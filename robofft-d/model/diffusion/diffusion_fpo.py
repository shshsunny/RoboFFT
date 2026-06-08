"""
FPODiffusion: Flow Policy Optimization for Diffusion Models
Adapting FPO's surrogate noise-matching loss (ELBO equivalence) to the DPPO-style diffusion-policy framework.

Key differences from DPPO:
- Omits the multi-step Denoising MDP chain evaluation.
- Calculates surrogate log probabilities via Noise Matching Loss weighted by Delta Log-SNR.
- Dramatically reduces memory and computation during PPO rollouts and updates.
"""

import math
import torch
import logging
import torch.nn.functional as F
import copy

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.diffusion_vpg import VPGDiffusion
from model.diffusion.sampling import make_timesteps, extract

class FPODiffusion(VPGDiffusion):
    def __init__(
        self,
        n_samples_per_action: int = 1,
        clip_ploss_coef: float = 0.2,
        aspo_ploss_coef: float = 0.2,
        switch_aspo_with_raw_advantage: bool = False,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        negdisc_coef: float = 0.5,
        negdisc_coef_end: float = None,
        clip_ploss_style: str = "ppo", # "ppo" or "asymmetric"
        clip_vloss_coef: float = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        norm_adv: bool = True,
        logprob_min: float = -200.0,
        logprob_max: float = 2.0,
        logratio_alpha: float = 1.0,
        use_per_sample_ratio: bool = True,
        gamma_denoising: float = 1.0,
        use_lambda_weight: bool = False,
        logprob_normalizer: float = 1.0,
        sample_t_type: str = 'full',
        surrogate_type: str = 'dsm',
        eta_train_max: float = 1.0,
        eta_train_min: float = 1.0,
        eta_train: float = 1.0,
        eta_eval: float = 1.0,
        clip_intermediate_actions: bool = False,
        bc_loss_type: str = 'mse',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_samples_per_action = n_samples_per_action
        self.norm_adv = norm_adv
        self.clip_ploss_coef = clip_ploss_coef
        self.aspo_ploss_coef = aspo_ploss_coef
        self.switch_aspo_with_raw_advantage = switch_aspo_with_raw_advantage
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.negdisc_coef = negdisc_coef
        self.negdisc_coef_end = negdisc_coef_end if negdisc_coef_end is not None else negdisc_coef
        self.clip_ploss_style = clip_ploss_style
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.clip_intermediate_actions = clip_intermediate_actions

        self.logprob_min = logprob_min
        self.logprob_max = logprob_max
        self.logratio_alpha = logratio_alpha
        self.use_per_sample_ratio = use_per_sample_ratio
        self.gamma_denoising = gamma_denoising
        self.use_lambda_weight = use_lambda_weight
        self.logprob_normalizer = logprob_normalizer
        self.sample_t_type = sample_t_type
        self.surrogate_type = surrogate_type
        self.eta_train = eta_train
        self.eta_train_min = eta_train_min
        self.eta_train_max = eta_train_max
        self.eta_eval = eta_eval

        self.bc_loss_type = bc_loss_type
        print("using fixed eta: train =", eta_train, ", eval =", eta_eval)
        if sample_t_type == 'full':
            assert self.n_samples_per_action % self.ft_denoising_steps == 0, "For 'full' sample_t_type, n_samples_per_action must be divisible by ft_denoising_steps to ensure proper indexing."
        
        # if use_anchor_loss: # initialize the anchor network
        self.actor_anchor = copy.deepcopy(self.actor)
        logging.info("Cloned anchor model")

        # Turn off gradients for original model
        for param in self.actor_anchor.parameters():
            param.requires_grad = False

        # precalculate delta-SNR coefficients for surrogate loss
        self._compute_delta_tilde_lambda()
        self._compute_gt()

    
    
    def _compute_delta_tilde_lambda(self):

        device = self.device

        if self.use_ddim:
            alphas = torch.flip(self.ddim_alphas[-self.ft_denoising_steps :], [0])
            alphas_prev = torch.flip(self.ddim_alphas_prev[-self.ft_denoising_steps :], [0])
        else:
            t_single = torch.arange(0, self.ft_denoising_steps, device=device)
            alphas = self.alphas_cumprod[t_single]
            alphas_prev = self.alphas_cumprod_prev[t_single]

        def get_log_snr(alpha_tensor):
            sigma_sq = 1.0 - alpha_tensor
            tilde_sigma_sq = torch.clamp(sigma_sq, min=1e-3)
            numerator = torch.clamp(1.0 - tilde_sigma_sq, min=1e-3)
            return torch.log(numerator) - torch.log(tilde_sigma_sq)

        lambda_current = get_log_snr(alphas)
        lambda_prev = get_log_snr(alphas_prev)

        self.delta_tilde_lambda = lambda_prev - lambda_current
        print(f"Computed Delta Tilde Lambda for FPO: {self.delta_tilde_lambda.cpu().numpy()}")

        if not self.use_lambda_weight:
            self.delta_tilde_lambda = torch.ones_like(self.delta_tilde_lambda, requires_grad=False)
            print("Using uniform weights for surrogate loss (no lambda weighting).")

    def _compute_delta_tilde_lambda_stable(self, c=2.0):
        device = self.device

        if self.use_ddim:
            alphas = torch.flip(self.ddim_alphas[-self.ft_denoising_steps :], [0])
            alphas_prev = torch.flip(self.ddim_alphas_prev[-self.ft_denoising_steps :], [0])
        else:
            t_single = torch.arange(0, self.ft_denoising_steps, device=device)
            alphas = self.alphas_cumprod[t_single]
            alphas_prev = self.alphas_cumprod_prev[t_single]

        def get_log_snr(alpha_tensor):
            sigma_sq = torch.clamp(1.0 - alpha_tensor, min=1e-6, max=1.0 - 1e-6)
            numerator = torch.clamp(alpha_tensor, min=1e-6, max=1.0 - 1e-6)
            return torch.log(numerator) - torch.log(sigma_sq)

        lambda_current = get_log_snr(alphas)
        lambda_prev = get_log_snr(alphas_prev)

        delta_lambda = torch.abs(lambda_prev - lambda_current)

        delta_lambda = torch.clamp(delta_lambda, max=100.0)

        w_lambda = torch.sigmoid(-lambda_current + c)

        self.delta_tilde_lambda = delta_lambda * w_lambda
        
        print(f"Stabilized Delta Tilde Lambda for FPO: {self.delta_tilde_lambda.cpu().numpy()}")

        if not self.use_lambda_weight:
            self.delta_tilde_lambda = torch.ones_like(self.delta_tilde_lambda, requires_grad=False)
            print("Using uniform weights for surrogate loss (no lambda weighting).")

    def _compute_gt(self):

        betas = self.betas
        sigmas = self.sqrt_one_minus_alphas_cumprod
        
        g_t = betas / (2.0 * sigmas)
        
        g_t_sq = g_t ** 2
        
        d_log_snr = self.delta_tilde_lambda if hasattr(self, 'delta_tilde_lambda') else None
        
        log.info("--- Diffusion Weight Schedule Analysis ---")
        log.info(f"{'t':>5} | {'beta_t':>10} | {'g_t^2 (CFM)':>12} | {'d_log_snr (ELBO)':>15}")
        log.info("-" * 55)
        
        scale_factor = 1.0 / g_t_sq[:self.ft_denoising_steps].mean() 
        g_t_sq_normalized = g_t_sq * scale_factor
        self.g_t_sq = g_t_sq_normalized

        for t in range(min(len(betas), self.ft_denoising_steps)):
            cfm_w = self.g_t_sq[t].item()
            elbo_w = d_log_snr[t].item() if d_log_snr is not None else 0.0
            log.info(f"{t:5d} | {betas[t]:10.6f} | {cfm_w:12.6f} | {elbo_w:15.6f}")


    def get_train_eta_by_t(self, t, x_shape):
        K = self.ft_denoising_steps

        eta = torch.ones_like(t, dtype=torch.float32, device=t.device)

        ft_mask = t < K
        if ft_mask.any():
            d = t[ft_mask].float()
            frac = d / max(K - 1, 1)

            eta_ft = self.eta_train_min * (
                self.eta_train_max / self.eta_train_min
            ) ** frac

            eta[ft_mask] = eta_ft

        eta = eta.view(-1, *([1] * (len(x_shape) - 1)))
        return eta
    

    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        policy='ft',
        deterministic=False,
    ):
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor_ft if policy == 'ft' else (self.actor if policy == 'base' else self.actor_anchor)
        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0: 
            cond_ft = {key: cond[key][ft_indices] for key in cond} 
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Let eta = predefined value
        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            base_logvar = extract(self.ddpm_logvar_clipped, t, x.shape)

            if deterministic:
                etas = torch.ones_like(mu).to(mu.device) * self.eta_eval
            else:
                etas = self.get_train_eta_by_t(t, x.shape)

            logvar = base_logvar + 2.0 * torch.log(torch.clamp(etas, min=1e-8))
            
            
        return mu, logvar, etas
    

    # override: remove chains from VPGDiffusion
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        policy='ft',
        eval_style='ddpm',
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            policy: which policy to use ('ft' for fine-tuned, 'base' for base, 'anchor' for anchor)
            eval_style: evaluation style ('ddpm', 'ddpm_clip', 'ddim')
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))

        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                policy=policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            if not deterministic: # train
                std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if eval_style == 'ddpm_clip':
                    std = torch.clip(std, min=min_sampling_denoising_std)
                elif eval_style == 'ddpm': # default evaluation style
                    if t == 0:
                        std = torch.zeros_like(std)
                    else:
                        std = torch.clip(std, min=1e-3)
                elif eval_style == 'ddim':
                    std = torch.zeros_like(std)

            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if (self.final_action_clip_value is not None and i == len(t_all) - 1) or self.clip_intermediate_actions:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return x
    
    
    def generate_target(self, x0, eps=None, t=None, denoising_inds=None):
        B = x0.shape[0]
        if t is None and denoising_inds is None: # sample time
            if self.use_ddim:

                candidate_t = torch.flip(self.ddim_t[-self.ft_denoising_steps :], [0])

            else:
                candidate_t = torch.arange(
                    start=0,
                    end=self.ft_denoising_steps,
                    device=self.device,
                )
            
            if self.sample_t_type == 'full':
                assert self.n_samples_per_action % self.ft_denoising_steps == 0, "For 'full' sample_t_type, n_samples_per_action must be divisible by ft_denoising_steps to ensure proper indexing."
                denoising_inds = torch.arange(self.n_samples_per_action, device=self.device).long() % self.ft_denoising_steps # (N_mc,) 例如 ft_denoising_steps=4, n_samples_per_action=8，则 denoising_inds=[0,1,2,3,0,1,2,3]
                denoising_inds = denoising_inds.unsqueeze(0).expand(B, -1) # (B, N_mc)
                t = candidate_t[denoising_inds]
            else:
                denoising_inds = torch.randint(0, len(candidate_t), (B, self.n_samples_per_action), device=self.device).long() # (B, N_mc)
                t = candidate_t[denoising_inds] # (B, N_mc)


        if eps is None: # sample eps
            eps = torch.randn((B, self.n_samples_per_action, *x0.shape[1:]), device=self.device)

        # x0: (B, N_mc, Ta, Da) -> (B*N_mc, Ta, Da)
        x0_expand = x0.unsqueeze(1).expand(-1, self.n_samples_per_action, *([-1]* (x0.ndim-1)))
        x0_flat = x0_expand.reshape(B * self.n_samples_per_action, *x0.shape[1:])
        
        # t: (B, N_mc) -> (B*N_mc,)
        t_flat = t.reshape(-1)
        # eps: (B, N_mc, Ta, Da) -> (B*N_mc, Ta, Da)
        eps_flat = eps.reshape(B * self.n_samples_per_action, *x0.shape[1:])
        
        x_noisy_flat = self.q_sample(x_start=x0_flat, t=t_flat, noise=eps_flat)
        
        x_noisy = x_noisy_flat.view(B, self.n_samples_per_action, *x0.shape[1:])

        return x_noisy, eps, t, denoising_inds
    

    def get_logprobs_eps_t(self, cond, actions, eps=None, t=None, denoising_inds=None, loss_type='mse'):
        
        B = actions.shape[0]
        
        x_noisy, eps, t, denoising_inds = self.generate_target(actions, eps=eps, t=t, denoising_inds=denoising_inds)

        x_noisy_flat = x_noisy.flatten(end_dim=1) # (B*N_mc, Ta, Da)
        t_flat = t.flatten()
        denoising_inds_flat = denoising_inds.flatten() # (B * N_mc)
        eps_flat = eps.flatten(end_dim=1) # (B*N_mc, Ta, Da)

        cond_flat = {
            key: val.unsqueeze(1).expand(-1, self.n_samples_per_action, *([-1]*(val.ndim-1))).flatten(end_dim=1)
            for key, val in cond.items()
        }

        eps_hat_flat = self.actor_ft(x_noisy_flat, t_flat, cond=cond_flat)

        if self.surrogate_type == 'dsm':
            if loss_type == 'mse':
                _loss = F.mse_loss(eps_hat_flat, eps_flat, reduction='none')  # (B*N_mc, Ta, Da)
            elif loss_type == 'huber':
                _loss = F.smooth_l1_loss(eps_hat_flat, eps_flat, reduction='none')  # (B*N_mc, Ta, Da)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            coef = extract(self.delta_tilde_lambda, denoising_inds_flat, _loss.shape) / 2.0
            
            dsm_loss_flat = coef * _loss
            dsm_loss = dsm_loss_flat.view(B, self.n_samples_per_action, *actions.shape[1:])

            logprob = -dsm_loss.mean(dim=-1).sum(dim=-1)  # (B, N_mc)

        elif self.surrogate_type == 'cfm':

            if loss_type == 'mse':
                _loss = F.mse_loss(eps_hat_flat, eps_flat, reduction='none')  # (B*N_mc, Ta, Da)
            elif loss_type == 'huber':
                _loss = F.smooth_l1_loss(eps_hat_flat, eps_flat, reduction='none')  # (B*N_mc, Ta, Da)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            coef = extract(self.g_t_sq, denoising_inds_flat, _loss.shape) / 2.0
            
            dsm_loss_flat = coef * _loss
            dsm_loss = dsm_loss_flat.view(B, self.n_samples_per_action, *actions.shape[1:])


            logprob = -dsm_loss.mean(dim=-1).sum(dim=-1)
        
        assert logprob.shape == (B, self.n_samples_per_action), f"Expected logprob shape {(B, self.n_samples_per_action)}, but got {logprob.shape}"
        
        logprob = logprob * self.logprob_normalizer
        return logprob, eps, t, denoising_inds



    def loss(
        self,
        obs,
        actions,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        eps,
        t,
        denoising_inds,
        use_bc_loss=False,
        use_anchor_loss=False,
        itr=0,
        total_itr=1,
    ):
        
        assert eps is not None and t is not None and denoising_inds is not None
        newlogprobs, _, _, _ = self.get_logprobs_eps_t(obs, actions, eps=eps, t=t, denoising_inds=denoising_inds)
        newlogprobs_raw_min = newlogprobs.min().item()
        newlogprobs_raw_max = newlogprobs.max().item()
        oldlogprobs_raw_min = oldlogprobs.min().item()
        oldlogprobs_raw_max = oldlogprobs.max().item()
        newlogprobs_clipfrac = ((newlogprobs < self.logprob_min) | (newlogprobs > self.logprob_max)).float().mean()
        oldlogprobs_clipfrac = ((oldlogprobs < self.logprob_min) | (oldlogprobs > self.logprob_max)).float().mean()
        raw_advantages = advantages.clone()
        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages
        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)

        # Ratio
        logratio = newlogprobs - oldlogprobs
        logratio_raw_mean = logratio.mean().item()
        logratio_raw_std = logratio.std().item()
        logratio_clip_frac = ((logratio < -10.0) | (logratio > 10.0)).float().mean()
        logratio = logratio.clamp(min=-10.0, max=10.0)
        ratio = logratio.exp()
        scaled_logratio = logratio * self.logratio_alpha

        if not self.use_per_sample_ratio:
            scaled_logratio = scaled_logratio.mean(dim=-1, keepdim=True)

        scaled_ratio = scaled_logratio.exp()
        advantages = advantages.unsqueeze(-1)
        raw_advantages = raw_advantages.unsqueeze(-1)

        clip_ploss_coef = self.clip_ploss_coef

        with torch.no_grad():
            approx_kl = ((scaled_ratio - 1) - scaled_logratio).mean()
            clipfrac = ((scaled_ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        # Policy loss (PPO-Clip)
        pg_loss1 = -advantages * scaled_ratio
        pg_loss2 = -advantages * torch.clamp(scaled_ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
        log.info(f"shapes - advantages: {advantages.shape}, scaled_ratio: {scaled_ratio.shape}, pg_loss1: {pg_loss1.shape}, pg_loss2: {pg_loss2.shape}")
        pg_loss_ppo = torch.max(pg_loss1, pg_loss2)
        if self.clip_ploss_style == "ppo":
            pg_loss = pg_loss_ppo.mean()
        elif self.clip_ploss_style == "asymmetric":
            pg_loss_spo = -advantages * scaled_ratio + torch.abs(advantages) * (scaled_ratio - 1) ** 2 / (2 * self.aspo_ploss_coef)
            if self.switch_aspo_with_raw_advantage:
                pg_loss = torch.where(raw_advantages >= 0, pg_loss_ppo, pg_loss_spo).mean()
            else:
                pg_loss = torch.where(advantages >= 0, pg_loss_ppo, pg_loss_spo).mean()
        elif self.clip_ploss_style == 'negdisc':
            negdisc_coef = (self.negdisc_coef * (total_itr - itr) + self.negdisc_coef_end * itr) / total_itr
            pg_loss_disc = pg_loss_ppo * negdisc_coef
            pg_loss = torch.where(advantages >= 0, pg_loss_ppo, pg_loss_disc).mean()
        else:
            raise NotImplementedError(f"Unknown clip_ploss_style={self.clip_ploss_style}. Supported styles: 'ppo', 'asymmetric', 'negdisc'")

        # Value loss
        newvalues = self.critic(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        bc_loss = torch.tensor(0.0, device=self.device)
        if use_bc_loss:
            samples = self.forward(
                cond=obs,
                deterministic=False,
                policy='base',
            )
            bc_logprobs, _, _, _ = self.get_logprobs_eps_t(
                obs,
                samples,
                loss_type=self.bc_loss_type,
            ) # (B, N_mc)

            bc_loss = -bc_logprobs.mean()
        
        anchor_loss = torch.tensor(0.0, device=self.device)
        if use_anchor_loss:
            samples = self.forward(
                cond=obs,
                deterministic=False,
                policy='anchor',
            )
            anchor_logprobs, _, _, _ = self.get_logprobs_eps_t(
                obs,
                samples,
            )
            anchor_loss = -anchor_logprobs.mean()


        logprob_stats = {
            "newlogprobs_raw_min": newlogprobs_raw_min,
            "newlogprobs_raw_max": newlogprobs_raw_max,
            "oldlogprobs_raw_min": oldlogprobs_raw_min,
            "oldlogprobs_raw_max": oldlogprobs_raw_max,
            "newlogprobs_clipfrac": newlogprobs_clipfrac.item(),
            "oldlogprobs_clipfrac": oldlogprobs_clipfrac.item(),
            "logratio_raw_mean": logratio_raw_mean,
            "logratio_raw_std": logratio_raw_std,
            "logratio_clip_frac": logratio_clip_frac.item(),
        }

        return (
            pg_loss,
            torch.tensor(0.0, device=self.device), 
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            self.eta_train,
            logprob_stats,
            anchor_loss,
        )