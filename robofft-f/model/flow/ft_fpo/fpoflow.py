# MIT License

# Copyright (c) 2026 RoboFFT Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_flow import FlowMLP, NoisyFlowMLP
Sample = namedtuple("Sample", "trajectories chains")

class FPOFlow(nn.Module):
    def __init__(self, 
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 logratio_alpha,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type,
                 sample_t_type,
                 n_samples_per_action,
                 matching_style,
                 clip_ploss_style='ppo',
                 noise_decay_schedule=False,
                 noise_decay_start=0,
                 zero_start_eval=False,
                 zero_start_train=False,
                 use_per_sample_ratio=True, # default borrowed from FPO++
                 ):
        
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
        self.ft_denoising_steps = ft_denoising_steps    # could be adjusted
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type:str = noise_scheduler_type

        # prevent extreme values sampled from gaussian. deviation from mean should stay within `randn_clip_value` times of std.
        self.randn_clip_value:float = randn_clip_value
        
        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std:float = min_sampling_denoising_std

        # Minimum and maximum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std:float = min_logprob_denoising_std
        self.max_logprob_denoising_std:float = max_logprob_denoising_std
        self.noise_decay_schedule = noise_decay_schedule
        self.noise_decay_start = noise_decay_start
        
        # Minimum and maximum logprobability in each batch, cutoff within this range to prevent policy collapse
        self.logprob_min:float= logprob_min
        self.logprob_max:float= logprob_max
        self.logratio_alpha:float = logratio_alpha
        
        self.clip_ploss_coef:float = clip_ploss_coef
        self.clip_ploss_coef_base:float = clip_ploss_coef_base
        self.clip_ploss_coef_rate:float = clip_ploss_coef_rate
        self.clip_ploss_style:str = clip_ploss_style
        self.clip_vloss_coef:float = clip_vloss_coef
        
        # clip intermediate actions during inference
        self.denoised_clip_value:float = denoised_clip_value
        self.logprob_debug_sample=logprob_debug_sample
        self.logprob_debug_recalculate=logprob_debug_recalculate
        
        # noise network settings
        self.learn_explore_time_embedding=learn_explore_time_embedding
        self.time_dim_explore=time_dim_explore
        self.use_time_independent_noise=use_time_independent_noise
        self.noise_hidden_dims=noise_hidden_dims
        self.explore_net_activation_type=explore_net_activation_type
        self.sample_t_type = sample_t_type
        self.n_samples_per_action = n_samples_per_action
        self.matching_style = matching_style
        
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)  # it was false for previous experiments for hopper walker halfcheetah
        for param in self.actor_old.parameters():
            param.requires_grad = False             # don't train this copy, just use it to load checkpoint. 
        self.actor_old.to(self.device)
        
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True
        
        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning")
        
        self.critic = critic
        self.critic = self.critic.to(self.device)
        
        self.report_network_params()

        self.zero_start_eval = zero_start_eval
        self.zero_start_train = zero_start_train
        self.use_per_sample_ratio = use_per_sample_ratio
    
    def init_actor_ft(self, policy_copy):
        self.actor_ft = policy_copy
    
    def check_gradient_flow(self):
        print(f"{next(self.actor_ft.policy.parameters()).requires_grad}") #True
        print(f"{next(self.actor_ft.mlp_logvar.parameters()).requires_grad}")#True
        print(f"{next(self.actor_ft.time_embedding_explore.parameters()).requires_grad}")#True
        print(f"{self.actor_ft.logvar_min.requires_grad}")#False
        print(f"{self.actor_ft.logvar_max.requires_grad}")#False
        
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )
    
    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
            print(f"network_path={network_path}, self.device={self.device}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema:
                ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
            else:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            print(f"actor_network_data={actor_network_data.keys()}")
        else:
            logging.warning("No actor policy path provided. Not loading any actor policy. Start from randomly initialized policy.")
    
    @torch.no_grad()
    def sample_first_point(self, B:int, eval_mode=False)->Tuple[torch.Tensor, torch.Tensor]: # for denoising (inference)
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        if (eval_mode and self.zero_start_eval) or (not eval_mode and self.zero_start_train):
            xt = torch.zeros(B, self.horizon_steps * self.action_dim)
        else: 
            xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device) # shape: (B, )                   # mean() or sum() 
        xt=xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob
    
    # for flow matching loss calculation
    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        supported_time_sample_type = ['uniform', 'linspace', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            return torch.rand((batch_size, self.n_samples_per_action), device=self.device) # NOTE: FPO n_samples_per_action
        elif time_sample_type == 'linspace':
            return torch.as_tensor(torch.randint(low=0, high=self.inference_steps, size=(batch_size, self.n_samples_per_action), device=self.device), dtype=torch.float32) * (1.0/self.inference_steps)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size, self.n_samples_per_action), device=self.device)
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)   # Default beta
            s = kwargs.get("s", 0.999)       # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size, self.n_samples_per_action)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')
    
    # for flow matching loss calculation
    def generate_target(self, x1: Tensor, eps=None, t=None) -> tuple:
        assert x1.ndim == 3 or x1.ndim == 4
        if x1.ndim == 3:
            assert x1.shape[1:] == (self.horizon_steps, self.action_dim)
            x1 = x1.unsqueeze(1)
        else:
            assert x1.shape[1:] == (1, self.horizon_steps, self.action_dim) or x1.shape[1:] == (self.n_samples_per_action, self.horizon_steps, self.action_dim)
        x0 = eps # (batch_size, N_mc, horizon_steps, action_dim)
        if eps is None or t is None:
            t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
            x0 = eps = torch.randn((x1.shape[0], self.n_samples_per_action, *x1.shape[-2:]), dtype=torch.float32, device=self.device) # a.k.a. eps
        
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0 # ground truth velocity, a.k.a. target
        return xt, v, (eps, t)
    
    # for flow matching loss calculation
    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], -1, 1, 1)).to(self.device)  # RoboFFT-F Authors revised on 04/23/2025
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    # flow matching loss calculation
    def get_logprobs_eps_t(self, 
                     cond:dict, 
                     x:Tensor,
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     normalize_act_horizon_dimension=False,
                     eps=None,
                     t=None,
                     ):
        B = x.shape[0]
        xt, v, (eps, t) = self.generate_target(x, eps=eps, t=t)
        xt_flat = xt.reshape((-1, *xt.shape[2:]))
        t_flat = t.reshape((-1, ))
        obs = {
                        key: value.unsqueeze(1)\
                            .expand(-1, self.n_samples_per_action, *([-1] * (value.ndim-1)))\
                            .reshape(-1, *value.shape[1:])
                        for key, value in cond.items()
        }
        v_hat = self.actor_ft(xt_flat, t_flat, obs)
        v_hat = v_hat.reshape((B, -1, *v_hat.shape[1:]))
        
        
        if self.matching_style == "u":
            cfm_loss = F.mse_loss(input=v_hat, target=v, reduction='none')
        elif self.matching_style == "u_but_supervise_as_eps":
            delta_x = -t.unsqueeze(-1).unsqueeze(-1) * v_hat
            assert xt.shape == delta_x.shape, f"xt.shape={xt.shape}, delta_x.shape={delta_x.shape}"
            x0_hat = xt + delta_x
            cfm_loss = F.mse_loss(input=x0_hat, target=eps, reduction='none')
        else:
            raise NotImplementedError(f"Unknown matching_style={self.matching_style}. Supported styles: 'u', 'u_but_supervise_as_eps'")
        logprob = -cfm_loss.sum(dim=(-2,-1)) # (B, N_mc)
        
        if normalize_denoising_horizon:
            logprob = logprob / (self.inference_steps + 1)
            
        if normalize_act_space_dimension:
            logprob = logprob / self.action_dim
        
        if normalize_act_horizon_dimension:
            logprob = logprob / self.horizon_steps
        
        return logprob, eps, t
        
    
    @torch.no_grad()
    def get_actions(self, 
                    cond:dict, 
                    eval_mode:bool, 
                    clip_intermediate_actions=True,
                    itr=None,
                    n_train_itr=None
                    ):
        B=cond["state"].shape[0]
        dt = (1.0/self.inference_steps)* torch.ones(B, self.horizon_steps, self.action_dim, device=self.device)
        sqrt_dt = torch.sqrt(dt)
        steps = torch.linspace(0, 1-1/self.inference_steps,self.inference_steps).repeat(B, 1).to(self.device)
        xt, logprob_init = self.sample_first_point(B, eval_mode=eval_mode)
        
        for i in range(self.inference_steps):
            t = steps[:,i]
            vt = self.actor_ft.forward(xt, t, cond)
            xt += vt * dt
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            if not eval_mode and self.min_sampling_denoising_std > 0.0:
                if self.min_sampling_denoising_std == self.max_logprob_denoising_std:
                    noise_scale = self.min_sampling_denoising_std
                elif self.noise_decay_schedule and itr is not None:
                    noise_scale = self.max_logprob_denoising_std - (self.max_logprob_denoising_std - self.min_logprob_denoising_std) * max(0, itr-self.noise_decay_start)/(n_train_itr - self.noise_decay_start)
                else:
                    raise NotImplementedError("Only support constant noise schedule for now. Please set min_sampling_denoising_std and max_logprob_denoising_std to the same value if you want to use a constant noise schedule.")
                noise = torch.randn_like(xt) * (noise_scale * sqrt_dt)
                xt = xt + noise
            
            # prevent last action overflow
            if i == self.inference_steps-1 and (self.act_min is not None or self.act_max is not None):
                xt = xt.clamp_(self.act_min, self.act_max)                      
        return xt
      
    
    def loss(
        self,
        obs,
        actions, # a.k.a. x
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        eps,
        t,
        use_bc_loss=False,
        bc_loss_type='W2',
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        normalize_act_horizon_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        assert eps is not None and t is not None
        newlogprobs, _, _ = self.get_logprobs_eps_t(obs, 
                                                actions,
                                                normalize_denoising_horizon=normalize_denoising_horizon,
                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                normalize_act_horizon_dimension=normalize_act_horizon_dimension,
                                                eps=eps, t=t
                                                )
        # logprobs: (batch_size, N_mc)
        # eps: (batch_size, N_mc, horizon_steps, action_dim)
        # t: (batch_size, N_mc)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
        

        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")
        

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        # RoboFFT: logratio scale calibration
        scaled_logratio = logratio * self.logratio_alpha


        if not self.use_per_sample_ratio:
            scaled_logratio = scaled_logratio.mean(dim=1) # mean over N_mc

        scaled_ratio = scaled_logratio.exp()
        
        # Get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()
        
        advantages = advantages.unsqueeze(1)
        # Policy loss
        pg_loss_ppo1 = -advantages * scaled_ratio
        pg_loss_ppo2 = -advantages * torch.clamp(scaled_ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        
        pg_loss_ppo = torch.max(pg_loss_ppo1, pg_loss_ppo2)
        if self.clip_ploss_style == "ppo":
            pg_loss = pg_loss_ppo.mean()
        elif self.clip_ploss_style == "asymmetric":
            pg_loss_spo = -advantages * scaled_ratio + torch.abs(advantages) * (scaled_ratio - 1) ** 2 / (2 * self.clip_ploss_coef)
            pg_loss = torch.where(advantages >= 0, pg_loss_ppo, pg_loss_spo).mean()
        else:
            raise NotImplementedError(f"Unknown clip_ploss_style={self.clip_ploss_style}. Supported styles: 'ppo', 'asymmetric'")
        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use. 
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # Entropy loss is deprecated
        entropy_loss = 0.0
        
        # BC loss is deprecated
        bc_loss = 0.0
        noise_std = torch.tensor(self.min_sampling_denoising_std)
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),
        )

