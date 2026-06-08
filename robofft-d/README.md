# RoboFFT-D: Forward Fine-tuning for Diffusion Robot Policies

This repository is the **DPPO-based** implementation of **RoboFFT-D**, the diffusion-policy branch of RoboFFT. It is intended to reproduce the Robomimic experiments for online RL fine-tuning of diffusion robot policies, together with the diffusion-side baselines used in the RoboFFT paper.

RoboFFT-D fine-tunes diffusion policies in the **forward noising process**. It replaces reverse-chain policy-gradient likelihoods with a weighted denoising surrogate likelihood and applies a PPO-style update to online trajectories. The implementation therefore stays close to the original diffusion behavior-cloning objective while retaining the DPPO codebase's environment, rollout, critic, and training infrastructure.

## Repository scope

Main methods and configs:

| Method | Role in RoboFFT paper | Configs |
| --- | --- | --- |
| `RoboFFT-D` | Ours, forward fine-tuning for diffusion policies | `cfg/robomimic/finetune/<task>/robofft_diffusion_mlp.yaml`, `robofft_diffusion_mlp_img.yaml` |
| `DPPO` | Reverse-process diffusion PPO baseline | `cfg/robomimic/finetune/<task>/ft_ppo_diffusion_mlp.yaml`, `ft_ppo_diffusion_mlp_img.yaml` |
| `DRWR` | Diffusion reward-weighted regression baseline | `cfg/robomimic/finetune/<task>/ft_rwr_diffusion_mlp.yaml` |

The flow-side methods in the RoboFFT paper, including **RoboFFT-F**, **FPO**, and **ReinFlow**, should be reproduced from the ReinFlow-based RoboFFT-F repository.

## Installation

Create a Python environment and install the package in editable mode:

```bash
conda create -n robofft-d python=3.8 -y
conda activate robofft-d
pip install -e .
pip install -e .[robomimic]
```

Then install MuJoCo / Robosuite / Robomimic dependencies. The original DPPO installation notes are retained in the [installation](installation/) subdirectory.

For RoboFFT-D's main Robomimic experiments, the relevant part is MuJoCo + Robomimic / Robosuite. D3IL and Furniture-Bench are not needed for the main RoboFFT-D Robomimic results.

Set data, log, and WandB paths:

```bash
bash script/set_path.sh
source ~/.bashrc
conda activate robofft-d
```

The script defines:

```bash
ROBOFFT_D_DATA_DIR      # data and normalization files
ROBOFFT_D_LOG_DIR       # logs and checkpoints
ROBOFFT_D_WANDB_ENTITY  # optional
```

To disable WandB for a quick check, append `wandb=null` to the training command.

## Data and checkpoints

The Robomimic fine-tuning configs expect normalization statistics at:

```text
${ROBOFFT_D_DATA_DIR}/robomimic/<task>/normalization.npz
```

The launcher downloads normalization files, training data files and known checkpoints through `script/download_url.py` when configured paths are missing. If you use your own diffusion pretraining checkpoint, override `base_policy_path` explicitly:

```bash
python script/run.py   --config-name=robofft_diffusion_mlp   --config-dir=cfg/robomimic/finetune/can   base_policy_path=/path/to/pretrained/state_3000.pt   wandb=null
```

## Running RoboFFT-D on Robomimic

Supported Robomimic tasks:

```text
lift, can, square, transport
```

State-input fine-tuning:

```bash
TASK=can
python script/run.py   --config-name=robofft_diffusion_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

Pixel-input fine-tuning:

```bash
TASK=can
python script/run.py   --config-name=robofft_diffusion_mlp_img   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

Use the same command pattern for `lift`, `square`, and `transport`.

## Running diffusion-side baselines

DPPO baseline:

```bash
TASK=can
python script/run.py   --config-name=ft_ppo_diffusion_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null

python script/run.py   --config-name=ft_ppo_diffusion_mlp_img   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

DRWR baseline:

```bash
TASK=can
python script/run.py   --config-name=ft_rwr_diffusion_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

Note that DRWR is currently reproduced only for state-input experiments in this release.

## Optional pretraining

You can skip pretraining if you use released checkpoints. To train a Robomimic diffusion policy from demonstrations:

```bash
TASK=can
python script/run.py   --config-name=pre_diffusion_mlp   --config-dir=cfg/robomimic/pretrain/${TASK}   wandb=null

python script/run.py   --config-name=pre_diffusion_mlp_img   --config-dir=cfg/robomimic/pretrain/${TASK}   wandb=null
```

The resulting checkpoints are saved under the run's `checkpoint/` directory and can be passed to fine-tuning through `base_policy_path=/path/to/state_*.pt`.

## Evaluation and video recording

Evaluate a checkpoint with the corresponding eval config:

```bash
TASK=can
python script/run.py   --config-name=eval_diffusion_mlp   --config-dir=cfg/robomimic/eval/${TASK}   base_policy_path=/path/to/checkpoint/state_*.pt   ft_denoising_steps=10   wandb=null

python script/run.py   --config-name=eval_diffusion_mlp_img   --config-dir=cfg/robomimic/eval/${TASK}   base_policy_path=/path/to/checkpoint/state_*.pt   ft_denoising_steps=10   wandb=null
```

To save Robomimic videos, set:

```bash
env.save_video=True train.render.freq=<eval_interval> train.render.num=<num_videos>
```

## Implementation map

Core RoboFFT-D implementation:

```text
model/diffusion/diffusion_fpo.py
agent/finetune/train_fpo_diffusion_agent.py
agent/finetune/train_fpo_diffusion_img_agent.py
```

Important implementation details:

- `FPODiffusion.generate_target(...)` samples forward noising times and noise perturbations for action chunks.
- `FPODiffusion.get_logprobs_eps_t(...)` converts DSM / CFM-style denoising loss into a surrogate log-probability.
- `FPODiffusion.loss(...)` forms the PPO-style clipped objective from surrogate log-ratios.
- `use_lambda_weight` applies log-SNR / lambda weighting to the denoising surrogate.
- `logratio_alpha` calibrates the scale of the surrogate log-ratio.
- `n_samples_per_action` controls Monte-Carlo repetitions.
- `use_anchor_loss` and `anchor_loss_coeff` regularize the fine-tuned policy toward the best-so-far target policy.

DPPO and DRWR baselines:

```text
model/diffusion/diffusion_ppo.py
model/diffusion/diffusion_rwr.py
agent/finetune/train_ppo_diffusion_agent.py
agent/finetune/train_ppo_diffusion_img_agent.py
agent/finetune/train_rwr_diffusion_agent.py
```

Robomimic environment wrappers:

```text
env/gym_utils/wrapper/robomimic_lowdim.py
env/gym_utils/wrapper/robomimic_image.py
env/gym_utils/wrapper/multi_step.py
```

## Citation

```bibtex
@inproceedings{anonymous2026robofft,
  title     = {RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process},
  author    = {Anonymous Author(s)},
  booktitle = {Submitted to the 10th Conference on Robot Learning},
  year      = {2026}
}
```

This codebase builds on DPPO. Please also cite the original DPPO project when using the corresponding components.
