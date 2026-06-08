# RoboFFT-F: Forward Fine-tuning for Flow Robot Policies

This repository is the **ReinFlow-based** implementation of **RoboFFT-F**, the flow-policy branch of RoboFFT. It is intended to reproduce the Robomimic experiments for online RL fine-tuning of rectified-flow robot policies, together with the flow-side baselines used in the RoboFFT paper.

RoboFFT-F performs policy-gradient fine-tuning in the **forward noising / flow-matching space**. Instead of optimizing through the entire reverse denoising chain, it reuses the weighted denoising / flow-matching loss as a surrogate likelihood inside a PPO-style objective. This keeps imitation pretraining and RL fine-tuning aligned under the same forward-process objective while avoiding the long-chain reverse-MDP optimization used by methods such as ReinFlow and DPPO.

## Repository scope

This release keeps a ReinFlow-style project structure while renaming package paths and runtime environment variables for RoboFFT-F. The main RoboFFT-F experiments use **Robomimic**.

Main methods and configs:

| Method | Role in RoboFFT paper | Configs |
| --- | --- | --- |
| `RoboFFT-F` | Ours, forward fine-tuning for flow policies | `cfg/robomimic/finetune/<task>/robofft_reflow_mlp.yaml`, `robofft_reflow_mlp_img.yaml` |
| `FPO` | Forward-process baseline without the RoboFFT exploration / ratio-calibration choices | `cfg/robomimic/finetune/<task>/ft_fpo_reflow_mlp.yaml`, `ft_fpo_reflow_mlp_img.yaml` |
| `ReinFlow` | Reverse-process flow fine-tuning baseline | `cfg/robomimic/finetune/<task>/ft_ppo_reflow_mlp.yaml`, `ft_ppo_reflow_mlp_img.yaml` |

The diffusion-side methods in the RoboFFT paper, including **RoboFFT-D**, **DPPO**, and **DRWR**, should be reproduced from the DPPO-based RoboFFT-D repository rather than this flow repository.

## Installation

Create a Python environment and install the package in editable mode:

```bash
conda create -n robofft-f python=3.8 -y
conda activate robofft-f
pip install -e .
pip install -e .[robomimic]
```

Then install MuJoCo / Robosuite / Robomimic dependencies following the Robomimic part of the [RoboFFT-F installation guide](installation/robofft_f-setup.md), which is adapted from the original ReinFlow setup notes.

For headless training, make sure MuJoCo rendering is correctly configured. The launcher reads `sim_device` from the config and uses EGL when `sim_device` is provided; otherwise it falls back to OSMesa.

Finally set data, log, and WandB paths:

```bash
bash script/set_path.sh
source ~/.bashrc
conda activate robofft-f
```

The script defines:

```bash
ROBOFFT_F_DIR       # repository root
ROBOFFT_F_DATA_DIR  # data and normalization files
ROBOFFT_F_LOG_DIR   # logs and checkpoints
ROBOFFT_F_WANDB_ENTITY  # optional
```

To disable WandB for a quick check, append `wandb=null` to the training command.

## Data and checkpoints

The Robomimic fine-tuning configs expect normalization statistics at:

```text
${ROBOFFT_F_DATA_DIR}/robomimic/<task>/normalization.npz
```

The launcher downloads normalization, demonstration data and checkpoint files through `script/download_url.py` when the configured paths are missing. If you use your own pre-trained flow policy, override `base_policy_path` explicitly:

```bash
python script/run.py   --config-name=robofft_reflow_mlp   --config-dir=cfg/robomimic/finetune/can   base_policy_path=/path/to/pretrained/state_3000.pt   wandb=null
```

## Running RoboFFT-F on Robomimic

Supported Robomimic tasks:

```text
lift, can, square, transport
```

State-input fine-tuning:

```bash
TASK=can
python script/run.py   --config-name=robofft_reflow_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

Pixel-input fine-tuning:

```bash
TASK=can
python script/run.py   --config-name=robofft_reflow_mlp_img   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

Use the same command pattern for `lift`, `square`, and `transport`.

## Running flow-side baselines

FPO baseline:

```bash
TASK=can
python script/run.py   --config-name=ft_fpo_reflow_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null

python script/run.py   --config-name=ft_fpo_reflow_mlp_img   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

ReinFlow-style reverse fine-tuning baseline:

```bash
TASK=can
python script/run.py   --config-name=ft_ppo_reflow_mlp   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null

python script/run.py   --config-name=ft_ppo_reflow_mlp_img   --config-dir=cfg/robomimic/finetune/${TASK}   wandb=null
```

## Optional pretraining

You can skip pretraining if you use released checkpoints. To train a Robomimic flow policy from demonstrations:

```bash
TASK=can
python script/run.py   --config-name=pre_reflow_mlp   --config-dir=cfg/robomimic/pretrain/${TASK}   wandb=null

TASK=can
python script/run.py   --config-name=pre_reflow_mlp_img   --config-dir=cfg/robomimic/pretrain/${TASK}   wandb=null
```

## Evaluation and video recording

Evaluate a checkpoint with the corresponding eval config:

```bash
TASK=can
python script/run.py   --config-name=eval_reflow_mlp   --config-dir=cfg/robomimic/eval/${TASK}   base_policy_path=/path/to/checkpoint/state_*.pt   wandb=null

python script/run.py   --config-name=eval_reflow_mlp_img   --config-dir=cfg/robomimic/eval/${TASK}   base_policy_path=/path/to/checkpoint/state_*.pt   wandb=null
```

To save Robomimic videos, set:

```bash
env.save_video=True train.render.freq=<eval_interval> train.render.num=<num_videos>
```

## Implementation map

Core RoboFFT-F implementation:

```text
model/flow/ft_fpo/fpoflow.py
agent/finetune/robofft_f/train_fpo_flow_agent.py
agent/finetune/robofft_f/train_fpo_flow_img_agent.py
```

Important implementation details:

- `FPOFlow.get_logprobs_eps_t(...)` computes the forward-process flow-matching surrogate log-probability.
- `FPOFlow.loss(...)` forms the PPO-style clipped objective from surrogate log-ratios.
- `logratio_alpha` calibrates the surrogate log-ratio scale.
- `min_sampling_denoising_std`, `min_logprob_denoising_std`, and `max_logprob_denoising_std` control SDE-style exploration noise and likelihood evaluation.
- `sample_t_type=linspace` samples forward times from the inference grid used by the flow sampler.
- `n_samples_per_action` controls Monte-Carlo repetitions for the surrogate loss.

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

This codebase builds on ReinFlow and DPPO. Please also cite the original projects when using the corresponding components.
