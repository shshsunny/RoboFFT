# RoboFFT-D Environment Setup

This directory contains the diffusion branch of RoboFFT. The shared usage instructions, pretraining commands, fine-tuning commands, baseline list, and data/checkpoint layout are documented in the root `README.md`.

## Installation

```bash
cd robofft-d
conda create -n robofft-d python=3.8 -y
conda activate robofft-d
pip install -e .
pip install -e .[robomimic]
```

Install MuJoCo / Robosuite / Robomimic dependencies as required by your system. For the main RoboFFT-D experiments, only the Robomimic-related dependencies are required; D3IL and Furniture-Bench are not needed for the main Robomimic tables.

## Path setup

```bash
bash script/set_path.sh
source ~/.bashrc
conda activate robofft-d
```

When `robofft-f/` and `robofft-d/` are placed under the same release root, the defaults are:

```bash
ROBOFFT_D_DATA_DIR=<repo-root>/data
ROBOFFT_D_LOG_DIR=<repo-root>/log/robofft-d
```

`ROBOFFT_D_DATA_DIR` stores both processed Robomimic data and released RoboFFT-D base-policy checkpoints. `ROBOFFT_D_LOG_DIR` stores RoboFFT-D training logs and newly produced checkpoints.

To disable WandB, append `wandb=null` to a command.

## Run commands

Please use the root `README.md` for pretraining, fine-tuning, baseline, and evaluation commands.

The original DPPO README is retained as `README_DPPO.md` for provenance.
