# RoboFFT-F Environment Setup

This directory contains the flow / ReFlow branch of RoboFFT. The shared usage instructions, pretraining commands, fine-tuning commands, baseline list, and data/checkpoint layout are documented in the root `README.md`.

## Installation

```bash
cd robofft-f
conda create -n robofft-f python=3.8 -y
conda activate robofft-f
pip install -e .
pip install -e .[robomimic]
```

Install MuJoCo / Robosuite / Robomimic dependencies as required by your system. For the main RoboFFT-F experiments, only the Robomimic-related dependencies are required; Gym, Kitchen, D3IL, and Furniture-Bench are not needed for the main Robomimic tables.

## Path setup

```bash
bash script/set_path.sh
source ~/.bashrc
conda activate robofft-f
```

When `robofft-f/` and `robofft-d/` are placed under the same release root, the defaults are:

```bash
ROBOFFT_F_DATA_DIR=<repo-root>/data
ROBOFFT_F_LOG_DIR=<repo-root>/log/robofft-f
```

`ROBOFFT_F_DATA_DIR` stores both processed Robomimic data and released RoboFFT-F base-policy checkpoints. `ROBOFFT_F_LOG_DIR` stores RoboFFT-F training logs and newly produced checkpoints.

To disable WandB, append `wandb=null` to a command.

## Run commands

Please use the root `README.md` for pretraining, fine-tuning, baseline, and evaluation commands.

The original ReinFlow README is retained as `README_ReinFlow.md` for provenance.
