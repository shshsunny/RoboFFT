#!/bin/bash

##################### Paths #####################
#### RoboFFT-F Authors revised from DPPO's original repository

# Set default paths
DEFAULT_DIR="${PWD}"
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"

# Prompt the user for input, allowing overrides
read -p "Enter the place where your robofft_f script lies: [default: ${DEFAULT_DIR}], press ENTER to use default: " DIR
ROBOFFT_F_DIR=${DIR:-$DEFAULT_DIR}  # Use user input or default if input is empty

read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], press ENTER to use default: " DATA_DIR
ROBOFFT_F_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], press ENTER to use default: " LOG_DIR
ROBOFFT_F_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

# Export to current session
export ROBOFFT_F_DIR="$ROBOFFT_F_DIR"
export ROBOFFT_F_DATA_DIR="$ROBOFFT_F_DATA_DIR"
export ROBOFFT_F_LOG_DIR="$ROBOFFT_F_LOG_DIR"

# Confirm the paths with the user
echo "Script directory set to: $ROBOFFT_F_DIR"
echo "Data directory set to: $ROBOFFT_F_DATA_DIR"
echo "Log directory set to: $ROBOFFT_F_LOG_DIR"

# Append environment variables to .bashrc
echo "export ROBOFFT_F_DIR=\"$ROBOFFT_F_DIR\"" >> ~/.bashrc
echo "export ROBOFFT_F_DATA_DIR=\"$ROBOFFT_F_DATA_DIR\"" >> ~/.bashrc
echo "export ROBOFFT_F_LOG_DIR=\"$ROBOFFT_F_LOG_DIR\"" >> ~/.bashrc

echo "Environment variables ROBOFFT_F_DIR, ROBOFFT_F_DATA_DIR and ROBOFFT_F_LOG_DIR added to .bashrc and applied to the current session."

# Set verbose logging
echo -e "# verbose debug
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1">> ~/.bashrc
echo "Suppressed D4RL import errors, turned on verbose debugging for HYDRA, CUDA, and TORCH_USE_CUDA_DSA"

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), press ENTER to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export ROBOFFT_F_WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $ROBOFFT_F_WANDB_ENTITY"

  # Append environment variable to .bashrc
  echo "export ROBOFFT_F_WANDB_ENTITY=\"$ENTITY\"" >> ~/.bashrc
  
  echo "Environment variable ROBOFFT_F_WANDB_ENTITY added to .bashrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi