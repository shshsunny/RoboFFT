"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from urllib.request import urlretrieve
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _is_google_drive_folder(url: str) -> bool:
    return "drive.google.com" in url and "/folders/" in url


def _is_google_drive_url(url: str) -> bool:
    return "drive.google.com" in url


def _download_file(url: str, output_path: str):
    """Download a single file.

    Google Drive links still use gdown for backward compatibility. Other URLs
    are treated as direct file URLs, which lets the release configs use
    Hugging Face / ModelScope / institutional mirrors without changing the
    launcher again.
    """
    if _is_google_drive_url(url):
        return gdown.download(url=url, output=output_path, fuzzy=True)
    return urlretrieve(url, output_path)


def _download_dataset(url: str, dataset_path: str):
    """Download a dataset asset.

    Legacy DPPO/ReinFlow dataset URLs are Google Drive folders. RoboFFT release
    dataset URLs should preferably be direct links to train.npz.
    """
    if _is_google_drive_folder(url):
        return gdown.download_folder(url=url, output=os.path.dirname(dataset_path))
    return _download_file(url, dataset_path)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = cfg.train_dataset_path
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        _download_dataset(download_url, download_target)

    # For for-tuning: download normalization if needed
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(
            f"Downloading normalization statistics from {download_url} to {download_target}"
        )
        _download_file(download_url, download_target)

    # For for-tuning: download checkpoint if needed
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(
                f"Unknown checkpoint path. Did you specify the correct path to the policy you trained?"
            )
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        _download_file(download_url, download_target)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
