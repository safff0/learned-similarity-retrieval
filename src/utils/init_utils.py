import argparse
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(argv: list[str]) -> DictConfig:
    """
    Load YAML config from ``--config <path>`` and merge any remaining
    dot-notation overrides (e.g. ``optimizer.lr=5e-4``) on top.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args(argv)
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: str) -> str:
    """
    Resolve a device name. ``"auto"`` picks cuda if available; an explicit
    ``"cuda"`` falls back to cpu with a warning if cuda is unavailable.
    """
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        return "cpu"
    return name
