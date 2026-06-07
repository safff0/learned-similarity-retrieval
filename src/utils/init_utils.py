import argparse
import os
import random
import logging
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from src.utils.config import Config

logger = logging.getLogger(__name__)


def load_config(argv: list[str]) -> Config:
    """
    Load YAML config from ``--config <path>`` and merge any remaining
    dot-notation overrides (e.g. ``optimizer.lr=5e-4``) on top.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args(argv)
    schema = OmegaConf.structured(Config)
    raw_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, raw_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def set_worker_seed(worker_id: int) -> None:
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
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
    result = "cpu"
    if name == "auto":
        if torch.cuda.is_available():
            result = "cuda"
        elif torch.mps.is_available():
            result = "mps"
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
    if name == "mps" and not torch.mps.is_available():
        logger.warning("mps requested but unavailable; falling back to cpu")
    logger.info(f"torch device resolved to [{result}]")
    if result == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return result


def init_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
