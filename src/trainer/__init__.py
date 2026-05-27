from collections.abc import Iterable
from dataclasses import asdict

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

from src.trainer.base_trainer import BaseTrainer
from src.trainer.trainer import Trainer
from src.utils.config import Config


def build_optimizer(config: Config, params: Iterable[Parameter]) -> Optimizer:
    """
    Build an optimizer by name from :mod:`torch.optim`.

    ``config.optimizer.name`` is the class name (e.g. ``Adam``, ``SGD``,
    ``AdamW``); remaining keys are passed as kwargs to the constructor.
    """
    spec = config.optimizer.params
    name = config.optimizer.name
    cls = getattr(torch.optim, name)
    return cls(params, **spec)


__all__ = ["BaseTrainer", "Trainer", "build_optimizer"]
