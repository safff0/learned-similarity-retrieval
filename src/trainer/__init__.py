import torch

from src.trainer.base_trainer import BaseTrainer
from src.trainer.trainer import Trainer


def build_optimizer(config, params):
    """
    Build an optimizer by name from :mod:`torch.optim`.

    ``config.optimizer.name`` is the class name (e.g. ``Adam``, ``SGD``,
    ``AdamW``); remaining keys are passed as kwargs to the constructor.
    """
    spec = dict(config.optimizer)
    name = spec.pop("name")
    cls = getattr(torch.optim, name)
    return cls(params, **spec)


__all__ = ["BaseTrainer", "Trainer", "build_optimizer"]
