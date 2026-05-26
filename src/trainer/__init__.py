from torch import optim

from src.trainer.base_trainer import BaseTrainer
from src.trainer.trainer import Trainer

_OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


def build_optimizer(cfg, params):
    name = cfg.optimizer.name
    kwargs = {k: v for k, v in cfg.optimizer.items() if k != "name"}
    return _OPTIMIZERS[name](params, **kwargs)


__all__ = ["BaseTrainer", "Trainer", "build_optimizer", "_OPTIMIZERS"]
