from src.model.baseline_model import BaselineModel

_MODELS = {
    "baseline": BaselineModel,
}


def build_model(cfg):
    name = cfg.model.name
    kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    return _MODELS[name](**kwargs)


__all__ = ["BaselineModel", "build_model", "_MODELS"]
