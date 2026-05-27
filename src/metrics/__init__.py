import importlib
import pkgutil

from src.metrics.tracker import MetricTracker
from src.registry import build
from src.utils.config import Config

for _, _mod_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_mod_name}")


def build_metrics(config: Config) -> dict[str, list]:
    """
    Build metric instances grouped by stage.

    Reads ``config.metrics.train`` and ``config.metrics.inference`` (both lists
    of specs with ``name:`` keys).
    """
    out = {"train": [], "inference": []}
    for stage in ("train", "inference"):
        entries = getattr(config.metrics, stage, []) or []
        for entry in entries:
            out[stage].append(build("metric", entry))
    return out


__all__ = ["MetricTracker", "build_metrics"]
