from src.metrics.base_metric import BaseMetric
from src.metrics.example import ExampleMetric
from src.metrics.tracker import MetricTracker

_METRICS = {
    "example": ExampleMetric,
}


def build_metrics(cfg) -> dict:
    out = {"train": [], "inference": []}
    for stage in ("train", "inference"):
        entries = cfg.metrics.get(stage, []) or []
        for entry in entries:
            entry = dict(entry)
            name = entry.pop("name")
            out[stage].append(_METRICS[name](**entry))
    return out


__all__ = ["BaseMetric", "ExampleMetric", "MetricTracker", "build_metrics", "_METRICS"]
