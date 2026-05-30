from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class Hitrate(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[float, int]:
        valid = loss_mask.bool()
        n = int(valid.sum().item())
        if n == 0:
            return 0.0, 0
        topk = logits.topk(self._k, dim=-1).indices
        hits = (topk == target.unsqueeze(-1)).any(dim=-1)
        return float((hits & valid).sum().item()) / n, n
