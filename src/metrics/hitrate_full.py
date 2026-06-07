from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class HitrateFull(BaseMetric):
    def __init__(
        self,
        k: int,
        last_only: bool = False,
        filter_history: bool = True,
        overfetch_factor: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._k = k
        self._last_only = last_only
        self._filter_history = filter_history
        self._overfetch_factor = overfetch_factor

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        history_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[float, int]:
        valid = loss_mask.bool()
        if self._last_only:
            last_valid = valid.cumsum(dim=1) == valid.sum(dim=1, keepdim=True)
            valid = valid & last_valid
        positions = valid.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return 0.0, 0

        b_idx = positions[:, 0]
        t_idx = positions[:, 1]
        target_items = target[b_idx, t_idx].long()
        n = target_items.numel()
        if n == 0:
            return 0.0, 0

        step_logits = logits[b_idx, t_idx]
        overfetch_k = min(
            step_logits.size(1),
            max(self._k + history_ids.size(1) + 1, self._k * self._overfetch_factor),
        )
        candidate_ids = torch.topk(step_logits, k=overfetch_k, dim=1).indices

        target_mask = candidate_ids == target_items.unsqueeze(1)
        valid_mask = candidate_ids != 0
        if self._filter_history:
            histories = history_ids[b_idx].long()
            invalid_mask = (candidate_ids.unsqueeze(-1) == histories.unsqueeze(1)).any(dim=-1)
            valid_mask = valid_mask & (~invalid_mask | target_mask)
        kept_rank = valid_mask.cumsum(dim=1)
        hits = (target_mask & valid_mask & (kept_rank <= self._k)).any(dim=1)
        return hits.float().mean().item(), n
