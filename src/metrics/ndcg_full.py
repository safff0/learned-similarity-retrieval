from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class NDCGFull(BaseMetric):
    def __init__(
        self,
        k: int,
        last_only: bool = False,
        filter_history: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._k = k
        self._last_only = last_only
        self._filter_history = filter_history

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
        step_logits = logits[b_idx, t_idx]
        target_items = target[b_idx, t_idx].long()
        candidate_ids = torch.argsort(step_logits, dim=1, descending=True)

        target_mask = candidate_ids == target_items.unsqueeze(1)
        valid_mask = candidate_ids != 0
        if self._filter_history:
            histories = history_ids[b_idx].long()
            invalid_mask = (candidate_ids.unsqueeze(-1) == histories.unsqueeze(1)).any(dim=-1)
            valid_mask = valid_mask & (~invalid_mask | target_mask)
        kept_rank = valid_mask.cumsum(dim=1)
        # IDCG@K = 1 for a single relevant target, so NDCG = 1 / log2(rank + 1)
        # when target is found within top-K after history filtering; 0 otherwise.
        ndcg = torch.where(
            target_mask & valid_mask & (kept_rank <= self._k),
            1.0 / torch.log2(kept_rank.clamp_min(1).float() + 1.0),
            torch.zeros_like(kept_rank, dtype=torch.float32),
        ).amax(dim=1)
        n = ndcg.numel()
        return ndcg.mean().item(), n
