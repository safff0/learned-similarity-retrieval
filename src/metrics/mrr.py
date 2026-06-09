from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class MRR(BaseMetric):
    """Mean Reciprocal Rank with history filtering. Retrieves the entire item
    catalog through ``model.retrieve_topk`` so the rank is exact."""

    def __init__(
        self,
        last_only: bool = False,
        filter_history: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_only = last_only
        self._filter_history = filter_history

    def __call__(
        self,
        retrieval_queries: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        history_ids: torch.Tensor,
        user_id: torch.Tensor,
        model: Any,
        next_retrieval_queries: torch.Tensor | None = None,
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
        if self._last_only and next_retrieval_queries is not None:
            query_embeddings = next_retrieval_queries[b_idx]
        else:
            query_embeddings = retrieval_queries[b_idx, t_idx]
        target_items = target[b_idx, t_idx].long()
        query_user_ids = user_id[b_idx].long()
        n = target_items.numel()
        if n == 0:
            return 0.0, 0

        _, candidate_ids = model.retrieve_topk(
            query_embeddings=query_embeddings,
            k=model.retrieval_item_count,
            user_ids=query_user_ids,
        )
        target_mask = candidate_ids == target_items.unsqueeze(1)
        valid_mask = candidate_ids != 0
        if self._filter_history:
            histories = history_ids[b_idx].long()
            invalid_mask = (candidate_ids.unsqueeze(-1) == histories.unsqueeze(1)).any(dim=-1)
            valid_mask = valid_mask & (~invalid_mask | target_mask)
        kept_rank = valid_mask.cumsum(dim=1)
        reciprocal = torch.where(
            target_mask & valid_mask,
            1.0 / kept_rank.clamp_min(1).float(),
            torch.zeros_like(kept_rank, dtype=torch.float32),
        ).amax(dim=1)
        return reciprocal.mean().item(), n
