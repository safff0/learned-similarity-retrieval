from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class NDCG(BaseMetric):
    """NDCG@K with history filtering. With a single relevant target the
    IDCG@K is 1, so the metric simplifies to ``1/log2(rank+1)`` when the
    target lands in top-K after history filtering, else 0."""

    def __init__(
        self,
        k: int,
        overfetch_factor: int = 4,
        last_only: bool = False,
        filter_history: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._k = k
        self._overfetch_factor = overfetch_factor
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

        overfetch_k = min(
            model.retrieval_item_count,
            max(
                self._k + history_ids.size(1) + 1,
                self._k * self._overfetch_factor,
            ),
        )
        cache_key = (
            int(query_embeddings.data_ptr()),
            tuple(query_embeddings.shape),
            int(query_user_ids.data_ptr()),
        )
        retrieval_cache = getattr(model, "_metric_retrieval_cache", None)
        cached_ids = None
        cached_k = -1
        if retrieval_cache is not None:
            cached = retrieval_cache.get(cache_key)
            if cached is not None:
                cached_k, cached_ids = cached
        if cached_ids is None or cached_k < overfetch_k:
            _, candidate_ids = model.retrieve_topk(
                query_embeddings=query_embeddings,
                k=overfetch_k,
                user_ids=query_user_ids,
            )
            retrieval_cache = getattr(model, "_metric_retrieval_cache", None)
            if retrieval_cache is None:
                retrieval_cache = {}
                setattr(model, "_metric_retrieval_cache", retrieval_cache)
            retrieval_cache[cache_key] = (overfetch_k, candidate_ids)
        else:
            candidate_ids = cached_ids[:, :overfetch_k]

        target_mask = candidate_ids == target_items.unsqueeze(1)
        valid_mask = candidate_ids != 0
        if self._filter_history:
            histories = history_ids[b_idx].long()
            invalid_mask = (candidate_ids.unsqueeze(-1) == histories.unsqueeze(1)).any(dim=-1)
            valid_mask = valid_mask & (~invalid_mask | target_mask)
        kept_rank = valid_mask.cumsum(dim=1)
        ndcg = torch.where(
            target_mask & valid_mask & (kept_rank <= self._k),
            1.0 / torch.log2(kept_rank.clamp_min(1).float() + 1.0),
            torch.zeros_like(kept_rank, dtype=torch.float32),
        ).amax(dim=1)
        return ndcg.mean().item(), n

    def prepare(self, model: Any, **kwargs: Any) -> None:
        setattr(model, "_metric_retrieval_cache", {})
        if hasattr(model, "prepare_retrieval_index"):
            model.prepare_retrieval_index()

    def cleanup(self, model: Any, **kwargs: Any) -> None:
        if hasattr(model, "_metric_retrieval_cache"):
            delattr(model, "_metric_retrieval_cache")
        if hasattr(model, "clear_retrieval_index"):
            model.clear_retrieval_index()
