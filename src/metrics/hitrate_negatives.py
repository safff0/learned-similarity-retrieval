from typing import Any

import torch

from src.metrics.base_metric import BaseMetric
from src.registry import register


@register("metric")
class HitrateNegatives(BaseMetric):
    def __init__(self, k: int, n_negatives: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k
        self._n_negatives = n_negatives

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        history_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[float, int]:
        item_count = logits.shape[-1]
        valid = loss_mask.bool()
 
        # preprocess
        positions = valid.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return 0.0, 0
        b_idx = positions[:, 0]
        t_idx = positions[:, 1]
        target_items = target[b_idx, t_idx].long()
        N = target_items.numel()
        if N == 0:
            return 0.0, 0
        
        # make allowed mask
        allowed = torch.ones((N, item_count), dtype=torch.float32, device=logits.device)
        allowed[:, 0] = 0.0
        allowed[torch.arange(N, device=logits.device), target_items] = 0.0
        histories = history_ids[b_idx].long()
        row_idx = torch.arange(N, device=logits.device).unsqueeze(1)
        row_idx = row_idx.expand(histories.size())
        allowed[row_idx, histories] = 0.0
        
        # leave only targets that have enough negatives
        enough = allowed.sum(dim=-1) >= self._n_negatives
        if not enough.any():
            return 0.0, 0
        allowed = allowed[enough]
        b_idx = b_idx[enough]
        t_idx = t_idx[enough]
        target_items = target_items[enough]
        N = target_items.numel()

        # count hitrate 1 vs n_negatives
        negatives = torch.multinomial(
            allowed,
            num_samples=self._n_negatives,
            replacement=False
        )
        candidates = torch.cat([negatives, target_items.unsqueeze(1)], dim=1)
        scores = logits[b_idx, t_idx]
        candidate_scores = scores.gather(1, candidates)
        topk_idx = candidate_scores.topk(min(self._k, self._n_negatives + 1), dim=-1).indices
        hits = (topk_idx == self._n_negatives).any(dim=-1)
        hitrate = hits.float().mean().item()
        return hitrate, N