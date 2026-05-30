import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from abc import abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class UserHistoryItem:
    user_id: int
    history_ids: torch.Tensor                       # (L,) input ids = seq[:-1]
    history_features: torch.Tensor                  # (L, F) features for inputs
    target: torch.Tensor | None = None              # (L,) shifted next ids = seq[1:]
    target_feedback: torch.Tensor | None = None     # (L,) shifted next ratings
    loss_mask: torch.Tensor | None = None           # (L,) bool, True = score this target
    timestamp: int | None = None


@dataclass
class UserHistoryBatch:
    user_id: torch.Tensor           # (B,)
    history_ids: torch.Tensor       # (B, L_max)
    history_features: torch.Tensor  # (B, L_max, F)
    target: torch.Tensor            # (B, L_max)
    target_feedback: torch.Tensor   # (B, L_max)
    mask: torch.Tensor              # (B, L_max) bool, True = valid input (not padding)
    loss_mask: torch.Tensor         # (B, L_max) bool, True = target counts toward loss/metrics
    timestamp: torch.Tensor         # (B,)


class BaseDataset(Dataset):
    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
    ) -> None:
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def load_object(self, path: str | Path) -> torch.Tensor:
        data_object = torch.load(path)
        return data_object

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]], limit: int | None, shuffle_index: bool
    ) -> list[dict[str, Any]]:
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index

    @abstractmethod
    def __getitem__(self, ind: int) -> UserHistoryItem:
        pass