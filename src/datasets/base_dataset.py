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
    history_ids: torch.Tensor           # item ids in user's history
    history_features: torch.Tensor      # features of items in user's history
    target: int | None = None           # next item in user's future history
    target_feedback: int | None = None  # feedback if present
    timestamp: int | None = None


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