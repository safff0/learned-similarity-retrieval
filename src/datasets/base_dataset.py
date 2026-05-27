import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.
    """

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
