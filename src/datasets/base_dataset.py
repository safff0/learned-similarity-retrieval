import random

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Index-driven base dataset.

    Subclasses produce an ``index`` (list of dicts with at least ``path`` and
    ``label`` keys); this class handles shuffling/limiting and the
    load + preprocess hooks.
    """

    def __init__(self, index, limit=None, shuffle_index=False):
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index)
        index = self._sort_index(index)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        record = self._index[idx]
        data_object = self.load_object(record["path"])
        data_object = self.preprocess_data(data_object)
        return {"data_object": data_object, "labels": record["label"]}

    def load_object(self, path):
        return torch.load(path)

    def preprocess_data(self, data):
        return data

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "every index entry needs a 'path' field"
            assert "label" in entry, "every index entry needs a 'label' field"

    @staticmethod
    def _filter_records_from_dataset(index):
        return index

    @staticmethod
    def _sort_index(index):
        return index

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            index = list(index)
            random.Random(42).shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
