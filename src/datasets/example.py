from src.datasets.base_dataset import BaseDataset


class ExampleDataset(BaseDataset):
    def __init__(self, partition: str = "train", dataset_length: int = 100, **kwargs):
        index = self._create_index(partition, dataset_length)
        super().__init__(index, **kwargs)

    def _create_index(self, partition: str, dataset_length: int):
        raise NotImplementedError(
            "Build the data index here: a list of {'path': ..., 'label': ...} dicts "
            "(or whatever schema your retrieval task needs)."
        )
