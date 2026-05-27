import logging
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.registry import register
from src.utils.io_utils import ROOT_PATH, read_json, write_json

logger = logging.getLogger(__name__)


@register("dataset")
class ExampleDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self,
        input_length: int,
        n_classes: int,
        dataset_length: int,
        partition: str = "train",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            partition (str): partition name (used for the on-disk cache path).
        """
        index_path = ROOT_PATH / "data" / "example" / partition / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(
                input_length, n_classes, dataset_length, partition
            )

        super().__init__(index, *args, **kwargs)

    def _create_index(
        self, input_length: int, n_classes: int, dataset_length: int, partition: str
    ) -> list[dict[str, Any]]:
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            partition (str): partition name.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "example" / partition
        data_path.mkdir(exist_ok=True, parents=True)

        # to get pretty object names
        number_of_zeros = int(np.log10(dataset_length)) + 1

        # In this example, we create a synthesized dataset. However, in real
        # tasks, you should process dataset metadata and append it
        # to index.
        logger.info("Creating Example Dataset")
        for i in tqdm(range(dataset_length)):
            example_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            example_data = torch.randn(input_length)
            example_label = torch.randint(n_classes, size=(1,)).item()
            torch.save(example_data, example_path)

            index.append({"path": str(example_path), "label": example_label})

        write_json(index, str(data_path / "index.json"))

        return index
