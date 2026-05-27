from collections.abc import Iterator
from itertools import repeat
from dataclasses import asdict
from typing import Any

from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn
from src.registry import build
from src.utils.init_utils import set_worker_seed
from src.utils.config import Config


def inf_loop(dataloader: DataLoader) -> Iterator[dict[str, Any]]:
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config: Config, device: str) -> dict[str, DataLoader]:
    """
    Create dataloaders for each of the dataset partitions.

    Each entry under ``config.data.partitions`` is a dataset spec (with a
    ``name:`` key resolved against the registry) plus the dataset's kwargs.

    Args:
        config (DictConfig): experiment config.
        device (str): device (kept for parity; not used here).
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
    """
    dataloaders = {}
    for partition_name, partition_spec in config.data.partitions.items():
        dataset = build("dataset", partition_spec)

        assert config.data.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.data.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        dataloaders[partition_name] = DataLoader(
            dataset,
            collate_fn=collate_fn,
            drop_last=(partition_name == "train"),
            shuffle=(partition_name == "train"),
            worker_init_fn=set_worker_seed,
            **asdict(config.data.dataloader),
        )

    return dataloaders
