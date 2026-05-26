from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn


def build_dataloaders(cfg) -> dict:
    from src.datasets import _DATASETS

    dataset_cls = _DATASETS[cfg.data.name]
    dataset_kwargs = {
        k: v for k, v in cfg.data.items() if k not in {"name", "batch_size", "num_workers"}
    }

    train_dataset = dataset_cls(partition="train", **dataset_kwargs)
    val_dataset = dataset_cls(partition="val", **dataset_kwargs)

    common = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )
    return {
        "train": DataLoader(train_dataset, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_dataset, shuffle=False, **common),
    }
