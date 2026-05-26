from src.datasets.example import ExampleDataset

_DATASETS = {
    "example": ExampleDataset,
}

from src.datasets.data_utils import build_dataloaders  # noqa: E402

__all__ = ["ExampleDataset", "build_dataloaders", "_DATASETS"]
