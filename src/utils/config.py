from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObjectConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetPartitionConfig:
    name: str = "ExampleDataset"
    partition: str = "train"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    batch_size: int = 10
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class DataConfig:
    partitions: dict[str, DatasetPartitionConfig] = field(
        default_factory=lambda: {
            "train": DatasetPartitionConfig(
                name="ExampleDataset",
                partition="train",
                params={
                    "input_length": 1024,
                    "n_classes": 10,
                    "dataset_length": 100,
                },
            ),
            "val": DatasetPartitionConfig(
                name="ExampleDataset",
                partition="val",
                params={
                    "input_length": 1024,
                    "n_classes": 10,
                    "dataset_length": 50,
                },
            ),
        }
    )
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


@dataclass
class ModelConfig(ObjectConfig):
    name: str = "BaselineModel"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_feats": 1024,
            "fc_hidden": 512,
            "n_class": 10,
        }
    )


@dataclass
class OptimizerConfig(ObjectConfig):
    name: str = "Adam"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 3.0e-4,
        }
    )


@dataclass
class MetricsConfig:
    device: str = "auto"

    # Could be names only:
    train: list[str] = field(default_factory=list)
    inference: list[str] = field(default_factory=list)

    # Or, if you later want metrics with params, use:
    # train: list[ObjectConfig] = field(default_factory=list)
    # inference: list[ObjectConfig] = field(default_factory=list)


@dataclass
class TrainerConfig:
    device: str = "auto"
    seed: int = 1
    n_epochs: int = 10
    epoch_len: int | None = None
    log_step: int = 50
    save_period: int = 5
    save_dir: str = "./saved/default"
    tb_dir: str = "./runs/default"
    device_tensors: list[str] = field(
        default_factory=lambda: ["data_object", "labels"]
    )
    loss_names: list[str] = field(default_factory=lambda: ["loss"])
    max_grad_norm: float | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
