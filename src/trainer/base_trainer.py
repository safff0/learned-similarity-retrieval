from abc import abstractmethod
from pathlib import Path

import torch

from src.metrics.tracker import MetricTracker
from src.utils.io_utils import save_checkpoint


class BaseTrainer:
    """Owns the epoch loop, checkpointing, and metric tracking.

    Subclasses must implement ``process_batch`` (per-batch forward / loss / step)
    and ``_log_batch`` (anything extra to log beyond scalars).
    """

    def __init__(self, model, optimizer, dataloaders, metrics, config, writer, device):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.config = config
        self.writer = writer
        self.device = device

        self.epochs: int = config.trainer.num_epochs
        self.log_interval: int = config.trainer.log_interval
        self.ckpt_dir = Path(config.trainer.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.is_train = False

    def train(self) -> None:
        try:
            for epoch in range(1, self.epochs + 1):
                self._train_epoch(epoch)
                for partition in self.dataloaders:
                    if partition == "train":
                        continue
                    self._evaluation_epoch(epoch, partition)
                save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch}.ckpt",
                    self.model,
                    self.optimizer,
                    epoch,
                )
        except KeyboardInterrupt:
            save_checkpoint(
                self.ckpt_dir / "interrupt.ckpt", self.model, self.optimizer, -1
            )
            raise

    def _train_epoch(self, epoch: int) -> dict:
        self.is_train = True
        self.model.train()
        tracker = self._make_tracker("train")
        loader = self.dataloaders["train"]
        for i, batch in enumerate(loader):
            batch = self.process_batch(batch, tracker)
            if i % self.log_interval == 0:
                self._log_batch(i, batch, mode="train")
                step = (epoch - 1) * len(loader) + i
                for k, v in tracker.result().items():
                    self.writer.add_scalar(f"train/{k}", v, step)
                print(f"[epoch {epoch} step {i}] {tracker.result()}")
        return tracker.result()

    def _evaluation_epoch(self, epoch: int, partition: str) -> dict:
        self.is_train = False
        self.model.eval()
        tracker = self._make_tracker("inference")
        with torch.no_grad():
            for i, batch in enumerate(self.dataloaders[partition]):
                batch = self.process_batch(batch, tracker)
                self._log_batch(i, batch, mode=partition)
        for k, v in tracker.result().items():
            self.writer.add_scalar(f"{partition}/{k}", v, epoch)
        print(f"[epoch {epoch} {partition}] {tracker.result()}")
        return tracker.result()

    def _make_tracker(self, stage: str) -> MetricTracker:
        loss_names = list(self.config.trainer.loss_names)
        metric_names = [m.name for m in self.metrics[stage]]
        return MetricTracker(*loss_names, *metric_names)

    def move_batch_to_device(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        return batch

    @abstractmethod
    def process_batch(self, batch, metrics: MetricTracker):
        raise NotImplementedError

    @abstractmethod
    def _log_batch(self, batch_idx: int, batch, mode: str = "train") -> None:
        raise NotImplementedError
