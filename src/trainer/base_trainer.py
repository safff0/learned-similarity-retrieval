from abc import abstractmethod
import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.datasets.base_dataset import UserHistoryBatch
from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.config import Config

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model: nn.Module,
        metrics: dict[str, list],
        optimizer: Optimizer,
        config: Config,
        device: str,
        dataloaders: dict[str, DataLoader],
        writer: SummaryWriter | None,
        epoch_len: int | None = None,
    ) -> None:
        """
        Args:
            model (nn.Module): PyTorch model. Expected to return a dict that
                includes a 'loss' key during training (no separate criterion
                module in this template).
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            writer (SummaryWriter): TensorBoard writer.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device

        self.log_step = config.trainer.log_step or 50

        self.model = model
        self.optimizer = optimizer

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = self.cfg_trainer.save_period

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.trainer.loss_names,
            "grad_norm",
            *[m.alias for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.trainer.loss_names,
            *[m.alias for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir
        self.checkpoint_dir = Path(config.trainer.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            logger.warning("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self) -> None:
        """
        Full training logic: training model for an epoch, evaluating it on
        non-train partitions, and saving checkpoints periodically.
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                logger.info("    %-15s: %s", key, value)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        last_train_metrics = {}
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            batch, outputs = self.process_batch(
                batch,
                metrics=self.train_metrics,
            )

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                global_step = (epoch - 1) * self.epoch_len + batch_idx
                self._log_scalars(self.train_metrics, "train", global_step)
                self._log_batch(batch_idx, batch, outputs)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = dict(last_train_metrics)

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch: int, part: str, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch, outputs = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self._log_scalars(self.evaluation_metrics, part, epoch * self.epoch_len)
            self._log_batch(
                batch_idx, batch, outputs, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def move_batch_to_device(self, batch: UserHistoryBatch) -> UserHistoryBatch:
        """
        Move every tensor-valued field of the batch onto the trainer's device.

        Args:
            batch: UserHistoryBatch produced by the dataloader's collate_fn.
        Returns:
            batch: the same dataclass with its tensor fields on device.
        """
        for f in dataclasses.fields(batch):
            value = getattr(batch, f.name)
            if isinstance(value, torch.Tensor):
                setattr(batch, f.name, value.to(self.device))
        return batch

    def _clip_grad_norm(self) -> None:
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.cfg_trainer.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg_trainer.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: float = 2) -> float:
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    @abstractmethod
    def _log_batch(
        self,
        batch_idx: int,
        batch: UserHistoryBatch,
        outputs: dict[str, Any],
        mode: str = "train",
    ) -> None:
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (UserHistoryBatch): batch after going through 'process_batch'.
            outputs (dict): model outputs (loss + auxiliary tensors).
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, mode: str, step: int) -> None:
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
            mode (str): "train" or partition name.
            step (int): global step for the writer.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{mode}/{metric_name}",
                metric_tracker.avg(metric_name),
                step,
            )

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save the checkpoint.

        Args:
            epoch (int): current epoch number.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        torch.save(state, filename)
        logger.info("Saving checkpoint: %s ...", filename)
