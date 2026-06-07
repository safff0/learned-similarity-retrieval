from typing import Any

from src.datasets.base_dataset import UserHistoryBatch
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(
        self,
        batch: UserHistoryBatch,
        metrics: MetricTracker,
    ) -> tuple[UserHistoryBatch, dict[str, Any]]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The model is called as ``model(batch)`` and is expected to return
        a dict that includes a single aggregated loss under the key 'loss'.

        Args:
            batch (UserHistoryBatch): batch from the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (UserHistoryBatch): the (device-resident) batch.
            outputs (dict): model outputs (loss + auxiliary tensors).
        """
        batch = self.move_batch_to_device(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(batch)

        if self.is_train:
            outputs["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.trainer.loss_names:
            if loss_name in outputs:
                metrics.update(loss_name, outputs[loss_name].item())

        # Flatten batch + outputs so metrics receive a single **kwargs dict.
        # ``vars(batch)`` is a shallow view of the dataclass fields — no copy.
        # Outputs override batch on key collision.
        flat = {**vars(batch), **outputs, "model": self.model}
        for met in metric_funcs:
            value, n = met(**flat)
            if n > 0:
                metrics.update(met.alias, value, n=n)
        return batch, outputs

    def _log_batch(
        self,
        batch_idx: int,
        batch: UserHistoryBatch,
        outputs: dict[str, Any],
        mode: str = "train",
    ) -> None:
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (UserHistoryBatch): batch after going through 'process_batch'.
            outputs (dict): model outputs (loss + auxiliary tensors).
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from your batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
