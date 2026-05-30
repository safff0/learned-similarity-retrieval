import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(
        self,
        *keys: str,
        writer: SummaryWriter | None = None,
    ) -> None:
        """
        Args:
            *keys (list[str]): metric identifiers — must be unique. For
                user-defined metrics this is ``BaseMetric.alias``; loss
                names and ``grad_norm`` use their own string.
            writer (SummaryWriter | None): experiment tracker. Not used in
                this code version. Can be used to log metrics from each batch.
        """
        duplicates = [k for k in set(keys) if keys.count(k) > 1]
        if duplicates:
            raise ValueError(
                f"MetricTracker got duplicate keys {duplicates}. Set a distinct "
                "`alias` on each metric in the config so they don't collide."
            )
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self) -> None:
        """
        Reset all metrics after epoch end.
        """
        self._data.loc[:, :] = 0

    def update(self, key: str, value: float, n: int = 1) -> None:
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key: str) -> float:
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def result(self) -> dict[str, float]:
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict mapping metric key to average value.
        """
        return dict(self._data.average)

    def keys(self) -> pd.Index:
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
