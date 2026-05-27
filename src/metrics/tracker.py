import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys: str, writer: SummaryWriter | None = None) -> None:
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (SummaryWriter | None): experiment tracker. Not used in
                this code version. Can be used to log metrics from each batch.
        """
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
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._data.average)

    def keys(self) -> pd.Index:
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
