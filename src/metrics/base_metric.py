from abc import abstractmethod
from typing import Any


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(
        self,
        alias: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            alias (str | None): identifier used as the tracker key AND the
                display label in logs/tensorboard. Must be unique across
                metrics within the same stage. Defaults to the class name,
                so multiple instances of the same metric class MUST set
                distinct aliases via config (e.g. "HR@5", "HR@10").
        """
        self.alias = alias if alias is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch: Any) -> tuple[float, int]:
        """
        Defines metric calculation logic for a given batch.

        Returns:
            (value, n): the metric value AND the number of eval items
                ``value`` was computed over. The tracker uses ``n`` to
                weight the running mean so batches with different counts
                of valid eval positions aggregate exactly.
        """
        raise NotImplementedError()
