from abc import abstractmethod
from typing import Any


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch: Any) -> float:
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()
