from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, k: int = 10, name: str | None = None):
        super().__init__(name)
        self.k = k

    def __call__(self, **batch):
        raise NotImplementedError(
            "Compute a retrieval metric (recall@k, ndcg@k, ...) from batch outputs."
        )
