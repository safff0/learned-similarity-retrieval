class BaseMetric:
    def __init__(self, name: str | None = None):
        self.name = name if name is not None else type(self).__name__

    def __call__(self, **batch):
        raise NotImplementedError
