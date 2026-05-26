class MetricTracker:
    def __init__(self, *keys: str):
        self._keys: list[str] = list(keys)
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self._sums = {k: 0.0 for k in self._keys}
        self._counts = {k: 0 for k in self._keys}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self._sums:
            self._keys.append(key)
            self._sums[key] = 0.0
            self._counts[key] = 0
        self._sums[key] += float(value) * n
        self._counts[key] += n

    def avg(self, key: str) -> float:
        if self._counts.get(key, 0) == 0:
            return 0.0
        return self._sums[key] / self._counts[key]

    def result(self) -> dict[str, float]:
        return {k: self.avg(k) for k in self._keys}

    def keys(self) -> list[str]:
        return list(self._keys)
