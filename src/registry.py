from collections.abc import Callable
from dataclasses import asdict
from typing import Any


_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str) -> Callable[[type], type]:
    """Decorator: register a class under ``kind`` keyed by its class name.

    Example:
        @register("model")
        class BaselineModel(nn.Module): ...
    """

    def deco(cls):
        _REGISTRY.setdefault(kind, {})[cls.__name__] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    """Look up a registered class by ``kind`` + ``name`` (no instantiation)."""
    if kind not in _REGISTRY or name not in _REGISTRY[kind]:
        available = sorted(_REGISTRY.get(kind, {}).keys())
        raise KeyError(f"unknown {kind} '{name}'; registered: {available}")
    return _REGISTRY[kind][name]


def build(kind: str, spec: Any) -> Any:
    """Instantiate a registered class from a config spec.

    The spec is a dict-like with a ``name`` key (resolved against the registry)
    plus remaining keys passed as kwargs to the class constructor.
    """
    name = spec.name
    spec = spec.params
    cls = get(kind, name)
    return cls(**spec)


def registered(kind: str) -> list[str]:
    """List names registered under a kind (useful for debugging configs)."""
    return sorted(_REGISTRY.get(kind, {}).keys())
