from __future__ import annotations

from core.base import Detector, Segmenter, VLM

_REGISTRY: dict[str, dict[str, type]] = {
    "detector": {},
    "segmenter": {},
    "vlm": {},
}

CATEGORY_BASE = {
    "detector": Detector,
    "segmenter": Segmenter,
    "vlm": VLM,
}


def register(category: str, name: str):
    """Decorator. Usage: @register("detector", "yolov8")"""
    assert category in _REGISTRY, f"Unknown category: {category}"

    def wrapper(cls):
        assert issubclass(cls, CATEGORY_BASE[category]), \
            f"{cls.__name__} must subclass {CATEGORY_BASE[category].__name__}"
        assert name not in _REGISTRY[category], \
            f"Duplicate registration: {category}/{name}"
        _REGISTRY[category][name] = cls
        return cls

    return wrapper


def create(category: str, name: str, **kwargs):
    """Factory. Returns instantiated model."""
    assert category in _REGISTRY, f"Unknown category: {category}"
    assert name in _REGISTRY[category], \
        f"Not found: {category}/{name}. Available: {list(_REGISTRY[category].keys())}"
    return _REGISTRY[category][name](**kwargs)


def get_class(category: str, name: str) -> type:
    """Return registered class without instantiating."""
    assert category in _REGISTRY, f"Unknown category: {category}"
    assert name in _REGISTRY[category], \
        f"Not found: {category}/{name}. Available: {list(_REGISTRY[category].keys())}"
    return _REGISTRY[category][name]


def list_models(category: str) -> list[str]:
    assert category in _REGISTRY, f"Unknown category: {category}"
    return list(_REGISTRY[category].keys())
