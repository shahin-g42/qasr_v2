"""Standalone decision core; importing this namespace does not import Torch."""

from importlib import import_module

from .codes import (
    AUTOJEV_MODEL,
    AUTOJEV_REVISION,
    AUTOJEV_SOURCE_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    MAX_OPTIONS,
    discover_codes,
)

__version__ = "0.1.0"
__all__ = [
    "AUTOJEV_MODEL", "AUTOJEV_REVISION", "AUTOJEV_SOURCE_REVISION", "BASE_MODEL",
    "BASE_REVISION", "MAX_OPTIONS", "DecisionModel", "PreparedBatch", "build_meta_model",
    "discover_codes", "load_full_state", "load_model",
]


def __getattr__(name: str):
    if name == "PreparedBatch":
        return getattr(import_module(".processing", __name__), name)
    if name in {"DecisionModel", "build_meta_model", "load_full_state", "load_model"}:
        return getattr(import_module(".modeling", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
