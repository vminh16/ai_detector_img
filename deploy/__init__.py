"""Deploy/runtime package for the active web inference stack."""

from __future__ import annotations

from typing import Any

__all__ = ["InferencePipeline", "PredictionResult"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import InferencePipeline, PredictionResult

        exports = {
            "InferencePipeline": InferencePipeline,
            "PredictionResult": PredictionResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
