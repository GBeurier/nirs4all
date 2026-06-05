"""
Analysis utilities for visualization.

Exposed lazily (PEP 562): ``transfer`` imports matplotlib and ``shap`` imports
shap/numba at module load, so they are imported only on first attribute access
rather than eagerly at ``import nirs4all``. Accessing ``ShapAnalyzer`` raises the
underlying ImportError/AttributeError if shap is unavailable or numpy-incompatible.
"""
import importlib
from typing import Any

_LAZY_EXPORTS = {
    "BranchAnalyzer": "nirs4all.visualization.analysis.branch",
    "BranchSummary": "nirs4all.visualization.analysis.branch",
    "PreprocPCAEvaluator": "nirs4all.visualization.analysis.transfer",
    "ShapAnalyzer": "nirs4all.visualization.analysis.shap",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
