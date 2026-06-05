"""
Visualization tools for NIRS data analysis.

Symbols are exposed lazily (PEP 562): the submodules below import matplotlib (and,
for the shap analyzer, shap/numba) at module load, so they are imported only on
first attribute access rather than eagerly at ``import nirs4all``.
"""
import importlib
from typing import Any

# Public name -> submodule that defines it.
_LAZY_EXPORTS = {
    "PredictionAnalyzer": "nirs4all.visualization.predictions",
    "show_figures": "nirs4all.visualization.display",
    "PipelineDiagram": "nirs4all.visualization.pipeline_diagram",
    "plot_pipeline_diagram": "nirs4all.visualization.pipeline_diagram",
    "BranchAnalyzer": "nirs4all.visualization.analysis.branch",
    "BranchSummary": "nirs4all.visualization.analysis.branch",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
