"""AOM-Ridge: dual / kernel Ridge with operator-mixture preprocessing.

Self-contained package under ``bench/AOM_v0/Ridge``. Operators are imported
from ``bench/AOM_v0/aompls``; this package never modifies them.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["AOMRidgeRegressor"]


def __getattr__(name: str):
    if name == "AOMRidgeRegressor":
        return import_module(".estimators", __name__).AOMRidgeRegressor
    raise AttributeError(f"module 'aomridge' has no attribute {name!r}")
