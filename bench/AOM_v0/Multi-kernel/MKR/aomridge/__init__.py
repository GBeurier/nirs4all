"""AOM-Ridge: dual / kernel Ridge with operator-mixture preprocessing.

Self-contained package under ``bench/aom_v0/Multi-kernel/MKR``. Operators are imported
from ``bench/aom_v0/Multi-kernel/aompls``; this package never modifies them.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AOMRidgeRegressor",
    "AOMRidgeClassifier",
    "AOMMultiKernelRidge",
    "AOMKernelizer",
]


def __getattr__(name: str):
    if name == "AOMRidgeRegressor":
        return import_module(".estimators", __name__).AOMRidgeRegressor
    if name == "AOMRidgeClassifier":
        return import_module(".classification", __name__).AOMRidgeClassifier
    if name == "AOMMultiKernelRidge":
        return import_module(".mkr_estimator", __name__).AOMMultiKernelRidge
    if name == "AOMKernelizer":
        return import_module(".kernelizer", __name__).AOMKernelizer
    raise AttributeError(f"module 'aomridge' has no attribute {name!r}")
