"""AOM-Ridge: dual / kernel Ridge with operator-mixture preprocessing.

Self-contained package under ``bench/AOM_v0/Ridge``. Operators are imported
from ``bench/AOM_v0/aompls``; this package never modifies them.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AOMRidgeRegressor",
    "AOMRidgeClassifier",
    "AOMMultiKernelRidge",
    "AOMKernelizer",
    "AOMRidgePLS",
    "AOMRidgePLSCV",
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
    if name == "AOMRidgePLS":
        return import_module(".aom_ridge_pls", __name__).AOMRidgePLS
    if name == "AOMRidgePLSCV":
        return import_module(".aom_ridge_pls", __name__).AOMRidgePLSCV
    raise AttributeError(f"module 'aomridge' has no attribute {name!r}")
