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
    "AOMRidgeAutoSelector",
    "AOMRidgeBlender",
    "AOMMultiBranchMKL",
    "AOMLocalRidge",
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
    if name == "AOMRidgeAutoSelector":
        return import_module(".auto_selector", __name__).AOMRidgeAutoSelector
    if name == "AOMRidgeBlender":
        return import_module(".blender", __name__).AOMRidgeBlender
    if name == "AOMMultiBranchMKL":
        return import_module(".multi_branch_mkl", __name__).AOMMultiBranchMKL
    if name == "AOMLocalRidge":
        return import_module(".local_ridge", __name__).AOMLocalRidge
    raise AttributeError(f"module 'aomridge' has no attribute {name!r}")
