"""AOM-Ridge family public entry point for `nirs4all`.

Re-exports the canonical AOM-Ridge implementation from
``nirs4all.operators.models._aom_nirs.ridge`` (vendored copy of the
``aom-nirs`` package; once ``aom-nirs`` is on PyPI this file will
switch to ``from aom_nirs.ridge import ...``).

The AOM-Ridge family is the Talanta paper's strongest empirical
contribution: ``AOMRidgeBlender`` reaches median RMSEP ratio 0.918
versus Ridge-default on the strict 32-dataset intersection
(Wilcoxon Holm-corrected p = 2.6e-4, 27/32 wins).
"""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.ridge import (
    AOMKernelizer,
    AOMLocalRidge,
    AOMMultiBranchMKL,
    AOMMultiKernelRidge,
    AOMRidgeAutoSelector,
    AOMRidgeBlender,
    AOMRidgeClassifier,
    AOMRidgePLS,
    AOMRidgePLSCV,
    AOMRidgeRegressor,
)

__all__ = [
    "AOMRidgeRegressor",
    "AOMRidgeClassifier",
    "AOMRidgeBlender",
    "AOMRidgeAutoSelector",
    "AOMRidgePLS",
    "AOMRidgePLSCV",
    "AOMMultiKernelRidge",
    "AOMKernelizer",
    "AOMMultiBranchMKL",
    "AOMLocalRidge",
]
