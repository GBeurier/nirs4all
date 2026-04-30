"""AOM-MKM: multi-kernel mixed model with REML variance components.

Self-contained package under ``bench/aom_v0/Multi-kernel/MkM``. Operators are imported
from ``bench/aom_v0/Multi-kernel/aompls``; the kernelizer mirrors the AOM-Ridge mkR
kernelizer.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["AOMMultiKernelMixedModel"]


def __getattr__(name: str):
    if name == "AOMMultiKernelMixedModel":
        return import_module(".estimator", __name__).AOMMultiKernelMixedModel
    raise AttributeError(f"module 'mkm' has no attribute {name!r}")
