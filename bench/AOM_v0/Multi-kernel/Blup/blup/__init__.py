"""AOM-BLUP: prediction-decomposition layer over AOM-MKM.

Wraps an ``AOMMultiKernelMixedModel`` to expose per-block prediction
contributions. The decomposition identity ``sum components == predict``
holds by construction.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["AOMMultiKernelBLUP"]


def __getattr__(name: str):
    if name == "AOMMultiKernelBLUP":
        return import_module(".estimator", __name__).AOMMultiKernelBLUP
    raise AttributeError(f"module 'blup' has no attribute {name!r}")
