"""Kernelizer adapter for MKM.

Delegates to the MKR-side ``AOMKernelizer`` (centred + trace-normalised
block kernels) so we don't duplicate the implementation. The MKM package
only needs the trained block kernels and the ability to compute cross
kernels using training-side moments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Import via the standard sys-path setup done in conftest.py / runtime.
from aomridge.kernelizer import (
    AOMKernelizer,
    BlockKernelStats,
    kernel_alignment_matrix,
)

__all__ = [
    "AOMKernelizer",
    "BlockKernelStats",
    "kernel_alignment_matrix",
]
