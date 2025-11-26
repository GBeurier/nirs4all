"""PLS (Partial Least Squares) model operators for nirs4all.

This module provides PLS-based model operators that are sklearn-compatible
for use in nirs4all pipelines. Supports both NumPy and JAX backends.

Backward compatibility: All model classes are imported from their respective
modules and exposed here for convenience and legacy imports.

Classes
-------
PLSDA
    PLS Discriminant Analysis for classification tasks.
IKPLS
    Improved Kernel PLS with NumPy/JAX backend (fast implementation).
OPLS
    Orthogonal PLS for regression (removes Y-orthogonal variation).
OPLSDA
    Orthogonal PLS Discriminant Analysis for classification.
MBPLS
    Multiblock PLS for fusing multiple X blocks.
DiPLS
    Dynamic PLS for time-lagged process data.
SparsePLS
    Sparse PLS with L1 regularization for variable selection.
SIMPLS
    SIMPLS algorithm for PLS regression (de Jong 1993).
"""

# Import all model classes from their new modules for backward compatibility
from .plsda import PLSDA
from .ikpls import IKPLS
from .opls import OPLS
from .oplsda import OPLSDA
from .mbpls import MBPLS
from .diplS import DiPLS
from .sparsepls import SparsePLS
from .simpls import SIMPLS

__all__ = [
    "PLSDA",
    "IKPLS",
    "OPLS",
    "OPLSDA",
    "MBPLS",
    "DiPLS",
    "SparsePLS",
    "SIMPLS",
]
