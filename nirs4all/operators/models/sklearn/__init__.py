"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from .pls import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS
from .lwpls import LWPLS
from .ipls import IntervalPLS
from .robust_pls import RobustPLS

__all__ = [
    "PLSDA",
    "IKPLS",
    "OPLS",
    "OPLSDA",
    "MBPLS",
    "DiPLS",
    "SparsePLS",
    "LWPLS",
    "SIMPLS",
    "IntervalPLS",
    "RobustPLS",
]
