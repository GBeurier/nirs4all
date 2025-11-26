"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from .pls import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS
from .lwpls import LWPLS

__all__ = [
    "PLSDA",
    "IKPLS",
    "OPLS",
    "OPLSDA",
    "MBPLS",
    "DiPLS",
    "SparsePLS",
    "LWPLS",
]
