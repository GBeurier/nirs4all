"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from .lwpls import LWPLS
from .ipls import IntervalPLS
from .robust_pls import RobustPLS
from .recursive_pls import RecursivePLS
from .kopls import KOPLS
from .plsda import PLSDA
from .ikpls import IKPLS
from .opls import OPLS
from .oplsda import OPLSDA
from .mbpls import MBPLS
from .dipls import DiPLS
from .sparsepls import SparsePLS
from .simpls import SIMPLS
from .nlpls import KernelPLS, NLPLS, KPLS
from .oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .fckpls import FCKPLS, FractionalPLS, FractionalConvFeaturizer


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
    "RecursivePLS",
    "KOPLS",
    "KernelPLS",
    "NLPLS",
    "KPLS",
    "OKLMPLS",
    "IdentityFeaturizer",
    "PolynomialFeaturizer",
    "RBFFeaturizer",
    "FCKPLS",
    "FractionalPLS",
    "FractionalConvFeaturizer",
]