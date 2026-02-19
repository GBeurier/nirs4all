"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from .aom_pls import (
    AOMPLSRegressor,
    ComposedOperator,
    DetrendProjectionOperator,
    FFTBandpassOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WaveletProjectionOperator,
    default_operator_bank,
    extended_operator_bank,
)
from .aom_pls_classifier import AOMPLSClassifier
from .dipls import DiPLS
from .fckpls import FCKPLS, FractionalConvFeaturizer, FractionalPLS
from .ikpls import IKPLS
from .ipls import IntervalPLS
from .kopls import KOPLS
from .lwpls import LWPLS
from .mbpls import MBPLS
from .nlpls import KPLS, NLPLS, KernelPLS
from .oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .opls import OPLS
from .oplsda import OPLSDA
from .plsda import PLSDA
from .pop_pls import POPPLSRegressor, pop_pls_operator_bank
from .pop_pls_classifier import POPPLSClassifier
from .recursive_pls import RecursivePLS
from .robust_pls import RobustPLS
from .simpls import SIMPLS
from .sparsepls import SparsePLS

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
    "AOMPLSRegressor",
    "LinearOperator",
    "IdentityOperator",
    "SavitzkyGolayOperator",
    "DetrendProjectionOperator",
    "ComposedOperator",
    "NorrisWilliamsOperator",
    "FiniteDifferenceOperator",
    "WaveletProjectionOperator",
    "FFTBandpassOperator",
    "default_operator_bank",
    "extended_operator_bank",
    "POPPLSRegressor",
    "pop_pls_operator_bank",
    "AOMPLSClassifier",
    "POPPLSClassifier",
]
