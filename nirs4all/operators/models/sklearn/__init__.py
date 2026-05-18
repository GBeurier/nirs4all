"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from .aom_fast import (
    FastAOMConfig,
    FastAOMPLSRidge,
    HardAOMChainPLSRidge,
    SingleChainPLSRidge,
    SoftAOMChainPLSRidge,
    SparseMultiKernelRidge,
)
from .aom_pls import (
    AOMPLSRegressor,
    ComposedOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    POPPLSRegressor,
    SavitzkyGolayOperator,
    WhittakerOperator,
    bank_by_name,
    compact_bank,
    default_bank,
    default_operator_bank,
    extended_bank,
)
from .aom_pls_aomlib import AOMPLSAomlibRegressor
from .aom_pls_classifier import AOMPLSClassifier
from .aom_ridge import (
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
from .pcr import PCR
from .plsda import PLSDA
from .pop_pls_classifier import POPPLSClassifier
from .recursive_pls import RecursivePLS
from .robust_pls import RobustPLS
from .simpls import SIMPLS
from .sparsepls import SparsePLS
from .tabpfn_nirs import TabPFNNIRSRegressor

__all__ = [
    "PLSDA",
    "IKPLS",
    "OPLS",
    "OPLSDA",
    "PCR",
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
    "POPPLSRegressor",
    "AOMPLSAomlibRegressor",
    "AOMPLSClassifier",
    "POPPLSClassifier",
    "LinearSpectralOperator",
    "IdentityOperator",
    "SavitzkyGolayOperator",
    "DetrendProjectionOperator",
    "ComposedOperator",
    "NorrisWilliamsOperator",
    "FiniteDifferenceOperator",
    "WhittakerOperator",
    "default_operator_bank",
    "default_bank",
    "compact_bank",
    "extended_bank",
    "bank_by_name",
    # AOM-Ridge family (new)
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
    # FastAOM family (new)
    "FastAOMPLSRidge",
    "FastAOMConfig",
    "SingleChainPLSRidge",
    "HardAOMChainPLSRidge",
    "SoftAOMChainPLSRidge",
    "SparseMultiKernelRidge",
    # TabPFN NIRS-tuned regressor (fixed AGG preprocessing, no HPO)
    "TabPFNNIRSRegressor",
]
