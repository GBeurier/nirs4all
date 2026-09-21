"""Scikit-learn model operators.

This module provides wrappers and utilities for using scikit-learn models
as operators in nirs4all pipelines.
"""

from importlib import import_module

from .dipls import DiPLS
from .fckpls import FCKPLS, FractionalConvFeaturizer
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
from .recursive_pls import RecursivePLS
from .robust_pls import RobustPLS
from .simpls import SIMPLS
from .sparsepls import SparsePLS
from .tabpfn_nirs import TabPFNNIRSRegressor

_LAZY_EXPORT_MODULES = {
    **dict.fromkeys(
        (
            "FastAOMConfig",
            "FastAOMPLSRidge",
            "HardAOMChainPLSRidge",
            "SingleChainPLSRidge",
            "SoftAOMChainPLSRidge",
            "SparseMultiKernelRidge",
        ),
        ".aom_fast",
    ),
    **dict.fromkeys(
        (
            "AOMPLSRegressor",
            "ComposedOperator",
            "DetrendProjectionOperator",
            "FiniteDifferenceOperator",
            "IdentityOperator",
            "LinearSpectralOperator",
            "NorrisWilliamsOperator",
            "POPPLSRegressor",
            "SavitzkyGolayOperator",
            "WhittakerOperator",
            "bank_by_name",
            "compact_bank",
            "default_bank",
            "default_operator_bank",
            "extended_bank",
        ),
        ".aom_pls",
    ),
    "AOMPLSAomlibRegressor": ".aom_pls_aomlib",
    "AOMPLSClassifier": ".aom_pls_classifier",
    **dict.fromkeys(
        (
            "AOMKernelizer",
            "AOMLocalRidge",
            "AOMMultiBranchMKL",
            "AOMMultiKernelRidge",
            "AOMRidgeAutoSelector",
            "AOMRidgeBlender",
            "AOMRidgeClassifier",
            "AOMRidgePLS",
            "AOMRidgePLSCV",
            "AOMRidgeRegressor",
        ),
        ".aom_ridge",
    ),
    "POPPLSClassifier": ".pop_pls_classifier",
}


def __getattr__(name: str):
    """Load AOM/POP families only when their public symbols are requested."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

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
