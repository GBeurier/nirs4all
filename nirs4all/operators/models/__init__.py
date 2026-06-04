"""
Models module for presets.

This module contains model definitions and references organized by framework.
TensorFlow and PyTorch models are loaded lazily to avoid importing heavy
frameworks at package load time.
"""
import sys
from typing import TYPE_CHECKING

from .base import BaseModelOperator

# TensorFlow models are loaded lazily to avoid importing TensorFlow at startup
# Use: from nirs4all.operators.models.tensorflow import nicon, generic
# Or access via __getattr__ below
# Import meta-model stacking
from .meta import BranchScope, CoverageStrategy, MetaModel, StackingConfig, StackingLevel, TestAggregation
from .residual import ResidualModel
from .selection import (
    AllPreviousModelsSelector,
    DiversitySelector,
    ExplicitModelSelector,
    ModelCandidate,
    SelectorFactory,
    SourceModelSelector,
    TopKByMetricSelector,
)

# Import sklearn models (lightweight, always available)
from .sklearn import IKPLS, KOPLS, LWPLS, MBPLS, OPLS, OPLSDA, PCR, PLSDA, SIMPLS, DiPLS, IntervalPLS, RecursivePLS, RobustPLS, SparsePLS
from .sklearn.aom_fast import (
    FastAOMConfig,
    FastAOMPLSRidge,
    HardAOMChainPLSRidge,
    SingleChainPLSRidge,
    SoftAOMChainPLSRidge,
    SparseMultiKernelRidge,
)
from .sklearn.aom_pls import (
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
from .sklearn.aom_pls_classifier import AOMPLSClassifier
from .sklearn.aom_ridge import (
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
from .sklearn.fckpls import FCKPLS, FractionalConvFeaturizer
from .sklearn.nlpls import KPLS, NLPLS, KernelPLS
from .sklearn.oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .sklearn.pop_pls_classifier import POPPLSClassifier
from .sklearn.tabpfn_nirs import TabPFNNIRSRegressor

# Lazy loading for TensorFlow models
_tensorflow_exports = None

def _get_tensorflow_exports():
    """Lazily load TensorFlow model exports."""
    global _tensorflow_exports
    if _tensorflow_exports is None:
        from nirs4all.utils.backend import is_available
        if is_available('tensorflow'):
            from .tensorflow import generic, nicon
            _tensorflow_exports = {}
            # Collect all exported names from tensorflow modules
            for mod in [nicon, generic]:
                for name in getattr(mod, '__all__', []):
                    _tensorflow_exports[name] = getattr(mod, name)
        else:
            _tensorflow_exports = {}
    return _tensorflow_exports

def __getattr__(name):
    """Lazy attribute access for TensorFlow models."""
    tf_exports = _get_tensorflow_exports()
    if name in tf_exports:
        return tf_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseModelOperator",
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
    # AOM-PLS
    "AOMPLSRegressor",
    "POPPLSRegressor",
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
    # Classifiers
    "AOMPLSClassifier",
    "POPPLSClassifier",
    # AOM-Ridge family
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
    # FastAOM family
    "FastAOMPLSRidge",
    "FastAOMConfig",
    "SingleChainPLSRidge",
    "HardAOMChainPLSRidge",
    "SoftAOMChainPLSRidge",
    "SparseMultiKernelRidge",
    # TabPFN NIRS-tuned regressor (fixed AGG preprocessing, no HPO)
    "TabPFNNIRSRegressor",
    # Meta-model stacking
    "MetaModel",
    "ResidualModel",
    "StackingConfig",
    "CoverageStrategy",
    "TestAggregation",
    "BranchScope",
    "StackingLevel",
    # Source model selection
    "SourceModelSelector",
    "AllPreviousModelsSelector",
    "ExplicitModelSelector",
    "TopKByMetricSelector",
    "DiversitySelector",
    "SelectorFactory",
    "ModelCandidate",
]
