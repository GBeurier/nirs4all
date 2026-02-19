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
from .sklearn import IKPLS, KOPLS, LWPLS, MBPLS, OPLS, OPLSDA, PLSDA, SIMPLS, DiPLS, IntervalPLS, RecursivePLS, RobustPLS, SparsePLS
from .sklearn.aom_pls import (
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
from .sklearn.aom_pls_classifier import AOMPLSClassifier
from .sklearn.fckpls import FCKPLS, FractionalConvFeaturizer, FractionalPLS
from .sklearn.nlpls import KPLS, NLPLS, KernelPLS
from .sklearn.oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .sklearn.pop_pls import POPPLSRegressor, pop_pls_operator_bank
from .sklearn.pop_pls_classifier import POPPLSClassifier

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
    # AOM-PLS
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
    # POP-PLS
    "POPPLSRegressor",
    "pop_pls_operator_bank",
    # Classifiers
    "AOMPLSClassifier",
    "POPPLSClassifier",
    # Meta-model stacking
    "MetaModel",
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
