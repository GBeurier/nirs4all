"""
Models module for presets.

This module contains model definitions and references organized by framework.
TensorFlow and PyTorch models are loaded lazily to avoid importing heavy
frameworks at package load time.
"""
import sys
from typing import TYPE_CHECKING

from .base import BaseModelOperator

# Import sklearn models (lightweight, always available)
from .sklearn import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS, LWPLS, IntervalPLS, RobustPLS, RecursivePLS, KOPLS
from .sklearn.nlpls import KernelPLS, NLPLS, KPLS
from .sklearn.oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .sklearn.fckpls import FCKPLS, FractionalPLS, FractionalConvFeaturizer

# TensorFlow models are loaded lazily to avoid importing TensorFlow at startup
# Use: from nirs4all.operators.models.tensorflow import nicon, generic
# Or access via __getattr__ below

# Import meta-model stacking
from .meta import MetaModel, StackingConfig, CoverageStrategy, TestAggregation, BranchScope, StackingLevel
from .selection import (
    SourceModelSelector,
    AllPreviousModelsSelector,
    ExplicitModelSelector,
    TopKByMetricSelector,
    DiversitySelector,
    SelectorFactory,
    ModelCandidate,
)

# Lazy loading for TensorFlow models
_tensorflow_exports = None


def _get_tensorflow_exports():
    """Lazily load TensorFlow model exports."""
    global _tensorflow_exports
    if _tensorflow_exports is None:
        from nirs4all.utils.backend import is_available
        if is_available('tensorflow'):
            from .tensorflow import nicon, generic
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
