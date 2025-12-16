"""
Models module for presets.

This module contains model definitions and references organized by framework.
"""

from .base import BaseModelOperator

# Import sklearn models
from .sklearn import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS, LWPLS, IntervalPLS, RobustPLS, RecursivePLS, KOPLS
from .sklearn.nlpls import KernelPLS, NLPLS, KPLS
from .sklearn.oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
from .sklearn.fckpls import FCKPLS, FractionalPLS, FractionalConvFeaturizer

# Import TensorFlow models
from .tensorflow.nicon import *
from .tensorflow.generic import *

# Import PyTorch models (currently commented out)
# from .pytorch.nicon import *
# from .pytorch.generic import *

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
