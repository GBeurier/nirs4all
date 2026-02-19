"""
Model controllers module for nirs4all.

This module contains model controllers for different machine learning frameworks.
All model controllers support training, fine-tuning with Optuna, and prediction modes.

Controllers follow the operator-controller pattern where:
- Operators (in nirs4all.operators.models) define WHAT models to use
- Controllers (here) define HOW to execute them
"""

from .autogluon_model import AutoGluonModelController
from .base_model import BaseModelController
from .jax_model import JaxModelController
from .meta_model import MetaModelController
from .sklearn_model import SklearnModelController

# Phase 2: Stacking subpackage
from .stacking import (
    FoldAlignmentValidator,
    ReconstructionResult,
    ReconstructorConfig,
    TrainingSetReconstructor,
    ValidationResult,
)
from .tensorflow_model import TensorFlowModelController
from .torch_model import PyTorchModelController

__all__ = [
    'BaseModelController',
    'SklearnModelController',
    'TensorFlowModelController',
    'PyTorchModelController',
    'JaxModelController',
    'AutoGluonModelController',
    'MetaModelController',
    # Stacking components
    'TrainingSetReconstructor',
    'FoldAlignmentValidator',
    'ValidationResult',
    'ReconstructionResult',
    'ReconstructorConfig',
]
