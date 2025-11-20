"""
Model controllers module for nirs4all.

This module contains model controllers for different machine learning frameworks.
All model controllers support training, fine-tuning with Optuna, and prediction modes.

Controllers follow the operator-controller pattern where:
- Operators (in nirs4all.operators.models) define WHAT models to use
- Controllers (here) define HOW to execute them
"""

from .base_model import BaseModelController
from .sklearn_model import SklearnModelController
from .tensorflow_model import TensorFlowModelController
from .torch_model import PyTorchModelController
from .jax_model import JaxModelController

__all__ = [
    'BaseModelController',
    'SklearnModelController',
    'TensorFlowModelController',
    'PyTorchModelController',
    'JaxModelController',
]
