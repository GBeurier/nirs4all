"""
Utility functions for the nirs4all package.
"""

from .backend_utils import (
    TF_AVAILABLE,
    # TORCH_AVAILABLE,
    framework,
    is_tensorflow_available,
    # is_torch_available,
    is_keras_available,
    is_jax_available,
    is_gpu_available
)

from .PCA_analyzer import PreprocPCAEvaluator

__all__ = [
    'TF_AVAILABLE',
    # 'TORCH_AVAILABLE',
    'framework',
    'is_tensorflow_available',
    # 'is_torch_available',
    'is_keras_available',
    'is_jax_available',
    'is_gpu_available',
    'PreprocPCAEvaluator'
]
