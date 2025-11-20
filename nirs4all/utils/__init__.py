"""
Utility functions for the nirs4all package.

This module contains true utility functions for terminal output, backend detection, etc.
Core functionality has been moved to appropriate modules:
- Metrics/evaluation → nirs4all.core.metrics
- Task types → nirs4all.core.task_type, nirs4all.core.task_detection
- Binning → nirs4all.data.binning
- Balancing → nirs4all.controllers.data.balancing
- Artifact serialization → nirs4all.pipeline.artifact_serialization
"""

from .backend import (
    TF_AVAILABLE,
    # TORCH_AVAILABLE,
    framework,
    is_tensorflow_available,
    # is_torch_available,
    is_keras_available,
    is_jax_available,
    is_gpu_available
)

__all__ = [
    'TF_AVAILABLE',
    # 'TORCH_AVAILABLE',
    'framework',
    'is_tensorflow_available',
    # 'is_torch_available',
    'is_keras_available',
    'is_jax_available',
    'is_gpu_available',
]
