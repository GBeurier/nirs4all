"""
Utility functions for the nirs4all package.

This module contains true utility functions for terminal output, backend detection, etc.
Core functionality has been moved to appropriate modules:
- Metrics/evaluation → nirs4all.core.metrics
- Task types → nirs4all.core.task_type, nirs4all.core.task_detection
- Binning → nirs4all.data.binning
- Balancing → nirs4all.controllers.data.balancing
- Artifact serialization → nirs4all.pipeline.artifact_serialization
- Header units → nirs4all.utils.header_units
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

from .header_units import (
    AXIS_LABELS,
    DEFAULT_AXIS_LABEL,
    get_axis_label,
    get_x_values_and_label,
    should_invert_x_axis,
    apply_x_axis_limits,
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
    # Header unit utilities
    'AXIS_LABELS',
    'DEFAULT_AXIS_LABEL',
    'get_axis_label',
    'get_x_values_and_label',
    'should_invert_x_axis',
    'apply_x_axis_limits',
]
