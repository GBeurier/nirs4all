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
    # Availability checks (fast, no imports)
    is_available,
    is_tensorflow_available,
    is_torch_available,
    is_keras_available,
    is_jax_available,
    is_gpu_available,

    # Requirement enforcement
    require_backend,
    check_backend_available,
    BackendNotAvailableError,

    # Framework decorator
    framework,

    # Lazy constants (for backward compat)
    TF_AVAILABLE,
    TORCH_AVAILABLE,
    JAX_AVAILABLE,
    KERAS_AVAILABLE,

    # Info utilities
    get_backend_info,
    get_gpu_info,
    print_backend_status,

    # Lazy import helper
    lazy_import,

    # Cache management
    clear_availability_cache,
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
    # Backend detection (fast, no imports)
    'is_available',
    'is_tensorflow_available',
    'is_torch_available',
    'is_keras_available',
    'is_jax_available',
    'is_gpu_available',

    # Requirements
    'require_backend',
    'check_backend_available',
    'BackendNotAvailableError',

    # Framework decorator
    'framework',

    # Lazy availability constants (backward compat)
    'TF_AVAILABLE',
    'TORCH_AVAILABLE',
    'JAX_AVAILABLE',
    'KERAS_AVAILABLE',

    # Info utilities
    'get_backend_info',
    'get_gpu_info',
    'print_backend_status',
    'lazy_import',
    'clear_availability_cache',

    # Header unit utilities
    'AXIS_LABELS',
    'DEFAULT_AXIS_LABEL',
    'get_axis_label',
    'get_x_values_and_label',
    'should_invert_x_axis',
    'apply_x_axis_limits',
]
