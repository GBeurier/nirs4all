"""
NIRS4All - A comprehensive package for Near-Infrared Spectroscopy data processing and analysis.

This package provides tools for spectroscopy data handling, preprocessing, model building,
and pipeline management with support for multiple ML backends.
"""

__version__ = "0.9.7"
__author__ = "NIRS4All Project"

# Core pipeline components - most commonly used
from .pipeline import PipelineRunner, PipelineConfigs, PipelineHistory
from .controllers import register_controller, CONTROLLER_REGISTRY

# Utility functions for backend detection
from .utils import (
    is_tensorflow_available,
    is_torch_available,
    is_gpu_available,
    framework
)

# Make commonly used classes available at package level
__all__ = [
    # Pipeline components
    "PipelineRunner",
    "PipelineConfigs",
    "PipelineHistory",

    # Controller system
    "register_controller",
    "CONTROLLER_REGISTRY",

    # Utilities
    "is_tensorflow_available",
    "is_torch_available",
    "is_gpu_available",
    "framework"
]
