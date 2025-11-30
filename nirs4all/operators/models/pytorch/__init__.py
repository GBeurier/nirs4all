"""PyTorch model operators.

This module provides PyTorch neural network models for use in nirs4all pipelines.
"""

from .nicon import *
from .generic import *
from .spectral_transformer import *

__all__ = [
    # Exports from nicon and generic will be added here
    # SpectralTransformer exports
    'SpectralTransformer',
    'spectral_transformer',
    'spectral_transformer_classification',
    'spectral_transformer_small',
    'spectral_transformer_small_classification',
    'spectral_transformer_large',
    'spectral_transformer_large_classification',
]
