"""
TabPFN Integration for NIRS Analysis
=====================================

This module provides tools for using TabPFN (Tabular Prior-data Fitted Network)
with NIRS spectral data. TabPFN is a transformer-based model that can solve
tabular problems in a second.

Key components:
- SpectralLatentFeatures: Transforms NIRS spectra into TabPFN-friendly features
- SpectralLatentFeaturesLite: Lightweight version for faster processing
- Hyperparameter search spaces for TabPFN tuning
- TabPFN configuration utilities (tabpfn_config)
- Integration utilities with nirs4all pipelines

References:
    Hollmann et al. (2023). TabPFN: A Transformer That Solves Small
    Tabular Classification Problems in a Second. ICLR 2023.
"""

from .spectral_latent_features import SpectralLatentFeatures, SpectralLatentFeaturesLite

# Conditionally import tabpfn_config utilities
try:
    from .tabpfn_config import (
        CLASSIFIER_MODELS,
        REGRESSOR_MODELS,
        TABPFN_AVAILABLE,
        build_finetune_params,
        create_model,
        generate_inference_configs,
        get_model_class,
        get_model_paths,
    )
    __all__ = [
        'SpectralLatentFeatures',
        'SpectralLatentFeaturesLite',
        'TABPFN_AVAILABLE',
        'CLASSIFIER_MODELS',
        'REGRESSOR_MODELS',
        'get_model_class',
        'get_model_paths',
        'create_model',
        'build_finetune_params',
        'generate_inference_configs',
    ]
except ImportError:
    __all__ = ['SpectralLatentFeatures', 'SpectralLatentFeaturesLite']
