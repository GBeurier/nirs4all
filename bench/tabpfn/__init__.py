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
- Integration utilities with nirs4all pipelines

References:
    Hollmann et al. (2023). TabPFN: A Transformer That Solves Small
    Tabular Classification Problems in a Second. ICLR 2023.
"""

from .spectral_latent_features import SpectralLatentFeatures, SpectralLatentFeaturesLite

__all__ = ['SpectralLatentFeatures', 'SpectralLatentFeaturesLite']
