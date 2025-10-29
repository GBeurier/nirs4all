"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.
"""

from .dataset import SpectroDataset
from .config import DatasetConfigs
from .predictions import Predictions, PredictionResult, PredictionResultsList
from ..visualization.predictions import PredictionAnalyzer

# Provide backward-compatible imports for feature components
from .feature_components import (
    FeatureSource,
    FeatureLayout,
    HeaderUnit,
    normalize_layout,
    normalize_header_unit,
)

__all__ = [
    "SpectroDataset",
    "DatasetConfigs",
    "Predictions",
    "PredictionResult",
    "PredictionResultsList",
    "PredictionAnalyzer",
    "FeatureSource",
    "FeatureLayout",
    "HeaderUnit",
    "normalize_layout",
    "normalize_header_unit",
]
