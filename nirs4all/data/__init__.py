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
from ._features import (
    FeatureSource,
    FeatureLayout,
    HeaderUnit,
    normalize_layout,
    normalize_header_unit,
)

# Signal type management
from .signal_type import (
    SignalType,
    SignalTypeInput,
    normalize_signal_type,
    SignalTypeDetector,
    detect_signal_type,
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
    # Signal type
    "SignalType",
    "SignalTypeInput",
    "normalize_signal_type",
    "SignalTypeDetector",
    "detect_signal_type",
]
