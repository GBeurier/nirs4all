"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.
"""

from .dataset import SpectroDataset
from .config import DatasetConfigs
from .predictions import Predictions, PredictionResult, PredictionResultsList
from ..visualization.predictions import PredictionAnalyzer

__all__ = ["SpectroDataset", "DatasetConfigs", "Predictions", "PredictionResult", "PredictionResultsList", "PredictionAnalyzer"]
