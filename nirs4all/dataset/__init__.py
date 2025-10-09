"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.
"""

__version__ = "0.1.0"
__author__ = "NIRS4All Project"

# Main public API
from .dataset import SpectroDataset
from .dataset_config import DatasetConfigs
from .predictions import Predictions, PredictionResult, PredictionResultsList
from .prediction_analyzer import PredictionAnalyzer

__all__ = ["SpectroDataset", "DatasetConfigs", "Predictions", "PredictionResult", "PredictionResultsList", "PredictionAnalyzer"]
