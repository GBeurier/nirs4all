"""
Predictions result containers.

Public classes:
    - PredictionResult: Enhanced dict for a single prediction record.
    - PredictionResultsList: Enhanced list of PredictionResult objects.
"""

from .result import PredictionResult, PredictionResultsList

__all__ = [
    'PredictionResult',
    'PredictionResultsList',
]
