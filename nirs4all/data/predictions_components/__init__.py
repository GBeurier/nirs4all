"""
Predictions components package.

This package provides modular components for prediction management:
    - storage: Low-level DataFrame storage backend
    - serializer: Data serialization/deserialization
    - indexer: Fast filtering and lookup operations
    - ranker: Ranking and top-k selection
    - aggregator: Partition data aggregation
    - query: Catalog query operations
    - result: User-facing result containers
    - schemas: DataFrame schema definitions

Public classes exported for backward compatibility:
    - PredictionResult
    - PredictionResultsList
"""

from .result import PredictionResult, PredictionResultsList
from .storage import PredictionStorage
from .serializer import PredictionSerializer
from .schemas import PREDICTION_SCHEMA
from .array_registry import ArrayRegistry, ARRAY_SCHEMA

__all__ = [
    'PredictionResult',
    'PredictionResultsList',
    'PredictionStorage',
    'PredictionSerializer',
    'PREDICTION_SCHEMA',
    'ARRAY_SCHEMA',
    'ArrayRegistry',
]
