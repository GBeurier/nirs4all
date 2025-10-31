"""
Indexer components for modular, maintainable index management.

This package provides focused components that handle distinct responsibilities
within the indexer system:

- IndexStore: Low-level DataFrame storage and query execution
- QueryBuilder: Convert Selector dicts to Polars expressions
- SampleManager: Sample ID generation and tracking
- AugmentationTracker: Origin/augmented sample relationships
- ProcessingManager: Processing list operations with native Polars lists
- ParameterNormalizer: Input validation and normalization
"""

from .index_store import IndexStore
from .query_builder import QueryBuilder
from .sample_manager import SampleManager
from .augmentation_tracker import AugmentationTracker
from .processing_manager import ProcessingManager
from .parameter_normalizer import ParameterNormalizer

__all__ = [
    "IndexStore",
    "QueryBuilder",
    "SampleManager",
    "AugmentationTracker",
    "ProcessingManager",
    "ParameterNormalizer",
]
