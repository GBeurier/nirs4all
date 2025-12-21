"""Data operators for merge and source handling.

This module provides operators for branch merging, source handling,
and related data manipulation operations.
"""

from .merge import (
    MergeMode,
    SelectionStrategy,
    AggregationStrategy,
    ShapeMismatchStrategy,
    BranchPredictionConfig,
    MergeConfig,
)

__all__ = [
    # Enums
    "MergeMode",
    "SelectionStrategy",
    "AggregationStrategy",
    "ShapeMismatchStrategy",
    # Dataclasses
    "BranchPredictionConfig",
    "MergeConfig",
]
