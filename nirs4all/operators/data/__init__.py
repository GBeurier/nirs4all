"""Data operators for merge and source handling.

This module provides operators for branch merging, source handling,
and related data manipulation operations.
"""

from .merge import (
    MergeMode,
    BranchType,
    DisjointSelectionCriterion,
    SelectionStrategy,
    AggregationStrategy,
    ShapeMismatchStrategy,
    SourceMergeStrategy,
    SourceIncompatibleStrategy,
    BranchPredictionConfig,
    MergeConfig,
    SourceMergeConfig,
)

__all__ = [
    # Enums
    "MergeMode",
    "BranchType",
    "DisjointSelectionCriterion",
    "SelectionStrategy",
    "AggregationStrategy",
    "ShapeMismatchStrategy",
    "SourceMergeStrategy",
    "SourceIncompatibleStrategy",
    # Dataclasses
    "BranchPredictionConfig",
    "MergeConfig",
    "SourceMergeConfig",
]
