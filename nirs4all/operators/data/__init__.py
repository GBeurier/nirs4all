"""Data operators for merge and source handling.

This module provides operators for branch merging, source handling,
and related data manipulation operations.
"""

from .merge import (
    AggregationStrategy,
    BranchPredictionConfig,
    BranchType,
    DisjointSelectionCriterion,
    MergeConfig,
    MergeMode,
    SelectionStrategy,
    ShapeMismatchStrategy,
    SourceIncompatibleStrategy,
    SourceMergeConfig,
    SourceMergeStrategy,
)
from .repetition import (
    RepetitionConfig,
    UnequelRepsStrategy,
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
    "UnequelRepsStrategy",
    # Dataclasses
    "BranchPredictionConfig",
    "MergeConfig",
    "SourceMergeConfig",
    "RepetitionConfig",
]
