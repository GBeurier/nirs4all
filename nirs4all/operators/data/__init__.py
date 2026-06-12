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
    MetaFeatureAdapter,
    MetaFeaturePlan,
    MetaRowDomain,
    MissingPredictionPolicy,
    SelectionProtocol,
    SelectionStrategy,
    ShapeMismatchStrategy,
    SourceIncompatibleStrategy,
    SourceMergeConfig,
    SourceMergeStrategy,
    StackingFitContract,
    StackingSelectionProtocol,
)
from .rep_fusion import RepFusionConfig
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
    "MetaRowDomain",
    "MetaFeatureAdapter",
    "MissingPredictionPolicy",
    "StackingSelectionProtocol",
    "SelectionProtocol",
    # Dataclasses
    "BranchPredictionConfig",
    "MergeConfig",
    "SourceMergeConfig",
    "RepetitionConfig",
    "RepFusionConfig",
    "MetaFeaturePlan",
    "StackingFitContract",
]
