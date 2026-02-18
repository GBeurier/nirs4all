"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.
"""

from ..visualization.predictions import PredictionAnalyzer

# Provide backward-compatible imports for feature components
from ._features import (
    FeatureLayout,
    FeatureSource,
    HeaderUnit,
    normalize_header_unit,
    normalize_layout,
)
from .config import DatasetConfigs
from .dataset import SpectroDataset

# Parser utilities
from .parsers import (
    ConfigNormalizer,
    normalize_config,
)

# Partition utilities (Phase 4)
from .partition import (
    PartitionAssigner,
    PartitionError,
    PartitionResult,
)
from .predictions import MergeReport, PredictionResult, PredictionResultsList, Predictions

# Schema types (Phase 1 - new in refactoring)
from .schema import (
    ColumnConfig,
    ConfigValidator,
    DatasetConfigSchema,
    FileConfig,
    LoadingParams,
    PartitionConfig,
    TaskType,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

# Selection utilities (Phase 3)
from .selection import (
    ColumnSelectionError,
    ColumnSelector,
    LinkingError,
    RoleAssigner,
    RoleAssignmentError,
    RowSelectionError,
    RowSelector,
    SampleLinker,
)

# Signal type management
from .signal_type import (
    SignalType,
    SignalTypeDetector,
    SignalTypeInput,
    detect_signal_type,
    normalize_signal_type,
)

__all__ = [
    "SpectroDataset",
    "DatasetConfigs",
    "MergeReport",
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
    # Schema (new)
    "DatasetConfigSchema",
    "FileConfig",
    "ColumnConfig",
    "PartitionConfig",
    "LoadingParams",
    "TaskType",
    "ConfigValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    # Parsers (new)
    "ConfigNormalizer",
    "normalize_config",
    # Selection (Phase 3)
    "ColumnSelector",
    "ColumnSelectionError",
    "RowSelector",
    "RowSelectionError",
    "RoleAssigner",
    "RoleAssignmentError",
    "SampleLinker",
    "LinkingError",
    # Partition (Phase 4)
    "PartitionAssigner",
    "PartitionError",
    "PartitionResult",
]
