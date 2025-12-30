"""
SpectroDataset - A specialized dataset API for spectroscopy data.

This module provides zero-copy, multi-source aware dataset management
with transparent versioning and fine-grained indexing capabilities.

Submodules:
    synthetic: Synthetic NIRS spectra generation tools.
"""

from .dataset import SpectroDataset
from .config import DatasetConfigs
from .predictions import Predictions, PredictionResult, PredictionResultsList
from ..visualization.predictions import PredictionAnalyzer

# Synthetic data generation submodule
from . import synthetic

# Provide backward-compatible imports for feature components
from ._features import (
    FeatureSource,
    FeatureLayout,
    HeaderUnit,
    normalize_layout,
    normalize_header_unit,
)

# Signal type management
from .signal_type import (
    SignalType,
    SignalTypeInput,
    normalize_signal_type,
    SignalTypeDetector,
    detect_signal_type,
)

# Schema types (Phase 1 - new in refactoring)
from .schema import (
    DatasetConfigSchema,
    FileConfig,
    ColumnConfig,
    PartitionConfig,
    LoadingParams,
    TaskType,
    ConfigValidator,
    ValidationResult,
    ValidationError,
    ValidationWarning,
)

# Parser utilities
from .parsers import (
    ConfigNormalizer,
    normalize_config,
)

# Selection utilities (Phase 3)
from .selection import (
    ColumnSelector,
    ColumnSelectionError,
    RowSelector,
    RowSelectionError,
    RoleAssigner,
    RoleAssignmentError,
    SampleLinker,
    LinkingError,
)

# Partition utilities (Phase 4)
from .partition import (
    PartitionAssigner,
    PartitionError,
    PartitionResult,
)

__all__ = [
    "SpectroDataset",
    "DatasetConfigs",
    "Predictions",
    "PredictionResult",
    "PredictionResultsList",
    "PredictionAnalyzer",
    # Synthetic data generation
    "synthetic",
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
