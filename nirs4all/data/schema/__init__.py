"""
Schema module for dataset configuration.

This module provides Pydantic-based schema models for dataset configuration,
providing type safety, validation, and clear documentation of the configuration format.

The schema supports:
- Legacy format (train_x, test_x, etc.) - fully implemented
- New files syntax (planned for future phases)
- Multi-source datasets with sources syntax
- Feature variations for preprocessed data or multi-variable datasets
"""

from .config import (
    AggregateMethod,
    CategoricalMode,
    ColumnConfig,
    ColumnSelection,
    # Core config models
    DatasetConfigSchema,
    FileConfig,
    FoldConfig,
    FoldDefinition,
    HeaderUnit,
    LoadingParams,
    NAFillConfig,
    NAFillMethod,
    NAPolicy,
    PartitionConfig,
    PartitionType,
    # Type aliases
    PathOrArray,
    PreprocessingApplied,
    SharedMetadataConfig,
    SharedTargetsConfig,
    SignalTypeEnum,
    # Source config models (Phase 6)
    SourceConfig,
    SourceFileConfig,
    # Enums
    TaskType,
    # Variation config models (Phase 7)
    VariationConfig,
    VariationFileConfig,
    VariationMode,
)
from .validation import (
    ConfigValidator,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

__all__ = [
    # Core config models
    "DatasetConfigSchema",
    "FileConfig",
    "ColumnConfig",
    "PartitionConfig",
    "LoadingParams",
    "FoldConfig",
    "FoldDefinition",
    # Source config models (Phase 6)
    "SourceConfig",
    "SourceFileConfig",
    "SharedTargetsConfig",
    "SharedMetadataConfig",
    # Variation config models (Phase 7)
    "VariationConfig",
    "VariationFileConfig",
    "PreprocessingApplied",
    # Enums
    "TaskType",
    "HeaderUnit",
    "SignalTypeEnum",
    "PartitionType",
    "NAPolicy",
    "NAFillMethod",
    "NAFillConfig",
    "CategoricalMode",
    "AggregateMethod",
    "VariationMode",
    # Type aliases
    "PathOrArray",
    "ColumnSelection",
    # Validation
    "ConfigValidator",
    "ValidationError",
    "ValidationWarning",
    "ValidationResult",
]
