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
    # Core config models
    DatasetConfigSchema,
    FileConfig,
    ColumnConfig,
    PartitionConfig,
    LoadingParams,
    FoldConfig,
    FoldDefinition,
    # Source config models (Phase 6)
    SourceConfig,
    SourceFileConfig,
    SharedTargetsConfig,
    SharedMetadataConfig,
    # Variation config models (Phase 7)
    VariationConfig,
    VariationFileConfig,
    PreprocessingApplied,
    # Enums
    TaskType,
    HeaderUnit,
    SignalTypeEnum,
    PartitionType,
    NAPolicy,
    CategoricalMode,
    AggregateMethod,
    VariationMode,
    # Type aliases
    PathOrArray,
    ColumnSelection,
)

from .validation import (
    ConfigValidator,
    ValidationError,
    ValidationWarning,
    ValidationResult,
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
