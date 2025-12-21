"""
Schema definitions for dataset configuration.

This module defines Pydantic models for validating and normalizing dataset configurations.
It supports both the legacy format (train_x, test_x, etc.) and the planned new format
(files, sources, variations).

The models provide:
- Type validation and coercion
- Default value handling
- Clear documentation via Field descriptions
- Serialization/deserialization
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator


# =============================================================================
# Custom type handling for numpy arrays
# =============================================================================

def _validate_path_or_array(v: Any) -> Any:
    """Validate path or array, passing through as-is for numpy arrays."""
    # Pass through numpy arrays without validation
    if isinstance(v, np.ndarray):
        return v
    # Pass through lists (might be paths or arrays)
    if isinstance(v, list):
        return v
    # Pass through strings and Path objects
    if isinstance(v, (str, Path)):
        return v
    return v


# Type alias for path or array with custom validator
PathOrArrayType = Annotated[Any, BeforeValidator(_validate_path_or_array)]

# =============================================================================
# Enums for configuration options
# =============================================================================


class TaskType(str, Enum):
    """Task type for the dataset."""

    AUTO = "auto"
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class HeaderUnit(str, Enum):
    """Unit type for spectral headers."""

    WAVENUMBER = "cm-1"
    WAVELENGTH = "nm"
    NONE = "none"
    TEXT = "text"
    INDEX = "index"


class SignalTypeEnum(str, Enum):
    """Signal type for spectral data."""

    AUTO = "auto"
    ABSORBANCE = "absorbance"
    REFLECTANCE = "reflectance"
    REFLECTANCE_PERCENT = "reflectance%"
    TRANSMITTANCE = "transmittance"
    TRANSMITTANCE_PERCENT = "transmittance%"
    LOG_1_R = "log(1/R)"
    KUBELKA_MUNK = "kubelka-munk"


class PartitionType(str, Enum):
    """Partition assignment type."""

    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class NAPolicy(str, Enum):
    """Policy for handling NA/missing values."""

    AUTO = "auto"
    REMOVE = "remove"
    ABORT = "abort"


class CategoricalMode(str, Enum):
    """Mode for handling categorical columns in Y data."""

    AUTO = "auto"
    PRESERVE = "preserve"
    NONE = "none"


class AggregateMethod(str, Enum):
    """Method for aggregating predictions."""

    MEAN = "mean"
    MEDIAN = "median"
    VOTE = "vote"


class VariationMode(str, Enum):
    """Mode for handling feature variations.

    Feature variations represent different "views" of the same samples,
    such as pre-computed preprocessing variants or different variables
    from time series data.
    """

    SEPARATE = "separate"   # Each variation runs as independent pipeline
    CONCAT = "concat"       # Horizontal concatenation of all variations
    SELECT = "select"       # Use only specified variations
    COMPARE = "compare"     # Run each variation and rank by performance


# =============================================================================
# Type aliases
# =============================================================================

# Path can be a string path or a numpy array (for in-memory data)
# Use Any to avoid Pydantic schema generation issues with numpy.ndarray
PathOrArray = Any  # Union[str, Path, np.ndarray, List[Union[str, Path]]]

# Column selection can be indices, names, regex, etc.
ColumnSelection = Union[
    List[int],           # List of column indices
    List[str],           # List of column names
    str,                 # Range string like "2:-1" or regex pattern
    Dict[str, Any],      # Complex selection like {"regex": "^feature_.*"}
]


# =============================================================================
# Loading parameters schema
# =============================================================================


class LoadingParams(BaseModel):
    """Parameters for loading data files.

    These parameters control how CSV and other files are parsed.
    Parameters can be specified at global, partition, or file level,
    with more specific levels overriding general ones.
    """

    model_config = ConfigDict(extra="allow")  # Allow extra fields for forward compatibility

    delimiter: Optional[str] = Field(
        default=None,
        description="Field delimiter for CSV files. Default: ';' if not specified."
    )

    decimal_separator: Optional[str] = Field(
        default=None,
        description="Decimal separator for numeric values. Default: '.'"
    )

    has_header: Optional[bool] = Field(
        default=None,
        description="Whether the first row is a header. Default: True"
    )

    header_unit: Optional[Union[HeaderUnit, str]] = Field(
        default=None,
        description="Unit type for headers: 'cm-1', 'nm', 'none', 'text', 'index'. Default: 'cm-1'"
    )

    signal_type: Optional[Union[SignalTypeEnum, str]] = Field(
        default=None,
        description="Signal type: 'absorbance', 'reflectance', etc. Default: auto-detect"
    )

    encoding: Optional[str] = Field(
        default=None,
        description="File encoding. Default: 'utf-8'"
    )

    na_policy: Optional[Union[NAPolicy, str]] = Field(
        default=None,
        description="How to handle NA values: 'remove' or 'abort'. Default: 'remove'"
    )

    categorical_mode: Optional[Union[CategoricalMode, str]] = Field(
        default=None,
        description="How to handle categorical columns: 'auto', 'preserve', 'none'. Default: 'auto'"
    )

    @field_validator("header_unit", mode="before")
    @classmethod
    def normalize_header_unit(cls, v: Any) -> Optional[Union[HeaderUnit, str]]:
        """Normalize header_unit to enum if possible."""
        if v is None:
            return None
        if isinstance(v, HeaderUnit):
            return v
        if isinstance(v, str):
            try:
                return HeaderUnit(v.lower())
            except ValueError:
                return v  # Keep as string for validation error
        return v

    @field_validator("signal_type", mode="before")
    @classmethod
    def normalize_signal_type(cls, v: Any) -> Optional[Union[SignalTypeEnum, str]]:
        """Normalize signal_type to enum if possible."""
        if v is None:
            return None
        if isinstance(v, SignalTypeEnum):
            return v
        if isinstance(v, str):
            try:
                return SignalTypeEnum(v.lower())
            except ValueError:
                return v  # Keep as string for validation error
        return v

    def merge_with(self, other: Optional["LoadingParams"]) -> "LoadingParams":
        """Merge with another LoadingParams, self taking precedence.

        Args:
            other: Another LoadingParams to merge with (lower priority).

        Returns:
            New LoadingParams with merged values.
        """
        if other is None:
            return self

        # Start with other's values, then override with self's non-None values
        merged_data = other.model_dump(exclude_none=True)
        self_data = self.model_dump(exclude_none=True)
        merged_data.update(self_data)

        return LoadingParams(**merged_data)


# =============================================================================
# Column configuration schema (for future files syntax)
# =============================================================================


class ColumnConfig(BaseModel):
    """Configuration for column selection and role assignment.

    This is a stub for future implementation of the files syntax.
    Currently, column selection is handled by the loader directly.
    """

    model_config = ConfigDict(extra="forbid")

    features: Optional[ColumnSelection] = Field(
        default=None,
        description="Columns to use as features (X)."
    )

    targets: Optional[ColumnSelection] = Field(
        default=None,
        description="Columns to use as targets (Y)."
    )

    metadata: Optional[ColumnSelection] = Field(
        default=None,
        description="Columns to use as metadata."
    )


# =============================================================================
# Partition configuration schema
# =============================================================================


class PartitionConfig(BaseModel):
    """Configuration for partition assignment.

    Supports multiple partition methods:
    - Static: Assign entire file to a partition (use `type`)
    - Column-based: Partition based on column values (use `column`)
    - Percentage-based: Split by percentage (use `train`, `test` with percentages)
    - Index-based: Explicit index lists (use `train`, `test` with lists)
    - Index file: Load indices from external files (use `train_file`, `test_file`)

    Examples:
        # Static partition (entire file)
        partition:
          type: train

        # Column-based partition
        partition:
          column: "split"
          train_values: ["train", "training"]
          test_values: ["test", "validation"]

        # Percentage-based partition
        partition:
          train: "80%"
          test: "80%:100%"
          shuffle: true
          random_state: 42

        # Index-based partition
        partition:
          train: [0, 1, 2, 3, 4]
          test: [5, 6, 7, 8, 9]

        # Index file partition
        partition:
          train_file: "train_indices.txt"
          test_file: "test_indices.txt"
    """

    model_config = ConfigDict(extra="allow")

    # --- Static partition ---
    type: Optional[PartitionType] = Field(
        default=None,
        description="Static partition type: 'train', 'test', or 'predict'. "
                    "Assigns entire file to this partition."
    )

    # --- Column-based partition ---
    column: Optional[str] = Field(
        default=None,
        description="Column name containing partition labels."
    )

    train_values: Optional[List[str]] = Field(
        default=None,
        description="Values in partition column that indicate training data."
    )

    test_values: Optional[List[str]] = Field(
        default=None,
        description="Values in partition column that indicate test data."
    )

    predict_values: Optional[List[str]] = Field(
        default=None,
        description="Values in partition column that indicate predict-only data."
    )

    unknown_policy: Optional[Literal["error", "ignore", "train"]] = Field(
        default=None,
        description="How to handle unknown values in partition column. "
                    "'error': raise exception, 'ignore': skip rows, 'train': include in training."
    )

    # --- Percentage-based partition ---
    train: Optional[Union[str, List[int]]] = Field(
        default=None,
        description="Training partition: percentage string ('80%', '0:80%') or list of indices."
    )

    test: Optional[Union[str, List[int]]] = Field(
        default=None,
        description="Test partition: percentage string ('20%', '80%:100%') or list of indices."
    )

    predict: Optional[Union[str, List[int]]] = Field(
        default=None,
        description="Predict partition: percentage string or list of indices."
    )

    shuffle: Optional[bool] = Field(
        default=None,
        description="Whether to shuffle data before percentage-based splitting."
    )

    random_state: Optional[int] = Field(
        default=None,
        description="Random state for shuffle and sampling operations."
    )

    stratify: Optional[str] = Field(
        default=None,
        description="Column name for stratified splitting (maintains class proportions)."
    )

    # --- Index file partition ---
    train_file: Optional[str] = Field(
        default=None,
        description="Path to file containing training indices."
    )

    test_file: Optional[str] = Field(
        default=None,
        description="Path to file containing test indices."
    )

    predict_file: Optional[str] = Field(
        default=None,
        description="Path to file containing predict indices."
    )

    @model_validator(mode="after")
    def validate_partition_method(self) -> "PartitionConfig":
        """Validate that partition specification is consistent."""
        methods_used = []

        # Check static partition
        if self.type is not None:
            methods_used.append("static (type)")

        # Check column-based partition
        if self.column is not None:
            methods_used.append("column-based")

        # Check percentage/index partition
        has_percentage_or_index = (
            self.train is not None or
            self.test is not None or
            self.predict is not None
        )
        if has_percentage_or_index:
            methods_used.append("percentage/index")

        # Check file-based partition
        has_file = (
            self.train_file is not None or
            self.test_file is not None or
            self.predict_file is not None
        )
        if has_file:
            methods_used.append("file-based")

        # Allow combining methods in some cases
        # For now, just validate that we have at least one method if any fields are set
        return self

    def to_assigner_spec(self) -> Union[str, Dict[str, Any], None]:
        """Convert this config to a spec for PartitionAssigner.

        Returns:
            Partition specification for PartitionAssigner.assign().
        """
        # Static partition
        if self.type is not None:
            return self.type.value

        # Column-based partition
        if self.column is not None:
            spec = {"column": self.column}
            if self.train_values:
                spec["train_values"] = self.train_values
            if self.test_values:
                spec["test_values"] = self.test_values
            if self.predict_values:
                spec["predict_values"] = self.predict_values
            if self.unknown_policy:
                spec["unknown_policy"] = self.unknown_policy
            return spec

        # File-based partition
        if self.train_file or self.test_file or self.predict_file:
            spec = {}
            if self.train_file:
                spec["train_file"] = self.train_file
            if self.test_file:
                spec["test_file"] = self.test_file
            if self.predict_file:
                spec["predict_file"] = self.predict_file
            return spec

        # Percentage/index partition
        if self.train is not None or self.test is not None or self.predict is not None:
            spec = {}
            if self.train is not None:
                spec["train"] = self.train
            if self.test is not None:
                spec["test"] = self.test
            if self.predict is not None:
                spec["predict"] = self.predict
            if self.shuffle is not None:
                spec["shuffle"] = self.shuffle
            if self.random_state is not None:
                spec["random_state"] = self.random_state
            if self.stratify is not None:
                spec["stratify"] = self.stratify
            return spec

        return None


# =============================================================================
# Fold configuration schema
# =============================================================================


class FoldDefinition(BaseModel):
    """Definition of a single cross-validation fold.

    Specifies which sample indices belong to training and validation sets.
    """

    model_config = ConfigDict(extra="forbid")

    train: List[int] = Field(
        description="Sample indices for training in this fold."
    )

    val: List[int] = Field(
        default_factory=list,
        description="Sample indices for validation in this fold. "
                    "Also accepts 'test' as an alias."
    )


class FoldConfig(BaseModel):
    """Configuration for cross-validation fold definitions.

    Supports multiple ways to specify folds:
    - Inline: List of FoldDefinition objects
    - File: Path to a fold file (CSV, JSON, YAML)
    - Column: Column name in metadata containing fold assignments

    Examples:
        # Inline fold definitions
        folds:
          - train: [0, 1, 2, 3, 4]
            val: [5, 6, 7, 8, 9]
          - train: [5, 6, 7, 8, 9]
            val: [0, 1, 2, 3, 4]

        # File reference
        folds:
          file: "path/to/folds.csv"
          format: auto

        # Column in metadata
        folds:
          column: "cv_fold"
    """

    model_config = ConfigDict(extra="forbid")

    # --- Inline fold definitions ---
    folds: Optional[List[FoldDefinition]] = Field(
        default=None,
        description="List of inline fold definitions."
    )

    # --- File reference ---
    file: Optional[str] = Field(
        default=None,
        description="Path to fold file (CSV, JSON, YAML, TXT)."
    )

    format: Optional[Literal["auto", "csv", "json", "yaml", "txt"]] = Field(
        default="auto",
        description="Format of fold file. 'auto' detects from extension."
    )

    # --- Column reference ---
    column: Optional[str] = Field(
        default=None,
        description="Column name in metadata containing fold assignments. "
                    "Each unique value becomes a validation fold."
    )

    @model_validator(mode="after")
    def validate_fold_source(self) -> "FoldConfig":
        """Validate that exactly one fold source is specified."""
        sources = []

        if self.folds is not None:
            sources.append("inline (folds)")
        if self.file is not None:
            sources.append("file")
        if self.column is not None:
            sources.append("column")

        if len(sources) > 1:
            raise ValueError(
                f"Multiple fold sources specified: {sources}. "
                "Use only one of: inline folds, file, or column."
            )

        return self

    def to_fold_list(self) -> Optional[List[Tuple[List[int], List[int]]]]:
        """Convert inline fold definitions to fold list.

        Returns:
            List of (train_indices, val_indices) tuples, or None if not inline.
        """
        if self.folds is None:
            return None

        return [(f.train, f.val) for f in self.folds]


# =============================================================================
# File configuration schema (for future files syntax)
# =============================================================================


class FileConfig(BaseModel):
    """Configuration for a single data file.

    This is a stub for future implementation of the files syntax.
    It describes how to load and interpret a single data file.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the data file."
    )

    partition: Optional[PartitionType] = Field(
        default=None,
        description="Which partition this file belongs to."
    )

    columns: Optional[ColumnConfig] = Field(
        default=None,
        description="Column selection and role assignment."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters specific to this file."
    )

    link_by: Optional[str] = Field(
        default=None,
        description="Column to use for linking with other files."
    )


# =============================================================================
# Source configuration schema (for multi-source datasets)
# =============================================================================


class SourceFileConfig(BaseModel):
    """Configuration for a single file within a source.

    Similar to FileConfig but simplified for source context.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the data file."
    )

    partition: Optional[PartitionType] = Field(
        default=None,
        description="Which partition this file belongs to."
    )

    columns: Optional[ColumnConfig] = Field(
        default=None,
        description="Column selection and role assignment."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters specific to this file."
    )


class SourceConfig(BaseModel):
    """Configuration for a single feature source in multi-source datasets.

    A source represents a distinct feature set, typically from different
    instruments, sensors, or measurement types. Each source has its own
    files, loading parameters, and signal type.

    Examples:
        # NIR spectrometer source
        sources:
          - name: "NIR"
            files:
              - path: data/NIR_train.csv
                partition: train
              - path: data/NIR_test.csv
                partition: test
            params:
              header_unit: nm
              signal_type: absorbance

        # Multi-source with shared targets
        sources:
          - name: "NIR"
            files: [...]
          - name: "MIR"
            files: [...]
        targets:
          path: data/targets.csv
          link_by: sample_id
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Unique identifier for this source (e.g., 'NIR', 'MIR', 'sensor1')."
    )

    files: Optional[List[Union[str, SourceFileConfig, Dict[str, Any]]]] = Field(
        default=None,
        description="List of files for this source. Can be paths or file configs."
    )

    # Convenience: direct paths for train/test instead of files list
    train_x: Optional[str] = Field(
        default=None,
        description="Direct path to training features for this source."
    )

    test_x: Optional[str] = Field(
        default=None,
        description="Direct path to test features for this source."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters for all files in this source."
    )

    link_by: Optional[str] = Field(
        default=None,
        description="Column to use for linking samples across sources."
    )

    @model_validator(mode="after")
    def validate_source_files(self) -> "SourceConfig":
        """Validate that source has at least one data source."""
        has_files = self.files is not None and len(self.files) > 0
        has_direct = self.train_x is not None or self.test_x is not None

        if not has_files and not has_direct:
            raise ValueError(
                f"Source '{self.name}' must have either 'files' or 'train_x'/'test_x' specified."
            )

        if has_files and has_direct:
            raise ValueError(
                f"Source '{self.name}' cannot have both 'files' and 'train_x'/'test_x'. "
                f"Use one format or the other."
            )

        return self

    def get_train_paths(self) -> List[str]:
        """Get all training file paths for this source.

        Returns:
            List of paths to training files.
        """
        if self.train_x:
            return [self.train_x]

        paths = []
        if self.files:
            for f in self.files:
                if isinstance(f, str):
                    # Infer partition from path
                    lower_path = f.lower()
                    if any(p in lower_path for p in ('train', 'cal')):
                        paths.append(f)
                elif isinstance(f, (SourceFileConfig, dict)):
                    file_dict = f if isinstance(f, dict) else f.model_dump()
                    partition = file_dict.get('partition')
                    if partition == PartitionType.TRAIN or partition == 'train':
                        paths.append(file_dict['path'])
                    elif partition is None:
                        # Infer from path
                        lower_path = file_dict['path'].lower()
                        if any(p in lower_path for p in ('train', 'cal')):
                            paths.append(file_dict['path'])
        return paths

    def get_test_paths(self) -> List[str]:
        """Get all test file paths for this source.

        Returns:
            List of paths to test files.
        """
        if self.test_x:
            return [self.test_x]

        paths = []
        if self.files:
            for f in self.files:
                if isinstance(f, str):
                    # Infer partition from path
                    lower_path = f.lower()
                    if any(p in lower_path for p in ('test', 'val')):
                        paths.append(f)
                elif isinstance(f, (SourceFileConfig, dict)):
                    file_dict = f if isinstance(f, dict) else f.model_dump()
                    partition = file_dict.get('partition')
                    if partition == PartitionType.TEST or partition == 'test':
                        paths.append(file_dict['path'])
                    elif partition is None:
                        # Infer from path
                        lower_path = file_dict['path'].lower()
                        if any(p in lower_path for p in ('test', 'val')):
                            paths.append(file_dict['path'])
        return paths


class SharedTargetsConfig(BaseModel):
    """Configuration for shared targets in multi-source datasets.

    When using multiple sources, targets can be shared across all sources.
    This configuration specifies how to load and link targets.

    Examples:
        # Simple shared targets
        targets:
          path: data/targets.csv
          link_by: sample_id

        # With column selection
        targets:
          path: data/all_data.csv
          columns: [0]  # First column is target
          link_by: sample_id
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the targets file."
    )

    columns: Optional[ColumnSelection] = Field(
        default=None,
        description="Column selection for targets (if file has multiple columns)."
    )

    link_by: Optional[str] = Field(
        default=None,
        description="Column to use for linking targets to feature sources."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters for the targets file."
    )

    partition: Optional[PartitionType] = Field(
        default=None,
        description="If specified, this targets file is for a specific partition only."
    )


class SharedMetadataConfig(BaseModel):
    """Configuration for shared metadata in multi-source datasets.

    When using multiple sources, metadata can be shared across all sources.
    This configuration specifies how to load and link metadata.

    Examples:
        # Simple shared metadata
        metadata:
          path: data/metadata.csv
          link_by: sample_id
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the metadata file."
    )

    columns: Optional[ColumnSelection] = Field(
        default=None,
        description="Column selection for metadata (if file has multiple columns)."
    )

    link_by: Optional[str] = Field(
        default=None,
        description="Column to use for linking metadata to feature sources."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters for the metadata file."
    )

    partition: Optional[PartitionType] = Field(
        default=None,
        description="If specified, this metadata file is for a specific partition only."
    )


# =============================================================================
# Variation configuration schema (for feature variations / preprocessed data)
# =============================================================================


class PreprocessingApplied(BaseModel):
    """Metadata about preprocessing that was applied offline.

    This is informational only - helps track provenance of preprocessed data.

    Example:
        preprocessing_applied:
          - type: "SNV"
            description: "Standard Normal Variate"
            software: "OPUS 8.0"
          - type: "SG_smooth"
            params:
              window: 15
              polyorder: 2
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        description="Type of preprocessing (e.g., 'SNV', 'MSC', 'SG_derivative')."
    )

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the preprocessing."
    )

    software: Optional[str] = Field(
        default=None,
        description="Software used to apply the preprocessing."
    )

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters used for the preprocessing."
    )


class VariationFileConfig(BaseModel):
    """Configuration for a single file within a variation.

    Similar to SourceFileConfig but for variation context.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the data file."
    )

    partition: Optional[PartitionType] = Field(
        default=None,
        description="Which partition this file belongs to."
    )

    columns: Optional[ColumnConfig] = Field(
        default=None,
        description="Column selection and role assignment."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters specific to this file."
    )

    header: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Header configuration (unit, signal_type, etc.)."
    )


class VariationConfig(BaseModel):
    """Configuration for a single feature variation.

    A variation represents a different "view" of the same samples, such as:
    - Pre-computed preprocessing (SNV, MSC, derivatives)
    - Different variables from time series data
    - Different feature representations

    All variations must have the same number of samples (rows).

    Examples:
        # Simple variation
        variations:
          - name: "raw"
            files:
              - path: data/spectra_raw.csv
                partition: train

        # Variation with preprocessing provenance
        variations:
          - name: "snv"
            description: "SNV preprocessed spectra"
            preprocessing_applied:
              - type: "SNV"
                software: "OPUS 8.0"
            files:
              - path: data/spectra_snv.csv
                partition: train

        # Using direct paths
        variations:
          - name: "raw"
            train_x: data/X_raw_train.csv
            test_x: data/X_raw_test.csv
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Unique identifier for this variation (e.g., 'raw', 'snv', 'derivative')."
    )

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this variation."
    )

    files: Optional[List[Union[str, VariationFileConfig, Dict[str, Any]]]] = Field(
        default=None,
        description="List of files for this variation. Can be paths or file configs."
    )

    # Convenience: direct paths for train/test instead of files list
    train_x: Optional[str] = Field(
        default=None,
        description="Direct path to training features for this variation."
    )

    test_x: Optional[str] = Field(
        default=None,
        description="Direct path to test features for this variation."
    )

    params: Optional[LoadingParams] = Field(
        default=None,
        description="Loading parameters for all files in this variation."
    )

    preprocessing_applied: Optional[List[PreprocessingApplied]] = Field(
        default=None,
        description="Provenance information about preprocessing applied offline."
    )

    @model_validator(mode="after")
    def validate_variation_files(self) -> "VariationConfig":
        """Validate that variation has at least one data source."""
        has_files = self.files is not None and len(self.files) > 0
        has_direct = self.train_x is not None or self.test_x is not None

        if not has_files and not has_direct:
            raise ValueError(
                f"Variation '{self.name}' must have either 'files' or 'train_x'/'test_x' specified."
            )

        if has_files and has_direct:
            raise ValueError(
                f"Variation '{self.name}' cannot have both 'files' and 'train_x'/'test_x'. "
                f"Use one format or the other."
            )

        return self

    def get_train_paths(self) -> List[str]:
        """Get all training file paths for this variation.

        Returns:
            List of paths to training files.
        """
        if self.train_x:
            return [self.train_x]

        paths = []
        if self.files:
            for f in self.files:
                if isinstance(f, str):
                    # Infer partition from path
                    lower_path = f.lower()
                    if any(p in lower_path for p in ('train', 'cal')):
                        paths.append(f)
                elif isinstance(f, (VariationFileConfig, dict)):
                    file_dict = f if isinstance(f, dict) else f.model_dump()
                    partition = file_dict.get('partition')
                    if partition == PartitionType.TRAIN or partition == 'train':
                        paths.append(file_dict['path'])
                    elif partition is None:
                        # Infer from path
                        lower_path = file_dict['path'].lower()
                        if any(p in lower_path for p in ('train', 'cal')):
                            paths.append(file_dict['path'])
        return paths

    def get_test_paths(self) -> List[str]:
        """Get all test file paths for this variation.

        Returns:
            List of paths to test files.
        """
        if self.test_x:
            return [self.test_x]

        paths = []
        if self.files:
            for f in self.files:
                if isinstance(f, str):
                    # Infer partition from path
                    lower_path = f.lower()
                    if any(p in lower_path for p in ('test', 'val')):
                        paths.append(f)
                elif isinstance(f, (VariationFileConfig, dict)):
                    file_dict = f if isinstance(f, dict) else f.model_dump()
                    partition = file_dict.get('partition')
                    if partition == PartitionType.TEST or partition == 'test':
                        paths.append(file_dict['path'])
                    elif partition is None:
                        # Infer from path
                        lower_path = file_dict['path'].lower()
                        if any(p in lower_path for p in ('test', 'val')):
                            paths.append(file_dict['path'])
        return paths


# =============================================================================
# Main dataset configuration schema
# =============================================================================


class DatasetConfigSchema(BaseModel):
    """Complete dataset configuration schema.

    This model represents the normalized, validated form of a dataset configuration.
    It supports both the legacy format (train_x, test_x, etc.) and is designed
    to be extensible for the new files syntax.

    All input configurations are normalized to this schema before processing.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for forward compatibility
        validate_assignment=True,
        arbitrary_types_allowed=True,  # Allow numpy arrays
    )

    # --- Dataset identification ---
    name: Optional[str] = Field(
        default=None,
        description="Dataset name. Derived from file/folder path if not specified."
    )

    description: Optional[str] = Field(
        default=None,
        description="Optional description of the dataset."
    )

    # --- Task configuration ---
    task_type: Optional[TaskType] = Field(
        default=None,
        description="Task type: 'regression', 'binary_classification', 'multiclass_classification', 'auto'"
    )

    # --- Legacy format paths ---
    # Using Any type to support numpy arrays without schema generation issues
    train_x: Optional[Any] = Field(
        default=None,
        description="Training features: path(s) to CSV file(s) or numpy array."
    )

    train_y: Optional[Any] = Field(
        default=None,
        description="Training targets: path to CSV file, column indices, or numpy array."
    )

    train_group: Optional[Any] = Field(
        default=None,
        description="Training metadata: path to CSV file or DataFrame."
    )

    test_x: Optional[Any] = Field(
        default=None,
        description="Test features: path(s) to CSV file(s) or numpy array."
    )

    test_y: Optional[Any] = Field(
        default=None,
        description="Test targets: path to CSV file, column indices, or numpy array."
    )

    test_group: Optional[Any] = Field(
        default=None,
        description="Test metadata: path to CSV file or DataFrame."
    )

    # --- Legacy format filters (for column selection from X) ---
    train_x_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from train_x."
    )

    train_y_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from train_y (or from train_x if train_y is None)."
    )

    train_group_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from train_group."
    )

    test_x_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from test_x."
    )

    test_y_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from test_y (or from test_x if test_y is None)."
    )

    test_group_filter: Optional[List[int]] = Field(
        default=None,
        description="Column indices to select from test_group."
    )

    # --- Loading parameters (legacy format) ---
    global_params: Optional[LoadingParams] = Field(
        default=None,
        description="Global loading parameters applied to all files."
    )

    train_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters applied to all training files."
    )

    test_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters applied to all test files."
    )

    train_x_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading train_x."
    )

    train_y_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading train_y."
    )

    train_group_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading train_group."
    )

    test_x_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading test_x."
    )

    test_y_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading test_y."
    )

    test_group_params: Optional[LoadingParams] = Field(
        default=None,
        description="Parameters for loading test_group."
    )

    # --- Aggregation settings ---
    aggregate: Optional[Union[str, bool]] = Field(
        default=None,
        description="Aggregation column name, True for y-based aggregation, or None."
    )

    aggregate_method: Optional[AggregateMethod] = Field(
        default=None,
        description="Aggregation method: 'mean', 'median', or 'vote'."
    )

    aggregate_exclude_outliers: Optional[bool] = Field(
        default=None,
        description="Whether to exclude outliers before aggregation."
    )

    # --- New format (stubs for future implementation) ---
    files: Optional[List[FileConfig]] = Field(
        default=None,
        description="List of file configurations (new format)."
    )

    sources: Optional[List[SourceConfig]] = Field(
        default=None,
        description="Multi-source configuration for sensor fusion / multi-instrument data."
    )

    shared_targets: Optional[Union[SharedTargetsConfig, List[SharedTargetsConfig]]] = Field(
        default=None,
        description="Shared targets configuration for multi-source datasets."
    )

    shared_metadata: Optional[Union[SharedMetadataConfig, List[SharedMetadataConfig]]] = Field(
        default=None,
        description="Shared metadata configuration for multi-source datasets."
    )

    variations: Optional[List[VariationConfig]] = Field(
        default=None,
        description="Feature variations configuration for preprocessed data or multi-variable datasets."
    )

    variation_mode: Optional[VariationMode] = Field(
        default=None,
        description="How to handle feature variations: 'separate', 'concat', 'select', 'compare'."
    )

    variation_select: Optional[List[str]] = Field(
        default=None,
        description="When variation_mode='select', list of variation names to use."
    )

    variation_prefix: Optional[bool] = Field(
        default=None,
        description="When variation_mode='concat', whether to prefix column names with variation names."
    )

    # --- Fold configuration ---
    folds: Optional[Union[FoldConfig, List[Dict[str, Any]], str]] = Field(
        default=None,
        description="Cross-validation fold configuration. Can be: "
                    "FoldConfig object, list of fold dicts, or path to fold file."
    )

    # --- Validators ---

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_task_type(cls, v: Any) -> Optional[TaskType]:
        """Normalize task_type to enum."""
        if v is None:
            return None
        if isinstance(v, TaskType):
            return v
        if isinstance(v, str):
            try:
                return TaskType(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid task_type: '{v}'. "
                    f"Valid values: {[t.value for t in TaskType]}"
                )
        return v

    @field_validator("aggregate_method", mode="before")
    @classmethod
    def normalize_aggregate_method(cls, v: Any) -> Optional[AggregateMethod]:
        """Normalize aggregate_method to enum."""
        if v is None:
            return None
        if isinstance(v, AggregateMethod):
            return v
        if isinstance(v, str):
            try:
                return AggregateMethod(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid aggregate_method: '{v}'. "
                    f"Valid values: {[m.value for m in AggregateMethod]}"
                )
        return v

    @field_validator(
        "global_params", "train_params", "test_params",
        "train_x_params", "train_y_params", "train_group_params",
        "test_x_params", "test_y_params", "test_group_params",
        mode="before"
    )
    @classmethod
    def parse_loading_params(cls, v: Any) -> Optional[LoadingParams]:
        """Parse dict to LoadingParams if needed."""
        if v is None:
            return None
        if isinstance(v, LoadingParams):
            return v
        if isinstance(v, dict):
            return LoadingParams(**v)
        return v

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: Any) -> Optional[List[SourceConfig]]:
        """Parse sources list to SourceConfig objects."""
        if v is None:
            return None
        if not isinstance(v, list):
            return v

        parsed = []
        for item in v:
            if isinstance(item, SourceConfig):
                parsed.append(item)
            elif isinstance(item, dict):
                parsed.append(SourceConfig(**item))
            else:
                raise ValueError(f"Invalid source config type: {type(item)}")
        return parsed

    @field_validator("shared_targets", mode="before")
    @classmethod
    def parse_shared_targets(cls, v: Any) -> Optional[Union[SharedTargetsConfig, List[SharedTargetsConfig]]]:
        """Parse shared targets configuration."""
        if v is None:
            return None
        if isinstance(v, SharedTargetsConfig):
            return v
        if isinstance(v, dict):
            return SharedTargetsConfig(**v)
        if isinstance(v, list):
            return [
                SharedTargetsConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v

    @field_validator("shared_metadata", mode="before")
    @classmethod
    def parse_shared_metadata(cls, v: Any) -> Optional[Union[SharedMetadataConfig, List[SharedMetadataConfig]]]:
        """Parse shared metadata configuration."""
        if v is None:
            return None
        if isinstance(v, SharedMetadataConfig):
            return v
        if isinstance(v, dict):
            return SharedMetadataConfig(**v)
        if isinstance(v, list):
            return [
                SharedMetadataConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v

    @field_validator("variations", mode="before")
    @classmethod
    def parse_variations(cls, v: Any) -> Optional[List[VariationConfig]]:
        """Parse variations list to VariationConfig objects."""
        if v is None:
            return None
        if not isinstance(v, list):
            return v

        parsed = []
        for item in v:
            if isinstance(item, VariationConfig):
                parsed.append(item)
            elif isinstance(item, dict):
                parsed.append(VariationConfig(**item))
            else:
                raise ValueError(f"Invalid variation config type: {type(item)}")
        return parsed

    @field_validator("variation_mode", mode="before")
    @classmethod
    def normalize_variation_mode(cls, v: Any) -> Optional[VariationMode]:
        """Normalize variation_mode to enum."""
        if v is None:
            return None
        if isinstance(v, VariationMode):
            return v
        if isinstance(v, str):
            try:
                return VariationMode(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid variation_mode: '{v}'. "
                    f"Valid values: {[m.value for m in VariationMode]}"
                )
        return v

    @model_validator(mode="after")
    def validate_data_sources(self) -> "DatasetConfigSchema":
        """Validate that at least one data source is specified."""
        has_legacy = self.train_x is not None or self.test_x is not None
        has_files = self.files is not None and len(self.files) > 0
        has_sources = self.sources is not None and len(self.sources) > 0
        has_variations = self.variations is not None and len(self.variations) > 0

        if not has_legacy and not has_files and not has_sources and not has_variations:
            # This is a warning, not an error - empty configs are allowed
            # The actual error will be raised when trying to load data
            pass

        # Validate variation_select if provided
        if self.variation_select and has_variations:
            variation_names = {v.name for v in self.variations}
            unknown_variations = set(self.variation_select) - variation_names
            if unknown_variations:
                raise ValueError(
                    f"variation_select contains unknown variation names: {unknown_variations}. "
                    f"Available variations: {variation_names}"
                )

        # Validate variation_mode 'select' requires variation_select
        if self.variation_mode == VariationMode.SELECT and not self.variation_select:
            raise ValueError(
                "variation_mode='select' requires 'variation_select' to specify which variations to use."
            )

        return self

    # --- Utility methods ---

    def get_effective_params(
        self,
        partition: str,
        data_type: str
    ) -> LoadingParams:
        """Get effective loading parameters for a specific data file.

        Parameters are merged with precedence: specific > partition > global.

        Args:
            partition: 'train' or 'test'
            data_type: 'x', 'y', or 'group'

        Returns:
            Merged LoadingParams.
        """
        # Start with global params
        result = self.global_params or LoadingParams()

        # Merge partition-level params
        partition_params = self.train_params if partition == "train" else self.test_params
        if partition_params:
            result = partition_params.merge_with(result)

        # Merge file-specific params
        file_params_attr = f"{partition}_{data_type}_params"
        file_params = getattr(self, file_params_attr, None)
        if file_params:
            result = file_params.merge_with(result)

        return result

    def is_legacy_format(self) -> bool:
        """Check if this config uses legacy format (train_x/test_x)."""
        return (
            self.train_x is not None or
            self.test_x is not None or
            self.train_y is not None or
            self.test_y is not None
        )

    def is_files_format(self) -> bool:
        """Check if this config uses new files format."""
        return self.files is not None and len(self.files) > 0

    def is_multi_source(self) -> bool:
        """Check if this config has multiple feature sources."""
        # Legacy format multi-source
        if isinstance(self.train_x, list) and len(self.train_x) > 1:
            return True
        if isinstance(self.test_x, list) and len(self.test_x) > 1:
            return True
        # New format multi-source
        if self.sources is not None and len(self.sources) > 1:
            return True
        return False

    def is_sources_format(self) -> bool:
        """Check if this config uses the new sources format."""
        return self.sources is not None and len(self.sources) > 0

    def get_source_names(self) -> List[str]:
        """Get names of all sources in this config.

        Returns:
            List of source names, or empty list if not multi-source.
        """
        if self.sources:
            return [s.name for s in self.sources]
        return []

    def get_source_count(self) -> int:
        """Get the number of feature sources.

        Returns:
            Number of sources (1 for single-source, >1 for multi-source).
        """
        if self.sources:
            return len(self.sources)
        # Legacy format
        if isinstance(self.train_x, list):
            return len(self.train_x)
        if isinstance(self.test_x, list):
            return len(self.test_x)
        return 1

    def is_variations_format(self) -> bool:
        """Check if this config uses the variations format."""
        return self.variations is not None and len(self.variations) > 0

    def get_variation_names(self) -> List[str]:
        """Get names of all variations in this config.

        Returns:
            List of variation names, or empty list if no variations.
        """
        if self.variations:
            return [v.name for v in self.variations]
        return []

    def get_variation_count(self) -> int:
        """Get the number of feature variations.

        Returns:
            Number of variations.
        """
        if self.variations:
            return len(self.variations)
        return 0

    def get_selected_variations(self) -> List[VariationConfig]:
        """Get the variations to use based on variation_mode and variation_select.

        For mode='select', returns only the selected variations.
        For other modes, returns all variations.

        Returns:
            List of VariationConfig objects to use.
        """
        if not self.variations:
            return []

        if self.variation_mode == VariationMode.SELECT and self.variation_select:
            selected = []
            for v in self.variations:
                if v.name in self.variation_select:
                    selected.append(v)
            return selected

        return list(self.variations)

    def variations_to_legacy_format(self) -> Dict[str, Any]:
        """Convert variations format to legacy format for backward compatibility.

        This converts the variations syntax to the train_x/test_x format
        that existing loaders understand. The conversion depends on variation_mode:

        - separate: Returns config for first variation (caller handles multiple runs)
        - concat: Returns list of paths to be concatenated
        - select: Returns config for selected variations only
        - compare: Same as separate (caller handles comparison)

        Returns:
            Dictionary with legacy format configuration.
        """
        if not self.variations:
            return self.to_dict()

        result = {}

        # Copy non-variation fields
        if self.name:
            result['name'] = self.name
        if self.description:
            result['description'] = self.description
        if self.task_type:
            result['task_type'] = self.task_type.value if hasattr(self.task_type, 'value') else self.task_type
        if self.global_params:
            result['global_params'] = self.global_params.model_dump(exclude_none=True)
        if self.aggregate:
            result['aggregate'] = self.aggregate
        if self.aggregate_method:
            result['aggregate_method'] = self.aggregate_method.value if hasattr(self.aggregate_method, 'value') else self.aggregate_method

        # Get variations to use
        variations_to_use = self.get_selected_variations()
        mode = self.variation_mode or VariationMode.SEPARATE

        if mode in (VariationMode.SEPARATE, VariationMode.COMPARE):
            # For separate/compare, return first variation
            # Caller is responsible for handling multiple variations
            if variations_to_use:
                first_var = variations_to_use[0]
                train_paths = first_var.get_train_paths()
                test_paths = first_var.get_test_paths()

                if train_paths:
                    result['train_x'] = train_paths[0] if len(train_paths) == 1 else train_paths
                if test_paths:
                    result['test_x'] = test_paths[0] if len(test_paths) == 1 else test_paths

                if first_var.params:
                    result['train_x_params'] = first_var.params.model_dump(exclude_none=True)
                    result['test_x_params'] = first_var.params.model_dump(exclude_none=True)

        elif mode == VariationMode.CONCAT:
            # Concatenate all variations (multi-source style)
            train_x_paths = []
            test_x_paths = []
            train_x_params = []
            test_x_params = []

            for var in variations_to_use:
                train_paths = var.get_train_paths()
                test_paths = var.get_test_paths()

                if train_paths:
                    train_x_paths.extend(train_paths)
                    if var.params:
                        for _ in train_paths:
                            train_x_params.append(var.params.model_dump(exclude_none=True))

                if test_paths:
                    test_x_paths.extend(test_paths)
                    if var.params:
                        for _ in test_paths:
                            test_x_params.append(var.params.model_dump(exclude_none=True))

            if train_x_paths:
                result['train_x'] = train_x_paths if len(train_x_paths) > 1 else train_x_paths[0]
            if test_x_paths:
                result['test_x'] = test_x_paths if len(test_x_paths) > 1 else test_x_paths[0]
            if train_x_params:
                result['train_x_params'] = train_x_params if len(train_x_params) > 1 else train_x_params[0]
            if test_x_params:
                result['test_x_params'] = test_x_params if len(test_x_params) > 1 else test_x_params[0]

        elif mode == VariationMode.SELECT:
            # Same as concat but only for selected variations
            train_x_paths = []
            test_x_paths = []

            for var in variations_to_use:
                train_paths = var.get_train_paths()
                test_paths = var.get_test_paths()

                if train_paths:
                    train_x_paths.extend(train_paths)
                if test_paths:
                    test_x_paths.extend(test_paths)

            if train_x_paths:
                result['train_x'] = train_x_paths if len(train_x_paths) > 1 else train_x_paths[0]
            if test_x_paths:
                result['test_x'] = test_x_paths if len(test_x_paths) > 1 else test_x_paths[0]

        # Handle shared targets
        if self.shared_targets:
            targets_list = self.shared_targets if isinstance(self.shared_targets, list) else [self.shared_targets]
            for targets in targets_list:
                if targets.partition == PartitionType.TRAIN or targets.partition == 'train' or targets.partition is None:
                    result['train_y'] = targets.path
                    if targets.params:
                        result['train_y_params'] = targets.params.model_dump(exclude_none=True)
                if targets.partition == PartitionType.TEST or targets.partition == 'test' or targets.partition is None:
                    result['test_y'] = targets.path
                    if targets.params:
                        result['test_y_params'] = targets.params.model_dump(exclude_none=True)

        # Handle shared metadata
        if self.shared_metadata:
            metadata_list = self.shared_metadata if isinstance(self.shared_metadata, list) else [self.shared_metadata]
            for metadata in metadata_list:
                if metadata.partition == PartitionType.TRAIN or metadata.partition == 'train' or metadata.partition is None:
                    result['train_group'] = metadata.path
                    if metadata.params:
                        result['train_group_params'] = metadata.params.model_dump(exclude_none=True)
                if metadata.partition == PartitionType.TEST or metadata.partition == 'test' or metadata.partition is None:
                    result['test_group'] = metadata.path
                    if metadata.params:
                        result['test_group_params'] = metadata.params.model_dump(exclude_none=True)

        # Store variation metadata for advanced processing
        result['_variations'] = [
            {
                'name': v.name,
                'description': v.description,
                'preprocessing_applied': [p.model_dump() for p in v.preprocessing_applied] if v.preprocessing_applied else None,
                'params': v.params.model_dump(exclude_none=True) if v.params else None,
            }
            for v in variations_to_use
        ]
        result['_variation_mode'] = mode.value if mode else None

        return result

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert sources or variations format to legacy format for backward compatibility.

        This converts the sources/variations syntax to the train_x/test_x array syntax
        that existing loaders understand.

        Returns:
            Dictionary with legacy format configuration.
        """
        # Check for variations format first
        if self.is_variations_format():
            return self.variations_to_legacy_format()

        if not self.sources:
            return self.to_dict()

        result = {}

        # Copy non-source fields
        if self.name:
            result['name'] = self.name
        if self.description:
            result['description'] = self.description
        if self.task_type:
            result['task_type'] = self.task_type.value if hasattr(self.task_type, 'value') else self.task_type
        if self.global_params:
            result['global_params'] = self.global_params.model_dump(exclude_none=True)
        if self.aggregate:
            result['aggregate'] = self.aggregate
        if self.aggregate_method:
            result['aggregate_method'] = self.aggregate_method.value if hasattr(self.aggregate_method, 'value') else self.aggregate_method

        # Convert sources to train_x/test_x lists
        train_x_paths = []
        test_x_paths = []
        train_x_params = []
        test_x_params = []

        for source in self.sources:
            train_paths = source.get_train_paths()
            test_paths = source.get_test_paths()

            if train_paths:
                train_x_paths.extend(train_paths)
                if source.params:
                    for _ in train_paths:
                        train_x_params.append(source.params.model_dump(exclude_none=True))

            if test_paths:
                test_x_paths.extend(test_paths)
                if source.params:
                    for _ in test_paths:
                        test_x_params.append(source.params.model_dump(exclude_none=True))

        if train_x_paths:
            result['train_x'] = train_x_paths if len(train_x_paths) > 1 else train_x_paths[0]
        if test_x_paths:
            result['test_x'] = test_x_paths if len(test_x_paths) > 1 else test_x_paths[0]

        if train_x_params:
            result['train_x_params'] = train_x_params if len(train_x_params) > 1 else train_x_params[0]
        if test_x_params:
            result['test_x_params'] = test_x_params if len(test_x_params) > 1 else test_x_params[0]

        # Handle shared targets
        if self.shared_targets:
            targets_list = self.shared_targets if isinstance(self.shared_targets, list) else [self.shared_targets]
            for targets in targets_list:
                if targets.partition == PartitionType.TRAIN or targets.partition == 'train' or targets.partition is None:
                    result['train_y'] = targets.path
                    if targets.params:
                        result['train_y_params'] = targets.params.model_dump(exclude_none=True)
                if targets.partition == PartitionType.TEST or targets.partition == 'test' or targets.partition is None:
                    result['test_y'] = targets.path
                    if targets.params:
                        result['test_y_params'] = targets.params.model_dump(exclude_none=True)

        # Handle shared metadata
        if self.shared_metadata:
            metadata_list = self.shared_metadata if isinstance(self.shared_metadata, list) else [self.shared_metadata]
            for metadata in metadata_list:
                if metadata.partition == PartitionType.TRAIN or metadata.partition == 'train' or metadata.partition is None:
                    result['train_group'] = metadata.path
                    if metadata.params:
                        result['train_group_params'] = metadata.params.model_dump(exclude_none=True)
                if metadata.partition == PartitionType.TEST or metadata.partition == 'test' or metadata.partition is None:
                    result['test_group'] = metadata.path
                    if metadata.params:
                        result['test_group_params'] = metadata.params.model_dump(exclude_none=True)

        # Store source metadata for advanced processing
        result['_sources'] = [
            {
                'name': s.name,
                'link_by': s.link_by,
                'params': s.params.model_dump(exclude_none=True) if s.params else None,
            }
            for s in self.sources
        ]

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfigSchema":
        """Create from dictionary."""
        return cls(**data)
