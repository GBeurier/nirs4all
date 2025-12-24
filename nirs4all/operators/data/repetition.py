"""Repetition transformation operator configuration.

This module provides configuration dataclasses for transforming spectral
repetitions (multiple spectra per sample) into either separate sources
or additional preprocessings.

When samples have multiple repetitions (e.g., 4 spectra per leaf sample),
these operators reshape the dataset structure:

- `rep_to_sources`: Each repetition becomes a separate data source
  Input: 1 source × (120 samples, 1 pp, 500 features)
  Output: 4 sources × (30 samples, 1 pp, 500 features)

- `rep_to_pp`: Repetitions become additional preprocessing slots
  Input: 1 source × (120 samples, 1 pp, 500 features)
  Output: 1 source × (30 samples, 4 pp, 500 features)

Example:
    >>> # Transform 4 repetitions per sample into 4 sources
    >>> {"rep_to_sources": "Sample_ID"}
    >>>
    >>> # Transform repetitions into preprocessing dimension
    >>> {"rep_to_pp": "Sample_ID"}
    >>>
    >>> # Group by target value instead of metadata column
    >>> {"rep_to_sources": "y"}
    >>>
    >>> # Advanced configuration with options
    >>> {"rep_to_sources": {
    ...     "column": "Sample_ID",
    ...     "on_unequal": "drop",
    ...     "source_names": "rep_{i}"
    ... }}
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import warnings


class UnequelRepsStrategy(Enum):
    """Strategy for handling samples with unequal repetition counts.

    When samples have different numbers of repetitions, this controls
    how the transformation handles the mismatch.

    Attributes:
        ERROR: Raise an error if repetition counts differ (default, strictest).
        PAD: Pad shorter groups with NaN/zeros to match the longest.
        DROP: Drop samples that don't have the expected repetition count.
        TRUNCATE: Truncate all groups to the minimum repetition count.
    """

    ERROR = "error"
    PAD = "pad"
    DROP = "drop"
    TRUNCATE = "truncate"


@dataclass
class RepetitionConfig:
    """Configuration for repetition transformation operations.

    This dataclass provides configuration for `rep_to_sources` and `rep_to_pp`
    keywords, which reshape datasets based on sample repetitions.

    Repetitions are identified by a metadata column (e.g., "Sample_ID") that
    groups multiple spectra belonging to the same physical sample.

    Attributes:
        column: Metadata column identifying sample groups, or special values:
            - None (default): Use dataset's `aggregate` column from DatasetConfigs
            - "y": Group by target values
            - str: Explicit metadata column name
        on_unequal: Strategy when samples have different repetition counts.
            - "error" (default): Raise error if counts differ
            - "pad": Pad shorter groups with NaN to match longest
            - "drop": Drop samples without expected repetition count
            - "truncate": Use minimum count across all samples
        expected_reps: Expected number of repetitions per sample.
            If None (default), inferred from data (mode of group sizes).
            If specified, validates all groups match this count.
        source_names: Naming template for new sources (rep_to_sources only).
            - None (default): Uses "rep_0", "rep_1", etc.
            - str with {i}: Template like "rep_{i}" or "spectrum_{i}"
            - List[str]: Explicit names for each repetition
        pp_names: Naming template for new preprocessings (rep_to_pp only).
            - None (default): Uses "{original}_rep{i}" format
            - str with {i} and {pp}: Template like "{pp}_r{i}"
            - List[str]: Explicit names (length = n_reps * n_existing_pp)
        preserve_order: Whether to preserve sample order within groups.
            If True (default), repetitions are ordered by their row position.
            If False, order within groups is undefined.
        aggregate_metadata: How to handle metadata after grouping.
            - "first" (default): Keep metadata from first repetition
            - "validate": Ensure all reps have identical metadata, error if not
            - "drop": Remove metadata columns that differ across repetitions

    Example:
        >>> # Use dataset's aggregate column (simplest)
        >>> RepetitionConfig()
        >>>
        >>> # Simple column-based grouping
        >>> RepetitionConfig(column="Sample_ID")
        >>>
        >>> # Group by target value with padding
        >>> RepetitionConfig(column="y", on_unequal="pad")
        >>>
        >>> # Explicit repetition count validation
        >>> RepetitionConfig(
        ...     column="Leaf_ID",
        ...     expected_reps=4,
        ...     on_unequal="error"
        ... )
        >>>
        >>> # Custom source naming
        >>> RepetitionConfig(
        ...     column="Sample_ID",
        ...     source_names="measurement_{i}"
        ... )
    """

    column: Optional[str] = None  # None = use dataset.aggregate
    on_unequal: str = "error"
    expected_reps: Optional[int] = None
    source_names: Optional[Union[str, List[str]]] = None
    pp_names: Optional[Union[str, List[str]]] = None
    preserve_order: bool = True
    aggregate_metadata: str = "first"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Note: column=None is valid (means use dataset.aggregate at runtime)

        # Validate on_unequal
        valid_strategies = ("error", "pad", "drop", "truncate")
        if self.on_unequal not in valid_strategies:
            raise ValueError(
                f"on_unequal must be one of {valid_strategies}, got '{self.on_unequal}'"
            )

        # Validate expected_reps
        if self.expected_reps is not None:
            if not isinstance(self.expected_reps, int) or self.expected_reps < 1:
                raise ValueError(
                    f"expected_reps must be a positive integer, got {self.expected_reps}"
                )

        # Validate aggregate_metadata
        valid_agg = ("first", "validate", "drop")
        if self.aggregate_metadata not in valid_agg:
            raise ValueError(
                f"aggregate_metadata must be one of {valid_agg}, got '{self.aggregate_metadata}'"
            )

        # Validate source_names format
        if isinstance(self.source_names, str):
            if "{i}" not in self.source_names:
                warnings.warn(
                    f"source_names template '{self.source_names}' does not contain '{{i}}'. "
                    "All sources will have the same name. Consider using 'rep_{{i}}' format.",
                    UserWarning,
                    stacklevel=2
                )

        # Validate pp_names format
        if isinstance(self.pp_names, str):
            if "{i}" not in self.pp_names and "{pp}" not in self.pp_names:
                warnings.warn(
                    f"pp_names template '{self.pp_names}' does not contain '{{i}}' or '{{pp}}'. "
                    "All preprocessings will have the same name.",
                    UserWarning,
                    stacklevel=2
                )

    @property
    def is_y_grouping(self) -> bool:
        """Check if grouping by target values.

        Returns:
            True if column is "y" (case-insensitive).
        """
        return self.column is not None and self.column.lower() == "y"

    @property
    def uses_dataset_aggregate(self) -> bool:
        """Check if using dataset's aggregate column.

        Returns:
            True if column is None (will use dataset.aggregate at runtime).
        """
        return self.column is None

    def resolve_column(self, dataset_aggregate: Optional[str]) -> str:
        """Resolve the actual column to use for grouping.

        Args:
            dataset_aggregate: The aggregate value from dataset (column name, "y", or None).

        Returns:
            The resolved column name to use.

        Raises:
            ValueError: If no column specified and dataset has no aggregate setting.
        """
        if self.column is not None:
            return self.column

        # Use dataset's aggregate setting
        if dataset_aggregate is None:
            raise ValueError(
                "No column specified for repetition grouping and dataset has no "
                "aggregate setting. Either specify a column in the step "
                "(e.g., {'rep_to_sources': 'Sample_ID'}) or set aggregate in "
                "DatasetConfigs. [Error: REP-E001]"
            )

        return dataset_aggregate

    def get_unequal_strategy(self) -> UnequelRepsStrategy:
        """Get the unequal handling strategy as an enum.

        Returns:
            UnequelRepsStrategy enum value.
        """
        return UnequelRepsStrategy(self.on_unequal)

    def get_source_name(self, rep_index: int) -> str:
        """Generate source name for a given repetition index.

        Args:
            rep_index: Zero-based repetition index.

        Returns:
            Source name string.
        """
        if self.source_names is None:
            return f"rep_{rep_index}"
        elif isinstance(self.source_names, str):
            return self.source_names.format(i=rep_index)
        elif isinstance(self.source_names, list):
            if rep_index < len(self.source_names):
                return self.source_names[rep_index]
            return f"rep_{rep_index}"
        return f"rep_{rep_index}"

    def get_pp_name(self, rep_index: int, original_pp: str) -> str:
        """Generate preprocessing name for a given repetition and original processing.

        Args:
            rep_index: Zero-based repetition index.
            original_pp: Original preprocessing name (e.g., "raw", "snv").

        Returns:
            New preprocessing name string.
        """
        if self.pp_names is None:
            return f"{original_pp}_rep{rep_index}"
        elif isinstance(self.pp_names, str):
            return self.pp_names.format(i=rep_index, pp=original_pp)
        elif isinstance(self.pp_names, list):
            # For list, compute flat index
            # This is complex because we need to know n_existing_pp
            # Fall back to default naming for lists (handled by caller)
            return f"{original_pp}_rep{rep_index}"
        return f"{original_pp}_rep{rep_index}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation for manifest storage.
        """
        result = {
            "column": self.column,
            "on_unequal": self.on_unequal,
            "preserve_order": self.preserve_order,
            "aggregate_metadata": self.aggregate_metadata,
        }

        if self.expected_reps is not None:
            result["expected_reps"] = self.expected_reps

        if self.source_names is not None:
            result["source_names"] = self.source_names

        if self.pp_names is not None:
            result["pp_names"] = self.pp_names

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepetitionConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary representation. If 'column' is missing, uses None (aggregate).

        Returns:
            RepetitionConfig instance.
        """
        return cls(
            column=data.get("column"),  # None if not present = use aggregate
            on_unequal=data.get("on_unequal", "error"),
            expected_reps=data.get("expected_reps"),
            source_names=data.get("source_names"),
            pp_names=data.get("pp_names"),
            preserve_order=data.get("preserve_order", True),
            aggregate_metadata=data.get("aggregate_metadata", "first"),
        )

    @classmethod
    def from_step_value(cls, value: Union[str, bool, Dict[str, Any], None]) -> "RepetitionConfig":
        """Create config from step value (string, bool, or dict).

        Handles multiple syntax styles:
        - None or True: Use dataset's aggregate column
        - str: Explicit column name (or "y" for target grouping)
        - dict: Full configuration with options

        Args:
            value: Step value - column name, True/None for aggregate, or config dict.

        Returns:
            RepetitionConfig instance.

        Example:
            >>> # Use dataset aggregate (simplest)
            >>> RepetitionConfig.from_step_value(True)
            >>> RepetitionConfig.from_step_value(None)
            >>>
            >>> # Explicit column
            >>> RepetitionConfig.from_step_value("Sample_ID")
            >>>
            >>> # Advanced syntax
            >>> RepetitionConfig.from_step_value({
            ...     "column": "Sample_ID",
            ...     "on_unequal": "drop"
            ... })
        """
        if value is None or value is True:
            return cls(column=None)  # Use dataset aggregate
        elif isinstance(value, str):
            return cls(column=value)
        elif isinstance(value, dict):
            return cls.from_dict(value)
        elif isinstance(value, cls):
            return value
        else:
            raise ValueError(
                f"Invalid repetition config type: {type(value).__name__}. "
                f"Expected string (column name), True (use aggregate), or dict with configuration."
            )


# Expose for imports
__all__ = [
    "RepetitionConfig",
    "UnequelRepsStrategy",
]
