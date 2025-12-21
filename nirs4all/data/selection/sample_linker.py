"""
Sample linker for dataset configuration.

This module provides key-based sample linking across multiple data files,
enabling joining of features, targets, and metadata by a common identifier.

Example:
    >>> linker = SampleLinker()
    >>> result = linker.link(
    ...     {"features": features_df, "targets": targets_df},
    ...     link_by="sample_id"
    ... )
    >>> print(result.linked_data)  # Joined DataFrame
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd


class LinkingError(Exception):
    """Raised when sample linking fails."""
    pass


@dataclass
class LinkingResult:
    """Result of a sample linking operation.

    Attributes:
        linked_data: Dictionary of linked DataFrames (key column removed).
        key_column: The column used for linking.
        matched_keys: Set of keys present in all sources.
        missing_keys: Dictionary mapping source names to their missing keys.
        sample_count: Number of linked samples.
        report: Detailed linking report.
    """
    linked_data: Dict[str, pd.DataFrame]
    key_column: str
    matched_keys: Set[Any]
    missing_keys: Dict[str, Set[Any]]
    sample_count: int
    report: Dict[str, Any] = field(default_factory=dict)


class SampleLinker:
    """Link samples across multiple data files by key column.

    Supports multiple linking modes:
    - "inner": Keep only samples present in all sources (default)
    - "left": Keep all samples from the first source
    - "outer": Keep all samples from any source

    Example:
        >>> linker = SampleLinker()
        >>> result = linker.link(
        ...     {
        ...         "X": features_df,    # Has columns: sample_id, feature1, feature2
        ...         "Y": targets_df,     # Has columns: sample_id, target
        ...         "M": metadata_df,    # Has columns: sample_id, group, date
        ...     },
        ...     link_by="sample_id"
        ... )
        >>> # Linked DataFrames have aligned rows
        >>> X_linked = result.linked_data["X"]  # Without sample_id column
    """

    def __init__(
        self,
        mode: str = "inner",
        on_missing: str = "warn",
    ):
        """Initialize the sample linker.

        Args:
            mode: Linking mode - "inner", "left", or "outer".
            on_missing: Action when keys are missing - "warn", "error", or "ignore".
        """
        if mode not in ("inner", "left", "outer"):
            raise ValueError(f"Invalid mode: {mode}. Expected 'inner', 'left', or 'outer'.")
        if on_missing not in ("warn", "error", "ignore"):
            raise ValueError(f"Invalid on_missing: {on_missing}. Expected 'warn', 'error', or 'ignore'.")

        self.mode = mode
        self.on_missing = on_missing

    def link(
        self,
        sources: Dict[str, pd.DataFrame],
        link_by: str,
        keep_key_column: bool = False,
    ) -> LinkingResult:
        """Link multiple data sources by key column.

        Args:
            sources: Dictionary mapping source names to DataFrames.
                Each DataFrame must have the key column.
            link_by: Name of the column to use for linking.
            keep_key_column: Whether to keep the key column in output DataFrames.

        Returns:
            LinkingResult with linked DataFrames.

        Raises:
            LinkingError: If linking fails (missing key columns, no matches, etc.).
        """
        if not sources:
            raise LinkingError("No sources provided for linking.")

        if len(sources) < 2:
            # Single source - nothing to link, just return as-is
            source_name, df = list(sources.items())[0]
            if link_by not in df.columns:
                raise LinkingError(
                    f"Key column '{link_by}' not found in source '{source_name}'. "
                    f"Available columns: {df.columns.tolist()[:10]}"
                )

            key_values = set(df[link_by].unique())
            output_df = df if keep_key_column else df.drop(columns=[link_by])

            return LinkingResult(
                linked_data={source_name: output_df},
                key_column=link_by,
                matched_keys=key_values,
                missing_keys={source_name: set()},
                sample_count=len(df),
                report={"mode": "single_source", "original_count": len(df)},
            )

        # Validate key column exists in all sources
        for name, df in sources.items():
            if link_by not in df.columns:
                raise LinkingError(
                    f"Key column '{link_by}' not found in source '{name}'. "
                    f"Available columns: {df.columns.tolist()[:10]}"
                )

        # Get key sets from each source
        key_sets: Dict[str, Set[Any]] = {}
        for name, df in sources.items():
            key_sets[name] = set(df[link_by].unique())

        # Determine matched keys based on mode
        all_keys = set.union(*key_sets.values())
        source_names = list(sources.keys())

        if self.mode == "inner":
            # Keys present in ALL sources
            matched_keys = set.intersection(*key_sets.values())
        elif self.mode == "left":
            # Keys from first source
            matched_keys = key_sets[source_names[0]]
        else:  # outer
            # Keys from ANY source
            matched_keys = all_keys

        # Calculate missing keys for each source
        missing_keys: Dict[str, Set[Any]] = {}
        for name, keys in key_sets.items():
            missing_keys[name] = matched_keys - keys

        # Handle missing keys
        total_missing = sum(len(mk) for mk in missing_keys.values())
        if total_missing > 0:
            if self.on_missing == "error":
                raise LinkingError(
                    f"Missing keys detected in {len([k for k, v in missing_keys.items() if v])} sources. "
                    f"Total missing: {total_missing}. Use on_missing='warn' to continue."
                )
            elif self.on_missing == "warn":
                import warnings
                warnings.warn(
                    f"Missing keys detected during linking. "
                    f"Total missing: {total_missing}. "
                    f"Use on_missing='ignore' to suppress this warning."
                )

        # Filter and align DataFrames
        linked_data: Dict[str, pd.DataFrame] = {}
        report: Dict[str, Any] = {
            "mode": self.mode,
            "original_counts": {name: len(df) for name, df in sources.items()},
            "matched_key_count": len(matched_keys),
            "source_key_counts": {name: len(keys) for name, keys in key_sets.items()},
        }

        for name, df in sources.items():
            # Filter to matched keys
            filtered_df = df[df[link_by].isin(matched_keys)].copy()

            # Sort by key for alignment
            filtered_df = filtered_df.sort_values(by=link_by).reset_index(drop=True)

            # Remove key column if not keeping
            if not keep_key_column:
                filtered_df = filtered_df.drop(columns=[link_by])

            linked_data[name] = filtered_df

        # Verify alignment
        sample_counts = [len(df) for df in linked_data.values()]
        if len(set(sample_counts)) > 1:
            # Handle duplicates - some sources might have multiple rows per key
            report["duplicate_keys_detected"] = True
            report["sample_counts_per_source"] = {
                name: len(df) for name, df in linked_data.items()
            }

        sample_count = sample_counts[0] if sample_counts else 0

        return LinkingResult(
            linked_data=linked_data,
            key_column=link_by,
            matched_keys=matched_keys,
            missing_keys=missing_keys,
            sample_count=sample_count,
            report=report,
        )

    def link_aligned(
        self,
        sources: Dict[str, pd.DataFrame],
        validate: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Link sources that are already aligned by row index.

        This is a simpler linking method for sources that are guaranteed
        to have matching rows (same samples in same order).

        Args:
            sources: Dictionary of aligned DataFrames.
            validate: Whether to validate that all sources have same row count.

        Returns:
            Dictionary of DataFrames (unchanged, just validated).

        Raises:
            LinkingError: If validation fails.
        """
        if validate:
            row_counts = {name: len(df) for name, df in sources.items()}
            unique_counts = set(row_counts.values())

            if len(unique_counts) > 1:
                raise LinkingError(
                    f"Sources have different row counts: {row_counts}. "
                    f"Cannot link aligned sources with mismatched rows."
                )

        return {name: df.copy() for name, df in sources.items()}

    def create_sample_index(
        self,
        sources: Dict[str, pd.DataFrame],
        link_by: str,
    ) -> pd.DataFrame:
        """Create a sample index showing key presence across sources.

        Args:
            sources: Dictionary of source DataFrames.
            link_by: Key column name.

        Returns:
            DataFrame with keys as index and boolean columns per source.
        """
        key_sets: Dict[str, Set[Any]] = {}
        for name, df in sources.items():
            if link_by in df.columns:
                key_sets[name] = set(df[link_by].unique())
            else:
                key_sets[name] = set()

        all_keys = set.union(*key_sets.values()) if key_sets else set()

        index_data = {
            link_by: list(all_keys),
        }
        for name, keys in key_sets.items():
            index_data[f"in_{name}"] = [k in keys for k in all_keys]

        result = pd.DataFrame(index_data)
        result["in_all"] = result[[f"in_{name}" for name in key_sets]].all(axis=1)
        result = result.set_index(link_by)

        return result


def link_xy(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    link_by: str,
    mode: str = "inner",
) -> tuple:
    """Convenience function to link X and Y DataFrames.

    Args:
        x_df: Features DataFrame.
        y_df: Targets DataFrame.
        link_by: Key column name.
        mode: Linking mode.

    Returns:
        Tuple of (X_linked, Y_linked) DataFrames.
    """
    linker = SampleLinker(mode=mode)
    result = linker.link({"X": x_df, "Y": y_df}, link_by=link_by)

    return result.linked_data["X"], result.linked_data["Y"]


def link_xym(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    m_df: pd.DataFrame,
    link_by: str,
    mode: str = "inner",
) -> tuple:
    """Convenience function to link X, Y, and metadata DataFrames.

    Args:
        x_df: Features DataFrame.
        y_df: Targets DataFrame.
        m_df: Metadata DataFrame.
        link_by: Key column name.
        mode: Linking mode.

    Returns:
        Tuple of (X_linked, Y_linked, M_linked) DataFrames.
    """
    linker = SampleLinker(mode=mode)
    result = linker.link({"X": x_df, "Y": y_df, "M": m_df}, link_by=link_by)

    return (
        result.linked_data["X"],
        result.linked_data["Y"],
        result.linked_data["M"],
    )
