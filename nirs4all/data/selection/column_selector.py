"""
Column selector for dataset configuration.

This module provides flexible column selection for DataFrames, supporting
multiple selection syntaxes including indices, names, ranges, regex patterns,
and exclusion.

Example:
    >>> selector = ColumnSelector()
    >>> # By name
    >>> cols = selector.select(df, ["col1", "col2"])
    >>> # By index
    >>> cols = selector.select(df, [0, 1, 2])
    >>> # By range (slice syntax)
    >>> cols = selector.select(df, "2:-1")
    >>> # By regex pattern
    >>> cols = selector.select(df, {"regex": "^feature_.*"})
    >>> # By exclusion
    >>> cols = selector.select(df, {"exclude": ["id", "date"]})
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class ColumnSelectionError(Exception):
    """Raised when column selection fails."""
    pass


# Type alias for column selection specification
ColumnSpec = Union[
    int,                     # Single index
    str,                     # Single name or range string
    List[int],               # List of indices
    List[str],               # List of names
    Dict[str, Any],          # Complex selection (regex, exclude, etc.)
    slice,                   # Python slice object
    None,                    # Select all columns
]


@dataclass
class SelectionResult:
    """Result of a column selection operation.

    Attributes:
        indices: List of selected column indices (0-based).
        names: List of selected column names.
        data: The selected DataFrame subset.
    """
    indices: List[int]
    names: List[str]
    data: pd.DataFrame


class ColumnSelector:
    """Flexible column selector for DataFrames.

    Supports multiple selection methods:
    - By name: `["col1", "col2"]` or `"col_name"`
    - By index: `[0, 1, 2]` or `0`
    - By range: `"2:-1"` (slice syntax as string)
    - By regex pattern: `{"regex": "^feature_.*"}`
    - By exclusion: `{"exclude": ["id", "date"]}`
    - Combined: `{"include": [0, 1], "exclude": ["id"]}`

    Example:
        >>> selector = ColumnSelector()
        >>> result = selector.select(df, "2:-1")
        >>> print(result.names)  # Column names in range
        >>> print(result.data)   # Selected columns as DataFrame
    """

    def __init__(self, case_sensitive: bool = True):
        """Initialize the column selector.

        Args:
            case_sensitive: Whether column name matching is case-sensitive.
        """
        self.case_sensitive = case_sensitive

    def select(
        self,
        df: pd.DataFrame,
        selection: ColumnSpec,
    ) -> SelectionResult:
        """Select columns from a DataFrame.

        Args:
            df: The DataFrame to select columns from.
            selection: Column selection specification. Can be:
                - None: Select all columns
                - int: Single column index
                - str: Single column name or range string ("2:-1")
                - List[int]: List of column indices
                - List[str]: List of column names
                - Dict: Complex selection (see class docstring)

        Returns:
            SelectionResult with indices, names, and selected data.

        Raises:
            ColumnSelectionError: If selection is invalid or columns not found.
        """
        if selection is None:
            # Select all columns
            return SelectionResult(
                indices=list(range(len(df.columns))),
                names=df.columns.tolist(),
                data=df.copy(),
            )

        if isinstance(selection, int):
            return self._select_by_single_index(df, selection)

        if isinstance(selection, str):
            return self._select_by_string(df, selection)

        if isinstance(selection, slice):
            return self._select_by_slice(df, selection)

        if isinstance(selection, (list, tuple)):
            return self._select_by_list(df, selection)

        if isinstance(selection, dict):
            return self._select_by_dict(df, selection)

        raise ColumnSelectionError(
            f"Unsupported selection type: {type(selection).__name__}. "
            f"Expected int, str, list, dict, or None."
        )

    def _select_by_single_index(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> SelectionResult:
        """Select a single column by index."""
        n_cols = len(df.columns)

        # Handle negative indices
        if index < 0:
            index = n_cols + index

        if index < 0 or index >= n_cols:
            raise ColumnSelectionError(
                f"Column index {index} out of range. "
                f"DataFrame has {n_cols} columns (0-{n_cols - 1})."
            )

        col_name = df.columns[index]
        return SelectionResult(
            indices=[index],
            names=[col_name],
            data=df.iloc[:, [index]].copy(),
        )

    def _select_by_string(
        self,
        df: pd.DataFrame,
        selection: str,
    ) -> SelectionResult:
        """Select by string (column name or range)."""
        # Check if it's a range/slice syntax
        if ":" in selection:
            return self._select_by_range_string(df, selection)

        # Otherwise, treat as column name
        return self._select_by_name(df, selection)

    def _select_by_name(
        self,
        df: pd.DataFrame,
        name: str,
    ) -> SelectionResult:
        """Select a single column by name."""
        col_names = df.columns.tolist()

        if self.case_sensitive:
            if name in col_names:
                idx = col_names.index(name)
                return SelectionResult(
                    indices=[idx],
                    names=[name],
                    data=df[[name]].copy(),
                )
        else:
            # Case-insensitive search
            lower_names = [n.lower() for n in col_names]
            if name.lower() in lower_names:
                idx = lower_names.index(name.lower())
                actual_name = col_names[idx]
                return SelectionResult(
                    indices=[idx],
                    names=[actual_name],
                    data=df[[actual_name]].copy(),
                )

        raise ColumnSelectionError(
            f"Column '{name}' not found. Available columns: {col_names[:10]}"
            + ("..." if len(col_names) > 10 else "")
        )

    def _select_by_range_string(
        self,
        df: pd.DataFrame,
        range_str: str,
    ) -> SelectionResult:
        """Select columns by range string (slice syntax).

        Examples:
            "2:-1" -> columns from index 2 to second-to-last
            ":5" -> first 5 columns
            "3:" -> columns from index 3 to end
            "1:10:2" -> columns 1, 3, 5, 7, 9
        """
        n_cols = len(df.columns)

        # Parse the range string
        parts = range_str.split(":")
        if len(parts) == 2:
            start_str, stop_str = parts
            step_str = None
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
        else:
            raise ColumnSelectionError(
                f"Invalid range format: '{range_str}'. "
                f"Expected format: 'start:stop' or 'start:stop:step'."
            )

        # Parse start, stop, step
        try:
            start = int(start_str) if start_str.strip() else None
            stop = int(stop_str) if stop_str.strip() else None
            step = int(step_str) if step_str and step_str.strip() else None
        except ValueError as e:
            raise ColumnSelectionError(
                f"Invalid range values in '{range_str}': {e}"
            )

        # Create slice and select
        slc = slice(start, stop, step)
        return self._select_by_slice(df, slc)

    def _select_by_slice(
        self,
        df: pd.DataFrame,
        slc: slice,
    ) -> SelectionResult:
        """Select columns by slice object."""
        n_cols = len(df.columns)
        indices = list(range(*slc.indices(n_cols)))

        if not indices:
            raise ColumnSelectionError(
                f"Slice {slc} results in empty selection for DataFrame with {n_cols} columns."
            )

        names = [df.columns[i] for i in indices]
        return SelectionResult(
            indices=indices,
            names=names,
            data=df.iloc[:, indices].copy(),
        )

    def _select_by_list(
        self,
        df: pd.DataFrame,
        selection: Sequence,
    ) -> SelectionResult:
        """Select columns by list of indices or names."""
        if not selection:
            raise ColumnSelectionError("Empty selection list provided.")

        # Determine if list contains indices or names
        first_item = selection[0]

        if isinstance(first_item, (int, np.integer)):
            return self._select_by_index_list(df, list(selection))
        elif isinstance(first_item, str):
            return self._select_by_name_list(df, list(selection))
        else:
            raise ColumnSelectionError(
                f"Unsupported list item type: {type(first_item).__name__}. "
                f"Expected int or str."
            )

    def _select_by_index_list(
        self,
        df: pd.DataFrame,
        indices: List[int],
    ) -> SelectionResult:
        """Select columns by list of indices."""
        n_cols = len(df.columns)
        resolved_indices = []

        for idx in indices:
            # Handle negative indices
            resolved_idx = idx if idx >= 0 else n_cols + idx

            if resolved_idx < 0 or resolved_idx >= n_cols:
                raise ColumnSelectionError(
                    f"Column index {idx} out of range. "
                    f"DataFrame has {n_cols} columns (0-{n_cols - 1})."
                )
            resolved_indices.append(resolved_idx)

        names = [df.columns[i] for i in resolved_indices]
        return SelectionResult(
            indices=resolved_indices,
            names=names,
            data=df.iloc[:, resolved_indices].copy(),
        )

    def _select_by_name_list(
        self,
        df: pd.DataFrame,
        names: List[str],
    ) -> SelectionResult:
        """Select columns by list of names."""
        col_names = df.columns.tolist()
        resolved_indices = []
        resolved_names = []

        for name in names:
            if self.case_sensitive:
                if name in col_names:
                    idx = col_names.index(name)
                    resolved_indices.append(idx)
                    resolved_names.append(name)
                else:
                    raise ColumnSelectionError(
                        f"Column '{name}' not found. "
                        f"Available: {col_names[:10]}" +
                        ("..." if len(col_names) > 10 else "")
                    )
            else:
                # Case-insensitive search
                lower_names = [n.lower() for n in col_names]
                if name.lower() in lower_names:
                    idx = lower_names.index(name.lower())
                    resolved_indices.append(idx)
                    resolved_names.append(col_names[idx])
                else:
                    raise ColumnSelectionError(
                        f"Column '{name}' not found (case-insensitive). "
                        f"Available: {col_names[:10]}" +
                        ("..." if len(col_names) > 10 else "")
                    )

        return SelectionResult(
            indices=resolved_indices,
            names=resolved_names,
            data=df[resolved_names].copy(),
        )

    def _select_by_dict(
        self,
        df: pd.DataFrame,
        selection: Dict[str, Any],
    ) -> SelectionResult:
        """Select columns by dictionary specification.

        Supported keys:
        - "regex": Regular expression pattern to match column names
        - "exclude": Columns to exclude (names or indices)
        - "include": Columns to include (names or indices)
        - "startswith": Prefix to match
        - "endswith": Suffix to match
        - "contains": Substring to match
        - "dtype": Select columns by dtype
        """
        col_names = df.columns.tolist()
        n_cols = len(col_names)

        # Start with all columns or specified include set
        if "include" in selection:
            include_result = self.select(df, selection["include"])
            selected_indices = set(include_result.indices)
        else:
            selected_indices = set(range(n_cols))

        # Apply regex filter
        if "regex" in selection:
            pattern = selection["regex"]
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                raise ColumnSelectionError(f"Invalid regex pattern '{pattern}': {e}")

            matching = {i for i, name in enumerate(col_names) if regex.search(name)}
            selected_indices &= matching

        # Apply startswith filter
        if "startswith" in selection:
            prefix = selection["startswith"]
            if self.case_sensitive:
                matching = {i for i, name in enumerate(col_names) if name.startswith(prefix)}
            else:
                matching = {i for i, name in enumerate(col_names) if name.lower().startswith(prefix.lower())}
            selected_indices &= matching

        # Apply endswith filter
        if "endswith" in selection:
            suffix = selection["endswith"]
            if self.case_sensitive:
                matching = {i for i, name in enumerate(col_names) if name.endswith(suffix)}
            else:
                matching = {i for i, name in enumerate(col_names) if name.lower().endswith(suffix.lower())}
            selected_indices &= matching

        # Apply contains filter
        if "contains" in selection:
            substring = selection["contains"]
            if self.case_sensitive:
                matching = {i for i, name in enumerate(col_names) if substring in name}
            else:
                matching = {i for i, name in enumerate(col_names) if substring.lower() in name.lower()}
            selected_indices &= matching

        # Apply dtype filter
        if "dtype" in selection:
            dtype_spec = selection["dtype"]
            matching = set()
            for i, col in enumerate(col_names):
                col_dtype = df[col].dtype
                if self._dtype_matches(col_dtype, dtype_spec):
                    matching.add(i)
            selected_indices &= matching

        # Apply exclude filter
        if "exclude" in selection:
            exclude_spec = selection["exclude"]
            exclude_result = self.select(df, exclude_spec)
            selected_indices -= set(exclude_result.indices)

        if not selected_indices:
            raise ColumnSelectionError(
                f"Selection {selection} resulted in no columns. "
                f"Available columns: {col_names[:10]}" +
                ("..." if len(col_names) > 10 else "")
            )

        # Sort indices to maintain column order
        sorted_indices = sorted(selected_indices)
        names = [col_names[i] for i in sorted_indices]

        return SelectionResult(
            indices=sorted_indices,
            names=names,
            data=df.iloc[:, sorted_indices].copy(),
        )

    def _dtype_matches(self, actual_dtype, dtype_spec: str) -> bool:
        """Check if a dtype matches a specification."""
        dtype_str = str(actual_dtype).lower()
        spec_lower = dtype_spec.lower()

        # Handle common dtype categories
        dtype_categories = {
            "numeric": ["int", "float", "complex"],
            "integer": ["int"],
            "float": ["float"],
            "string": ["object", "string", "str"],
            "categorical": ["category"],
            "datetime": ["datetime"],
            "bool": ["bool"],
        }

        if spec_lower in dtype_categories:
            return any(cat in dtype_str for cat in dtype_categories[spec_lower])

        # Direct match
        return spec_lower in dtype_str

    def parse_selection(
        self,
        selection: Any,
        available_columns: List[str],
    ) -> List[int]:
        """Parse a selection specification and return column indices.

        This is a convenience method for when you don't have a DataFrame
        but want to validate and resolve a selection.

        Args:
            selection: Column selection specification.
            available_columns: List of available column names.

        Returns:
            List of column indices.

        Raises:
            ColumnSelectionError: If selection is invalid.
        """
        # Create a dummy DataFrame with just the column names
        dummy_df = pd.DataFrame(columns=available_columns)
        result = self.select(dummy_df, selection)
        return result.indices
