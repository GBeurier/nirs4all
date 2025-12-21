"""
Row selector for dataset configuration.

This module provides flexible row selection for DataFrames, supporting
multiple selection syntaxes including indices, ranges, percentages,
conditions, and random sampling.

Example:
    >>> selector = RowSelector()
    >>> # By index
    >>> rows = selector.select(df, [0, 1, 2])
    >>> # By range
    >>> rows = selector.select(df, "0:100")
    >>> # By percentage
    >>> rows = selector.select(df, "0:80%")
    >>> # By condition
    >>> rows = selector.select(df, {"where": {"column": "quality", "op": ">", "value": 0.5}})
    >>> # Random sample
    >>> rows = selector.select(df, {"sample": 100, "random_state": 42})
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class RowSelectionError(Exception):
    """Raised when row selection fails."""
    pass


# Type alias for row selection specification
RowSpec = Union[
    int,                     # Single index
    str,                     # Range string or percentage
    List[int],               # List of indices
    Dict[str, Any],          # Complex selection (where, sample, etc.)
    slice,                   # Python slice object
    None,                    # Select all rows
]


@dataclass
class RowSelectionResult:
    """Result of a row selection operation.

    Attributes:
        indices: List of selected row indices (from original DataFrame index).
        mask: Boolean mask for the selection.
        data: The selected DataFrame subset.
    """
    indices: List[int]
    mask: pd.Series
    data: pd.DataFrame


class RowSelector:
    """Flexible row selector for DataFrames.

    Supports multiple selection methods:
    - All rows: `None`
    - By index: `[0, 1, 2]` or `0`
    - By range: `"0:100"` (slice syntax as string)
    - By percentage: `"0:80%"` or `"80%:100%"`
    - By condition: `{"where": {"column": "quality", "op": ">", "value": 0.5}}`
    - Random sample: `{"sample": 100, "random_state": 42}`
    - Stratified sample: `{"sample": 100, "stratify": "class", "random_state": 42}`
    - Head/Tail: `{"head": 100}` or `{"tail": 50}`

    Example:
        >>> selector = RowSelector()
        >>> result = selector.select(df, "0:80%")
        >>> print(len(result.data))  # 80% of rows
    """

    # Supported comparison operators
    OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
        "contains": lambda a, b: b in str(a),
        "startswith": lambda a, b: str(a).startswith(b),
        "endswith": lambda a, b: str(a).endswith(b),
        "isna": lambda a, b: pd.isna(a),
        "notna": lambda a, b: pd.notna(a),
        "regex": lambda a, b: bool(re.search(b, str(a))),
    }

    def __init__(self, default_random_state: Optional[int] = None):
        """Initialize the row selector.

        Args:
            default_random_state: Default random state for sampling operations.
        """
        self.default_random_state = default_random_state

    def select(
        self,
        df: pd.DataFrame,
        selection: RowSpec,
    ) -> RowSelectionResult:
        """Select rows from a DataFrame.

        Args:
            df: The DataFrame to select rows from.
            selection: Row selection specification. Can be:
                - None: Select all rows
                - int: Single row index
                - str: Range string ("0:100") or percentage ("0:80%")
                - List[int]: List of row indices
                - Dict: Complex selection (see class docstring)

        Returns:
            RowSelectionResult with indices, mask, and selected data.

        Raises:
            RowSelectionError: If selection is invalid or rows not found.
        """
        if selection is None:
            # Select all rows
            n_rows = len(df)
            return RowSelectionResult(
                indices=list(range(n_rows)),
                mask=pd.Series([True] * n_rows, index=df.index),
                data=df.copy(),
            )

        if isinstance(selection, int):
            return self._select_by_single_index(df, selection)

        if isinstance(selection, str):
            return self._select_by_string(df, selection)

        if isinstance(selection, slice):
            return self._select_by_slice(df, selection)

        if isinstance(selection, (list, tuple)):
            return self._select_by_index_list(df, list(selection))

        if isinstance(selection, dict):
            return self._select_by_dict(df, selection)

        raise RowSelectionError(
            f"Unsupported selection type: {type(selection).__name__}. "
            f"Expected int, str, list, dict, or None."
        )

    def _select_by_single_index(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> RowSelectionResult:
        """Select a single row by index."""
        n_rows = len(df)

        # Handle negative indices
        if index < 0:
            index = n_rows + index

        if index < 0 or index >= n_rows:
            raise RowSelectionError(
                f"Row index {index} out of range. "
                f"DataFrame has {n_rows} rows (0-{n_rows - 1})."
            )

        mask = pd.Series([False] * n_rows, index=df.index)
        mask.iloc[index] = True

        return RowSelectionResult(
            indices=[index],
            mask=mask,
            data=df.iloc[[index]].copy(),
        )

    def _select_by_string(
        self,
        df: pd.DataFrame,
        selection: str,
    ) -> RowSelectionResult:
        """Select by string (range or percentage)."""
        # Check if it's a percentage-based selection
        if "%" in selection:
            return self._select_by_percentage(df, selection)

        # Check if it's a range/slice syntax
        if ":" in selection:
            return self._select_by_range_string(df, selection)

        raise RowSelectionError(
            f"Invalid string selection: '{selection}'. "
            f"Expected range ('0:100') or percentage ('0:80%')."
        )

    def _select_by_percentage(
        self,
        df: pd.DataFrame,
        selection: str,
    ) -> RowSelectionResult:
        """Select rows by percentage range.

        Examples:
            "0:80%" -> First 80% of rows
            "80%:100%" -> Last 20% of rows
            "20%:80%" -> Middle 60% of rows
        """
        n_rows = len(df)

        # Parse percentage string
        parts = selection.split(":")
        if len(parts) != 2:
            raise RowSelectionError(
                f"Invalid percentage format: '{selection}'. "
                f"Expected 'start%:end%' or 'start:end%'."
            )

        start_str, end_str = parts

        # Parse start
        if start_str.strip().endswith("%"):
            start_pct = float(start_str.strip().rstrip("%")) / 100.0
            start_idx = int(n_rows * start_pct)
        elif start_str.strip():
            start_idx = int(start_str.strip())
        else:
            start_idx = 0

        # Parse end
        if end_str.strip().endswith("%"):
            end_pct = float(end_str.strip().rstrip("%")) / 100.0
            end_idx = int(n_rows * end_pct)
        elif end_str.strip():
            end_idx = int(end_str.strip())
        else:
            end_idx = n_rows

        # Validate
        start_idx = max(0, min(start_idx, n_rows))
        end_idx = max(0, min(end_idx, n_rows))

        if start_idx >= end_idx:
            raise RowSelectionError(
                f"Invalid percentage range: '{selection}'. "
                f"Start ({start_idx}) must be less than end ({end_idx})."
            )

        indices = list(range(start_idx, end_idx))
        mask = pd.Series([False] * n_rows, index=df.index)
        mask.iloc[indices] = True

        return RowSelectionResult(
            indices=indices,
            mask=mask,
            data=df.iloc[indices].copy(),
        )

    def _select_by_range_string(
        self,
        df: pd.DataFrame,
        range_str: str,
    ) -> RowSelectionResult:
        """Select rows by range string (slice syntax).

        Examples:
            "0:100" -> First 100 rows
            ":50" -> First 50 rows
            "50:" -> Rows from 50 to end
            "10:100:2" -> Every other row from 10 to 100
        """
        n_rows = len(df)

        # Parse the range string
        parts = range_str.split(":")
        if len(parts) == 2:
            start_str, stop_str = parts
            step_str = None
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
        else:
            raise RowSelectionError(
                f"Invalid range format: '{range_str}'. "
                f"Expected 'start:stop' or 'start:stop:step'."
            )

        # Parse start, stop, step
        try:
            start = int(start_str) if start_str.strip() else None
            stop = int(stop_str) if stop_str.strip() else None
            step = int(step_str) if step_str and step_str.strip() else None
        except ValueError as e:
            raise RowSelectionError(
                f"Invalid range values in '{range_str}': {e}"
            )

        # Create slice and select
        slc = slice(start, stop, step)
        return self._select_by_slice(df, slc)

    def _select_by_slice(
        self,
        df: pd.DataFrame,
        slc: slice,
    ) -> RowSelectionResult:
        """Select rows by slice object."""
        n_rows = len(df)
        indices = list(range(*slc.indices(n_rows)))

        if not indices:
            raise RowSelectionError(
                f"Slice {slc} results in empty selection for DataFrame with {n_rows} rows."
            )

        mask = pd.Series([False] * n_rows, index=df.index)
        mask.iloc[indices] = True

        return RowSelectionResult(
            indices=indices,
            mask=mask,
            data=df.iloc[indices].copy(),
        )

    def _select_by_index_list(
        self,
        df: pd.DataFrame,
        indices: List[int],
    ) -> RowSelectionResult:
        """Select rows by list of indices."""
        n_rows = len(df)
        resolved_indices = []

        for idx in indices:
            # Handle negative indices
            resolved_idx = idx if idx >= 0 else n_rows + idx

            if resolved_idx < 0 or resolved_idx >= n_rows:
                raise RowSelectionError(
                    f"Row index {idx} out of range. "
                    f"DataFrame has {n_rows} rows (0-{n_rows - 1})."
                )
            resolved_indices.append(resolved_idx)

        mask = pd.Series([False] * n_rows, index=df.index)
        mask.iloc[resolved_indices] = True

        return RowSelectionResult(
            indices=resolved_indices,
            mask=mask,
            data=df.iloc[resolved_indices].copy(),
        )

    def _select_by_dict(
        self,
        df: pd.DataFrame,
        selection: Dict[str, Any],
    ) -> RowSelectionResult:
        """Select rows by dictionary specification.

        Supported keys:
        - "where": Condition for filtering
        - "sample": Number of rows to sample
        - "sample_frac": Fraction of rows to sample
        - "stratify": Column for stratified sampling
        - "random_state": Random state for sampling
        - "shuffle": Whether to shuffle before selecting
        - "head": Select first N rows
        - "tail": Select last N rows
        """
        n_rows = len(df)
        mask = pd.Series([True] * n_rows, index=df.index)

        # Apply where condition
        if "where" in selection:
            where_mask = self._apply_where_condition(df, selection["where"])
            mask = mask & where_mask

        # Apply head
        if "head" in selection:
            head_n = selection["head"]
            head_mask = pd.Series([False] * n_rows, index=df.index)
            head_mask.iloc[:head_n] = True
            mask = mask & head_mask

        # Apply tail
        if "tail" in selection:
            tail_n = selection["tail"]
            tail_mask = pd.Series([False] * n_rows, index=df.index)
            tail_mask.iloc[-tail_n:] = True
            mask = mask & tail_mask

        # Get filtered DataFrame
        filtered_df = df[mask]

        # Apply sampling
        if "sample" in selection or "sample_frac" in selection:
            random_state = selection.get("random_state", self.default_random_state)
            stratify_col = selection.get("stratify")

            if "sample" in selection:
                n_sample = selection["sample"]
                if n_sample > len(filtered_df):
                    n_sample = len(filtered_df)
            else:
                frac = selection["sample_frac"]
                n_sample = int(len(filtered_df) * frac)

            if stratify_col:
                # Stratified sampling
                if stratify_col not in filtered_df.columns:
                    raise RowSelectionError(
                        f"Stratify column '{stratify_col}' not found in DataFrame."
                    )
                sampled_df = self._stratified_sample(
                    filtered_df, n_sample, stratify_col, random_state
                )
            else:
                # Simple random sampling
                sampled_df = filtered_df.sample(n=n_sample, random_state=random_state)

            filtered_df = sampled_df

        # Apply shuffle (if not sampling, which already shuffles)
        elif selection.get("shuffle", False):
            random_state = selection.get("random_state", self.default_random_state)
            filtered_df = filtered_df.sample(frac=1.0, random_state=random_state)

        # Build final result
        final_indices = [df.index.get_loc(idx) for idx in filtered_df.index]
        final_mask = pd.Series([False] * n_rows, index=df.index)
        final_mask.loc[filtered_df.index] = True

        return RowSelectionResult(
            indices=final_indices,
            mask=final_mask,
            data=filtered_df.copy(),
        )

    def _apply_where_condition(
        self,
        df: pd.DataFrame,
        condition: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.Series:
        """Apply a where condition to create a boolean mask.

        Condition format:
            {"column": "col_name", "op": ">", "value": 0.5}

        Multiple conditions (AND):
            [
                {"column": "col1", "op": ">", "value": 0.5},
                {"column": "col2", "op": "==", "value": "A"}
            ]

        Multiple conditions (OR):
            {"or": [
                {"column": "col1", "op": ">", "value": 0.5},
                {"column": "col2", "op": "==", "value": "A"}
            ]}
        """
        n_rows = len(df)

        # Handle OR condition
        if isinstance(condition, dict) and "or" in condition:
            mask = pd.Series([False] * n_rows, index=df.index)
            for sub_condition in condition["or"]:
                sub_mask = self._apply_single_condition(df, sub_condition)
                mask = mask | sub_mask
            return mask

        # Handle AND condition (list)
        if isinstance(condition, list):
            mask = pd.Series([True] * n_rows, index=df.index)
            for sub_condition in condition:
                sub_mask = self._apply_single_condition(df, sub_condition)
                mask = mask & sub_mask
            return mask

        # Handle single condition
        return self._apply_single_condition(df, condition)

    def _apply_single_condition(
        self,
        df: pd.DataFrame,
        condition: Dict[str, Any],
    ) -> pd.Series:
        """Apply a single condition."""
        column = condition.get("column")
        op = condition.get("op", "==")
        value = condition.get("value")

        if column is None:
            raise RowSelectionError(
                f"Condition missing 'column' key: {condition}"
            )

        if column not in df.columns:
            raise RowSelectionError(
                f"Column '{column}' not found in DataFrame. "
                f"Available: {df.columns.tolist()[:10]}"
            )

        if op not in self.OPERATORS:
            raise RowSelectionError(
                f"Unknown operator '{op}'. "
                f"Available: {list(self.OPERATORS.keys())}"
            )

        # Apply the operator
        op_func = self.OPERATORS[op]

        if op in ("isna", "notna"):
            # Unary operators
            mask = df[column].apply(lambda x: op_func(x, None))
        else:
            mask = df[column].apply(lambda x: op_func(x, value))

        return mask

    def _stratified_sample(
        self,
        df: pd.DataFrame,
        n_sample: int,
        stratify_col: str,
        random_state: Optional[int],
    ) -> pd.DataFrame:
        """Perform stratified sampling.

        Samples proportionally from each stratum (unique value of stratify_col).
        """
        np.random.seed(random_state)

        groups = df.groupby(stratify_col)
        n_groups = len(groups)
        samples_per_group = n_sample // n_groups
        remainder = n_sample % n_groups

        sampled_dfs = []
        group_names = list(groups.groups.keys())

        for i, (name, group) in enumerate(groups):
            # Distribute remainder to first groups
            n_group_sample = samples_per_group + (1 if i < remainder else 0)
            n_group_sample = min(n_group_sample, len(group))

            if n_group_sample > 0:
                sampled_dfs.append(group.sample(n=n_group_sample, random_state=random_state))

        if sampled_dfs:
            return pd.concat(sampled_dfs)
        return df.iloc[:0]  # Empty DataFrame with same structure
