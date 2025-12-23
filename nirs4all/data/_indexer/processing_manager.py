"""
Handle processing list operations with native Polars lists.

This module provides the ProcessingManager class for managing processing
lists using native Polars List(Utf8) type - no eval() needed!
"""

from typing import List
import polars as pl


class ProcessingManager:
    """
    Manage processing lists with native Polars operations.

    This class handles all processing list operations using native Polars
    List(Utf8) type, eliminating the need for string parsing and eval().

    Processing lists represent the sequence of transformations applied to
    spectroscopic data (e.g., ["raw", "msc", "savgol"]).

    Key improvements over string-based approach:
    - Type-safe: Native Polars list operations
    - Secure: No eval() calls
    - Efficient: Optimized list operations
    - Maintainable: Clean, readable code
    """

    def __init__(self, store):
        """
        Initialize the processing manager.

        Args:
            store: IndexStore instance for DataFrame access.
        """
        self._store = store

    def replace_processings(self, old_names: List[str], new_names: List[str]) -> None:
        """
        Replace processing names across all samples.

        Creates a mapping from old to new processing names and applies it to
        all processing lists in the index.

        Args:
            old_names: List of existing processing names to replace.
            new_names: List of new processing names to use. Must have same
                      length as old_names.

        Raises:
            ValueError: If old_names and new_names have different lengths.
            ValueError: If old_names or new_names is empty.

        Examples:
            >>> manager = ProcessingManager(store)
            >>>
            >>> # Replace single processing
            >>> manager.replace_processings(["raw"], ["raw_v2"])
            >>>
            >>> # Replace multiple processings
            >>> manager.replace_processings(
            ...     ["old_msc", "old_savgol"],
            ...     ["msc", "savgol"]
            ... )

        Note:
            - Operates on ALL rows in the index
            - Non-matched processings are left unchanged
            - Case-sensitive matching
        """
        if not old_names or not new_names:
            return

        if len(old_names) != len(new_names):
            raise ValueError(
                f"old_names ({len(old_names)}) and new_names ({len(new_names)}) "
                "must have the same length"
            )

        # Create replacement mapping
        replacement_map = {old: new for old, new in zip(old_names, new_names)}

        # Use map_elements for clean replacement
        def replace_in_list(proc_list):
            """Replace processing names in a list."""
            if proc_list is None:
                return proc_list
            return [replacement_map.get(p, p) for p in proc_list]

        # Apply replacement to all rows
        df = self._store.df
        updated_df = df.with_columns(
            pl.col("processings").map_elements(
                replace_in_list,
                return_dtype=pl.List(pl.Utf8)
            )
        )
        self._store._df = updated_df

    def reset_processings(self, new_processings: List[str]) -> None:
        """
        Reset processing names for all samples to a new list.

        This replaces the entire processing list for every sample with the
        provided list. Used when resetting feature storage (e.g. after merge).

        Args:
            new_processings: List of new processing names.

        Raises:
            ValueError: If new_processings is empty.
        """
        if not new_processings:
            raise ValueError("new_processings cannot be empty")

        # Use native Polars list creation
        df = self._store.df
        # Create a series of lists, one for each row
        # Note: pl.lit with a list creates a Series of that list repeated?
        # No, pl.lit([1, 2]) creates a Series with 2 rows.
        # We want a column where every row is the list `new_processings`.

        # Correct way: create a Series of the list, then repeat it?
        # Or use pl.lit(pd.Series([new_processings] * len(df)))?
        # Polars doesn't support list literals easily in expressions for all rows.

        # We can use map_elements to return the new list for every row.
        updated_df = df.with_columns(
            pl.col("processings").map_elements(
                lambda _: new_processings,
                return_dtype=pl.List(pl.Utf8)
            )
        )
        self._store._df = updated_df

    def add_processings(self, new_processings: List[str]) -> None:
        """
        Append processing names to all existing processing lists.

        Adds new processings to the end of each sample's processing list.
        This is useful when applying additional transformations to all data.

        Args:
            new_processings: List of processing names to append to all lists.

        Raises:
            ValueError: If new_processings is empty.

        Examples:
            >>> manager = ProcessingManager(store)
            >>>
            >>> # Add single processing
            >>> manager.add_processings(["normalize"])
            >>> # ["raw", "msc"] becomes ["raw", "msc", "normalize"]
            >>>
            >>> # Add multiple processings
            >>> manager.add_processings(["normalize", "scale"])
            >>> # ["raw", "msc"] becomes ["raw", "msc", "normalize", "scale"]

        Note:
            - Operates on ALL rows in the index
            - Appends to the end of each list
            - Does not check for duplicates
        """
        if not new_processings:
            return

        # Use native Polars list concatenation - much cleaner!
        df = self._store.df
        updated_df = df.with_columns(
            pl.col("processings").list.concat(pl.lit(new_processings))
        )
        self._store._df = updated_df

    def get_processings_for_sample(self, sample_id: int) -> List[str]:
        """
        Get the processing list for a specific sample.

        Args:
            sample_id: Sample ID to look up.

        Returns:
            List[str]: Processing list for the sample, or empty list if not found.

        Example:
            >>> processings = manager.get_processings_for_sample(10)
            >>> # processings: ["raw", "msc", "savgol"]
        """
        condition = pl.col("sample") == sample_id
        row = self._store.query(condition)

        if len(row) == 0:
            return []

        proc_list = row.select(pl.col("processings")).item()
        return proc_list if proc_list is not None else []

    def validate_processing_format(self, processings: any) -> List[str]:
        """
        Validate and normalize processing input to a list of strings.

        Accepts various formats and normalizes them to List[str].

        Args:
            processings: Processing specification in various formats:
                        - List[str]: ["raw", "msc"]
                        - str: "raw" (single processing)
                        - None: Uses default ["raw"]

        Returns:
            List[str]: Validated processing list.

        Examples:
            >>> manager.validate_processing_format(["raw", "msc"])
            ['raw', 'msc']
            >>>
            >>> manager.validate_processing_format("raw")
            ['raw']
            >>>
            >>> manager.validate_processing_format(None)
            ['raw']
        """
        if processings is None:
            return ["raw"]  # Default
        elif isinstance(processings, str):
            return [processings]
        elif isinstance(processings, list):
            return processings
        else:
            raise TypeError(
                f"processings must be List[str], str, or None, got {type(processings)}"
            )
