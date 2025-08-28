import polars as pl
from nirs4all.dataset.types import (
    Selector, SourceSelector, OutputData, InputData, Layout,
    SampleIndices, PartitionType, ProcessingList, SampleConfig
)
from typing import Dict, List, Union, Any, Optional
import numpy as np


class Indexer:
    """
    Index manager for samples used in ML/DL pipelines.
    Optimizes contiguous access and manages filtering.

    This class is designed to retrieve data during ML pipelines.
    For example, it can be used to get all test samples from branch 2,
    including augmented samples, for specific processings such as
    ["raw", "savgol", "gaussian"].
    """

    def __init__(self):
        self.df = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32), # row index - 1 value per line
            "sample": pl.Series([], dtype=pl.Int32), # index of the sample in the db
            "origin": pl.Series([], dtype=pl.Int32), # For data augmentation. index of the original sample. If sample is original, it's the same as sample index else it's a new one.
            "partition": pl.Series([], dtype=pl.Categorical), # is the sample in "train" set or "test" set
            "group": pl.Series([], dtype=pl.Int8), # group index - a metadata to aggregate samples per types or cluster, etc.
            "branch": pl.Series([], dtype=pl.Int8), # the branch of the pipeline where the sample is used
            "processings": pl.Series([], dtype=pl.Categorical), # the list of processing that has been applied to the sample
            "augmentation": pl.Series([], dtype=pl.Categorical), # the type of augmentation applied to generate the augmented sample
        })

        self.default_values = {
            "partition": "train",
            "group": 0,
            "branch": 0,
            "processings": ["raw"],
        }

    def _apply_filters(self, selector: Selector) -> pl.DataFrame:
        condition = self._build_filter_condition(selector)
        return self.df.filter(condition)

    def _build_filter_condition(self, selector: Selector) -> pl.Expr:
        conditions = []
        for col, value in selector.items():
            if col not in self.df.columns:
                continue
            if isinstance(value, list):
                conditions.append(pl.col(col).is_in(value))
            elif value is None:
                conditions.append(pl.col(col).is_null())
            else:
                conditions.append(pl.col(col) == value)

        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond

        return condition

    def x_indices(self, selector: Selector) -> np.ndarray:
        filtered_df = self._apply_filters(selector) if selector else self.df
        indices = filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)
        return indices

    def y_indices(self, selector: Selector) -> np.ndarray:
        filtered_df = self._apply_filters(selector) if selector else self.df
        result = filtered_df.with_columns(
            pl.when(pl.col("origin").is_null() | pl.col("origin").is_nan())
            .then(pl.col("sample"))
            .otherwise(pl.col("origin")).cast(pl.Int32).alias("y_index")
        )
        return result["y_index"].to_numpy().astype(np.int32)

    def register_samples(self, count: int, partition: str = "train") -> List[int]:
        new_samples_indices = list(range(self.next_sample_index(), self.next_sample_index() + count))
        new_rows_indices = list(range(self.next_row_index(), self.next_row_index() + count))
        self.df = self.df.vstack(pl.DataFrame({
            "row": pl.Series(new_rows_indices, dtype=pl.Int32),
            "sample": pl.Series(new_samples_indices, dtype=pl.Int32),
            "origin": pl.Series(new_samples_indices, dtype=pl.Int32),
            "partition": pl.Series([partition] * count, dtype=pl.Categorical),
            "group": pl.Series([self.default_values["group"]] * count, dtype=pl.Int8),
            "branch": pl.Series([self.default_values["branch"]] * count, dtype=pl.Int8),
            "processings": pl.Series([str(self.default_values["processings"])] * count, dtype=pl.Categorical),
            "augmentation": pl.Series([None] * count, dtype=pl.Categorical),
        }))

        return new_samples_indices

    def _prepare_indices(self, count: int, sample_indices: Optional[SampleIndices], origin_indices: Optional[SampleIndices]) -> tuple[List[int], List[int], List[Optional[int]]]:
        """Prepare sample and origin indices for batch insertion."""
        next_row_idx = self.next_row_index()
        row_ids = list(range(next_row_idx, next_row_idx + count))

        if sample_indices is None:
            next_sample_idx = self.next_sample_index()
            sample_ids = list(range(next_sample_idx, next_sample_idx + count))
            origins = self._prepare_origins_for_new_samples(count, origin_indices)
        else:
            sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")
            origins = self._prepare_origins_for_existing_samples(sample_ids, origin_indices, count)

        return row_ids, sample_ids, origins

    def _prepare_origins_for_new_samples(self, count: int, origin_indices: Optional[SampleIndices]) -> List[Optional[int]]:
        """Prepare origin indices for new samples."""
        if origin_indices is None:
            return [None] * count
        else:
            origins = self._normalize_indices(origin_indices, count, "origin_indices")
            return [int(x) if x is not None else None for x in origins]

    def _prepare_origins_for_existing_samples(self, sample_ids: List[int], origin_indices: Optional[SampleIndices], count: int) -> List[Optional[int]]:
        """Prepare origin indices for existing samples."""
        if origin_indices is None:
            return [int(x) for x in sample_ids]
        else:
            origins = self._normalize_indices(origin_indices, count, "origin_indices")
            return [int(x) for x in origins]

    def _normalize_indices(self, indices: SampleIndices, count: int, param_name: str) -> List[int]:
        """Normalize various index formats to a list of integers."""
        if isinstance(indices, (int, np.integer)):
            return [indices] * count
        elif isinstance(indices, np.ndarray):
            result = indices.tolist()
        else:
            result = list(indices)

        if len(result) != count:
            raise ValueError(f"{param_name} length ({len(result)}) must match count ({count})")
        return result

    def _prepare_column_values(self, count: int, group: Union[int, List[int]], branch: Union[int, List[int]],
                              processings: Union[ProcessingList, List[ProcessingList]],
                              augmentation: Optional[Union[str, List[str]]]) -> tuple:
        """Prepare column values for batch insertion."""
        groups = self._normalize_single_or_list(group, count, "group")
        branches = self._normalize_single_or_list(branch, count, "branch")
        processings_list = self._prepare_processings(processings, count)
        augmentations = self._normalize_single_or_list(augmentation, count, "augmentation", allow_none=True)

        return groups, branches, processings_list, augmentations

    def _normalize_single_or_list(self, value: Union[Any, List[Any]], count: int, param_name: str, allow_none: bool = False) -> List[Any]:
        """Normalize single value or list to a list of specified length."""
        if value is None and allow_none:
            return [None] * count
        elif isinstance(value, (int, np.integer, str)) or value is None:
            return [value] * count
        else:
            result = list(value)
            if len(result) != count:
                raise ValueError(f"{param_name} length ({len(result)}) must match count ({count})")
            return result

    def _prepare_processings(self, processings: Union[ProcessingList, List[ProcessingList], str, List[str]], count: int) -> List[ProcessingList]:
        """Prepare processings list with proper validation."""
        if processings is None:
            return [self.default_values["processings"]] * count
        elif isinstance(processings, str):
            # Single string representation for all samples
            return [processings] * count
        elif isinstance(processings, list) and len(processings) > 0 and isinstance(processings[0], str):
            # Check if it's string representations or actual string processing names
            first_item = processings[0]
            if first_item.startswith("[") and first_item.endswith("]"):
                # List of string representations - each for a different sample
                if len(processings) != count:
                    raise ValueError(f"processings length ({len(processings)}) must match count ({count})")
                return processings
            else:
                # Actual processing names - single list for all samples
                return [processings] * count
        else:
            # List of processing lists
            result = list(processings)
            if len(result) != count:
                raise ValueError(f"processings length ({len(result)}) must match count ({count})")
            return result

    def _create_sample_dataframe(self, row_ids: List[int], sample_ids: List[int], origins: List[Optional[int]],
                                partition: str, groups: List[int], branches: List[int],
                                processings_list: List[ProcessingList], augmentations: List[Optional[str]],
                                kwargs: Dict[str, Any], count: int) -> pl.DataFrame:
        """Create the DataFrame for new samples."""
        # Handle additional kwargs
        additional_cols = {}
        for col, value in kwargs.items():
            if col in self.df.columns:
                if isinstance(value, (list, np.ndarray)):
                    if len(value) != count:
                        raise ValueError(f"{col} length ({len(value)}) must match count ({count})")
                    additional_cols[col] = list(value)
                else:
                    additional_cols[col] = [value] * count

        # Convert processings to strings for Polars compatibility
        processings_strings = [str(p) if isinstance(p, list) else p for p in processings_list]

        new_data = {
            "row": pl.Series(row_ids, dtype=pl.Int32),
            "sample": pl.Series(sample_ids, dtype=pl.Int32),
            "origin": pl.Series(origins, dtype=pl.Int32),
            "partition": pl.Series([partition] * count, dtype=pl.Categorical),
            "group": pl.Series(groups, dtype=pl.Int8),
            "branch": pl.Series(branches, dtype=pl.Int8),
            "processings": pl.Series(processings_strings, dtype=pl.Categorical),
            "augmentation": pl.Series(augmentations, dtype=pl.Categorical),
        }

        # Add additional columns
        for col, values in additional_cols.items():
            expected_dtype = self.df.schema[col]
            new_data[col] = pl.Series(values, dtype=expected_dtype)

        return pl.DataFrame(new_data)

    def add_samples(
        self,
        count: int,
        partition: PartitionType = "train",
        sample_indices: Optional[SampleIndices] = None,
        origin_indices: Optional[SampleIndices] = None,
        group: Union[int, List[int]] = 0,
        branch: Union[int, List[int]] = 0,
        processings: Union[ProcessingList, List[ProcessingList]] = None,
        augmentation: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> List[int]:
        """
        Add multiple samples to the indexer efficiently.

        Args:
            count: Number of samples to add
            partition: Data partition ("train", "test", "val")
            sample_indices: Specific sample IDs to use. If None, auto-increment
            origin_indices: Original sample IDs for augmented samples
            group: Group ID(s) - single value or list of values
            branch: Branch ID(s) - single value or list of values
            processings: Processing steps - single list or list of lists
            augmentation: Augmentation type(s) - single value or list
            **kwargs: Additional column overrides

        Returns:
            List of sample indices that were added
        """
        if count <= 0:
            return []

        # Prepare indices and column values using helper methods
        row_ids, sample_ids, origins = self._prepare_indices(count, sample_indices, origin_indices)
        groups, branches, processings_list, augmentations = self._prepare_column_values(
            count, group, branch, processings, augmentation
        )

        # Create and append the new DataFrame
        new_df = self._create_sample_dataframe(
            row_ids, sample_ids, origins, partition, groups, branches,
            processings_list, augmentations, kwargs, count
        )
        self.df = pl.concat([self.df, new_df], how="vertical")

        return sample_ids

    def _prepare_row_indices(self, n_rows: int) -> tuple[range, range]:
        """Prepare row and sample index ranges."""
        next_row_index = self.next_row_index()
        next_sample_index = self.next_sample_index()
        row_indices = range(next_row_index, next_row_index + n_rows)
        sample_indices = range(next_sample_index, next_sample_index + n_rows)
        return row_indices, sample_indices

    def _handle_sample_column(self, n_rows: int, new_indices: Dict[str, Any], sample_indices: range) -> tuple[pl.Series, List[int]]:
        """Handle sample column creation and determine sample values."""
        if "sample" in new_indices:
            val = new_indices["sample"]
            if isinstance(val, list):
                if len(val) != n_rows:
                    raise ValueError(f"Override list for 'sample' should have {n_rows} elements")
                sample_values = val
            else:
                sample_values = [val] * n_rows
        else:
            sample_values = list(sample_indices)

        return pl.Series(sample_values, dtype=pl.Int32), sample_values

    def _handle_origin_column(self, sample_values: List[int], new_indices: Dict[str, Any]) -> Optional[pl.Series]:
        """Handle origin column creation if not explicitly provided."""
        if "origin" not in new_indices:
            return pl.Series(sample_values, dtype=pl.Int32)
        return None

    def _handle_processings_column(self, val: Any, n_rows: int, expected_dtype: pl.DataType) -> pl.Series:
        """Handle processings column with string conversion."""
        if isinstance(val, list) and len(val) == n_rows and all(isinstance(item, list) for item in val):
            # Per-row processings - each element is a processing list
            processings_strings = [str(p) for p in val]
            return pl.Series(processings_strings, dtype=expected_dtype)
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
            # Single processing list for all rows
            processing_str = str(val)
            return pl.Series([processing_str] * n_rows, dtype=expected_dtype)
        else:
            # Other cases - single processing for all rows
            processing_str = str(val) if isinstance(val, list) else val
            return pl.Series([processing_str] * n_rows, dtype=expected_dtype)

    def _handle_regular_column(self, col: str, val: Any, n_rows: int, expected_dtype: pl.DataType) -> pl.Series:
        """Handle regular column creation with validation."""
        if isinstance(val, list):
            if len(val) != n_rows:
                raise ValueError(f"Override list for '{col}' should have {n_rows} elements")
            return pl.Series(val, dtype=expected_dtype)
        else:
            return pl.Series([val] * n_rows, dtype=expected_dtype)

    def add_rows(
        self,
        n_rows: int,
        new_indices: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Add rows to the indexer with optional column overrides."""
        if n_rows <= 0:
            return []

        new_indices = new_indices or {}
        row_indices, sample_indices = self._prepare_row_indices(n_rows)

        cols: Dict[str, pl.Series] = {}

        # Handle row column
        cols["row"] = pl.Series(row_indices, dtype=pl.Int32)

        # Handle sample column and determine sample values
        sample_series, sample_values = self._handle_sample_column(n_rows, new_indices, sample_indices)
        cols["sample"] = sample_series

        # Handle origin column (if not explicitly provided, set to sample values)
        origin_series = self._handle_origin_column(sample_values, new_indices)
        if origin_series is not None:
            cols["origin"] = origin_series

        # Handle remaining columns
        for col in self.df.columns:
            if col in ["row", "sample"] or (col == "origin" and col in cols):
                continue

            val = new_indices.get(col, self.default_values.get(col, None))
            expected_dtype = self.df.schema[col]

            if col == "processings":
                cols[col] = self._handle_processings_column(val, n_rows, expected_dtype)
            else:
                cols[col] = self._handle_regular_column(col, val, n_rows, expected_dtype)

        # Create and append new DataFrame
        new_df = pl.DataFrame(cols)
        indices = new_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32).tolist()
        self.df = pl.concat([self.df, new_df], how="vertical")
        return indices




    def update_by_filter(self, selector: Selector, updates: Dict[str, Any]) -> None:
        condition = self._build_filter_condition(selector)

        for col, value in updates.items():
            self.df = self.df.with_columns(
                pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

    def next_row_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["row"].max()) + 1

    def next_sample_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["sample"].max()) + 1

    def get_column_values(self, col: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        # Apply filters if provided, otherwise use the full dataframe
        filtered_df = self._apply_filters(filters) if filters else self.df
        return filtered_df.select(pl.col(col)).to_series().to_list()

    def uniques(self, col: str) -> List[Any]:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        return self.df.select(pl.col(col)).unique().to_series().to_list()


    def augment_rows(self, samples: List[int], count: Union[int, List[int]], augmentation_id: str) -> List[int]:
        """
        Create augmented samples based on existing samples.

        Args:
            samples: List of sample IDs to augment
            count: Number of augmentations per sample (int) or list of counts per sample
            augmentation_id: String identifier for the augmentation type

        Returns:
            List of new sample IDs for the augmented samples
        """
        if not samples:
            return []

        # Normalize count to list
        if isinstance(count, int):
            count_list = [count] * len(samples)
        else:
            count_list = list(count)
            if len(count_list) != len(samples):
                raise ValueError("count must be an int or a list with the same length as samples")

        total_augmentations = sum(count_list)
        if total_augmentations == 0:
            return []

        # Get sample data for the samples to augment
        sample_filter = pl.col("sample").is_in(samples)
        filtered_df = self.df.filter(sample_filter).sort("sample")

        if len(filtered_df) != len(samples):
            missing = set(samples) - set(filtered_df["sample"].to_list())
            raise ValueError(f"Samples not found in indexer: {missing}")

        # Prepare data for augmented samples
        origin_indices = []
        partitions = []
        groups = []
        branches = []
        processings_list = []

        for i, (sample_id, sample_count) in enumerate(zip(samples, count_list)):
            if sample_count <= 0:
                continue

            # Get the original sample data
            sample_row = filtered_df.filter(pl.col("sample") == sample_id).row(0, named=True)

            # Repeat data for each augmentation of this sample
            origin_indices.extend([sample_id] * sample_count)
            partitions.extend([sample_row["partition"]] * sample_count)
            groups.extend([sample_row["group"]] * sample_count)
            branches.extend([sample_row["branch"]] * sample_count)
            # Since processings are stored as strings, we need to keep them as strings
            processings_list.extend([sample_row["processings"]] * sample_count)

        # Create augmented samples using add_samples
        # Use first partition as default since partitions should be consistent
        partition = partitions[0] if partitions else "train"

        augmented_ids = self.add_samples(
            count=total_augmentations,
            partition=partition,
            origin_indices=origin_indices,
            group=groups[0] if len(set(groups)) == 1 else groups,
            branch=branches[0] if len(set(branches)) == 1 else branches,
            processings=processings_list,
            augmentation=augmentation_id
        )

        return augmented_ids

    def __repr__(self):
        return str(self.df)