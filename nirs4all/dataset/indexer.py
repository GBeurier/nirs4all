from typing import Dict, List, Union, Any, Optional, overload
import numpy as np
import polars as pl

from nirs4all.dataset.helpers import Selector, SampleIndices, PartitionType, ProcessingList, IndexDict


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
        # Enable StringCache for consistent categorical encodings
        pl.enable_string_cache()

        self.df = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32),  # row index - 1 value per line
            "sample": pl.Series([], dtype=pl.Int32),  # index of the sample in the db
            "origin": pl.Series([], dtype=pl.Int32),  # For data augmentation. index of the original sample. If sample is original, it's the same as sample index else it's a new one.
            "partition": pl.Series([], dtype=pl.Categorical),  # is the sample in "train" set or "test" set
            "group": pl.Series([], dtype=pl.Int8),  # group index - a metadata to aggregate samples per types or cluster, etc.
            "branch": pl.Series([], dtype=pl.Int8),  # the branch of the pipeline where the sample is used
            "processings": pl.Series([], dtype=pl.Utf8),  # the list of processing that has been applied to the sample (stored as string)
            "augmentation": pl.Series([], dtype=pl.Categorical),  # the type of augmentation applied to generate the augmented sample
        })

        self.default_values = {
            "partition": "train",
            # "group": 0,
            # "branch": 0,
            "processings": ["raw"],
        }

    def _apply_filters(self, selector: Selector) -> pl.DataFrame:
        condition = self._build_filter_condition(selector)
        return self.df.filter(condition)

    def _build_filter_condition(self, selector: Selector) -> pl.Expr:
        conditions = []
        for col, value in selector.items():
            if col not in self.df.columns or col == "processings":
                continue
            if isinstance(value, list):
                conditions.append(pl.col(col).is_in(value))
            elif value is None:
                conditions.append(pl.col(col).is_null())
            else:
                conditions.append(pl.col(col) == value)

        # Handle empty conditions (empty selector)
        if not conditions:
            return pl.lit(True)

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
            pl.when(pl.col("origin").is_null())
            .then(pl.col("sample"))
            .otherwise(pl.col("origin")).cast(pl.Int32).alias("y_index")
        )
        return result["y_index"].to_numpy().astype(np.int32)

    def replace_processings(self, source_processings: List[str], new_processings: List[str]) -> None:
        """
        Replace processing names for a specific source.

        Args:
            source_processings: List of existing processing names to replace
            new_processings: List of new processing names to set
        """
        if not source_processings or not new_processings:
            return

        def replace_proc(proc_str: str) -> str:
            try:
                proc_list = eval(proc_str)
                if not isinstance(proc_list, list):
                    return proc_str

                # Create a mapping from old to new processing names
                replacement_map = {old: new for old, new in zip(source_processings, new_processings)}

                # Replace each processing name if it exists in the map
                updated = [replacement_map.get(proc, proc) for proc in proc_list]
                return str(updated)
            except Exception:
                return proc_str

        # Apply replacement to all rows (no filtering needed for replacement)
        self.df = self.df.with_columns(
            pl.col("processings").map_elements(replace_proc, return_dtype=pl.Utf8)
        )

    def add_processings(self, new_processings: List[str]) -> None:
        """
        Add new processing names to all existing processing lists.

        Args:
            new_processings: List of new processing names to add to existing lists
        """
        if not new_processings:
            return

        def append_processings(proc_str: str) -> str:
            try:
                proc_list = eval(proc_str)
                if not isinstance(proc_list, list):
                    proc_list = [proc_str]
                # Add new processings to the existing list
                updated_list = proc_list + new_processings
                return str(updated_list)
            except Exception:
                # If parsing fails, create new list with new processings
                return str(new_processings)

        # Apply to all rows
        self.df = self.df.with_columns(
            pl.col("processings").map_elements(append_processings, return_dtype=pl.Utf8)
        )

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

    def _prepare_processings(self, processings: Union[ProcessingList, List[ProcessingList], str, List[str], None], count: int) -> List[str]:
        """Prepare processings list with proper validation and string conversion."""
        if processings is None:
            return [str(self.default_values["processings"])] * count
        elif isinstance(processings, str):
            # Single string representation for all samples
            return [processings] * count
        elif isinstance(processings, list) and len(processings) > 0:
            if isinstance(processings[0], str) and processings[0].startswith("[") and processings[0].endswith("]"):
                # List of string representations - each for a different sample
                if len(processings) != count:
                    raise ValueError(f"processings length ({len(processings)}) must match count ({count})")
                return processings
            elif isinstance(processings[0], str):
                # Actual processing names - single list for all samples
                return [str(processings)] * count
            elif isinstance(processings[0], list):
                # List of processing lists
                if len(processings) != count:
                    raise ValueError(f"processings length ({len(processings)}) must match count ({count})")
                return [str(p) for p in processings]
            else:
                # Other cases - convert to string
                if len(processings) == count:
                    return [str(p) for p in processings]
                else:
                    return [str(processings)] * count
        else:
            # Other cases - single processing for all samples
            return [str(processings)] * count

    def _convert_indexdict_to_params(self, index_dict: IndexDict, count: int) -> Dict[str, Any]:
        """Convert IndexDict to method parameters, similar to _apply_filters pattern."""
        params = {}

        # Handle special mappings
        if "sample" in index_dict:
            params["sample_indices"] = index_dict["sample"]
        if "origin" in index_dict:
            params["origin_indices"] = index_dict["origin"]

        # Handle direct mappings
        direct_mappings = ["partition", "group", "branch", "processings", "augmentation"]
        for key in direct_mappings:
            if key in index_dict:
                params[key] = index_dict[key]

        # Handle any other columns as overrides
        for key, value in index_dict.items():
            if key not in ["sample", "origin"] + direct_mappings:
                params[key] = value

        return params

    def _append(self,
                count: int,
                *,
                partition: PartitionType = "train",
                sample_indices: Optional[SampleIndices] = None,
                origin_indices: Optional[SampleIndices] = None,
                group: Optional[Union[int, List[int]]] = None,
                branch: Optional[Union[int, List[int]]] = None,
                processings: Union[ProcessingList, List[ProcessingList], str, List[str], None] = None,
                augmentation: Optional[Union[str, List[str]]] = None,
                **overrides) -> List[int]:
        """
        Core method to append samples to the indexer.

        Args:
            count: Number of samples to add
            partition: Data partition ("train", "test", "val")
            sample_indices: Specific sample IDs to use. If None, auto-increment
            origin_indices: Original sample IDs for augmented samples
            group: Group ID(s) - single value or list of values
            branch: Branch ID(s) - single value or list of values
            processings: Processing steps - single list or list of lists or string representations
            augmentation: Augmentation type(s) - single value or list
            **overrides: Additional column overrides

        Returns:
            List of sample indices that were added
        """
        if count <= 0:
            return []

        # Prepare row indices
        next_row_idx = self.next_row_index()
        row_ids = list(range(next_row_idx, next_row_idx + count))

        # Prepare sample indices and origins
        if sample_indices is None:
            next_sample_idx = self.next_sample_index()
            sample_ids = list(range(next_sample_idx, next_sample_idx + count))
            if origin_indices is None:
                origins = [None] * count
            else:
                origins_normalized = self._normalize_indices(origin_indices, count, "origin_indices")
                origins = [int(x) if x is not None else None for x in origins_normalized]
        else:
            sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")
            if origin_indices is None:
                origins = [int(x) for x in sample_ids]
            else:
                origins_normalized = self._normalize_indices(origin_indices, count, "origin_indices")
                origins = [int(x) for x in origins_normalized]

        # Prepare column values
        groups = self._normalize_single_or_list(group, count, "group")
        branches = self._normalize_single_or_list(branch, count, "branch")
        processings_list = self._prepare_processings(processings, count)
        augmentations = self._normalize_single_or_list(augmentation, count, "augmentation", allow_none=True)

        # Handle additional overrides
        additional_cols = {}
        for col, value in overrides.items():
            if col in self.df.columns and col not in ["row", "sample", "origin", "partition", "group", "branch", "processings", "augmentation"]:
                if isinstance(value, (list, np.ndarray)):
                    if len(value) != count:
                        raise ValueError(f"{col} length ({len(value)}) must match count ({count})")
                    additional_cols[col] = list(value)
                else:
                    additional_cols[col] = [value] * count

        # Create new DataFrame
        new_data = {
            "row": pl.Series(row_ids, dtype=pl.Int32),
            "sample": pl.Series(sample_ids, dtype=pl.Int32),
            "origin": pl.Series(origins, dtype=pl.Int32),
            "partition": pl.Series([partition] * count, dtype=pl.Categorical),
            "group": pl.Series(groups, dtype=pl.Int8),
            "branch": pl.Series(branches, dtype=pl.Int8),
            "processings": pl.Series(processings_list, dtype=pl.Utf8),
            "augmentation": pl.Series(augmentations, dtype=pl.Categorical),
        }

        # Add additional columns with proper casting
        for col, values in additional_cols.items():
            expected_dtype = self.df.schema[col]
            new_data[col] = pl.Series(values, dtype=expected_dtype)

        new_df = pl.DataFrame(new_data)
        self.df = pl.concat([self.df, new_df], how="vertical")

        return sample_ids

    def add_samples(
        self,
        count: int,
        partition: PartitionType = "train",
        sample_indices: Optional[SampleIndices] = None,
        origin_indices: Optional[SampleIndices] = None,
        group: Optional[Union[int, List[int]]] = None,
        branch: Optional[Union[int, List[int]]] = None,
        processings: Union[ProcessingList, List[ProcessingList], None] = None,
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
        return self._append(
            count,
            partition=partition,
            sample_indices=sample_indices,
            origin_indices=origin_indices,
            group=group,
            branch=branch,
            processings=processings,
            augmentation=augmentation,
            **kwargs
        )

    def add_samples_dict(
        self,
        count: int,
        indices: Optional[IndexDict] = None,
        **kwargs
    ) -> List[int]:
        """
        Add multiple samples using dictionary-based parameter specification.

        This method provides a cleaner API for specifying sample parameters
        using a dictionary, similar to the filtering API pattern.

        Args:
            count: Number of samples to add
            indices: Dictionary containing column specifications {
                "partition": "train|test|val",
                "sample": [list of sample IDs] or single ID,
                "origin": [list of origin IDs] or single ID,
                "group": [list of groups] or single group,
                "branch": [list of branches] or single branch,
                "processings": processing configuration,
                "augmentation": augmentation type,
                ... (any other column)
            }
            **kwargs: Additional column overrides (take precedence over indices)

        Returns:
            List of sample indices that were added

        Example:
            # Add samples with dictionary specification
            indexer.add_samples_dict(3, {
                "partition": "train",
                "group": [1, 2, 1],
                "processings": ["raw", "msc"]
            })
        """
        if indices is None:
            indices = {}
        params = self._convert_indexdict_to_params(indices, count)
        params.update(kwargs)
        return self._append(count, **params)

    def add_rows(self, n_rows: int, new_indices: Optional[Dict[str, Any]] = None) -> List[int]:
        """Add rows to the indexer with optional column overrides."""
        if n_rows <= 0:
            return []

        new_indices = new_indices or {}

        # Extract arguments for _append
        kwargs = {}

        # Handle special mappings
        if "sample" in new_indices:
            kwargs["sample_indices"] = new_indices["sample"]
        if "origin" in new_indices:
            kwargs["origin_indices"] = new_indices["origin"]
        elif "sample" not in new_indices:
            # For add_rows, default origin to sample indices when not explicitly set
            next_sample_idx = self.next_sample_index()
            kwargs["origin_indices"] = list(range(next_sample_idx, next_sample_idx + n_rows))

        # Handle direct mappings
        for key in ["partition", "group", "branch", "processings", "augmentation"]:
            if key in new_indices:
                kwargs[key] = new_indices[key]

        # Handle any other overrides
        for key, value in new_indices.items():
            if key not in ["sample", "origin", "partition", "group", "branch", "processings", "augmentation"]:
                kwargs[key] = value

        return self._append(n_rows, **kwargs)

    def add_rows_dict(
        self,
        n_rows: int,
        indices: IndexDict,
        **kwargs
    ) -> List[int]:
        """
        Add rows using dictionary-based parameter specification.

        This method provides a cleaner API for specifying row parameters
        using a dictionary, similar to the filtering API pattern.

        Args:
            n_rows: Number of rows to add
            indices: Dictionary containing column specifications {
                "partition": "train|test|val",
                "sample": [list of sample IDs] or single ID,
                "origin": [list of origin IDs] or single ID,
                "group": [list of groups] or single group,
                "branch": [list of branches] or single branch,
                "processings": processing configuration,
                "augmentation": augmentation type,
                ... (any other column)
            }
            **kwargs: Additional column overrides (take precedence over indices)

        Returns:
            List of sample indices that were added

        Example:
            # Add rows with dictionary specification
            indexer.add_rows_dict(2, {
                "partition": "val",
                "sample": [100, 101],
                "group": 5
            })
        """
        if n_rows <= 0:
            return []

        params = self._convert_indexdict_to_params(indices, n_rows)
        params.update(kwargs)  # kwargs take precedence
        return self._append(n_rows, **params)

    def register_samples(self, count: int, partition: PartitionType = "train") -> List[int]:
        """Register samples using the unified _append method."""
        return self._append(count, partition=partition)

    def register_samples_dict(
        self,
        count: int,
        indices: IndexDict,
        **kwargs
    ) -> List[int]:
        """
        Register samples using dictionary-based parameter specification.

        Args:
            count: Number of samples to register
            indices: Dictionary containing column specifications
            **kwargs: Additional column overrides (take precedence over indices)

        Returns:
            List of sample indices that were registered

        Example:
            indexer.register_samples_dict(5, {"partition": "test", "group": 2})
        """
        params = self._convert_indexdict_to_params(indices, count)
        params.update(kwargs)  # kwargs take precedence
        return self._append(count, **params)

    def update_by_filter(self, selector: Selector, updates: Dict[str, Any]) -> None:
        condition = self._build_filter_condition(selector)

        for col, value in updates.items():
            # Cast the literal value to the expected column type
            cast_value = pl.lit(value).cast(self.df.schema[col])
            self.df = self.df.with_columns(
                pl.when(condition).then(cast_value).otherwise(pl.col(col)).alias(col)
            )

    def update_by_indices(self, sample_indices: SampleIndices, updates: Dict[str, Any]) -> None:
        sample_ids = self._normalize_indices(sample_indices, len(sample_indices) if isinstance(sample_indices, (list, np.ndarray)) else 1, "sample_indices")
        condition = pl.col("row").is_in(sample_ids)

        for col, value in updates.items():
            # Cast the literal value to the expected column type
            cast_value = pl.lit(value).cast(self.df.schema[col])
            self.df = self.df.with_columns(
                pl.when(condition).then(cast_value).otherwise(pl.col(col)).alias(col)
            )


    def next_row_index(self) -> int:
        if len(self.df) == 0:
            return 0
        max_val = self.df["row"].max()
        return int(max_val) + 1 if max_val is not None else 0

    def next_sample_index(self) -> int:
        if len(self.df) == 0:
            return 0
        max_val = self.df["sample"].max()
        return int(max_val) + 1 if max_val is not None else 0

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

        # Create augmented samples using _append
        # Use first partition as default since partitions should be consistent
        partition = partitions[0] if partitions else "train"

        augmented_ids = self._append(
            total_augmentations,
            partition=partition,
            origin_indices=origin_indices,
            group=groups,
            branch=branches,
            processings=processings_list,
            augmentation=augmentation_id
        )

        return augmented_ids

    def __repr__(self):
        return str(self.df)

    def __str__(self):
        # Get columns to include (excluding sample and origin, and row)
        cols_to_include = [col for col in self.df.columns if col not in ["sample", "origin", "row"]]

        if not cols_to_include:
            return "No indexable columns found"

        if len(self.df) == 0:
            return "No rows found"

        # Group by all columns and count (include null values)
        combinations = self.df.select(cols_to_include).group_by(cols_to_include).agg(
            pl.len().alias("count")
        ).sort("count", descending=True)

        # Format output
        summary = []
        for row in combinations.to_dicts():
            # Build the combination string, skipping null values
            parts = []
            for col in cols_to_include:
                value = row[col]
                # Skip null values
                if value is None:
                    continue

                if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                    # Already formatted as list string
                    parts.append(f"{value}")
                else:
                    # Format other values appropriately
                    parts.append(f'"{value}"')

            # Only add to summary if we have at least one non-null value
            if parts:
                combination_str = ", ".join(parts)
                count = row["count"]
                summary.append(f"{combination_str}: {count} samples")

        parts_str = "\n- ".join(summary)
        return f"Indexes:\n- {parts_str}"
