from typing import Dict, List, Union, Any, Optional, overload, Mapping
import numpy as np
import polars as pl

from nirs4all.data.types import Selector, SampleIndices, PartitionType, ProcessingList, IndexDict
from nirs4all.data._indexer import (
    IndexStore,
    QueryBuilder,
    SampleManager,
    AugmentationTracker,
    ProcessingManager,
    ParameterNormalizer,
)


class Indexer:
    """
    Index manager for samples used in ML/DL pipelines.
    Optimizes contiguous access and manages filtering.

    This class is designed to retrieve data during ML pipelines.
    For example, it can be used to get all test samples from branch 2,
    including augmented samples, for specific processings such as
    ["raw", "savgol", "gaussian"].

    The Indexer uses a component-based architecture for maintainability:
    - IndexStore: DataFrame storage and queries
    - QueryBuilder: Selector to Polars expression conversion
    - SampleManager: ID generation
    - AugmentationTracker: Origin/augmented relationships
    - ProcessingManager: Processing list operations
    - ParameterNormalizer: Input validation
    """

    def __init__(self):
        # Initialize components
        self._store = IndexStore()
        self._query_builder = QueryBuilder(valid_columns=self._store.columns)
        self._sample_manager = SampleManager(self._store)
        self._augmentation_tracker = AugmentationTracker(self._store, self._query_builder)
        self._processing_manager = ProcessingManager(self._store)
        self._parameter_normalizer = ParameterNormalizer(default_processings=["raw"])

    @property
    def df(self) -> pl.DataFrame:
        """
        Get the underlying DataFrame for backward compatibility.

        Returns:
            pl.DataFrame: The complete index DataFrame.

        Note:
            Direct DataFrame access is provided for backward compatibility.
            Prefer using indexer methods when possible.
        """
        return self._store.df

    @property
    def default_values(self) -> Dict[str, Any]:
        """
        Get default values for backward compatibility.

        Returns:
            Dict[str, Any]: Default values used when parameters are None.
        """
        return {
            "partition": "train",
            "processings": ["raw"],
        }

    def _ensure_selector_dict(self, selector: Any) -> Dict[str, Any]:
        """Ensure selector is a dictionary."""
        if selector is None:
            return {}

        # Handle ExecutionContext (duck typing)
        if hasattr(selector, "selector") and hasattr(selector, "state"):
            return dict(selector.selector)

        if isinstance(selector, Mapping):
            return dict(selector)

        return {}

    def _apply_filters(self, selector: Selector) -> pl.DataFrame:
        """Apply selector filters and return filtered DataFrame."""
        selector = self._ensure_selector_dict(selector)
        condition = self._query_builder.build(selector, exclude_columns=["processings"])
        return self._store.query(condition)

    def _build_filter_condition(self, selector: Selector) -> pl.Expr:
        """Build a Polars filter expression from selector."""
        selector = self._ensure_selector_dict(selector)
        return self._query_builder.build(selector, exclude_columns=["processings"])

    def x_indices(self, selector: Selector, include_augmented: bool = True, include_excluded: bool = False) -> np.ndarray:
        """
        Get sample indices with optional augmented sample aggregation.

        This method implements two-phase selection to prevent data leakage:
        1. Phase 1: Get base samples (sample == origin)
        2. Phase 2: Get augmented versions of those base samples

        Args:
            selector: Filter criteria dictionary. Supported keys:
                - partition: "train"|"test"|"val" or list
                - group: int or list of ints
                - branch: int or list of ints
                - augmentation: str, list, or None for null check
                - Any other indexed columns
            include_augmented: If True, include augmented versions of selected samples.
                             If False, return only base samples (sample == origin).
                             Default True for backward compatibility.
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True for diagnostics, reporting, or viewing excluded samples.

        Returns:
            np.ndarray: Array of sample indices (dtype: np.int32). When include_augmented=True,
                       includes base samples and their augmented versions. When False, only
                       base samples where sample == origin.

        Raises:
            KeyError: If selector contains invalid column names.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, partition="train")
            >>> indexer.augment_rows([0, 1], 2, "flip")
            >>>
            >>> # Get all train samples (base + augmented)
            >>> all_train = indexer.x_indices({"partition": "train"})
            >>> # Returns: [0, 1, 2, 3, 4, 5, 6, 7, 8] (5 base + 4 augmented)
            >>>
            >>> # Get only base train samples
            >>> base_train = indexer.x_indices({"partition": "train"}, include_augmented=False)
            >>> # Returns: [0, 1, 2, 3, 4] (5 base only)
            >>>
            >>> # Mark sample as excluded and filter it
            >>> indexer.mark_excluded([0], reason="outlier")
            >>> filtered = indexer.x_indices({"partition": "train"})
            >>> # Returns: [1, 2, 3, 4, ...] (sample 0 and its augmentations excluded)
            >>>
            >>> # Include excluded samples (for diagnostics)
            >>> all_samples = indexer.x_indices({"partition": "train"}, include_excluded=True)

        Note:
            The two-phase selection ensures that augmented samples from other partitions
            are NOT included, preventing data leakage in cross-validation scenarios.
        """
        # Build the exclusion filter
        excluded_filter = self._query_builder.build_excluded_filter(include_excluded)

        if not include_augmented:
            # Simple case: filter for base samples only
            condition = self._build_filter_condition(selector)
            base_condition = condition & self._query_builder.build_base_samples_filter() & excluded_filter
            filtered_df = self._store.query(base_condition)
            return filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)

        # Two-phase selection using augmentation tracker
        # Pass both the base condition AND the exclusion filter
        condition = self._build_filter_condition(selector) & excluded_filter
        return self._augmentation_tracker.get_all_samples_with_augmentations(
            condition, additional_filter=excluded_filter
        )

    def y_indices(self, selector: Selector, include_augmented: bool = True, include_excluded: bool = False) -> np.ndarray:
        """
        Get y indices for samples. Returns origin indices for y-value lookup.

        For augmented samples, this method maps them to their base samples (origins)
        since y-values only exist for base samples. This enables proper target retrieval
        when working with augmented data.

        Args:
            selector: Filter criteria dictionary. Same format as x_indices().
                     See x_indices() for supported keys.
            include_augmented: If True (default), include augmented samples mapped to their origins.
                             If False, return only base sample origins (sample == origin).
                             Default True for backward compatibility with original behavior.
            include_excluded: If True, include samples marked as excluded.
                            If False (default), exclude samples marked as excluded=True.
                            Use True for diagnostics, reporting, or viewing excluded samples.

        Returns:
            np.ndarray: Array of origin sample indices for y-value lookup (dtype: np.int32).
                       When include_augmented=True (default), augmented samples are included
                       and each is mapped to its origin. When False, only base samples are
                       returned (sample == origin).

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, partition="train")
            >>> indexer.augment_rows([0, 1], 2, "flip")
            >>>
            >>> # Get origins for all train samples (base + augmented)
            >>> y_idx = indexer.y_indices({"partition": "train"})
            >>> # Returns: [0, 1, 2, 3, 4, 0, 0, 1, 1]
            >>> # (5 base origins + 4 augmented mapped to origins 0, 0, 1, 1)
            >>>
            >>> # Use with targets
            >>> targets = np.array([10, 20, 30, 40, 50])  # 5 base samples
            >>> x_idx = indexer.x_indices({"partition": "train"})
            >>> y_idx = indexer.y_indices({"partition": "train"})
            >>> X = all_spectra[x_idx]  # Get spectra (includes augmented)
            >>> y = targets[y_idx]   # Get targets (augmented samples use origin's target)
            >>>
            >>> # Get only base sample origins
            >>> base_origins = indexer.y_indices({"partition": "train"}, include_augmented=False)
            >>> # Returns: [0, 1, 2, 3, 4]
            >>>
            >>> # Exclude filtered samples
            >>> indexer.mark_excluded([0], reason="outlier")
            >>> filtered_y = indexer.y_indices({"partition": "train"})
            >>> # Sample 0 and its augmentations excluded from result

        Note:
            The length and order of y_indices() output always corresponds to x_indices()
            output with the same selector and include_augmented parameters. This ensures
            X and y arrays are properly aligned for training.
        """
        # Build the exclusion filter
        excluded_filter = self._query_builder.build_excluded_filter(include_excluded)

        filtered_df = self._apply_filters(selector) if selector else self._store.df
        # Apply exclusion filter
        filtered_df = filtered_df.filter(excluded_filter)

        if not include_augmented:
            # Return only base sample origins (sample == origin)
            base_condition = self._query_builder.build_base_samples_filter()
            return filtered_df.filter(base_condition).select(pl.col("origin")).to_series().to_numpy().astype(np.int32)

        # Include augmented samples: all origins are returned (with augmented samples mapped)
        return filtered_df.select(pl.col("origin")).to_series().to_numpy().astype(np.int32)

    def get_augmented_for_origins(self, origin_samples: List[int]) -> np.ndarray:
        """
        Get all augmented samples for given origin sample IDs.

        This method is used to retrieve augmented versions of base samples,
        enabling two-phase selection that prevents data leakage across CV folds.

        Args:
            origin_samples: List of origin sample IDs to find augmented versions for.
                          Can be empty list.

        Returns:
            np.ndarray: Array of augmented sample IDs (dtype: np.int32). Only includes
                       samples where origin is in origin_samples AND sample != origin
                       (actual augmented samples, not base samples).

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(3, partition="train")
            >>> indexer.augment_rows([0, 1], 2, "flip")
            >>>
            >>> # Get base samples
            >>> base_samples = indexer.x_indices({"partition": "train"}, include_augmented=False)
            >>> # base_samples: [0, 1, 2]
            >>>
            >>> # Get their augmented versions
            >>> augmented = indexer.get_augmented_for_origins(base_samples.tolist())
            >>> # augmented: [3, 4, 5, 6] (2 augmented each for samples 0 and 1)
            >>>
            >>> # Combine for full dataset
            >>> all_samples = np.concatenate([base_samples, augmented])
            >>> # all_samples: [0, 1, 2, 3, 4, 5, 6]

        Note:
            This method does not filter by partition, group, or other criteria. It returns
            ALL augmented samples for the given origins, regardless of their attributes.
            Use x_indices() for filtered retrieval with automatic augmentation handling.
        """
        return self._augmentation_tracker.get_augmented_for_origins(origin_samples)

    def get_origin_for_sample(self, sample_id: int) -> Optional[int]:
        """
        Get origin sample ID for a given sample.

        With the current design, all samples have origin set:
        - Base samples: origin == sample (self-referencing)
        - Augmented samples: origin != sample (references base sample)

        Args:
            sample_id: Sample ID to look up.

        Returns:
            Optional[int]: Origin sample ID, or None if sample not found in index.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(2, partition="train")
            >>> indexer.augment_rows([0], 1, "flip")
            >>>
            >>> # For augmented sample
            >>> origin = indexer.get_origin_for_sample(2)  # Sample 2 is augmentation of 0
            >>> print(origin)  # 0
            >>>
            >>> # For base sample
            >>> origin = indexer.get_origin_for_sample(0)  # Sample 0 is base
            >>> print(origin)  # 0 (self-referencing)
            >>>
            >>> # For non-existent sample
            >>> origin = indexer.get_origin_for_sample(999)
            >>> print(origin)  # None

        Note:
            This is a single-sample lookup. For batch operations, use y_indices()
            which is more efficient for retrieving origins for multiple samples.
        """
        return self._augmentation_tracker.get_origin_for_sample(sample_id)

    def replace_processings(self, source_processings: List[str], new_processings: List[str]) -> None:
        """
        Replace processing names across all samples.

        Creates a mapping from old to new processing names and applies it to
        all processing lists in the index.

        Args:
            source_processings: List of existing processing names to replace.
            new_processings: List of new processing names to set. Must have
                           same length as source_processings.

        Raises:
            ValueError: If source_processings and new_processings have different lengths.
            ValueError: If source_processings or new_processings is empty.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, processings=["raw", "old_msc", "savgol"])
            >>>
            >>> # Replace single processing
            >>> indexer.replace_processings(["old_msc"], ["msc"])
            >>> # Now all samples have ["raw", "msc", "savgol"]
            >>>
            >>> # Replace multiple processings
            >>> indexer.replace_processings(
            ...     ["raw", "savgol"],
            ...     ["raw_v2", "savgol_v2"]
            ... )
            >>> # Now all samples have ["raw_v2", "msc", "savgol_v2"]

        Note:
            - Operates on ALL rows in the index
            - Non-matched processings are left unchanged
            - Case-sensitive matching
            - Use this method when renaming processings after pipeline changes
        """
        self._processing_manager.replace_processings(source_processings, new_processings)

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
        self._processing_manager.reset_processings(new_processings)

    def add_processings(self, new_processings: List[str]) -> None:
        """
        Append processing names to all existing processing lists.

        Adds new processings to the end of each sample's processing list.
        This is useful when applying additional transformations to all data.

        Args:
            new_processings: List of new processing names to add to existing lists.

        Raises:
            ValueError: If new_processings is empty.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, processings=["raw", "msc"])
            >>>
            >>> # Add single processing
            >>> indexer.add_processings(["normalize"])
            >>> # All samples now have ["raw", "msc", "normalize"]
            >>>
            >>> # Add multiple processings
            >>> indexer.add_processings(["scale", "center"])
            >>> # All samples now have ["raw", "msc", "normalize", "scale", "center"]

        Note:
            - Operates on ALL rows in the index
            - Appends to the end of each processing list
            - Does not check for duplicates (allows intentional reprocessing)
            - Use this method when adding pipeline steps to existing data
        """
        self._processing_manager.add_processings(new_processings)

    def _normalize_indices(self, indices: SampleIndices, count: int, param_name: str) -> List[int]:
        """Normalize various index formats to a list of integers (internal helper)."""
        return self._parameter_normalizer.normalize_indices(indices, count, param_name)

    def _normalize_single_or_list(self, value: Union[Any, List[Any]], count: int, param_name: str, allow_none: bool = False) -> List[Any]:
        """Normalize single value or list to a list of specified length (internal helper)."""
        return self._parameter_normalizer.normalize_single_or_list(value, count, param_name, allow_none)

    def _prepare_processings(self, processings: Union[ProcessingList, List[ProcessingList], str, List[str], None], count: int) -> List[List[str]]:
        """Prepare processings list with proper validation (internal helper)."""
        return self._parameter_normalizer.prepare_processings(processings, count)

    def _convert_indexdict_to_params(self, index_dict: IndexDict, count: int) -> Dict[str, Any]:
        """Convert IndexDict to method parameters (internal helper)."""
        return self._parameter_normalizer.convert_indexdict_to_params(index_dict, count)

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
        Core method to append samples to the indexer (internal).

        Args:
            count: Number of samples to add
            partition: Data partition ("train", "test", "val")
            sample_indices: Specific sample IDs to use. If None, auto-increment
            origin_indices: Original sample IDs for augmented samples
            group: Group ID(s) - single value or list of values
            branch: Branch ID(s) - single value or list of values
            processings: Processing steps - single list or list of lists
            augmentation: Augmentation type(s) - single value or list
            **overrides: Additional column overrides

        Returns:
            List of sample indices that were added
        """
        if count <= 0:
            return []

        # Generate row and sample IDs using SampleManager
        row_ids = self._sample_manager.generate_row_ids(count)

        if sample_indices is None:
            sample_ids = self._sample_manager.generate_sample_ids(count)
            if origin_indices is None:
                # Base samples: origin = sample (self-referencing)
                origins = sample_ids.copy()
            else:
                origins = self._normalize_indices(origin_indices, count, "origin_indices")
        else:
            sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")
            if origin_indices is None:
                # Base samples: origin = sample (self-referencing)
                origins = [int(x) for x in sample_ids]
            else:
                origins = self._normalize_indices(origin_indices, count, "origin_indices")

        # Normalize column values
        groups = self._normalize_single_or_list(group, count, "group")
        branches = self._normalize_single_or_list(branch, count, "branch")
        processings_list = self._prepare_processings(processings, count)  # Now returns List[List[str]]
        augmentations = self._normalize_single_or_list(augmentation, count, "augmentation", allow_none=True)

        # Handle additional overrides
        additional_cols = {}
        for col, value in overrides.items():
            if col in self._store.columns and col not in ["row", "sample", "origin", "partition", "group", "branch", "processings", "augmentation"]:
                if isinstance(value, (list, np.ndarray)):
                    if len(value) != count:
                        raise ValueError(f"{col} length ({len(value)}) must match count ({count})")
                    additional_cols[col] = list(value)
                else:
                    additional_cols[col] = [value] * count

        # Create new DataFrame with native List type for processings
        new_data = {
            "row": pl.Series(row_ids, dtype=pl.Int32),
            "sample": pl.Series(sample_ids, dtype=pl.Int32),
            "origin": pl.Series(origins, dtype=pl.Int32),
            "partition": pl.Series([partition] * count, dtype=pl.Categorical),
            "group": pl.Series(groups, dtype=pl.Int8),
            "branch": pl.Series(branches, dtype=pl.Int8),
            "processings": pl.Series(processings_list, dtype=pl.List(pl.Utf8)),  # Native list!
            "augmentation": pl.Series(augmentations, dtype=pl.Categorical),
            "excluded": pl.Series([False] * count, dtype=pl.Boolean),  # Default: not excluded
            "exclusion_reason": pl.Series([None] * count, dtype=pl.Utf8),  # Default: no reason
        }

        # Add additional columns with proper casting
        for col, values in additional_cols.items():
            expected_dtype = self._store.schema[col]
            new_data[col] = pl.Series(values, dtype=expected_dtype)

        # Append to store
        self._store.append(new_data)

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

        This is the primary method for registering samples in the index. Samples can be
        base samples or augmented samples, with flexible parameter specification.

        Args:
            count: Number of samples to add. Must be positive.
            partition: Data partition ("train", "test", "val"). Default "train".
            sample_indices: Specific sample IDs to use. If None, auto-increment from
                          current max. Can be:
                          - int: Single ID repeated for all samples
                          - List[int]: One ID per sample (length must match count)
                          - np.ndarray: One ID per sample (length must match count)
            origin_indices: Original sample IDs for augmented samples. If None, samples
                          are treated as base samples (origin = sample). Same format
                          options as sample_indices.
            group: Group ID(s) for sample categorization. Can be:
                  - int: Single group for all samples
                  - List[int]: One group per sample (length must match count)
                  - None: No group assignment
            branch: Pipeline branch ID(s). Same format as group.
            processings: Processing transformations applied. Can be:
                        - None: Uses default ["raw"]
                        - List[str]: Single list for all samples (e.g., ["raw", "msc"])
                        - List[List[str]]: One list per sample (length must match count)
            augmentation: Augmentation type(s). Same format as group, but allows None values.
            **kwargs: Additional column values. Must match count if list/array.

        Returns:
            List[int]: List of sample IDs that were added. Length equals count.

        Raises:
            ValueError: If count <= 0, or if list/array parameter lengths don't match count.
            TypeError: If parameter types are invalid.

        Examples:
            >>> indexer = Indexer()
            >>>
            >>> # Add 5 base train samples with default settings
            >>> ids = indexer.add_samples(5)
            >>> # ids: [0, 1, 2, 3, 4]
            >>>
            >>> # Add test samples with specific processings
            >>> test_ids = indexer.add_samples(
            ...     3,
            ...     partition="test",
            ...     processings=["raw", "msc", "savgol"]
            ... )
            >>>
            >>> # Add samples with different groups
            >>> grouped_ids = indexer.add_samples(
            ...     4,
            ...     partition="train",
            ...     group=[1, 1, 2, 2],
            ...     processings=["raw"]
            ... )
            >>>
            >>> # Add augmented samples (references existing samples as origins)
            >>> aug_ids = indexer.add_samples(
            ...     2,
            ...     partition="train",
            ...     origin_indices=[0, 1],  # Augmentations of samples 0 and 1
            ...     augmentation="flip"
            ... )

        Note:
            - Auto-incrementing sample IDs start from 0 or next available ID
            - Base samples have origin == sample (self-referencing)
            - Augmented samples have origin != sample (references base sample)
            - Single values are broadcast to all samples
            - Lists/arrays must match count exactly
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
        """
        Update rows matching a selector filter.

        Args:
            selector: Filter criteria dictionary (same format as x_indices).
            updates: Dictionary of column:value pairs to update.

        Example:
            >>> indexer.update_by_filter({"partition": "train", "group": 1}, {"branch": 2})
        """
        condition = self._build_filter_condition(selector)
        self._store.update_by_condition(condition, updates)

    def update_by_indices(self, sample_indices: SampleIndices, updates: Dict[str, Any]) -> None:
        """
        Update rows by sample indices.

        Args:
            sample_indices: Sample IDs to update (int, list, or array).
            updates: Dictionary of column:value pairs to update.

        Example:
            >>> indexer.update_by_indices([0, 1, 2], {"group": 5})
        """
        count = len(sample_indices) if isinstance(sample_indices, (list, np.ndarray)) else 1
        sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")
        condition = self._query_builder.build_sample_filter(sample_ids)
        self._store.update_by_condition(condition, updates)

    def next_row_index(self) -> int:
        """
        Get the next available row index.

        Returns:
            int: Next row index (max + 1, or 0 if empty).

        Example:
            >>> next_idx = indexer.next_row_index()
        """
        return self._sample_manager.next_row_id()

    def next_sample_index(self) -> int:
        """
        Get the next available sample index.

        Returns:
            int: Next sample index (max + 1, or 0 if empty).

        Example:
            >>> next_idx = indexer.next_sample_index()
        """
        return self._sample_manager.next_sample_id()

    def get_column_values(self, col: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Get column values, optionally filtered.

        Args:
            col: Column name to retrieve.
            filters: Optional selector dictionary for filtering.

        Returns:
            List[Any]: Column values.

        Example:
            >>> partitions = indexer.get_column_values("partition")
            >>> train_groups = indexer.get_column_values("group", {"partition": "train"})
        """
        condition = self._build_filter_condition(filters) if filters else None
        return self._store.get_column(col, condition)

    def uniques(self, col: str) -> List[Any]:
        """
        Get unique values in a column.

        Args:
            col: Column name.

        Returns:
            List[Any]: Unique values in the column.

        Example:
            >>> unique_partitions = indexer.uniques("partition")
        """
        return self._store.get_unique(col)

    def augment_rows(self, samples: List[int], count: Union[int, List[int]], augmentation_id: str) -> List[int]:
        """
        Create augmented samples based on existing samples.

        This method creates new augmented samples that reference existing base samples
        as their origins. The augmented samples inherit all attributes (partition, group,
        branch, processings) from their origin samples.

        Args:
            samples: List of sample IDs to augment. Must exist in the index.
            count: Number of augmentations per sample. Can be:
                  - int: Same count for all samples
                  - List[int]: One count per sample (length must match samples)
            augmentation_id: String identifier for the augmentation type
                           (e.g., "flip", "rotate", "noise").

        Returns:
            List[int]: List of new sample IDs for the augmented samples.

        Raises:
            ValueError: If samples list is empty, if count list length doesn't match
                       samples length, or if any sample IDs are not found.

        Examples:
            >>> indexer = Indexer()
            >>> base_ids = indexer.add_samples(3, partition="train", processings=["raw", "msc"])
            >>>
            >>> # Create 2 augmentations for each base sample
            >>> aug_ids = indexer.augment_rows(base_ids, 2, "flip")
            >>> # aug_ids: [3, 4, 5, 6, 7, 8] (2 per sample)
            >>>
            >>> # Different counts per sample
            >>> aug_ids2 = indexer.augment_rows([0, 1], [1, 3], "rotate")
            >>> # aug_ids2: [9, 10, 11, 12] (1 for sample 0, 3 for sample 1)
            >>>
            >>> # Verify augmented samples reference their origins
            >>> origin = indexer.get_origin_for_sample(aug_ids[0])
            >>> print(origin)  # base_ids[0]

        Note:
            - Augmented samples inherit partition, group, branch, and processings from origins
            - origin field is set to the base sample ID
            - augmentation field is set to augmentation_id
            - Useful for data augmentation in ML pipelines (flips, rotations, noise, etc.)
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
        sample_filter = self._query_builder.build_sample_filter(samples)
        filtered_df = self._store.query(sample_filter).sort("sample")

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
            # Processings are now native lists, not strings
            processings_list.extend([sample_row["processings"]] * sample_count)

        # Create augmented samples using _append
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

    def __repr__(self) -> str:
        """
        String representation showing the DataFrame.

        Returns:
            str: String representation of the index DataFrame.
        """
        return str(self._store.df)

    def __str__(self) -> str:
        """
        Human-readable summary of indexed samples.

        Returns:
            str: Summary showing sample counts by combination of attributes.
        """
        df = self._store.df
        cols_to_include = [col for col in df.columns if col not in ["sample", "origin", "row"]]

        if not cols_to_include:
            return "No indexable columns found"

        if len(df) == 0:
            return "No rows found"

        # Group by all columns and count
        combinations = df.select(cols_to_include).group_by(cols_to_include).agg(
            pl.len().alias("count")
        ).sort("count", descending=True)

        # Format output
        summary = []
        for row in combinations.to_dicts():
            parts = []
            for col in cols_to_include:
                value = row[col]
                if value is None:
                    continue

                # Handle native list type for processings
                if col == "processings" and isinstance(value, list):
                    parts.append(f"{col} - {value}")
                elif isinstance(value, str):
                    parts.append(f'{col} - "{value}"')
                else:
                    parts.append(f"{col} - {value}")

            if parts:
                combination_str = ", ".join(parts)
                count = row["count"]
                summary.append(f"{combination_str}: {count} samples")

        parts_str = "\n- ".join(summary)
        return f"Indexes:\n- {parts_str}"

    # ==================== Sample Filtering Methods ====================

    def mark_excluded(
        self,
        sample_indices: SampleIndices,
        reason: Optional[str] = None,
        cascade_to_augmented: bool = True
    ) -> int:
        """
        Mark samples as excluded from training.

        Excluded samples are automatically filtered out from x_indices() and y_indices()
        calls unless include_excluded=True is explicitly passed. This provides a
        non-destructive way to remove outliers or corrupted samples from training.

        Args:
            sample_indices: Sample IDs to exclude. Can be:
                          - int: Single sample ID
                          - List[int]: List of sample IDs
                          - np.ndarray: Array of sample IDs
            reason: Optional string describing why samples are excluded
                   (e.g., "outlier", "corrupted", "low_quality").
            cascade_to_augmented: If True (default), also exclude augmented samples
                                 derived from the specified base samples. This prevents
                                 data leakage from augmented versions of excluded samples.

        Returns:
            int: Number of samples marked as excluded.

        Raises:
            ValueError: If sample_indices is empty.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, partition="train")
            >>> indexer.augment_rows([0, 1], 2, "flip")
            >>>
            >>> # Mark sample 0 as excluded (outlier detection)
            >>> n_excluded = indexer.mark_excluded([0], reason="iqr_outlier")
            >>> # n_excluded: 3 (sample 0 + 2 augmented versions)
            >>>
            >>> # Verify exclusion
            >>> train_samples = indexer.x_indices({"partition": "train"})
            >>> # Sample 0 and its augmentations no longer included
            >>>
            >>> # View excluded samples
            >>> excluded_df = indexer.get_excluded_samples()

        Note:
            - Exclusion is non-destructive: data remains in the indexer
            - Use mark_included() to reverse exclusion
            - Excluded samples can still be accessed via include_excluded=True
            - Cascade prevents data leakage from augmented versions
        """
        count = len(sample_indices) if isinstance(sample_indices, (list, np.ndarray)) else 1
        sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")

        if not sample_ids:
            return 0

        all_samples_to_exclude = set(sample_ids)

        # Cascade to augmented samples if requested
        if cascade_to_augmented:
            augmented = self._augmentation_tracker.get_augmented_for_origins(sample_ids)
            all_samples_to_exclude.update(augmented.tolist())

        # Build condition and update
        condition = self._query_builder.build_sample_filter(list(all_samples_to_exclude))
        updates = {"excluded": True}
        if reason is not None:
            updates["exclusion_reason"] = reason

        self._store.update_by_condition(condition, updates)
        return len(all_samples_to_exclude)

    def mark_included(
        self,
        sample_indices: Optional[SampleIndices] = None,
        cascade_to_augmented: bool = True
    ) -> int:
        """
        Remove exclusion flag from samples.

        This method reverses the effect of mark_excluded(), re-including samples
        in x_indices() and y_indices() results.

        Args:
            sample_indices: Sample IDs to include. Can be:
                          - int: Single sample ID
                          - List[int]: List of sample IDs
                          - np.ndarray: Array of sample IDs
                          - None: Include ALL currently excluded samples
            cascade_to_augmented: If True (default), also include augmented samples
                                 derived from the specified base samples.

        Returns:
            int: Number of samples marked as included.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, partition="train")
            >>> indexer.mark_excluded([0, 1], reason="outlier")
            >>>
            >>> # Re-include sample 0
            >>> n_included = indexer.mark_included([0])
            >>> # n_included: 1
            >>>
            >>> # Re-include all excluded samples
            >>> n_included = indexer.mark_included()  # No argument = all excluded

        Note:
            - Clears both the excluded flag and exclusion_reason
            - Useful for iterative filtering or correcting previous exclusions
        """
        if sample_indices is None:
            # Include all currently excluded samples
            condition = pl.col("excluded") == True  # noqa: E712
            count = len(self._store.query(condition))
            self._store.update_by_condition(condition, {"excluded": False, "exclusion_reason": None})
            return count

        count = len(sample_indices) if isinstance(sample_indices, (list, np.ndarray)) else 1
        sample_ids = self._normalize_indices(sample_indices, count, "sample_indices")

        if not sample_ids:
            return 0

        all_samples_to_include = set(sample_ids)

        # Cascade to augmented samples if requested
        if cascade_to_augmented:
            augmented = self._augmentation_tracker.get_augmented_for_origins(sample_ids)
            all_samples_to_include.update(augmented.tolist())

        # Build condition and update
        condition = self._query_builder.build_sample_filter(list(all_samples_to_include))
        self._store.update_by_condition(condition, {"excluded": False, "exclusion_reason": None})
        return len(all_samples_to_include)

    def get_excluded_samples(self, selector: Optional[Selector] = None) -> pl.DataFrame:
        """
        Get DataFrame of excluded samples with their exclusion reasons.

        Args:
            selector: Optional filter criteria to narrow down the query.
                     If None, returns all excluded samples.

        Returns:
            pl.DataFrame: DataFrame containing excluded samples with columns:
                         sample, origin, partition, group, branch, exclusion_reason.

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(5, partition="train")
            >>> indexer.mark_excluded([0, 1], reason="outlier")
            >>>
            >>> # Get all excluded samples
            >>> excluded_df = indexer.get_excluded_samples()
            >>> print(excluded_df)
            >>>
            >>> # Get excluded samples from train partition only
            >>> train_excluded = indexer.get_excluded_samples({"partition": "train"})

        Note:
            Returns a Polars DataFrame for efficient processing.
            Use .to_pandas() if pandas DataFrame is needed.
        """
        # Start with excluded=True filter
        condition = pl.col("excluded") == True  # noqa: E712

        # Add selector conditions if provided
        if selector:
            selector_condition = self._build_filter_condition(selector)
            condition = condition & selector_condition

        return self._store.query(condition).select([
            "sample", "origin", "partition", "group", "branch",
            "augmentation", "exclusion_reason"
        ])

    def get_exclusion_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of exclusions by reason.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_excluded: Total number of excluded samples
                - total_samples: Total number of samples in indexer
                - exclusion_rate: Ratio of excluded to total samples
                - by_reason: Dict mapping reason strings to counts
                - by_partition: Dict mapping partition names to excluded counts

        Examples:
            >>> indexer = Indexer()
            >>> indexer.add_samples(10, partition="train")
            >>> indexer.mark_excluded([0, 1], reason="outlier")
            >>> indexer.mark_excluded([2], reason="low_quality")
            >>>
            >>> summary = indexer.get_exclusion_summary()
            >>> print(summary)
            >>> # {
            >>> #     'total_excluded': 3,
            >>> #     'total_samples': 10,
            >>> #     'exclusion_rate': 0.3,
            >>> #     'by_reason': {'outlier': 2, 'low_quality': 1},
            >>> #     'by_partition': {'train': 3}
            >>> # }
        """
        df = self._store.df
        total_samples = len(df)

        # Filter to excluded samples
        excluded_df = df.filter(pl.col("excluded") == True)  # noqa: E712
        total_excluded = len(excluded_df)

        # Group by reason
        by_reason = {}
        if total_excluded > 0:
            reason_counts = excluded_df.group_by("exclusion_reason").agg(
                pl.len().alias("count")
            ).to_dicts()
            for row in reason_counts:
                reason = row["exclusion_reason"] if row["exclusion_reason"] else "unspecified"
                by_reason[reason] = row["count"]

        # Group by partition
        by_partition = {}
        if total_excluded > 0:
            partition_counts = excluded_df.group_by("partition").agg(
                pl.len().alias("count")
            ).to_dicts()
            for row in partition_counts:
                by_partition[row["partition"]] = row["count"]

        return {
            "total_excluded": total_excluded,
            "total_samples": total_samples,
            "exclusion_rate": total_excluded / total_samples if total_samples > 0 else 0.0,
            "by_reason": by_reason,
            "by_partition": by_partition,
        }

    def reset_exclusions(self, selector: Optional[Selector] = None) -> int:
        """
        Remove all exclusion flags matching the selector.

        This is a convenience method equivalent to calling mark_included() on all
        excluded samples matching the selector.

        Args:
            selector: Optional filter criteria. If None, resets ALL exclusions.

        Returns:
            int: Number of samples reset.

        Examples:
            >>> # Reset all exclusions
            >>> n_reset = indexer.reset_exclusions()
            >>>
            >>> # Reset only train partition exclusions
            >>> n_reset = indexer.reset_exclusions({"partition": "train"})
        """
        # Build condition for excluded samples
        condition = pl.col("excluded") == True  # noqa: E712

        # Add selector conditions if provided
        if selector:
            selector_condition = self._build_filter_condition(selector)
            condition = condition & selector_condition

        # Count before update
        count = len(self._store.query(condition))

        # Update to include
        self._store.update_by_condition(condition, {"excluded": False, "exclusion_reason": None})
        return count
