# Indexer Docstring Specifications

This document provides complete Google-style docstring specifications for all `Indexer` methods, both current and refactored.

---

## Current Indexer Methods - Docstring Updates

### Core Retrieval Methods

#### `x_indices()`
```python
def x_indices(self, selector: Selector, include_augmented: bool = True) -> np.ndarray:
    """
    Get sample indices with optional augmented sample aggregation.

    This method implements two-phase selection to prevent data leakage:
    1. Phase 1: Get base samples (sample == origin)
    2. Phase 2: Get augmented versions of those base samples

    Args:
        selector: Filter criteria dictionary. Supported keys:
            - partition: str or List[str] - Data partition(s) ("train", "test", "val")
            - group: int or List[int] - Group ID(s)
            - branch: int or List[int] - Branch ID(s)
            - augmentation: str or List[str] - Augmentation type(s)
            Any column in the index DataFrame can be used as a filter.
        include_augmented: If True, include augmented versions of selected samples.
            If False, return only base samples (sample == origin).
            Default True for backward compatibility.

    Returns:
        np.ndarray: Array of sample indices (dtype: np.int32). When include_augmented=True,
            returns base samples followed by their augmented versions. Order is stable
            within each group (base samples, then augmented samples).

    Raises:
        KeyError: If selector contains invalid column names.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(3, partition="train")  # samples 0, 1, 2
        >>> indexer.add_samples(2, partition="train", origin_indices=[0, 1], augmentation="aug_0")  # samples 3, 4
        >>>
        >>> # Get all train samples (base + augmented)
        >>> all_train = indexer.x_indices({"partition": "train"})
        >>> print(all_train)  # [0, 1, 2, 3, 4]
        >>>
        >>> # Get only base train samples
        >>> base_train = indexer.x_indices({"partition": "train"}, include_augmented=False)
        >>> print(base_train)  # [0, 1, 2]
        >>>
        >>> # Filter by multiple criteria
        >>> group0_train = indexer.x_indices({"partition": "train", "group": 0})
        >>>
        >>> # Empty selector returns all samples
        >>> all_samples = indexer.x_indices({})

    Note:
        The two-phase selection ensures that augmented samples from other partitions
        or groups are not accidentally included, preventing data leakage in
        cross-validation scenarios.
    """
```

#### `y_indices()`
```python
def y_indices(self, selector: Selector, include_augmented: bool = True) -> np.ndarray:
    """
    Get y indices for samples. Returns origin indices for y-value lookup.

    For augmented samples, this method maps them to their base samples (origins)
    since y-values only exist for base samples. This enables proper target retrieval
    when working with augmented data.

    Args:
        selector: Filter criteria dictionary. Same format as x_indices().
        include_augmented: If True (default), include augmented samples mapped to their origins.
            If False, return only base sample origins (sample == origin).
            Default True for backward compatibility with original behavior.

    Returns:
        np.ndarray: Array of origin sample indices for y-value lookup (dtype: np.int32).
            - For base samples: Returns the sample ID itself
            - For augmented samples: Returns the origin sample ID
            Length matches the number of samples returned by x_indices() with same parameters.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(3, partition="train")  # samples 0, 1, 2 (base)
        >>> indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # samples 3, 4 (augmented)
        >>>
        >>> # Get origin indices for all train samples (base + augmented mapped to origins)
        >>> origins_with_aug = indexer.y_indices({"partition": "train"})
        >>> print(origins_with_aug)  # [0, 1, 2, 0, 0] - note: samples 3,4 map to origin 0
        >>>
        >>> # Get origin indices only for base train samples
        >>> origins_base = indexer.y_indices({"partition": "train"}, include_augmented=False)
        >>> print(origins_base)  # [0, 1, 2]
        >>>
        >>> # Typical usage with dataset
        >>> x_indices = indexer.x_indices({"partition": "train"})
        >>> y_indices = indexer.y_indices({"partition": "train"})
        >>> X = features[x_indices]  # Get features for all samples
        >>> y = targets[y_indices]   # Get targets (augmented samples use origin's target)

    Note:
        The length and order of y_indices() output always corresponds to x_indices()
        output with the same selector and include_augmented parameters.
    """
```

#### `get_augmented_for_origins()`
```python
def get_augmented_for_origins(self, origin_samples: List[int]) -> np.ndarray:
    """
    Get all augmented samples for given origin sample IDs.

    This method is used to retrieve augmented versions of base samples,
    enabling two-phase selection that prevents data leakage across CV folds.

    Args:
        origin_samples: List of origin sample IDs to find augmented versions for.
            These should be base sample IDs (where sample == origin).

    Returns:
        np.ndarray: Array of augmented sample IDs (dtype: np.int32). Only includes
            samples where origin is in origin_samples AND sample != origin.
            Empty array if no augmented samples exist.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(3, partition="train")  # base: 0, 1, 2
        >>> indexer.add_samples(2, partition="train", origin_indices=[0, 0], augmentation="aug_0")  # aug: 3, 4
        >>> indexer.add_samples(1, partition="train", origin_indices=[1], augmentation="aug_1")  # aug: 5
        >>>
        >>> # Get augmented samples for origin 0
        >>> augmented = indexer.get_augmented_for_origins([0])
        >>> print(augmented)  # [3, 4]
        >>>
        >>> # Get augmented samples for multiple origins
        >>> augmented = indexer.get_augmented_for_origins([0, 1])
        >>> print(augmented)  # [3, 4, 5]
        >>>
        >>> # Two-phase selection example (prevents leakage)
        >>> base_samples = indexer.x_indices({"partition": "train"}, include_augmented=False)
        >>> augmented = indexer.get_augmented_for_origins(base_samples.tolist())
        >>> all_samples = np.concatenate([base_samples, augmented])

    Note:
        This method does not filter by partition, group, or other criteria. It returns
        ALL augmented samples for the given origins, regardless of their metadata.
        Use x_indices() for filtered retrieval.
    """
```

#### `get_origin_for_sample()`
```python
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
            - For base samples: Returns the sample_id itself
            - For augmented samples: Returns the ID of the base sample

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(2, partition="train")  # samples 0, 1 (base)
        >>> indexer.add_samples(1, partition="train", origin_indices=[0], augmentation="aug_0")  # sample 2 (augmented)
        >>>
        >>> # For augmented sample
        >>> origin = indexer.get_origin_for_sample(2)
        >>> print(origin)  # 0 (points to base sample)
        >>>
        >>> # For base sample
        >>> origin = indexer.get_origin_for_sample(0)
        >>> print(origin)  # 0 (points to itself)
        >>>
        >>> # For non-existent sample
        >>> origin = indexer.get_origin_for_sample(999)
        >>> print(origin)  # None

    Note:
        This is a single-sample lookup. For batch operations, use y_indices()
        which is more efficient.
    """
```

---

### Sample Addition Methods

#### `add_samples()`
```python
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
        partition: Data partition ("train", "test", "val", "validation"). Default "train".
        sample_indices: Specific sample IDs to use. If None, auto-increment from
            next available ID. Can be:
            - Single int: Use same ID for all (not recommended)
            - List[int]: One ID per sample
            - np.ndarray: Array of IDs
        origin_indices: Original sample IDs for augmented samples. If None,
            samples are treated as base samples (origin = sample). Can be:
            - Single int: All augmented from same origin
            - List[int]: One origin per sample
            - np.ndarray: Array of origins
        group: Group ID(s) for organizing samples into clusters. Can be:
            - None: No group assignment
            - Single int: All samples get same group
            - List[int]: One group per sample
        branch: Pipeline branch ID(s). Can be:
            - None: No branch assignment
            - Single int: All samples get same branch
            - List[int]: One branch per sample
        processings: Processing steps applied. Can be:
            - None: Use default ["raw"]
            - List[str]: Same processing chain for all samples, e.g. ["raw", "msc"]
            - List[List[str]]: Different chain per sample
            - str: Pre-formatted string representation
            - List[str]: Pre-formatted strings, one per sample
        augmentation: Augmentation type identifier. Can be:
            - None: Base samples (not augmented)
            - str: Same augmentation type for all
            - List[str]: Different type per sample
        **kwargs: Additional column values. Must match count if list/array.

    Returns:
        List[int]: List of sample IDs that were added. Length equals count.

    Raises:
        ValueError: If count <= 0, or if list/array parameter lengths don't match count.
        TypeError: If parameter types are invalid.

    Examples:
        >>> indexer = Indexer()
        >>>
        >>> # Add 5 base samples to train partition
        >>> ids = indexer.add_samples(5, partition="train", group=0)
        >>> print(ids)  # [0, 1, 2, 3, 4]
        >>>
        >>> # Add 2 augmented samples from origins 0 and 1
        >>> aug_ids = indexer.add_samples(
        ...     2,
        ...     partition="train",
        ...     origin_indices=[0, 1],
        ...     augmentation="aug_savgol"
        ... )
        >>> print(aug_ids)  # [5, 6]
        >>>
        >>> # Add samples with different groups per sample
        >>> ids = indexer.add_samples(3, group=[0, 1, 0], partition="train")
        >>>
        >>> # Add test samples with custom processing
        >>> ids = indexer.add_samples(
        ...     10,
        ...     partition="test",
        ...     processings=["raw", "msc", "savgol"],
        ...     branch=1
        ... )
        >>>
        >>> # Add samples with different processing per sample
        >>> ids = indexer.add_samples(
        ...     2,
        ...     processings=[["raw"], ["raw", "msc"]],
        ...     partition="val"
        ... )

    Note:
        - Auto-incrementing sample IDs start from 0 or next available ID
        - Origin indices default to sample indices for base samples
        - All list/array parameters must have length == count
        - Single values are broadcast to all samples
    """
```

#### `add_samples_dict()`
```python
def add_samples_dict(
    self,
    count: int,
    indices: Optional[IndexDict] = None,
    **kwargs
) -> List[int]:
    """
    Add multiple samples using dictionary-based parameter specification.

    This method provides a cleaner API for specifying sample parameters
    using a dictionary, similar to the filtering API pattern. Preferred
    for complex sample additions.

    Args:
        count: Number of samples to add. Must be positive.
        indices: Dictionary containing column specifications. Supported keys:
            - "partition": str - Data partition ("train", "test", "val")
            - "sample": int or List[int] - Sample IDs
            - "origin": int or List[int] - Origin IDs for augmented samples
            - "group": int or List[int] - Group IDs
            - "branch": int or List[int] - Branch IDs
            - "processings": ProcessingList - Processing chain(s)
            - "augmentation": str or List[str] - Augmentation type(s)
            - Any other column name with appropriate value(s)
        **kwargs: Additional column overrides. Take precedence over indices dict.

    Returns:
        List[int]: List of sample IDs that were added. Length equals count.

    Raises:
        ValueError: If count <= 0, or if list/array parameter lengths don't match count.
        KeyError: If indices contains invalid column names.

    Examples:
        >>> indexer = Indexer()
        >>>
        >>> # Add samples with dictionary specification
        >>> ids = indexer.add_samples_dict(3, {
        ...     "partition": "train",
        ...     "group": [1, 2, 1],
        ...     "processings": ["raw", "msc"]
        ... })
        >>> print(ids)  # [0, 1, 2]
        >>>
        >>> # Add augmented samples
        >>> aug_ids = indexer.add_samples_dict(2, {
        ...     "origin": [0, 1],
        ...     "augmentation": "aug_gaussian",
        ...     "partition": "train"
        ... })
        >>> print(aug_ids)  # [3, 4]
        >>>
        >>> # Override dict values with kwargs
        >>> ids = indexer.add_samples_dict(
        ...     5,
        ...     {"partition": "train", "group": 0},
        ...     group=1  # Overrides group from dict
        ... )
        >>>
        >>> # Add with custom metadata
        >>> ids = indexer.add_samples_dict(10, {
        ...     "partition": "test",
        ...     "fold_id": 2,
        ...     "cv_split": "outer_fold_1"
        ... })

    Note:
        - Special key mappings: "sample" → "sample_indices", "origin" → "origin_indices"
        - kwargs take precedence over indices dictionary
        - Cleaner than add_samples() for complex parameter combinations
    """
```

#### `add_rows()`
```python
def add_rows(self, n_rows: int, new_indices: Optional[Dict[str, Any]] = None) -> List[int]:
    """
    Add rows to the indexer with optional column overrides.

    Legacy method for backward compatibility. Prefer add_samples_dict() for new code.

    Args:
        n_rows: Number of rows to add. Must be positive.
        new_indices: Dictionary of column overrides. Same format as add_samples_dict().

    Returns:
        List[int]: List of sample IDs that were added.

    Raises:
        ValueError: If n_rows <= 0 or parameter lengths don't match.

    Examples:
        >>> indexer = Indexer()
        >>> ids = indexer.add_rows(5, {"partition": "train", "group": 0})

    Note:
        This method is maintained for backward compatibility. New code should use
        add_samples_dict() which has a clearer API.
    """
```

#### `add_rows_dict()`
```python
def add_rows_dict(
    self,
    n_rows: int,
    indices: IndexDict,
    **kwargs
) -> List[int]:
    """
    Add rows using dictionary-based parameter specification.

    Identical to add_samples_dict() but with "rows" terminology for
    backward compatibility.

    Args:
        n_rows: Number of rows to add.
        indices: Dictionary containing column specifications.
        **kwargs: Additional column overrides (take precedence over indices).

    Returns:
        List[int]: List of sample indices that were added.

    Examples:
        >>> ids = indexer.add_rows_dict(2, {
        ...     "partition": "val",
        ...     "sample": [100, 101],
        ...     "group": 5
        ... })

    Note:
        Prefer add_samples_dict() for new code. This method exists for consistency
        with add_rows().
    """
```

#### `register_samples()`
```python
def register_samples(self, count: int, partition: PartitionType = "train") -> List[int]:
    """
    Register samples with minimal parameters.

    Simplified method for basic sample registration with auto-generated IDs.
    Useful for quick dataset setup.

    Args:
        count: Number of samples to register. Must be positive.
        partition: Data partition. Default "train".

    Returns:
        List[int]: List of registered sample IDs.

    Raises:
        ValueError: If count <= 0.

    Examples:
        >>> indexer = Indexer()
        >>> train_ids = indexer.register_samples(100, partition="train")
        >>> test_ids = indexer.register_samples(20, partition="test")

    Note:
        This is a convenience method equivalent to:
        add_samples(count, partition=partition)
    """
```

#### `register_samples_dict()`
```python
def register_samples_dict(
    self,
    count: int,
    indices: IndexDict,
    **kwargs
) -> List[int]:
    """
    Register samples using dictionary-based parameter specification.

    Args:
        count: Number of samples to register.
        indices: Dictionary containing column specifications.
        **kwargs: Additional column overrides (take precedence over indices).

    Returns:
        List[int]: List of sample indices that were registered.

    Examples:
        >>> ids = indexer.register_samples_dict(5, {
        ...     "partition": "test",
        ...     "group": 2
        ... })

    Note:
        Identical to add_samples_dict(). Exists for API consistency.
    """
```

---

### Augmentation Methods

#### `augment_rows()`
```python
def augment_rows(
    self,
    samples: List[int],
    count: Union[int, List[int]],
    augmentation_id: str
) -> List[int]:
    """
    Create augmented samples based on existing samples.

    This method creates new augmented samples that reference existing base samples
    as their origins. The augmented samples inherit metadata from their origins
    (partition, group, branch, processings) but have unique sample IDs.

    Args:
        samples: List of sample IDs to augment. Must be existing sample IDs.
        count: Number of augmentations per sample. Can be:
            - Single int: Same count for all samples
            - List[int]: Different count per sample (must match len(samples))
        augmentation_id: String identifier for the augmentation type (e.g.,
            "aug_savgol", "aug_gaussian", "noise_0.1").

    Returns:
        List[int]: List of new sample IDs for the augmented samples. Length equals
            sum(count) if count is list, or len(samples) * count if count is int.

    Raises:
        ValueError: If samples not found, count is invalid, or count list length
            doesn't match samples length.
        KeyError: If sample IDs don't exist in the index.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(3, partition="train", group=0)  # [0, 1, 2]
        >>>
        >>> # Create 2 augmented versions of each sample
        >>> aug_ids = indexer.augment_rows([0, 1, 2], count=2, augmentation_id="aug_noise")
        >>> print(aug_ids)  # [3, 4, 5, 6, 7, 8]
        >>>
        >>> # Augmented samples inherit origin metadata
        >>> origin_0_aug = indexer.get_augmented_for_origins([0])
        >>> print(origin_0_aug)  # [3, 4]
        >>>
        >>> # Different augmentation counts per sample
        >>> more_aug = indexer.augment_rows(
        ...     [0, 1],
        ...     count=[1, 3],
        ...     augmentation_id="aug_savgol"
        ... )
        >>> print(more_aug)  # [9, 10, 11, 12] (1 for sample 0, 3 for sample 1)
        >>>
        >>> # Chain augmentations
        >>> base_ids = indexer.add_samples(5, partition="train")
        >>> first_aug = indexer.augment_rows(base_ids, 2, "aug_noise")
        >>> second_aug = indexer.augment_rows(base_ids, 1, "aug_smooth")
        >>> # Now each base sample has 3 augmented versions

    Note:
        - Augmented samples get metadata (partition, group, branch, processings) from origins
        - Each augmented sample maintains origin reference for y-value lookup
        - Origin must be an existing sample in the index
        - Augmentation ID helps track augmentation method used
    """
```

---

### Update Methods

#### `update_by_filter()`
```python
def update_by_filter(self, selector: Selector, updates: Dict[str, Any]) -> None:
    """
    Update samples matching filter criteria.

    Efficiently updates multiple samples at once based on selector criteria.
    Useful for bulk metadata updates.

    Args:
        selector: Filter criteria dictionary. Same format as x_indices().
        updates: Dictionary of column updates {column_name: new_value}.
            Values are broadcast to all matching rows.

    Raises:
        KeyError: If selector or updates contain invalid column names.
        TypeError: If update values don't match column types.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(10, partition="train", group=0)
        >>>
        >>> # Update all group 0 samples to group 1
        >>> indexer.update_by_filter(
        ...     {"partition": "train", "group": 0},
        ...     {"group": 1}
        ... )
        >>>
        >>> # Update branch for all augmented samples
        >>> indexer.update_by_filter(
        ...     {"augmentation": "aug_noise"},
        ...     {"branch": 2}
        ... )
        >>>
        >>> # Update multiple columns
        >>> indexer.update_by_filter(
        ...     {"partition": "test"},
        ...     {"group": 5, "branch": 1}
        ... )

    Note:
        - Updates are atomic per column
        - All matching rows get the same value
        - Type casting is automatic based on column schema
    """
```

#### `update_by_indices()`
```python
def update_by_indices(self, sample_indices: SampleIndices, updates: Dict[str, Any]) -> None:
    """
    Update specific samples by their row indices.

    Updates samples identified by row indices (not sample IDs). Use with caution
    as row indices can change.

    Args:
        sample_indices: Row indices to update. Can be:
            - Single int: Update one row
            - List[int]: Update multiple rows
            - np.ndarray: Array of row indices
        updates: Dictionary of column updates {column_name: new_value}.

    Raises:
        KeyError: If updates contain invalid column names.
        IndexError: If row indices are out of bounds.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(5, partition="train")
        >>>
        >>> # Update first row
        >>> indexer.update_by_indices(0, {"group": 1})
        >>>
        >>> # Update multiple rows
        >>> indexer.update_by_indices([0, 1, 2], {"branch": 2})

    Note:
        Prefer update_by_filter() which is more intuitive and less error-prone.
        This method uses row indices, not sample IDs.
    """
```

---

### Processing Management Methods

#### `replace_processings()`
```python
def replace_processings(self, source_processings: List[str], new_processings: List[str]) -> None:
    """
    Replace processing names for a specific source.

    Updates all processing lists that contain the source processings,
    replacing them with new names. Used when renaming or updating
    processing steps.

    Args:
        source_processings: List of existing processing names to replace.
        new_processings: List of new processing names to set.
            Must have same length as source_processings.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(5, processings=["raw", "old_proc", "msc"])
        >>>
        >>> # Rename processing step
        >>> indexer.replace_processings(
        ...     ["old_proc"],
        ...     ["new_proc"]
        ... )
        >>> # Now samples have ["raw", "new_proc", "msc"]
        >>>
        >>> # Replace multiple processings
        >>> indexer.replace_processings(
        ...     ["proc_old_1", "proc_old_2"],
        ...     ["proc_new_1", "proc_new_2"]
        ... )

    Note:
        - Only exact matches are replaced
        - Processing lists are stored as strings internally
        - Empty source_processings or new_processings does nothing
    """
```

#### `add_processings()`
```python
def add_processings(self, new_processings: List[str]) -> None:
    """
    Add new processing names to all existing processing lists.

    Appends new processing steps to every sample's processing list.
    Useful for adding preprocessing steps retroactively.

    Args:
        new_processings: List of new processing names to add to existing lists.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(5, processings=["raw", "msc"])
        >>>
        >>> # Add normalization step to all samples
        >>> indexer.add_processings(["normalize"])
        >>> # Now all samples have ["raw", "msc", "normalize"]
        >>>
        >>> # Add multiple processings
        >>> indexer.add_processings(["gaussian", "savgol"])
        >>> # Now ["raw", "msc", "normalize", "gaussian", "savgol"]

    Note:
        - Appends to ALL samples in the index
        - Processing order is preserved
        - Empty new_processings does nothing
    """
```

---

### Utility Methods

#### `next_row_index()`
```python
def next_row_index(self) -> int:
    """
    Get next available row index.

    Returns:
        int: Next sequential row index. Returns 0 if DataFrame is empty.

    Examples:
        >>> indexer = Indexer()
        >>> print(indexer.next_row_index())  # 0
        >>> indexer.add_samples(5)
        >>> print(indexer.next_row_index())  # 5
    """
```

#### `next_sample_index()`
```python
def next_sample_index(self) -> int:
    """
    Get next available sample index.

    Returns:
        int: Next sequential sample index. Returns 0 if DataFrame is empty.

    Examples:
        >>> indexer = Indexer()
        >>> print(indexer.next_sample_index())  # 0
        >>> indexer.add_samples(10)
        >>> print(indexer.next_sample_index())  # 10
    """
```

#### `get_column_values()`
```python
def get_column_values(self, col: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Get column values with optional filtering.

    Retrieves all values from a specific column, optionally filtered by criteria.

    Args:
        col: Column name to retrieve values from.
        filters: Optional filter criteria. Same format as x_indices() selector.

    Returns:
        List[Any]: List of column values. Order matches DataFrame row order.

    Raises:
        ValueError: If column does not exist.
        KeyError: If filters contain invalid column names.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(5, group=[0, 0, 1, 1, 2])
        >>>
        >>> # Get all groups
        >>> groups = indexer.get_column_values("group")
        >>> print(groups)  # [0, 0, 1, 1, 2]
        >>>
        >>> # Get groups for train partition only
        >>> train_groups = indexer.get_column_values(
        ...     "group",
        ...     filters={"partition": "train"}
        ... )
        >>>
        >>> # Get augmentation types
        >>> aug_types = indexer.get_column_values(
        ...     "augmentation",
        ...     filters={"augmentation": pl.col("augmentation").is_not_null()}
        ... )
    """
```

#### `uniques()`
```python
def uniques(self, col: str) -> List[Any]:
    """
    Get unique values in a column.

    Args:
        col: Column name to get unique values from.

    Returns:
        List[Any]: List of unique values in the column.

    Raises:
        ValueError: If column does not exist.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(10, group=[0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        >>>
        >>> # Get unique groups
        >>> unique_groups = indexer.uniques("group")
        >>> print(unique_groups)  # [0, 1, 2] (order may vary)
        >>>
        >>> # Get unique partitions
        >>> partitions = indexer.uniques("partition")
        >>> print(partitions)  # ["train", "test", "val"]
    """
```

---

### String Representations

#### `__repr__()`
```python
def __repr__(self) -> str:
    """
    Return string representation of the DataFrame.

    Returns:
        str: String representation showing the Polars DataFrame.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(3)
        >>> print(repr(indexer))
        # Prints Polars DataFrame representation
    """
```

#### `__str__()`
```python
def __str__(self) -> str:
    """
    Return human-readable summary of index contents.

    Groups samples by their metadata (partition, group, branch, processings,
    augmentation) and displays counts for each unique combination.

    Returns:
        str: Formatted summary of sample distributions.

    Examples:
        >>> indexer = Indexer()
        >>> indexer.add_samples(5, partition="train", group=0)
        >>> indexer.add_samples(3, partition="test", group=1)
        >>> print(indexer)
        Indexes:
        - partition - "train", group - 0, processings - "['raw']": 5 samples
        - partition - "test", group - 1, processings - "['raw']": 3 samples
    """
```

---

## Internal Helper Methods (Current Implementation)

These methods are internal (`_` prefix) and may change during refactoring:

### `_apply_filters()`
```python
def _apply_filters(self, selector: Selector) -> pl.DataFrame:
    """
    Apply selector filters to DataFrame and return filtered result.

    Internal method for executing filter operations.

    Args:
        selector: Filter criteria dictionary.

    Returns:
        pl.DataFrame: Filtered DataFrame.
    """
```

### `_build_filter_condition()`
```python
def _build_filter_condition(self, selector: Selector) -> pl.Expr:
    """
    Build Polars expression from selector dictionary.

    Internal method for constructing filter expressions.

    Args:
        selector: Filter criteria dictionary.

    Returns:
        pl.Expr: Polars expression for filtering.
    """
```

### `_normalize_indices()`
```python
def _normalize_indices(self, indices: SampleIndices, count: int, param_name: str) -> List[int]:
    """
    Normalize various index formats to a list of integers.

    Internal normalization helper.

    Args:
        indices: Index values to normalize.
        count: Expected count.
        param_name: Parameter name for error messages.

    Returns:
        List[int]: Normalized index list.

    Raises:
        ValueError: If length doesn't match count.
    """
```

### `_normalize_single_or_list()`
```python
def _normalize_single_or_list(
    self,
    value: Union[Any, List[Any]],
    count: int,
    param_name: str,
    allow_none: bool = False
) -> List[Any]:
    """
    Normalize single value or list to a list of specified length.

    Internal normalization helper.

    Args:
        value: Value to normalize.
        count: Expected count.
        param_name: Parameter name for error messages.
        allow_none: Whether None is allowed.

    Returns:
        List[Any]: Normalized value list.
    """
```

### `_prepare_processings()`
```python
def _prepare_processings(
    self,
    processings: Union[ProcessingList, List[ProcessingList], str, List[str], None],
    count: int
) -> List[str]:
    """
    Prepare processings list with proper validation and string conversion.

    Internal method for processing list normalization.

    Args:
        processings: Processing specification.
        count: Number of samples.

    Returns:
        List[str]: List of processing string representations.
    """
```

### `_convert_indexdict_to_params()`
```python
def _convert_indexdict_to_params(self, index_dict: IndexDict, count: int) -> Dict[str, Any]:
    """
    Convert IndexDict to method parameters.

    Internal conversion helper.

    Args:
        index_dict: Dictionary of index specifications.
        count: Number of samples.

    Returns:
        Dict[str, Any]: Converted parameters.
    """
```

### `_append()`
```python
def _append(
    self,
    count: int,
    *,
    partition: PartitionType = "train",
    sample_indices: Optional[SampleIndices] = None,
    origin_indices: Optional[SampleIndices] = None,
    group: Optional[Union[int, List[int]]] = None,
    branch: Optional[Union[int, List[int]]] = None,
    processings: Union[ProcessingList, List[ProcessingList], str, List[str], None] = None,
    augmentation: Optional[Union[str, List[str]]] = None,
    **overrides
) -> List[int]:
    """
    Core method to append samples to the indexer.

    Internal implementation method used by all public add methods.

    Args:
        count: Number of samples to add.
        partition: Data partition.
        sample_indices: Sample IDs.
        origin_indices: Origin sample IDs.
        group: Group assignment(s).
        branch: Branch assignment(s).
        processings: Processing configurations.
        augmentation: Augmentation type(s).
        **overrides: Additional column overrides.

    Returns:
        List[int]: List of added sample IDs.
    """
```

---

*Document Version: 1.0*
*Date: 2025-10-29*
*Author: GitHub Copilot with nirs4all team*
