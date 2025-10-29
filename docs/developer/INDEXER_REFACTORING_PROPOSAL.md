# Indexer Refactoring Proposal

## Executive Summary

The `Indexer` class has grown to handle multiple distinct responsibilities, making it difficult to maintain and test. This proposal outlines a comprehensive refactoring that:

1. **Extracts components** into focused, single-responsibility classes
2. **Maintains public API signatures** for backward compatibility
3. **Improves performance** through optimized polars operations
4. **Enhances maintainability** with clear separation of concerns
5. **Completes documentation** with proper Google-style docstrings

---

## Current Issues

### 1. **Multiple Responsibilities**
The `Indexer` class currently handles:
- DataFrame management (storage, filtering, updates)
- Sample index generation and tracking
- Augmentation management (origins, augmented samples)
- Processing list manipulation (string-based)
- Parameter normalization and validation
- Data retrieval (x_indices, y_indices)

### 2. **API Inconsistency**
Multiple overlapping methods for similar operations:
- `add_samples()` vs `add_samples_dict()` vs `add_rows()` vs `add_rows_dict()` vs `register_samples()` vs `register_samples_dict()`
- All route through internal `_append()` method
- Dictionary-based API (`*_dict` methods) provides cleaner interface but coexists with legacy methods

### 3. **String-based Processing Lists**
Processing lists are stored as string representations of Python lists:
- Error-prone parsing with `eval()` - security and reliability concerns
- Difficult to query and filter efficiently
- Manual string manipulation in `replace_processings()` and `add_processings()`
- **Solution**: Migrate to native Polars `List(Utf8)` type for clean, efficient storage

### 4. **Complex Parameter Normalization**
Multiple helper methods (`_normalize_indices`, `_normalize_single_or_list`, `_prepare_processings`) handle parameter variations with intricate logic.

### 5. **Missing or Incomplete Docstrings**
Several methods lack proper Google-style docstrings or have incomplete documentation.

---

## Proposed Architecture

### Component Structure

```
nirs4all/data/
├── indexer.py                          # Main facade (simplified)
└── indexer_components/
    ├── __init__.py
    ├── index_store.py                  # DataFrame storage & filtering
    ├── sample_manager.py               # Sample registration & index generation
    ├── augmentation_tracker.py         # Origin/augmented relationship management
    ├── processing_manager.py           # Processing list operations
    ├── parameter_normalizer.py         # Input validation & normalization
    └── query_builder.py                # Selector to polars expression conversion
```

---

## Component Details

### 1. **IndexStore** (`index_store.py`)
**Responsibility:** Low-level DataFrame storage and query execution

```python
class IndexStore:
    """
    Low-level storage for sample index data using Polars DataFrame.

    Handles DataFrame initialization, schema management, and basic
    CRUD operations. Does not contain business logic.

    Attributes:
        df (pl.DataFrame): Sample index storage

    Examples:
        >>> store = IndexStore()
        >>> store.append_rows(data_dict)
        >>> filtered = store.filter(condition_expr)
    """

    def __init__(self):
        """Initialize empty DataFrame with proper schema."""

    def append_rows(self, data: Dict[str, pl.Series]) -> None:
        """Append rows to the DataFrame."""

    def filter(self, condition: pl.Expr) -> pl.DataFrame:
        """Execute filter and return result DataFrame."""

    def update_rows(self, condition: pl.Expr, updates: Dict[str, Any]) -> None:
        """Update rows matching condition."""

    def get_column_values(self, col: str, condition: Optional[pl.Expr] = None) -> List[Any]:
        """Get column values, optionally filtered."""

    def get_unique_values(self, col: str) -> List[Any]:
        """Get unique values for a column."""

    def next_row_index(self) -> int:
        """Get next available row index."""

    def count_rows(self, condition: Optional[pl.Expr] = None) -> int:
        """Count rows matching condition."""

    @property
    def columns(self) -> List[str]:
        """Get list of column names."""

    def __repr__(self) -> str:
        """Return DataFrame representation."""
```

**Benefits:**
- Clear separation of storage from logic
- Easy to test in isolation
- Simple to replace with alternative backends if needed
- Encapsulates all polars-specific operations

---

### 2. **QueryBuilder** (`query_builder.py`)
**Responsibility:** Convert Selector dictionaries to Polars expressions

```python
class QueryBuilder:
    """
    Builds Polars filter expressions from Selector dictionaries.

    Translates high-level filter specifications into optimized
    Polars expressions for efficient DataFrame queries.

    Examples:
        >>> builder = QueryBuilder()
        >>> expr = builder.build({"partition": "train", "group": [0, 1]})
        >>> # Returns: (pl.col("partition") == "train") & pl.col("group").is_in([0, 1])
    """

    def build(self, selector: Selector) -> pl.Expr:
        """
        Build filter expression from selector dictionary.

        Args:
            selector: Filter specifications {column: value(s)}

        Returns:
            Polars expression for filtering

        Examples:
            >>> builder.build({"partition": "train"})
            >>> builder.build({"group": [0, 1, 2]})
            >>> builder.build({"partition": "train", "branch": 0})
        """

    def build_base_sample_filter(self, selector: Selector) -> pl.Expr:
        """
        Build filter for base samples (sample == origin).

        Args:
            selector: Filter specifications

        Returns:
            Expression that filters for base samples matching selector
        """

    def build_augmented_filter(self, origin_ids: List[int]) -> pl.Expr:
        """
        Build filter for augmented samples of given origins.

        Args:
            origin_ids: List of origin sample IDs

        Returns:
            Expression: (origin in list) & (sample != origin)
        """
```

**Benefits:**
- Centralizes query logic
- Easier to optimize query performance
- Consistent expression building
- Testable in isolation

---

### 3. **SampleManager** (`sample_manager.py`)
**Responsibility:** Sample ID generation and registration

```python
class SampleManager:
    """
    Manages sample ID generation and tracking.

    Handles auto-incrementing sample IDs, sample registration,
    and sample count tracking.

    Attributes:
        _next_sample_id (int): Next available sample ID

    Examples:
        >>> manager = SampleManager()
        >>> ids = manager.generate_sample_ids(5)
        >>> # Returns: [0, 1, 2, 3, 4]
    """

    def __init__(self, initial_id: int = 0):
        """
        Initialize sample manager.

        Args:
            initial_id: Starting sample ID (default: 0)
        """

    def generate_sample_ids(self, count: int) -> List[int]:
        """
        Generate sequential sample IDs.

        Args:
            count: Number of IDs to generate

        Returns:
            List of new sample IDs

        Examples:
            >>> manager = SampleManager()
            >>> ids = manager.generate_sample_ids(3)
            >>> # Returns: [0, 1, 2]
            >>> more_ids = manager.generate_sample_ids(2)
            >>> # Returns: [3, 4]
        """

    def set_next_id(self, next_id: int) -> None:
        """
        Set the next sample ID to use.

        Args:
            next_id: Next sample ID

        Raises:
            ValueError: If next_id is negative
        """

    @property
    def next_id(self) -> int:
        """Get next available sample ID."""
```

**Benefits:**
- Simple, focused responsibility
- Thread-safe ID generation (if needed in future)
- Clear API for sample tracking

---

### 4. **AugmentationTracker** (`augmentation_tracker.py`)
**Responsibility:** Manage origin/augmented sample relationships

```python
class AugmentationTracker:
    """
    Tracks relationships between base and augmented samples.

    Provides methods for two-phase sample selection that prevents
    data leakage in cross-validation scenarios.

    Examples:
        >>> tracker = AugmentationTracker(store, query_builder)
        >>> # Get base samples
        >>> base = tracker.get_base_samples({"partition": "train"})
        >>> # Get their augmented versions
        >>> augmented = tracker.get_augmented_for_origins(base.tolist())
    """

    def __init__(self, store: IndexStore, query_builder: QueryBuilder):
        """
        Initialize augmentation tracker.

        Args:
            store: Index storage
            query_builder: Query expression builder
        """

    def get_base_samples(self, selector: Selector) -> np.ndarray:
        """
        Get base sample IDs (sample == origin) matching selector.

        Args:
            selector: Filter criteria

        Returns:
            Array of base sample IDs

        Examples:
            >>> base = tracker.get_base_samples({"partition": "train", "group": 0})
        """

    def get_augmented_for_origins(self, origin_ids: List[int]) -> np.ndarray:
        """
        Get augmented sample IDs for given origin IDs.

        Phase 2 of two-phase selection for leak prevention.

        Args:
            origin_ids: List of origin sample IDs

        Returns:
            Array of augmented sample IDs (origin in list, sample != origin)

        Examples:
            >>> augmented = tracker.get_augmented_for_origins([0, 1, 2])
        """

    def get_origin_for_sample(self, sample_id: int) -> Optional[int]:
        """
        Get origin sample ID for a given sample.

        Args:
            sample_id: Sample ID to lookup

        Returns:
            Origin sample ID, or None if sample not found

        Examples:
            >>> origin = tracker.get_origin_for_sample(100)  # Returns e.g. 10
        """

    def create_augmented_samples(
        self,
        origin_ids: List[int],
        augmentation_id: str,
        count_per_origin: Union[int, List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare augmented sample data for given origins.

        Retrieves origin sample metadata and replicates it for
        augmented samples with new sample IDs.

        Args:
            origin_ids: Origin sample IDs to augment
            augmentation_id: Augmentation type identifier
            count_per_origin: Number of augmentations per origin

        Returns:
            List of dictionaries with sample data for augmented samples

        Raises:
            ValueError: If origin samples not found
        """
```

**Benefits:**
- Encapsulates leak prevention logic
- Clear two-phase selection pattern
- Easier to test augmentation scenarios
- Documents the critical augmentation workflow

---

### 5. **ProcessingManager** (`processing_manager.py`)
**Responsibility:** Handle processing list operations

```python
class ProcessingManager:
    """
    Manages processing list operations and transformations.

    Works with native Polars List(Utf8) column type for clean,
    efficient processing list storage without eval() or string parsing.

    Examples:
        >>> manager = ProcessingManager()
        >>> # Processing lists are native Polars lists
        >>> # No string conversion needed!
    """

    def replace_processings(
        self,
        store: IndexStore,
        source_processings: List[str],
        new_processings: List[str]
    ) -> None:
        """
        Replace processing names across all samples.

        Uses native Polars list operations for efficient replacement.

        Args:
            store: Index storage
            source_processings: Processing names to replace
            new_processings: New processing names

        Examples:
            >>> manager.replace_processings(
            ...     store,
            ...     ["proc_old_1", "proc_old_2"],
            ...     ["proc_new_1", "proc_new_2"]
            ... )
        """

    def add_processings(
        self,
        store: IndexStore,
        new_processings: List[str]
    ) -> None:
        """
        Add new processing names to all existing processing lists.

        Uses Polars list concatenation for efficient appending.

        Args:
            store: Index storage
            new_processings: Processing names to append

        Examples:
            >>> manager.add_processings(store, ["gaussian", "normalize"])
        """
```

**Benefits:**
- Native Polars List type - no eval() needed
- Clean, type-safe operations
- Efficient list operations (replace, append)
- Clear API for processing modifications

---

### 6. **ParameterNormalizer** (`parameter_normalizer.py`)
**Responsibility:** Validate and normalize input parameters

```python
class ParameterNormalizer:
    """
    Normalizes and validates parameters for sample operations.

    Converts various input formats (single values, lists, arrays)
    into consistent list formats required by IndexStore.

    Examples:
        >>> normalizer = ParameterNormalizer()
        >>> groups = normalizer.normalize_single_or_list(0, 5, "group")
        >>> # Returns: [0, 0, 0, 0, 0]
    """

    @staticmethod
    def normalize_indices(
        indices: SampleIndices,
        count: int,
        param_name: str
    ) -> List[int]:
        """
        Normalize various index formats to a list of integers.

        Args:
            indices: Sample indices (int, list, or ndarray)
            count: Expected number of indices
            param_name: Parameter name (for error messages)

        Returns:
            List of integer indices

        Raises:
            ValueError: If length doesn't match count

        Examples:
            >>> ParameterNormalizer.normalize_indices([1, 2, 3], 3, "sample_indices")
            [1, 2, 3]
            >>> ParameterNormalizer.normalize_indices(5, 3, "sample_indices")
            [5, 5, 5]
        """

    @staticmethod
    def normalize_single_or_list(
        value: Union[Any, List[Any]],
        count: int,
        param_name: str,
        allow_none: bool = False
    ) -> List[Any]:
        """
        Normalize single value or list to a list of specified length.

        Args:
            value: Single value or list
            count: Expected number of values
            param_name: Parameter name (for error messages)
            allow_none: Whether None is allowed as a value

        Returns:
            List of values with length == count

        Raises:
            ValueError: If list length doesn't match count or None when not allowed

        Examples:
            >>> ParameterNormalizer.normalize_single_or_list("train", 3, "partition")
            ['train', 'train', 'train']
            >>> ParameterNormalizer.normalize_single_or_list([0, 1, 2], 3, "group")
            [0, 1, 2]
        """

    @staticmethod
    def prepare_processings(
        processings: Union[ProcessingList, List[ProcessingList], str, List[str], None],
        count: int,
        default: List[str]
    ) -> List[str]:
        """
        Prepare processing lists with proper validation and string conversion.

        Handles various input formats:
        - None: Use default
        - Single list: Apply to all samples
        - List of lists: One per sample
        - String representation(s): Already formatted

        Args:
            processings: Processing specification
            count: Number of samples
            default: Default processing list if None

        Returns:
            List of processing string representations

        Raises:
            ValueError: If list count doesn't match sample count

        Examples:
            >>> ParameterNormalizer.prepare_processings(["raw", "msc"], 2, ["raw"])
            ["['raw', 'msc']", "['raw', 'msc']"]
            >>> ParameterNormalizer.prepare_processings([["raw"], ["msc"]], 2, ["raw"])
            ["['raw']", "['msc']"]
        """

    @staticmethod
    def convert_indexdict_to_params(
        index_dict: IndexDict,
        count: int
    ) -> Dict[str, Any]:
        """
        Convert IndexDict to normalized method parameters.

        Maps dictionary keys to parameter names and normalizes values.
        Special handling for "sample" → "sample_indices" and
        "origin" → "origin_indices".

        Args:
            index_dict: Dictionary of index specifications
            count: Number of samples being added

        Returns:
            Dictionary of normalized parameters

        Examples:
            >>> ParameterNormalizer.convert_indexdict_to_params(
            ...     {"partition": "train", "sample": [0, 1], "group": 0},
            ...     2
            ... )
            {'partition': 'train', 'sample_indices': [0, 1], 'group': 0}
        """
```

**Benefits:**
- Centralizes all validation logic
- Consistent error messages
- Easier to test edge cases
- Clear documentation of expected formats

---

### 7. **Indexer** (Refactored Facade)
**Responsibility:** Public API orchestration

The main `Indexer` class becomes a thin facade that:
1. Delegates to components
2. Maintains public API signatures
3. Provides convenience methods

```python
class Indexer:
    """
    Index manager for samples used in ML/DL pipelines.

    Optimizes contiguous access and manages filtering. This is a facade
    that coordinates specialized components for storage, querying,
    augmentation, and parameter normalization.

    This class is designed to retrieve data during ML pipelines.
    For example, it can be used to get all test samples from branch 2,
    including augmented samples, for specific processings such as
    ["raw", "savgol", "gaussian"].

    Attributes:
        df (pl.DataFrame): Direct access to underlying DataFrame (for debugging)

    Examples:
        >>> indexer = Indexer()
        >>> # Add samples
        >>> indexer.add_samples(5, partition="train", group=0)
        >>> # Get train sample indices
        >>> train_indices = indexer.x_indices({"partition": "train"})
    """

    def __init__(self):
        """Initialize indexer with component dependencies."""
        self._store = IndexStore()
        self._query_builder = QueryBuilder()
        self._sample_manager = SampleManager()
        self._augmentation_tracker = AugmentationTracker(
            self._store, self._query_builder
        )
        self._processing_manager = ProcessingManager()
        self._normalizer = ParameterNormalizer()

        self.default_values = {
            "partition": "train",
            "processings": ["raw"],
        }

    @property
    def df(self) -> pl.DataFrame:
        """Access underlying DataFrame for debugging/inspection."""
        return self._store.df

    # Public API methods delegate to components...
    # (Implementation details in next section)
```

---

## Refactored Method Implementations

### Data Retrieval Methods

```python
def x_indices(
    self,
    selector: Selector,
    include_augmented: bool = True
) -> np.ndarray:
    """
    Get sample indices with optional augmented sample aggregation.

    This method implements two-phase selection to prevent data leakage:
    1. Phase 1: Get base samples (sample == origin)
    2. Phase 2: Get augmented versions of those base samples

    Args:
        selector: Filter criteria (partition, group, branch, etc.)
        include_augmented: If True, include augmented versions of selected samples.
                         If False, return only base samples (sample == origin).
                         Default True for backward compatibility.

    Returns:
        Array of sample indices

    Examples:
        >>> # Get all train samples (base + augmented)
        >>> all_train = indexer.x_indices({"partition": "train"})
        >>> # Get only base train samples
        >>> base_train = indexer.x_indices({"partition": "train"}, include_augmented=False)
    """
    if not include_augmented:
        # Simple case: delegate to augmentation tracker
        return self._augmentation_tracker.get_base_samples(selector)

    # Two-phase selection
    base_indices = self._augmentation_tracker.get_base_samples(selector)

    if len(base_indices) == 0:
        return base_indices

    # Get augmented versions
    augmented_indices = self._augmentation_tracker.get_augmented_for_origins(
        base_indices.tolist()
    )

    if len(augmented_indices) > 0:
        return np.concatenate([base_indices, augmented_indices])

    return base_indices


def y_indices(
    self,
    selector: Selector,
    include_augmented: bool = True
) -> np.ndarray:
    """
    Get y indices for samples. Returns origin indices for y-value lookup.

    For augmented samples, this method maps them to their base samples (origins)
    since y-values only exist for base samples.

    Args:
        selector: Filter criteria (partition, group, branch, etc.)
        include_augmented: If True (default), include augmented samples mapped to their origins.
                         If False, return only base sample origins (sample == origin).
                         Default True for backward compatibility with original behavior.

    Returns:
        Array of origin sample indices for y-value lookup. When include_augmented=True (default),
        augmented samples are included and mapped to their origins. When False, only
        base samples are returned (sample == origin).

    Examples:
        >>> # Get origin indices for all train samples (base + augmented mapped to origins)
        >>> origins_with_aug = indexer.y_indices({"partition": "train"})
        >>> # Get origin indices only for base train samples
        >>> origins_base = indexer.y_indices({"partition": "train"}, include_augmented=False)
    """
    condition = self._query_builder.build(selector) if selector else pl.lit(True)

    if not include_augmented:
        condition = condition & (pl.col("sample") == pl.col("origin"))

    filtered_df = self._store.filter(condition)
    return filtered_df.select(pl.col("origin")).to_series().to_numpy().astype(np.int32)
```

### Sample Addition Methods

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

    Args:
        count: Number of samples to add
        partition: Data partition ("train", "test", "val")
        sample_indices: Specific sample IDs to use. If None, auto-increment
        origin_indices: Original sample IDs for augmented samples. If None,
                       samples are treated as base samples (origin = sample).
        group: Group ID(s) - single value or list of values
        branch: Branch ID(s) - single value or list of values
        processings: Processing steps - single list or list of lists
        augmentation: Augmentation type(s) - single value or list
        **kwargs: Additional column overrides

    Returns:
        List of sample indices that were added

    Raises:
        ValueError: If parameter list lengths don't match count

    Examples:
        >>> # Add 5 base samples to train partition
        >>> ids = indexer.add_samples(5, partition="train", group=0)
        >>> # Returns: [0, 1, 2, 3, 4]

        >>> # Add 2 augmented samples
        >>> aug_ids = indexer.add_samples(
        ...     2,
        ...     partition="train",
        ...     origin_indices=[0, 1],
        ...     augmentation="aug_savgol"
        ... )
        >>> # Returns: [5, 6]

        >>> # Add samples with different groups
        >>> ids = indexer.add_samples(3, group=[0, 1, 0])
    """
    return self._add_samples_internal(
        count=count,
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

    Examples:
        >>> # Add samples with dictionary specification
        >>> indexer.add_samples_dict(3, {
        ...     "partition": "train",
        ...     "group": [1, 2, 1],
        ...     "processings": ["raw", "msc"]
        ... })

        >>> # Add augmented samples
        >>> indexer.add_samples_dict(2, {
        ...     "origin": [0, 1],
        ...     "augmentation": "aug_gaussian"
        ... })
    """
    if indices is None:
        indices = {}
    params = self._normalizer.convert_indexdict_to_params(indices, count)
    params.update(kwargs)  # kwargs take precedence
    return self._add_samples_internal(count, **params)
```

---

## Performance Optimizations

### 1. **Batch Operations**
Replace multiple single-row operations with bulk operations:

```python
# Before: Multiple filter operations
for sample_id in sample_ids:
    row = df.filter(pl.col("sample") == sample_id)
    # process row

# After: Single filter with is_in
filtered = df.filter(pl.col("sample").is_in(sample_ids))
```

### 2. **Lazy Evaluation**
Use polars lazy API for complex queries:

```python
def x_indices_lazy(self, selector: Selector) -> np.ndarray:
    """Use lazy evaluation for complex queries."""
    lf = self._store.df.lazy()
    condition = self._query_builder.build(selector)
    result = lf.filter(condition).select(pl.col("sample")).collect()
    return result.to_series().to_numpy()
```

---

## Migration Strategy

### Phase 1: Extract Components (Week 1)
1. Create `indexer_components/` directory
2. Implement `IndexStore` with existing DataFrame logic
3. **Migrate processings column from `pl.Utf8` to `pl.List(pl.Utf8)` (~25 lines)**
4. Implement `QueryBuilder` with filter logic
5. Implement `ParameterNormalizer` with normalization methods
6. Write unit tests for each component

### Phase 2: Refactor Support Classes (Week 1-2)
1. Implement `SampleManager`
2. Implement `AugmentationTracker`
3. Implement `ProcessingManager` (simplified - no string conversion needed)
4. Write unit tests for each

### Phase 3: Refactor Main Indexer (Week 2)
1. Replace internal implementations with component delegation
2. Ensure all public APIs remain unchanged
3. Update internal helper methods
4. Add comprehensive docstrings

### Phase 4: Testing & Validation (Week 2-3)
1. Run existing test suite (should pass without changes)
2. Add new component-level tests
3. Add integration tests
4. Performance benchmarking

### Phase 5: Documentation (Week 3)
1. Complete all Google-style docstrings
2. Update developer documentation
3. Add architecture diagrams
4. Create migration guide for internal code

---

## Testing Strategy

### Unit Tests Structure

```
tests/unit/data/indexer_components/
├── test_index_store.py
├── test_query_builder.py
├── test_sample_manager.py
├── test_augmentation_tracker.py
├── test_processing_manager.py
└── test_parameter_normalizer.py

tests/unit/data/
├── test_indexer_public_api.py        # Existing API tests
├── test_indexer_augmentation.py      # Existing augmentation tests
└── test_indexer_integration.py       # Component integration tests
```

### Test Coverage Goals
- **Component tests**: 95%+ coverage
- **Integration tests**: Cover all public API methods
- **Performance tests**: Benchmark key operations
- **Backward compatibility**: All existing tests must pass

### New Test Categories

1. **Component Isolation Tests**
   - Each component tested independently
   - Mock dependencies
   - Test edge cases and error conditions

2. **Integration Tests**
   - Test component interactions
   - Verify data flow through components
   - Test complex scenarios

3. **Performance Regression Tests**
   - Benchmark before/after refactoring
   - Ensure no performance degradation
   - Target: Same or better performance

---

## Documentation Plan

### 1. **Complete Google-Style Docstrings**

All methods will have complete docstrings including:
- One-line summary
- Detailed description
- Args section with types and descriptions
- Returns section with type and description
- Raises section for exceptions
- Examples section with usage patterns

### 2. **Architecture Documentation**

Create `docs/developer/indexer_architecture.md`:
- Component diagram
- Data flow diagrams
- Responsibility matrix
- Design patterns used
- Extension points

### 3. **API Reference**

Update `docs/api/indexer.md`:
- All public methods documented
- Usage patterns and examples
- Migration guide from old patterns
- Best practices

---

## Backward Compatibility

### Guaranteed Compatibility

All public method signatures remain unchanged:
- `x_indices(selector, include_augmented=True)`
- `y_indices(selector, include_augmented=True)`
- `add_samples(...)`
- `add_samples_dict(...)`
- `register_samples(...)`
- `augment_rows(...)`
- `update_by_filter(...)`
- `get_column_values(...)`
- `uniques(...)`
- All other public methods

### Internal API Changes

Internal methods (prefixed with `_`) may change:
- `_append()` - remains but delegates to components
- `_apply_filters()` - replaced by QueryBuilder
- `_build_filter_condition()` - replaced by QueryBuilder
- Helper methods moved to ParameterNormalizer

### Property Access

Direct DataFrame access remains:
```python
indexer.df  # Still returns the polars DataFrame
```

---

## Benefits Summary

### Maintainability
- ✅ Clear separation of concerns
- ✅ Single responsibility per component
- ✅ Easier to understand and modify
- ✅ Reduced cognitive load

### Testability
- ✅ Components testable in isolation
- ✅ Better test coverage
- ✅ Easier to mock dependencies
- ✅ Faster test execution

### Performance
- ✅ Optimized polars operations
- ✅ Query expression caching
- ✅ Batch operations
- ✅ Lazy evaluation support

### Documentation
- ✅ Complete Google-style docstrings
- ✅ Clear component responsibilities
- ✅ Usage examples throughout
- ✅ Architecture documentation

### Extensibility
- ✅ Easy to add new components
- ✅ Clear extension points
- ✅ Alternative backends possible
- ✅ Pluggable components

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation:**
- Maintain all public APIs
- Comprehensive test suite
- Gradual rollout with feature flags

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark before/after
- Performance tests in CI
- Profile and optimize hot paths

### Risk 3: Increased Complexity
**Mitigation:**
- Clear component boundaries
- Excellent documentation
- Simple facade API

### Risk 4: Testing Overhead
**Mitigation:**
- Automated test generation
- Test fixtures and factories
- Parallel test execution

---

## Resolved Decisions

Based on team feedback and pragmatic considerations:

1. **Processing List Storage**: ✅ **Migrate to Polars `List(Utf8)` type**
   - Native Polars support eliminates `eval()` security/reliability concerns
   - Clean, type-safe operations
   - ~25 lines of code changes
   - No performance regression
   - Easy migration path

2. **Query Caching**: ❌ **Not implementing**
   - Query performance is not a bottleneck
   - Added complexity not justified
   - Keep implementation simple

3. **Thread Safety**: ❌ **Not needed**
   - Parallelization will use duplicated indexer instances
   - No shared state requirements
   - Simpler implementation

4. **Alternative Backends**: ℹ️ **Design for flexibility, implement as needed**
   - Current Polars implementation optimal for use case
   - Component architecture allows backend swap if needed
   - See "Backend Alternatives" section for detailed analysis

---

## Success Criteria

1. ✅ All existing tests pass
2. ✅ 95%+ test coverage on new components
3. ✅ No performance regression (within 5%)
4. ✅ All public APIs documented with Google-style docstrings
5. ✅ Component architecture documented
6. ✅ Code review approved by 2+ maintainers
7. ✅ Successfully runs all integration tests
8. ✅ Documentation reviewed and approved

---

## Backend Alternatives Analysis

The component architecture is designed to make backend replacement feasible if requirements change. Here's a detailed analysis of potential alternatives to Polars:

### Current: Polars DataFrame

**Characteristics:**
- In-memory columnar storage
- Rust-powered, highly optimized
- Rich query API with expressions
- No external dependencies (beyond Polars itself)

**Strengths:**
- ✅ Excellent performance for in-memory operations
- ✅ Low memory footprint compared to Pandas
- ✅ Clean API with method chaining
- ✅ Good for datasets up to millions of rows
- ✅ Zero setup - just import and use

**Limitations:**
- ❌ Not ideal for datasets larger than available RAM
- ❌ No persistence without manual save/load
- ❌ Limited multi-process sharing
- ❌ Smaller ecosystem than Pandas

**Best For:** Current nirs4all use case - moderate dataset sizes (thousands to hundreds of thousands of samples), in-memory processing, single-process workflows.

---

### Alternative 1: DuckDB

**Characteristics:**
- Embedded SQL analytical database
- Columnar storage with compression
- Can work with larger-than-memory datasets
- SQL query interface

**Implementation Approach:**
```python
class DuckDBIndexStore(IndexStore):
    def __init__(self, path: Optional[str] = None):
        self.conn = duckdb.connect(path or ':memory:')
        self._create_schema()

    def filter(self, condition: str) -> List[Dict]:
        # Convert Selector to SQL WHERE clause
        return self.conn.execute(f"SELECT * FROM samples WHERE {condition}").fetchall()
```

**Strengths:**
- ✅ Handles larger-than-memory datasets efficiently
- ✅ Persistent storage with ACID transactions
- ✅ SQL interface familiar to many users
- ✅ Excellent analytical query performance
- ✅ Can query Parquet files directly
- ✅ Zero-copy integration with Arrow/Polars

**Limitations:**
- ❌ SQL abstraction layer adds complexity
- ❌ Requires SQL query generation
- ❌ Less intuitive for simple operations
- ❌ Slightly higher overhead for small datasets

**When to Consider:**
- Datasets > 1M samples (larger than comfortable RAM)
- Need for persistent storage with transactions
- Integration with SQL-based tools/workflows
- Multiple processes need to access same index

**Migration Effort:** Medium (2-3 days)
- Create DuckDBIndexStore implementing IndexStore interface
- Convert Selector to SQL WHERE clauses in QueryBuilder
- Add connection management
- Test data persistence and recovery

---

### Alternative 2: SQLite

**Characteristics:**
- Lightweight SQL database
- Single-file storage
- ACID transactions
- Ubiquitous availability

**Strengths:**
- ✅ Zero configuration
- ✅ Persistent, reliable storage
- ✅ Works on any platform
- ✅ Multi-process safe with proper locking
- ✅ Battle-tested for decades

**Limitations:**
- ❌ Slower than Polars for analytical queries
- ❌ Row-oriented storage (less efficient for analytics)
- ❌ SQL overhead for simple operations
- ❌ Limited analytical functions

**When to Consider:**
- Need rock-solid persistence
- Multi-process access required
- Integration with existing SQLite-based systems
- Datasets too large for comfortable in-memory handling

**Migration Effort:** Medium (2-3 days)
- Similar to DuckDB but with more basic SQL features
- Need to implement analytical operations in Python

---

### Alternative 3: Pandas DataFrame

**Characteristics:**
- Most popular Python data analysis library
- Rich ecosystem and community
- Extensive functionality

**Strengths:**
- ✅ Massive ecosystem and community
- ✅ Familiar to most Python data scientists
- ✅ Rich integration with other libraries
- ✅ Extensive documentation and examples

**Limitations:**
- ❌ 2-5x slower than Polars for most operations
- ❌ Higher memory usage
- ❌ Some inconsistencies in API design
- ❌ Not as optimized for performance

**When to Consider:**
- Team strongly prefers Pandas familiarity
- Need specific Pandas-only functionality
- Integration with Pandas-heavy codebase

**Migration Effort:** Low (1-2 days)
- Very similar API to Polars
- Mostly drop-in replacement with minor adjustments
- Main changes in expression syntax

**Note:** Not recommended - Polars is objectively better for this use case.

---

### Alternative 4: PyArrow Table

**Characteristics:**
- Columnar in-memory format
- Zero-copy interoperability
- Part of Apache Arrow ecosystem

**Strengths:**
- ✅ Memory-efficient columnar storage
- ✅ Zero-copy data sharing
- ✅ Fast C++ implementation
- ✅ Standard format for data exchange
- ✅ Good integration with Polars/DuckDB

**Limitations:**
- ❌ Limited query API compared to Polars
- ❌ More verbose for complex operations
- ❌ Less feature-rich than Polars
- ❌ Requires more manual operation composition

**When to Consider:**
- Heavy integration with Arrow ecosystem
- Need zero-copy data exchange
- Memory constraints are critical
- Working with other Arrow-based tools

**Migration Effort:** Medium (2-3 days)
- Need to implement more query logic manually
- Polars can read/write Arrow, so hybrid approach possible

---

### Alternative 5: Custom Dict/List Structure

**Characteristics:**
- Native Python data structures
- Maximum simplicity and control
- No external dependencies

**Implementation Example:**
```python
class DictIndexStore:
    def __init__(self):
        self.rows = []  # List of dicts

    def filter(self, condition_func):
        return [row for row in self.rows if condition_func(row)]
```

**Strengths:**
- ✅ Zero dependencies
- ✅ Maximum control and transparency
- ✅ Simple to debug and understand
- ✅ No library-specific quirks

**Limitations:**
- ❌ No optimization - O(n) for everything
- ❌ High memory overhead
- ❌ Manual implementation of all operations
- ❌ Slow for datasets > 10k samples

**When to Consider:**
- Minimizing dependencies is critical
- Dataset is very small (< 1000 samples)
- Need maximum control over implementation

**Migration Effort:** High (4-5 days)
- Need to implement all filtering/querying logic
- Implement all aggregations manually
- No performance optimizations out of the box

**Note:** Not recommended for production use.

---

### Alternative 6: Redis / In-Memory Database

**Characteristics:**
- Distributed in-memory data store
- Key-value with data structures
- Pub/sub capabilities

**Strengths:**
- ✅ Distributed across machines
- ✅ Persistence options
- ✅ Pub/sub for event notifications
- ✅ Multi-process/multi-machine sharing

**Limitations:**
- ❌ Network overhead
- ❌ Complex setup and management
- ❌ Not optimized for analytical queries
- ❌ Significant operational complexity

**When to Consider:**
- Microservices architecture
- Multiple processes/machines need access
- Real-time updates across services
- Need pub/sub for index changes

**Migration Effort:** High (1-2 weeks)
- Completely different paradigm
- Need to design key structure
- Handle network failures and reconnection
- Deploy and manage Redis infrastructure

**Note:** Significant overkill for current use case.

---

### Recommendation: Polars → DuckDB Migration Path

The most realistic alternative to Polars is **DuckDB** for scenarios involving:
1. Datasets exceeding comfortable RAM (> 1M samples)
2. Need for persistence with ACID guarantees
3. Multi-process index access
4. Integration with SQL-based workflows

**Hybrid Approach** (Best of both worlds):
```python
class IndexStore:
    """Abstract base for different backends."""
    pass

class PolarsIndexStore(IndexStore):
    """Fast in-memory - current implementation."""
    pass

class DuckDBIndexStore(IndexStore):
    """Persistent, larger-than-memory."""
    pass

# Factory pattern for selection
def create_index_store(backend="polars", **kwargs):
    if backend == "polars":
        return PolarsIndexStore()
    elif backend == "duckdb":
        return DuckDBIndexStore(**kwargs)
```

**When to Implement:**
- User reports memory issues with large datasets
- Feature request for persistent index
- Clear use case for SQL access

**Estimated Effort:** 3-4 days for DuckDB backend implementation

---

### Backend Decision Matrix

| Backend | Speed | Memory | Persistence | Setup | Best For |
|---------|-------|--------|-------------|-------|----------|
| **Polars** (current) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | **Current use case** |
| **DuckDB** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Large datasets, persistence |
| **SQLite** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Multi-process, persistence |
| **Pandas** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Team familiarity |
| **PyArrow** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Arrow ecosystem |
| **Dict/List** | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | Tiny datasets only |
| **Redis** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | Distributed systems |

---

## Next Steps

1. **Review this proposal** with the team
2. **Approve architecture** and component design
3. **Create tracking issues** for each phase
4. **Set up branch** for refactoring work
5. **Begin Phase 1** implementation
6. **Regular sync meetings** to track progress

---

## Conclusion

This refactoring will transform the `Indexer` class from a monolithic class handling multiple concerns into a well-organized, maintainable, and performant system of focused components. The refactoring maintains complete backward compatibility while setting up a solid foundation for future enhancements.

The component-based architecture makes the code easier to understand, test, and extend, while the comprehensive documentation ensures that future developers can quickly grasp the system's design and make contributions confidently.

---

*Document Version: 1.0*
*Date: 2025-10-29*
*Author: GitHub Copilot with nirs4all team*
