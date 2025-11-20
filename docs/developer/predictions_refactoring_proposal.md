# Predictions Class Refactoring Proposal

**Date**: October 29, 2025
**Version**: 0.4.1
**Status**: Proposal - Awaiting Review

---

## Executive Summary

The `predictions.py` module has grown to **2400+ lines** with multiple responsibilities mixed together, making it difficult to maintain, test, and extend. This proposal outlines a comprehensive refactoring to separate concerns into focused, testable components while maintaining backward compatibility with existing public APIs.

**Key Design Decisions** (resolved):
- **No pluggable backends** - Keep Polars-based storage
- **No async support** - Calls to predictions are rare
- **Hybrid serialization** - JSON for metadata (human-readable, external parsing), Parquet for arrays (performance)
- **No caching** - Unnecessary complexity for rare calls
- **No schema migration** - Maintain compatibility with current logic
- **Preserve public API** - All currently importable classes/functions stay accessible via `from nirs4all.data import ...`

### Key Problems Identified

1. **Mixed Responsibilities**: Data storage, serialization, ranking, filtering, visualization, and catalog management all in one class
2. **Inconsistent APIs**: Multiple methods doing similar things (`top()`, `top_k()`, `get_best()`)
3. **Performance Issues**: Repeated JSON serialization/deserialization for large arrays, inefficient filtering
4. **Poor Testability**: Tightly coupled components make unit testing difficult
5. **Documentation Gaps**: Missing or incomplete docstrings, inconsistent style
6. **Code Duplication**: Similar filtering logic repeated across methods

---

## Proposed Architecture

### Component Breakdown

```
nirs4all/data/
├── predictions.py                    # Main facade (public API preserved + public component functions)
└── predictions_components/
    ├── __init__.py                   # Re-exports for public API (e.g., PredictionResult, PredictionResultsList)
    ├── storage.py                    # Polars DataFrame storage backend
    ├── serializer.py                 # Hybrid: JSON for metadata, Parquet for arrays
    ├── indexer.py                    # Fast lookups and filtering (no caching)
    ├── ranker.py                     # Ranking and top-k operations
    ├── aggregator.py                 # Partition aggregation logic
    ├── query.py                      # Catalog query operations
    ├── result.py                     # PredictionResult & PredictionResultsList (public classes)
    └── schemas.py                    # DataFrame schema definitions
```

**Public API Design**:
- All currently importable classes/functions remain accessible via `from nirs4all.data import ...`
- Main facade: `Predictions` class in `predictions.py`
- Public result classes: `PredictionResult`, `PredictionResultsList` (from `predictions_components/result.py`)
- Components expose necessary public functions but internal implementation stays modular

---

## Detailed Component Design

### 1. `storage.py` - PredictionStorage

**Responsibility**: Core DataFrame operations and storage

```python
class PredictionStorage:
    """
    Low-level storage backend using Polars DataFrame.

    Handles:
    - DataFrame schema management
    - CRUD operations (add, filter, merge, clear)
    - File I/O (JSON, Parquet)
    - Schema validation
    """

    def __init__(self, schema: Dict[str, pl.DataType])
    def add_row(self, row_dict: Dict[str, Any]) -> str
    def add_rows(self, rows: List[Dict[str, Any]]) -> List[str]
    def filter(self, **criteria) -> pl.DataFrame
    def get_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]
    def merge(self, other: 'PredictionStorage', deduplicate: bool = False) -> None
    def clear(self) -> None
    def to_dataframe(self) -> pl.DataFrame
    def save_json(self, filepath: Path) -> None
    def load_json(self, filepath: Path) -> None
    def save_parquet(self, meta_path: Path, data_path: Path) -> None
    def load_parquet(self, meta_path: Path, data_path: Path) -> None
```

**Benefits**:
- Single source of truth for storage operations
- Easy to mock for testing
- Can swap backend (e.g., SQLite, DuckDB) without affecting other components

---

### 2. `serializer.py` - PredictionSerializer

**Responsibility**: Data serialization and deserialization

```python
class PredictionSerializer:
    """
    Handles all serialization/deserialization operations.

    Supports:
    - JSON encoding/decoding for metadata and simple fields
    - Parquet for array data (y_true, y_pred, sample_indices)
    - CSV export with metadata headers
    - Parquet split format (meta + data)
    - Hash generation for IDs

    Design Decision:
    - Metadata stays in JSON for human readability and external parsing
    - Arrays (y_true, y_pred, sample_indices) use Parquet for efficiency
    - Polars handles in-memory operations
    """

    @staticmethod
    def serialize_row(row: Dict[str, Any]) -> Dict[str, str]

    @staticmethod
    def deserialize_row(row: Dict[str, str]) -> Dict[str, Any]

    @staticmethod
    def generate_id(row: Dict[str, Any]) -> str

    @staticmethod
    def to_csv(predictions: List[Dict], filepath: Path, mode: str = "single") -> None

    @staticmethod
    def from_csv(filepath: Path) -> List[Dict[str, Any]]

    @staticmethod
    def save_arrays_parquet(
        predictions_df: pl.DataFrame,
        parquet_path: Path,
        array_columns: List[str] = ["y_true", "y_pred", "sample_indices"]
    ) -> None

    @staticmethod
    def load_arrays_parquet(parquet_path: Path) -> pl.DataFrame

    @staticmethod
    def numpy_to_bytes(arr: np.ndarray) -> bytes

    @staticmethod
    def bytes_to_numpy(data: bytes) -> np.ndarray
```

**Benefits**:
- Centralized serialization logic
- Hybrid format: JSON for metadata, Parquet for arrays
- Performance optimization with minimal code changes
- Easy to extend with new formats

---

### 3. `indexer.py` - PredictionIndexer

**Responsibility**: Fast filtering and lookup operations

```python
class PredictionIndexer:
    """
    Optimized filtering and indexing for predictions.

    Features:
    - Multi-column filtering (no caching - calls are rare)
    - Unique value extraction
    - Complex query building
    """

    def __init__(self, storage: PredictionStorage)

    def filter(
        self,
        dataset_name: Optional[str] = None,
        partition: Optional[str] = None,
        config_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fold_id: Optional[str] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> pl.DataFrame

    def get_unique_values(self, column: str) -> List[Any]

    def build_filter_expression(self, **criteria) -> pl.Expr

    def get_datasets(self) -> List[str]
    def get_partitions(self) -> List[str]
    def get_configs(self) -> List[str]
    def get_models(self) -> List[str]
    def get_folds(self) -> List[str]
```

**Benefits**:
- Separates filtering logic from storage
- Simple and direct filtering without caching overhead
- Clear interface for filter operations

---

### 4. `ranker.py` - PredictionRanker

**Responsibility**: Ranking and top-k selection

```python
class PredictionRanker:
    """
    Handles ranking predictions by metrics.

    Features:
    - Top-k selection with flexible ranking
    - Multi-metric scoring
    - Partition-aware ranking
    - Ascending/descending sort
    """

    def __init__(self, storage: PredictionStorage, serializer: PredictionSerializer)

    def top(
        self,
        n: int,
        rank_metric: str = "",
        rank_partition: str = "val",
        display_partition: str = "test",
        ascending: bool = True,
        group_by_fold: bool = False,
        **filters
    ) -> PredictionResultsList

    def get_best(
        self,
        metric: str = "",
        ascending: bool = True,
        **filters
    ) -> Optional[PredictionResult]

    def _compute_rank_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str
    ) -> float

    def _build_model_key(
        self,
        row: Dict[str, Any],
        group_by_fold: bool
    ) -> Tuple[str, ...]
```

**Benefits**:
- Consolidates ranking logic
- Clear separation from storage/filtering
- Easy to add new ranking strategies

---

### 5. `aggregator.py` - PartitionAggregator

**Responsibility**: Partition data aggregation

```python
class PartitionAggregator:
    """
    Aggregates prediction data across partitions.

    Features:
    - Train/val/test partition combining
    - Nested dictionary structure creation
    - Metadata preservation
    """

    def __init__(self, storage: PredictionStorage, indexer: PredictionIndexer)

    def aggregate_partitions(
        self,
        prediction_id: str,
        partitions: List[str] = ["train", "val", "test"]
    ) -> Dict[str, Dict[str, Any]]

    def add_partition_data(
        self,
        results: List[PredictionResult],
        partitions: List[str]
    ) -> List[PredictionResult]

    def merge_partition_arrays(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ) -> Dict[str, Dict]
```

**Benefits**:
- Isolated partition logic
- Reusable across different contexts
- Clear data flow

---

### 6. `query.py` - CatalogQueryEngine

**Responsibility**: Catalog-specific query operations

```python
class CatalogQueryEngine:
    """
    High-level query operations for catalog browsing.

    Features:
    - Best pipeline queries
    - Cross-dataset comparisons
    - Run listing and summary
    - Metric-based filtering
    """

    def __init__(self, storage: PredictionStorage, indexer: PredictionIndexer)

    def query_best(
        self,
        dataset_name: Optional[str] = None,
        metric: str = "test_score",
        n: int = 10,
        ascending: bool = False
    ) -> pl.DataFrame

    def filter_by_criteria(
        self,
        dataset_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        metric_thresholds: Optional[Dict[str, float]] = None
    ) -> pl.DataFrame

    def compare_across_datasets(
        self,
        pipeline_hash: str,
        metric: str = "test_score"
    ) -> pl.DataFrame

    def list_runs(
        self,
        dataset_name: Optional[str] = None
    ) -> pl.DataFrame

    def get_summary_stats(
        self,
        metric: str = "test_score"
    ) -> Dict[str, float]
```

**Benefits**:
- Separates catalog from core predictions
- Clear interface for reporting
- Can be extended with advanced analytics

---

### 7. `result.py` - Result Classes

**Responsibility**: User-facing result containers

```python
class PredictionResult(dict):
    """
    Enhanced dictionary for a single prediction with convenience methods.

    Features:
    - Property accessors (id, model_name, dataset_name, etc.)
    - save_to_csv() - save individual result
    - eval_score() - compute metrics on-the-fly
    - summary() - generate tab report
    """

    # Keep existing implementation but improve docstrings


class PredictionResultsList(list):
    """
    List container for PredictionResult objects with batch operations.

    Features:
    - save() - batch CSV export
    - get() - retrieve by ID
    - filter() - chain filtering
    - Iterator support
    """

    # Keep existing implementation but improve docstrings
```

**Benefits**:
- User-friendly interface
- Chainable operations
- Clear API surface

---

### 8. `schemas.py` - Schema Definitions

**Responsibility**: DataFrame schema constants

```python
# Centralized schema definitions
PREDICTION_SCHEMA = {
    "id": pl.Utf8,
    "dataset_name": pl.Utf8,
    "dataset_path": pl.Utf8,
    "config_name": pl.Utf8,
    "config_path": pl.Utf8,
    "pipeline_uid": pl.Utf8,
    "step_idx": pl.Int64,
    "op_counter": pl.Int64,
    "model_name": pl.Utf8,
    "model_classname": pl.Utf8,
    "model_path": pl.Utf8,
    "fold_id": pl.Utf8,
    "sample_indices": pl.Utf8,
    "weights": pl.Utf8,
    "metadata": pl.Utf8,
    "partition": pl.Utf8,
    "y_true": pl.Utf8,
    "y_pred": pl.Utf8,
    "val_score": pl.Float64,
    "test_score": pl.Float64,
    "train_score": pl.Float64,
    "metric": pl.Utf8,
    "task_type": pl.Utf8,
    "n_samples": pl.Int64,
    "n_features": pl.Int64,
    "preprocessings": pl.Utf8,
    "best_params": pl.Utf8,
}

# Metadata-only schema for catalog queries
METADATA_SCHEMA = {...}

# Array-only schema for data storage
ARRAY_SCHEMA = {...}
```

---

## Refactored Main Class

### `predictions.py` - Predictions (Facade)

```python
class Predictions:
    """
    Main facade for prediction management.

    Delegates to specialized components while maintaining
    backward-compatible public API.

    Architecture:
        - Storage: PredictionStorage (DataFrame backend)
        - Serializer: PredictionSerializer (I/O operations)
        - Indexer: PredictionIndexer (filtering)
        - Ranker: PredictionRanker (top-k selection)
        - Aggregator: PartitionAggregator (partition combining)
        - Query: CatalogQueryEngine (catalog operations)

    Examples:
        >>> # Create and add predictions
        >>> pred = Predictions()
        >>> pred.add_prediction(
        ...     dataset_name="wheat",
        ...     model_name="PLS_10",
        ...     y_true=y_true,
        ...     y_pred=y_pred
        ... )

        >>> # Query top models
        >>> top_5 = pred.top(n=5, rank_metric="rmse", rank_partition="val")
        >>> top_5[0].save_to_csv("best_model.csv")

        >>> # Save and load
        >>> pred.save_to_file("predictions.json")
        >>> loaded = Predictions.load("predictions.json")
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize Predictions with optional file loading.

        Args:
            filepath: Optional path to load predictions from (JSON or Parquet).
        """
        self._storage = PredictionStorage(PREDICTION_SCHEMA)
        self._serializer = PredictionSerializer()
        self._indexer = PredictionIndexer(self._storage)
        self._ranker = PredictionRanker(self._storage, self._serializer)
        self._aggregator = PartitionAggregator(self._storage, self._indexer)
        self._query = CatalogQueryEngine(self._storage, self._indexer)

        if filepath:
            self.load_from_file(filepath)

    # ========== Public API (Preserved) ==========

    def add_prediction(self, **kwargs) -> str:
        """Add single prediction (delegate to storage)."""
        # Serialize numpy arrays
        serialized = self._serializer.serialize_row(kwargs)
        # Generate ID
        prediction_id = self._serializer.generate_id(serialized)
        serialized['id'] = prediction_id
        # Store
        self._storage.add_row(serialized)
        return prediction_id

    def add_predictions(self, **kwargs) -> None:
        """Add multiple predictions (delegate to storage)."""
        # Implementation delegates to storage
        pass

    def filter_predictions(self, **kwargs) -> List[Dict[str, Any]]:
        """Filter predictions (delegate to indexer + serializer)."""
        df = self._indexer.filter(**kwargs)
        rows = df.to_dicts()
        return [self._serializer.deserialize_row(r) for r in rows]

    def top(self, n: int, **kwargs) -> PredictionResultsList:
        """Get top N predictions (delegate to ranker)."""
        return self._ranker.top(n, **kwargs)

    def get_best(self, **kwargs) -> Optional[PredictionResult]:
        """Get best prediction (delegate to ranker)."""
        return self._ranker.get_best(**kwargs)

    # Deprecated but kept for compatibility
    def top_k(self, k: int = 5, **kwargs) -> PredictionResultsList:
        """DEPRECATED: Use top() instead."""
        warnings.warn("top_k() is deprecated, use top() instead", DeprecationWarning)
        return self.top(n=k, **kwargs)

    # ... other public methods delegate similarly
```

---

## Migration Strategy

### Phase 1: Component Extraction (Week 1)

1. Create `predictions_components/` directory structure
2. Extract `PredictionStorage` and `PredictionSerializer` (lowest dependencies)
   - Implement hybrid serialization: JSON for metadata, Parquet for arrays (`y_true`, `y_pred`, `sample_indices`)
3. Update imports in `predictions.py`, keep all logic inline
4. Ensure public classes (`PredictionResult`, `PredictionResultsList`) remain importable from `nirs4all.data`
5. Run existing tests - **no breakage expected**

### Phase 2: Indexer and Ranker (Week 2)

1. Extract `PredictionIndexer` (filtering logic, no caching)
2. Extract `PredictionRanker` (top/top_k/get_best logic)
3. Update `Predictions` to delegate to these components
4. Run tests - verify behavior unchanged

### Phase 3: Aggregator and Query (Week 3)

1. Extract `PartitionAggregator`
2. Extract `CatalogQueryEngine`
3. Update `Predictions` to use these components
4. Run tests

### Phase 4: Documentation and Deprecation (Week 4)

1. Add comprehensive Google-style docstrings to all components
2. Mark deprecated methods (`top_k`, redundant helpers)
3. Update user documentation
4. Create migration guide for advanced users

### Phase 5: Test Enhancement (Ongoing)

1. Write unit tests for each component in isolation
2. Add integration tests for common workflows
3. Achieve >85% code coverage

---

## API Changes and Deprecations

### Kept (Unchanged)

✅ `add_prediction()` - Core functionality
✅ `add_predictions()` - Batch add
✅ `filter_predictions()` - General filtering
✅ `top()` - Primary ranking method
✅ `get_best()` - Get single best
✅ `save_to_file()` / `load_from_file()` - Persistence
✅ `merge_predictions()` - Combining results
✅ `num_predictions` property
✅ `get_datasets()`, `get_models()`, etc. - Convenience methods
✅ `PredictionResult` class
✅ `PredictionResultsList` class

### Deprecated (Still Work, Warning)

⚠️ `top_k()` → Use `top()` instead
⚠️ `get_similar()` → Use `filter_predictions()` + first
⚠️ `_apply_dataframe_filters()` → Internal only
⚠️ `_deserialize_rows()` → Internal only
⚠️ `_add_partition_data()` → Internal only

### New (Enhanced Functionality)

✨ Component-based architecture for easier testing
✨ Performance improvements from centralized operations
✨ Better error messages and validation
✨ Consistent docstring style

---

## Testing Strategy

### Unit Tests

Each component gets isolated unit tests:

```
tests/unit/predictions_components/
├── test_storage.py
├── test_serializer.py
├── test_indexer.py
├── test_ranker.py
├── test_aggregator.py
├── test_query.py
└── test_result.py
```

**Example** (`test_ranker.py`):

```python
def test_top_returns_correct_count():
    storage = PredictionStorage(PREDICTION_SCHEMA)
    serializer = PredictionSerializer()
    ranker = PredictionRanker(storage, serializer)

    # Add 10 predictions
    for i in range(10):
        storage.add_row({...})

    # Get top 5
    results = ranker.top(n=5, rank_metric="rmse")

    assert len(results) == 5
    assert isinstance(results, PredictionResultsList)
```

### Integration Tests

Keep existing integration tests, ensure they pass:

```
tests/integration/
├── test_predictions_workflow.py
├── test_pipeline_predictions.py
└── test_catalog_queries.py
```

### Performance Tests

Add benchmarks for key operations:

```python
def test_top_performance_with_1000_predictions():
    pred = Predictions()
    # Add 1000 predictions
    ...

    start = time.time()
    top_10 = pred.top(10, rank_metric="rmse")
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should complete in < 1 second
```

---

## Documentation Updates

### Module Docstrings

Each component gets comprehensive module-level documentation:

```python
"""
Prediction ranking and top-k selection.

This module provides the PredictionRanker class for ranking and selecting
top-performing models from a prediction storage backend.

Features:
    - Top-k selection by arbitrary metrics
    - Partition-aware ranking (train/val/test)
    - Fold-grouped or cross-fold ranking
    - Ascending/descending sort orders

Examples:
    >>> storage = PredictionStorage(PREDICTION_SCHEMA)
    >>> ranker = PredictionRanker(storage, serializer)
    >>>
    >>> # Get top 5 models by validation RMSE
    >>> top_5 = ranker.top(
    ...     n=5,
    ...     rank_metric="rmse",
    ...     rank_partition="val",
    ...     dataset_name="wheat"
    ... )
    >>>
    >>> # Get single best model
    >>> best = ranker.get_best(metric="r2", ascending=False)

Classes:
    PredictionRanker: Main ranking engine

See Also:
    - PredictionStorage: Backend storage
    - PredictionIndexer: Filtering operations
    - PartitionAggregator: Partition data handling
"""
```

### Google-Style Docstrings

Every public method gets complete docstrings:

```python
def top(
    self,
    n: int,
    rank_metric: str = "",
    rank_partition: str = "val",
    display_partition: str = "test",
    ascending: bool = True,
    group_by_fold: bool = False,
    **filters
) -> PredictionResultsList:
    """
    Get top n models ranked by a metric on a specific partition.

    Ranks models by performance on `rank_partition`, then returns
    their data from `display_partition`. Useful for validation-based
    selection with test set evaluation.

    Args:
        n: Number of top models to return.
        rank_metric: Metric to rank by. If empty, uses stored score column
            (val_score/test_score). Examples: "rmse", "r2", "accuracy".
        rank_partition: Partition to compute ranking scores from.
            One of ["train", "val", "test"]. Default: "val".
        display_partition: Partition to return prediction data from.
            One of ["train", "val", "test"]. Default: "test".
        ascending: Sort order. True for ascending (lower is better),
            False for descending (higher is better). Default: True.
        group_by_fold: If True, treat each fold as a separate model.
            If False, aggregate across folds. Default: False.
        **filters: Additional filter criteria (dataset_name, config_name, etc.).

    Returns:
        PredictionResultsList containing top n models, sorted by rank_metric.
        Each result is a PredictionResult with prediction data from
        display_partition.

    Raises:
        ValueError: If rank_partition or display_partition is invalid.
        KeyError: If rank_metric is not computable for the data.

    Examples:
        >>> # Get top 5 models by validation RMSE, show test results
        >>> top_5 = pred.top(
        ...     n=5,
        ...     rank_metric="rmse",
        ...     rank_partition="val",
        ...     display_partition="test"
        ... )
        >>>
        >>> # Get top 3 models by test R2 (higher is better)
        >>> top_3 = pred.top(
        ...     n=3,
        ...     rank_metric="r2",
        ...     rank_partition="test",
        ...     ascending=False
        ... )
        >>>
        >>> # Filter by dataset before ranking
        >>> wheat_top = pred.top(
        ...     n=10,
        ...     rank_metric="mae",
        ...     dataset_name="wheat"
        ... )

    See Also:
        - get_best(): Get single best model
        - filter_predictions(): Pre-filter before ranking
        - PredictionResult.eval_score(): Compute additional metrics

    Notes:
        - If rank_metric matches the stored metric in the data, uses
          precomputed scores for efficiency.
        - For custom metrics, computes on-the-fly from y_true/y_pred.
        - Automatically handles metric directionality (RMSE lower is better,
          R2 higher is better).
    """
    # Implementation
    pass
```

---

## Performance Optimizations

### 1. Lazy Deserialization

Only deserialize arrays when accessed:

```python
class LazyPredictionResult(PredictionResult):
    """PredictionResult with lazy array deserialization."""

    def __init__(self, row: Dict[str, str], serializer: PredictionSerializer):
        self._raw = row
        self._serializer = serializer
        self._deserialized = {}

    @property
    def y_true(self) -> np.ndarray:
        if 'y_true' not in self._deserialized:
            self._deserialized['y_true'] = self._serializer.json_to_numpy(
                self._raw['y_true']
            )
        return self._deserialized['y_true']
```

### 2. Caching Common Queries

~~Removed - caching not needed for rare prediction calls~~

### 3. Batch Operations

Optimize batch adds:

```python
def add_predictions(self, **kwargs) -> List[str]:
    """Add multiple predictions in a single batch."""
    rows = self._prepare_batch_rows(**kwargs)

    # Vectorized serialization
    serialized = [self._serializer.serialize_row(r) for r in rows]

    # Batch ID generation
    ids = [self._serializer.generate_id(r) for r in serialized]
    for i, row in enumerate(serialized):
        row['id'] = ids[i]

    # Single DataFrame concat instead of multiple
    self._storage.add_rows(serialized)

    return ids
```

---

## Error Handling Improvements

### Validation

Add input validation at component boundaries:

```python
class PredictionRanker:
    def top(self, n: int, rank_partition: str = "val", **kwargs):
        # Validate inputs
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        valid_partitions = ["train", "val", "test"]
        if rank_partition not in valid_partitions:
            raise ValueError(
                f"rank_partition must be one of {valid_partitions}, "
                f"got '{rank_partition}'"
            )

        # Continue with logic...
```

### Better Error Messages

```python
def get_best(self, metric: str = "", **filters):
    results = self.top(n=1, rank_metric=metric, **filters)

    if not results:
        filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items())
        raise ValueError(
            f"No predictions found matching criteria: {filter_desc}. "
            f"Available datasets: {self._indexer.get_datasets()}"
        )

    return results[0]
```

---

## Backward Compatibility

### Deprecation Warnings

```python
import warnings

class Predictions:
    def top_k(self, k: int = 5, **kwargs):
        warnings.warn(
            "top_k() is deprecated and will be removed in v0.6.0. "
            "Use top(n=k, ...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.top(n=k, **kwargs)
```

### Shim Layer

For internal methods that may be used by advanced users:

```python
def _apply_dataframe_filters(self, df: pl.DataFrame, **kwargs):
    """INTERNAL: Use PredictionIndexer.filter() instead."""
    warnings.warn(
        "_apply_dataframe_filters() is internal API and may change. "
        "Use filter_predictions() instead.",
        FutureWarning,
        stacklevel=2
    )
    return self._indexer.filter(**kwargs)
```

---

## Risk Assessment

### Low Risk

✅ Component extraction (storage, serializer) - no logic changes
✅ Adding docstrings - only documentation
✅ Adding tests - quality improvement

### Medium Risk

⚠️ Refactoring `top()`/`top_k()` - complex logic, heavily tested
⚠️ Partition aggregation changes - intricate data flow
⚠️ Array serialization to Parquet - must preserve exact behavior and compatibility

### High Risk

❌ Changing DataFrame schema - breaks persistence
❌ Removing public methods - breaks user code
❌ Breaking backward compatibility - must maintain all currently importable public API
❌ Performance regressions - must benchmark

### Mitigation

1. **Comprehensive test suite** before starting refactoring
2. **Incremental rollout** - one component at a time
3. **Benchmark suite** to catch performance regressions
4. **Beta period** with deprecation warnings before removal
5. **Migration guide** for advanced users (if needed)
6. **Verify public API compatibility** - ensure all currently importable classes/functions remain accessible

---

## Success Metrics

### Code Quality

- [ ] Lines of code in `predictions.py`: **2400 → <800**
- [ ] Average method length: **50 lines → <20 lines**
- [ ] Cyclomatic complexity: **High → Low**
- [ ] Test coverage: **~60% → >85%**

### Maintainability

- [ ] All public methods have complete docstrings
- [ ] All components independently testable
- [ ] Clear separation of concerns
- [ ] Easy to add new features (e.g., new storage backend)

### Performance

- [ ] Top-k with 1000 predictions: **<1s**
- [ ] Filter 10,000 predictions: **<500ms**
- [ ] Save/load 5000 predictions: **<2s**

### Compatibility

- [ ] All existing tests pass without modification
- [ ] No breaking changes to public API
- [ ] Deprecation warnings for old methods
- [ ] Migration guide available

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Storage & Serializer | 3 days | `storage.py`, `serializer.py`, `schemas.py` |
| Phase 2: Indexer & Ranker | 4 days | `indexer.py`, `ranker.py`, updated `Predictions` |
| Phase 3: Aggregator & Query | 3 days | `aggregator.py`, `query.py` |
| Phase 4: Documentation | 3 days | Complete docstrings, migration guide |
| Phase 5: Testing | 5 days | Unit + integration tests, >85% coverage |
| **Total** | **18 days** | Fully refactored, tested, documented module |

---

## Next Steps

1. ✅ **Review this proposal** - Feedback received and integrated
2. ✅ **Approve architecture** - Design decisions resolved
3. **Set up branch** - Create `refactor/predictions-0.4.1` branch
4. **Write baseline tests** - Ensure current behavior captured
5. **Begin Phase 1** - Extract storage and serializer components (with Parquet for arrays)

---

## Appendix A: Example Usage After Refactoring

### Simple Workflow

```python
from nirs4all.data.predictions import Predictions

# Create predictions
pred = Predictions()

# Add predictions (unchanged API)
pred.add_prediction(
    dataset_name="wheat",
    model_name="PLS_10",
    partition="test",
    y_true=y_true,
    y_pred=y_pred,
    test_score=0.85
)

# Query top models (cleaner API)
top_5 = pred.top(
    n=5,
    rank_metric="rmse",
    rank_partition="val",
    display_partition="test"
)

# Access results (unchanged)
best = top_5[0]
print(f"Best model: {best.model_name}, RMSE: {best.eval_score(['rmse'])}")
best.save_to_csv("best_model.csv")

# Save (unchanged)
pred.save_to_file("predictions.json")
```

### Advanced Usage (Components)

```python
from nirs4all.data.predictions import Predictions
from nirs4all.data._predictions import PredictionIndexer, PredictionRanker

# For advanced users who need fine-grained control
pred = Predictions()

# Direct component access
indexer = pred._indexer
filtered_df = indexer.filter(dataset_name="wheat", model_name="PLS")

# Custom ranking logic
ranker = pred._ranker
custom_results = ranker.top(
    n=10,
    rank_metric="custom_metric",
    group_by_fold=True
)
```

---

## Appendix B: File Structure After Refactoring

```
nirs4all/data/
├── predictions.py                          # 800 lines (down from 2400)
│   └── class Predictions (facade)
│   └── Public functions preserved for direct import
│
└── predictions_components/
    ├── __init__.py                         # Re-exports public classes for backward compatibility
    │   └── from .result import PredictionResult, PredictionResultsList
    │   └── from .serializer import PredictionSerializer (if needed publicly)
    │
    ├── storage.py                          # ~200 lines
    │   └── class PredictionStorage
    │
    ├── serializer.py                       # ~250 lines
    │   └── class PredictionSerializer
    │       └── Hybrid: JSON for metadata, Parquet for arrays
    │
    ├── indexer.py                          # ~150 lines
    │   └── class PredictionIndexer (no caching)
    │
    ├── ranker.py                           # ~300 lines
    │   └── class PredictionRanker
    │
    ├── aggregator.py                       # ~150 lines
    │   └── class PartitionAggregator
    │
    ├── query.py                            # ~200 lines
    │   └── class CatalogQueryEngine
    │
    ├── result.py                           # ~300 lines (PUBLIC)
    │   └── class PredictionResult
    │   └── class PredictionResultsList
    │
    └── schemas.py                          # ~50 lines
        └── PREDICTION_SCHEMA
        └── METADATA_SCHEMA
        └── ARRAY_SCHEMA
```

**Total**: ~2430 lines → ~2400 lines (same total, better organized)

**Public API Access**:
```python
# All these imports continue to work
from nirs4all.data import Predictions, PredictionResult, PredictionResultsList
from nirs4all.data.predictions import Predictions
from nirs4all.data._predictions import PredictionResult, PredictionResultsList
```

---

## Appendix C: Docstring Template

### Function/Method Docstring

```python
def method_name(
    self,
    arg1: Type1,
    arg2: Optional[Type2] = None,
    *args,
    **kwargs
) -> ReturnType:
    """
    One-line summary of what this method does.

    More detailed explanation if needed. Can span multiple lines.
    Explain the purpose, behavior, and any important details.

    Args:
        arg1: Description of arg1. What it's used for, valid values, etc.
        arg2: Description of arg2. Mention default behavior if None.
        *args: Description of variadic positional arguments.
        **kwargs: Description of keyword arguments. Common keys:
            - key1: Description of key1
            - key2: Description of key2

    Returns:
        Description of return value. Type information, structure,
        special values, etc.

    Raises:
        ValueError: When arg1 is invalid or out of range.
        KeyError: When required key missing in kwargs.
        RuntimeError: When operation cannot be completed.

    Examples:
        >>> # Example 1: Basic usage
        >>> result = obj.method_name(value1, value2)
        >>>
        >>> # Example 2: With optional arguments
        >>> result = obj.method_name(value1, arg2=custom_value)
        >>>
        >>> # Example 3: Edge case
        >>> result = obj.method_name([], None)

    See Also:
        - related_method(): Brief description of relationship
        - OtherClass.method(): Cross-reference to related functionality

    Notes:
        - Important implementation detail or caveat
        - Performance consideration
        - Deprecated/alternative approach

    Warning:
        Critical warning about usage (e.g., data loss, side effects)
    """
    pass
```

### Class Docstring

```python
class ClassName:
    """
    One-line summary of what this class represents.

    Longer description of the class purpose, responsibilities,
    and how it fits into the larger system.

    Attributes:
        attr1: Description of public attribute 1.
        attr2: Description of public attribute 2.

    Examples:
        >>> # Basic usage
        >>> obj = ClassName(arg1, arg2)
        >>> result = obj.method()
        >>>
        >>> # Advanced usage
        >>> obj = ClassName.from_file("path.json")
        >>> obj.process()
        >>> obj.save()

    See Also:
        - RelatedClass: How it relates to this class
        - module.function(): Related functionality

    Notes:
        - Thread safety: This class is/is not thread-safe
        - Immutability: Instances are/are not immutable
        - Inheritance: Designed for inheritance or final
    """

    def __init__(self, arg1: Type1, arg2: Type2):
        """
        Initialize ClassName with required parameters.

        Args:
            arg1: Description of arg1.
            arg2: Description of arg2.

        Raises:
            ValueError: If arguments are invalid.
        """
        pass
```

---

## Questions & Discussion

### Open Questions - RESOLVED

1. **Storage Backend**: ✅ **NO** - No need for pluggable backends (SQLite, DuckDB). Keep current Polars approach.

2. **Async Support**: ✅ **NO** - Async I/O not needed. Calls to predictions are rare.

3. **Serialization Format**: ✅ **HYBRID APPROACH**
   - **Parquet for arrays**: `y_true`, `y_pred`, and `sample_indices` stored in Parquet format for performance
   - **JSON for metadata**: Keep JSON for human-readable metadata and external parsing compatibility
   - **Polars for operations**: Continue using Polars DataFrame for in-memory operations
   - **Rationale**: JSON is good for metadata and external interop, but storing arrays (especially large prediction vectors) in JSON is inefficient

4. **Caching Strategy**: ✅ **NO CACHING** - Remove caching from `PredictionIndexer`. Calls to predictions are rare enough that caching adds unnecessary complexity without meaningful performance benefit.

5. **Versioning**: ✅ **NO SCHEMA MIGRATION** - Refactoring maintains compatibility with current schema logic (except Parquet optimization for arrays). No migration handling needed.

### Discussion Points - RESOLVED

- **Component Granularity**: ✅ **APPROVED** - Component structure is well-designed

- **Performance Targets**: ✅ **APPROVED** - <1s top-k and <500ms filter targets are realistic and acceptable

- **Test Coverage Goal**: ✅ **85% TARGET** - 85% coverage is perfect if critical and systematic paths are tested. Focus on high-value test coverage rather than arbitrary 100%.

- **Migration Timeline**: ✅ **FLEXIBLE** - 18-day timeline is a guideline, actual duration is flexible

- **Public API**: ✅ **PRESERVE CURRENT PUBLIC API** - Keep all currently public functions accessible via `from nirs4all.data import ...`. Components should expose public functions that are currently importable directly, not just through the `Predictions` facade.

---

**End of Proposal**

*This proposal has been reviewed and approved for implementation. All open questions resolved and design decisions finalized. Ready to proceed with refactoring.*

---

## Changelog

**October 29, 2025** - Initial proposal created
**October 29, 2025** - Feedback integrated:
  - ✅ No pluggable backends, no async support
  - ✅ Hybrid serialization: JSON for metadata, Parquet for arrays
  - ✅ No caching in indexer
  - ✅ No schema migration needed
  - ✅ 85% test coverage target approved
  - ✅ Public API preservation confirmed - all currently importable classes/functions remain accessible
  - ✅ Proposal approved for implementation
