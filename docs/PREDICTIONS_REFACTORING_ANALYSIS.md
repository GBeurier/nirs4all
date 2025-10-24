# Predictions Class Refactoring Analysis

## Redundancies Identified

### 1. Filtering Methods

**Existing Method: `filter_predictions()`** (line 844)
- Returns: `List[Dict[str, Any]]` with deserialized arrays
- Usage: 19 usages across codebase (io.py, prediction_analyzer.py, self-calls)
- Features:
  - Filters by dataset_name, partition, config_name, model_name, fold_id, step_idx
  - Deserializes JSON fields (y_true, y_pred, sample_indices, weights, metadata, best_params)
  - Returns list of dicts

**New Method: `filter_by_criteria()`** (line 2151)
- Returns: `pl.DataFrame`
- Usage: 1 test (test_phase4_query_reporting.py)
- Features:
  - Filters by dataset_name, date_range, metric_thresholds
  - Returns Polars DataFrame (no deserialization)

**Recommendation**:
- **KEEP** `filter_predictions()` - widely used, different return type
- **REFACTOR** `filter_by_criteria()` to use `filter_predictions()` internally for consistency
- Create internal `_apply_dataframe_filters()` helper for common filtering logic

---

### 2. Top/Best Selection Methods

**Existing Method: `top()`** (line 1282)
- Returns: `PredictionResultsList` (custom list of dicts)
- Usage: 3 usages (self-calls in predictions.py, prediction_analyzer.py)
- Features:
  - Ranks by metric on specific partition (default: val)
  - Displays metrics on different partition (default: test)
  - Computes metrics on-the-fly if needed
  - Supports fold-level grouping
  - Complex aggregation and ranking logic

**New Method: `query_best()`** (line 2117)
- Returns: `pl.DataFrame`
- Usage: 4 tests (test_phase4_query_reporting.py)
- Features:
  - Simple sort by metric column
  - Returns top N rows
  - Lightweight, no computation

**Recommendation**:
- **KEEP** `top()` - complex ranking logic, widely used
- **KEEP** `query_best()` - simple, lightweight, different use case
- **RENAME** `query_best()` → `get_best_by_metric()` to clarify difference from `top()`
- Add docstring note explaining when to use each:
  - Use `top()` for: Complex ranking, cross-partition analysis, metric computation
  - Use `get_best_by_metric()` for: Simple sorting of catalog metadata

---

## Refactoring Plan

### Phase 1: Internal Helper Methods
Create internal methods for common operations:

```python
def _apply_dataframe_filters(
    self,
    df: pl.DataFrame,
    dataset_name: Optional[str] = None,
    partition: Optional[str] = None,
    config_name: Optional[str] = None,
    model_name: Optional[str] = None,
    fold_id: Optional[str] = None,
    step_idx: Optional[int] = None,
    **kwargs
) -> pl.DataFrame:
    """Apply common filters to DataFrame."""
    # Consolidate filtering logic
```

### Phase 2: Refactor filter_by_criteria()
Make it use `_apply_dataframe_filters()`:

```python
def filter_by_criteria(
    self,
    dataset_name: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    metric_thresholds: Optional[Dict[str, float]] = None
) -> pl.DataFrame:
    """Filter predictions by multiple criteria (catalog-focused)."""
    df = self._apply_dataframe_filters(self._df, dataset_name=dataset_name)
    # Apply date_range and metric_thresholds
    return df
```

### Phase 3: Update filter_predictions()
Optionally use `_apply_dataframe_filters()` internally:

```python
def filter_predictions(
    self,
    dataset_name: Optional[str] = None,
    partition: Optional[str] = None,
    # ... other params
) -> List[Dict[str, Any]]:
    """Filter predictions and return as list of dictionaries."""
    df = self._apply_dataframe_filters(
        self._df,
        dataset_name=dataset_name,
        partition=partition,
        # ...
    )
    # Deserialize and return as list of dicts
    return self._deserialize_rows(df)
```

### Phase 4: Add Deserialization Helper
```python
def _deserialize_rows(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Deserialize JSON fields in DataFrame rows."""
    results = []
    for row in df.to_dicts():
        row["sample_indices"] = json.loads(row["sample_indices"])
        # ... other fields
        results.append(row)
    return results
```

### Phase 5: Rename and Document
- Rename `query_best()` → `get_best_by_metric()`
- Add clear docstrings explaining use cases
- Update tests

---

## Method Comparison Table

| Method | Return Type | Use Case | Deserialization | Computation |
|--------|-------------|----------|-----------------|-------------|
| `filter_predictions()` | `List[Dict]` | Get full prediction data for analysis | Yes | No |
| `filter_by_criteria()` | `pl.DataFrame` | Query catalog metadata | No | No |
| `top()` | `PredictionResultsList` | Rank models by computed metrics | Yes | Yes |
| `query_best()` | `pl.DataFrame` | Simple sort of catalog | No | No |

---

## Backward Compatibility

**CRITICAL**: `filter_predictions()` is used in 19 places:
- `nirs4all/pipeline/io.py` (1 usage)
- `nirs4all/dataset/predictions.py` (4 internal usages)
- `nirs4all/dataset/prediction_analyzer.py` (5 usages)

Any changes to `filter_predictions()` must maintain exact same signature and return type.

---

## Implementation Priority

1. **HIGH**: Create internal helpers (`_apply_dataframe_filters`, `_deserialize_rows`)
2. **MEDIUM**: Refactor `filter_by_criteria()` to use helpers
3. **MEDIUM**: Refactor `filter_predictions()` to use helpers (careful testing)
4. **LOW**: Rename `query_best()` → `get_best_by_metric()`
5. **LOW**: Update documentation

---

## Testing Requirements

- All 39 workspace tests must pass
- All existing tests must pass (no regression)
- New tests for internal helpers
- Update Phase 4 tests if renaming `query_best()`
