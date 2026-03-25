# Audit: Metadata-Based Prediction Aggregation

**Issue**: #36 -- Allowing metadata usage for prediction management
**Related**: #37 -- DuckDB locking on successive runs (potential SQLite migration)
**Date**: 2026-03-25

---

## 1. Current State

### 1.1 How Predictions Are Created

During pipeline execution, the model controller (`controllers/models/base_model.py`) assembles prediction data including per-partition metadata. The method `_build_fold_prediction_data` (line ~1988) explicitly extracts dataset metadata for each partition (train, val, test):

```python
# base_model.py ~line 2011
for partition_name in ['train', 'val', 'test']:
    meta_df = dataset.metadata({"partition": partition_name})
    metadata_dict = {}
    for col in meta_df.columns:
        col_data = meta_df[col].to_numpy()
        metadata_dict[col] = col_data.tolist()
    partition_metadata[partition_name] = metadata_dict
```

This metadata is then passed to `prediction_store.add_prediction(..., metadata=metadata, ...)` in `_add_all_predictions` (line ~2106).

### 1.2 How Metadata Flows Into the Buffer

`Predictions.add_prediction()` calls `_build_prediction_row()` which stores metadata as a plain dict in the buffer row:

```python
# predictions.py line 152
"metadata": metadata if metadata is not None else {},
```

The metadata dict has the structure `{column_name: [value_per_sample, ...]}` -- one list entry per prediction sample.

### 1.3 How Metadata Is Used for Aggregation (In-Memory Only)

The `_apply_aggregation` method (line 1070) reads `metadata` from the buffer row and uses it to group predictions:

```python
# predictions.py line 1102-1112
if by_repetition not in metadata:
    warnings.warn(f"Aggregation column '{by_repetition}' not found in metadata...")
    return y_true, y_pred, y_proba, False
group_ids = np.asarray(metadata[by_repetition])
```

This works correctly **within a single pipeline execution** because the metadata lives in the in-memory buffer alongside y_true/y_pred arrays.

### 1.4 How Predictions Are Persisted

When `Predictions.flush()` is called, two things happen:

1. **DuckDB metadata row** via `WorkspaceStore.save_prediction()` -- stores scalar fields (scores, model name, fold_id, partition, etc.). The `metadata` dict is **not passed** to `save_prediction()` at all. The method signature has no metadata parameter.

2. **Parquet array row** via `ArrayStore.save_batch()` -- stores y_true, y_pred, y_proba, sample_indices, weights. The Parquet schema (`_PARQUET_SCHEMA` in `array_store.py` line 72) has **no metadata columns**.

### 1.5 How Predictions Are Loaded Back

When loading from a workspace (`Predictions._populate_buffer_from_store`), the buffer is rebuilt from DuckDB query results + Parquet arrays. Since metadata was never persisted, the reconstructed buffer rows have **empty metadata dicts**.

### 1.6 Storage Schema Summary

**DuckDB `predictions` table** (`store_schema.py` line 96):
- prediction_id, pipeline_id, chain_id, dataset_name, model_name, model_class
- fold_id, partition, val_score, test_score, train_score, metric, task_type
- n_samples, n_features, scores (JSON), best_params (JSON)
- preprocessings, branch_id, branch_name, exclusion_count, exclusion_rate
- refit_context, created_at
- **No metadata column.**

**Parquet schema** (`array_store.py` line 72):
- prediction_id, dataset_name, model_name, fold_id, partition, metric, val_score, task_type
- y_true, y_pred, y_proba, y_proba_shape, sample_indices, weights
- **No metadata column.**

---

## 2. Gap Analysis

### 2.1 The Core Problem

Metadata is available in-memory during pipeline execution but is **discarded on persistence**. Once predictions are flushed to storage, the link between individual prediction samples and their dataset metadata is permanently lost.

This means:
- `predictions.top(5, by_repetition="Sample_ID")` works **during** a pipeline run (metadata in buffer).
- `predictions.top(5, by_repetition="Sample_ID")` **fails silently** when loading from a workspace (metadata missing, aggregation skipped with warning).
- Any post-hoc aggregation by a metadata column that wasn't the pipeline's repetition column is impossible.

### 2.2 Where Metadata Is Lost

The loss occurs at exactly two points:

1. **`Predictions.flush()`** (line 679) -- does not include `row["metadata"]` in the data sent to either `save_prediction` or `save_batch`.

2. **`Predictions._populate_buffer_from_store()`** (line 322) -- when rebuilding the buffer from stored data, no metadata is available to restore.

### 2.3 The `sample_indices` Partial Bridge

The Parquet schema does store `sample_indices` (integer indices of samples within a partition). In theory, this could be used to re-join predictions with the original dataset's metadata at query time -- but only if:
- The original dataset is still available and unchanged.
- The index semantics are stable (i.e., the indices correspond to rows in the dataset's partition).
- The caller provides the dataset explicitly.

Currently, no such re-joining mechanism exists.

### 2.4 Repetition Column vs. Arbitrary Metadata

The `Predictions` class has `set_repetition_column()` which stores a column name but **not** the column data. When `by_repetition=True` is used, it resolves to the column name and then looks it up in `row["metadata"]` -- which is empty after reload.

The issue author specifically asks for aggregation by metadata columns **different from** the repetition column used during calibration. This means even if the repetition column's data were persisted, the feature request requires arbitrary metadata column access.

---

## 3. Implementation Proposal

### 3.1 Strategy: Store Per-Sample Metadata in Parquet

The most practical approach is to store the per-sample metadata dict alongside the prediction arrays in Parquet. This keeps the metadata co-located with the arrays it describes, avoids DuckDB schema changes, and is naturally portable.

### 3.2 Changes Required

#### 3.2.1 ArrayStore Parquet Schema (`array_store.py`)

Add a `sample_metadata` column to `_PARQUET_SCHEMA`:

```python
_PARQUET_SCHEMA = pa.schema([
    # ... existing columns ...
    ("sample_metadata", pa.utf8()),  # JSON-encoded {col: [values...]}
])
```

JSON encoding is chosen because:
- Metadata columns have mixed types (strings, floats, ints).
- The number and names of columns vary per dataset.
- JSON is self-describing and compatible with any future storage backend.

Update `_records_to_table()` to serialize the metadata dict to JSON. Update `load_batch()` / `load_single()` to deserialize it back.

**Size consideration**: For a dataset with 200 samples and 5 metadata columns of short strings, this adds ~2-5 KB per prediction row (Zstd-compressed). For typical workspaces with hundreds of predictions, this is negligible.

#### 3.2.2 Predictions Flush (`predictions.py`)

In `flush()`, include `row["metadata"]` in the array record passed to `save_batch`:

```python
array_records.append({
    # ... existing fields ...
    "sample_metadata": row.get("metadata", {}),
})
```

#### 3.2.3 Predictions Load (`predictions.py`)

In `_populate_buffer_from_store()`, when loading arrays, also restore metadata:

```python
arrays = store.array_store.load_single(pred_id, ...)
if arrays:
    # ... existing array loading ...
    metadata = arrays.get("sample_metadata", {})
```

Pass it to `add_prediction(..., metadata=metadata, ...)`.

Similarly update `_load_portable_parquet()`.

#### 3.2.4 No DuckDB Schema Changes

The DuckDB `predictions` table does not need modification. Per-sample metadata is a property of the array data, not of the prediction record's scalar metadata. Keeping it in Parquet is the right separation.

#### 3.2.5 Backward Compatibility

Existing Parquet files without the `sample_metadata` column will return `None` for that field when read. The load path should handle this gracefully (default to empty dict). No migration is needed -- old files simply have no metadata, which is the current behavior.

### 3.3 Alternative: Re-Join from Dataset at Query Time

Instead of storing metadata, provide a method to re-join predictions with the original dataset:

```python
predictions.set_dataset_context(dataset)
predictions.top(5, by_repetition="Sample_ID")  # Uses dataset metadata via sample_indices
```

**Pros**: No storage changes. No data duplication.
**Cons**: Requires the original dataset to be available. Fragile if dataset changes. `sample_indices` semantics must be exactly right. Does not work for portable Parquet files.

This could be offered as a complementary feature but should not replace persistence.

### 3.4 API Surface

No new public API methods are strictly required. The existing `top(by_repetition="Column")` already supports arbitrary column names -- it just fails when metadata is missing. Once metadata is persisted, the existing API works as expected.

Optionally, the `_apply_aggregation` method could be enhanced to produce clearer error messages when metadata is missing, distinguishing between "column not persisted" and "column not in dataset".

---

## 4. DuckDB to SQLite Migration Considerations

### 4.1 What Is DuckDB-Specific in This Feature

**Nothing.** The proposed changes are entirely in the Parquet layer (`ArrayStore`) and the in-memory `Predictions` buffer. The DuckDB schema (`store_schema.py`) and `WorkspaceStore` are untouched.

Specifically:
- `ArrayStore` uses PyArrow + Polars for Parquet I/O. No DuckDB dependency.
- The `Predictions` class buffer is plain Python dicts.
- The `flush()` path writes metadata to Parquet, not DuckDB.

### 4.2 Should This Feature Wait for the Migration?

**No.** This feature is storage-backend-agnostic by design. The Parquet sidecar files (`arrays/*.parquet`) are independent of whether the relational metadata lives in DuckDB or SQLite. The `ArrayStore` class has no DuckDB imports or dependencies.

If/when DuckDB is replaced by SQLite (issue #37):
- `WorkspaceStore` and `store_schema.py` would change.
- `store_queries.py` SQL syntax might need adjustments.
- `ArrayStore` would remain identical.
- The metadata persistence proposed here would remain identical.

### 4.3 DuckDB Locking and This Feature

Issue #37 describes IOError on `store.duckdb` due to file locking on successive runs with different datasets. This feature does not exacerbate the locking issue because:
- Metadata is stored in Parquet (one file per dataset), not DuckDB.
- Parquet writes use atomic temp-file-then-rename, no file locks.
- No additional DuckDB writes are introduced.

---

## 5. Implementation Checklist

| # | File | Change | Effort |
|---|------|--------|--------|
| 1 | `pipeline/storage/array_store.py` | Add `sample_metadata` (JSON utf8) to `_PARQUET_SCHEMA`; serialize in `_records_to_table`; deserialize in `load_batch` | Small |
| 2 | `data/predictions.py` (`flush`) | Include `row["metadata"]` in `array_records` dict | Trivial |
| 3 | `data/predictions.py` (`_populate_buffer_from_store`) | Restore metadata from loaded arrays into buffer | Small |
| 4 | `data/predictions.py` (`_load_portable_parquet`) | Restore metadata from Parquet columns | Small |
| 5 | Tests: unit test for round-trip metadata persistence | Verify metadata survives flush -> reload -> aggregation | Medium |
| 6 | Tests: integration test for `top(by_repetition=col)` after reload | End-to-end with workspace store | Medium |

**Total estimated effort**: Small to medium. No architectural changes. No schema migrations. No breaking changes.

---

## 6. Risks and Edge Cases

1. **Large metadata columns**: If a dataset has many metadata columns with long string values, the serialized JSON could become large. Mitigation: optionally filter to only persist metadata columns that the user has marked for aggregation (e.g., via `dataset.set_aggregate()` or a new `persist_metadata_columns` option). However, for v1 persisting all metadata is simpler and the Zstd compression in Parquet handles this well.

2. **Metadata column type fidelity**: JSON serialization may lose numpy dtype information (e.g., int64 becomes Python int). This is acceptable since metadata is used for grouping (equality checks), not arithmetic.

3. **Dataset evolution**: If a dataset is modified after training (columns added/removed), stored metadata reflects the state at training time. This is correct behavior -- predictions should be self-contained snapshots.

4. **Memory footprint**: Loading metadata for all predictions into the buffer increases memory usage. For workspaces with thousands of predictions, this could be noticeable. Mitigation: the `load_arrays=False` path already skips array loading; metadata loading could follow the same flag.
