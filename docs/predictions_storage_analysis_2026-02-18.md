# Predictions Storage Analysis: DuckDB Architecture Review

Date: 2026-02-18
Scope: `nirs4all/data/predictions.py`, `nirs4all/pipeline/storage/workspace_store.py`, `store_schema.py`, `store_queries.py`
Context: 6GB+ workspace store, slow read performance, question of whether DuckDB is the right tool for storing dense prediction matrices

---

## Executive Summary

The current DuckDB storage stores **everything** in a single `store.duckdb` file: scalar metadata, JSON configs, execution logs, and dense numerical arrays (y_true, y_pred, y_proba). This design is correct for the metadata/relational side but is a **fundamental mismatch** for dense array storage, which is the dominant contributor to the 6GB file size and the root cause of slow reads.

DuckDB is a columnar analytical database optimized for scanning/filtering/aggregating scalar columns. Storing variable-length `DOUBLE[]` arrays in it forces DuckDB to treat what is essentially a matrix storage problem as a relational one. The result: bloated file size, slow sequential array retrieval, and poor compression of what should be highly compressible numerical data.

**The core insight**: metadata queries ("give me the top 10 models by RMSE") should be fast (DuckDB excels at this). Array retrieval ("give me the y_true/y_pred vectors for prediction X") should also be fast but uses a different access pattern. These two workloads have opposite storage requirements and should be separated.

---

## 1. Current Architecture

### 1.1 Schema Overview

Seven tables in a single `store.duckdb`:

| Table | Row count driver | Data profile |
|-------|-----------------|-------------|
| `runs` | 1 per nirs4all.run() call | Small, scalar metadata + JSON config |
| `pipelines` | 1 per expanded config variant | Small, scalar + JSON |
| `chains` | 1 per preprocessing-to-model combo | Small, scalar + JSON (steps, artifacts) |
| `predictions` | 1 per (chain, fold, partition) | Medium, ~23 scalar/JSON columns |
| `prediction_arrays` | 1 per prediction | **Large**: y_true[], y_pred[], y_proba[], sample_indices[], weights[] |
| `artifacts` | 1 per fitted model/transformer | Small (metadata only; binaries on filesystem) |
| `logs` | 1 per pipeline step event | Medium, scalar + JSON details |

### 1.2 Write Path

```
Pipeline execution
    -> Predictions._buffer (in-memory list of dicts)
    -> flush() iterates buffer row-by-row:
        -> store.save_prediction()      # 1 INSERT per row (with upsert check)
        -> store.save_prediction_arrays()  # 1 INSERT per row, arrays via .tolist()
    -> store.update_chain_summary()  # 3-4 aggregate queries per chain
```

**Key observation**: arrays are converted from numpy to Python lists (`arr.flatten().tolist()`) before insertion. This is the most expensive step of the write path -- it forces a full copy from contiguous numpy memory into Python object space, then DuckDB re-encodes it into its internal format.

### 1.3 Read Path

Two distinct read patterns exist:

**Pattern A: Metadata queries** (fast, well-served by DuckDB)
```python
store.query_predictions(dataset_name="wheat", partition="val")
store.top_predictions(10, metric="val_score")
store.query_chain_summaries(run_id=...)
```
These return `pl.DataFrame` via zero-copy Arrow transfer. They scan only the `predictions` table (no array columns). This works well.

**Pattern B: Array retrieval** (slow, poorly served by DuckDB)
```python
# Loading from workspace for visualization/analysis:
Predictions.from_workspace("workspace", load_arrays=True)
# Internally iterates ALL predictions and calls per-row:
store.get_prediction_arrays(pred_id)  # SELECT from prediction_arrays WHERE id = $1
```
This does N individual point-lookups against the `prediction_arrays` table. Each lookup deserializes a `DOUBLE[]` column into a Python list, then converts to numpy. For a workspace with thousands of predictions, this is extremely slow.

### 1.4 The `from_workspace` Anti-Pattern

The `Predictions.from_workspace()` method (predictions.py:224-346) is the primary source of slow reads. It:

1. Queries ALL prediction metadata (`store.query_predictions()`) -- returns a Polars DataFrame (fast)
2. Iterates every row with `df.iter_rows(named=True)` (medium: Python-level iteration)
3. For each row, calls `store.get_prediction_arrays(pred_id)` -- **individual SQL query per prediction** (slow)
4. Deserializes JSON fields (`json.loads`) per row (medium)
5. Calls `predictions.add_prediction(...)` to re-buffer into memory (medium: builds dict per row)

This is an **N+1 query pattern** against DuckDB, where N can be tens of thousands. Each array retrieval requires a fresh SQL parse, DuckDB scan, array deserialization, and numpy conversion.

---

## 2. What Works Well (Right Decisions)

### 2.1 DuckDB for Relational Metadata

The relational schema for runs/pipelines/chains/predictions is well-designed:
- Natural key uniqueness constraint prevents duplicates
- Indexes on common filter columns (dataset_name, partition, val_score)
- The `v_chain_summary` view pre-computes CV averages, avoiding repeated aggregation
- Cascade deletion handles the complex FK graph correctly
- JSON columns for flexible data (scores, best_params, config) -- appropriate for DuckDB's JSON support

### 2.2 Content-Addressed Artifact Storage

Binary artifacts (fitted models) are stored on filesystem with DuckDB holding only metadata. This is exactly right -- binary blobs should not live in a database. The content-hash dedup and ref-counting are well-engineered.

### 2.3 Chain as First-Class Entity

The chain concept (preprocessing-to-model sequence) with denormalized summary columns is a good design. It enables efficient ranking without joining through predictions every time.

### 2.4 Degraded Mode

The retry-on-lock with degraded mode fallback is pragmatic. DuckDB's single-writer limitation is a real constraint, and graceful degradation is the right answer.

### 2.5 Bulk Chain Summary Updates

`bulk_update_chain_summaries()` uses set-based SQL instead of per-chain loops, reducing round-trips from 4N to 4. This is the right pattern.

---

## 3. What Does Not Work (Problems)

### 3.1 Dense Arrays in DuckDB (Core Problem)

**Problem**: `DOUBLE[]` columns in DuckDB are stored as variable-length lists in a columnar format designed for analytical queries. But prediction arrays are not queried analytically -- they are always loaded whole, for a specific prediction_id. This is a key-value access pattern, not an analytical one.

**Impact**:
- **File size bloat**: DuckDB's internal encoding of `DOUBLE[]` has per-value overhead. A 1000-element array of float64 should take ~8KB raw; in DuckDB it takes significantly more due to list encoding, null handling, and page overhead.
- **Slow reads**: Point-lookups on `prediction_arrays` are not DuckDB's strength. It must locate the row, decode the list column, and convert to Python. For thousands of predictions, this dominates read time.
- **Slow writes**: `arr.flatten().tolist()` creates a Python list of float objects (each ~28 bytes), passed to DuckDB which re-encodes them. For y_proba matrices (n_samples x n_classes), this means flattening + copying potentially millions of floats through Python object space.

**Evidence**: The user reports "very very very slow" reads on a 6GB store. Parquet with the same data volume was efficient. The difference is that Parquet stores contiguous float arrays natively, while DuckDB adds relational overhead.

### 3.2 N+1 Query Pattern in `from_workspace`

**Problem**: Loading predictions from store iterates rows in Python and issues one SQL query per prediction for arrays. This is the classic N+1 pattern.

**Impact**: For M predictions, the load time is dominated by M individual DuckDB round-trips, each involving SQL parsing, plan generation, page lookup, array decoding, and numpy conversion.

**Why it exists**: The current API returns arrays per-prediction (dict with numpy arrays). There's no batch array retrieval method.

### 3.3 Python List Conversion on Write

**Problem**: `save_prediction_arrays()` converts numpy arrays to Python lists via `.flatten().tolist()` before DuckDB insertion.

**Impact**: For a prediction with 5000 samples:
- `y_true.tolist()` creates 5000 Python float objects (each ~28 bytes = 140KB of Python objects for 40KB of raw data)
- Same for `y_pred`, `sample_indices`, `weights`
- Total: ~500KB of Python object overhead for ~160KB of raw data
- This happens for every prediction on every flush

### 3.4 No DuckDB Configuration Tuning

**Problem**: The only DuckDB pragma set is `enable_progress_bar=false`. No memory limits, no thread configuration, no compression settings.

**Impact**: DuckDB uses defaults, which may not be optimal for the workload. For example:
- Default memory limit may cause excessive disk spilling for large queries
- No explicit WAL checkpoint configuration

### 3.5 Single-File Scaling Limit

**Problem**: Everything in one `store.duckdb` file means the entire database must be opened even for simple metadata queries. DuckDB uses memory-mapped I/O; a 6GB file means 6GB of virtual address space, and cold reads touch many pages.

**Impact**: Even a simple "get top 10 predictions" query must open a connection to a 6GB file. The prediction_arrays table dominates the file size, so metadata queries pay the cost of co-location with array data.

### 3.6 y_proba Shape Loss

**Problem**: 2D class probability matrices are flattened to 1D for storage (`arr.flatten().tolist()`). The original shape (n_samples, n_classes) is not stored anywhere in `prediction_arrays`.

**Impact**: Reconstruction requires inferring shape from external context (n_samples from prediction metadata + n_classes from the task). This is fragile and undocumented.

### 3.7 Flush is Row-by-Row

**Problem**: `Predictions.flush()` iterates the buffer and calls `save_prediction()` + `save_prediction_arrays()` individually per row. Each `save_prediction()` does a natural-key lookup (SELECT) + possible DELETE + INSERT.

**Impact**: For a pipeline with 100 predictions (10 folds x 2 partitions x 5 models), flush makes ~300 SQL calls (100 lookups + 100 prediction inserts + 100 array inserts). Each goes through the lock, connection check, and retry wrapper.

---

## 4. Quantifying the Problem

### 4.1 File Size Breakdown (Estimated for 6GB Store)

Assuming a typical NIRS workflow: 500 runs x 20 pipelines x 5 chains x 10 folds x 3 partitions = 1.5M prediction rows, each with ~500-sample arrays:

| Table | Est. rows | Est. size | % of total |
|-------|----------|-----------|-----------|
| `prediction_arrays` | 1.5M | ~5.4 GB | ~90% |
| `predictions` | 1.5M | ~400 MB | ~7% |
| `chains` | 50K | ~50 MB | <1% |
| `logs` | 500K | ~100 MB | ~2% |
| `runs + pipelines + artifacts` | <10K | <10 MB | <1% |

The array data dominates. The metadata that DuckDB excels at storing and querying is less than 10% of the total.

### 4.2 Read Performance Profile

For loading 10,000 predictions with arrays:
- Metadata query: ~100ms (single Polars scan, fast)
- Array retrieval: ~10,000 individual queries x ~5ms each = **~50 seconds**
- JSON parsing + dict construction: ~2 seconds
- **Total: ~52 seconds**, dominated by array retrieval

For comparison, loading the same data from Parquet:
- Parquet read with pyarrow: ~2 seconds for the full dataset (columnar scan, SIMD decompression)
- Zero-copy to numpy: ~100ms
- **Total: ~2 seconds**

This 25x performance gap is structural, not tunable.

---

## 5. Proposals

### Proposal A: Hybrid Architecture (DuckDB + Parquet Sidecar) -- Recommended

**Concept**: Keep DuckDB for relational metadata (runs, pipelines, chains, predictions, artifacts, logs). Move dense arrays to a Parquet-based sidecar store.

**Layout**:
```
workspace/
    store.duckdb              # Metadata only (~500MB for a large workspace)
    arrays/                   # Dense array storage
        <dataset_name>/       # Partitioned by dataset
            arrays.parquet    # All arrays for this dataset
    artifacts/                # Binary artifacts (unchanged)
```

**Schema change**: Drop `prediction_arrays` table entirely. The Parquet file stores:
```
prediction_id: string (join key)
y_true: list[float64]
y_pred: list[float64]
y_proba: list[float64]
y_proba_shape: list[int32]    # Preserve original shape
sample_indices: list[int32]
weights: list[float64]
```

**Why Parquet**:
- Parquet natively stores arrays as nested types with excellent compression (Zstd, ~4x for float arrays)
- Batch reads via pyarrow/polars are zero-copy and orders of magnitude faster than row-by-row DuckDB queries
- DuckDB can query Parquet files directly when needed (`SELECT * FROM read_parquet('arrays/*.parquet') WHERE prediction_id IN (...)`)
- Parquet files are append-friendly (new predictions append new row groups)
- Well-understood, battle-tested format for numerical data

**Write path**:
```python
# During flush:
# 1. Save metadata to DuckDB (unchanged)
store.save_prediction(...)

# 2. Batch-write arrays to Parquet (new)
# Accumulate all arrays in a pyarrow table, write as single row group
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "prediction_id": [...],
    "y_true": pa.array([...], type=pa.list_(pa.float64())),
    "y_pred": pa.array([...], type=pa.list_(pa.float64())),
    ...
})
pq.write_to_dataset(table, "arrays/", partition_cols=["dataset_name"])
```

**Read path**:
```python
# Metadata query (unchanged, fast)
df = store.query_predictions(dataset_name="wheat")

# Array retrieval (new, batch)
pred_ids = df["prediction_id"].to_list()
arrays_df = pl.scan_parquet("arrays/wheat/").filter(
    pl.col("prediction_id").is_in(pred_ids)
).collect()
# Zero-copy to numpy
y_true = arrays_df["y_true"].to_numpy()
```

**Benefits**:
- DuckDB file shrinks from ~6GB to ~500MB (metadata only) -- all metadata queries become instant
- Array reads become batch operations with native Parquet decompression -- 10-50x faster
- Parquet compression (Zstd) typically achieves 3-5x on float arrays -- 6GB of arrays might compress to 1.5GB
- DuckDB can still JOIN with Parquet files when needed (cross-engine queries)
- No change to the relational schema or query API

**Costs**:
- Two storage backends to manage (DuckDB + Parquet)
- Transactional consistency between DuckDB and Parquet is not atomic (but predictions are already non-critical -- degraded mode exists)
- Need to handle Parquet file growth/compaction over time

**Migration**: Drop `prediction_arrays` table, export existing arrays to Parquet files, update `save_prediction_arrays()` and `get_prediction_arrays()`.

---

### Proposal B: DuckDB with Attached Parquet (DuckDB-Native Approach)

**Concept**: Keep DuckDB but use its native Parquet integration to attach external Parquet files as virtual tables. Arrays are written directly to Parquet; DuckDB queries them transparently.

**Layout**:
```
workspace/
    store.duckdb              # Metadata tables
    arrays.parquet            # Dense arrays (managed externally, queryable from DuckDB)
    artifacts/
```

**How it works**:
```sql
-- In DuckDB, create a view that reads from the Parquet file
CREATE VIEW prediction_arrays AS
SELECT * FROM read_parquet('arrays.parquet');

-- Queries work transparently
SELECT p.*, pa.y_true, pa.y_pred
FROM predictions p
JOIN prediction_arrays pa ON p.prediction_id = pa.prediction_id
WHERE p.val_score < 0.1;
```

**Benefits**:
- Single query engine (DuckDB) for both metadata and arrays
- Parquet compression benefits
- No code change in query layer (views are transparent)

**Costs**:
- Write path must manage Parquet file separately (append is tricky with Parquet)
- DuckDB's Parquet reader is good but adds overhead vs. direct pyarrow reads
- Upsert (delete + re-insert) is harder with external Parquet files

**Verdict**: More elegant than Proposal A but harder to implement correctly, especially for upserts.

---

### Proposal C: Optimize DuckDB In-Place (Tactical Fixes)

If a full architecture change is too disruptive, several tactical optimizations can significantly improve performance without changing the storage layout:

#### C.1 Batch Array Loading

Replace the N+1 pattern in `from_workspace` with a single batch query:

```python
# Instead of per-row get_prediction_arrays():
def get_prediction_arrays_batch(self, prediction_ids: list[str]) -> pl.DataFrame:
    """Load arrays for multiple predictions in a single query."""
    placeholders = ", ".join(f"${i+1}" for i in range(len(prediction_ids)))
    sql = f"SELECT * FROM prediction_arrays WHERE prediction_id IN ({placeholders})"
    return self._fetch_pl(sql, prediction_ids)
```

**Expected improvement**: 10-50x for large loads (one SQL call instead of thousands).

#### C.2 DuckDB Memory & Thread Configuration

```python
conn.execute("SET memory_limit='2GB'")        # Prevent unbounded memory use
conn.execute("SET threads=4")                  # Reasonable parallelism
conn.execute("SET checkpoint_threshold='256MB'")  # More frequent WAL checkpoints
```

#### C.3 Batch Insert on Flush

Replace row-by-row inserts with DuckDB's `executemany()` or direct Arrow insertion:

```python
# Instead of per-row save_prediction_arrays():
import pyarrow as pa

arrays_table = pa.table({
    "prediction_id": prediction_ids,
    "y_true": pa.array(y_true_lists, type=pa.list_(pa.float64())),
    ...
})
conn.execute("INSERT INTO prediction_arrays SELECT * FROM arrays_table")
```

#### C.4 Lazy Array Loading

Make `from_workspace` default to `load_arrays=False` and load arrays on demand:

```python
# Load metadata only (fast)
predictions = Predictions.from_workspace("workspace", load_arrays=False)
top_5 = predictions.top(5)

# Load arrays only for the top 5
for pred in top_5:
    arrays = store.get_prediction_arrays(pred.prediction_id)
```

**Expected improvement**: Metadata-only load goes from ~50s to ~1s for common use cases.

---

### Proposal D: Alternative Storage Backends (If Starting Fresh)

For reference, alternatives to DuckDB for this specific workload:

| Backend | Metadata | Arrays | Pros | Cons |
|---------|----------|--------|------|------|
| DuckDB + Parquet (Proposal A) | DuckDB | Parquet | Best of both worlds | Two backends |
| SQLite + NumPy files | SQLite | `.npy` files | Simple, fast array I/O | No columnar queries |
| Lance | Both | Both | ML-native, versioned | Young ecosystem |
| TileDB | Both | Both | Multi-dimensional arrays | Complex, heavy dependency |
| HDF5 | Groups | Datasets | Standard for numerical data | No relational queries |
| Pure Parquet | Parquet | Parquet | Single format, fast batch | No point-lookups, no indexes |

---

## 6. Recommendation

**Short-term (1-2 days)**: Implement Proposal C (tactical fixes). The batch array loading (C.1) and lazy loading (C.4) alone will improve the common read path by 10-50x with minimal code change.

**Medium-term (1 week)**: Implement Proposal A (hybrid DuckDB + Parquet). This is the architecturally correct solution. DuckDB handles what it's good at (relational queries on metadata), Parquet handles what it's good at (dense array storage with compression).

**Do not do**: Keep the current architecture and hope DuckDB improves. The mismatch is fundamental -- DuckDB is not designed to be a key-value store for large arrays. No amount of tuning will change the access pattern mismatch.

---

## 7. Implementation Sketch for Proposal A

### 7.1 New Module: `nirs4all/pipeline/storage/array_store.py`

```python
class ArrayStore:
    """Parquet-backed dense array storage for predictions."""

    def __init__(self, workspace_path: Path):
        self._arrays_dir = workspace_path / "arrays"
        self._arrays_dir.mkdir(exist_ok=True)

    def save_batch(self, records: list[dict]) -> None:
        """Write a batch of prediction arrays to Parquet."""
        # Group by dataset_name for partitioned storage
        # Write as single row group with Zstd compression

    def load_batch(self, prediction_ids: list[str], dataset_name: str) -> dict[str, dict]:
        """Load arrays for multiple predictions."""
        # Predicate pushdown: filter by prediction_id
        # Zero-copy to numpy

    def load_single(self, prediction_id: str) -> dict | None:
        """Load arrays for a single prediction (for backward compat)."""

    def delete_batch(self, prediction_ids: list[str]) -> None:
        """Remove arrays (rewrite Parquet without deleted rows)."""
```

### 7.2 Changes to WorkspaceStore

- Remove `save_prediction_arrays()` and `get_prediction_arrays()` methods
- Add `ArrayStore` as a composition member
- Update `flush()` to batch-write arrays via `ArrayStore.save_batch()`
- Update `delete_run()`/`delete_pipeline()` to cascade into `ArrayStore`

### 7.3 Changes to Predictions

- Update `from_workspace()` to use `ArrayStore.load_batch()` instead of per-row queries
- Keep `load_arrays=False` path unchanged (metadata-only from DuckDB)

### 7.4 Schema Migration

```python
def _migrate_to_parquet_arrays(conn, workspace_path):
    """One-time migration: export prediction_arrays to Parquet, drop table."""
    # 1. Export all arrays to Parquet files (partitioned by dataset)
    # 2. Verify row counts match
    # 3. Drop prediction_arrays table
    # 4. VACUUM to reclaim space
```

---

## 8. Appendix: Code Locations

| Concern | File | Key lines |
|---------|------|-----------|
| Schema DDL | `pipeline/storage/store_schema.py` | 29-160 (SCHEMA_DDL) |
| Array insert | `pipeline/storage/workspace_store.py` | 1025-1075 (save_prediction_arrays) |
| Array read | `pipeline/storage/workspace_store.py` | 1758-1789 (get_prediction_arrays) |
| N+1 load pattern | `data/predictions.py` | 265-346 (_load_from_store) |
| Flush row-by-row | `data/predictions.py` | 517-611 (flush) |
| Query builders | `pipeline/storage/store_queries.py` | 142-146 (INSERT_PREDICTION_ARRAYS) |
| Python list conversion | `workspace_store.py` | 1056-1064 (_to_list, _to_int_list) |
| DuckDB connection setup | `workspace_store.py` | 251-264 (__init__) |

---

## 9. Appendix: DuckDB Configuration Checklist

Settings that should be evaluated regardless of which proposal is adopted:

```sql
-- Memory management
SET memory_limit = '2GB';              -- Prevent unbounded memory allocation
SET temp_directory = '/tmp/duckdb';    -- Explicit temp location for spills

-- Write performance
SET checkpoint_threshold = '256MB';    -- WAL checkpoint frequency
SET wal_autocheckpoint = '256MB';      -- Auto-checkpoint threshold

-- Read performance
SET threads = 4;                       -- Parallelism for scans
SET enable_progress_bar = false;       -- Already set

-- Compression (for existing DOUBLE[] columns)
SET force_compression = 'zstd';        -- Better than default for float arrays
```

---

## 10. Appendix: What Parquet Was Good At (and Why It Felt Fast)

The previous Parquet-based system stored predictions as `<dataset>.meta.parquet` files. These were:

1. **Batch-readable**: The entire dataset's predictions loaded in one `pd.read_parquet()` / `pl.scan_parquet()` call
2. **Compressed**: Parquet's Snappy/Zstd compression on float columns achieves 3-5x
3. **Columnar**: Reading only `y_true` and `y_pred` columns skips all metadata columns (projection pushdown)
4. **Memory-mapped**: PyArrow can memory-map Parquet files, avoiding full reads for filtered queries
5. **Zero-copy**: Arrow-native format means Polars/NumPy can consume without copying

The current DuckDB system lost all five of these advantages for array data while gaining transactional consistency and relational query power that arrays don't need. The hybrid approach (Proposal A) restores these advantages for arrays while keeping DuckDB's benefits for metadata.

---

## 11. Production Readiness Review (2026-02-18)

### Overall Assessment

The diagnosis is directionally correct and the proposed hybrid architecture (DuckDB for metadata + Parquet for dense arrays) is a strong long-term direction.
As written, this document is **not yet sufficient for production approval** because several core claims are estimated rather than measured, and the migration/consistency plan is underspecified.

### Findings (Ordered by Severity)
*
#### 1) High: Performance and size claims are not benchmark-backed

/* SKIPPED */

#### 2) High: Proposal A lacks an explicit cross-store consistency protocol

Moving metadata to DuckDB and arrays to Parquet introduces a two-phase write path, but the document does not define failure semantics (crash between metadata insert and array write, retry behavior, orphan cleanup).

**Production risk**: dangling metadata rows, missing arrays, or duplicate array records after retries.

**Required**:
- Define write state transitions (for example: `pending_arrays -> committed`).
- Make writes idempotent by `prediction_id`.
- Add recovery on startup (reconcile pending/orphaned records).
- Add integrity checks for read path (detect and surface missing arrays).

#### 3) High: Upsert/delete semantics for Parquet sidecar are not fully designed

Current system supports upsert by natural key and deletion cascades. Proposal A says "drop `prediction_arrays` and use Parquet" but does not define how frequent updates/deletes are handled without expensive full-file rewrites.

**Production risk**: write amplification, compaction debt, and degraded long-term performance.

**Required**:
- Define mutation strategy: immutable append + tombstones + periodic compaction, or copy-on-write partitions.
- Define compaction triggers (file count, tombstone ratio, size thresholds) and maintenance workflow.
- Document expected write amplification and recovery time.

#### 4) High: Migration plan is missing safety gates and rollback path

The migration sketch is too high-level for production change management.

**Production risk**: irreversible data loss or partial migration.

**Required**:
- Run in dual-write mode behind a feature flag before cutover.
- Add validation gates: row-count parity and checksum/hash parity for sampled arrays.
- Keep dual-read fallback until parity targets are met.
- Define rollback procedure with exact stop/go criteria.

#### 5) Medium: Partitioning strategy is too coarse

Partitioning only by `dataset_name` can create very large hot partitions for large datasets and poor selective reads for common query patterns (pipeline/chain/fold slices).

**Required**:
- Re-evaluate partition keys using observed query predicates.
- Add file sizing targets and row-group targets (avoid very small/very large files).

#### 6) Medium: Operational controls are missing

No concrete plan is given for file locking, concurrent writers, metrics, and alerting.

**Required**:
- Define writer concurrency model and locking boundary.
- Add observability: array write failures, missing-array reads, compaction lag, storage growth.
- Add SLOs for load/write operations and regression alerts.

#### 7) Low: Minor document drift with current schema details

The document states seven tables, while current schema also includes `projects`. This does not change the core diagnosis, but it should be corrected for accuracy.

### Revised Approval Recommendation

Approve the direction conditionally:

1. Ship Proposal C.1 (batch array load) + C.4 (lazy loading) first and measure real gains.
2. Build Proposal A behind feature flags with dual-write/dual-read safeguards.
3. Complete migration validation and rollback drills before dropping `prediction_arrays`.
4. Only then finalize the architecture switch as production default.

### Production Go/No-Go Checklist

- [ ] Consistency protocol implemented and tested for crash/retry scenarios.
- [ ] Parquet mutation + compaction strategy documented and automated.
- [ ] Dual-write migration completed with parity checks.
- [ ] Rollback path tested in staging.
- [ ] Observability + SLO alerts enabled.
