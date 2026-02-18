# Backlog: Option A — Hybrid DuckDB + Parquet Storage

**Date**: 2026-02-18
**Status**: Draft — Pending team approval
**Source**: `predictions_storage_analysis_2026-02-18.md` + production readiness review
**Scope**: `nirs4all/data/predictions.py`, `nirs4all/pipeline/storage/`

---

## Goal

Replace the `prediction_arrays` DuckDB table with **one Parquet file per dataset**. DuckDB keeps all relational metadata (runs, pipelines, chains, predictions, artifacts, logs). Dense numerical arrays (y_true, y_pred, y_proba, sample_indices, weights) move to self-describing Parquet files with Zstd compression.

Each Parquet file embeds enough metadata (model name, fold, partition, metric, score) to be **independently browsable** — a user can grab a single `.parquet` file, open it in any Parquet-aware tool (Polars, pandas, DuckDB CLI, Datasette), and understand what's inside without the DuckDB store.

**Target outcomes**:
- DuckDB file shrinks from ~6 GB to ~500 MB (metadata only)
- Array reads go from ~50 s to ~2 s for 10 000 predictions
- All existing query, ranking, filtering, and merge APIs remain unchanged
- One file per dataset = visually clean workspace, easy manual cleanup
- Simple migration: one command, reversible, zero data loss

---

## Phased Delivery

| Phase | Name | Description | Depends on |
|-------|------|-------------|------------|
| **1** | Core implementation | ArrayStore module + WorkspaceStore integration + batch flush | — |
| **2** | Migration | Migrate existing DuckDB stores, validation, rollback | Phase 1 |
| **3** | User-facing API | `Predictions(path)`, merge stores, maintenance helpers | Phase 1 |
| **4** | Cutover & cleanup | Drop `prediction_arrays`, remove legacy code | Phase 2 + 3 |

Full implementation in one pass. No intermediate "tactical fix" phase.

---

## Phase 1 — Core Implementation

### 1.1 Create `nirs4all/pipeline/storage/array_store.py`

New class `ArrayStore` with this public API:

```python
class ArrayStore:
    def __init__(self, base_dir: Path):
        """base_dir is the workspace root. Arrays live under base_dir / 'arrays'."""

    # --- Write ---
    def save_batch(self, records: list[dict]) -> int:
        """Write prediction arrays to Parquet. Returns rows written.
        Each dict: {prediction_id, dataset_name, model_name, fold_id, partition,
                     metric, val_score, task_type, y_true, y_pred, y_proba,
                     y_proba_shape, sample_indices, weights}.
        Grouped by dataset_name; each group appended to its dataset Parquet file.
        Writes are idempotent by prediction_id (duplicates resolved on compact).
        """

    # --- Read ---
    def load_batch(self, prediction_ids: list[str], dataset_name: str | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Load arrays for multiple predictions. Returns {prediction_id: {y_true, y_pred, ...}}.
        Uses predicate pushdown on prediction_id. Zero-copy to numpy where possible.
        If dataset_name given, reads only that file.
        """

    def load_single(self, prediction_id: str, dataset_name: str | None = None) -> dict[str, np.ndarray] | None:
        """Convenience: load arrays for one prediction."""

    def load_dataset(self, dataset_name: str) -> pl.DataFrame:
        """Load the full Parquet file for a dataset. Returns all columns (arrays + metadata).
        This is the 'portable read' — the returned DataFrame is self-describing.
        """

    # --- Delete ---
    def delete_batch(self, prediction_ids: set[str], dataset_name: str | None = None) -> int:
        """Mark prediction_ids as deleted (tombstone). Returns count.
        Physical removal happens during compact().
        """

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete the entire Parquet file for a dataset. Returns True if file existed."""

    # --- Maintenance ---
    def compact(self, dataset_name: str | None = None) -> dict:
        """Rewrite Parquet file(s): apply tombstones, deduplicate prediction_ids, re-sort.
        Returns stats: {dataset: {rows_before, rows_after, rows_removed, bytes_before, bytes_after}}.
        If dataset_name is None, compact all datasets.
        """

    def stats(self) -> dict:
        """Return storage stats: {total_files, total_rows, total_bytes, datasets: {name: {rows, bytes}}}."""

    def integrity_check(self, expected_ids: set[str] | None = None) -> dict:
        """Check Parquet health. Returns {orphan_ids: [...], missing_ids: [...], corrupt_files: [...]}.
        If expected_ids given, cross-check against it.
        """

    def list_datasets(self) -> list[str]:
        """Return dataset names that have Parquet files."""
```

### 1.2 Parquet layout — one file per dataset

```
workspace/
    store.duckdb                       # Metadata only (~500 MB for a large workspace)
    arrays/
        wheat.parquet                  # All predictions for dataset "wheat"
        corn.parquet                   # All predictions for dataset "corn"
        milk_fat.parquet               # etc.
        _tombstones.json               # {prediction_id: timestamp} — pending deletes
    artifacts/                         # Binary artifacts (unchanged)
```

One file per dataset means:
- Finished project? Delete the `.parquet` file.
- Want to share predictions? Copy one file.
- Want to browse externally? Open it in any Parquet tool.
- Clean visual layout: `ls arrays/` shows exactly your datasets.

### 1.3 Parquet schema — self-describing and portable

Each Parquet file contains **arrays + lightweight metadata** so it can be used standalone:

| Column | Arrow type | Purpose |
|--------|-----------|---------|
| `prediction_id` | `utf8` | Join key to DuckDB `predictions` table |
| `dataset_name` | `utf8` | Self-describing: which dataset this belongs to |
| `model_name` | `utf8` | Portable: which model produced this prediction |
| `fold_id` | `utf8` | Portable: which CV fold |
| `partition` | `utf8` | Portable: "train" / "val" / "test" |
| `metric` | `utf8` | Portable: metric name (e.g. "rmse", "accuracy") |
| `val_score` | `float64` | Portable: validation score for quick ranking |
| `task_type` | `utf8` | Portable: "regression" / "classification" |
| `y_true` | `list<float64>` | Dense array |
| `y_pred` | `list<float64>` | Dense array |
| `y_proba` | `list<float64>` | Dense array (flattened; shape in `y_proba_shape`) |
| `y_proba_shape` | `list<int32>` | `[n_samples, n_classes]` — fixes shape loss bug |
| `sample_indices` | `list<int32>` | Dense array |
| `weights` | `list<float64>` | Dense array |

**Portable use case**: a user copies `wheat.parquet` to another machine and runs:

```python
import polars as pl
df = pl.read_parquet("wheat.parquet")
# Browse: model names, scores, partitions — all there
df.filter(pl.col("partition") == "val").sort("val_score").head(10)
# Access arrays directly
y_true = df.row(0, named=True)["y_true"]
```

The full metadata (scores dict, best_params, preprocessings, branch info, timestamps) stays in DuckDB. The Parquet carries just enough for standalone browsing and basic ranking. Users who need the full picture use `Predictions(workspace_path)` which joins both sources.

**Compression**: Zstd level 3 (good ratio, fast decompression).
**Row group size**: Target 128 MB uncompressed per row group.

### 1.4 Wire ArrayStore into WorkspaceStore

**File**: `workspace_store.py`

Compose `ArrayStore` as a member of `WorkspaceStore`:

```python
class WorkspaceStore:
    def __init__(self, workspace_path, ...):
        ...
        self._array_store = ArrayStore(workspace_path)
```

**Write path** — replace `save_prediction_arrays()`:
- Remove the DuckDB `INSERT INTO prediction_arrays` code path entirely.
- Array writes go through `ArrayStore.save_batch()`.
- During `flush()`: accumulate all array records in a list, call `save_batch()` once at the end of the flush cycle (not row-by-row).
- Write order: **(1)** write arrays to Parquet, **(2)** insert metadata into DuckDB. A crash between (1) and (2) leaves orphan array rows (harmless, cleaned by `compact()`).

**Read path** — replace `get_prediction_arrays()` / `get_prediction_arrays_batch()`:
- Single prediction: `ArrayStore.load_single(prediction_id, dataset_name)`
- Batch: `ArrayStore.load_batch(prediction_ids, dataset_name)`
- Full dataset: `ArrayStore.load_dataset(dataset_name)`

**Delete path** — update cascade deletes:
- `delete_prediction()`: also call `ArrayStore.delete_batch({prediction_id})`
- `delete_run()` / `delete_pipeline()`: collect affected prediction_ids, call `ArrayStore.delete_batch(ids)`
- `delete_dataset()`: call `ArrayStore.delete_dataset(dataset_name)` (deletes the whole file)

### 1.5 Update `store_queries.py`

- Remove `INSERT_PREDICTION_ARRAYS`, `DELETE_PREDICTION_ARRAYS`, `GET_PREDICTION_ARRAYS` constants.
- Remove `CASCADE_DELETE_*_PREDICTION_ARRAYS` queries.
- Array operations no longer touch SQL.

### 1.6 Update `store_schema.py`

- Remove `prediction_arrays` from `SCHEMA_DDL` for new stores.
- Existing stores with `prediction_arrays` are handled by migration (Phase 2).

### 1.7 Batch flush optimization

**File**: `predictions.py::flush()`

```python
# Current: row-by-row (N SQL calls for arrays)
for entry in self._buffer:
    store.save_prediction(...)           # metadata → DuckDB
    store.save_prediction_arrays(...)    # arrays → DuckDB (SLOW)

# New: batch (1 Parquet write for all arrays)
array_records = []
for entry in self._buffer:
    store.save_prediction(...)           # metadata → DuckDB
    array_records.append({...})          # accumulate
store.array_store.save_batch(array_records)  # arrays → Parquet (FAST, one write)
```

### 1.8 DuckDB connection tuning

**File**: `workspace_store.py::__init__`

Apply on every connection open (benefits metadata queries regardless of array backend):
```sql
SET memory_limit = '2GB';
SET threads = 4;
SET checkpoint_threshold = '256MB';
```

### 1.9 Writer concurrency model

Single writer, consistent with DuckDB's existing single-writer model. The DuckDB lock already serializes writers. Parquet writes happen inside the same lock scope. No additional locking needed.

Read access is lock-free (Parquet files are immutable snapshots; new rows are appended as new row groups, old rows are removed by rewriting during `compact()`).

### 1.10 Crash consistency protocol

Write order: **(1)** write Parquet row group, **(2)** insert metadata into DuckDB.

On startup recovery (`WorkspaceStore.__init__`):
- Parquet prediction_ids not in DuckDB → orphaned arrays → cleaned by next `compact()`.
- DuckDB prediction_ids not in Parquet → missing arrays → surfaced by `integrity_check()`.

A crash between (1) and (2) leaves orphan array rows (harmless). A crash before (1) leaves no partial state.

---

## Phase 2 — Migration

### 2.1 Migration function

**File**: `nirs4all/pipeline/storage/migration.py` (new)

```python
def migrate_arrays_to_parquet(
    workspace_path: str | Path,
    *,
    batch_size: int = 10_000,
    verify: bool = True,
    dry_run: bool = False,
) -> MigrationReport:
    """Migrate prediction_arrays from DuckDB to Parquet sidecar files.

    Steps:
        1. Open store in read-only mode.
        2. Query distinct dataset_names from predictions table.
        3. For each dataset (one at a time):
           a. Stream prediction_ids + arrays from prediction_arrays in batches.
           b. Join with predictions table to get portable metadata columns
              (model_name, fold_id, partition, metric, val_score, task_type).
           c. Write to arrays/<dataset_name>.parquet via ArrayStore.
           d. Verify: sample 1% of rows, compare array checksums DuckDB vs Parquet.
        4. If all datasets passed verification and dry_run=False:
           a. DROP TABLE prediction_arrays.
           b. VACUUM to reclaim space.
        5. Return MigrationReport.

    Rollback: if verify fails or error occurs, delete arrays/ directory. DuckDB is untouched.
    """
```

### 2.2 MigrationReport dataclass

```python
@dataclass
class MigrationReport:
    total_rows: int
    rows_migrated: int
    datasets_migrated: list[str]
    verification_passed: bool
    verification_sample_size: int
    verification_mismatches: int
    duckdb_size_before: int       # bytes
    duckdb_size_after: int        # bytes
    parquet_total_size: int       # bytes
    duration_seconds: float
    errors: list[str]
```

### 2.3 CLI entry point

```bash
# Migrate a workspace
python -m nirs4all.pipeline.storage.migration /path/to/workspace

# Dry run (read-only, report only)
python -m nirs4all.pipeline.storage.migration /path/to/workspace --dry-run

# Verify an already-migrated workspace
python -m nirs4all.pipeline.storage.migration /path/to/workspace --verify-only
```

### 2.4 Auto-detection on store open

**File**: `workspace_store.py::__init__`

| `prediction_arrays` table exists? | `arrays/` directory exists? | Behavior |
|---|----|---|
| Yes | No | **Legacy store.** Log warning: `"Legacy array storage detected. Run 'python -m nirs4all.pipeline.storage.migration <path>' to upgrade."` Read arrays from DuckDB. |
| No | Yes | **Migrated store.** Use ArrayStore for all array operations. |
| Yes | Yes | **Mid-migration.** Read from Parquet (preferred), fall back to DuckDB for missing rows. |
| No | No | **New store** (or store with no predictions yet). Use ArrayStore. |

No silent auto-migration. The user must explicitly run the migration command.

### 2.5 Rollback procedure

Migration only drops `prediction_arrays` after all datasets pass verification. Before that point:
- Rollback = delete the `arrays/` directory. Store is untouched.
- The `--dry-run` flag lets users preview without any writes.

After `prediction_arrays` is dropped:
- Rollback requires restoring from backup. Migration logs a reminder:
  `"Back up store.duckdb before migrating: cp workspace/store.duckdb workspace/store.duckdb.bak"`

---

## Phase 3 — User-Facing API

### 3.1 Easy loading constructors

**File**: `predictions.py`

```python
class Predictions:
    def __init__(self, db_path: str | Path | None = None, *, dataset_name: str | None = None, load_arrays: bool = False):
        """Load predictions from a workspace store.

        Args:
            db_path: Path to workspace directory, store.duckdb file, or a single .parquet file.
                     If a .parquet file: loads standalone predictions from that file only
                     (portable mode — no DuckDB, limited metadata).
                     If a directory or .duckdb file: opens full workspace (DuckDB + Parquet arrays).
                     If None: creates an empty in-memory Predictions instance.
            dataset_name: Filter to specific dataset. None = all datasets.
            load_arrays: Whether to eagerly load dense arrays. Default False (lazy).
        """

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "Predictions":
        """Explicit file-based loading. Accepts workspace dir, .duckdb, or .parquet."""

    @classmethod
    def from_workspace(cls, workspace_path: str | Path, **kwargs) -> "Predictions":
        """Existing method — delegates to __init__."""

    @classmethod
    def from_parquet(cls, parquet_path: str | Path) -> "Predictions":
        """Load predictions from a standalone .parquet file (portable mode).
        All portable columns are available (model_name, fold_id, partition, metric, val_score).
        Arrays are loaded eagerly. Full metadata (scores, best_params, etc.) is not available.
        """
```

**Acceptance**:
- `Predictions("/path/to/workspace")` — full workspace mode.
- `Predictions.from_file("/path/to/store.duckdb")` — full workspace mode.
- `Predictions.from_parquet("arrays/wheat.parquet")` — portable mode, arrays + basic metadata.
- `Predictions("/path/to/wheat.parquet")` — auto-detects portable mode.

### 3.2 Merge databases

**File**: `predictions.py`

```python
@classmethod
def merge_stores(
    cls,
    sources: list[str | Path],
    target: str | Path,
    *,
    on_conflict: str = "keep_best",  # "keep_best" | "keep_latest" | "keep_all" | "skip"
    datasets: list[str] | None = None,
) -> MergeReport:
    """Merge multiple workspace stores into a single target store.

    Args:
        sources: List of workspace paths to merge from.
        target: Workspace path to merge into. Created if it does not exist.
        on_conflict: Strategy when same natural key exists in multiple sources:
            - "keep_best": keep the prediction with best val_score
            - "keep_latest": keep the most recently created prediction
            - "keep_all": keep all (may create duplicates if natural keys collide)
            - "skip": skip predictions that already exist in target
        datasets: Filter to specific datasets. None = merge all.

    Returns:
        MergeReport with counts per source, conflicts resolved, errors.
    """
```

Also keep the existing `merge_predictions(other)` instance method for in-memory merging.

**Acceptance**: merging 3 workspaces of 1 GB each into a single target completes without error, deduplicates correctly, arrays are present in the target Parquet files.

### 3.3 Maintenance helpers

**File**: `predictions.py` (instance methods on `Predictions` loaded from a store)

```python
def clean_dead_links(self, *, dry_run: bool = False) -> dict:
    """Remove predictions whose array data is missing (orphaned metadata).
    Also remove array data whose prediction metadata is missing (orphaned arrays).
    Returns: {metadata_orphans_removed: int, array_orphans_removed: int}.
    """

def remove_bottom(self, fraction: float, *, metric: str = "val_score", partition: str = "val",
                   dataset_name: str | None = None, dry_run: bool = False) -> dict:
    """Remove the bottom X% of predictions by score.
    Args:
        fraction: 0.0 to 1.0 — fraction to remove (e.g. 0.2 = remove worst 20%).
        metric: Score column to rank by.
        partition: Partition to evaluate.
        dataset_name: Filter to a dataset. None = all.
        dry_run: If True, return what would be removed without deleting.
    Returns: {removed: int, remaining: int, threshold_score: float}.
    """

def remove_dataset(self, dataset_name: str, *, dry_run: bool = False) -> dict:
    """Remove all predictions (metadata + arrays) for a dataset.
    Deletes the dataset's .parquet file entirely.
    Returns: {predictions_removed: int, parquet_deleted: bool}.
    """

def remove_run(self, run_id: str, *, dry_run: bool = False) -> dict:
    """Remove all predictions for a run. Cascades through pipelines, chains, predictions, arrays.
    Returns: {runs_removed: int, predictions_removed: int, arrays_removed: int}.
    """

def compact(self, dataset_name: str | None = None) -> dict:
    """Compact Parquet files: apply pending deletes, deduplicate.
    Returns: ArrayStore.compact() stats.
    """

def store_stats(self) -> dict:
    """Return combined stats: DuckDB metadata size, Parquet array size,
    row counts per table, per-dataset breakdown (file size, prediction count).
    """
```

### 3.4 Fast query access patterns

These already exist but are documented here for completeness and to ensure they remain fast after migration:

```python
# Already exists — metadata only, sub-second on DuckDB
predictions.top(10, rank_metric="rmse", rank_partition="val")
predictions.get_best(metric="r2")
predictions.filter_predictions(dataset_name="wheat", partition="val")
predictions.to_dataframe()  # Polars DataFrame of metadata

# New convenience: direct DuckDB SQL for power users
predictions.query(sql: str) -> pl.DataFrame
    """Run arbitrary read-only SQL against the metadata store.
    Arrays are not included (use load_arrays on results).
    Example: predictions.query("SELECT dataset_name, COUNT(*) FROM predictions GROUP BY 1")
    """
```

---

## Phase 4 — Cutover & Cleanup

### 4.1 Auto-migration on open

Once Phases 1-3 are validated, update `store_schema.py::_migrate_schema` to auto-migrate legacy stores on open:

```python
# Migration: move prediction_arrays to Parquet sidecar
if _table_exists(conn, "prediction_arrays"):
    _migrate_arrays_to_parquet(conn, workspace_path)
    conn.execute("DROP TABLE prediction_arrays")
```

This replaces the "log warning" behavior from Phase 2.4 with automatic migration.

### 4.2 Remove dead code

- Remove `save_prediction_arrays()` and `get_prediction_arrays()` DuckDB methods from `workspace_store.py`.
- Remove `INSERT_PREDICTION_ARRAYS`, `DELETE_PREDICTION_ARRAYS`, `GET_PREDICTION_ARRAYS` from `store_queries.py`.
- Remove `prediction_arrays` from `SCHEMA_DDL` in `store_schema.py`.
- Remove `_to_list()`, `_to_int_list()` helper methods.
- Remove the `y_proba` flattening workaround (shape is now stored in Parquet).
- Remove legacy-store detection logic from `workspace_store.py::__init__` (no more fallback to DuckDB arrays).

---

## File Change Summary

| File | Phase | Changes |
|------|-------|---------|
| `pipeline/storage/array_store.py` | 1 | **New file** — ArrayStore class |
| `pipeline/storage/workspace_store.py` | 1, 4 | Replace array methods with ArrayStore delegation (P1), remove dead code (P4) |
| `pipeline/storage/store_schema.py` | 1, 4 | Remove `prediction_arrays` DDL (P1 for new stores, P4 auto-migrate old) |
| `pipeline/storage/store_queries.py` | 1, 4 | Remove array query constants (P1), final cleanup (P4) |
| `pipeline/storage/migration.py` | 2 | **New file** — migration function + CLI |
| `data/predictions.py` | 1, 3 | Batch flush (P1), constructors + merge + maintenance (P3) |

---

## Testing Strategy

### Unit tests

| Test | Phase | What it covers |
|------|-------|---------------|
| `test_array_store_save_load` | 1 | Round-trip: save_batch → load_batch, verify numpy equality |
| `test_array_store_single_file_per_dataset` | 1 | Two datasets → two `.parquet` files, correct content in each |
| `test_array_store_delete_compact` | 1 | Tombstone → compact → verify rows removed, file shrinks |
| `test_array_store_integrity_check` | 1 | Inject orphans/missing → check detected |
| `test_array_store_y_proba_shape` | 1 | 2D y_proba survives round-trip with correct shape |
| `test_array_store_portable_columns` | 1 | Parquet contains model_name, fold_id, partition, metric, val_score, task_type |
| `test_workspace_store_flush_batch` | 1 | flush() writes arrays in one batch, not row-by-row |
| `test_migration_roundtrip` | 2 | Create DuckDB store with arrays → migrate → verify all data accessible via Parquet |
| `test_migration_dry_run` | 2 | Dry run leaves store unchanged |
| `test_migration_verification` | 2 | Inject a bad row → verify migration detects mismatch |
| `test_predictions_easy_loading` | 3 | `Predictions(path)`, `Predictions.from_file(path)` |
| `test_predictions_from_parquet` | 3 | `Predictions.from_parquet("wheat.parquet")` — portable mode, basic ranking works |
| `test_merge_stores` | 3 | Merge 2 stores → verify dedup and array presence |
| `test_clean_dead_links` | 3 | Inject orphans → clean → verify |
| `test_remove_bottom` | 3 | Add 100 predictions → remove bottom 20% → verify 80 remain |
| `test_remove_dataset_deletes_parquet` | 3 | `remove_dataset("wheat")` → `wheat.parquet` no longer exists |

### Integration tests

| Test | Phase | What it covers |
|------|-------|---------------|
| `test_full_pipeline_with_parquet_arrays` | 1 | `nirs4all.run()` → `predictions.top()` → arrays load correctly |
| `test_migrate_real_workspace` | 2 | Migrate a test workspace with 1 000 predictions, verify end-to-end |

### Performance benchmarks (not blocking, tracked)

| Benchmark | Target |
|-----------|--------|
| Load 10 000 predictions (metadata only) | < 1 s |
| Load 10 000 predictions (with arrays) | < 5 s |
| Flush 1 000 predictions | < 5 s |
| `top(10)` on 100 000 predictions | < 200 ms |
| Migration of 1 GB store | < 60 s |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Crash between Parquet write and DuckDB insert | Low | Medium (orphan arrays) | Write order enforced; orphans cleaned by `compact()` |
| Single Parquet file per dataset grows very large | Medium | Medium (slow rewrites on compact) | Acceptable: even 1 M predictions per dataset is ~4 GB compressed; `compact()` streams and rewrites; for truly huge datasets this is still faster than DuckDB arrays |
| Migration fails mid-way on large store | Low | High (partial state) | Migration is incremental by dataset; each dataset is atomic; rollback = delete `arrays/` dir |
| Concurrent writers corrupt Parquet files | Low | High | Single-writer model inherited from DuckDB lock; documented as constraint |
| Portable metadata columns get out of sync with DuckDB | Low | Low (cosmetic) | Portable columns are written once at flush time from the same source; no ongoing sync needed |
| New dependency on PyArrow | — | — | PyArrow is already a transitive dependency via Polars; no new dep |

---

## Out of Scope

- Changing the relational schema (runs, pipelines, chains, predictions, artifacts, logs) — no changes
- Changing the content-addressed artifact storage — no changes
- Adding a new database backend (SQLite, Lance, TileDB) — not in this iteration
- Multi-writer support — not needed (single-writer is the existing model)
- Remote/cloud storage for Parquet files — not in this iteration

---

## Definition of Done

Each phase is independently shippable. A phase is done when:

1. All listed tests pass.
2. Existing test suite passes without modification (no regressions).
3. `ruff check .` clean.
4. Code reviewed and merged.

**Full project done** when Phase 4 is shipped and the `prediction_arrays` table no longer exists in any active workspace.
