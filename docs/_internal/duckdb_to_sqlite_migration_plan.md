# DuckDB to SQLite Migration Plan

Date: 2026-03-25
Context: Issue #37 (DuckDB locking prevents successive runs from the same workspace) and issue #36 (metadata-based prediction aggregation) both touch the storage layer. This document analyzes the migration path and how it interacts with pending features.

---

## Problem Statement

DuckDB enforces single-writer exclusivity at the file level. When a Python process holds a connection to `store.duckdb`, no other process (or even the same process in certain scenarios) can open it. This causes `IOError: Could not set lock on file` when:

1. Running successive `nirs4all.run()` calls on different datasets from the same workspace
2. Having a Jupyter notebook open with an active workspace while launching another run
3. Any scenario where a previous connection wasn't properly closed (crash, interrupt)

SQLite, in WAL mode, supports concurrent readers and serialized writers without exclusive file locks, making it a better fit for nirs4all's usage pattern.

---

## Current DuckDB Usage Inventory

### Files involved

| File | DuckDB usage |
|------|-------------|
| `pipeline/storage/store_schema.py` | DDL: table/view/index definitions |
| `pipeline/storage/store_queries.py` | All SQL queries (INSERT, SELECT, UPDATE, DELETE) |
| `pipeline/storage/workspace_store.py` | `duckdb.connect()`, connection management, retry-on-lock, query execution |
| `pipeline/storage/array_store.py` | Parquet files only (no DuckDB dependency) |
| `data/predictions.py` | Uses WorkspaceStore API (no direct DuckDB calls) |
| `pipeline/execution/orchestrator.py` | Creates WorkspaceStore instance |

### DuckDB-specific features used

| Feature | Where | SQLite equivalent |
|---------|-------|-------------------|
| `duckdb.connect()` | workspace_store.py:250 | `sqlite3.connect()` |
| `PRAGMA enable_progress_bar=false` | workspace_store.py:253 | Not needed |
| `JSON` column type | store_schema.py (config, scores, etc.) | `TEXT` + json functions or `JSON` (SQLite 3.38+) |
| `DOUBLE` type | store_schema.py | `REAL` |
| `BIGINT` type | store_schema.py | `INTEGER` |
| `TIMESTAMP DEFAULT current_timestamp` | store_schema.py | Same syntax works |
| `VARCHAR` type | store_schema.py | `TEXT` |
| `CREATE INDEX IF NOT EXISTS` | store_schema.py | Same syntax |
| `REFERENCES` (foreign keys) | store_schema.py | Same syntax, need `PRAGMA foreign_keys=ON` |
| `fetchone()`, `fetchall()` | workspace_store.py | Same API |
| Parameterized queries (`?` or `$1`) | store_queries.py | `?` placeholders |
| Thread safety with RLock | workspace_store.py:247 | `check_same_thread=False` + RLock |

### DuckDB-specific features NOT used

- Columnar analytics / OLAP queries
- Parquet integration (done separately via ArrayStore)
- Vector types
- Window functions (beyond basic SQL)
- DuckDB extensions

**Key insight**: The DuckDB usage is essentially relational CRUD. There are no columnar analytics or DuckDB-specific query features that would be hard to port.

---

## Migration Strategy

### Approach: Thin abstraction + direct replacement

Since all DuckDB access goes through `WorkspaceStore`, the migration is contained:

1. **Replace `duckdb` import with `sqlite3`** in `workspace_store.py`
2. **Adjust DDL types** in `store_schema.py` (mostly cosmetic: `VARCHAR`→`TEXT`, `DOUBLE`→`REAL`)
3. **Adjust query syntax** in `store_queries.py` (parameter style: DuckDB uses `$1`, SQLite uses `?`)
4. **Enable WAL mode** on connect: `PRAGMA journal_mode=WAL`
5. **Enable foreign keys**: `PRAGMA foreign_keys=ON`
6. **JSON handling**: SQLite 3.38+ has built-in JSON functions. For older versions, store JSON as TEXT and parse in Python.

### File changes estimate

| File | Change scope |
|------|-------------|
| `store_schema.py` | Type name adjustments. ~30 min. |
| `store_queries.py` | Parameter placeholder style (`$1` → `?`). Grep and replace. ~1 hour. |
| `workspace_store.py` | Connection management rewrite. Drop retry-on-lock logic (WAL eliminates this). ~2-3 hours. |
| `requirements` / `pyproject.toml` | Remove `duckdb` dependency, `sqlite3` is stdlib. |
| Tests | Update any DuckDB-specific assertions. ~1 hour. |

### Data migration tool

Existing workspaces with `store.duckdb` need a migration path:

```python
def migrate_duckdb_to_sqlite(workspace_path: Path) -> None:
    """One-time migration from store.duckdb to store.sqlite."""
    # 1. Open DuckDB read-only
    # 2. Create SQLite with new schema
    # 3. Copy all rows table by table
    # 4. Rename store.duckdb -> store.duckdb.bak
    # 5. Verify SQLite integrity
```

This should be triggered automatically when `WorkspaceStore.__init__` detects `store.duckdb` but no `store.sqlite`.

---

## Interaction with Issue #36 (Metadata in Predictions)

Issue #36 requests the ability to aggregate predictions using metadata columns that weren't part of the pipeline. This requires schema changes to the predictions table (or a sidecar metadata table).

### Should #36 wait for the migration?

**Yes, preferably.** Reasons:

1. **Avoid double schema migration**: Adding metadata columns to DuckDB now means writing a DuckDB migration AND a DuckDB→SQLite migration. Better to do it once on SQLite.
2. **Schema changes are simpler in SQLite**: Adding columns or tables during development is straightforward with `ALTER TABLE`.
3. **The locking fix (#37) is a prerequisite anyway**: If users can't run successive queries on the same workspace, they can't effectively use metadata-based aggregation.

### If #36 can't wait

If the feature is needed before migration, implement it storage-agnostically:

- Add a `sample_metadata` JSON column to the `predictions` table (both DuckDB and eventual SQLite support this)
- Or create a `prediction_metadata` sidecar Parquet file alongside the existing array Parquet files
- Keep the aggregation logic in `Predictions` class (Python-side, not SQL-side)

---

## Implementation Order

Recommended sequence:

1. **Migrate DuckDB → SQLite** (fixes #37, removes external dependency)
2. **Add metadata storage to predictions** (implements #36 on the new SQLite schema)
3. **Add prediction aggregation by metadata** (builds on #36 metadata storage)

This order avoids rework and resolves the blocking locking issue first.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Performance regression for large workspaces | SQLite WAL mode is fast for write-heavy CRUD. Benchmark with 10K+ predictions. |
| JSON query performance in SQLite | Keep JSON queries simple (extract, not filter). Complex filtering stays in Python. |
| Existing user workspaces break | Auto-migration tool + backup of original DuckDB file |
| `sqlite3` version too old for JSON | Require Python 3.11+ (bundles SQLite 3.39+). Already the minimum Python version. |
| Concurrent write contention in WAL | WAL handles this with retry. Much better than DuckDB's exclusive lock. |

---

## What Does NOT Change

- **ArrayStore** (Parquet files) — completely independent of the DB engine
- **Predictions in-memory API** — `top()`, `get_best()`, `flush()` interface stays the same
- **Artifact storage** — content-addressed joblib files, no DB dependency
- **Run manifests** — YAML files, no DB dependency
