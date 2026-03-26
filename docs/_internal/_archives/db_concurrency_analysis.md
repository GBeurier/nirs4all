# DuckDB Concurrency & Crash Recovery Analysis

**Date**: 2026-02-24
**Status**: Analysis complete — action proposals ready for backlog

---

## Executive Summary

The current DuckDB integration has **fundamental lifecycle and concurrency flaws** that cause three classes of failures:
1. **Orphaned lock/WAL files** after crashes or ungraceful exits
2. **Lock conflicts during parallel execution** (multi-process writes)
3. **Silent data loss** via irreversible degraded mode

The root cause is architectural: DuckDB connections are **never closed** in the primary code path (`nirs4all.run()`), and the retry/degradation mechanism masks errors rather than recovering from them. This is **fixable without changing the storage engine**, but requires disciplined lifecycle management and a redesigned error recovery strategy.

---

## 1. Root Cause Analysis

### 1.1 Store Never Closed (CRITICAL)

**The primary path — `nirs4all.run()` — never closes the DuckDB connection.**

Call chain:
```
nirs4all.run()
  → PipelineRunner(**kwargs)
    → PipelineOrchestrator.__init__()          # Opens DuckDB at line 106
      → WorkspaceStore(workspace_path)          # duckdb.connect() at line 247
  → runner.run(pipeline, dataset)
    → orchestrator.execute()                    # Uses store throughout
  → return RunResult(predictions, _runner=runner)  # Runner (and store) kept alive
  # ← NO close() EVER CALLED
```

**Files involved:**
- `pipeline/storage/workspace_store.py:247` — `duckdb.connect()` opens the connection
- `pipeline/storage/workspace_store.py:373-382` — `close()` method exists but is never called
- `pipeline/execution/orchestrator.py:106` — creates store, never closes it
- `api/run.py:395-503` — creates runner, returns RunResult, no cleanup

**Consequence:** The DuckDB connection is held open for the entire lifetime of the `RunResult` object. When the Python process exits (or crashes), the connection is abandoned. DuckDB's WAL file and potentially its lock file are left on disk. The next `nirs4all.run()` call may encounter stale locks.

**Evidence:** Three orphaned WAL files exist in the repository:
```
bench/tabpfn_paper/test_workspace/store.duckdb.wal
bench/tabpfn_paper/AOM_workspace/store.duckdb.wal
bench/tabpfn_paper/workspace/store.duckdb.wal
```

**Who DOES close?** Only two callers close correctly:
- `Session.__exit__()` (api/session.py:399-401) — context manager
- `predict()` API (api/predict.py:227) — try/finally

### 1.2 Misleading WAL Configuration

```python
# workspace_store.py lines 249-255
# Enable WAL mode for concurrent read/write          ← COMMENT
self._conn.execute("PRAGMA enable_progress_bar=false")  ← ACTUAL CODE (unrelated!)

self._conn.execute("SET memory_limit = '2GB'")
self._conn.execute("SET threads = 4")
self._conn.execute("SET checkpoint_threshold = '256MB'")
```

The comment says "Enable WAL mode" but the code does not set any WAL-related pragma. DuckDB defaults to WAL mode for file-based databases, so this isn't technically wrong — but the misleading comment suggests the developer intended to configure WAL behavior and never did. No `busy_timeout`, no `wal_autocheckpoint`, no explicit journal mode.

### 1.3 Degraded Mode Is Irreversible and Silent (CRITICAL)

**Location:** `workspace_store.py:108-144` and `292-334`

When a DuckDB `TransactionException` persists after 5 retries (~3.1 seconds total), the store sets `self._degraded = True`. After that:

```python
def _retry_on_lock(func):
    def wrapper(self, *args, **kwargs):
        if self._degraded:
            return None          # ← Silently skips ALL future writes
        ...
```

**Problems:**
1. **Irreversible** — no mechanism to clear `_degraded`. Once set, the store is permanently broken for the lifetime of the Python process.
2. **Silent** — returns `None` instead of raising. Callers don't check return values. The user sees "run completed" but predictions were never persisted.
3. **Cascading** — one transient lock conflict (e.g., from an orphaned WAL) permanently disables persistence for all subsequent operations.

### 1.4 Thread Lock Doesn't Protect Against Process-Level Conflicts

```python
# workspace_store.py:243
self._lock = threading.RLock()
```

This `RLock` serializes access within a single process. But the real contention is **cross-process**: DuckDB uses OS-level file locks on the `.duckdb` file. When parallel workers run via `joblib/loky` (separate processes), the main process holds the file lock. If the main process crashes mid-execution, the OS should release the lock — but:
- On network filesystems (NFS, SMB): locks may persist indefinitely
- On WSL2: file lock behavior can be inconsistent with Windows-side processes
- DuckDB's internal lock acquisition timeout is very short

### 1.5 Parallel Execution Architecture

**Location:** `pipeline/execution/orchestrator.py:290-489`

The parallel path correctly avoids multi-process DuckDB access:

```python
parallel_executor.store = None   # line 318
runtime_context_copy = RuntimeContext(store=None, ...)  # line 347
```

Workers execute variants without touching DuckDB. After workers complete, the main process reconstructs store state by writing pipeline records, chains, and predictions back to DuckDB in a loop (lines 397-488).

**But the vulnerability is:**
1. The main process holds the DuckDB connection open during the entire parallel phase
2. If ANY worker crashes (loky kills the process), the main process may be interrupted
3. The reconstruction loop (lines 397-488) wraps each variant in try/except, but a persistent lock conflict causes degraded mode → all remaining variants lose their store records
4. The reconstruction does multiple sequential writes per variant (begin_pipeline, register_artifact × N, save_chain × N, flush_predictions, complete_pipeline) — any single write failure triggers degradation

### 1.6 Schema Migration Is Not Transactional

**Location:** `pipeline/storage/store_schema.py:438-469`

```python
def create_schema(conn, workspace_path=None):
    for statement in SCHEMA_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)  # Each DDL auto-committed separately
```

If the process crashes mid-migration, the database is in a partially migrated state. Since each statement uses `IF NOT EXISTS`, this is mostly safe for table creation, but the `_migrate_schema()` call at line 455 does `ALTER TABLE` and `DROP TABLE` operations that are not idempotent.

### 1.7 Migration Function Has Unclosed Connection

**Location:** `pipeline/storage/migration.py:110`

```python
conn = duckdb.connect(str(db_path))
try:
    ...
    conn.execute("DROP TABLE prediction_arrays")
    conn.execute("VACUUM")
    ...
finally:
    # Connection IS closed in this path (line ~200)
```

The main `migrate_arrays_to_parquet` function does use try/finally correctly. However, `verify_migration` at line 332 also closes properly. This specific issue from initial analysis was a false alarm — the migration code is correctly written.

---

## 2. Diagnosis: Fixable or Restructure?

### Is DuckDB the right tool?

DuckDB is designed as an **in-process OLAP database**. Its concurrency model:
- **Single writer** at a time (file-level lock)
- **Multiple readers** concurrent with one writer (WAL mode)
- **Not designed for multi-process writes**

Our usage:
- Single writer (main process) — **compatible**
- Workers read-only or no-access — **compatible**
- Connection held open indefinitely — **incompatible with crash safety**
- No close() calls — **incompatible with lock management**

**Verdict: DuckDB is appropriate for this use case.** The issues are in our lifecycle management, not in DuckDB's capabilities. We don't need multi-process writes — workers correctly use `store=None`. The problems are:
1. We don't close connections when done
2. We don't recover gracefully from stale locks
3. We enter an irreversible degraded mode on any lock conflict

### What would "restructure" mean?

Switching to SQLite or another engine would have the same issues if we don't close connections. Switching to a file-based approach (JSON/YAML manifests) would lose query capabilities. The current hybrid architecture (DuckDB metadata + Parquet arrays) is sound — we just need to fix the lifecycle.

---

## 3. Action Proposals

### P0: Critical — Fix Store Lifecycle (must-do, high impact)

#### A1: Add `__del__` safety net to WorkspaceStore

```python
def __del__(self) -> None:
    if self._conn is not None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None
```

This is a safety net, not a primary mechanism. Python's `__del__` is not guaranteed to run, but it catches the common case of abandoned stores. Low effort, immediate benefit.

#### A2: Close store after orchestrator.execute() completes

In `api/run.py`, after the run loop completes, explicitly close the store:

```python
try:
    for pipeline_idx, single_pipeline in enumerate(pipelines):
        ...
except Exception:
    ...
    raise
finally:
    # Close the store to release DuckDB file locks
    store = getattr(getattr(runner, 'orchestrator', None), 'store', None)
    if store is not None:
        store.close()
```

This is the primary fix. The RunResult should not hold a live store reference. If post-run operations need the store (e.g., `result.export()`), they should reopen it.

#### A3: Add context manager protocol to WorkspaceStore

```python
def __enter__(self) -> "WorkspaceStore":
    return self

def __exit__(self, *exc) -> None:
    self.close()
```

Then callers can use `with WorkspaceStore(path) as store:` for deterministic cleanup.

#### A4: Checkpoint + close at orchestrator boundaries

After `orchestrator.execute()` returns (line 763), force a WAL checkpoint before closing:

```python
def close(self) -> None:
    with self._lock:
        if self._conn is not None:
            try:
                self._conn.execute("CHECKPOINT")
            except Exception:
                pass
            self._conn.close()
            self._conn = None
```

This ensures the WAL is flushed to the main database file before releasing the lock.

### P1: High — Fix Degraded Mode

#### B1: Replace irreversible degradation with per-operation retry

Remove the `_degraded` flag entirely. Instead, each write operation should:
1. Try the operation with retry
2. On persistent failure, log a warning and skip **that specific operation**
3. NOT set a global flag that disables all future operations

```python
def _execute_with_retry(self, sql, params=None, *, max_retries=_MAX_RETRIES, base_delay=_BASE_DELAY):
    with self._lock:
        conn = self._ensure_open()
        for attempt in range(max_retries + 1):
            try:
                conn.execute(sql, params or [])
                return True
            except duckdb.TransactionException as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("DuckDB lock conflict (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    time.sleep(delay)
        logger.error("DuckDB write failed after %d retries — this operation skipped", max_retries)
        return False  # Caller can check, but doesn't cascade
```

#### B2: Add stale lock recovery

Before entering retry, check if the lock file is stale:

```python
def _try_recover_stale_lock(self) -> bool:
    """Attempt recovery from stale DuckDB locks (e.g., after crash)."""
    wal_path = self._workspace_path / "store.duckdb.wal"
    if wal_path.exists():
        logger.warning("Found orphaned WAL file: %s — attempting recovery", wal_path)
        try:
            # Close current connection
            if self._conn is not None:
                self._conn.close()
            # Reopen — DuckDB will replay the WAL automatically
            db_path = self._workspace_path / "store.duckdb"
            self._conn = duckdb.connect(str(db_path))
            self._conn.execute("CHECKPOINT")  # Force WAL flush
            logger.info("WAL recovery successful")
            return True
        except Exception as e:
            logger.error("WAL recovery failed: %s", e)
    return False
```

#### B3: Increase retry budget

The current 3.1s budget is too aggressive. Use 10+ seconds with jitter:

```python
_MAX_RETRIES = 8
_BASE_DELAY = 0.2  # 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6 → ~50s total
```

Add random jitter to avoid thundering herd:

```python
delay = base_delay * (2 ** attempt) * (0.5 + random.random())
```

### P2: Medium — Robustness Improvements

#### C1: Reconnect on connection loss

If `_ensure_open()` fails or a read raises `IOException`, automatically reconnect:

```python
def _ensure_open(self) -> duckdb.DuckDBPyConnection:
    if self._conn is None:
        db_path = self._workspace_path / "store.duckdb"
        self._conn = duckdb.connect(str(db_path))
        # Re-apply pragmas
        self._apply_pragmas()
        # Re-create schema (idempotent)
        create_schema(self._conn, workspace_path=self._workspace_path)
    return self._conn
```

#### C2: Fix misleading WAL comment

Replace:
```python
# Enable WAL mode for concurrent read/write
self._conn.execute("PRAGMA enable_progress_bar=false")
```
With:
```python
self._conn.execute("PRAGMA enable_progress_bar=false")
```

Just remove the misleading comment. DuckDB defaults to WAL mode for file databases.

#### C3: Clean orphaned temp files in ArrayStore

After crash, `.parquet.tmp` files can remain in `arrays/`. Add a cleanup on init:

```python
# In ArrayStore.__init__
for tmp in (self._arrays_dir).glob("*.parquet.tmp"):
    tmp.unlink(missing_ok=True)
```

#### C4: Batch store reconstruction after parallel execution

Instead of writing records one-by-one in the parallel reconstruction loop (which does 5+ writes per variant), batch them into a single transaction:

```python
# orchestrator.py parallel reconstruction
conn = self.store._ensure_open()
conn.execute("BEGIN TRANSACTION")
try:
    for result in successful_results:
        # ... all writes ...
    conn.execute("COMMIT")
except Exception:
    conn.execute("ROLLBACK")
    raise
```

This reduces the window for lock conflicts from N × 5 writes to a single transaction.

### P3: Low — Quality of Life

#### D1: Add store health check

```python
def health_check(self) -> dict:
    """Return store health status."""
    wal_exists = (self._workspace_path / "store.duckdb.wal").exists()
    return {
        "connected": self._conn is not None,
        "degraded": self._degraded,
        "wal_pending": wal_exists,
        "db_size": (self._workspace_path / "store.duckdb").stat().st_size,
    }
```

#### D2: Add CLI command for WAL recovery

```bash
nirs4all workspace recover /path/to/workspace
```

Opens the store, forces a checkpoint, closes cleanly. Simple but useful after crashes.

#### D3: Log store lifecycle events

```python
logger.debug("WorkspaceStore opened: %s", db_path)
# ... on close:
logger.debug("WorkspaceStore closed: %s", db_path)
```

---

## 4. Backlog (Prioritized)

| # | Action | Priority | Effort | Impact | Description |
|---|--------|----------|--------|--------|-------------|
| 1 | A2 | P0 | S | Critical | Close store in `api/run.py` after run completes |
| 2 | A4 | P0 | S | Critical | Checkpoint before close in `WorkspaceStore.close()` |
| 3 | A1 | P0 | XS | High | Add `__del__` safety net to WorkspaceStore |
| 4 | A3 | P0 | XS | High | Add `__enter__`/`__exit__` to WorkspaceStore |
| 5 | B1 | P1 | M | Critical | Replace irreversible degraded mode with per-op skip |
| 6 | B3 | P1 | S | High | Increase retry budget to ~50s with jitter |
| 7 | C2 | P2 | XS | Low | Fix misleading WAL comment |
| 8 | C4 | P2 | M | Medium | Batch parallel reconstruction in single transaction |
| 9 | C1 | P2 | S | Medium | Auto-reconnect on connection loss |
| 10 | B2 | P2 | M | Medium | Add stale lock recovery (reopen + checkpoint) |
| 11 | C3 | P2 | XS | Low | Clean orphaned `.parquet.tmp` on ArrayStore init |
| 12 | D2 | P3 | S | Low | CLI `workspace recover` command |
| 13 | D1 | P3 | XS | Low | Store health check method |
| 14 | D3 | P3 | XS | Low | Log store lifecycle events |

**Effort scale**: XS = <30 min, S = 1-2h, M = half-day

### Recommended Implementation Order

**Phase 1 — Stop the bleeding (items 1-4, 6-7):**
Close connections properly, add safety nets. This alone will eliminate most orphaned WAL/lock issues and prevent the degraded mode from being triggered in the first place.

**Phase 2 — Fix recovery (items 5, 8-10):**
Make the system resilient to edge cases. Replace irreversible degradation with graceful per-operation handling. Batch parallel reconstruction to minimize lock exposure.

**Phase 3 — Polish (items 11-14):**
Quality of life improvements for debugging and maintenance.

---

## 5. Files Affected

| File | Issue | Fix |
|------|-------|-----|
| `pipeline/storage/workspace_store.py` | No `__del__`, no `__enter__/__exit__`, misleading comment, irreversible degraded mode | A1, A3, A4, B1, B2, B3, C1, C2 |
| `api/run.py` | Never closes store after run | A2 |
| `pipeline/execution/orchestrator.py` | Reconstruction loop not batched | C4 |
| `pipeline/storage/array_store.py` | No temp file cleanup | C3 |
| `cli/commands/workspace.py` | Some commands don't close store (lines 56-127) | A3 (context manager) |
| `data/predictions.py` | `from_workspace()` opens store, doesn't always close | A3 |

---

## 6. Independent Review (Fact Check + Gaps)

This section audits the analysis above against the current codebase and DuckDB documentation.

### 6.1 Fact-check results

| Claim | Verdict | Review notes |
|---|---|---|
| `nirs4all.run()` keeps a live DuckDB connection via `RunResult(_runner=runner)` and does not close it | ✅ Correct | Verified in `api/run.py`, `api/result.py`, `pipeline/runner.py`, `pipeline/execution/orchestrator.py`, `pipeline/storage/workspace_store.py`. |
| Only `Session.__exit__()` and `predict()` close correctly | ⚠️ Incomplete | `predict()` only closes in the `chain_id` path. The model-based `predict()` path builds a `PipelineRunner` without an explicit close. |
| WAL configuration is missing (`no wal_autocheckpoint`) | ❌ Incorrect | `workspace_store.py` sets `checkpoint_threshold = '256MB'`; in DuckDB this is the alias of `wal_autocheckpoint`. The misleading comment is still real, but WAL checkpoint tuning is present. |
| Parallel execution causes multi-process write contention in current architecture | ⚠️ Partially outdated | Worker processes explicitly run with `store=None`; writes happen in main process reconstruction. Current lock risk is primarily from external concurrent processes or a previously degraded store, not worker-side writes. |
| Degraded mode is irreversible and can silently drop persistence | ✅ Correct (severity understated) | Beyond decorated methods returning `None`, `_execute_with_retry()` no-ops when degraded. Critical lifecycle methods can return generated IDs even when nothing was written. |
| Schema migration is non-transactional and unsafe | ⚠️ Partially correct | Per-statement execution is non-transactional, but many migration steps are idempotent. The larger risk is `_auto_migrate_prediction_arrays()` dropping the legacy table without explicit verification/rollback. |
| Migration module leaves unclosed connections | ✅ Corrected in document | `migration.py` closes connections properly in `finally` blocks. |

### 6.2 Additional problems not captured above

1. `api/retrain.py` returns `RunResult(_runner=runner)` and has the same open-store lifecycle issue as `api/run.py`.
2. CLI workspace commands (`workspace_list_runs`, `workspace_query_best`, `workspace_query_filter`, `workspace_stats`) open `WorkspaceStore` and do not close it.
3. Degraded mode can corrupt parity between metadata and arrays:
   - `save_prediction()` can return `None` in degraded mode.
   - `Predictions.flush()` can still append array rows using that `None` prediction ID.
4. Query-only flows always open a read-write store. This increases lock contention risk and is avoidable for read-only commands.
5. `PipelineRunner` always builds `PipelineOrchestrator` + `WorkspaceStore`, even in predict/explain flows that may not require DB access.

### 6.3 Review of proposed fixes (what to adjust)

1. Keep the diagnosis that DuckDB is viable, but update wording from “parallel worker writes” to “lifecycle + degraded-mode + external process contention.”
2. Reframe A2 (close in `api/run.py`) to avoid breaking post-run APIs:
   - add `PipelineRunner.close()` and `RunResult.close()` (plus context-manager support),
   - keep explicit caller-controlled lifecycle instead of implicit immediate close.
3. Replace “skip failed write and continue” (B1) for critical writes with fail-fast behavior:
   - critical operations (`begin_*`, `save_chain`, `save_prediction`, `complete_*`) should raise after retries,
   - non-critical logging/status writes may degrade gracefully.
4. Drop WAL-file-based stale-lock recovery heuristic (B2): WAL presence alone is not a reliable stale-lock signal.
5. Keep retry/backoff improvements, but make timeout budget configurable and bounded (avoid fixed ~50s global stalls).
6. Move/guard auto-migration:
   - prefer explicit migration command path with verification,
   - avoid destructive table-drop in implicit startup migration without a verified checkpoint.

### 6.4 Suggested revised priority

1. **P0**: lifecycle closure primitives (`PipelineRunner.close`, `RunResult.close`), remove silent degraded no-ops for critical writes.
2. **P0**: fix CLI/API call sites to close stores deterministically (including `retrain` and query commands).
3. **P1**: introduce read-only store mode for query paths.
4. **P1**: harden migration strategy (explicit verified migration; reduce implicit destructive behavior).
5. **P2**: comment/logging cleanup, optional temp-file cleanup, retry tuning refinements.

### 6.5 Documentation/source anchors used for fact-check

- DuckDB docs: [Handling Concurrency](https://duckdb.org/docs/stable/connect/concurrency)
- DuckDB docs: [Configuration Overview (`duckdb_settings`)](https://duckdb.org/docs/stable/configuration/overview)
- DuckDB docs: [ALTER TABLE (transactional semantics)](https://duckdb.org/docs/stable/sql/statements/alter_table)
