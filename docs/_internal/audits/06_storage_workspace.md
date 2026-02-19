# Storage and Workspace Modules - Release Blockers Only (2026-02-19)

## Active Findings

(none)

## Beta Release Tasks (Open)

(none)

## Resolved Findings
- `S-01 [RESOLVED]` `save_chain()` TOCTOU window eliminated — SELECT (dataset name resolution) and INSERT now execute inside the same `with self._lock:` block (`pipeline/storage/workspace_store.py`).
- `S-02 [RESOLVED]` `pickle.loads` trust boundary explicitly documented: `_deserialize_artifact` docstring, `nosec` annotation, debug logging, and inline comment at the call site in `load_artifact()`.
- `S-03 [RESOLVED]` `ArrayStore` cross-process write safety documented in module-level docstring and `save_batch()` method docstring. Workers use `store=None` in parallel execution to avoid concurrent Parquet writes (`pipeline/storage/array_store.py`).
- `S-04 [RESOLVED]` DuckDB tuning externalized via `DuckDBConfig` dataclass with `memory_limit`, `threads`, and `checkpoint_threshold` fields. `WorkspaceStore.__init__` accepts optional `duckdb_config` parameter.
- `S-05 [RESOLVED]` Retry logic unified: `_retry_on_lock` decorator now delegates to `_call_with_retry`, the single retry implementation shared with `_execute_with_retry`. No duplicated backoff/degraded-mode logic.
- `S-06 [RESOLVED]` Auto-migration now controllable via `auto_migrate: bool = True` parameter on `WorkspaceStore.__init__`. When `False`, `create_schema` runs without the workspace path so legacy migrations are skipped.
- `S-07 [RESOLVED]` `bulk_update_chain_summaries` temp-table hardened: unique `uuid4` suffix per call prevents name collisions; `try/finally` guarantees cleanup.
- `S-08 [RESOLVED]` Dedicated concurrent-write thread-safety tests added for `ArrayStore` (`tests/unit/pipeline/storage/test_array_store.py::TestArrayStoreConcurrentWrites`).
