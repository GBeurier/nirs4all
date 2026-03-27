# Storage Migration Guide -- Webapp Developers

## Summary of changes

| Before (v0.8) | After (v0.9) |
|---|---|
| `store.duckdb` | `store.sqlite` |
| `duckdb` required dependency | `sqlite3` (stdlib) |
| `duckdb.connect()` | `sqlite3.connect()` |
| Exclusive file locking | WAL: concurrent readers + 1 writer |
| Parameters `$1, $2, $3` | Parameters `?, ?, ?` |

## Impact on the webapp

### No change to StoreAdapter

`StoreAdapter` only uses the public methods of `WorkspaceStore`.  All
methods are identical (same names, same arguments, same return types).

### File detection

The webapp searches for `store.sqlite` first, then `store.duckdb` as a
legacy fallback.  When an old `store.duckdb` is found,
`WorkspaceStore.__init__` automatically triggers migration to
`store.sqlite`.

### Polars DataFrame

`WorkspaceStore` query methods still return `pl.DataFrame`.  The internal
implementation changed (no more DuckDB zero-copy `.pl()`; Polars
DataFrames are now built from SQLite result rows), but the return type is
identical.

### Error messages and docstrings

All references to "DuckDB" in webapp error messages and docstrings have
been replaced with "workspace store" or "SQLite".

## Modified webapp files

| File | Nature of change |
|---|---|
| `api/workspace_manager.py` | Detection: `store.sqlite` + fallback `store.duckdb` |
| `api/aggregated_predictions.py` | Same + updated docstrings |
| `api/workspace.py` | Updated docstrings |
| `api/predictions.py` | Updated deprecation messages |
| `api/store_adapter.py` | No change (already database-agnostic) |
| `tests/test_aggregated_predictions_api.py` | `store.duckdb` -> `store.sqlite` in fixtures |
| `tests/test_store_integration.py` | Same |
| `docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md` | ~50 references updated |

## What the webapp does NOT need to do

- Install duckdb (was never a direct webapp dependency)
- Modify API endpoints (same routes, same responses)
- Modify the frontend (no visible UI change)
- Modify Pydantic models (same schemas)
