# Migration DuckDB to SQLite (v0.9)

Starting with v0.9, nirs4all uses **SQLite WAL** instead of DuckDB as the
metadata storage engine.  This resolves multi-process locking issues and
removes the ~50 MB `duckdb` binary dependency.

## What changes

### Workspace file

- The metadata file changes from `store.duckdb` to `store.sqlite`.
- Migration is **automatic**: the first time you open a pre-v0.9 workspace,
  nirs4all converts `store.duckdb` to `store.sqlite` and renames the
  original to `store.duckdb.bak`.

### duckdb dependency

- `duckdb` is no longer a required dependency.
- To migrate an existing workspace that still has `store.duckdb`:
  `pip install nirs4all[migration]`.
- New workspaces do not need duckdb at all.

### Identical behaviour

- All public APIs are unchanged: `run()`, `predict()`, `explain()`, etc.
- `WorkspaceStore` method signatures and return types (`pl.DataFrame`) are
  identical.
- `ArrayStore` (Parquet) and `artifacts/` (joblib) are unaffected.

## What does NOT change

| Component | Change |
|-----------|--------|
| `nirs4all.run()` | None |
| `nirs4all.predict()` | None |
| `nirs4all.explain()` | None |
| `Predictions` API | None |
| `WorkspaceStore` public signatures | None |
| `ArrayStore` (Parquet) | None |
| `artifacts/` (joblib) | None |
| Pipeline syntax | None |

## Required actions

### Python library users

No action for new workspaces.

For an existing workspace that still contains `store.duckdb`, migration is
transparent **if** duckdb is available in your environment.  Otherwise
install the migration extra once:

```bash
pip install nirs4all[migration]
```

### Developers importing WorkspaceStore directly

`WorkspaceStore` keeps exactly the same methods and signatures.  The only
visible change is the database filename (`store.sqlite` instead of
`store.duckdb`).

### Webapp developers

See [storage_migration_webapp.md](storage_migration_webapp.md) for details.
