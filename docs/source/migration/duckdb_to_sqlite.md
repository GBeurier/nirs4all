# Explicit DuckDB workspace conversion

Current nirs4all workspaces use `store.sqlite`. A workspace containing
`store.duckdb`, or a SQLite store containing the historical
`prediction_arrays` table, is a legacy source and is never migrated when the
runtime opens it.

## Runtime behaviour

Detection is read-only. `WorkspaceStore` raises `ConversionRequired` with
conversion guidance before it opens a recognized legacy store. It does not
rename, delete, or write the source, and it never creates a `.bak` file.

The explicit `engine="legacy"` runtime remains available through the agreed
rollback window, until after R4, for its supported workflows. It does not
enable DuckDB workspace conversion or make migration implicit.

## Convert into a separate output

The `nirs4all-tools` candidate package provides the installed console command;
a source checkout is not required. Candidate packages are currently
**unpublished**, so install the locally supplied candidate wheel rather than
assuming a package registry version exists. Include the `duckdb` and `parquet`
extras when converting a DuckDB workspace.

```bash
python -m pip install "./nirs4all_tools-<candidate>-py3-none-any.whl[duckdb,parquet]"

nirs4all-tools workspace inspect /data/workspace
nirs4all-tools workspace convert /data/workspace \
  --output /data/workspace-r2 --verify
```

`inspect` and `convert` leave the source path, inode, and bytes intact. The
output must be new and disjoint from the source; conversion is one-way and
never performed in place.

The conversion domain codes are:

| Code | Meaning |
|---:|---|
| `0` | Fully converted without warnings |
| `10` | Best-effort conversion completed, with unsupported content preserved opaque |
| `20` | Input is unsupported or refused in strict mode; no usable conversion was produced |

Other operational failures have distinct codes; consult the converter report
and `nirs4all-tools --help` before retrying.

## Roll back from R2 to R1

There is no reverse converter. Reinstall the signed R1 artifact and reopen the
original, unchanged source workspace. Keep the new R2-native output separately;
do not replace the original with it. This is why conversion requires a distinct
output path and preserves the source exactly.

Web application maintainers should also read
[Storage Migration Guide -- Webapp Developers](storage_migration_webapp.md).
