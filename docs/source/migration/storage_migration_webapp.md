# Storage Migration Guide -- Webapp Developers

## Supported runtime boundary

Current nirs4all workspaces use `store.sqlite`. The runtime may identify a
historical `store.duckdb` or a SQLite store containing the legacy
`prediction_arrays` table, but that identification is read-only.

`WorkspaceStore` raises `ConversionRequired` before opening a recognized
legacy store. It does not fall back to DuckDB, invoke a converter, rename the
source, or create a `.bak` file. A web application should surface the exception
and its operator guidance; it must not retry the source through another backend.

The explicit `engine="legacy"` runtime remains supported for its agreed
workflows until after R4. This rollback lane does not open or convert a DuckDB
workspace implicitly.

## Operator action

Conversion belongs to the separately installed `nirs4all-tools` console
command and always writes a new, disjoint output:

```bash
nirs4all-tools workspace inspect /data/workspace
nirs4all-tools workspace convert /data/workspace \
  --output /data/workspace-r2 --verify
```

`nirs4all-tools` `0.0.7` is published. Operators should install that exact
version for this support contract; the application must not automatically
install Tools or treat its component publication as V1 product promotion.

The workspace commands use stable domain codes: `0` means a clean conversion,
`10` means best-effort completion with unsupported content preserved opaque,
and `20` means unsupported or strict refusal. The source path, inode, and bytes
remain intact for all three outcomes.

## Application behaviour

- Open `store.sqlite` only through the supported runtime store.
- Catch `ConversionRequired` and display the source path plus the two commands
  above; do not invoke them from a request handler.
- Open the converted output only after the command returns `0` or an operator
  explicitly accepts a documented `10` result.
- Keep the original source and converted output at separate paths.

## R2 to R1 rollback

Rollback has no reverse-conversion step. Reinstall the signed R1 artifact and
reopen the original, unchanged source. Retain the R2-native output separately.
This preserves both auditability and the ability to resume on R2 later.
