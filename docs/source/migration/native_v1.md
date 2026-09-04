# Native V1 migration and compatibility guide

:::{warning}
This page documents the R4/V1 contract being qualified. R1 `0.13.0` is
published; R2 `1.0.0rc1` and R3 `1.0.0rc2` are distinct release candidates,
and R4 `1.0.0` has not been created. This pre-receipt guide does not assert
registry publication for either candidate and is not a stable-release
announcement.
:::

## Runtime boundary

The R3/R4 product architecture is unchanged:

- the Python library selects the native engine by default and fails closed;
- `engine="legacy"` remains available only when a direct Python caller selects
  the rollback profile explicitly, through the end of R4;
- Studio owns HTTP, WebSocket, store, jobs and scheduling in Rust;
- Studio may package CPython only as a bounded JSON-stdio host for an explicit
  library or plugin after capability preflight. CPython owns no listener,
  scheduler, store, job lifecycle or fallback;
- Studio, Web and strict Python product paths cannot select the legacy engine.

Installing Python is therefore the normal way to use the Python library and
its optional ML integrations. It is not an instruction to install a Python
backend for Studio.

## Migrate a workspace

Migration is an explicit, one-way operation in the separately published
`nirs4all-tools` `0.0.7` package. The runtime never starts it automatically and
never writes a legacy source.

```bash
python -m venv n4a-migration
n4a-migration/bin/python -m pip install \
  "nirs4all-tools[duckdb,parquet]==0.0.7"

n4a-migration/bin/nirs4all-tools workspace inspect /data/workspace
n4a-migration/bin/nirs4all-tools workspace convert /data/workspace \
  --output /data/workspace-v2 --verify
```

The source path, inode and bytes remain unchanged. Keep the source and output
at different paths. Exit code `0` means a clean verified conversion; `10`
means that unsupported content was preserved opaque and needs review; `20`
means unsupported input or strict refusal. There is no native-to-legacy reverse
conversion.

The authoritative read/write/migrate dispositions and retention window are in
the Tools [support matrix](https://github.com/GBeurier/nirs4all-tools/blob/codex/r4-sup002-tools/docs/contracts/legacy-support-matrix.v1.json)
and [support SLA](https://github.com/GBeurier/nirs4all-tools/blob/codex/r4-sup002-tools/docs/legacy-support-sla.md).

## Public API examples

The candidate default is native, so an ordinary run does not need an engine
selector:

```python
import nirs4all

result = nirs4all.run(pipeline=pipeline, dataset=dataset)
result.export("model.n4a")

prediction = nirs4all.predict("model.n4a", new_data)
with nirs4all.session() as session:
    second = nirs4all.run(pipeline=other_pipeline, dataset=dataset, session=session)
```

Unsupported native shapes are rejected during preflight; they are not retried
through legacy. The rollback path is deliberately different and visible:

```python
rollback_result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    engine="legacy",
)
```

Do not put that selector in Studio or Web configuration. See the
{doc}`native capability preflight </reference/native_capability_preflight>` and
{doc}`public interfaces </reference/public_interfaces>` references for the
operation-by-operation capability contract.

## ABI and schema identities

| Surface | V1 candidate identity | Compatibility rule |
|---|---|---|
| Methods C ABI | `libn4m` ABI `2.5.0`, SONAME major `2` | Call `n4m_check_abi_compatibility` before other ABI calls; bindings remain thin. |
| Methods model payload | N4MM format `1` and `2` | A fitted-model payload, not a complete `.n4a` pipeline archive. |
| Portable model/run archive | Core Archive V2 | Validate format, bounds, Methods identity and fingerprint before replay; no refit or legacy fallback on import. |
| Workspace | `nirs4all-workspace-v2` | SQLite metadata plus Parquet array sidecars; legacy stores require explicit Tools conversion. |
| Runtime envelopes | `rt_run_request.v1`, `rt_result.v1`, `rt_error.v1` | Producers and consumers use the frozen JSON schemas; no product-specific reinterpretation. |
| Studio product contract | Studio V1 HTTP/OpenAPI/WS snapshots | Rust is the sole control-plane owner; plugin-host responses cross bounded JSON stdio only. |

The Methods [ABI reference](https://github.com/GBeurier/nirs4all-methods/blob/codex/r4-doc002-methods/docs/abi/reference.md)
and the ecosystem [runtime schemas](https://github.com/GBeurier/nirs4all-ecosystem/tree/release/native-v1-candidate/docs/contracts/runtime)
are the lower-level authorities. This guide does not redefine them.

## Capability summary

| Request | Strict product behavior |
|---|---|
| `run`, `predict`, `session`, save/load/export | Native for the qualified V1 matrix; unsupported shapes refuse before significant work. |
| Full retrain | Native for the qualified archive path. |
| Transfer | Explicit Python-library plugin only when its preflight succeeds. |
| Finetune, unavailable explain/generate shapes | Explicit refusal in the strict profile; no implicit legacy execution. |
| Existing Python workflow during rollback | Direct Python call with explicit `engine="legacy"`; never Studio/Web. |

## FAQ

### Does Studio need Python?

Not as a backend. Studio's HTTP, WebSocket, store, jobs and scheduler are Rust.
A packaged CPython may exist only for an explicitly selected, preflighted
library/plugin call over bounded stdio.

### Can Studio fall back to the old backend?

No. An unavailable native or plugin capability returns a typed refusal. The
explicit legacy rollback profile is a direct Python-library feature only.

### Can I open an old workspace in place?

No. Inspect it with Tools, convert to a new disjoint directory, verify the
output, and retain the original for rollback.

### Is exit code 10 a clean conversion?

No. It means the command completed but preserved unsupported content opaque.
Read the migration and unsupported-content reports before using the output.

### How long are legacy readers supported?

The Tools `0.x` reader line is guaranteed through both complete releases after
the R2 flip: R3 and R4. Retirement is post-V1 only, after an announced
governance decision and migration notice.

### Does this page mean V1 is available?

No. R4 remains held until the product lock and remaining release gates are
green. Published component packages and passing candidate tests do not amount
to a stable V1 release.
