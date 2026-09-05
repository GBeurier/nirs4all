# Native V1 migration and compatibility guide

:::{warning}
Version `1.0.0` is published and is the compatibility baseline. Corrective
V1 qualification is in progress; this page distinguishes the general Python
API from the explicitly selected portable profile. A supported portable
subset alone does not establish general Python or Studio feature parity.
:::

## Runtime boundary

The R3/R4 product architecture is unchanged:

- the general Python `run()` API chooses a DAG-ML execution profile before
  execution: portable requests use `native`; other requests use `dag-ml`;
- an explicit `engine=` or `N4A_ENGINE` selector is respected unchanged;
- the explicit `engine="native"` portable profile fails closed; no execution
  failure is retried through another engine;
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

An ordinary general Python run does not need an engine selector. The actual
engine is available as `result.execution_engine` and in `result.per_dataset`.
For a strict portable request, select `engine="native"` explicitly. Its current
training contract requires a list pipeline, a splitter, a supported Methods
model and `dataset={"X": X, "y": y, "sample_ids": ids}`. Chart generation,
workspace/project/cache integration and broader host operators are separate
general-profile capabilities, not portable archive guarantees.

An omitted `save_charts` uses the selected profile's default (false for
portable, true for general DAG-ML/explicit legacy). Explicit options are never
silently discarded. Both profiles accept `verbose=0..3`.

The general profile restores concrete pipelines without a splitter through a
single native `REFIT` phase: all training rows are fitted once, and test rows
are never fitted. There is no cross-validation score (`cv_best_score` is NaN).
When a test partition exists, the historical validation alias points to this
test partition, with a warning and persisted `evaluation` metadata; it is not
an independent model-selection holdout. Generator/HPO variants of this
no-splitter profile remain under qualification.

Concrete pipeline chart commands are presentation-only. Spectra after a
transform are generated from the captured full-training refit transform,
without a new fit; they are explicitly not out-of-fold features. Saved charts
include an adjacent HTML description, exact numeric CSV inputs, and scored
fold membership JSON under `workspace/charts`. Visible-only charts print a
text alternative. Branch/source/augmentation stage snapshots not yet captured
are refused explicitly rather than rendered from a misleading raw substitute;
this remains an open general-profile parity gap.

Captured general-profile `.n4a` models replay through a native DAG `PREDICT`
phase with a Python scientific host. Public `predict()` and trained/loaded
`Session.predict()` do not train again; their metadata records the selected
engine, input cohort, source artifact fingerprint and `training_performed=False`.
Inference does not manufacture validation scores. An explicit `engine="native"`
still requires a portable Core archive and never silently loads a host model.
General archives contain trusted Python/joblib objects: load them only from a
trusted producer. Recorded SHA-256 digests detect corruption, not malicious
producers. Older DAG exports without a recorded digest retain a visible warning
and do not claim verified integrity. Loaded Sessions are bound to their source
archive fingerprint and refuse replacement after loading.

When Studio supplies `store_run_id`, the library attaches its pipeline results
to that existing running workspace run without completing or failing the parent
run; Studio retains that lifecycle. `should_stop` is cooperative cancellation:
checked before execution, between native scientific tasks, and before publishing
results. It does not interrupt an individual BLAS/estimator fit mid-call.

General model-local `finetune_params` with `approach="single"` can use the
host Optuna profile (`engine="optuna"` inside `finetune_params`, or inferred
from historical sampler/trial controls). Optuna proposes parameters; DAG owns
the bounded trial loop, native candidate scores and selection. Each outer
training scope gets its own training-only 80/20 tuning holdout, and the final
REFIT gets a separate search on the full training partition. Preprocessing and
target transforms are fitted inside each inner training split. This fixes the
earlier global-preprocessing/global-parameter leakage, so selected parameters
and CV values need not equal that historical implementation.

Trial counts, seeds, the installed optimizer's sampler grammar and verbosity
are honored. Its historical Grid-to-TPE conversion for noncategorical ranges
remains visible; provenance records the effective sampler. General exports
retain the searches and the selected fitted predictor, not a resumable Optuna
study or a Methods checkpoint. They remain trusted Python artifacts, not
portable Core packages. Adaptive grouped/individual search, progressive pruning,
parallel trials, persistent studies and multi-phase host searches still require
qualification; deterministic DAG parameter grids keep their separate contract.

Classification repetition voting preserves the native raw/CV selection scores.
Each fitted CV/REFIT task captures its real class probabilities, when the
classifier provides them, with explicit class and sample identities. The
library's existing ensemble and repetition aggregators produce the additional
train/test/aggregate presentation rows without fitting again. These rows carry
`classification_evidence` or `aggregate_evidence` metadata; their display scores
are not new native selection reports. A classifier without `predict_proba`
uses its real hard predictions, never fabricated one-hot probabilities.
In subprocess execution the same evidence travels in a host-only capture
sidecar, separate from the coordinator's native result protocol.

General `generate()` and its convenience methods select the installed
`nirs4all.python.synthesis.v1` library adapter when no engine/plugin selector
is supplied. This reuses the scientific synthesis builder, without a legacy
ML runner, HTTP backend or retry. `generate.preflight()` exposes that host
contract; explicit `engine="native"` remains strict and unsupported for
synthesis. An explicitly requested unknown plugin is not substituted.

General `explain()` similarly selects `nirs4all.python.shap.v1`. It accepts a
captured general `.n4a`, a trained DAG result, or a persisted `result.best`
selection and explains the full REFIT predictor without training it again.
Preprocessing and inverse target transforms remain inside that predictor.
`output_index` selects one target or class-probability column (default 0);
the returned two-dimensional SHAP values describe that output, not all classes.
Kernel SHAP operates on the complete input predictor and can be expensive on
wide spectra: `n_samples` limits background rows, while `sample_indices` chooses
the rows to explain. Recorded headers and relation aggregation levels are
preserved rather than labelling an aggregate as a raw observation.
CSV values, provenance and an HTML table are written before optional figures;
`visualizations=[]` disables figures. This does not claim portable Core SHAP
or saved individual CV-fold models. Explicit `engine="native"` remains strict.

For a qualified request:

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
