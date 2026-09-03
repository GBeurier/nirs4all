# Native Capability Preflight

This page is the public capability matrix for the locally qualified R2/R4
candidate. It does not announce a published package or installer. Release
availability and exact artifact digests remain controlled by the signed release
manifest.

The default Python engine is `native`. Unsupported work fails closed; no native
failure is redirected to the historical Python orchestrator. The only rollback
route is an explicit direct-Python call with `engine="legacy"`. That route is not
reachable from Studio's strict product boundary.

## Check capabilities before execution

The following checks do not train a model or run a scientific operation:

```python
from nirs4all.api.explain import explain_preflight
from nirs4all.api.generate import generate_preflight
from nirs4all.api.retrain import retrain_preflight

decisions = [
    retrain_preflight(mode="full", engine="native"),
    explain_preflight(engine="native"),
    generate_preflight(engine="native"),
]

for decision in decisions:
    print(decision.to_dict())
    # Call decision.require() only when refusal should raise.
```

Static pipeline consumers can inspect syntax and effect support without loading
a dataset:

```console
nirs4all keyword-registry --schema
nirs4all keyword-registry
```

The registry preflights declared syntax and effects. Dynamic constraints are
still checked at the `run()` boundary before the affected scientific operation.

## Limitations and their preflight

| Public limitation | Preflight | Refusal/remediation |
| --- | --- | --- |
| An omitted `engine` selects `native`. Ambient `N4A_ENGINE=legacy` or `dual` is forbidden. | Resolve the requested profile with `nirs4all.pipeline.engine.resolve_engine(...)`. | `ExecutionProfileError` identifies the forbidden profile. Pass `engine="legacy"` explicitly only for direct-Python rollback. |
| `allow_fallback=True` is not an execution route. | Call `resolve_engine(engine, allow_fallback=True)` before preparing data. | Native selectors refuse it. Select `engine="legacy"` explicitly when rollback is intended. |
| Archive V2 native training accepts raw PLS or exact row-wise SNV (`ddof=0`) -> Savitzky-Golay (`mode="interp"`, `deriv=0`, `delta=1`) -> PLS, with KFold and explicit target names for multi-target data. | Use the keyword registry for static syntax, then call `run(..., engine="native")`; its compiler validates the complete shape before fitting. | `NativeArchiveTrainingError` describes the unsupported shape. Simplify the pipeline, select another native engine deliberately, or use explicit legacy rollback. |
| `dag-ml` covers more of the pipeline language but not every operator shape. | Inspect the keyword registry, then use `run(..., engine="dag-ml")`; runtime validation occurs before the unsupported effect executes. | A structured `RtError` reports the missing capability and mitigation. There is no fallback. |
| A legacy Archive V1 loaded into a default-native `Session` is not silently replayed. | Inspect/convert it with `nirs4all-tools` before prediction, or deliberately test the explicit legacy lane. | Native prediction reports `legacy_archive_conversion_required`. Convert to the portable contract or pass `engine="legacy"` explicitly outside Studio. |
| Retrain support depends on mode, engine, plugin selector, session state, and bundle contents. | `retrain_preflight(...).to_dict()` is side-effect-free; native full retrain then validates exactly one bounded `train_pipeline.json` member. | `decision.require()` or `retrain()` returns the stable capability and mitigation. A plugin name does not imply that an adapter is installed. |
| Explain and synthetic generation remain Python-host features during this window. | `explain_preflight(...)` and `generate_preflight(...)`. | Native selection is refused; direct Python callers may select `engine="legacy"` explicitly. Studio cannot. |
| Studio accepts only its bounded JSON scientific-job contract. It does not expose arbitrary Python objects, paths, workspaces, schedulers, HTTP, or persistence to CPython. | Validate through `studio_scientific_job_v1`; it checks the strict profile and complete request before constructing a runner. | `StudioScientificJobError` returns a stable code to the Rust control plane. No Python HTTP service or product fallback is started. |
| Legacy/dual usage telemetry is local and opt-in. | Set `N4A_LEGACY_USAGE_COUNTER=1`, then read `get_legacy_engine_usage_counts()`. | The snapshot is process-only and data-free; it is not release telemetry and performs no network or persistent write. |

Conversion tooling has a separate deterministic result contract: exit `0` means
success, `10` means unsupported input, and `20` means corrupt or structurally
invalid input. Conversion never mutates its source. See the
{doc}`/migration/python-to-native` guide before moving a rollback archive.

## Related contracts

- {doc}`public_interfaces` — complete Python and Studio boundary surfaces.
- {doc}`native_conformal_finetuning_release_audit` — bounded native tuning and
  conformal capability audit.
- {doc}`/migration/python-to-native` — explicit inspection and conversion flow.
