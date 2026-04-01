# Prediction to Pipeline

**Date**: 2026-04-01  
**Scope**: Review of current "prediction -> pipeline / predict / retrain" capabilities and a corrected design direction

## Review summary

The previous draft identified real gaps, but it understated what the library already does and it put too much emphasis on reconstructing everything from the chain alone.

The codebase already supports:

- prediction from a prediction entry
- prediction from a prediction ID
- prediction from a `chain_id`
- retrain from a prediction entry
- extraction of a pipeline-like step list from a prediction source
- export of the canonical expanded pipeline config from `pipeline_id`

The main missing piece is not "can we do anything from a prediction?" but rather:

- there is no single public API that returns a pipeline definition from a prediction or chain
- the current semantics are not explicit about `full pipeline variant` vs `selected execution path`
- some metadata needed for exact round-trips is not persisted or not re-hydrated by `Predictions`

The strongest design is to treat `pipelines.expanded_config` as the canonical topology source, and to treat `chains` and traces as execution-path and artifact-selection metadata.

## Current capabilities of the lib regarding this feature

### 1. Prediction already works from a prediction entry or prediction ID

This is already implemented and tested.

- `PipelineRunner.predict(prediction_obj=..., dataset=...)` accepts:
  - a prediction dict
  - a prediction ID string
  - a bundle path
- `nirs4all.predict(model=..., data=...)` exposes the same model-based path at module level
- `PredictionResolver.resolve()` explicitly supports `SourceType.PREDICTION` and `SourceType.PREDICTION_ID`

Relevant code:

- `nirs4all/pipeline/runner.py`
- `nirs4all/pipeline/predictor.py`
- `nirs4all/pipeline/resolver.py`
- `tests/integration/pipeline/test_prediction_reuse_integration.py`

### 2. Prediction already works directly from `chain_id`

This also already exists.

- `WorkspaceStore.replay_chain(chain_id, X, wavelengths=None)` replays a stored chain directly
- `nirs4all.predict(chain_id=..., data=...)` is the public wrapper for this fast store-based path

Important limitation:

- the module-level `chain_id` path currently accepts only in-memory data forms (`numpy`, tuple, dict with `X`, `SpectroDataset`)
- it does not normalize dataset folders / CSV-style inputs the way the model-based path does

Relevant code:

- `nirs4all/pipeline/storage/workspace_store.py`
- `nirs4all/api/predict.py`

### 3. Retrain already works from a prediction source

This is more mature than the previous draft suggested.

- `PipelineRunner.retrain(source=..., dataset=..., mode=...)` already accepts prediction dicts, prediction IDs, bundles, folders, and other resolver-backed sources
- `nirs4all.retrain(source=..., data=...)` wraps it at module level
- `Retrainer` already supports:
  - `full`
  - `transfer`
  - `finetune`

Relevant code:

- `nirs4all/pipeline/retrainer.py`
- `nirs4all/api/retrain.py`
- `tests/integration/pipeline/test_retrain_integration.py`

### 4. Extraction of a pipeline-like step list already exists

There is already an extraction API:

- `PipelineRunner.extract(source)`
- `Retrainer.extract(source)`

This returns an `ExtractedPipeline` with editable `steps`.

This is the closest existing capability to "give me the pipeline corresponding to a prediction."

Important caveat:

- this is not yet positioned as a stable "pipeline definition export" API
- it returns an editable step list, not JSON/YAML export
- its exact semantics depend on how the resolver reconstructed the pipeline for that source

Relevant code:

- `nirs4all/pipeline/retrainer.py`
- `tests/integration/pipeline/test_retrain_integration.py`

### 5. Canonical pipeline topology is already stored

The store already has the best source of truth for full topology:

- `pipelines.expanded_config`
- `pipelines.generator_choices`

This matters because `expanded_config` preserves:

- branches
- merges
- stacking / meta-model structure
- wrappers like `y_processing`
- serialized splitter definitions when they were part of the expanded config

There is also already an export helper:

- `WorkspaceStore.export_pipeline_config(pipeline_id, output_path)`

Relevant code:

- `nirs4all/pipeline/storage/store_schema.py`
- `nirs4all/pipeline/storage/workspace_store.py`

### 6. Refit internals already reconstruct full pipeline variants from store metadata

The refit path already uses `pipeline_id -> expanded_config` as the reconstruction source.

This is important because it proves the codebase already prefers pipeline metadata over chain-only reconstruction when fidelity matters.

Relevant code:

- `nirs4all/pipeline/execution/refit/config_extractor.py`

### 7. Resolver store mode already combines pipeline topology with chain selection

`PredictionResolver._resolve_from_store()` already does something very close to the desired feature:

- uses the prediction to select the matching chain
- loads artifacts from that chain
- uses `pipeline.expanded_config` to rebuild executable steps

This is the right architectural direction.

Important caveat:

- the current reconstruction is `expanded_config[:model_step_idx]`
- that is good for many cases, but it is not always the exact selected execution path for a branch-local model

Relevant code:

- `nirs4all/pipeline/resolver.py`

### 8. Export exists in two different fidelity levels

There are two export paths today:

1. Resolver-based export via `runner.export(source, ...)`
   - writes `pipeline.json`
   - may write `trace.json`
   - is suitable as a richer reconstruction source

2. Store chain export via `store.export_chain(chain_id, ...)`
   - writes `chain.json`
   - does not write `pipeline.json`
   - is primarily a deployment / replay artifact, not a canonical topology export

This distinction is important. A bundle produced from `store.export_chain()` is prediction-oriented. A resolver-based bundle is much closer to a reconstruction bundle.

Relevant code:

- `nirs4all/pipeline/bundle/generator.py`
- `nirs4all/pipeline/bundle/loader.py`
- `nirs4all/pipeline/storage/workspace_store.py`

## What's missing

### 1. No public "pipeline definition from source" API

There is no single public API like:

- `nirs4all.pipeline_from(source, ...)`
- `runner.to_pipeline(source, ...)`
- `predictions.to_pipeline(...)`

The capability is scattered across `extract()`, the resolver, store exports, and bundle exports.

### 2. No direct `chain_id -> retrain` API

Prediction from `chain_id` exists, but retrain from `chain_id` does not.

This asymmetry is user-visible and confusing.

### 3. No explicit distinction between `variant pipeline` and `selected path`

For a prediction source there are two different things a user may mean:

- the full expanded pipeline variant that produced the prediction
- the exact execution path that led to the selected model

Those are not the same in branching / stacking scenarios.

Current code mixes these semantics:

- chain replay is path-like
- refit/config extraction is variant-like
- resolver store reconstruction is a prefix of `expanded_config`, which is often useful but not always semantically exact

### 4. Chain alone is not a reliable canonical topology source

The previous draft overestimated how much should be reconstructed from chain data alone.

What the chain does well:

- artifact mapping
- model-step targeting
- branch selection metadata
- deployable linear replay

What the chain does not preserve well enough for full topology reconstruction:

- branch/merge structure as an explicit nested config
- wrapper semantics such as `y_processing`
- fully faithful raw operator definitions in all cases
- exact source-level intent of the original expanded config

### 5. Some prediction metadata is lost across persistence boundaries

This is one of the most important practical gaps.

`Predictions._populate_buffer_from_store()` currently rehydrates only part of the stored prediction metadata. In particular:

- `chain_id` is not re-injected into the in-memory prediction rows
- `trace_id` is not persisted in the SQL predictions table
- `branch_path` is not persisted in the SQL predictions table
- `target_processing` is tracked in `Predictions.add_prediction()` but not persisted in SQL

This means:

- a re-opened `Predictions` object is weaker than the original in-memory prediction buffer
- exact source-to-chain disambiguation becomes harder than it needs to be

### 6. Chain step params are not reconstruction-grade in all cases

Two issues matter here:

1. `operator_class` in chains is a short name in practice, not a full import path
2. `operator_config` / `params` can be lossy for raw live objects

This is especially relevant for:

- splitters provided as live sklearn instances
- arbitrary class instances
- any step where the executor stored only sanitized config

So the chain is not a good sole input for generic object re-instantiation.

### 7. `y_processing` is still weakly represented in chain-only exports

Within the execution system, `operator_type` can distinguish `y_processing`, and bundle export with `pipeline.json` can recover it.

But in chain-only storage/export:

- the chain step itself stores only operator class + params
- the `y_processing` wrapper is not preserved as such

So a pure chain-to-config reconstruction still risks turning target transforms into ordinary preprocessing unless it also consults the pipeline config or trace metadata.

### 8. `chain_id` prediction API does not yet cover folder/CSV-style inputs

This is not a conceptual blocker, but it is a usability gap relative to the stated feature request.

Today:

- `predict(chain_id=..., data=X)` works
- `predict(chain_id=..., data="path/to/dataset_or_csv")` does not use the normal dataset normalization path

## Design proposition(s) with rationale

### Proposition A: Add a first-class pipeline definition resolver

**Recommendation**: add a public API centered on source resolution, not on chain reconstruction alone.

Suggested shape:

```python
resolve_pipeline_definition(
    source,
    *,
    view="variant",   # "variant" | "path"
    format="python",  # "python" | "json" | "yaml"
    workspace_path=None,
)
```

Suggested outputs:

- `view="variant"`:
  - return the canonical expanded pipeline variant from `pipelines.expanded_config`
  - preserve branches, merges, stacking, wrappers, and generator-expanded structure

- `view="path"`:
  - return the selected execution path to the target model
  - prefer trace-based branch extraction when a trace is available
  - fallback to chain-derived linear path when only chain data exists

Rationale:

- this builds on `PredictionResolver`, which already solves source detection and chain selection
- it avoids duplicating source handling logic across prediction, retrain, export, and docs
- it separates two valid but different user intents

### Why this is better than a chain-only `chain_to_pipeline()`

Because the full pipeline topology already exists in a better place:

- `pipelines.expanded_config`

The chain should be used for:

- artifact reuse
- target-model selection
- exact execution-path metadata

The chain should not be made the canonical source of topology unless there is no better source available.

### Proposition B: Make `chain_id` a first-class source for retrain and extraction

Suggested additions:

```python
nirs4all.retrain(chain_id="...", data=..., scope="variant")
runner.extract(chain_id="...", view="variant")
runner.extract_definition(chain_id="...", view="variant", format="yaml")
```

Design rule:

- `chain_id` should resolve to its parent `pipeline_id`
- default retrain behavior should use the canonical variant pipeline from `expanded_config`

Optional explicit scope:

- `scope="variant"`: retrain the full expanded pipeline variant that produced the chain
- `scope="path"`: retrain only the selected execution path

Rationale:

- `chain_id` prediction already exists, so retrain should mirror it
- `variant` is the safer default for complex topologies
- `path` is still useful for deployment-oriented or branch-local reuse

### Proposition C: Persist and rehydrate the metadata needed for exact round-trips

This is the highest-value structural improvement after the public API.

Recommended changes:

1. Extend persisted prediction metadata with:
   - `trace_id`
   - `branch_path`
   - `target_processing`

2. Ensure rehydration into `Predictions` preserves:
   - `chain_id`
   - `trace_id`
   - `branch_path`
   - `target_processing`

3. Keep `chain_id` authoritative when present

Rationale:

- this reduces ambiguity without inventing new reconstruction logic
- it makes stored predictions much more useful for exact replay, export, and pipeline lookup
- it improves determinism for branch-local models

### Proposition D: Enrich chain-based bundle export with canonical pipeline metadata

Recommended improvement:

- when exporting from `chain_id`, also include `pipeline.json` when the parent pipeline record is available
- include `pipeline_id`, `chain_id`, and selection metadata in the manifest

Suggested extra manifest fields:

- `pipeline_id`
- `chain_id`
- `branch_path`
- `trace_id` if known
- `view_hint`: `"path"` or `"variant"`

Rationale:

- current chain export is excellent for prediction replay
- it is weaker than resolver-based export for retrain / reconstruction
- adding `pipeline.json` removes an unnecessary fidelity gap between export paths

### Proposition E: Add a low-level chain-to-path helper, but keep it secondary

A low-level helper is still useful, but it should be explicitly path-oriented:

```python
chain_to_path_steps(chain, *, pipeline=None, trace=None) -> list[Any]
```

Use cases:

- diagnostics
- deployment introspection
- fallback when `expanded_config` is not available

It should not be the main recommended API for "give me the pipeline", because that phrase usually means topology-preserving config, not just replay steps.

## Suggested implementation order

### Phase 1: clarify and expose what already exists

1. Add `resolve_pipeline_definition(source, view=..., format=...)`
2. Add `chain_id` support to retrain/extract public APIs
3. Fix `api.predict(chain_id=...)` to use normal dataset normalization

### Phase 2: remove ambiguity and metadata loss

4. Persist `trace_id`, `branch_path`, `target_processing`
5. Rehydrate `chain_id` and related metadata in `Predictions`
6. Make `Predictions.to_pipeline()` and `Predictions.predict()/retrain()` thin convenience wrappers

### Phase 3: improve round-trip fidelity

7. Include `pipeline.json` in chain-based bundle export
8. Optionally add path-view helpers based on trace/chain
9. Optionally store richer chain step metadata (`operator_type`, full class path) for better fallback behavior

## Global insight on the feature and the related code

### 1. The codebase is closer to the target than the previous draft implied

The core ingredients are already present:

- prediction source resolution
- direct chain replay
- retrain infrastructure
- extraction API
- canonical expanded pipeline storage
- refit-time reconstruction of winning variants

This is not a greenfield feature. It is mostly a productization and API-clarity problem.

### 2. The canonical source of topology is already the pipeline record, not the chain

This is the key design insight.

If the goal is:

- exact branch/merge/stacking structure
- JSON/YAML export
- faithful retrain of the original variant

then `pipelines.expanded_config` is the right source.

If the goal is:

- fast prediction replay
- artifact reuse
- selected-path targeting

then the chain is the right source.

Trying to force the chain to serve both roles leads to avoidable complexity and weaker fidelity.

### 3. The biggest practical issue is metadata loss, not lack of storage tables

The store already contains enough information to do most of what is needed.

The more immediate weakness is that some of that information does not survive the trip back through `Predictions`, especially for persisted/re-opened results.

That is likely to hurt users more than the absence of a new reconstruction module.

### 4. Current extraction/retrain semantics need to be made explicit

`extract()` and retrain already exist, but for complex pipelines the meaning should be documented and formalized:

- are we extracting the full pipeline variant?
- or the selected model path?

Without that distinction, users can get surprising behavior around branching and stacking.

### 5. A few obvious improvements should be done even if the larger feature is postponed

These are small, high-value enhancements:

1. Preserve `chain_id` when loading predictions from store
2. Add `trace_id` / `branch_path` persistence
3. Make `predict(chain_id=...)` accept normalized dataset inputs
4. Include `pipeline.json` in chain-based bundle export
5. Return non-empty preprocessing metadata for chain-based `PredictResult`

### 6. One code-quality inconsistency is worth noting

Some comments/docstrings imply chains store fully qualified class names, but the execution system generally records short class names in practice.

That is acceptable for display and replay, but it is not good enough for a generic "recreate this operator from chain only" story.

If path-only reconstruction from chains becomes important, storing richer operator metadata would help. But it is not the first problem to solve.
