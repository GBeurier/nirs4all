# Pipeline Module Technical Debt Review

Date: 2026-02-07  
Scope: `nirs4all/pipeline/**`  
Focus: technical debt, redundancy, dead code, performance, reproducibility

## Executive Summary

The pipeline module is functionally rich but structurally overloaded. The main risks are:

1. Runtime/performance debt in execution hot paths (`executor`, `orchestrator`, `artifact_loader`).
2. Reproducibility debt (non-deterministic resolution and incomplete seed propagation).
3. Dead/dormant code and partially integrated features (notably step cache and legacy run entity).
4. High maintenance burden from parallel implementations in prediction/artifact-provider paths.
5. Low test coverage in critical modules where behavior is most complex.

The highest-leverage short-term actions are:

1. Decide and execute `StepCache` strategy: integrate end-to-end or remove.
2. Fix pipeline metadata semantics (`dataset_hash` mismatch).
3. Remove confirmed dead code and tighten static gates.
4. Reduce copy/materialization overhead in branch execution and shape tracing.
5. Make resolver and generator paths deterministic by default.

---

## Methodology

Audit combined static analysis, coverage snapshot review, and targeted code inspection.

### Static scans

- `ruff check nirs4all/pipeline --select C901`
- `ruff check nirs4all/pipeline --select F401,F841`

Results:

- Complexity findings: **63**
- Unused symbol findings: **77** (mostly F401, some F841)

### Size profile

- Files in `nirs4all/pipeline`: **76**
- LOC in `nirs4all/pipeline`: **34,257**
- Pipeline-focused tests (`tests/(unit|integration)/pipeline`): **88 files**, **36,705 LOC**

### Coverage snapshot

Source: existing `.coverage` in workspace (`.venv/bin/coverage report -m`).  
Note: this is a snapshot, not a fresh full-suite execution in this audit.

Lowest-covered pipeline modules (selected):

- `nirs4all/pipeline/run.py`: 0%
- `nirs4all/pipeline/minimal_predictor.py`: 8%
- `nirs4all/pipeline/storage/library.py`: 9%
- `nirs4all/pipeline/trace/extractor.py`: 12%
- `nirs4all/pipeline/storage/workspace_store.py`: 20%
- `nirs4all/pipeline/bundle/loader.py`: 24%
- `nirs4all/pipeline/config/context.py`: 27%
- `nirs4all/pipeline/execution/orchestrator.py`: 41%
- `nirs4all/pipeline/execution/executor.py`: 44%

---

## Debt Register (Prioritized)

### [P0] Step cache is implemented but not integrated into execution

Evidence:

- Cache is created and stats logged only in orchestrator:
  - `nirs4all/pipeline/execution/orchestrator.py:216`
  - `nirs4all/pipeline/execution/orchestrator.py:334`
- Builder/executor have no step cache dependency:
  - `nirs4all/pipeline/execution/builder.py:221`
  - `nirs4all/pipeline/execution/executor.py:35`
- Search shows no executor usage:
  - `rg "step_cache|StepCache"` => only orchestrator + tests.

Impact:

- Feature appears active but has no runtime effect.
- Maintenance and test cost with no production benefit.

Recommendation:

1. Either wire cache into `PipelineExecutor` hot path (`get/put` around deterministic preprocessing steps), or remove `StepCache` and related tests/docs.
2. Do not keep partially integrated performance features.

---

### [P0] Incorrect pipeline metadata: pipeline hash stored as `dataset_hash`

Evidence:

- Pipeline hash is computed from steps:
  - `nirs4all/pipeline/execution/executor.py:140`
- Passed to `begin_pipeline(... dataset_hash=...)`:
  - `nirs4all/pipeline/execution/executor.py:157`
- Store contract expects dataset content hash:
  - `nirs4all/pipeline/storage/workspace_store.py:357`

Impact:

- Incorrect provenance and compatibility checks.
- Cache invalidation keyed by dataset can misbehave or become misleading.

Recommendation:

1. Add explicit `pipeline_hash` column/field where needed.
2. Pass actual dataset content hash into `dataset_hash`.
3. Backfill/migrate existing rows with clear compatibility rules.

---

### [P0] Branch execution copies full feature tensors repeatedly

Evidence:

- Deep-copy restore per branch:
  - `nirs4all/pipeline/execution/executor.py:509`
- Context copy deep-copies full `custom` dict:
  - `nirs4all/pipeline/config/context.py:1024`
- Per-step immutable update pattern calls `with_step_number` every step:
  - `nirs4all/pipeline/execution/executor.py:421`
  - `nirs4all/pipeline/config/context.py:1080`

Impact:

- High memory pressure and CPU overhead in branch-heavy pipelines.
- Amplifies with large arrays and nested branch metadata in `custom`.

Recommendation:

1. Stop deep-copying array payloads in context custom state.
2. Store branch snapshots by lightweight handles/IDs instead of full arrays.
3. Apply copy-on-write for branch snapshots.

---

### [P1] Per-step shape tracing materializes data twice (2D + 3D), before and after each step

Evidence:

- Shape tracing calls:
  - `dataset.x(... layout="2d")` at `nirs4all/pipeline/execution/executor.py:1090`
  - `dataset.x(... layout="3d")` at `nirs4all/pipeline/execution/executor.py:1098`
- Called both pre/post step:
  - `nirs4all/pipeline/execution/executor.py:691`
  - `nirs4all/pipeline/execution/executor.py:710`
  - Branch path also does both:
    - `nirs4all/pipeline/execution/executor.py:575`
    - `nirs4all/pipeline/execution/executor.py:595`

Impact:

- Extra compute and allocations on every step, including branch multipliers.

Recommendation:

1. Gate shape tracing behind explicit debug flag/verbosity threshold.
2. Cache shape metadata when selector state is unchanged.
3. Record minimal metadata (already-known dimensions) when possible.

---

### [P1] Prediction flush path has quadratic matching behavior and eager array loading

Evidence:

- Candidate selection loops through `chain_rows` repeatedly per prediction:
  - `nirs4all/pipeline/execution/executor.py:307`
  - `nirs4all/pipeline/execution/executor.py:323`
  - `nirs4all/pipeline/execution/executor.py:332`
- Iterates predictions with `load_arrays=True`:
  - `nirs4all/pipeline/execution/executor.py:341`
- Unused import in hot function:
  - `nirs4all/pipeline/execution/executor.py:283` (`polars`).

Impact:

- O(predictions x chains) matching cost.
- Potential memory spike due eager arrays.

Recommendation:

1. Pre-index chains by `(model_step_idx, branch_key, model_class, preprocessings)`.
2. Stream prediction arrays lazily/chunked.
3. Remove unused `polars` import.

---

### [P1] Artifact loader step lookups are linear scans over all artifacts

Evidence:

- `load_for_step()` iterates `self._artifacts.items()` each call:
  - `nirs4all/pipeline/storage/artifacts/artifact_loader.py:364`

Impact:

- Prediction and explain workflows do repeated O(N) scans per step.

Recommendation:

1. Build indexes at loader init (`by_step`, `by_step_branch`, `by_step_branch_source`).
2. Keep linear scan only for fallback/debug paths.

---

### [P1] Resolver selects first filesystem candidate from unsorted iteration

Evidence:

- Iteration and candidate order depend on filesystem traversal:
  - `nirs4all/pipeline/resolver.py:525`
  - `nirs4all/pipeline/resolver.py:536`
- Returns `candidates[0]` without deterministic sort:
  - `nirs4all/pipeline/resolver.py:546`
  - `nirs4all/pipeline/resolver.py:559`
- Partial match (`pipeline_uid in subdir.name`) broadens ambiguity:
  - `nirs4all/pipeline/resolver.py:537`

Impact:

- Non-deterministic replay source selection across environments/runs.

Recommendation:

1. Require exact match by canonical IDs first.
2. Sort candidates deterministically (run timestamp, run id).
3. Treat partial-match fallback as explicit warning/error unless unique.

---

### [P1] Reproducibility controls are incomplete and mixed

Evidence:

- Global seed helper:
  - `nirs4all/pipeline/runner.py:42`
- Sets `PYTHONHASHSEED` at runtime:
  - `nirs4all/pipeline/runner.py:57`
- Generator expansion call without explicit seed:
  - `nirs4all/pipeline/config/pipeline_config.py:78`
- Generator API supports seed:
  - `nirs4all/pipeline/config/_generator/core.py:74`
- Reservoir sampling uses unseeded RNG when no seed:
  - `nirs4all/pipeline/config/_generator/iterator.py:284`

Impact:

- Results can vary unless users manually control all paths.
- Reproducibility behavior is implicit and fragmented.

Recommendation:

1. Add a single run-level seed policy and propagate seed everywhere.
2. Use deterministic default in generator module for bounded random ops.
3. Document non-deterministic operators explicitly.

---

### [P1] Import-time root logger mutation in config module

Evidence:

- Root logger modified at import:
  - `nirs4all/pipeline/config/pipeline_config.py:26`
  - `nirs4all/pipeline/config/pipeline_config.py:28`
  - `nirs4all/pipeline/config/pipeline_config.py:29`

Impact:

- Hidden global side effects, duplicate handlers, surprising logs in host apps/tests.

Recommendation:

1. Remove root logger mutation from module import path.
2. Route logging config through central bootstrap only.

---

### [P1] Hash identity is brittle and truncated

Evidence:

- Executor hash uses `default=str` and truncates MD5 to 6 chars:
  - `nirs4all/pipeline/execution/executor.py:1129`
  - `nirs4all/pipeline/execution/executor.py:1130`
- Config naming truncates hash in generated names:
  - `nirs4all/pipeline/config/pipeline_config.py:91`
  - `nirs4all/pipeline/config/pipeline_config.py:303`

Impact:

- Collision and instability risk in metadata IDs.
- `default=str` can hide non-serializable config semantics.

Recommendation:

1. Use canonical serialization for step configs.
2. Increase hash length (at least 12 hex chars) or move to SHA-256 truncated.
3. Keep display-short hash separate from identity hash.

---

### [P1] Memory-heavy defaults and retained dataset snapshots

Evidence:

- `keep_datasets=True` default:
  - `nirs4all/pipeline/runner.py:130`
- Raw and postprocessed arrays retained:
  - `nirs4all/pipeline/execution/orchestrator.py:252`
  - `nirs4all/pipeline/execution/orchestrator.py:303`

Impact:

- Elevated baseline memory use for large datasets.

Recommendation:

1. Default `keep_datasets=False`.
2. Add bounded snapshot policy (max datasets / max MB).

---

### [P2] Confirmed dead/unreferenced code in pipeline module

Confirmed unreferenced symbols by repository search:

- `nirs4all/pipeline/execution/executor.py:835` (`_filter_binaries_for_branch`)
- `nirs4all/pipeline/storage/store_queries.py:471` (`build_filter_clause`)
- `nirs4all/pipeline/config/_generator/constraints.py:213` (`count_with_constraints`)
- `nirs4all/pipeline/config/_generator/strategies/registry.py:96` (`get_all_strategies`)
- `nirs4all/pipeline/config/_generator/strategies/registry.py:110` (`clear_registry`)
- `nirs4all/pipeline/config/_generator/strategies/registry.py:119` (`_get_registry_state`)
- `nirs4all/pipeline/run.py` has no imports from `nirs4all` or `tests` outside itself.

Impact:

- Noise for maintainers and readers.
- Dead interfaces create false extension points.

Recommendation:

1. Remove or deprecate with timeline.
2. Add dead-code CI gate.

---

### [P2] Runtime path has dormant cross-run cache hooks

Evidence:

- Methods exist:
  - `nirs4all/pipeline/storage/artifacts/artifact_registry.py:1225`
  - `nirs4all/pipeline/storage/artifacts/artifact_registry.py:1255`
- Used in tests, not in runtime orchestration/executor flow.

Impact:

- Potentially valuable feature is not delivering runtime value.

Recommendation:

1. Integrate into run lifecycle (persist after successful run; consult before compute), or remove if out of scope.

---

### [P2] `StepRunner.execute` contains dead/unreachable block and stale comments

Evidence:

- Early return path:
  - `nirs4all/pipeline/steps/step_runner.py:200`
- Unreachable commented/planning block continues:
  - `nirs4all/pipeline/steps/step_runner.py:206`
  - `nirs4all/pipeline/steps/step_runner.py:231`
- Later return depends on tuple-branch locals:
  - `nirs4all/pipeline/steps/step_runner.py:233`

Impact:

- High confusion and high-risk editing zone.

Recommendation:

1. Refactor `execute()` into strict branches with typed return.
2. Remove stale commentary and unreachable paths.

---

### [P2] Redundant artifact-provider and prediction replay implementations

Evidence:

- Artifact-provider variants with overlapping filtering/selection logic:
  - `nirs4all/pipeline/config/context.py:544` (`MapArtifactProvider`)
  - `nirs4all/pipeline/config/context.py:727` (`LoaderArtifactProvider`)
  - `nirs4all/pipeline/minimal_predictor.py:222` (`MinimalArtifactProvider.get_artifacts_for_step`)
  - `nirs4all/pipeline/storage/artifacts/artifact_loader.py:337` (`load_for_step`)
- Dual prediction execution paths with substantial setup duplication:
  - `nirs4all/pipeline/predictor.py:161` (`_predict_with_minimal_pipeline`)
  - `nirs4all/pipeline/predictor.py:295` (`_predict_full_pipeline`)

Impact:

- Behavioral drift risk.
- Bug fixes must be replicated in multiple locations.

Recommendation:

1. Introduce a single artifact query service API.
2. Share one prediction execution setup path with strategy hooks.

---

### [P2] Static cleanliness debt (unused imports/locals)

Evidence:

- `77` unused-symbol findings (`F401/F841`) in `nirs4all/pipeline`.

Impact:

- Signals incomplete refactors and low signal-to-noise.

Recommendation:

1. Apply safe auto-fixes for F401/F841 where possible.
2. Add CI check for zero unused symbols in pipeline package.

---

### [P2] Complexity concentration in core orchestrators/resolver

Evidence:

- `63` complexity findings (`C901`).
- Highest concentrations:
  - `nirs4all/pipeline/resolver.py` (7)
  - `nirs4all/pipeline/execution/executor.py` (7)
  - `nirs4all/pipeline/bundle/loader.py` (5)
  - `nirs4all/pipeline/trace/extractor.py` (4)
  - `nirs4all/pipeline/config/component_serialization.py` (4)

Impact:

- Hard-to-test behavior and high regression probability.

Recommendation:

1. Split high-branch functions into pure helpers.
2. Use explicit state objects to reduce branch fan-out.

---

## Performance Risk Map

### High risk

1. Branch deep copies in executor/context (`executor.py:509`, `context.py:1024`).
2. Per-step input/output shape materialization (`executor.py:1090`, `executor.py:1098`).
3. Prediction flush matching loops and eager arrays (`executor.py:307`, `executor.py:341`).
4. Artifact loader full-scan lookup (`artifact_loader.py:364`).

### Medium risk

1. Dataset snapshot retention by default (`runner.py:130`, `orchestrator.py:252`, `orchestrator.py:303`).
2. Duplicate prediction/artifact paths increase optimization surface area.

---

## Reproducibility Risk Map

### High risk

1. Non-deterministic pipeline directory resolution (`resolver.py:525`, `resolver.py:546`).
2. Incomplete seed propagation (`pipeline_config.py:78`, `_generator/core.py:74`).
3. Unseeded fallback RNG (`iterator.py:284`).

### Medium risk

1. `PYTHONHASHSEED` runtime assignment gives false confidence (`runner.py:57`).
2. Short/truncated hashes for identity (`executor.py:1130`, `pipeline_config.py:91`).

---

## Redundancy Inventory

Primary redundancy clusters:

1. Artifact selection/filtering logic duplicated in:
   - `nirs4all/pipeline/config/context.py:628`
   - `nirs4all/pipeline/config/context.py:784`
   - `nirs4all/pipeline/minimal_predictor.py:222`
   - `nirs4all/pipeline/storage/artifacts/artifact_loader.py:337`
2. Prediction execution setup duplicated in:
   - `nirs4all/pipeline/predictor.py:161`
   - `nirs4all/pipeline/predictor.py:295`
3. Runtime feature flags and tracing concerns spread across executor and orchestrator without a central policy object.

---

## Dead Code Inventory

### Confirmed dead (no references outside defining module)

1. `nirs4all/pipeline/run.py` (module-level entity set)
2. `nirs4all/pipeline/execution/executor.py:835` (`_filter_binaries_for_branch`)
3. `nirs4all/pipeline/storage/store_queries.py:471` (`build_filter_clause`)
4. `nirs4all/pipeline/config/_generator/constraints.py:213` (`count_with_constraints`)
5. `nirs4all/pipeline/config/_generator/strategies/registry.py:96`
6. `nirs4all/pipeline/config/_generator/strategies/registry.py:110`
7. `nirs4all/pipeline/config/_generator/strategies/registry.py:119`

### Dormant integrations

1. Step cache lifecycle without execution integration.
2. Cross-run cache store hooks not used by runtime flow.

---

## Testability and Coverage Debt

Coverage is weakest in modules with the highest complexity and side effects:

1. Runtime core: `executor`, `orchestrator`, `workspace_store`.
2. Replay/trace extraction: `minimal_predictor`, `trace/extractor`.
3. Bundle and storage layers: `bundle/loader`, `storage/library`.

Risk:

- Critical paths can regress silently under refactor pressure.

Recommended minimum targets for next cycle:

1. `execution/executor.py` >= 70%
2. `execution/orchestrator.py` >= 70%
3. `resolver.py` >= 75%
4. `storage/workspace_store.py` >= 60% initial, then ratchet upward
5. Add deterministic replay tests covering ambiguous resolver candidates and seeded/unseeded generator paths.

---

## Remediation Roadmap

## Phase 0 (1-2 weeks): correctness and cleanup

1. Fix `dataset_hash` semantics (P0).
2. Remove confirmed dead code and stale unreachable blocks.
3. Resolve `StepCache` strategy (integrate or remove).
4. Remove import-time root logger mutation.
5. Clear F401/F841 debt in pipeline package.

## Phase 1 (2-4 weeks): performance hardening

1. Optimize branch snapshot handling (copy-on-write handles).
2. Add shape-trace gating and cheap-path mode.
3. Rework prediction flush matching with pre-indexed chain map.
4. Add artifact loader indexes for step/branch/source queries.

## Phase 2 (4-8 weeks): architecture consolidation

1. Unify artifact provider/query abstraction.
2. Collapse minimal/full prediction setup duplication into one engine with strategy hooks.
3. Reduce C901 hotspots in `executor`, `resolver`, `bundle/loader`.

## Phase 3 (ongoing): reproducibility and quality gates

1. Explicit deterministic policy object (seed, resolver policy, hash policy).
2. CI gates:
   - `ruff C901` budget by file.
   - Zero F401/F841 in `nirs4all/pipeline`.
   - Coverage thresholds for critical modules.
3. Add non-regression tests for replay determinism and branch artifact selection.

---

## Suggested CI Guardrails

1. `ruff check nirs4all/pipeline --select F401,F841` must pass.
2. Complexity budget for touched files (`C901`) must not increase.
3. Determinism tests:
   - resolver candidate selection with ambiguous directories.
   - generator expansion stability with fixed seed.
4. Coverage ratchet per critical module (fail on downward drift).

---

## Appendix: Commands Used

Static checks:

- `ruff check nirs4all/pipeline --select C901 --output-format concise`
- `ruff check nirs4all/pipeline --select F401,F841 --output-format concise`

Size/coverage:

- `rg --files nirs4all/pipeline | wc -l`
- `rg --files nirs4all/pipeline | xargs wc -l | tail -n 1`
- `.venv/bin/coverage report -m | rg '^nirs4all/pipeline/'`

Reference discovery:

- `rg -n "<symbol>" nirs4all/pipeline tests`
- `nl -ba <file> | sed -n '<range>p'`
