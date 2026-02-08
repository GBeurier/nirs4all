# Workspace/Predictions/Artifacts Technical Debt Review

Date: 2026-02-07  
Scope: `nirs4all/workspace/**`, `nirs4all/pipeline/storage/**`, `nirs4all/pipeline/resolver.py`, `nirs4all/data/predictions.py`, refit execution paths  
Focus: technical debt in workspace persistence, prediction lifecycle, and artifact replay/export reliability

## Executive Summary

The workspace/predictions/artifacts stack is feature-complete but has contract drift across layers.  
The core debt is not missing functionality; it is inconsistent semantics between trace, chain, store, resolver, and refit flows.

Highest-risk debt:

1. Refit prediction labels are applied after persistence in refit flows, which can write CV-like rows to store.
2. Fold key contracts are fragmented (`"0"`/`"final"` vs `"fold_0"`/`"fold_final"`), forcing normalization shims across modules.
3. Chain resolution in branched pipelines uses weak heuristics and string-matching on serialized JSON branch paths.

Highest-leverage paydown:

1. Normalize fold key contract once and enforce it at boundaries.
2. Make resolver strictly deterministic (parsed branch path matching, no first-row fallback without explicit tie-breaks).
3. Unify prediction persistence path and remove duplicated write semantics.

---

## Methodology

This is a debt-focused second pass over the workspace/predictions/artifacts process, based on targeted code inspection and test inspection.

Key files inspected:

- `nirs4all/pipeline/execution/executor.py`
- `nirs4all/pipeline/execution/refit/executor.py`
- `nirs4all/pipeline/execution/refit/stacking_refit.py`
- `nirs4all/pipeline/storage/chain_builder.py`
- `nirs4all/pipeline/storage/workspace_store.py`
- `nirs4all/pipeline/trace/execution_trace.py`
- `nirs4all/pipeline/config/context.py`
- `nirs4all/pipeline/resolver.py`
- `nirs4all/pipeline/storage/artifacts/artifact_registry.py`
- `nirs4all/data/predictions.py`
- `nirs4all/workspace/__init__.py`

Tests reviewed:

- `tests/unit/pipeline/execution/refit/test_refit_executor.py`
- `tests/unit/pipeline/execution/refit/test_refit_p2c.py`
- `tests/unit/pipeline/test_resolver.py`
- `tests/unit/pipeline/storage/test_chain_replay.py`

Execution note:

- Runtime tests were not executed in this environment because `pytest` is not installed (`python3 -m pytest` -> `No module named pytest`).

---

## Debt Register (Prioritized)

### [P0] Refit persistence lifecycle is split across phases (labeling after write)

Evidence:

- Refit execution writes via `executor.execute(...)` before relabeling:
  - `nirs4all/pipeline/execution/refit/executor.py:143`
  - `nirs4all/pipeline/execution/refit/executor.py:158`
- `PipelineExecutor.execute()` flushes predictions to store during execution:
  - `nirs4all/pipeline/execution/executor.py:231`
  - `nirs4all/pipeline/execution/executor.py:232`
  - `nirs4all/pipeline/execution/executor.py:351`
  - `nirs4all/pipeline/execution/executor.py:368`
- Same ordering exists in stacking refit branches/meta path:
  - `nirs4all/pipeline/execution/refit/stacking_refit.py:656`
  - `nirs4all/pipeline/execution/refit/stacking_refit.py:671`
  - `nirs4all/pipeline/execution/refit/stacking_refit.py:736`
  - `nirs4all/pipeline/execution/refit/stacking_refit.py:751`

Debt mechanics:

- Business semantics (`fold_id="final"`, `refit_context`) are transformed after store writes.
- In-memory and persisted representations can diverge.

Impact:

- Refit rows in store can be materially wrong for analytics/replay selection.
- `v_aggregated_predictions` excludes by `refit_context IS NULL`; wrong labels distort reporting (`nirs4all/pipeline/storage/store_schema.py:175`).

Paydown:

1. Move relabeling to pre-persistence stage, or make persistence refit-aware by phase.
2. Add an integration test that asserts store rows for refit have `fold_id="final"` and non-null `refit_context`.

---

### [P0] Fold key contract is fragmented across trace, chain, store, and providers

Evidence:

- Chain builder rewrites fold keys with `fold_` prefix:
  - `nirs4all/pipeline/storage/chain_builder.py:119`
  - `nirs4all/pipeline/storage/chain_builder.py:120`
- Export/replay detect refit via `"final"` key:
  - `nirs4all/pipeline/storage/workspace_store.py:1472`
  - `nirs4all/pipeline/storage/workspace_store.py:2053`
- Cleanup includes explicit compatibility shim for both styles:
  - `nirs4all/pipeline/storage/workspace_store.py:1824`
  - `nirs4all/pipeline/storage/workspace_store.py:1835`
- Trace type/deserialization assumes integer fold keys:
  - `nirs4all/pipeline/trace/execution_trace.py:80`
  - `nirs4all/pipeline/trace/execution_trace.py:126`
- Provider refit lookup assumes string key `"final"`:
  - `nirs4all/pipeline/config/context.py:913`

Debt mechanics:

- Multiple key dialects are in active use.
- Normalization happens ad hoc in different modules, not at a single boundary.

Impact:

- Fragile replay/export behavior and high regression risk.
- New features must duplicate normalization logic or break.

Paydown:

1. Define one canonical store contract for fold keys (`"0"`, `"1"`, `"final"`, `"avg"`, `"w_avg"`).
2. Normalize only at ingestion boundaries (trace import, chain build), never in downstream business logic.
3. Type fold IDs in trace/providers as `str | int` only if unavoidable, otherwise converge to `str`.

---

### [P0] Resolver chain selection is heuristic and non-deterministic under branch ambiguity

Evidence:

- Branch match compares raw serialized JSON string:
  - `nirs4all/pipeline/resolver.py:1303`
  - `nirs4all/pipeline/resolver.py:1304`
- Final fallback is first chain row:
  - `nirs4all/pipeline/resolver.py:1330`
- Resolver also includes legacy filesystem fallback path when store resolve fails:
  - `nirs4all/pipeline/resolver.py:588`
  - `nirs4all/pipeline/resolver.py:620`
  - `nirs4all/pipeline/resolver.py:1416`

Debt mechanics:

- Selection relies on partial metadata and brittle serialization assumptions.
- Determinism depends on incidental row ordering.

Impact:

- Wrong chain can be replayed in branched/multi-chain pipelines without hard failure.
- Hard-to-diagnose prediction mismatches.

Paydown:

1. Parse `branch_path` to canonical list and match structurally.
2. Introduce deterministic tie-break order and explicit error on unresolved ambiguity.
3. Add branch-heavy integration tests for resolver selection.

---

### [P1] Store-first and filesystem-legacy resolution both remain active

Evidence:

- Store-first then filesystem fallback behavior:
  - `nirs4all/pipeline/resolver.py:589`
  - `nirs4all/pipeline/resolver.py:590`
  - `nirs4all/pipeline/resolver.py:594`
  - `nirs4all/pipeline/resolver.py:628`
- Filesystem path dependencies still include manifest and pipeline JSON:
  - `nirs4all/pipeline/resolver.py:630`
  - `nirs4all/pipeline/resolver.py:1434`

Debt mechanics:

- Two control planes (DuckDB vs run-folder files) with overlapping responsibilities.
- Behavior depends on which path succeeds first.

Impact:

- Operational complexity and inconsistent replay behavior across environments.
- Migration and debugging burden.

Paydown:

1. Define an explicit mode policy: store-only, filesystem-only, or controlled fallback with observability.
2. Emit structured warnings whenever fallback path is used.
3. Long-term: retire legacy filesystem resolution for production runtime.

---

### [P1] Prediction persistence has duplicated write paths with divergent semantics

Evidence:

- Runtime write path in executor:
  - `nirs4all/pipeline/execution/executor.py:280`
- Separate `Predictions.flush()` write path with empty-string defaults for IDs:
  - `nirs4all/data/predictions.py:378`
  - `nirs4all/data/predictions.py:394`
  - `nirs4all/data/predictions.py:395`
- `Predictions.flush()` is not used in main pipeline execution path:
  - `rg "\\.flush\\("` only finds docs/examples for predictions flush usage.

Debt mechanics:

- Two persistence implementations encode lifecycle assumptions differently.
- Contract drift is likely over time.

Impact:

- Hidden behavioral differences if callers use `Predictions.flush()` directly.
- Increased maintenance cost and test surface.

Paydown:

1. Consolidate to one write path (executor helper or `Predictions.flush()`, not both).
2. If both are retained, formalize shared mapper and schema contract tests.

---

### [P1] Artifact dedup integrity check is incomplete

Evidence:

- `_find_existing_by_hash()` returns first file matching short hash pattern:
  - `nirs4all/pipeline/storage/artifacts/artifact_registry.py:1305`
  - `nirs4all/pipeline/storage/artifacts/artifact_registry.py:1308`

Debt mechanics:

- Comment says content should be verified, but bytes are not checked before reuse.

Impact:

- Low probability collision risk with high correctness impact.

Paydown:

1. Verify full content hash on candidate file before dedup reuse.
2. Add collision-focused test around short-hash matching path.

---

### [P1] Schema is weak on natural-key idempotency and duplicate prevention

Evidence:

- `predictions` table has PK on `prediction_id`, but no uniqueness on semantic keys:
  - `nirs4all/pipeline/storage/store_schema.py:75`
  - `nirs4all/pipeline/storage/store_schema.py:99`
- Indexes are performance-only, not integrity constraints:
  - `nirs4all/pipeline/storage/store_schema.py:186`
  - `nirs4all/pipeline/storage/store_schema.py:194`

Debt mechanics:

- Idempotency depends on caller-generated IDs and behavior discipline.

Impact:

- Duplicate semantic predictions are possible and difficult to detect.

Paydown:

1. Define natural key for prediction identity in training context.
2. Add uniqueness constraint (or upsert guard) for that natural key where appropriate.

---

### [P1] Chain replay API advertises wavelength awareness but transform path is generic

Evidence:

- Replay docs mention wavelength-aware operators:
  - `nirs4all/pipeline/storage/workspace_store.py:2025`
  - `nirs4all/pipeline/storage/workspace_store.py:2027`
- Transform application uses plain `.transform(X_current)`:
  - `nirs4all/pipeline/storage/workspace_store.py:2074`

Debt mechanics:

- Contract and implementation are misaligned.

Impact:

- Potentially incorrect replay for spectral transformers needing wavelength context.

Paydown:

1. Define and enforce transform contract for replay path (`transform(X, wavelengths=...)` when supported).
2. Add replay tests with wavelength-aware transformers.

---

### [P2] Workspace global state is process-global mutable singleton

Evidence:

- Global mutable `_active_workspace`:
  - `nirs4all/workspace/__init__.py:14`
  - `nirs4all/workspace/__init__.py:50`
  - `nirs4all/workspace/__init__.py:61`

Debt mechanics:

- Implicit ambient state controls storage root across runtime.

Impact:

- Concurrency and multi-tenant service scenarios are fragile.

Paydown:

1. Prefer explicit workspace injection through runner/store constructors.
2. Keep global helper as CLI convenience only, not production service dependency.

---

## Test and Observability Debt

### Coverage and test-shape gaps

1. Refit tests validate relabeling behavior but mostly with mocked executor/store boundaries.
   - `tests/unit/pipeline/execution/refit/test_refit_executor.py:440`
2. Resolver tests emphasize legacy artifact-id patterns and do not stress V3 + branch ambiguity.
   - `tests/unit/pipeline/test_resolver.py:137`
3. Chain replay unit test is delegation-only and does not validate transform/model contracts end-to-end.
   - `tests/unit/pipeline/storage/test_chain_replay.py:17`

### Missing debt controls

1. No contract tests for fold-key normalization across trace -> chain -> store -> replay.
2. No deterministic-selection tests for branched chain resolution with ambiguous metadata.
3. No integration test asserting persisted refit labels and contexts in DuckDB rows.

---

## Sequenced Paydown Plan

### Phase 1 (Integrity and Contracts)

1. Fix refit persistence ordering so persisted rows are already relabeled.
2. Normalize fold-key contract and remove downstream key-shim logic.
3. Add contract tests for refit labels and fold-key normalization.

### Phase 2 (Determinism and Resolution)

1. Refactor resolver chain selection to structural branch matching.
2. Add deterministic tie-breaks and fail-fast behavior on ambiguity.
3. Add branch-heavy resolver integration tests.

### Phase 3 (Consolidation)

1. Unify prediction write path and mapper logic.
2. Decide store-only vs dual-mode resolution strategy for production.
3. Add explicit observability for fallback path usage.

### Phase 4 (Hardening)

1. Enforce dedup verification by full-content hash.
2. Add natural-key uniqueness strategy for predictions.
3. Align replay transform contract with wavelength-aware operator support.

---

## Proposed Success Criteria

1. Refit rows in store always have correct `fold_id` and `refit_context`.
2. Fold key style is singular and documented at one boundary.
3. Resolver selection is deterministic and test-proven for branched pipelines.
4. Prediction persistence uses one canonical mapping path.
5. Replay/export behavior is contract-tested for refit and wavelength-aware chains.
