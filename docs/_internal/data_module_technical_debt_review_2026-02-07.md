# Data Module Technical Debt Review

Date: 2026-02-07  
Scope: `nirs4all/data/**`  
Focus: technical debt, redundancy, dead code, performance, reproducibility

## Executive Summary

Key risks:
- Memory and runtime blowups from feature processing growth and unbounded dataset caches.
- Reproducibility drift caused by inconsistent dataset hashing, global RNG seeding, and cache key collisions.
- Structural redundancy across loaders/parsers/selection/partition and dormant performance modules that are not wired into runtime.
- High-complexity hotspots and unused imports that slow maintenance and mask real defects.

Highest leverage actions:
- Fix dataset hash semantics and metadata consistency end-to-end.
- Replace name-only dataset caching with bounded, configuration-keyed caching (or remove it).
- Decide on one data loading stack (new loader registry vs legacy loader), then remove the other.
- Retire or wire the dormant performance components (DataCache, LazyDataset, StepCache) to avoid false confidence.

Context alignment:
- This review aligns with findings in `docs/_internal/spectrodataset_cache_memory_overflow_investigation.md` and `docs/_internal/cache_management_design.md` on memory amplification and cache correctness.

---

## Methodology

Static scans:
- `ruff check nirs4all/data --select C901` -> 46 complexity findings.
- `ruff check nirs4all/data --select F401,F841` -> 80 unused-import/unused-variable findings (67 auto-fixable).

Size profile:
- Files: 78
- LOC: 28,472
- Largest subpackages by LOC: `loaders` (4,300), `schema` (3,057), `dataset` (2,095), `parsers` (2,076), `_indexer` (1,823).

Test footprint:
- Unit tests: 86 files, 26,645 LOC in `tests/unit/data/**`.
- Integration tests: 3 files, 516 LOC in `tests/integration/data/**`.
- Additional tests referencing the data module outside those folders: 69 files.

Coverage snapshot:
- `.coverage` exists but contains no line data, and the `coverage` module is not installed in this environment. Per-file coverage could not be extracted in this audit.

---

## Debt Register (Prioritized)

### [P0] Dataset hash and metadata inconsistency

Evidence:
- `SpectroDataset.content_hash` caches `_content_hash_cache` and computes a full-data hash. `nirs4all/data/dataset.py:1481-1522`.
- `get_dataset_metadata` prefers `_content_hash` and falls back to a sampled MD5 hash. `nirs4all/data/dataset.py:1945-1955`.
- `set_content_hash` writes `_content_hash` only, not `_content_hash_cache`. `nirs4all/data/dataset.py:2032-2039`.

Impact:
- Metadata can report a hash that does not match `content_hash()`.
- Run manifests and cache keys can drift or collide, undermining reproducibility.
- Sampled hash fallback is not a true content hash and can collide for distinct datasets.

Recommendation:
- Use a single canonical attribute for the dataset hash and ensure all writers/readers agree.
- Replace the sampled MD5 fallback with a clear policy: either always compute `content_hash()` or store an explicit `quick_hash` with separate semantics.
- Propagate hash semantics into any run metadata consumers.

### [P0] DatasetConfigs cache is unbounded and keyed only by dataset name

Evidence:
- Cache is keyed by `name` and stores full arrays; no size bound or eviction. `nirs4all/data/config.py:367-393`.
- The cache is reused across different configs as long as the dataset name matches. `nirs4all/data/config.py:368-370`.

Impact:
- Reproducibility risk: same name with different config parameters can reuse stale arrays.
- Memory risk: cache retains full train/test arrays without eviction.

Recommendation:
- Key cache by a stable fingerprint of the full config, not by name alone.
- Add a byte budget and eviction policy, or remove caching and rely on explicit upstream cache layers.

### [P0] Feature processing growth and concatenation cause memory amplification

Evidence:
- `ArrayStorage.add_processing` concatenates the full 3D array on each new processing. `nirs4all/data/_features/array_storage.py:205-220`.
- `ArrayStorage.add_samples` concatenates full arrays on each sample addition. `nirs4all/data/_features/array_storage.py:161-162`.

Impact:
- Memory grows linearly with number of processings; peak memory spikes ~2x during concatenation.
- Generator-heavy pipelines can exhaust RAM.

Recommendation:
- Replace append-by-concat with chunked storage or pre-allocated growth.
- Introduce a bounded processing cache (or spill-to-disk policy) tied to byte budgets.
- Enforce guardrails (`max_processings_per_source`, `max_feature_ram_mb`).

### [P1] Dual loader stacks with inconsistent CSV auto-detection

Evidence:
- New loader registry imports `csv_loader_new` and also exposes legacy `load_csv`. `nirs4all/data/loaders/__init__.py:71-79`.
- `DatasetConfigs` still loads via legacy `loaders/loader.py`, which imports `load_csv` from `csv_loader`. `nirs4all/data/loaders/loader.py:8-11`.
- Legacy CSV auto-detection is explicitly disabled. `nirs4all/data/loaders/csv_loader.py:170-231`.

Impact:
- Two different loading paths for the same dataset can yield different parsing behavior.
- Debugging and reproducibility become path-dependent.

Recommendation:
- Pick a single loader stack as the source of truth.
- Rewire `DatasetConfigs` to the chosen stack, then deprecate/remove the other.
- Make auto-detection policy explicit and consistent across all entry points.

### [P1] Global RNG seeding inside data utilities

Evidence:
- Synthetic dataset loader sets global RNG seed. `nirs4all/data/loaders/loader.py:47-49`.
- Row selector’s stratified sampling sets global RNG seed. `nirs4all/data/selection/row_selector.py:536`.

Impact:
- Global RNG state is mutated, causing cross-run or cross-test nondeterminism.
- Downstream randomness becomes order-dependent.

Recommendation:
- Use a local `np.random.RandomState` or `numpy.random.Generator` instance.
- Avoid global `np.random.seed` inside library functions.

### [P1] Dormant caching and performance infrastructure

Evidence:
- `DataCache`, `LazyDataset`, and `LazyArray` are exported but not used in runtime. `nirs4all/data/performance/__init__.py:7-23`.
- Step cache relies on DataCache but is not wired into execution (see pipeline audit).
- Predictions cache APIs are stubs. `nirs4all/data/predictions.py:1186-1199`.
- `Features(cache=...)` flag is declared but not implemented. `nirs4all/data/features.py:21-28`.

Impact:
- Misleading API surface and maintenance overhead.
- False confidence about performance features that do not run.

Recommendation:
- Either wire these components end-to-end or deprecate/remove them.
- Document runtime caching behavior explicitly.

### [P1] Deep copy of preloaded datasets in DatasetConfigs

Evidence:
- Preloaded datasets are deep-copied per load. `nirs4all/data/config.py:361-365`.

Impact:
- Large memory and CPU overhead on variant-heavy pipelines.
- Copy semantics are opaque to users.

Recommendation:
- Use copy-on-write or shallow copy with immutable storage handles.
- Make copy behavior explicit and configurable.

### [P1] Metadata hashing path materializes full data

Evidence:
- `get_dataset_metadata` calls `self.x(None, layout="2d")` and stacks samples to compute a hash. `nirs4all/data/dataset.py:1949-1955`.

Impact:
- Unexpected memory allocations and CPU cost on a metadata call.
- Hash semantics depend on selector defaults (augmented/excluded).

Recommendation:
- Use `content_hash()` or a non-materializing hash function over storage.
- Keep metadata computation cheap and deterministic.

### [P2] High-complexity hotspots impede maintenance

Evidence:
- Ruff C901: 46 complexity violations across loaders, parsers, schema, config, predictions, detection. Examples include `DatasetConfigs.__init__` (`nirs4all/data/config.py:27`) and `load_csv` (`nirs4all/data/loaders/csv_loader.py:244`).

Impact:
- Elevated bug risk and higher onboarding cost.
- Small changes have unpredictable ripple effects.

Recommendation:
- Refactor high-complexity functions into smaller, testable units.
- Prefer composable helpers and shared parsing utilities.

### [P2] Unused imports and variables indicate code drift

Evidence:
- Ruff F401/F841: 80 findings (examples in `nirs4all/data/config.py`, `nirs4all/data/features.py`, `nirs4all/data/loaders/*`).

Impact:
- Reduced clarity and increased lint noise.
- Harder to identify real unused code.

Recommendation:
- Clean unused imports/variables and add linting to CI for data module.

---

## Redundancy and Dead Code Inventory

| Path | Status | Notes |
| --- | --- | --- |
| `nirs4all/data/performance/*` | Dormant | Exported but unused outside StepCache; not integrated into runtime. `nirs4all/data/performance/__init__.py:7-23`.
| `nirs4all/data/io.py` | Dead | Fully commented out legacy persistence implementation.
| `nirs4all/data/aggregation/*` | Unused | No runtime references found; no tests observed in data audit.
| `nirs4all/data/selection/*` | Unintegrated | Only exported and unit-tested; not wired into dataset loading or pipeline.
| `nirs4all/data/partition/*` | Unintegrated | Exported and tested, but not used in dataset loading/runtime.
| `nirs4all/data/detection/*` | Partial | Used by CLI but not integrated into loader stack or DatasetConfigs.
| `nirs4all/data/loaders/csv_loader.py` | Legacy | Still used by `DatasetConfigs` path; auto-detection disabled.
| `nirs4all/data/loaders/csv_loader_new.py` | Parallel | Used by new loader registry and archive loader.
| `nirs4all/data/_predictions/*` | Parallel | Still required by `Predictions`, but duplicates serialization/reporting logic.
| `nirs4all/data/schema/*` | Partial | Validation and schema exist; not enforced by DatasetConfigs or loader pipeline. |

---

## Performance Review

Main hotspots:
- Feature processing growth is unbounded and concatenation-based (`nirs4all/data/_features/array_storage.py:205-220`).
- Dataset loading cache retains full arrays indefinitely (`nirs4all/data/config.py:367-393`).
- Preloaded dataset deep copies can double memory on variant runs (`nirs4all/data/config.py:361-365`).
- Metadata hashing materializes full features (`nirs4all/data/dataset.py:1949-1955`).
- Predictions buffer stores arrays in memory until flush (`nirs4all/data/predictions.py:208-212`).

Secondary hotspots:
- `Metadata.to_numeric` uses row-by-row index mapping; can be O(N*M) on large datasets.
- `FeatureSource.__str__` computes full-array mean/var/min/max, which is expensive for large arrays.

---

## Reproducibility Review

Key reproducibility risks:
- Dataset hash inconsistency between `content_hash()` and metadata (`nirs4all/data/dataset.py:1481-1522`, `nirs4all/data/dataset.py:1945-1955`).
- Cache collisions due to name-only keying (`nirs4all/data/config.py:367-393`).
- Global RNG seeding inside data utilities (`nirs4all/data/loaders/loader.py:47-49`, `nirs4all/data/selection/row_selector.py:536`).
- Dual loader stacks with different parsing/detection behavior (`nirs4all/data/loaders/__init__.py:71-79`, `nirs4all/data/loaders/loader.py:8-11`).

---

## Testing and Observability Gaps

- No reliable per-file coverage snapshot available in this audit.
- Runtime cache behavior is largely implicit or stubbed (Predictions cache APIs are no-op).
- Memory growth and peak-RAM behavior are not instrumented in the data module.

---

## Recommended Plan (Sequenced)

1. Fix dataset hash semantics and metadata alignment.
2. Replace name-only dataset caching with bounded, config-fingerprinted caching.
3. Decide on and standardize one loader stack; deprecate/remove the other.
4. Introduce a bounded feature processing strategy and memory guardrails.
5. Remove or wire dormant performance features (DataCache, LazyDataset, StepCache, cache APIs).
6. Eliminate global RNG seeding in data utilities.
7. Refactor top C901 hotspots and clean unused imports/variables.

