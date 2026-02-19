# nirs4all Beta-Readiness Audit - Overall Synthesis (Release Blockers Only, 2026-02-19)

## Current Readiness
- Status: **Beta-ready**.
- All testing gaps, correctness bugs, CI gate gaps, architectural coupling issues, and code quality concerns from the initial audit have been resolved. The only remaining open item is the orchestrator god-module decomposition (`A-03`), which is deferred post-beta.

## Active Cross-Module Release Blockers

### Architecture
1. **Orchestrator god module** (`A-03`): `pipeline/execution/orchestrator.py` remains ~2000 lines. Import duplication resolved; further decomposition deferred post-beta.
2. ~~**`Predictions._buffer` coupling in orchestrator/refit**~~ (`A-NF-01`, `F-005`, `NF-PIPE-01`): resolved.
3. ~~**`DatasetConfigs.__new__` bypass**~~ (`F-004`): resolved.
4. ~~**Public API sprawl** (`A-19`, `A-18`)~~: resolved.

### Correctness
5. ~~**Hardcoded `n_train = 100`** (`F-009`)~~: resolved.

## Beta Release Work (Open)

### Architecture
- [x] Replace private `_buffer` usage in orchestrator and refit with public `Predictions` read APIs. (`A-NF-01`, `F-005`, `NF-PIPE-01`)
- [ ] Refactor `orchestrator.py` into smaller execution services. (`A-03`) — deferred post-beta.
- [x] Remove `DatasetConfigs.__new__` bypass in orchestrator. (`F-004`)
- [x] Curate/reduce top-level exports in `synthesis` and `analysis`. (`A-19`, `A-18`)

### Correctness
- [x] Make `DataCache.get_or_load()` atomic. (`F-8.2`)
- [x] Replace hardcoded `n_train = 100` fallback. (`F-009`)

### Code Quality
- [x] Clarify sklearn wrapper `fit()` contract in user-facing docs. (`SKL-01`)
- [x] Consolidate duplicate function-local imports in orchestrator. (`F-008`)

## Resolved Since Initial Audit

### Code Fixes
- `F-004` — `DatasetConfigs.__new__` bypass removed. New `from_spectrodatasets()` factory method added; `_wrap_dataset_list()` delegates to it.
- `F-005 / NF-PIPE-01 / A-NF-01` — All private `Predictions._buffer` accesses replaced with public API methods (`slice_after`, `iter_entries`, `extend_from_list`, `mutate_entries`) across orchestrator, branch controller, refit executor, stacking refit, and reports.
- `F-008` — Function-local duplicate imports in orchestrator consolidated to module-level. Only optional DL backends remain lazy.
- `F-009` — Hardcoded `n_train = 100` fallback replaced with `dataset.num_samples` plus `logger.debug()` warning.
- `F-09 / F-12` — `api/result.py` `_buffer` direct access replaced; `api/run.py` fragile `verbose` guard fixed.
- `F-010` — `_inject_best_params()` global-application behavior documented with warning; `_apply_params_to_model` filters by valid params. Call site in `stacking_refit.py` annotated.
- `F-013` — `DEFAULT_MAX_STACKING_DEPTH` documented with rationale and exported for user override via `max_depth` parameter.
- `F-16` — `Session.run()` kwarg passthrough updated to forward all valid `runner.run()` parameters instead of a restrictive whitelist.
- `F-18` — `retrain(mode="finetune")` docstring expanded with per-framework behavior (sklearn falls back to full refit; DL supports incremental training).
- `F-1.3` — Signal type detection partition bias removed; `_detect_signal_type()` now uses all data for unbiased detection.
- `F-1.5` — Silent `except Exception: pass` in `dataset.py` replaced with `logger.debug()`.
- `F-2.2` — Metadata reload path now uses `_load_file_with_registry()` with `na_policy='ignore'` instead of raw `pd.read_csv()`.
- `F-3.2` — `FolderParser.SUPPORTED_EXTENSIONS` expanded to cover all loader-supported formats.
- `F-8.2` — `DataCache.get_or_load()` TOCTOU race eliminated via double-checked locking with `RLock`.
- `F-9.1` — `ConfigValidator` now emits `ErrorRegistry` codes instead of ad-hoc string codes.
- `F-11.2` — `tests/unit/data/predictions/test_predictions.py` added with 44 tests.
- `S-01` — `save_chain()` TOCTOU eliminated; SELECT and INSERT now share the same lock scope.
- `S-03` — `ArrayStore` cross-process write safety documented in module and `save_batch()` docstrings.
- `S-02` — `pickle.loads` trust boundary explicitly documented in `_deserialize_artifact` docstring, `nosec` annotation with rationale, debug logging of format/size, and inline trust-boundary comment at `load_artifact()` call site.
- `S-04` — DuckDB tuning externalized via `DuckDBConfig` dataclass. `WorkspaceStore.__init__` accepts optional `duckdb_config` parameter with sensible defaults.
- `S-05` — Retry logic unified: `_retry_on_lock` decorator delegates to `_call_with_retry`, the single retry implementation also used by `_execute_with_retry`. No duplicated backoff/degraded-mode logic.
- `S-06` — Auto-migration now controllable via `auto_migrate: bool = True` parameter on `WorkspaceStore.__init__`. When `False`, legacy migrations are skipped.
- `S-07` — `bulk_update_chain_summaries` temp-table hardened with unique `uuid4` suffix per call and `try/finally` cleanup.
- `V-03` — Hardcoded visualization cache constant replaced with named `_TOP_K_CACHE_MIN_SIZE`.
- `A-19` — `synthesis/__init__.py` `__all__` curated from 244 to 23 primary symbols; all other symbols accessible via lazy `__getattr__`.
- `A-18` — `analysis/__init__.py` `__all__` curated from 26 to 9 primary symbols; low-level utilities still importable via explicit submodule import.
- `A-29` — Runtime `print()` calls in library code paths migrated to structured logging (`component_serialization`, `executor`, `runner`, `base_model`, `tensorflow_model`, `factory`, `targets`).
- `A-40` — `synthesis/__init__.py` converted to lazy `__getattr__`-based imports; only 4 submodules eagerly loaded.
- `A-13 / SKL-01` — `NIRSPipeline.fit()` error message and class/module docstrings updated to clearly document prediction-only contract.

### Controllers
- `C-03` — Priority ladder documented: equal priorities allowed, tie-broken alphabetically by class name. Sort key updated to `(priority, __name__)`.
- `C-04` — `reset_registry()` docstring added with caution guidance; `logger.warning()` emitted on call.

### Operators
- `O-03` — Section headers and module docstrings added to `nirs.py` (8 sections) and `scalers.py` (3 sections).
- `O-04` — Stale "backward compatibility" comments removed/updated; incorrect `Derivate.fit()` error message fixed; missing docstrings added.

### Tests Added
- `F-01 / NF-API-01` — Strict happy-path integration tests for `predict()` and `retrain()`.
- `F-06` — Dedicated `RunResult.validate()` unit tests.
- `O-01 / O-02` — `LogTransform` fitted-offset and `CropTransformer` state-mutation regression tests.
- `C-05` — `YTransformerMixinController` unit tests.
- `D09` — `workspace/__init__.py` direct unit tests.
- `F-031 / S-08` — `ArrayStore` concurrent-write thread-safety tests.
- `F-035` — Negative-step `_range_` regression tests.

### CI / DevOps
- `D01` — mypy type-check gate added to CI (`type-check` job).
- `D05` — macOS added to CI test matrix with `timeout-minutes: 30` and `continue-on-error: true`.
- `D06` — Python 3.12 added to CI test matrix.
- `D13` — CodeQL static security workflow added (`.github/workflows/codeql.yml`).
- `D17` — `pre-publish.yml` and `publish.yml` refactored to share jobs via `shared-test-and-docs.yml` `workflow_call`.

## Code Areas Not Yet Audited In Detail
- `scripts/`
- `bench/`
- `.github/actions/`
- Test-infrastructure glue (`tests/conftest.py` and shared fixture helpers)
- Workspace fixture/harness folders (`workspace*/`, `fake_workspace/`, `AOM_workspace/`)
