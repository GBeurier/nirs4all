# Beta-Readiness Audit: Pipeline Module (`nirs4all/pipeline/`) - Release Blockers Only (2026-02-19)

## Active Findings

None.

## Beta Release Tasks (Open)

None.

## Resolved Findings
- `F-004 [RESOLVED]` `DatasetConfigs.__new__` bypass removed. New `DatasetConfigs.from_spectrodatasets()` factory method added; `_wrap_dataset_list()` in orchestrator now delegates to it (`data/config.py`, `pipeline/execution/orchestrator.py`).
- `F-005 [RESOLVED]` Private `Predictions._buffer` access replaced with public API methods (`slice_after`, `iter_entries`, `extend_from_list`, `mutate_entries`, `num_predictions`, `filter_predictions`) across orchestrator, branch controller, refit executor, stacking refit, and reports (`data/predictions.py`, `pipeline/execution/orchestrator.py`, `controllers/data/branch.py`, `pipeline/execution/refit/executor.py`, `pipeline/execution/refit/stacking_refit.py`, `visualization/reports.py`).
- `F-008 [RESOLVED]` Function-local duplicate imports in orchestrator consolidated to module-level imports. Only optional deep-learning backends (`torch`, `tensorflow`) remain lazy-loaded (`pipeline/execution/orchestrator.py`).
- `F-009 [RESOLVED]` Hardcoded `n_train = 100` fallback replaced with `dataset.num_samples` and `logger.debug()` warning (`pipeline/execution/refit/executor.py`).
- `NF-PIPE-01 [RESOLVED]` See F-005.
- `F-001 [RESOLVED]` `ArrayStore.save_batch()` thread-safety verified by tests; cross-process write safety documented in module and method docstrings (`pipeline/storage/array_store.py`).
- `F-010 [RESOLVED]` `_inject_best_params()` global-application behavior documented with warning in docstring; `_apply_params_to_model` filters by valid parameter names via `set_params()` with per-key fallback. Call site in `stacking_refit.py` annotated (`pipeline/execution/refit/executor.py`).
- `F-013 [RESOLVED]` `DEFAULT_MAX_STACKING_DEPTH` documented with rationale (exponential complexity, 3 levels sufficient for real-world stacking); exported in `pipeline/execution/refit/__init__.py` for user override via `max_depth` parameter (`pipeline/execution/refit/stacking_refit.py`).
- `F-031 [RESOLVED]` Dedicated concurrent `ArrayStore.save_batch()` thread-safety tests added (`tests/unit/pipeline/storage/test_array_store.py::TestArrayStoreConcurrentWrites`).
- `F-035 [RESOLVED]` Negative-step `_range_` regression tests added (`tests/unit/pipeline/config/test_generator_strategies.py::TestRangeStrategy`).
