# Refit Runtime and Score Integration Audit

Date: 2026-02-08  
Scope: `nirs4all` refit execution, prediction ranking (`top`/`get_best`), logging/reporting, storage aggregation, and visualization analyzers/charts.

## Executive Answers

1. Are final (refit) scores for all models output in logs/reports?
- Not globally, and not at the same level as CV scores today.
- Current dataset-end reporting (`_print_best_predictions`) logs one "best prediction" and one tab report for that chosen entry, not a dedicated per-model refit final table.
- In simple refit, only the winning pipeline/model is retrained, so there is not even a "final score for all candidate models" artifact by design.

2. Are final scores integrated in `top()` at the same level as other scores, with possibility to request only final?
- Partially.
- `top()` can return refit rows, but defaults are CV-oriented (`rank_partition='val'`).
- With common mixed CV+refit buffers, `top(..., fold_id='final')` returns empty unless caller also switches rank partition (typically to `rank_partition='test'`).
- `RunResult` exposes dedicated refit accessors (`final`, `final_score`), but refit is not first-class in default `best/top` semantics.

## Current Runtime Management of Refit

### Orchestrator flow
- Refit runs as a second pass after all CV variants for a dataset (`nirs4all/pipeline/execution/orchestrator.py:325`).
- It reloads a pristine dataset before refit to avoid mutated state from pass 1 (`nirs4all/pipeline/execution/orchestrator.py:330`).
- Strategy dispatch is topology-based:
  - stacking/mixed merge -> `execute_stacking_refit`
  - competing branches -> `execute_competing_branches_refit`
  - separation -> `execute_separation_refit`
  - otherwise -> `execute_simple_refit`
  (`nirs4all/pipeline/execution/orchestrator.py:666`).

### Refit labeling and persistence contract
- Runtime context marks REFIT phase and sets fold/context labels (`nirs4all/pipeline/execution/refit/executor.py:127`).
- Executor flush applies runtime overrides before persistence (`nirs4all/pipeline/execution/executor.py:437`).
- Refit relabeling explicitly enforces `fold_id='final'` and context labels:
  - standalone: `refit_context='standalone'` (`nirs4all/pipeline/execution/refit/executor.py:340`)
  - stacking: `refit_context='stacking'` (`nirs4all/pipeline/execution/refit/stacking_refit.py:836`)
- Refit rows are merged into shared prediction stores (`nirs4all/pipeline/execution/refit/executor.py:168`, `nirs4all/pipeline/execution/orchestrator.py:711`).

### Selection scope for simple refit
- Winning config extraction is global-by-pipeline best CV score (`best_val`) and produces one winning configuration (`nirs4all/pipeline/execution/refit/config_extractor.py:103`).
- For non-stacking pipelines this means one final refit model, not one per candidate model.

## Outputs and Logging Status

### What is logged today
- Dataset-end "best prediction in run" uses `run_dataset_predictions.get_best()` (`nirs4all/pipeline/execution/orchestrator.py:745`).
- Tab report uses partition view of that same selected entry (`nirs4all/pipeline/execution/orchestrator.py:751`).
- Refit completion log exists but is count-oriented, not score-table oriented (`nirs4all/pipeline/execution/orchestrator.py:713`).

### Why refit finals are not first-class in default report
- `get_best()` ranks by `val` first, then falls back to `test` (`nirs4all/data/predictions.py:769`).
- When CV validation rows exist, default selection generally stays CV-centric.
- There is no dedicated "Refit final scores by model" report emitted at dataset-end.

## `top()` / `best` / RunResult Semantics

### Current behavior
- `Predictions.top()` defaults to `rank_partition='val'` (`nirs4all/data/predictions.py:486`).
- Partition filter is applied before extra filters (`nirs4all/data/predictions.py:563` then `:570`).
- `RunResult.best` is `predictions.top(n=1)` with default ranking (`nirs4all/api/result.py:275`).

### Practical consequence (important)
- In mixed CV+refit predictions, this call often returns empty:
  - `top(n=5, fold_id='final')`
- Because CV rows satisfy the default `val` partition prefilter first, then `fold_id='final'` is applied to already-val-only candidates.
- Refit-only retrieval currently requires caller intent, e.g.:
  - `top(n=5, fold_id='final', rank_partition='test')`

### Existing refit-specific accessors
- `RunResult.final` and `RunResult.final_score` exist (`nirs4all/api/result.py:390`, `:406`).
- `RunResult.cv_best` explicitly excludes refit rows (`nirs4all/api/result.py:422`).
- This is functional, but split into separate pathways instead of unified score-scope semantics.

## Storage and Aggregation Status

- Refit context exists in schema (`nirs4all/pipeline/storage/store_schema.py:98`).
- Aggregated view intentionally excludes refit rows: `WHERE p.refit_context IS NULL` (`nirs4all/pipeline/storage/store_schema.py:175`).
- Therefore `query_aggregated_predictions` / `query_top_aggregated_predictions` are CV-only by design.
- This exclusion is also codified in tests (`tests/unit/pipeline/execution/refit/test_refit_infrastructure.py:323`).

## Prediction Analyzer and Charts

- Analyzer cache/query defaults are CV-centric (`rank_partition='val'`) (`nirs4all/visualization/predictions.py:232`).
- Chart base ranking helper also defaults to `val` (`nirs4all/visualization/charts/base.py:183`).
- There is no dedicated refit score mode/scope across charts.
- Refit rows can be visualized only by manually passing compatible filters/partitions, so integration is partial and non-standard.

## Logical Target for Correct Overall Functioning

To integrate refit scores correctly across modules, define one explicit concept used everywhere:

- `score_scope = 'cv' | 'final' | 'all'`

Semantics:
- `cv`: rank on validation-centric CV entries (current default behavior).
- `final`: rank on refit final entries (`fold_id='final'`, partition typically `test`).
- `all`: include both with explicit tie-breaking/priority rules.

This removes ambiguous behavior and makes refit first-class without breaking CV workflows.

## Recommended Changes by Module

### 1) Ranking API (`Predictions`, `RunResult`)
- Add `score_scope` to `Predictions.top()` and `Predictions.get_best()`.
- Apply scope filtering before partition filtering.
- Define scope defaults:
  - `score_scope='cv'` for backward compatibility.
  - `score_scope='final'` convenience path for refit-only retrieval.
- Add `RunResult.top_final(...)` and optionally `RunResult.best_final` as explicit shortcuts.
- Keep `RunResult.final/final_score` as lightweight accessors.

### 2) Logging and printed reports
- Extend orchestrator dataset-end reporting to print both:
  - best CV summary
  - best/final refit summary (if available)
- Add a compact table: "Final refit scores by model/context" (model, fold=`final`, context, metric, test score).
- For non-stacking/simple refit, clearly log that final scope contains only the winning model by design.

### 3) Storage aggregation layer
- Keep existing CV-only view for compatibility.
- Add a second aggregated view or query mode that includes refit:
  - e.g. `v_aggregated_predictions_all` with `refit_context` preserved.
- Extend top-aggregated query builders with `include_refit` / `score_scope`.

### 4) Analyzer/charts
- Thread `score_scope` through `PredictionAnalyzer.get_cached_predictions(...)`.
- Add chart-level option to select `cv` vs `final` vs `all`.
- Set sensible defaults:
  - preserve `cv` as default
  - `final` auto-uses rank partition `test` unless overridden
- Label chart titles/subtitles with scope to avoid silent metric mixing.

### 5) Tests and contracts
- Add unit tests for:
  - `top(..., score_scope='final')` returning refit without requiring manual partition hacks.
  - `top(..., fold_id='final')` behavior consistency with scope.
  - `RunResult.best` + `best_final` semantics.
- Add integration tests for:
  - dataset-end logs include final summary when refit exists.
  - analyzer/charts in `final` scope.
  - aggregated query mode including refit.

## Minimal Safe Rollout Plan

1. Introduce `score_scope` in ranking API with backward-compatible defaults (`cv`).
2. Add explicit final summary logging/reporting at orchestrator dataset-end.
3. Expose `score_scope` in analyzer/charts API.
4. Add store aggregated mode including refit (without changing existing CV view behavior).
5. Add regression tests for all new contracts.

## Conclusion

Refit runtime execution and persistence are structurally in place.  
The gap is not "whether refit exists", but "where refit is first-class":
- default ranking/reporting is still CV-first,
- `top()` requires non-obvious parameters for refit-only results,
- store aggregated queries and charts do not treat final scores as a native scope.

The clean fix is a cross-module `score_scope` contract and aligned logging/reporting/query/chart behavior around that contract.
