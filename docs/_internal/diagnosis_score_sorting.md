# Diagnosis: Score Sorting / Ranking System

**Date**: 2026-03-24
**Status**: Multiple bugs confirmed — wrong sort order for classification metrics, fragmented direction logic

---

## 1. Score System Overview

The scoring system flow:

1. **Computation**: `ScoreCalculator` (`controllers/models/components/score_calculator.py`) calls `ModelUtils.get_best_score_metric()` to determine the primary metric and its direction (`higher_is_better`). Regression → `rmse` (lower is better). Classification → `balanced_accuracy` (higher is better).

2. **Storage**: Predictions are stored via `Predictions.add_prediction()` with `metric` (string) and `val_score`/`test_score`/`train_score` (floats). The **direction** (`higher_is_better`) is **NOT stored** — it must be re-inferred from the metric name string every time.

3. **Ranking**: `Predictions.top()` uses `_infer_ascending(metric)`. `WorkspaceStore.top_predictions()` uses `_infer_metric_ascending(metric)`. Multiple other locations have their own independent copies.

4. **Display**: `TabReportManager.generate_per_model_summary()` receives `ascending` from the caller and sorts accordingly.

---

## 2. Bugs Found

### BUG 1 (Critical): Hardcoded `ascending=True` in global summary

**File**: `nirs4all/pipeline/execution/orchestrator.py:1962`

```python
ascending=True,  # Will be adjusted based on metric
```

The comment says "Will be adjusted based on metric" but it is **never adjusted**. For classification tasks where `metric="balanced_accuracy"`, this sorts in ascending order (lowest first), meaning the **worst** model is ranked #1.

The per-dataset version at line 1555 correctly uses `asc = _infer_ascending(metric)`, making this an inconsistency between the two code paths.

**Fix**: Replace with `ascending=_infer_ascending(metric)`. The metric variable is already available at line 1948.

### BUG 2 (Moderate): CLI `--ascending` flag parsed but never passed

**File**: `nirs4all/cli/commands/workspace.py:79-83`

```python
top_df = store.top_predictions(
    n=args.n,
    dataset_name=args.dataset,
    metric=args.metric,
)
```

`args.ascending` is parsed (line 225) but never forwarded to `top_predictions()`. The function defaults to `ascending=True`, which is correct only for error metrics (rmse, mae). For classification metrics, results are shown in wrong order.

**Fix**: Pass `ascending=args.ascending` if user specified it, otherwise infer from metric.

### BUG 3 (Moderate): Incomplete hardcoded metric list in summary display

**File**: `nirs4all/controllers/models/base_model.py:1367`

```python
direction = get_symbols().direction(metric in ['r2', 'accuracy', 'balanced_accuracy'])
```

This inline list is missing: `f1`, `precision`, `recall`, `auc`, `roc_auc`, `kappa`, `rpd`, `rpiq`. Direction arrows are wrong for all these metrics.

**Fix**: Replace with a call to a centralized `is_higher_better(metric)` function.

### BUG 4 (Design): `METRIC_METADATA` in `pipeline/run.py` missing metrics

**File**: `nirs4all/pipeline/run.py:44-68`

`METRIC_METADATA` does not include `balanced_accuracy`, `kappa`, `rpiq`, or `specificity`. When `get_metric_info("balanced_accuracy")` is called, it falls to the `"default"` entry which happens to have `higher_is_better: True` — works by accident. But this is fragile and undocumented.

---

## 3. Root Cause: 8+ Duplicated Metric Direction Sets

The fundamental design issue is that **metric direction is defined in 8+ independent locations** with inconsistent coverage:

| Location | File | Approach | Missing Metrics |
|----------|------|----------|----------------|
| `_infer_ascending()` | `data/predictions.py:81` | "higher is better" set | — (most complete) |
| `_HIGHER_IS_BETTER_METRICS` | `pipeline/storage/workspace_store.py:190` | frozenset | — (tied with predictions.py) |
| `_infer_ascending()` | `pipeline/execution/refit/model_selector.py:271` | set | `kappa`, `rpd`, `rpiq` |
| `_infer_ascending()` | `pipeline/execution/refit/stacking_refit.py:441` | set | `kappa`, `rpd`, `rpiq` |
| inline list | `controllers/models/base_model.py:1367` | list | `f1`, `precision`, `recall`, `auc`, `roc_auc`, `kappa`, `rpd`, `rpiq` |
| `HIGHER_IS_BETTER_METRICS` | `visualization/charts/base.py:307` | list | — (most comprehensive list overall) |
| copy of above | `visualization/predictions.py:358` | list | Duplicate of base.py |
| copy of above | `visualization/charts/heatmap.py:785` | list | Duplicate of base.py |
| `_is_higher_better()` | `data/ensemble_utils.py:239` | set + heuristic | Has `"score"` with fallback matching |
| `LOWER_IS_BETTER_METRICS` | `controllers/shared/model_selector.py:62` | inverse set | Uses opposite approach (lists lower-is-better) |
| `METRIC_METADATA` | `pipeline/run.py:44` | dict | `balanced_accuracy`, `kappa`, `rpiq` |
| `METRIC_DIRECTION` | `optimization/optuna.py:44` | dict | Very small subset |

The inconsistencies mean that depending on which code path processes a score, the sort direction can differ. For example, `kappa` is recognized in `predictions.py` and `workspace_store.py` but NOT in `model_selector.py` or `stacking_refit.py`.

---

## 4. How `balanced_accuracy` Specifically Fails

| Stage | Code | Correct? |
|-------|------|----------|
| Score computation | `ScoreCalculator` → `ModelUtils.get_best_score_metric()` returns `("balanced_accuracy", True)` | ✅ |
| Per-dataset refit report | `orchestrator.py:1555` calls `_infer_ascending("balanced_accuracy")` → `False` → `ascending=False` | ✅ |
| **Global summary report** | `orchestrator.py:1962` hardcodes `ascending=True` | ❌ **Worst model shown as #1** |
| In-memory `top()` | `predictions.py:81` `_infer_ascending("balanced_accuracy")` → `False` | ✅ |
| Model selector (variant selection) | `model_selector.py:271` includes `balanced_accuracy` in higher-is-better set | ✅ |
| Summary direction arrow | `base_model.py:1367` includes `balanced_accuracy` in list | ✅ (but `f1` etc. are wrong) |

So the primary user-visible bug is: **the global summary table at the end of a multi-dataset run shows classification results in wrong order**.

---

## 5. Proposed Fix Strategy

### Immediate (Bug Fixes)

1. **`orchestrator.py:1962`** — Replace `ascending=True` with `ascending=_infer_ascending(metric)`
2. **`workspace.py:79`** — Pass `ascending` to `top_predictions()`
3. **`base_model.py:1367`** — Replace inline list with centralized function call

### Architectural (Eliminate Duplication)

4. **Create a single source of truth** — A new module (e.g., `nirs4all/core/metrics.py` or extend `pipeline/run.py`) that exports:
   - `is_higher_better(metric: str) -> bool`
   - `infer_ascending(metric: str) -> bool` (= `not is_higher_better`)

   Use the comprehensive set from `visualization/charts/base.py` as the reference, expanded to cover all known metrics.

5. **Replace all 8+ copies** with calls to this single function.

6. **Add `balanced_accuracy`, `kappa`, `rpiq`, `specificity`** to `METRIC_METADATA` in `pipeline/run.py`.

### Future Enhancement

7. **Store `higher_is_better` in prediction metadata** — Instead of re-inferring from the metric name string every time, store the boolean at prediction save time. This makes the system robust to unknown custom metrics where the user could specify direction explicitly.

---

## 6. Test Coverage Assessment

### Existing Tests

| Test File | What it covers | Gap |
|-----------|---------------|-----|
| `tests/unit/data/test_prediction_ranking.py:TestMetricDirection` | RMSE ascending, R2 descending | No `balanced_accuracy` or classification metrics |
| `tests/unit/pipeline/execution/refit/test_model_selector.py:TestMetricInference` | rmse, mse, r2, accuracy, f1 | No `balanced_accuracy` |
| `tests/unit/pipeline/storage/test_aggregated_predictions.py` | infer_ascending for rmse, r2, accuracy, mae | Limited coverage |
| `tests/unit/test_scoring_invariants.py` | Sorting with `ascending=True` and `metric="rmse"` | Always regression, never classification |

### Why Tests Didn't Catch It

- No test exercises the global summary code path (`_print_global_final_summary`)
- All sorting tests use regression metrics where `ascending=True` happens to be correct
- No test verifies that classification pipelines produce correctly sorted results

### Missing Tests (Required)

1. **Global summary sort order for classification**: Run a classification pipeline with multiple variants, assert that the global summary table shows highest `balanced_accuracy` first
2. **Cross-consistency test**: Assert that ALL copies of the metric direction logic agree for every known metric — this is the guard against future drift
3. **`Predictions.top()` with classification metrics**: Add test for `balanced_accuracy`, `f1`, `kappa` to `test_prediction_ranking.py`
4. **CLI `--ascending` flag**: Test that `workspace_query_best` respects the flag
5. **Direction arrow correctness**: Test that `_print_prediction_summary` shows correct arrow for `f1`, `recall`, `auc`, `kappa`, `rpd`
6. **Unknown metric fallback**: Test that a custom metric name (e.g., `"my_custom_score"`) gets a reasonable default direction (currently falls to `ascending=True` which is wrong if it's an accuracy-like metric)
7. **End-to-end integration**: Run classification pipeline → check `RunResult.best` actually has the highest balanced_accuracy (not lowest)
8. **`METRIC_METADATA` completeness**: Assert that every metric in the comprehensive visualization list is also present in `METRIC_METADATA`

### Existing Tests That Need Fixes

- `tests/unit/test_scoring_invariants.py` — Add classification variants (balanced_accuracy, f1) alongside the existing rmse tests
- `tests/unit/data/test_prediction_ranking.py:TestMetricDirection` — Add `balanced_accuracy`, `f1`, `kappa` direction tests
- `tests/unit/pipeline/execution/refit/test_model_selector.py:TestMetricInference` — Add `balanced_accuracy` test case

---

## 7. Full Inventory of Affected Files

| File | Line(s) | Issue | Severity |
|------|---------|-------|----------|
| `pipeline/execution/orchestrator.py` | 1962 | Hardcoded `ascending=True` | **Critical** |
| `cli/commands/workspace.py` | 79-83 | `args.ascending` ignored | Moderate |
| `controllers/models/base_model.py` | 1367 | Incomplete metric list for direction arrows | Moderate |
| `pipeline/run.py` | 44-68 | Missing metrics from `METRIC_METADATA` | Design |
| `pipeline/execution/refit/model_selector.py` | 271 | Duplicate `_infer_ascending`, missing metrics | Design |
| `pipeline/execution/refit/stacking_refit.py` | 441 | Duplicate `_infer_ascending`, missing metrics | Design |
| `data/predictions.py` | 72-82 | Duplicate `_infer_ascending` (source of truth candidate) | Design |
| `pipeline/storage/workspace_store.py` | 190-206 | Duplicate `_infer_metric_ascending` | Design |
| `controllers/shared/model_selector.py` | 62 | Different approach (`LOWER_IS_BETTER_METRICS`) | Design |
| `data/ensemble_utils.py` | 228-258 | Duplicate `_is_higher_better` with heuristic | Design |
| `visualization/charts/base.py` | 307-317 | Duplicate (most comprehensive list) | Design |
| `visualization/predictions.py` | 358-367 | Duplicate of base.py | Design |
| `visualization/charts/heatmap.py` | 785-795 | Duplicate of base.py | Design |
| `optimization/optuna.py` | 44-55 | Duplicate `METRIC_DIRECTION` | Design |

---

## 8. Relationship to Existing Analysis

There is an existing document at `docs/_internal/specifications/ranking_system_analysis.md` that describes the desired ranking logic. That document correctly identifies the need for ascending/descending inference (Step 4 in its flow diagram). The bugs documented here show that the implementation diverged from that specification — specifically, the global summary path hardcodes `ascending=True` instead of inferring it.
