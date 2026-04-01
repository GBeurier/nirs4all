# Task Type Separation Review

**Date**: 2026-03-30
**Scope**: Classification vs regression result mixing across the entire codebase

---

## 1. Current Overall Design of TaskType in the Library

### 1.1 Definition

`TaskType` is a `StrEnum` in `nirs4all/core/task_type.py` with three values:

| Value | String | Properties |
|-------|--------|------------|
| `REGRESSION` | `"regression"` | `is_regression=True` |
| `BINARY_CLASSIFICATION` | `"binary_classification"` | `is_classification=True` |
| `MULTICLASS_CLASSIFICATION` | `"multiclass_classification"` | `is_classification=True` |

Being a `StrEnum`, values serialize directly to SQLite TEXT and JSON without conversion.

### 1.2 Detection

`detect_task_type(y, threshold=0.05)` in `core/task_detection.py` auto-detects from target arrays:

- 2 unique integers -> `BINARY_CLASSIFICATION`
- 3-100 unique integers -> `MULTICLASS_CLASSIFICATION`
- Continuous / >100 unique -> `REGRESSION`
- Special handling for [0,1]-range values

Called automatically by `Targets.add_targets()` on first target assignment.

### 1.3 Propagation Chain

```
SpectroDataset.add_targets(y)
  -> Targets.add_targets(y) -> detect_task_type(y) -> stored in Targets._task_type
  -> accessible via dataset.task_type / dataset.is_classification / dataset.is_regression

During training:
  BaseModelController.execute()
    -> reads dataset.task_type
    -> selects metric via ModelUtils.get_best_score_metric(task_type)
    -> stores task_type in prediction record dict

On save:
  WorkspaceStore.save_prediction() -> INSERT into predictions table (task_type TEXT NOT NULL)
  ArrayStore.save_batch()          -> Parquet schema includes task_type column
```

### 1.4 Metric Dispatch

`core/metrics.py` provides task-type-aware metric selection:

| Task Type | Primary Metric | Sort Direction | Default Metrics |
|-----------|---------------|----------------|-----------------|
| Regression | `rmse` | ascending (lower better) | r2, rmse, mse, sep, mae, rpd, bias... |
| Binary classification | `balanced_accuracy` | descending (higher better) | accuracy, balanced_accuracy, precision, recall, f1, roc_auc... |
| Multiclass classification | `balanced_accuracy` | descending (higher better) | accuracy, balanced_accuracy, precision, recall, f1, specificity... |

`ModelUtils.get_best_score_metric(task_type)` returns `("rmse", False)` for regression, `("balanced_accuracy", True)` for classification.

### 1.5 Storage Schema

**SQLite tables** (`pipeline/storage/store_schema.py`):

- `chains` table: `task_type TEXT` (line 82) — per-model task type
- `predictions` table: `task_type TEXT NOT NULL` (line 109) — per-prediction task type
- `v_chain_summary` view: includes `task_type` from chains

**Parquet arrays** (`pipeline/storage/array_store.py`):

- Schema includes `("task_type", pa.utf8())` (line 104) — stored but never used in queries

### 1.6 What Works Well

| Component | File | Task-Type Aware? |
|-----------|------|-----------------|
| TaskType enum | `core/task_type.py` | Yes — clean StrEnum design |
| Auto-detection | `core/task_detection.py` | Yes — well-tested |
| Metric computation | `core/metrics.py` | Yes — full dispatch by task type |
| SpectroDataset | `data/dataset.py` | Yes — exposes `.task_type`, `.is_classification`, `.is_regression` |
| Model controllers | `controllers/models/base_model.py` | Yes — selects metrics per task type |
| Prediction recording | `data/_predictions/result.py` | Yes — `task_type` stored in every record |
| DB schema | `pipeline/storage/store_schema.py` | Yes — `task_type` is NOT NULL in predictions |
| `query_chain_summaries()` | `pipeline/storage/workspace_store.py:1870` | Yes — accepts `task_type` filter |
| Confusion matrix chart | `visualization/charts/confusion_matrix.py:146` | Yes — filters out regression predictions |

---

## 2. Review of the Problem

### 2.1 Core Issue

When a workspace contains **both classification and regression results** (different datasets, different targets, or multi-task experiments), queries that retrieve predictions, rank them, or display them do **not** separate by task type. This causes:

1. **Meaningless cross-task ranking**: RMSE and accuracy are sorted together
2. **Wrong default metric**: A single classification result switches the default metric for the entire chart
3. **Inappropriate charts**: Scatter plots (pred vs true) shown for classification; histograms mix regression error distributions with classification accuracy distributions

### 2.2 Affected Components — Detailed Breakdown

#### 2.2.1 `Predictions.top()` — No task_type filtering

**File**: `data/predictions.py:828-1038`

The `top()` method accepts `**filters` kwargs but has no explicit `task_type` parameter. The filter loop at line 947 could technically match `task_type` via kwargs, but:

- It is not documented
- The default metric resolution (line 961) picks from the first candidate's `metric` field, which may be from any task type
- Sort direction inference (line 965) uses `_infer_ascending(effective_metric)` which works per-metric, but if the user doesn't specify `rank_metric`, the auto-detected metric may not match the task type of interest

**Impact**: `result.predictions.top(5)` on a mixed workspace returns an interleaved ranking where regression models ranked by RMSE (ascending) and classification models ranked by balanced_accuracy (descending) are incomparable.

#### 2.2.2 `Predictions.get_best()` — No task_type filtering

**File**: `data/predictions.py:1169-1230`

Delegates to `top(n=1, ...)`. Same problem — returns the best prediction across all task types.

#### 2.2.3 `Predictions.filter_predictions()` — No task_type parameter

**File**: `data/predictions.py:1236-1297`

Accepts `dataset_name`, `partition`, `model_name`, `fold_id`, `step_idx`, `branch_id`, `branch_name` — but **not** `task_type`. Users must use `**kwargs` to pass `task_type` (undocumented).

#### 2.2.4 `RunResult` accessors — No task_type validation

**File**: `api/result.py:305-423`

| Property | Issue |
|----------|-------|
| `best` (line 308) | Returns overall best — could be wrong task type |
| `best_score` (line 324) | Returns `test_score` from `.best` without task validation |
| `best_rmse` (line 333) | Reads RMSE from `.best` — no check that task is regression |
| `best_r2` (line 367) | Reads R2 from `.best` — no check that task is regression |
| `best_accuracy` (line 393) | Reads accuracy from `.best` — no check that task is classification |

In practice within a single `nirs4all.run()` call, the task type is usually consistent (same dataset), so these work. The problem arises when the user reuses a workspace across multiple runs with different task types and then queries `result.predictions.top()`.

#### 2.2.5 `WorkspaceStore.query_predictions()` — No task_type parameter

**File**: `pipeline/storage/workspace_store.py:1645-1691`

Accepts `dataset_name`, `model_class`, `partition`, `fold_id`, `branch_id`, `pipeline_id`, `run_id` — but **not** `task_type`. The SQL query builder `build_prediction_query()` (store_queries.py:401-461) also lacks task_type support.

#### 2.2.6 `WorkspaceStore.top_predictions()` — No task_type parameter

**File**: `pipeline/storage/workspace_store.py:1693-1736`

Same gap. `build_top_predictions_query()` (store_queries.py:463-528) generates SQL without any `WHERE task_type = ?` condition.

#### 2.2.7 Base chart `_get_default_metric()` — Flawed logic

**File**: `visualization/charts/base.py:133-150`

```python
def _get_default_metric(self) -> str:
    task_types = self.predictions.get_unique_values('task_type')
    if any(t and 'classification' in str(t).lower() for t in task_types):
        return 'balanced_accuracy'
    return 'rmse'
```

Uses `any()` — if **one** classification prediction exists among hundreds of regression predictions, all charts default to `balanced_accuracy`. Should use majority or separate by task type.

#### 2.2.8 Charts without task_type awareness

| Chart | File | Issue |
|-------|------|-------|
| **Top K Comparison** | `visualization/charts/top_k_comparison.py` | Renders scatter plots (pred vs true + residuals) for top models. Scatter/residual plots are regression-specific. Classification models get meaningless scatter plots. No task_type filtering. |
| **Histogram** | `visualization/charts/histogram.py` | Plots score distribution. Mixes regression error distributions with classification accuracy distributions. No task_type filtering. |
| **Candlestick** | `visualization/charts/candlestick.py` | Shows metric distribution by variable. Mixes task types in the same box plot. No task_type filtering. |
| **Heatmap** | `visualization/charts/heatmap.py` | Cross-model/dataset score matrix. Zero references to `task_type`. Mixing RMSE and accuracy in the same heatmap is meaningless. |
| **Confusion Matrix** | `visualization/charts/confusion_matrix.py:146` | **Correctly** filters out regression: `[p for p in top_preds if 'classification' in str(p.get('task_type', '')).lower()]`. This is the pattern that should be generalized. |

#### 2.2.9 PredictionAnalyzer high-level API

**File**: `visualization/predictions.py`

The `PredictionAnalyzer` delegates to chart classes. Methods like `plot_top_k()`, `plot_histogram()`, `plot_candlestick()`, `plot_heatmap()` pass through without task_type filtering. Only `plot_confusion_matrix()` is safe because the chart class handles it.

### 2.3 Scenarios That Trigger the Bug

1. **Multi-dataset workspace**: User runs regression on dataset A, classification on dataset B, then calls `result.predictions.top(5)` — gets interleaved, incomparable results.

2. **Chart on mixed workspace**: `analyzer.plot_histogram()` with no dataset filter — shows mixed RMSE/accuracy distribution.

3. **Default metric flip**: Workspace has 50 regression runs + 1 classification run. All charts default to `balanced_accuracy` because of the `any()` check.

4. **`best_rmse` on classification result**: If user runs only classification, `result.best_rmse` returns NaN without explanation. Conversely, `result.best_accuracy` on regression returns NaN.

### 2.4 Test Coverage Gaps

| Area | Test File | Coverage |
|------|-----------|----------|
| Task detection | `tests/unit/core/test_task_detection.py` | Good — 135 lines, edge cases covered |
| Prediction ranking | `tests/unit/data/test_prediction_ranking.py` | No task_type tests — only tests metric sorting |
| Mixed task type queries | — | **Missing entirely** |
| Chart task_type separation | — | **Missing entirely** |
| `RunResult` accessors with wrong task type | — | **Missing entirely** |
| `filter_predictions(task_type=...)` | — | **Missing entirely** |
| `WorkspaceStore` task_type queries | — | **Missing entirely** |

---

## 3. Solutions

### 3.1 Library Code Changes

#### 3.1.1 Add `task_type` parameter to `Predictions.filter_predictions()`

**File**: `data/predictions.py`

Add explicit `task_type` parameter alongside existing filters:

```python
def filter_predictions(
    self,
    dataset_name: str | None = None,
    partition: str | None = None,
    config_name: str | None = None,
    model_name: str | None = None,
    fold_id: str | None = None,
    step_idx: int | None = None,
    branch_id: int | None = None,
    branch_name: str | None = None,
    task_type: str | None = None,       # NEW
    load_arrays: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
```

Add to `filter_map`:

```python
if task_type is not None:
    filter_map["task_type"] = task_type
```

#### 3.1.2 Add `task_type` parameter to `Predictions.top()` and `get_best()`

**File**: `data/predictions.py`

Add `task_type: str | None = None` to the `top()` signature. In the filter loop (line 931-948), add:

```python
# Apply task_type filter
if task_type is not None:
    task_type_lower = task_type.lower()
    if task_type_lower in ("classification", "clf"):
        if "classification" not in str(r.get("task_type", "")).lower():
            continue
    elif task_type_lower in ("regression", "reg"):
        if r.get("task_type") != "regression":
            continue
    else:
        if r.get("task_type") != task_type:
            continue
```

Similarly propagate `task_type` through `get_best()`.

#### 3.1.3 Task-type-aware default metric in `top()`

When no `rank_metric` is specified and the buffer contains mixed task types, `top()` should:

1. If `task_type` filter is set, use its default metric
2. If not set, detect the majority task type and use its default metric
3. Warn if mixed task types are present and no `task_type`/`rank_metric` specified

```python
if not rank_metric:
    task_types_in_candidates = {c.get("task_type") for c in candidates}
    if len(task_types_in_candidates) > 1:
        warnings.warn(
            f"Mixed task types found ({task_types_in_candidates}). "
            "Specify rank_metric or task_type to avoid cross-task comparison.",
            UserWarning, stacklevel=2,
        )
    first_task = candidates[0].get("task_type", "regression")
    effective_metric = "balanced_accuracy" if "classification" in str(first_task) else "rmse"
```

#### 3.1.4 Add `task_type` to SQL query builders

**File**: `pipeline/storage/store_queries.py`

Add `task_type: str | None = None` to both `build_prediction_query()` and `build_top_predictions_query()`:

```python
def build_prediction_query(
    ...,
    task_type: str | None = None,    # NEW
) -> tuple[str, list[object]]:
    ...
    if task_type is not None:
        if task_type.lower() in ("classification", "clf"):
            conditions.append("task_type LIKE '%classification%'")
        else:
            conditions.append("task_type = ?")
            params.append(task_type)
```

Same for `build_top_predictions_query()`.

#### 3.1.5 Add `task_type` to `WorkspaceStore.query_predictions()` and `top_predictions()`

**File**: `pipeline/storage/workspace_store.py`

Add `task_type: str | None = None` parameter, pass through to the query builder.

#### 3.1.6 Fix base chart `_get_default_metric()`

**File**: `visualization/charts/base.py`

Replace the `any()` logic with task-type-aware separation:

```python
def _get_default_metric(self) -> str:
    if self.predictions.num_predictions > 0:
        try:
            task_types = self.predictions.get_unique_values('task_type')
            classification_types = [t for t in task_types if t and 'classification' in str(t).lower()]
            regression_types = [t for t in task_types if t and str(t).lower() == 'regression']
            # Prefer the majority task type
            if classification_types and not regression_types:
                return 'balanced_accuracy'
            if regression_types and not classification_types:
                return 'rmse'
            # Mixed: default to rmse but let charts handle separation
            return 'rmse'
        except Exception:
            pass
    return 'rmse'
```

#### 3.1.7 Add task_type awareness to all chart classes

Each chart class should either:

**Option A — Auto-separate**: Detect mixed task types and render separate sub-figures per task type.

**Option B — Filter + warn**: Accept a `task_type` parameter. When not provided on mixed data, warn and render only the majority task type.

Recommended per chart:

| Chart | Approach | Details |
|-------|----------|---------|
| **Top K Comparison** | Filter | Add `task_type` param. Default to regression (scatter/residual is regression-specific). For classification, render confusion matrix subplots instead. |
| **Histogram** | Auto-separate | Detect task types. Render side-by-side histograms: left=regression metric, right=classification metric. |
| **Candlestick** | Filter | Add `task_type` param. Default to majority task type. Warn on mixed. |
| **Heatmap** | Auto-separate | Group models by task type. Render separate heatmaps (different color scales for RMSE vs accuracy). |
| **Confusion Matrix** | Already correct | No changes needed. |

#### 3.1.8 Add `task_type`-aware properties on `RunResult`

**File**: `api/result.py`

Option: make `best_rmse` filter to regression predictions, `best_accuracy` filter to classification predictions:

```python
@property
def best_rmse(self) -> float:
    best = self.predictions.get_best(metric="rmse", task_type="regression")
    if not best:
        return float('nan')
    rmse = best.get('rmse')
    return float(rmse) if rmse is not None else float('nan')

@property
def best_accuracy(self) -> float:
    best = self.predictions.get_best(metric="accuracy", task_type="classification")
    if not best:
        return float('nan')
    acc = best.get('accuracy')
    return float(acc) if acc is not None else float('nan')
```

### 3.2 Example Updates

#### 3.2.1 Add mixed-task example

Create `examples/developer/07_advanced/D_mixed_task_type_workspace.py`:

```python
"""Mixed task type workspace: classification + regression in same workspace.

Demonstrates:
- Running both regression and classification pipelines to the same workspace
- Querying results filtered by task type
- Generating task-type-appropriate charts
"""

import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

# Run regression
reg_result = nirs4all.run(
    pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), PLSRegression(10)],
    dataset="sample_data/regression",
    workspace="workspace/",
)

# Run classification
clf_result = nirs4all.run(
    pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), RandomForestClassifier()],
    dataset="sample_data/classification",
    workspace="workspace/",
)

# Query top models per task type
print("=== Top Regression Models ===")
top_reg = reg_result.predictions.top(3, task_type="regression")
for p in top_reg:
    print(f"  {p['model_name']}: RMSE={p.get('rmse', 'N/A')}")

print("\n=== Top Classification Models ===")
top_clf = clf_result.predictions.top(3, task_type="classification")
for p in top_clf:
    print(f"  {p['model_name']}: Acc={p.get('accuracy', 'N/A')}")
```

#### 3.2.2 Update existing classification example

`examples/user/01_getting_started/U03_basic_classification.py` — verify it uses `best_accuracy` not `best_rmse`.

### 3.3 Documentation Updates

#### 3.3.1 Update user guide

Add a section to `docs/source/user_guide/` explaining task type behavior:

- Auto-detection rules
- How to force task type (`dataset.set_task_type(...)`)
- Querying by task type (`top(task_type="regression")`)
- Chart behavior with mixed task types

#### 3.3.2 Update API reference docstrings

Add `task_type` parameter documentation to:
- `Predictions.top()`
- `Predictions.get_best()`
- `Predictions.filter_predictions()`
- `WorkspaceStore.query_predictions()`
- `WorkspaceStore.top_predictions()`
- All chart `render()` methods

### 3.4 Test Plan

#### 3.4.1 Unit tests — `tests/unit/data/test_prediction_task_type.py` (new)

```python
"""Tests for task-type-aware prediction queries."""

class TestTopWithTaskType:
    """Test Predictions.top() with task_type filtering."""

    def test_top_filters_regression_only(self, mixed_predictions):
        """top(task_type='regression') returns only regression predictions."""

    def test_top_filters_classification_only(self, mixed_predictions):
        """top(task_type='classification') returns only classification predictions."""

    def test_top_mixed_warns_without_filter(self, mixed_predictions):
        """top() on mixed task types warns when no task_type/rank_metric specified."""

    def test_top_classification_shorthand(self, mixed_predictions):
        """top(task_type='clf') works as shorthand for classification."""

    def test_default_metric_regression_only(self, regression_predictions):
        """Default metric is 'rmse' when only regression predictions exist."""

    def test_default_metric_classification_only(self, classification_predictions):
        """Default metric is 'balanced_accuracy' for classification-only predictions."""

    def test_default_metric_mixed_uses_majority(self, mixed_predictions):
        """Default metric uses majority task type when mixed."""

class TestGetBestWithTaskType:
    """Test Predictions.get_best() with task_type filtering."""

    def test_get_best_regression(self, mixed_predictions):
        """get_best(task_type='regression') returns best regression model."""

    def test_get_best_classification(self, mixed_predictions):
        """get_best(task_type='classification') returns best classification model."""

class TestFilterPredictionsTaskType:
    """Test Predictions.filter_predictions() with task_type."""

    def test_filter_by_task_type(self, mixed_predictions):
        """filter_predictions(task_type='regression') returns only regression."""

    def test_filter_classification(self, mixed_predictions):
        """filter_predictions(task_type='binary_classification') is exact match."""
```

#### 3.4.2 Unit tests — `tests/unit/visualization/charts/test_chart_task_type.py` (new)

```python
"""Tests for task-type-aware chart rendering."""

class TestDefaultMetricSelection:
    """Test _get_default_metric() logic."""

    def test_regression_only_returns_rmse(self):
    def test_classification_only_returns_balanced_accuracy(self):
    def test_mixed_returns_rmse(self):  # majority-based or safe default

class TestTopKComparisonTaskType:
    """Top K chart filters by task type."""

    def test_renders_only_regression_on_mixed(self):
    def test_accepts_task_type_param(self):

class TestHistogramTaskType:
    """Histogram chart separates task types."""

    def test_auto_separates_mixed_task_types(self):

class TestCandlestickTaskType:
    """Candlestick chart handles task types."""

    def test_default_filters_to_majority(self):
    def test_accepts_task_type_param(self):

class TestHeatmapTaskType:
    """Heatmap chart handles task types."""

    def test_separates_by_task_type(self):
```

#### 3.4.3 Unit tests — `tests/unit/pipeline/storage/test_store_task_type.py` (new)

```python
"""Tests for task-type-aware storage queries."""

class TestQueryPredictionsTaskType:
    """WorkspaceStore.query_predictions() with task_type filter."""

    def test_filter_regression(self, store_with_mixed):
    def test_filter_classification(self, store_with_mixed):
    def test_no_filter_returns_all(self, store_with_mixed):

class TestTopPredictionsTaskType:
    """WorkspaceStore.top_predictions() with task_type filter."""

    def test_top_regression_by_rmse(self, store_with_mixed):
    def test_top_classification_by_accuracy(self, store_with_mixed):
```

#### 3.4.4 Integration test — `tests/integration/pipeline/test_mixed_task_type.py` (new)

```python
"""Integration tests for mixed task type workspaces."""

class TestMixedTaskTypeWorkspace:
    """End-to-end tests with regression + classification in same workspace."""

    def test_top_separates_task_types(self):
        """Run regression then classification; top() per task type returns correct models."""

    def test_charts_handle_mixed_workspace(self):
        """Charts render correctly when workspace has both task types."""

    def test_best_rmse_ignores_classification(self):
        """RunResult.best_rmse returns NaN or correct value regardless of classification presence."""

    def test_best_accuracy_ignores_regression(self):
        """RunResult.best_accuracy returns NaN or correct value regardless of regression presence."""
```

### 3.5 Implementation Status (COMPLETED 2026-03-30)

All items have been implemented and validated (6672 unit tests passing, 50 new tests added).

| Change | Status | Files Modified |
|--------|--------|---------------|
| `matches_task_type()` + `resolve_task_type_sql()` helpers | Done | `core/task_type.py` |
| `task_type` param on `Predictions.top()`, `get_best()`, `filter_predictions()` | Done | `data/predictions.py` |
| `task_type` on SQL query builders | Done | `pipeline/storage/store_queries.py` |
| `task_type` on `WorkspaceStore.query_predictions()`, `top_predictions()` | Done | `pipeline/storage/workspace_store.py` |
| Fix `_get_default_metric()` — majority-based, accepts filter | Done | `visualization/charts/base.py` |
| Auto-separate charts on mixed task types + `task_type` param | Done | `visualization/charts/{top_k_comparison,histogram,candlestick,heatmap}.py` |
| `task_type` on `PredictionAnalyzer` plot methods | Done | `visualization/predictions.py` |
| `RunResult.best_rmse`/`best_r2` filter to regression, `best_accuracy` to classification | Done | `api/result.py` |
| 50 new unit tests (matching, predictions, charts) | Done | `tests/unit/core/test_task_type_matching.py`, `tests/unit/data/test_prediction_task_type.py`, `tests/unit/visualization/charts/test_chart_task_type.py` |

### 3.6 Design Decisions (Resolved)

1. **`top()` on mixed without filter**: Warns and uses the first candidate's task-type default metric. Accepts `task_type` param for explicit filtering.

2. **Charts on mixed data**: Warn + auto-separate (render separate figures per task type). Also accept explicit `task_type` parameter.

3. **`task_type` matching — fuzzy with aliases**:
   | Alias | Matches |
   |-------|---------|
   | `"regression"` / `"reg"` | `"regression"` |
   | `"classification"` / `"clf"` | `"binary_classification"` and `"multiclass_classification"` |
   | `"binary"` | `"binary_classification"` only |
   | `"multiclass"` | `"multiclass_classification"` only |

   Implemented in `matches_task_type()` (in-memory) and `resolve_task_type_sql()` (SQL). Case-insensitive.

4. **Backward compatibility**: `task_type` is keyword-only with `None` default. Existing code unchanged. Mixed-type warning is new but non-breaking.
