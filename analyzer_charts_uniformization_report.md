# Analyzer Charts Signature Uniformization Report

## Executive Summary

Analysis of 5 analyzer chart types reveals **significant inconsistencies** in parameter naming, default values, and title/label formatting. This report documents what needs to be standardized.

---

## Current Signature Analysis

### 1. **TopKComparisonChart** (`plot_top_k`)

**Current Signature:**
```python
def render(self, k: int = 5, rank_metric: str = 'rmse',
           rank_partition: str = 'val', display_partition: str = 'all',
           dataset_name: Optional[str] = None,
           figsize: Optional[tuple] = None, **filters) -> Figure:
```

**Issues:**
- ✅ Uses `rank_metric` and `rank_partition` (GOOD)
- ✅ Uses `display_partition` (GOOD)
- ❌ No `display_metric` parameter (inconsistent with heatmap)
- ❌ Default metric is `'rmse'` (regression-focused)

**Title/Labels:**
- Fig Title: `'Top {k} Models - Best {rank_metric.upper()} ({rank_partition})'`
- Chart Title: `'{model_name}\n{rank_metric.upper()}={rank_score_str} ({rank_partition})'`
- ⚠️ Shows ranking info but not display info clearly

---

### 2. **ConfusionMatrixChart** (`plot_confusion_matrix`)

**Current Signature:**
```python
def render(self, k: int = 5, metric: str = 'accuracy',
           rank_partition: str = 'val', display_partition: str = 'test',
           dataset_name: Optional[str] = None,
           figsize: Optional[tuple] = None, **filters) -> Union[Figure, List[Figure]]:
```

**Issues:**
- ❌ Uses `metric` instead of `rank_metric` (INCONSISTENT)
- ✅ Uses `rank_partition` and `display_partition` (GOOD)
- ❌ No `display_metric` (not needed for confusion matrix but inconsistent API)
- ✅ Default metric is `'accuracy'` (classification-focused, GOOD)

**Title/Labels:**
- Fig Title: `'Dataset: {ds} - Top {k} Models\nConfusion Matrices (ranked by {metric.upper()} on {rank_partition})'`
- Chart Title: `'{model_name}\n{metric.upper()}={score_str} ({display_partition})'`
- ⚠️ Uses display_partition for score display (good) but doesn't clarify ranking vs display

---

### 3. **ScoreHistogramChart** (`plot_histogram`)

**Current Signature:**
```python
def render(self, metric: str = 'rmse', dataset_name: Optional[str] = None,
           partition: Optional[str] = None, bins: int = 20,
           figsize: Optional[tuple] = None, **filters) -> Figure:
```

**Issues:**
- ❌ Uses `metric` instead of `rank_metric` or `display_metric` (VERY INCONSISTENT)
- ❌ Uses `partition` instead of `display_partition` (INCONSISTENT)
- ❌ NO ranking parameters (no rank_metric, no rank_partition)
- ❌ Default metric is `'rmse'` (regression-focused)
- ⚠️ This chart doesn't rank, so should only use `display_metric` and `display_partition`

**Title/Labels:**
- Title: `'Distribution of {metric.upper()} Scores\n({len(scores)} predictions, partition: {partition_label})'`
- ❌ Doesn't mention display vs rank (makes sense since no ranking)

---

### 4. **CandlestickChart** (`plot_candlestick`)

**Current Signature:**
```python
def render(self, variable: str, metric: str = 'rmse',
           dataset_name: Optional[str] = None, partition: Optional[str] = None,
           figsize: Optional[tuple] = None, **filters) -> Figure:
```

**Issues:**
- ❌ Uses `metric` instead of `rank_metric` or `display_metric` (VERY INCONSISTENT)
- ❌ Uses `partition` instead of `display_partition` (INCONSISTENT)
- ❌ NO ranking parameters
- ❌ Default metric is `'rmse'` (regression-focused)
- ⚠️ This chart doesn't rank, so should only use `display_metric` and `display_partition`

**Title/Labels:**
- Title: `'{metric.upper()} Distribution by {variable.replace("_", " ").title()}\n(partition: {partition})'`
- ❌ Doesn't clarify display vs rank

---

### 5. **HeatmapChart** (`plot_heatmap`)

**Current Signature:**
```python
def render(self, x_var: str, y_var: str,
           rank_metric: str = 'rmse', rank_partition: str = 'val',
           display_metric: str = '', display_partition: str = 'test',
           figsize: Optional[tuple] = None, normalize: bool = False,
           rank_agg: str = 'best', display_agg: str = 'mean',
           show_counts: bool = True, **filters) -> Figure:
```

**Issues:**
- ✅ Uses `rank_metric`, `rank_partition`, `display_metric`, `display_partition` (PERFECT!)
- ❌ Default rank_metric is `'rmse'` (regression-focused)
- ⚠️ `display_metric` defaults to `''` (empty string, then set to rank_metric)

**Title/Labels:**
- Title: `'{display_agg.title()} {display_metric.upper()} in {display_partition} (rank on {rank_agg} {rank_metric.upper()} in {rank_partition})'`
- ✅ **EXCELLENT** - Shows both ranking and display info clearly

---

## Summary Table of Issues

| Chart | rank_metric | rank_partition | display_metric | display_partition | Default Metric |
|-------|-------------|----------------|----------------|-------------------|----------------|
| **TopKComparison** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes | ❌ rmse |
| **ConfusionMatrix** | ❌ `metric` | ✅ Yes | ❌ No | ✅ Yes | ✅ accuracy |
| **Histogram** | ❌ `metric` | ❌ `partition` | ❌ No | ❌ No | ❌ rmse |
| **Candlestick** | ❌ `metric` | ❌ `partition` | ❌ No | ❌ No | ❌ rmse |
| **Heatmap** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ rmse |

---

## Required Changes

### A. Signature Uniformization

#### **Charts WITH Ranking** (TopKComparison, ConfusionMatrix, Heatmap)

**Standard signature should be:**
```python
def render(self,
           rank_metric: str = 'balanced_accuracy',  # or 'rmse'
           rank_partition: str = 'val',
           display_metric: str = '',  # defaults to rank_metric
           display_partition: str = 'test',
           **specific_params,
           **filters) -> Figure:
```

#### **Charts WITHOUT Ranking** (Histogram, Candlestick)

**Standard signature should be:**
```python
def render(self,
           display_metric: str = 'balanced_accuracy',  # or 'rmse'
           display_partition: str = 'test',
           **specific_params,
           **filters) -> Figure:
```

### B. Default Metric Logic

**Solution:**
- Use existing `task_type` from predictions (already computed and stored)
- **DON'T recompute** task_type - it's available in prediction records via `predictions._storage.df['task_type']`
- **Classification default:** `'balanced_accuracy'`
- **Regression default:** `'rmse'`

**Implementation approach:**
```python
def _get_default_metric(self) -> str:
    """Get default metric based on task type from predictions.

    Uses task_type already stored in predictions - does NOT recompute.
    """
    # Get task_type from first prediction record
    if self.predictions.num_predictions > 0:
        task_type = self.predictions._storage.df['task_type'][0]
        if task_type and 'classification' in task_type.lower():
            return 'balanced_accuracy'
    return 'rmse'
```

### C. Multi-Metric Display in Chart Titles

**Requirement:**
Allow displaying **multiple metrics and partitions** in individual chart titles while keeping the API call simple.

**Use Case Example:**
```python
# Simple call
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val')

# Each subplot should show:
# - Ranking info: "Ranked by RMSE (val)"
# - Display info for MULTIPLE metrics/partitions:
#   "train: RMSE=0.12, R²=0.88, MAE=0.10"
#   "val:   RMSE=0.15, R²=0.85, MAE=0.12"
#   "test:  RMSE=0.18, R²=0.82, MAE=0.14"
```

**Proposed Solution: `show_scores` Parameter**

Add a flexible `show_scores` parameter that controls what scores to display in chart titles:

```python
def render(self,
           k: int = 5,
           rank_metric: str = None,
           rank_partition: str = 'val',
           display_metric: str = '',
           display_partition: str = 'test',
           show_scores: Union[bool, str, List[str], Dict] = True,
           **kwargs) -> Figure:
    """
    Args:
        show_scores: Controls which scores to show in chart titles:
            - True (default): Show display_metric for display_partition only
            - False: Show no scores in titles (minimal)
            - 'all': Show display_metric for all partitions (train/val/test)
            - 'rank_only': Show only rank_metric on rank_partition
            - List[str]: Show specific metrics, e.g., ['rmse', 'r2', 'mae']
            - Dict: Full control, e.g., {
                'partitions': ['train', 'val', 'test'],
                'metrics': ['rmse', 'r2', 'mae']
              }
    """
```

**Examples:**

```python
# Default: show only display_metric on display_partition
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          display_metric='r2', display_partition='test')
# Chart title shows: "Ranked: RMSE=0.15 (val)\nDisplayed: R²=0.82 (test)"

# Show display_metric for all partitions
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          display_metric='r2', show_scores='all')
# Chart title shows:
# "Ranked: RMSE=0.15 (val)
#  train: R²=0.90 | val: R²=0.85 | test: R²=0.82"

# Show multiple metrics on test partition
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          show_scores=['rmse', 'r2', 'mae'])
# Chart title shows:
# "Ranked: RMSE=0.15 (val)
#  test: RMSE=0.18, R²=0.82, MAE=0.14"

# Full control: multiple metrics across multiple partitions
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          show_scores={
                              'partitions': ['train', 'val', 'test'],
                              'metrics': ['rmse', 'r2', 'mae']
                          })
# Chart title shows:
# "Ranked: RMSE=0.15 (val)
#  train: RMSE=0.12, R²=0.88, MAE=0.10
#  val:   RMSE=0.15, R²=0.85, MAE=0.12
#  test:  RMSE=0.18, R²=0.82, MAE=0.14"

# Minimal: only ranking info
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          show_scores='rank_only')
# Chart title shows: "Ranked: RMSE=0.15 (val)"

# No scores in titles
fig = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='val',
                          show_scores=False)
# Chart title shows: "RandomForest" only
```

**Implementation Helper:**

```python
class BaseChart(ABC):
    """Base chart class."""

    def _format_score_display(
        self,
        pred: Dict[str, Any],
        show_scores: Union[bool, str, List[str], Dict],
        rank_metric: str,
        rank_partition: str,
        display_metric: str = None,
        display_partition: str = None
    ) -> str:
        """Format scores for chart title based on show_scores parameter.

        Args:
            pred: Prediction dictionary with 'partitions' data
            show_scores: Control parameter for score display
            rank_metric: Metric used for ranking
            rank_partition: Partition used for ranking
            display_metric: Primary display metric
            display_partition: Primary display partition

        Returns:
            Formatted string for chart title
        """
        lines = []

        # Always show ranking info
        rank_score = pred.get(f'{rank_partition}_score')
        if rank_score is not None:
            lines.append(f"Ranked: {rank_metric.upper()}={rank_score:.4f} ({rank_partition})")

        # Handle show_scores parameter
        if show_scores is False:
            return ""  # No scores

        if show_scores == 'rank_only':
            return lines[0]  # Only ranking

        # Determine which metrics and partitions to show
        if show_scores is True:
            # Default: display_metric on display_partition
            metrics = [display_metric or rank_metric]
            partitions = [display_partition or 'test']
        elif show_scores == 'all':
            # Display_metric on all partitions
            metrics = [display_metric or rank_metric]
            partitions = ['train', 'val', 'test']
        elif isinstance(show_scores, list):
            # Multiple metrics on display_partition
            metrics = show_scores
            partitions = [display_partition or 'test']
        elif isinstance(show_scores, dict):
            # Full control
            metrics = show_scores.get('metrics', [display_metric or rank_metric])
            partitions = show_scores.get('partitions', [display_partition or 'test'])
        else:
            metrics = [display_metric or rank_metric]
            partitions = [display_partition or 'test']

        # Extract scores from prediction
        partitions_data = pred.get('partitions', {})

        for partition in partitions:
            partition_data = partitions_data.get(partition, {})
            if not partition_data:
                continue

            scores_str = []
            for metric in metrics:
                # Try to get pre-computed metric
                score = partition_data.get(metric)

                # If not found, compute from y_true and y_pred
                if score is None:
                    y_true = partition_data.get('y_true')
                    y_pred = partition_data.get('y_pred')
                    if y_true is not None and y_pred is not None:
                        try:
                            from nirs4all.core import metrics as evaluator
                            score = evaluator.eval(y_true, y_pred, metric)
                        except Exception:
                            continue

                if score is not None:
                    scores_str.append(f"{metric.upper()}={score:.4f}")

            if scores_str:
                if len(partitions) == 1:
                    # Single partition: inline format
                    lines.append(f"Displayed: {', '.join(scores_str)} ({partition})")
                else:
                    # Multiple partitions: one per line
                    lines.append(f"{partition}: {', '.join(scores_str)}")

        return '\n'.join(lines)
```

**Benefits:**
- ✅ **Simple by default**: `show_scores=True` shows only primary display_metric
- ✅ **Flexible**: Can show multiple metrics/partitions when needed
- ✅ **Backward compatible**: Default behavior unchanged
- ✅ **Clean API**: Single parameter controls all score display
- ✅ **No computation overhead**: Only computes metrics that are requested

### C. Title/Label Uniformization (Updated with Multi-Metric Support)

#### **Fig Title Pattern (for charts WITH ranking):**
```
"Top {k} {Chart_Type} - Ranked by {rank_agg} {rank_metric.upper()} ({rank_partition})"
```

Examples:
- `"Top 5 Models - Ranked by best RMSE (val)"`
- `"Top 3 Confusion Matrices - Ranked by best BALANCED_ACCURACY (val)"`

#### **Fig Title Pattern (for charts WITHOUT ranking):**
```
"{Chart_Type} - {display_metric.upper()} ({display_partition})"
```

Examples:
- `"Score Histogram - RMSE (test)"`
- `"Candlestick Distribution - BALANCED_ACCURACY (test)"`

#### **Chart/Subplot Title Pattern (flexible with show_scores):**

**Default behavior (show_scores=True):**
```
"{Item_Name}
Ranked: {rank_metric.upper()}={rank_score:.4f} ({rank_partition})
Displayed: {display_metric.upper()}={display_score:.4f} ({display_partition})"
```

**With show_scores='all':**
```
"{Item_Name}
Ranked: {rank_metric.upper()}={rank_score:.4f} ({rank_partition})
train: {display_metric.upper()}={score1:.4f} | val: {display_metric.upper()}={score2:.4f} | test: {display_metric.upper()}={score3:.4f}"
```

**With show_scores=['rmse', 'r2', 'mae']:**
```
"{Item_Name}
Ranked: RMSE={rank_score:.4f} (val)
test: RMSE={s1:.4f}, R²={s2:.4f}, MAE={s3:.4f}"
```

**With show_scores={'partitions': ['val', 'test'], 'metrics': ['rmse', 'r2']}:**
```
"{Item_Name}
Ranked: RMSE={rank_score:.4f} (val)
val:  RMSE={s1:.4f}, R²={s2:.4f}
test: RMSE={s3:.4f}, R²={s4:.4f}"
```

**For charts without ranking (show_scores=True or list):**
```
"{Item_Name}
{display_metric.upper()}={display_score:.4f} ({display_partition})"
```

**Or with multiple metrics:**
```
"{Item_Name}
{display_partition}: {metric1.UPPER()}={s1:.4f}, {metric2.UPPER()}={s2:.4f}, ..."
```

---

## Detailed Changes Required

### 1. **TopKComparisonChart**

**Signature Changes:**
```python
# BEFORE
def render(self, k: int = 5, rank_metric: str = 'rmse',
           rank_partition: str = 'val', display_partition: str = 'all', ...)

# AFTER
def render(self, k: int = 5,
           rank_metric: str = None,  # Auto-detect: balanced_accuracy or rmse
           rank_partition: str = 'val',
           display_metric: str = '',  # Not used for scatter plots but for consistency
           display_partition: str = 'all', ...)
```

**Title Changes:**
```python
# Fig Title - BEFORE
f'Top {k} Models - Best {rank_metric.upper()} ({rank_partition})'

# Fig Title - AFTER
f'Top {k} Models Comparison - Ranked by best {rank_metric.upper()} ({rank_partition})'

# Chart Title - BEFORE
f'{model_name}\n{rank_metric.upper()}={rank_score_str} ({rank_partition})'

# Chart Title - AFTER (with default show_scores=True)
title = self._format_score_display(
    pred, show_scores, rank_metric, rank_partition,
    display_metric, display_partition
)
f'{model_name}\n{title}'

# Example output (default):
# "RandomForest
#  Ranked: RMSE=0.1234 (val)
#  Displayed: R2=0.8765 (test)"

# Example output (show_scores={'partitions': ['train', 'val', 'test'], 'metrics': ['rmse', 'r2']}):
# "RandomForest
#  Ranked: RMSE=0.1234 (val)
#  train: RMSE=0.1100, R²=0.9000
#  val:   RMSE=0.1234, R²=0.8800
#  test:  RMSE=0.1400, R²=0.8500"
```

**New Parameter:**
```python
def render(self, k: int = 5,
           rank_metric: str = None,
           rank_partition: str = 'val',
           display_metric: str = '',
           display_partition: str = 'all',
           show_scores: Union[bool, str, List[str], Dict] = True,  # NEW
           **kwargs) -> Figure:
```

---

### 2. **ConfusionMatrixChart**

**Signature Changes:**
```python
# BEFORE
def render(self, k: int = 5, metric: str = 'accuracy',
           rank_partition: str = 'val', display_partition: str = 'test', ...)

# AFTER
def render(self, k: int = 5,
           rank_metric: str = None,  # Auto-detect: balanced_accuracy or rmse
           rank_partition: str = 'val',
           display_metric: str = '',  # Same as rank_metric for confusion matrix
           display_partition: str = 'test', ...)
```

**Title Changes:**
```python
# Fig Title - BEFORE
f'Dataset: {ds} - Top {k} Models\nConfusion Matrices (ranked by {metric.UPPER()} on {rank_partition})'

# Fig Title - AFTER
f'Dataset: {ds} - Top {k} Confusion Matrices\nRanked by best {rank_metric.upper()} ({rank_partition}), Displayed: {display_partition}'

# Chart Title - BEFORE
f'{model_name}\n{metric.upper()}={score_str} ({display_partition})'

# Chart Title - AFTER (with show_scores support)
title = self._format_score_display(
    pred, show_scores, rank_metric, rank_partition,
    display_metric, display_partition
)
f'{model_name}\n{title}'

# Example output (default show_scores=True):
# "RandomForest
#  Ranked: BALANCED_ACCURACY=0.8500 (val)
#  Displayed: BALANCED_ACCURACY=0.8200 (test)"
```

**New Parameter:**
```python
def render(self, k: int = 5,
           rank_metric: str = None,
           rank_partition: str = 'val',
           display_metric: str = '',
           display_partition: str = 'test',
           show_scores: Union[bool, str, List[str], Dict] = True,  # NEW
           **kwargs) -> Union[Figure, List[Figure]]:
```

---

### 3. **ScoreHistogramChart**

**Signature Changes:**
```python
# BEFORE
def render(self, metric: str = 'rmse', dataset_name: Optional[str] = None,
           partition: Optional[str] = None, bins: int = 20, ...)

# AFTER
def render(self,
           display_metric: str = None,  # Auto-detect: balanced_accuracy or rmse
           display_partition: str = 'test',
           dataset_name: Optional[str] = None,
           bins: int = 20, ...)
```

**Title Changes:**
```python
# BEFORE
f'Distribution of {metric.upper()} Scores\n({len(scores)} predictions, partition: {partition_label})'

# AFTER
f'Score Histogram - {display_metric.upper()} ({display_partition})\n{len(scores)} predictions'
```

---

### 4. **CandlestickChart**

**Signature Changes:**
```python
# BEFORE
def render(self, variable: str, metric: str = 'rmse',
           dataset_name: Optional[str] = None, partition: Optional[str] = None, ...)

# AFTER
def render(self, variable: str,
           display_metric: str = None,  # Auto-detect: balanced_accuracy or rmse
           display_partition: str = 'test',
           dataset_name: Optional[str] = None, ...)
```

**Title Changes:**
```python
# BEFORE
f'{metric.upper()} Distribution by {variable.replace("_", " ").title()}\n(partition: {partition})'

# AFTER
f'Candlestick - {display_metric.upper()} by {variable.replace("_", " ").title()}\nDisplayed: {display_partition}'
```

---

### 5. **HeatmapChart**

**Signature Changes:**
```python
# BEFORE
def render(self, x_var: str, y_var: str,
           rank_metric: str = 'rmse', rank_partition: str = 'val',
           display_metric: str = '', display_partition: str = 'test', ...)

# AFTER
def render(self, x_var: str, y_var: str,
           rank_metric: str = None,  # Auto-detect: balanced_accuracy or rmse
           rank_partition: str = 'val',
           display_metric: str = '',  # defaults to rank_metric
           display_partition: str = 'test', ...)
```

**Title Changes:**
```python
# BEFORE (already good!)
f'{display_agg.title()} {display_metric.upper()} in {display_partition} (rank on {rank_agg} {rank_metric.upper()} in {rank_partition})'

# AFTER (slight improvement)
f'Heatmap - {display_agg.title()} {display_metric.upper()} ({display_partition})\nRanked by {rank_agg} {rank_metric.UPPER()} ({rank_partition})'
```

---

## Implementation Priority

### Phase 1: Critical (Signature Uniformity)
1. Rename `metric` → `rank_metric` in ConfusionMatrixChart
2. Rename `metric` → `display_metric` and `partition` → `display_partition` in Histogram
3. Rename `metric` → `display_metric` and `partition` → `display_partition` in Candlestick
4. Add `display_metric` parameter to TopKComparison (even if unused)

### Phase 2: High (Default Metrics & Multi-Metric Display)
5. Implement task type detection from existing predictions data (use `predictions._storage.df['task_type']`)
6. Set metric defaults to `None` and resolve to `balanced_accuracy` or `rmse` in each chart's render method
7. Add `_get_default_metric()` helper method to BaseChart
8. **Add `show_scores` parameter to all charts with ranking** (TopK, ConfusionMatrix, Heatmap)
9. Implement `_format_score_display()` helper in BaseChart

### Phase 3: Medium (Titles & Labels)
10. Uniformize fig titles across all charts
11. Uniformize chart/subplot titles using `_format_score_display()` helper
12. Update labels to clearly separate ranking from display
13. Test various `show_scores` configurations

### Phase 4: Low (Documentation & Backward Compatibility)
14. Update docstrings to reflect new signatures and show_scores parameter
15. Update examples to use new parameter names
16. Add deprecation warnings for old parameter names (metric → rank_metric/display_metric)
17. Add examples showing show_scores usage in docs

---

## Backward Compatibility Strategy

**Option A: Deprecation Warnings (Recommended)**
```python
def render(self, rank_metric: str = None, metric: str = None, ...):
    if metric is not None:
        warnings.warn("'metric' is deprecated, use 'rank_metric' instead", DeprecationWarning)
        if rank_metric is None:
            rank_metric = metric
```

**Option B: Aggressive Removal**
- Remove old parameters entirely
- Update all examples and tests
- Add migration guide to CHANGELOG

---

## Testing Checklist

- [ ] Test TopKComparison with classification (balanced_accuracy default)
- [ ] Test TopKComparison with regression (rmse default)
- [ ] Test TopKComparison with show_scores='all'
- [ ] Test TopKComparison with show_scores=['rmse', 'r2', 'mae']
- [ ] Test TopKComparison with show_scores={'partitions': [...], 'metrics': [...]}
- [ ] Test ConfusionMatrix with new rank_metric parameter
- [ ] Test ConfusionMatrix with show_scores parameter
- [ ] Test Histogram with display_metric and display_partition
- [ ] Test Candlestick with display_metric and display_partition
- [ ] Test Heatmap with auto-detected defaults
- [ ] Test Heatmap with show_scores in cell annotations
- [ ] Verify all fig titles follow new pattern
- [ ] Verify all chart titles show rank vs display info correctly
- [ ] Verify multi-metric display works across all partitions
- [ ] Test backward compatibility with old parameter names
- [ ] Test that task_type is read from predictions, not recomputed
- [ ] Update integration tests (examples Q1-Q17)

---

## Conclusion

**Main Issues:**
1. **Inconsistent parameter naming** (metric vs rank_metric vs display_metric)
2. **Regression-biased defaults** (rmse everywhere instead of task-aware)
3. **Unclear titles** (don't distinguish ranking from display)
4. **Limited flexibility** (can't show multiple metrics/partitions in chart titles)

**Solutions:**
1. Standardize on `rank_metric`, `rank_partition`, `display_metric`, `display_partition`
2. Auto-detect task type from **existing** `predictions._storage.df['task_type']` (don't recompute)
3. Use `balanced_accuracy` (classification) or `rmse` (regression) as smart defaults
4. Uniformize titles to clearly show ranking vs display information
5. **Add `show_scores` parameter** for flexible multi-metric/multi-partition display in chart titles
6. Keep API simple: defaults work for 90% of use cases, advanced users can use `show_scores`

**Key Features of show_scores:**
- ✅ Simple default: `show_scores=True` shows only primary metric
- ✅ Show all partitions: `show_scores='all'`
- ✅ Show multiple metrics: `show_scores=['rmse', 'r2', 'mae']`
- ✅ Full control: `show_scores={'partitions': ['val', 'test'], 'metrics': ['rmse', 'r2']}`
- ✅ Minimal: `show_scores='rank_only'` or `show_scores=False`

**Estimated Effort:**
- Code changes: ~6-8 hours (added show_scores feature)
- Testing: ~3-4 hours (added multi-metric tests)
- Documentation: ~2-3 hours (show_scores examples)
- **Total: ~11-15 hours**
