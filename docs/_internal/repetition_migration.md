# Repetition API Migration Guide

This guide documents the migration from the old `aggregate`/`force_group` API to the new unified `repetition` API introduced in nirs4all 0.7.

## Overview

The new `repetition` parameter provides a unified way to handle sample repetitions (multiple spectral measurements per physical sample) across the data and pipeline layers.

**Important**: The visualization layer (PredictionAnalyzer) still uses `aggregate` parameter names. Only `DatasetConfigs` and `Predictions.top()` were migrated to the new naming.

## Quick Reference

| Old Syntax | New Syntax | Notes |
|------------|------------|-------|
| `DatasetConfigs(path, aggregate="Sample_ID")` | `DatasetConfigs(path, repetition="Sample_ID")` | Data layer |
| `{"split": KFold(5), "force_group": "Sample_ID"}` | `DatasetConfigs(path, repetition="Sample_ID")` + `KFold(5)` | Pipeline layer |
| `predictions.top(5, aggregate="X")` | `predictions.top(5, by_repetition="X")` | Data layer |
| `predictions.top(5, aggregate_method="mean")` | `predictions.top(5, repetition_method="mean")` | Data layer |
| `predictions.top(5, aggregate_exclude_outliers=True)` | `predictions.top(5, repetition_exclude_outliers=True)` | Data layer |
| YAML: `aggregate: sample_id` | YAML: `repetition: sample_id` | Config files |
| YAML: `aggregate_method: mean` | YAML: `repetition_method: mean` | Config files |
| YAML: `aggregate_exclude_outliers: true` | YAML: `repetition_exclude_outliers: true` | Config files |
| `PredictionAnalyzer(..., default_aggregate=...)` | *unchanged* | Visualization layer |
| `analyzer.plot_*(aggregate='...')` | *unchanged* | Visualization layer |

## Detailed Migration Patterns

### 1. DatasetConfigs: aggregate → repetition

**Before:**
```python
from nirs4all.data import DatasetConfigs

dataset_config = DatasetConfigs(
    "sample_data/regression",
    aggregate="Sample_ID"  # ← Deprecated
)
```

**After:**
```python
from nirs4all.data import DatasetConfigs

dataset_config = DatasetConfigs(
    "sample_data/regression",
    repetition="Sample_ID"  # ← New
)
```

### 2. Pipeline Steps: force_group → repetition in DatasetConfigs

The `force_group` parameter is no longer needed. Instead, set `repetition` once in DatasetConfigs and all splitters automatically respect it.

**Before:**
```python
# Had to specify force_group at each split step
pipeline = [
    {"split": KFold(n_splits=3), "force_group": "Sample_ID"},  # ← Deprecated
    PLSRegression(10)
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression"
)
```

**After:**
```python
# Set repetition once in DatasetConfigs
pipeline = [
    KFold(n_splits=3),  # Automatically groups by Sample_ID!
    PLSRegression(10)
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset=nirs4all.DatasetConfigs("sample_data/regression", repetition="Sample_ID")
)
```

### 3. Additional Grouping with group_by

For additional grouping columns beyond repetition, use `group_by`:

**Before:**
```python
# Not possible with force_group
```

**After:**
```python
# Combine repetition with additional grouping
pipeline = [
    {"split": KFold(5), "group_by": ["Year", "Location"]},
    PLSRegression(10)
]

# Final groups = (Sample_ID, Year, Location) tuples
result = nirs4all.run(
    pipeline=pipeline,
    dataset=nirs4all.DatasetConfigs(path, repetition="Sample_ID")
)
```

### 4. Opt-out of Repetition Grouping

Use `ignore_repetition=True` to disable automatic repetition grouping for specific experiments:

```python
pipeline = [
    {"split": KFold(5), "ignore_repetition": True},  # Ignores repetition
    PLSRegression(10)
]
```

### 5. Predictions.top(): aggregate → by_repetition

**Before:**
```python
# Get top models with aggregated metrics
top_models = predictions.top(
    5,
    rank_metric='rmse',
    aggregate='Sample_ID',           # ← Deprecated
    aggregate_method='mean',         # ← Deprecated
    aggregate_exclude_outliers=True  # ← Deprecated
)
```

**After:**
```python
# Get top models with repetition-aggregated metrics
top_models = predictions.top(
    5,
    rank_metric='rmse',
    by_repetition='Sample_ID',           # ← New
    repetition_method='mean',             # ← New
    repetition_exclude_outliers=True      # ← New
)

# Or use by_repetition=True to auto-resolve from dataset context
top_models = predictions.top(5, by_repetition=True)
```

### 6. YAML Configuration Files

**Before:**
```yaml
name: my_dataset
train_x: data/Xcal.csv
train_y: data/Ycal.csv
aggregate: sample_id
aggregate_method: mean
aggregate_exclude_outliers: false
```

**After:**
```yaml
name: my_dataset
train_x: data/Xcal.csv
train_y: data/Ycal.csv
repetition: sample_id
repetition_method: mean
repetition_exclude_outliers: false
```

### 7. Native Group Splitters

Native group splitters (`GroupKFold`, `StratifiedGroupKFold`) still work with the `group` parameter. They also respect `repetition` from DatasetConfigs if no explicit `group` is provided.

```python
# Still valid
{"split": GroupKFold(n_splits=3), "group": "Sample_ID"}

# Also valid - uses repetition from DatasetConfigs
pipeline = [GroupKFold(n_splits=3)]  # Uses repetition automatically
result = nirs4all.run(pipeline, dataset=DatasetConfigs(path, repetition="X"))
```

## Deprecation Warnings

The old parameters still work but trigger `DeprecationWarning`:

- `aggregate` in DatasetConfigs → use `repetition`
- `force_group` in split steps → use `repetition` in DatasetConfigs
- `aggregate` in Predictions.top() → use `by_repetition`
- `aggregate_method` → use `repetition_method`
- `aggregate_exclude_outliers` → use `repetition_exclude_outliers`

## Benefits of the New API

1. **Single declaration**: Set `repetition` once in DatasetConfigs, works everywhere
2. **Automatic propagation**: All splitters automatically respect repetition groups
3. **Composable**: Combine `repetition` with `group_by` for multi-column grouping
4. **Explicit opt-out**: Use `ignore_repetition=True` when needed
5. **Clearer semantics**: "repetition" clearly conveys the concept of sample measurement repetitions
