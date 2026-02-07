# Force Group Splitting: Universal Group Support

This guide explains the `force_group` parameter that enables **any sklearn-compatible splitter** to work with grouped samples, ensuring all samples from the same group stay together in train or test sets.

## Quick Start

```python
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold

# Use KFold with group-awareness
{"split": KFold(n_splits=5), "force_group": "Sample_ID"}

# Use ShuffleSplit with group-awareness
{"split": ShuffleSplit(test_size=0.2), "force_group": "Sample_ID"}

# Stratify on binned y values (for regression)
{"split": KFold(n_splits=5), "force_group": "y", "n_bins": 10}
```

## Why Force Group?

### The Problem

When dealing with repeated measurements (multiple spectra per sample), standard cross-validation can cause **data leakage**:

```python
# Problem: KFold may put measurements from the same sample in both train and test
{"split": KFold(n_splits=5)}  # Data leakage risk!
```

### The Traditional Solution

Use group-aware splitters like `GroupKFold`:

```python
{"split": GroupKFold(n_splits=5), "group": "Sample_ID"}
```

But this limits you to only group-aware splitters (`GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold`).

### The Force Group Solution

`force_group` wraps **any** splitter to add group-awareness:

```python
# Now ANY splitter works with groups!
{"split": KFold(n_splits=5), "force_group": "Sample_ID"}
{"split": ShuffleSplit(n_splits=10, test_size=0.2), "force_group": "Sample_ID"}
{"split": StratifiedKFold(n_splits=5), "force_group": "Sample_ID"}
```

## How It Works

1. **Aggregate**: Samples are grouped by the specified column
2. **Split**: The inner splitter works on "virtual samples" (one per group)
3. **Expand**: Fold indices are expanded back to original sample indices

```
Original Data (100 samples, 20 groups)
        ↓
   Aggregation (20 virtual samples)
        ↓
   Splitter operates on 20 samples
        ↓
   Expansion back to 100 samples
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `force_group` | `str` | Metadata column name for grouping, or `"y"` for target-based binning |
| `aggregation` | `str` | X aggregation method: `"mean"` (default), `"median"`, `"first"` |
| `y_aggregation` | `str` | Y aggregation method: `"mean"`, `"mode"`, `"first"` (auto-detected) |
| `n_bins` | `int` | Number of bins for `force_group="y"` (default: 5) |

## Usage Examples

### Basic Group Splitting

```python
from sklearn.model_selection import KFold

pipeline = [
    {"split": KFold(n_splits=5), "force_group": "Sample_ID"},
    PLSRegression(n_components=5)
]
```

### With Different Aggregation Methods

```python
# Use median aggregation (more robust to outliers)
{"split": KFold(n_splits=5), "force_group": "Sample_ID", "aggregation": "median"}

# Use first sample per group (fastest, no actual aggregation)
{"split": ShuffleSplit(test_size=0.2), "force_group": "Sample_ID", "aggregation": "first"}
```

### Y-Binning for Regression

Use `force_group="y"` to bin continuous target values into groups:

```python
# Bin y values into 10 quantile bins, then split by bins
{"split": KFold(n_splits=5), "force_group": "y", "n_bins": 10}
```

This ensures samples with similar y values tend to be in the same fold, providing more balanced y distribution across folds.

### Stratified Splitting with Groups

For classification with group-awareness:

```python
from sklearn.model_selection import StratifiedKFold

{
    "split": StratifiedKFold(n_splits=5),
    "force_group": "Sample_ID",
    "y_aggregation": "mode"  # Use most common class in group
}
```

## Comparison with `group` Parameter

| Feature | `group` | `force_group` |
|---------|---------|---------------|
| Works with GroupKFold | ✓ | ✓ |
| Works with KFold | ✗ | ✓ |
| Works with ShuffleSplit | ✗ | ✓ |
| Works with StratifiedKFold | ✗ | ✓ |
| Y-binning support | ✗ | ✓ |
| Aggregation options | ✗ | ✓ |

## Best Practices

1. **Choose appropriate aggregation**: Use `"mean"` for normal distributions, `"median"` for outlier robustness, `"first"` for speed

2. **Set n_bins appropriately**: For `force_group="y"`:
   - More bins = finer stratification but requires more samples
   - Fewer bins = more robust but coarser grouping
   - Recommended: 5-20 bins for datasets with 100+ samples

3. **Match y_aggregation to task**:
   - Classification: use `"mode"` (most common class)
   - Regression: use `"mean"` (average value)

4. **Prefer `force_group` over `group`** when using non-group-aware splitters to avoid silent failures

## Technical Details

Under the hood, `force_group` uses `GroupedSplitterWrapper`:

```python
from nirs4all.operators.splitters import GroupedSplitterWrapper

wrapper = GroupedSplitterWrapper(
    splitter=KFold(n_splits=5),
    aggregation="mean",
    y_aggregation="mean"
)

for train_idx, test_idx in wrapper.split(X, y, groups=sample_ids):
    # train_idx and test_idx are original sample indices
    # All samples from the same group are in the same fold
    pass
```

## Related

- [Writing Pipelines](./writing_pipelines.md): General pipeline authoring guide
- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference

```{seealso}
**Related Examples:**
- [U02: Group Splitting](../../../examples/user/05_cross_validation/U02_group_splitting.py) - GroupKFold and repetition-aware splitting
- [U04: Repetition Aggregation](../../../examples/user/05_cross_validation/U04_aggregation.py) - Handling repeated measurements
```
