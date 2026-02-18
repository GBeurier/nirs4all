# Cross-Validation Examples

This section covers cross-validation strategies to properly evaluate model performance on NIRS data.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-cv-strategies) | CV Strategies | ★★☆☆☆ | ~4 min |
| [U02](#u02-group-splitting) | Group Splitting | ★★☆☆☆ | ~3 min |
| [U03](#u03-sample-filtering) | Sample Filtering | ★★☆☆☆ | ~3 min |
| [U04](#u04-aggregation) | Aggregation | ★★☆☆☆ | ~3 min |

---

## U01: CV Strategies

**Select appropriate cross-validation for your data structure.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U01_cv_strategies.py)

### What You'll Learn

- Standard CV: KFold, ShuffleSplit, RepeatedKFold
- Stratified CV for classification
- Time-series CV for temporal data
- Leave-One-Out for small datasets

### CV Strategy Selection Guide

| Strategy | When to Use |
|----------|-------------|
| **KFold** | Standard regression, moderate-large datasets |
| **ShuffleSplit** | Flexible test size, many random splits |
| **RepeatedKFold** | Small datasets, need robust estimates |
| **StratifiedKFold** | Classification, class imbalance |
| **StratifiedShuffleSplit** | Classification + flexible splits |
| **TimeSeriesSplit** | Temporal/sequential data |
| **LeaveOneOut** | Very small datasets (<50 samples) |
| **SPXYFold** | Spatially representative folds via SPXY |
| **KennardStoneSplitter** | Kennard-Stone based train/test split |
| **KBinsStratifiedSplitter** | Stratification for continuous targets |

### KFold - Standard K-Fold

Divides data into K non-overlapping folds:

```python
from sklearn.model_selection import KFold

pipeline = [
    MinMaxScaler(),
    SNV(),

    # 5-fold cross-validation
    KFold(n_splits=5, shuffle=True, random_state=42),

    PLSRegression(n_components=10)
]
```

**Key parameters:**
- `n_splits`: Number of folds (typically 5-10)
- `shuffle`: Randomize before splitting (recommended)
- `random_state`: For reproducibility

### ShuffleSplit - Random Splits

More flexible than KFold:

```python
from sklearn.model_selection import ShuffleSplit

# 10 random splits with 25% test
ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
```

**Advantages:**
- Control test size exactly
- Number of splits independent of test size
- Good for large datasets

### RepeatedKFold - Multiple Repetitions

Repeats K-fold CV multiple times with different shuffles:

```python
from sklearn.model_selection import RepeatedKFold

# 5-fold repeated 3 times = 15 total evaluations
RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
```

**Best for:**
- Small datasets where variance is high
- When you need robust uncertainty estimates

### StratifiedKFold - Classification

Preserves class proportions in each fold:

```python
from sklearn.model_selection import StratifiedKFold

# Essential for imbalanced classification
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Critical for:**
- Imbalanced class distributions
- Multi-class classification
- Ensuring each fold has representative samples

### TimeSeriesSplit - Temporal Data

Expanding window approach for sequential data:

```python
from sklearn.model_selection import TimeSeriesSplit

# Train on past, test on future
TimeSeriesSplit(n_splits=5)
```

**Prevents:**
- Look-ahead bias
- Data leakage from future to past

**Use when:**
- Samples are ordered by time
- Temporal dependencies exist

### LeaveOneOut - Small Datasets

Leave exactly one sample out per fold:

```python
from sklearn.model_selection import LeaveOneOut

# N samples = N folds
LeaveOneOut()
```

**Trade-offs:**
- ✓ Maximum use of data for training
- ✓ N evaluations give complete picture
- ✗ Slow for large datasets
- ✗ High variance in estimates

---

## U02: Group Splitting

**Handle grouped/clustered data correctly.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U02_group_splitting.py)

### What You'll Learn

- GroupKFold for clustered samples
- GroupShuffleSplit for random group splits
- Handling biological replicates
- Avoiding data leakage from groups

### Why Group Splitting?

When samples are not independent:

```
Example: 5 measurements per fruit
├── Fruit 1: samples 1-5
├── Fruit 2: samples 6-10
├── Fruit 3: samples 11-15
└── ...

Wrong: Random split → Fruit 1's samples in both train AND test
Right: Group split → All of Fruit 1's samples in train OR test
```

### GroupKFold

```python
from sklearn.model_selection import GroupKFold

# Requires group labels
groups = [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5  # 5 groups of 5

pipeline = [
    MinMaxScaler(),
    SNV(),
    GroupKFold(n_splits=5),
    PLSRegression(n_components=10)
]

# Pass groups in dataset
result = nirs4all.run(
    pipeline=pipeline,
    dataset=(X, y, {"groups": groups})
)
```

### GroupShuffleSplit

Random group-aware splits:

```python
from sklearn.model_selection import GroupShuffleSplit

GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
```

### Common Grouping Scenarios

| Scenario | Group Definition |
|----------|------------------|
| Biological replicates | Sample/individual ID |
| Time series by subject | Subject ID |
| Multi-site study | Site ID |
| Batch effects | Batch ID |

---

## U03: Sample Filtering

**Filter samples based on criteria during cross-validation.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U03_sample_filtering.py)

### What You'll Learn

- Filtering outliers
- Conditioning on metadata
- Train/test set requirements

### Sample Filtering in Pipeline

```python
pipeline = [
    MinMaxScaler(),
    SNV(),

    # Filter: only include samples meeting criteria
    {"filter": {
        "y_range": (0, 100),       # Target value range
        "partition": ["train"]     # Which partitions to filter
    }},

    ShuffleSplit(n_splits=3),
    PLSRegression(n_components=10)
]
```

### Filtering Options

| Filter | Description |
|--------|-------------|
| `y_range` | Keep samples with y in (min, max) |
| `x_range` | Keep samples with X values in range |
| `metadata` | Filter based on metadata columns |
| `outliers` | Remove statistical outliers |

---

## U04: Aggregation

**Aggregate results across folds and variants.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/05_cross_validation/U04_aggregation.py)

### What You'll Learn

- Aggregating metrics across folds
- Statistical summaries
- Confidence intervals

### Understanding Aggregation

When you run a pipeline with cross-validation, you get:
- One prediction per fold
- Multiple folds per model configuration

**Aggregation** summarizes these:

```python
# Get predictions
result = nirs4all.run(pipeline=pipeline, dataset=dataset)

# Aggregate across folds
summary = result.predictions.aggregate(
    by=['model_name', 'preprocessings'],
    metrics=['rmse', 'r2'],
    aggregations=['mean', 'std', 'min', 'max']
)
```

### Aggregation Options

| Aggregation | Description |
|-------------|-------------|
| `mean` | Average across folds |
| `std` | Standard deviation |
| `min`, `max` | Range |
| `median` | Robust central tendency |
| `ci_95` | 95% confidence interval |

### Visualization with Aggregation

```python
analyzer = PredictionAnalyzer(result.predictions)

# Candlestick shows distribution
analyzer.plot_candlestick(
    variable="model_name",
    display_metric='rmse'
)

# Heatmap with aggregation
analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    aggregation='mean',  # or 'best', 'median'
    display_metric='rmse'
)
```

---

## CV Best Practices

### 1. Match CV to Data Structure

```python
# Independent samples
ShuffleSplit(n_splits=5, test_size=0.2)

# Classification
StratifiedKFold(n_splits=5)

# Grouped samples
GroupKFold(n_splits=5)

# Time series
TimeSeriesSplit(n_splits=5)
```

### 2. Number of Splits

| Dataset Size | Recommended Splits |
|--------------|-------------------|
| < 50 | LeaveOneOut or RepeatedKFold |
| 50-200 | 5-fold with repeats |
| 200-1000 | 5-10 fold |
| > 1000 | ShuffleSplit (faster) |

### 3. Always Shuffle for Regression

```python
# Good: Shuffle before splitting
KFold(n_splits=5, shuffle=True, random_state=42)

# Risky: Sequential splits may have patterns
KFold(n_splits=5, shuffle=False)  # Only for time series
```

### 4. Set random_state for Reproducibility

```python
# Reproducible
ShuffleSplit(n_splits=5, random_state=42)

# Different each run (not reproducible)
ShuffleSplit(n_splits=5)  # random_state=None
```

### 5. Validate Stratification

For classification, check class distribution in each fold:

```python
result = nirs4all.run(pipeline=pipeline, dataset=dataset)

# View fold distribution
nirs4all.run(
    pipeline=[..., "fold_chart", ...],
    dataset=dataset,
    plots_visible=True
)
```

---

## Running These Examples

```bash
cd examples

# Run all CV examples
./run.sh -n "U0*.py" -c user

# Run with plots to see fold distributions
python user/05_cross_validation/U01_cv_strategies.py --plots --show
```

## Next Steps

After mastering cross-validation:

- **Deployment**: Save and deploy trained models
- **Explainability**: Understand model decisions
- **Advanced**: Nested CV, custom splitters
