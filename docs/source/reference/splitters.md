# Splitters Reference

Splitters define how data is divided into training and test sets for cross-validation or hold-out evaluation. nirs4all provides NIRS-specific splitters alongside all standard sklearn splitters.

## Usage in Pipeline

```python
from nirs4all.operators.splitters import KennardStoneSplitter, SPXYFold
from sklearn.model_selection import ShuffleSplit

# Single hold-out split
pipeline = [
    SNV(),
    KennardStoneSplitter(test_size=0.2),
    {"model": PLSRegression(n_components=10)},
]

# K-fold cross-validation
pipeline = [
    SNV(),
    SPXYFold(n_splits=5),
    {"model": PLSRegression(n_components=10)},
]
```

---

## NIRS-Specific Splitters

All splitters below are imported from `nirs4all.operators.splitters`.

### Single-Split Methods

These produce a single train/test split.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `KennardStoneSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | Kennard-Stone algorithm: selects maximally diverse training samples using max-min distance criterion |
| `SPXYSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | SPXY (Sample set Partitioning based on joint X and Y distances): Kennard-Stone extended to include target information |
| `KMeansSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | K-Means clustering-based split: clusters samples and assigns cluster centers to training |
| `KBinsStratifiedSplitter` | `test_size`, `n_bins=10`, `strategy="uniform"`, `encode="ordinal"` | Stratified sampling using KBins discretization of continuous targets |
| `SystematicCircularSplitter` | `test_size` | Systematic circular sampling: orders by y-value, then selects at regular intervals |
| `SPlitSplitter` | `test_size` | Data twinning algorithm (Vakayil & Joseph 2022): selects a statistically representative subset |

### K-Fold Methods

These produce multiple folds for cross-validation.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `SPXYFold` | `n_splits=5`, `metric="euclidean"`, `y_metric="euclidean"`, `pca_components=None` | SPXY-based K-Fold: assigns samples to folds using joint X-Y distances for spatially representative folds |
| `SPXYGFold` | `n_splits=5`, `metric="euclidean"`, `y_metric="euclidean"`, `pca_components=None` | Group-aware SPXY K-Fold: respects group boundaries while using SPXY distance criterion |
| `BinnedStratifiedGroupKFold` | `n_splits=5`, `n_bins=10`, `strategy="quantile"`, `shuffle=False` | Stratified Group K-Fold with binned continuous targets: ensures balanced target distribution across folds while respecting groups |

### Group Wrapper

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `GroupedSplitterWrapper` | `splitter`, `aggregation="mean"`, `y_aggregation=None` | Wraps any sklearn splitter to add group-awareness; aggregates samples by group and ensures no group leakage |

Usage:
```python
from nirs4all.operators.splitters import GroupedSplitterWrapper
from sklearn.model_selection import KFold

# Any splitter becomes group-aware
wrapper = GroupedSplitterWrapper(KFold(n_splits=5))
```

---

## Commonly Used sklearn Splitters

These are imported from `sklearn.model_selection` and work directly in nirs4all pipelines.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `KFold` | `n_splits=5`, `shuffle=False` | Standard K-Fold cross-validation |
| `StratifiedKFold` | `n_splits=5`, `shuffle=False` | K-Fold with stratification on target classes |
| `ShuffleSplit` | `n_splits=10`, `test_size=0.1` | Random train/test splits with configurable sizes |
| `RepeatedKFold` | `n_splits=5`, `n_repeats=10` | Repeated K-Fold cross-validation |
| `LeaveOneOut` | *(none)* | Leave-one-out cross-validation |
| `GroupKFold` | `n_splits=5` | K-Fold respecting group boundaries |
| `StratifiedGroupKFold` | `n_splits=5`, `shuffle=False` | Stratified K-Fold respecting groups |

---

## SPXYFold Parameters Detail

`SPXYFold` supports several configurations for different use cases:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `y_metric` | `"euclidean"` | For regression (continuous y) -- default SPXY behavior |
| `y_metric` | `"hamming"` | For classification (categorical y) |
| `y_metric` | `None` | Ignore Y (pure Kennard-Stone, X-only selection) |
| `pca_components` | `int` or `None` | Apply PCA dimensionality reduction before distance computation |

---

## See Also

- {doc}`../reference/pipeline_keywords` -- Pipeline keyword syntax
- {doc}`../reference/filters` -- Sample filtering operators
- {doc}`../reference/models` -- Built-in models reference
