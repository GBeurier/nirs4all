# Pipeline Keywords Reference

Pipeline steps in nirs4all can be plain operators (class or instance) or **dict-wrapped steps** with special keywords that control how the operator is applied. This page documents every available keyword.

## Quick Reference

| Keyword | Purpose |
|---------|---------|
| `model` | Define model step |
| `y_processing` | Target (y) scaling with automatic inverse during prediction |
| `tag` | Mark samples with a tag without removing them |
| `exclude` | Remove flagged samples from training |
| `branch` | Parallel sub-pipelines (duplication) or sample splitting (separation) |
| `merge` | Combine branch outputs |
| `sample_augmentation` | Data augmentation applied to training samples |
| `feature_augmentation` | Feature-level augmentation with multiple action modes |
| `concat_transform` | Concatenate features from multiple transforms |
| `rep_to_sources` | Convert repetition groups to multi-source format |
| `rep_to_pp` | Convert repetition groups to preprocessing pipelines |
| `name` | Name a pipeline step for display and identification |

---

## `model`

Defines the model step in the pipeline.

**Syntax:**
```python
{"model": PLSRegression(n_components=10)}
```

**Behavior:** The operator is treated as the model to be trained and evaluated. During cross-validation, it is fit on training folds and evaluated on test folds. During refit, it is trained on the full dataset.

**Notes:**
- Any sklearn-compatible estimator with `fit()` and `predict()` methods can be used.
- A pipeline can have multiple model steps (e.g., in stacking).
- If a bare estimator is detected (not wrapped in a dict), nirs4all auto-detects it as a model based on whether it has a `predict()` method.

---

## `y_processing`

Applies a transformer to the target variable (y) before model training. The inverse transform is automatically applied to predictions.

**Syntax:**
```python
{"y_processing": MinMaxScaler()}
```

**Behavior:** The transformer is fit on y_train, then transforms y_train for model fitting. During prediction, the model output is inverse-transformed back to the original scale.

**Notes:**
- The transformer must implement `inverse_transform()` for prediction to work correctly.
- Common use: `MinMaxScaler()`, `StandardScaler()` for target normalization.

---

## `tag`

Marks samples with a tag for downstream analysis or branching. Tagged samples are NOT removed from the training set.

**Syntax:**
```python
# Single filter
{"tag": YOutlierFilter(method="iqr")}

# Multiple filters
{"tag": [YOutlierFilter(method="iqr"), XOutlierFilter(method="mahalanobis")]}
```

**Behavior:** The filter is evaluated and sample tags are stored in the dataset. Tags can be used later for separation branching via `{"branch": {"by_tag": ...}}`.

**Notes:**
- Tags are metadata annotations, not data modifications.
- Tag names are auto-generated from the filter class and method (e.g., `y_outlier_iqr`).
- Use `tag_name` parameter on the filter to set a custom tag name.

---

## `exclude`

Removes flagged samples from the training set. Test samples are never excluded.

**Syntax:**
```python
# Single filter
{"exclude": YOutlierFilter(method="iqr", threshold=1.5)}

# Multiple filters with mode
{"exclude": [
    YOutlierFilter(method="iqr"),
    XOutlierFilter(method="mahalanobis", threshold=3.0),
], "mode": "any"}
```

**Parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `mode` | `"any"` (default) | Exclude sample if ANY filter flags it |
| `mode` | `"all"` | Exclude sample only if ALL filters flag it |

**Behavior:** Filters are evaluated and matching samples are removed from the training indexer. The actual data arrays are not modified -- samples are masked out via the indexer.

---

## `branch`

Creates parallel sub-pipelines. Two modes: **duplication** (same samples, different pipelines) and **separation** (disjoint sample subsets).

### Duplication Branches

All branches receive the same samples and process them independently.

**Syntax:**
```python
{"branch": [
    [SNV(), PLSRegression(10)],
    [MSC(), RandomForestRegressor()],
]}
```

### Separation Branches

Branches receive disjoint subsets of samples.

**By metadata column:**
```python
{"branch": {"by_metadata": "site"}}
```

**By tag values:**
```python
{"branch": {"by_tag": "y_outlier_iqr", "values": {
    "clean": False,
    "outliers": True,
}}}
```

**By source (multi-source datasets):**
```python
{"branch": {"by_source": True, "steps": {
    "NIR": [SNV(), PLSRegression(10)],
    "markers": [MinMaxScaler(), Ridge()],
}}}
```

**By filter:**
```python
{"branch": {"by_filter": SampleFilter(...)}}
```

**Notes:**
- Duplication branches are typically followed by `{"merge": "predictions"}` for stacking.
- Separation branches are typically followed by `{"merge": "concat"}` to reassemble samples.

---

## `merge`

Combines outputs from preceding branches.

**Syntax:**
```python
# For duplication branches (stacking)
{"merge": "predictions"}   # Use OOF predictions as features for meta-model
{"merge": "features"}      # Use transformed features as input for next step
{"merge": "all"}           # Merge all available outputs

# For separation branches (reassembly)
{"merge": "concat"}        # Reassemble samples in original order

# For multi-source merging
{"merge": {"sources": "concat"}}
```

| Value | Use Case | Description |
|-------|----------|-------------|
| `"predictions"` | Duplication branches | Out-of-fold predictions become features for a meta-model |
| `"features"` | Duplication branches | Transformed feature matrices are concatenated horizontally |
| `"all"` | Duplication branches | All available outputs (features + predictions) |
| `"concat"` | Separation branches | Reassembles disjoint sample subsets in original order |

---

## `sample_augmentation`

Applies data augmentation to training samples. Augmented samples are added to the training set but not to the test set.

**Syntax:**
```python
# Single augmenter
{"sample_augmentation": GaussianAdditiveNoise(sigma=0.01)}

# Multiple augmenters (applied sequentially)
{"sample_augmentation": [
    GaussianAdditiveNoise(sigma=0.01),
    WavelengthShift(shift_range=(-1.0, 1.0)),
]}
```

**Behavior:** The augmentation operator is applied to training data only. Augmented copies are appended to the training set. During prediction, this step is skipped.

**Notes:**
- Augmentation is only applied during training, never during prediction.
- See {doc}`../reference/augmentations` for all available augmentation operators.

---

## `feature_augmentation`

Creates multiple preprocessing views of the same data. Supports three action modes.

**Syntax:**
```python
# Extend mode (default) - add new views alongside existing
{"feature_augmentation": [SNV(), SavitzkyGolay(deriv=1)]}

# With explicit action mode
{"feature_augmentation": [SNV(), Gaussian()], "action": "extend"}
{"feature_augmentation": [SNV(), Gaussian()], "action": "add"}
{"feature_augmentation": [SNV(), Gaussian()], "action": "replace"}
```

**Action Modes:**

| Action | Growth Pattern | Description |
|--------|---------------|-------------|
| `"extend"` (default) | Linear | Each operator runs independently on the base. New views are added to the set. |
| `"add"` | Multiplicative + originals | Each operator is chained on top of ALL existing views. Originals are kept. |
| `"replace"` | Multiplicative | Each operator is chained on top of ALL existing views. Originals are discarded. |

**Example with `"extend"` (starting from raw_A):**
```
raw_A, SNV, SavitzkyGolay  (3 views)
```

**Example with `"add"` (starting from raw_A):**
```
raw_A, raw_A+SNV, raw_A+SavitzkyGolay  (3 views)
```

**Example with `"replace"` (starting from raw_A):**
```
raw_A+SNV, raw_A+SavitzkyGolay  (2 views, raw_A discarded)
```

---

## `concat_transform`

Applies multiple transforms and concatenates the resulting features horizontally.

**Syntax:**
```python
{"concat_transform": [SNV(), SavitzkyGolay(deriv=1), Detrend()]}
```

**Behavior:** Each transform is applied independently to the same input. The resulting feature matrices are concatenated along the feature axis, producing a wider feature matrix.

**Notes:**
- Different from `feature_augmentation` which creates separate preprocessing views.
- `concat_transform` produces a single 2D matrix with concatenated features.

---

## `rep_to_sources`

Converts repetition groups (multiple spectra per physical sample) into a multi-source format.

**Syntax:**
```python
{"rep_to_sources": "Sample_ID"}
```

**Behavior:** Groups spectra by the specified metadata column (e.g., `"Sample_ID"`). Each repetition becomes a separate source in the multi-source dataset.

---

## `rep_to_pp`

Converts repetition groups into separate preprocessing pipelines.

**Syntax:**
```python
{"rep_to_pp": "Sample_ID"}
```

**Behavior:** Groups spectra by the specified metadata column. Each repetition becomes a separate preprocessing view, enabling per-repetition processing.

---

## `name`

Names a pipeline step for display and identification purposes.

**Syntax:**
```python
{"name": "scatter_correction", "step": SNV()}
```

**Behavior:** The step is executed normally, but uses the provided name in logs, reports, and visualization.

---

## Combining Keywords

Multiple keywords can be combined in a single dict when they apply to the same step:

```python
# Exclude with mode
{"exclude": [YOutlierFilter(), XOutlierFilter()], "mode": "any"}

# Feature augmentation with action
{"feature_augmentation": [SNV(), Detrend()], "action": "extend"}

# Branch with steps
{"branch": {"by_source": True, "steps": {"NIR": [...], "VIS": [...]}}}
```

---

## See Also

- {doc}`../reference/generator_keywords` -- Generator syntax (`_or_`, `_range_`, `_grid_`, etc.)
- {doc}`../reference/transforms` -- Available transforms
- {doc}`../reference/augmentations` -- Available augmentation operators
- {doc}`../reference/filters` -- Available filters for `tag` and `exclude`
- {doc}`../reference/models` -- Available models
- {doc}`../reference/splitters` -- Available splitters
