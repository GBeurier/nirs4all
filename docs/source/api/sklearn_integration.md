# sklearn Integration

The `nirs4all.sklearn` module provides sklearn-compatible wrappers for trained nirs4all pipelines, enabling integration with scikit-learn tools and SHAP explainers.

## Overview

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier
import shap

# Train with nirs4all
result = nirs4all.run(pipeline, dataset)

# Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

# Use sklearn-style interface
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Use with SHAP
explainer = shap.Explainer(pipe.predict, X_background)
shap_values = explainer(X_test)
```

## Key Concept

**Important:** `NIRSPipeline` is a **prediction wrapper**, not a training estimator. Training is always done via `nirs4all.run()`. The wrapper provides sklearn compatibility for:

- Prediction (`predict()`, `score()`)
- SHAP integration (via `predict()` callable)
- sklearn tools that only need prediction (e.g., `cross_val_predict`, `permutation_importance`)

```python
# ✅ Correct workflow
result = nirs4all.run(pipeline, dataset)  # Training
pipe = NIRSPipeline.from_result(result)   # Wrap for sklearn
pipe.predict(X_new)                        # Prediction works

# ❌ NIRSPipeline does NOT train
pipe = NIRSPipeline(steps=[...])
pipe.fit(X, y)  # Raises NotImplementedError
```

## Classes

### NIRSPipeline

sklearn-compatible regressor wrapper for trained nirs4all pipelines.

#### Class Methods

**`NIRSPipeline.from_result(result, fold=0)`**

Create wrapper from a `RunResult`.

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline

result = nirs4all.run(pipeline, dataset, save_artifacts=True)
pipe = NIRSPipeline.from_result(result)

# Specify which fold to use (default: 0)
pipe = NIRSPipeline.from_result(result, fold=2)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `RunResult` | Result from `nirs4all.run()` |
| `fold` | `int` | CV fold index to use (default: 0) |

**Returns:** `NIRSPipeline` instance.

---

**`NIRSPipeline.from_bundle(path)`**

Load wrapper from an exported `.n4a` bundle.

```python
from nirs4all.sklearn import NIRSPipeline

# Load from bundle (deployment scenario)
pipe = NIRSPipeline.from_bundle("exports/model.n4a")
y_pred = pipe.predict(X_new)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path to `.n4a` bundle file |

**Returns:** `NIRSPipeline` instance.

#### Instance Methods

**`predict(X)`**

Make predictions on new data.

```python
pipe = NIRSPipeline.from_bundle("model.n4a")
y_pred = pipe.predict(X_new)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `ndarray` | Input features (n_samples, n_features) |

**Returns:** `ndarray` of predictions (n_samples,) or (n_samples, n_outputs).

---

**`transform(X)`**

Apply preprocessing without model prediction.

```python
pipe = NIRSPipeline.from_result(result)
X_preprocessed = pipe.transform(X)
```

Useful for:
- Debugging preprocessing
- Getting intermediate features for analysis
- Base model predictions in stacking

---

**`score(X, y)`**

Calculate R² score (coefficient of determination).

```python
r2 = pipe.score(X_test, y_test)
print(f"R² = {r2:.4f}")
```

---

**`get_transformers()`**

Get list of preprocessing transformers.

```python
transformers = pipe.get_transformers()
for name, transformer in transformers:
    print(f"{name}: {type(transformer).__name__}")
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model_` | `object` | Underlying fitted model (for SHAP) |
| `is_fitted_` | `bool` | Always True (wrapper wraps fitted model) |
| `model_name` | `str` | Name of the model |
| `preprocessing_chain` | `list` | List of preprocessing step names |
| `model_step_index` | `int` | Index of model step in pipeline |
| `n_folds` | `int` | Number of CV folds |
| `fold_weights` | `list` | Weights for each fold |

### NIRSPipelineClassifier

Classification variant with `predict_proba()` and accuracy scoring.

```python
from nirs4all.sklearn import NIRSPipelineClassifier

# Classification pipeline
result = nirs4all.run(classification_pipeline, dataset)
pipe = NIRSPipelineClassifier.from_result(result)

# Classification-specific methods
y_pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)
accuracy = pipe.score(X_test, y_test)  # Uses accuracy, not R²
```

#### Additional Methods

**`predict_proba(X)`**

Get probability predictions (if model supports it).

```python
proba = pipe.predict_proba(X_test)
# proba.shape: (n_samples, n_classes)
```

## SHAP Integration

`NIRSPipeline` is designed to work seamlessly with SHAP:

### Kernel SHAP (Any Model)

```python
import shap
from nirs4all.sklearn import NIRSPipeline

# Create wrapper
pipe = NIRSPipeline.from_result(result)

# Use with Kernel SHAP (works with any model)
background = shap.kmeans(X_train, 50)  # Sample background data
explainer = shap.KernelExplainer(pipe.predict, background)
shap_values = explainer.shap_values(X_test[:10], nsamples=100)
```

### Explainer Auto-Detection

```python
import shap

# SHAP auto-detects best explainer type
explainer = shap.Explainer(pipe.predict, X_train[:100])
shap_values = explainer(X_test)

# Visualization
shap.summary_plot(shap_values)
shap.waterfall_plot(shap_values[0])
```

### Direct Model Access

For model-specific SHAP explainers (TreeExplainer, LinearExplainer):

```python
import shap

pipe = NIRSPipeline.from_result(result)

# Access underlying model
model = pipe.model_
print(f"Model type: {type(model).__name__}")

# Use model-specific explainer
if hasattr(model, 'feature_importances_'):
    # Tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
elif hasattr(model, 'coef_'):
    # Linear model
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)
```

## Examples

### Basic Workflow

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# 1. Train with nirs4all
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        PLSRegression(n_components=10)
    ],
    dataset="sample_data/regression",
    save_artifacts=True
)

# 2. Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

# 3. Use sklearn interface
from nirs4all.data import DatasetConfigs
dataset = DatasetConfigs("sample_data/regression")
for config, name in dataset.configs:
    ds = dataset.get_dataset(config, name)
    X_test, y_test = ds.x({})[:50], ds.y[:50]
    break

y_pred = pipe.predict(X_test)
r2 = pipe.score(X_test, y_test)
print(f"R² = {r2:.4f}")
```

### Export and Deploy

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline

# Training environment
result = nirs4all.run(pipeline, dataset, save_artifacts=True)
result.export("exports/production_model.n4a")

# Deployment environment (different machine/script)
pipe = NIRSPipeline.from_bundle("exports/production_model.n4a")

# Production predictions
y_pred = pipe.predict(new_data)
```

### Feature Importance with SHAP

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline
import shap
import matplotlib.pyplot as plt

# Train and wrap
result = nirs4all.run(pipeline, dataset)
pipe = NIRSPipeline.from_result(result)

# Get data
X = ...  # Your spectral data

# SHAP analysis
background = X[:100]
explainer = shap.Explainer(pipe.predict, background)
shap_values = explainer(X[100:150])

# Visualize
shap.summary_plot(shap_values, feature_names=[f"λ{i}" for i in range(X.shape[1])])
plt.title("Wavelength Importance")
plt.tight_layout()
plt.savefig("shap_summary.png")
```

### Cross-Validation Compatibility

```python
from nirs4all.sklearn import NIRSPipeline
from sklearn.model_selection import cross_val_predict

# Note: NIRSPipeline is for prediction only
# For CV during training, use nirs4all.run() with ShuffleSplit

# But you can use cross_val_predict for prediction validation
# (using pre-trained model)
pipe = NIRSPipeline.from_bundle("model.n4a")

# This works because it only needs predict()
# Note: cv=1 means no CV, just train-test split
from sklearn.model_selection import permutation_importance
perm_importance = permutation_importance(pipe, X_test, y_test, n_repeats=10)
```

## Best Practices

### 1. Save Artifacts for Wrapping

```python
# ✅ Enable artifacts for NIRSPipeline.from_result()
result = nirs4all.run(pipeline, dataset, save_artifacts=True)
pipe = NIRSPipeline.from_result(result)

# ❌ Without artifacts, from_result() may fail
result = nirs4all.run(pipeline, dataset, save_artifacts=False)
```

### 2. Export for Deployment

```python
# Development: train and export
result = nirs4all.run(pipeline, dataset)
result.export("model.n4a")

# Production: load from bundle
pipe = NIRSPipeline.from_bundle("model.n4a")
# No need for training data or original pipeline
```

### 3. Choose Appropriate SHAP Explainer

```python
import shap

# For any model (slower but universal)
explainer = shap.KernelExplainer(pipe.predict, background)

# For tree-based models (fast)
explainer = shap.TreeExplainer(pipe.model_)

# For linear models (fast)
explainer = shap.LinearExplainer(pipe.model_, background)

# For neural networks (gradient-based)
explainer = shap.DeepExplainer(pipe.model_, background)
```

### 4. Handle High-Dimensional NIRS Data

```python
# NIRS data often has 2000+ wavelengths
# Use subsampling for SHAP efficiency

# Sample fewer background points
background = shap.kmeans(X_train, 50)  # Not all samples

# Explain fewer test samples
shap_values = explainer(X_test[:20], nsamples=200)

# Or use feature groups
from nirs4all.operators.transforms import wavelength_bins
groups = wavelength_bins(n_features=X.shape[1], bin_size=50)
```

## Limitations

1. **No Training**: `NIRSPipeline.fit()` raises `NotImplementedError`. Use `nirs4all.run()` for training.

2. **Single Fold**: Default uses fold 0. For ensemble prediction across folds, use `nirs4all.predict()` with `all_predictions=True`.

3. **Preprocessing in Wrapper**: The wrapper applies preprocessing internally. SHAP values are for preprocessed features, not raw wavelengths.

4. **sklearn Tools Requiring fit()**: Tools like `GridSearchCV` won't work because they call `fit()`. Use nirs4all's generator syntax for hyperparameter search instead.

## See Also

- [Module-Level API](module_api.md) - `nirs4all.run()`, `predict()`, etc.
- [Q_sklearn_wrapper.py](https://github.com/gbeurier/nirs4all/blob/main/examples/Q_sklearn_wrapper.py) - Complete example
- [Q41_sklearn_shap.py](https://github.com/gbeurier/nirs4all/blob/main/examples/Q41_sklearn_shap.py) - SHAP example
- {doc}`/user_guide/index` - Complete user guide
