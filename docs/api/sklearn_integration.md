# sklearn Integration

The `nirs4all.sklearn` module provides sklearn-compatible wrappers for prediction and SHAP integration.

## Overview

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline

# Train with nirs4all
result = nirs4all.run(pipeline, dataset)

# Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

# Use sklearn interface
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Works with SHAP
import shap
explainer = shap.Explainer(pipe.predict, X_background)
```

## Important

**NIRSPipeline is a prediction wrapper, not a training estimator.**

```python
# ✅ Correct
result = nirs4all.run(pipeline, dataset)  # Training
pipe = NIRSPipeline.from_result(result)   # Wrap

# ❌ Wrong - NIRSPipeline doesn't train
pipe = NIRSPipeline(steps=[...])
pipe.fit(X, y)  # Raises NotImplementedError
```

## Classes

### NIRSPipeline

sklearn-compatible regressor wrapper.

**Class Methods:**

```python
# From RunResult
pipe = NIRSPipeline.from_result(result, fold=0)

# From exported bundle
pipe = NIRSPipeline.from_bundle("model.n4a")
```

**Instance Methods:**

```python
y_pred = pipe.predict(X)      # Predictions
score = pipe.score(X, y)      # R² score
X_prep = pipe.transform(X)    # Preprocessing only
transformers = pipe.get_transformers()  # List of transformers
```

**Properties:**

| Property | Description |
|----------|-------------|
| `model_` | Underlying fitted model (for SHAP) |
| `is_fitted_` | Always True |
| `model_name` | Name of the model |
| `preprocessing_chain` | Preprocessing step names |
| `n_folds` | Number of CV folds |

### NIRSPipelineClassifier

Classification variant with additional methods:

```python
pipe = NIRSPipelineClassifier.from_result(result)
proba = pipe.predict_proba(X)  # Probability predictions
accuracy = pipe.score(X, y)    # Uses accuracy, not R²
```

## SHAP Integration

```python
import shap
from nirs4all.sklearn import NIRSPipeline

pipe = NIRSPipeline.from_result(result)

# Kernel SHAP (any model)
explainer = shap.KernelExplainer(pipe.predict, shap.kmeans(X_train, 50))
shap_values = explainer.shap_values(X_test)

# Or access underlying model for specialized explainers
model = pipe.model_
if hasattr(model, 'feature_importances_'):
    explainer = shap.TreeExplainer(model)
```

## Examples

```python
# Complete workflow
import nirs4all
from nirs4all.sklearn import NIRSPipeline
import shap

# Train
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="data",
    save_artifacts=True
)

# Export for deployment
result.export("model.n4a")

# Load in production
pipe = NIRSPipeline.from_bundle("model.n4a")
y_pred = pipe.predict(X_new)

# SHAP analysis
explainer = shap.Explainer(pipe.predict, X_background)
shap_values = explainer(X_test)
shap.summary_plot(shap_values)
```

## See Also

- [Module-Level API](module_api.md)
- [Q_sklearn_wrapper.py](https://github.com/gbeurier/nirs4all/blob/main/examples/Q_sklearn_wrapper.py)
- [Q41_sklearn_shap.py](https://github.com/gbeurier/nirs4all/blob/main/examples/Q41_sklearn_shap.py)
