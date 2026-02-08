# Model Training

This guide covers how to train models in NIRS4ALL, including cross-validation and accessing results.

## Overview

Model training in NIRS4ALL follows a simple pattern:

1. Add a cross-validation splitter to your pipeline
2. Add one or more model steps
3. Run the pipeline and access predictions

```python
pipeline = [
    # ... preprocessing steps ...
    ShuffleSplit(n_splits=5),                    # Cross-validation
    {"model": PLSRegression(n_components=10)}    # Model
]
```

## Basic Model Training

### Single Model

The simplest case - train one model:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
import nirs4all

pipeline = [
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    verbose=1
)

print(f"Best RMSE: {result.best_score:.4f}")
```

### Multiple Models

Compare multiple models in one run:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

pipeline = [
    ShuffleSplit(n_splits=3),

    # Each model is trained and evaluated
    {"model": PLSRegression(n_components=5)},
    {"model": PLSRegression(n_components=10)},
    {"model": PLSRegression(n_components=15)},
    {"model": Ridge(alpha=1.0)},
    {"model": RandomForestRegressor(n_estimators=100)},
]

result = nirs4all.run(pipeline, "sample_data/regression", verbose=1)

# Compare results
for pred in result.top(n=5, display_metrics=['rmse', 'r2']):
    print(f"{pred['model_name']}: RMSE={pred['rmse']:.4f}")
```

### Named Models

Give models custom names for easier identification:

```python
pipeline = [
    ShuffleSplit(n_splits=3),

    {"name": "PLS-5", "model": PLSRegression(n_components=5)},
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF-100", "model": RandomForestRegressor(n_estimators=100)},
]
```

## Cross-Validation

### Available Splitters

NIRS4ALL supports all sklearn splitters:

| Splitter | Use Case |
|----------|----------|
| `ShuffleSplit` | Random train/val splits (recommended for regression) |
| `KFold` | K consecutive folds |
| `StratifiedKFold` | Maintains class distribution (classification) |
| `GroupKFold` | Keeps groups together |
| `LeaveOneOut` | Leave one sample out (small datasets) |
| `TimeSeriesSplit` | Respects temporal order |

### ShuffleSplit (Recommended)

```python
from sklearn.model_selection import ShuffleSplit

pipeline = [
    # 5 random 75%/25% train/val splits
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=10)}
]
```

### KFold

```python
from sklearn.model_selection import KFold

pipeline = [
    # 5-fold cross-validation
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=10)}
]
```

### StratifiedKFold (Classification)

```python
from sklearn.model_selection import StratifiedKFold

pipeline = [
    # Maintains class balance in each fold
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    {"model": LogisticRegression()}
]
```

### GroupKFold (Group Data)

When samples belong to groups (e.g., multiple measurements per subject):

```python
from sklearn.model_selection import GroupKFold

pipeline = [
    # Keeps all samples from a group in same fold
    # Groups come from metadata column
    {"force_group_splitting": "subject_id"},
    GroupKFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
```

## Accessing Results

### The Result Object

`nirs4all.run()` returns a `RunResult` with convenient accessors:

```python
result = nirs4all.run(pipeline, dataset)

# Quick access
result.best_score        # Best model's primary metric (RMSE for regression)
result.best              # Best prediction entry (dict)
result.num_predictions   # Total number of prediction entries

# Top performers
result.top(n=5)                                    # Top 5 by default metric
result.top(n=5, rank_metric='r2')                  # Top 5 by R²
result.top(n=5, display_metrics=['rmse', 'r2'])    # Include specific metrics

# Full predictions list
result.predictions       # PredictionResultsList object
```

### Prediction Entry Structure

Each prediction entry contains:

```python
pred = result.best
print(pred['model_name'])      # "PLSRegression"
print(pred['rmse'])            # 0.1234
print(pred['r2'])              # 0.9876
print(pred['fold_id'])         # 0
print(pred['preprocessings'])  # "MinMaxScaler | SNV"
print(pred['y_true'])          # numpy array
print(pred['y_pred'])          # numpy array
```

### Filtering Results

```python
# Get predictions for specific model
pls_preds = [p for p in result.predictions if 'PLS' in p['model_name']]

# Get predictions for specific fold
fold_0 = [p for p in result.predictions if p['fold_id'] == 0]

# Get test partition only
test_preds = [p for p in result.predictions if p['partition'] == 'test']
```

## Regression Models

### PLS Regression (Recommended for NIRS)

```python
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
```

**Key parameters:**
- `n_components`: Number of latent variables (typically 1-30 for NIRS)

### Ridge Regression

```python
from sklearn.linear_model import Ridge

pipeline = [
    ShuffleSplit(n_splits=5),
    {"model": Ridge(alpha=1.0)}
]
```

**Key parameters:**
- `alpha`: Regularization strength (higher = more regularization)

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

pipeline = [
    ShuffleSplit(n_splits=5),
    {"model": RandomForestRegressor(n_estimators=100, random_state=42)}
]
```

**Key parameters:**
- `n_estimators`: Number of trees (100-500 typical)
- `max_depth`: Maximum tree depth (None = unlimited)

### Support Vector Regression

```python
from sklearn.svm import SVR

pipeline = [
    ShuffleSplit(n_splits=5),
    {"model": SVR(kernel='rbf', C=1.0)}
]
```

## Classification Models

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

pipeline = [
    StratifiedKFold(n_splits=5),
    {"model": LogisticRegression(max_iter=1000)}
]

result = nirs4all.run(
    pipeline,
    "sample_data/classification",
    verbose=1
)
```

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

pipeline = [
    StratifiedKFold(n_splits=5),
    {"model": RandomForestClassifier(n_estimators=100)}
]
```

### Support Vector Classifier

```python
from sklearn.svm import SVC

pipeline = [
    StratifiedKFold(n_splits=5),
    {"model": SVC(kernel='rbf', probability=True)}
]
```

## Target Processing

Scale or transform targets:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipeline = [
    # Scale features
    MinMaxScaler(),

    # Scale targets (applied before training, inverted after)
    {"y_processing": MinMaxScaler()},
    # OR
    {"y_processing": StandardScaler()},

    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
]
```

:::{note}
Target scaling is automatically inverted when making predictions, so metrics are computed on the original scale.
:::

## Model Persistence

### Export for Production

```python
# Export best model
result.export("exports/my_model.n4a")
```

### Load and Use

```python
from nirs4all.pipeline import load_bundle

bundle = load_bundle("exports/my_model.n4a")
y_pred = bundle.predict(X_new)
```

## Tips and Best Practices

### Choosing n_components for PLS

```python
# Try a range of components
pipeline = [
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(n_components=5)},
    {"model": PLSRegression(n_components=10)},
    {"model": PLSRegression(n_components=15)},
    {"model": PLSRegression(n_components=20)},
]

result = nirs4all.run(pipeline, dataset)

# Find optimal
for pred in result.top(n=5, display_metrics=['rmse']):
    print(f"{pred['model_name']}: RMSE={pred['rmse']:.4f}")
```

### Random State for Reproducibility

```python
# Set random_state for reproducible results
pipeline = [
    ShuffleSplit(n_splits=5, random_state=42),
    {"model": RandomForestRegressor(n_estimators=100, random_state=42)}
]
```

### Preprocessing Before Model

Always apply preprocessing before the splitter:

```python
pipeline = [
    # Preprocessing first
    MinMaxScaler(),
    StandardNormalVariate(),

    # Then cross-validation
    ShuffleSplit(n_splits=5),

    # Then model
    {"model": PLSRegression(n_components=10)}
]
```

## Complete Example

```python
"""Complete model training example."""

import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import StandardNormalVariate, FirstDerivative

# Define pipeline
pipeline = [
    # Preprocessing
    MinMaxScaler(),
    StandardNormalVariate(),
    FirstDerivative(),

    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Cross-validation
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),

    # Multiple models
    {"name": "PLS-5", "model": PLSRegression(n_components=5)},
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "PLS-15", "model": PLSRegression(n_components=15)},
    {"name": "RF-100", "model": RandomForestRegressor(n_estimators=100, random_state=42)},
]

# Run pipeline
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="ModelComparison",
    verbose=1,
    save_artifacts=True
)

# View results
print(f"\n📊 Results Summary:")
print(f"   Total predictions: {result.num_predictions}")
print(f"   Best RMSE: {result.best_score:.4f}")

print("\n🏆 Top 5 Models:")
for i, pred in enumerate(result.top(n=5, display_metrics=['rmse', 'r2']), 1):
    print(f"   {i}. {pred['model_name']}: RMSE={pred['rmse']:.4f}, R²={pred['r2']:.4f}")

# Export best model
result.export("exports/best_model.n4a")
print("\n✅ Best model exported to exports/best_model.n4a")
```

## See Also

- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax
- {doc}`/reference/metrics` - Available evaluation metrics
- {doc}`/user_guide/pipelines/writing_pipelines` - Pipeline construction guide
- {doc}`/user_guide/models/hyperparameter_tuning` - Automated tuning with Optuna

```{seealso}
**Related Examples:**
- [U01: Hello World](../../../examples/user/01_getting_started/U01_hello_world.py) - Your first training pipeline
- [U02: Basic Regression](../../../examples/user/01_getting_started/U02_basic_regression.py) - Training with preprocessing and visualization
- [U01: Multi-Model](../../../examples/user/04_models/U01_multi_model.py) - Compare multiple models in one run
```
