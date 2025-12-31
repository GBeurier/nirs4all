# Deployment Examples

This section covers saving, loading, and deploying trained NIRS4ALL models for production use.

```{contents} On this page
:local:
:depth: 2
```

## Overview

| Example | Topic | Difficulty | Duration |
|---------|-------|------------|----------|
| [U01](#u01-save-load-predict) | Save, Load, Predict | ★★☆☆☆ | ~4 min |
| [U02](#u02-export-bundle) | Export Bundle | ★★☆☆☆ | ~3 min |
| [U03](#u03-workspace-management) | Workspace Management | ★★☆☆☆ | ~3 min |
| [U04](#u04-sklearn-integration) | sklearn Integration | ★★☆☆☆ | ~3 min |

---

## U01: Save, Load, and Predict

**Save trained models and use them for prediction on new data.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/06_deployment/U01_save_load_predict.py)

### What You'll Learn

- Automatic model saving with PipelineRunner
- Prediction with prediction entries
- Prediction with model IDs
- Verifying prediction consistency

### Training with Model Saving

Enable `save_artifacts=True` to persist trained models:

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

# Define pipeline
pipeline = [
    MinMaxScaler(),
    SNV(),
    FirstDerivative(),
    ShuffleSplit(n_splits=3, random_state=42),
    {"model": PLSRegression(n_components=10)}
]

# Run with saving enabled
runner = PipelineRunner(save_artifacts=True, verbose=1)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "MyPipeline"),
    DatasetConfigs("sample_data/regression")
)

# Get best model info
best = predictions.top(n=1)[0]
model_id = best['id']
print(f"Model ID: {model_id}")
```

### Prediction Methods

#### Method 1: Using Prediction Entry

```python
# Create predictor
predictor = PipelineRunner()

# New data for prediction
new_data = DatasetConfigs({'X_test': 'path/to/new_data.csv'})

# Predict using the prediction entry directly
new_predictions, _ = predictor.predict(best, new_data)
print(f"Predictions shape: {new_predictions.shape}")
```

#### Method 2: Using Model ID

```python
# Predict using just the model ID string
predictor = PipelineRunner()
new_predictions, _ = predictor.predict(model_id, new_data)
```

### Prediction on NumPy Arrays

```python
import numpy as np

# Create synthetic new data
X_new = np.random.randn(10, 2151)  # Must match training feature count

# Create dataset from array
new_data = DatasetConfigs({'X_test': X_new})
predictions, _ = predictor.predict(model_id, new_data)
```

### Model Storage Location

Models are saved in the workspace:

```
workspace/runs/<run_id>/
├── model.pkl          # Trained model
├── preprocessor.pkl   # Preprocessing pipeline
├── metadata.json      # Configuration info
└── ...
```

---

## U02: Export Bundle

**Export portable model bundles for distribution.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/06_deployment/U02_export_bundle.py)

### What You'll Learn

- Creating portable `.n4a` bundles
- Bundle contents and structure
- Importing bundles
- Version compatibility

### Creating a Bundle

```python
from nirs4all.pipeline.bundle import export_bundle, import_bundle

# After training, export the best model
export_bundle(
    prediction_entry=best,
    output_path="my_model.n4a",
    include_metadata=True
)
```

### Bundle Contents

A `.n4a` bundle is a compressed archive containing:

| File | Description |
|------|-------------|
| `model.pkl` | Trained model |
| `pipeline.pkl` | Full preprocessing pipeline |
| `metadata.json` | Training info, metrics |
| `requirements.txt` | Python dependencies |
| `manifest.json` | Bundle structure info |

### Importing a Bundle

```python
# Load bundle
predictor = import_bundle("my_model.n4a")

# Use for prediction
predictions = predictor.predict(X_new)
```

### Bundle Portability

Bundles are designed to be portable:

- ✓ Self-contained (all preprocessing included)
- ✓ Version info for compatibility checks
- ✓ Can be shared via email, cloud storage, etc.
- ⚠ Requires compatible Python/library versions

---

## U03: Workspace Management

**Organize and manage your training artifacts.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/06_deployment/U03_workspace_management.py)

### What You'll Learn

- Workspace structure
- Artifact cleanup
- Library management
- Configuration

### Workspace Structure

```
workspace/
├── runs/              # Individual training runs
│   ├── run_20241231_123456/
│   │   ├── model.pkl
│   │   ├── predictions.json
│   │   └── charts/
│   └── ...
├── library/           # Curated model library
├── exports/           # Exported bundles
├── logs/              # Training logs
└── examples_output/   # Example outputs
```

### Workspace Configuration

```python
from nirs4all.workspace import WorkspaceConfig

# Configure workspace location
config = WorkspaceConfig(
    root="./my_workspace",
    max_runs=100,           # Keep last 100 runs
    auto_cleanup=True       # Remove old runs
)
```

### Artifact Cleanup

```python
from nirs4all.workspace import cleanup_workspace

# Remove runs older than 30 days
cleanup_workspace(
    max_age_days=30,
    keep_best_n=10  # Always keep top 10 models
)
```

### Model Library

Curate your best models:

```python
from nirs4all.workspace import add_to_library, list_library

# Add model to library
add_to_library(
    model_id=best['id'],
    name="Production_PLS_v1",
    tags=["production", "sugar_content"]
)

# List library
for model in list_library():
    print(f"{model['name']}: {model['metrics']}")
```

---

## U04: sklearn Integration

**Use NIRS4ALL models with scikit-learn workflows.**

[📄 View source code](https://github.com/GBeurier/nirs4all/blob/main/examples/user/06_deployment/U04_sklearn_integration.py)

### What You'll Learn

- SklearnWrapper for pipeline compatibility
- Using with sklearn utilities
- Cross-validation with sklearn
- Integration with sklearn pipelines

### SklearnWrapper

Wrap trained models for sklearn compatibility:

```python
from nirs4all.sklearn import SklearnWrapper

# Wrap a trained model
wrapper = SklearnWrapper(prediction_entry=best)

# Use like any sklearn estimator
predictions = wrapper.predict(X_new)

# Works with sklearn utilities
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, wrapper.predict(X_test))
```

### sklearn Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Create wrapper
wrapper = SklearnWrapper(prediction_entry=best)

# Use sklearn cross-validation
scores = cross_val_score(wrapper, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV RMSE: {np.sqrt(-scores.mean()):.4f}")
```

### sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline

# Create sklearn pipeline with NIRS4ALL model
sklearn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SklearnWrapper(prediction_entry=best))
])

# Fit and predict
sklearn_pipeline.fit(X_train, y_train)
predictions = sklearn_pipeline.predict(X_test)
```

### Grid Search with sklearn

```python
from sklearn.model_selection import GridSearchCV

# Note: Hyperparameter tuning should be done in NIRS4ALL
# Use sklearn GridSearch only for final pipeline tuning

param_grid = {'model__scale_factor': [0.9, 1.0, 1.1]}
grid_search = GridSearchCV(sklearn_pipeline, param_grid, cv=5)
grid_search.fit(X, y)
```

---

## Deployment Best Practices

### 1. Always Validate Before Deployment

```python
# Load the saved model
predictor = PipelineRunner()
preds, _ = predictor.predict(model_id, test_dataset)

# Verify against training results
assert np.allclose(preds[:5], reference_preds[:5])
```

### 2. Document Model Requirements

```python
# Include in bundle metadata
export_bundle(
    prediction_entry=best,
    output_path="model.n4a",
    metadata={
        "description": "Sugar content prediction for NIR spectra",
        "input_shape": (None, 2151),
        "wavelength_range": "1000-2500 nm",
        "preprocessing": "SNV + FirstDerivative",
        "training_date": "2024-12-31",
        "training_rmse": 1.23
    }
)
```

### 3. Version Your Models

```python
# Use semantic versioning
add_to_library(
    model_id=best['id'],
    name="SugarModel_v2.1.0",
    changelog="Improved preprocessing, added MSC"
)
```

### 4. Monitor Prediction Quality

```python
# Log predictions for monitoring
import logging

logger = logging.getLogger("nirs4all.predictions")
logger.info(f"Prediction batch: n={len(X_new)}, mean={preds.mean():.2f}")
```

---

## Running These Examples

```bash
cd examples

# Run all deployment examples
./run.sh -n "U0*.py" -c user

# Run save/load example
python user/06_deployment/U01_save_load_predict.py
```

## Next Steps

After mastering deployment:

- **Explainability**: Understand model decisions with SHAP
- **Transfer Learning**: Adapt models to new instruments
- **Advanced Pipelines**: Complex branching and merging
