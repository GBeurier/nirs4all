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

- Training and automatic model saving
- Prediction on new data
- Exporting and loading models
- Verifying prediction consistency

### Training with Model Saving

Train a pipeline using the module-level API:

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from nirs4all.operators.transforms import SNV, FirstDerivative

# Define pipeline
pipeline = [
    MinMaxScaler(),
    SNV(),
    FirstDerivative(),
    ShuffleSplit(n_splits=3, random_state=42),
    {"model": PLSRegression(n_components=10)}
]

# Run training
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    name="MyPipeline",
    verbose=1,
)

# Get best model info
print(f"Best RMSE: {result.best_rmse:.4f}")
```

### Exporting and Predicting

#### Export the Trained Model

```python
# Export as a portable .n4a bundle
result.export("my_model.n4a")
```

#### Predict on New Data

```python
# Predict using the exported bundle
new_predictions = nirs4all.predict("my_model.n4a", "path/to/new_data.csv")
print(f"Predictions shape: {new_predictions.shape}")
```

### Prediction on NumPy Arrays

```python
import numpy as np

# Create synthetic new data
X_new = np.random.randn(10, 2151)  # Must match training feature count

# Predict directly from array
predictions = nirs4all.predict("my_model.n4a", X_new)
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
import nirs4all

# After training, export the best model as a bundle
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    verbose=1,
)
result.export("my_model.n4a")
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

### Loading a Bundle for Prediction

```python
import nirs4all

# Predict using the exported bundle
predictions = nirs4all.predict("my_model.n4a", X_new)
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

### NIRSPipeline

Load a trained bundle as a sklearn-compatible estimator:

```python
from nirs4all.sklearn import NIRSPipeline

# Load from exported bundle
model = NIRSPipeline.from_bundle("my_model.n4a")

# Use like any sklearn estimator
predictions = model.predict(X_new)

# Works with sklearn utilities
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, model.predict(X_test))
```

### SHAP Integration

```python
import shap
from nirs4all.sklearn import NIRSPipeline

# Load model as sklearn-compatible wrapper
model = NIRSPipeline.from_bundle("my_model.n4a")

# Use with SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
```

### sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from nirs4all.sklearn import NIRSPipeline

# Create sklearn pipeline with nirs4all model
model = NIRSPipeline.from_bundle("my_model.n4a")
sklearn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])

# Fit and predict
sklearn_pipeline.fit(X_train, y_train)
predictions = sklearn_pipeline.predict(X_test)
```

---

## Deployment Best Practices

### 1. Always Validate Before Deployment

```python
import nirs4all

# Load and predict with the exported bundle
preds = nirs4all.predict("my_model.n4a", test_dataset)

# Verify against training results
assert np.allclose(preds[:5], reference_preds[:5])
```

### 2. Document Model Requirements

```python
# Export with descriptive naming
result.export("sugar_model_snv_pls.n4a")

# Training metadata is automatically included in the bundle:
# - Pipeline configuration
# - Training metrics (RMSE, R2)
# - Dataset information
# - Library versions
```

### 3. Version Your Models

```python
# Use descriptive versioned file names for your exports
result.export("sugar_model_v2.1.0.n4a")
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
