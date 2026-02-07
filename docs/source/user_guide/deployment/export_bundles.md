# Export and Deployment

This guide covers exporting trained pipelines as standalone bundles for deployment, sharing, or archival.

## Overview

The bundle export feature allows you to package a trained pipeline into a self-contained file that can be:

- **Deployed** without the full nirs4all workspace
- **Shared** with colleagues or clients
- **Archived** for reproducibility years later
- **Run** on edge devices or in containers

## Bundle Formats

nirs4all supports two bundle formats:

### `.n4a` Format (Full Bundle)

A ZIP archive containing:
- `manifest.json`: Bundle metadata (chain_id, model_class, preprocessings, etc.)
- `chain.json`: Chain steps with fold and shared artifact references
- `artifacts/`: Directory with serialized model and transformer binaries

**Best for**: Deployment with nirs4all, archival, full reproducibility

### `.n4a.py` Format (Portable Script)

A single Python file with:
- Embedded artifacts (base64 encoded)
- Standalone `predict()` function
- No nirs4all dependency (only numpy, scipy, scikit-learn, joblib required)

**Best for**: Lightweight deployment, edge devices, sharing with non-nirs4all users

:::{warning}
**Model Compatibility**: The `.n4a.py` portable script only supports **sklearn-compatible models** (PLSRegression, RandomForest, SVM, XGBoost, etc.). Deep learning models (PyTorch, JAX, TensorFlow) require the full `.n4a` bundle format and the corresponding framework installed.
:::

## Model Compatibility

Not all models work with all export formats:

| Model Type | `.n4a` Bundle | `.n4a.py` Script |
|------------|---------------|------------------|
| **sklearn** (PLS, SVM, RF, etc.) | ✅ Full support | ✅ Full support |
| **sklearn transformers** (MinMaxScaler, SNV, etc.) | ✅ Full support | ✅ Full support |
| **XGBoost / LightGBM / CatBoost** | ✅ Full support | ⚠️ Requires library |
| **PyTorch models** | ✅ Full support | ❌ Not supported |
| **JAX models** | ✅ Full support | ❌ Not supported |
| **TensorFlow / Keras** | ✅ Full support | ❌ Not supported |

### Why Deep Learning Models Don't Work with `.n4a.py`

1. **State dict vs full model**: PyTorch/JAX save state dicts, not executable models
2. **Model class required**: You need the Python class definition to reconstruct the model
3. **Framework dependency**: These models require their full framework at runtime
4. **Device management**: GPU/CPU placement logic is framework-specific

For deep learning models, always use the `.n4a` bundle format and ensure the target environment has nirs4all plus the required framework (torch, jax, tensorflow) installed.

## Pipeline Feature Support

The `.n4a` bundle format supports all nirs4all pipeline features:

| Feature | `.n4a` Bundle | `.n4a.py` Script |
|---------|---------------|------------------|
| **Linear pipelines** | ✅ Full support | ✅ Full support |
| **CV ensemble (fold averaging)** | ✅ Full support | ✅ Full support |
| **Weighted fold averaging** | ✅ Full support | ✅ Full support |
| **Branching pipelines** | ✅ Full support | ❌ Not supported |
| **Stacking (meta-models)** | ✅ Full support | ❌ Not supported |
| **Nested branches** | ✅ Full support | ❌ Not supported |

The `.n4a` bundle stores the complete execution trace, enabling accurate replay of complex pipelines including branches and stacking configurations.

## Basic Export

### Export to .n4a Bundle

```python
import nirs4all

# Train your pipeline
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
)

# Export best model to .n4a bundle
result.export("exports/wheat_model.n4a")

# Export a specific prediction's model
result.export("exports/specific_model.n4a", prediction_id="abc123")
```

### Export to Portable Script

```python
# Export to portable Python script
result.export("exports/wheat_predictor.n4a.py", format="n4a.py")
```

## Export Options

The `result.export()` method accepts the following parameters:

```python
result.export(
    path="exports/model.n4a",          # Output path
    prediction_id=None,                 # Specific prediction (default: best)
    format="n4a",                       # Format: "n4a" or "n4a.py"
)
```

**Parameters:**
- `path`: Path for the output bundle file
- `prediction_id`: Export a specific prediction's chain (default: best model)
- `format`: Bundle format, either `"n4a"` (default) or `"n4a.py"`

### Store-Level Export

For advanced export operations, use `WorkspaceStore` directly:

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Export a chain by ID
store.export_chain(chain_id, Path("model.n4a"))

# Export pipeline config as JSON
store.export_pipeline_config(pipeline_id, Path("config.json"))

# Export run metadata as YAML
store.export_run(run_id, Path("run_summary.yaml"))

# Export filtered predictions as Parquet
store.export_predictions_parquet(Path("results.parquet"), dataset_name="wheat")
```

## Lightweight Artifact Access

For cases where you need direct access to individual artifacts from a chain:

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Load a specific artifact by ID
artifact = store.load_artifact(artifact_id)

# Get the filesystem path of an artifact
path = store.get_artifact_path(artifact_id)
```

## Using Exported Bundles

### Predict from .n4a Bundle

```python
import nirs4all

# Predict directly from bundle (no workspace needed)
preds = nirs4all.predict("exports/wheat_model.n4a", new_data)
```

### Use Portable Script

The `.n4a.py` script can be run standalone:

```bash
# Command line usage
python wheat_predictor.n4a.py input_spectra.csv output_predictions.csv
```

Or imported in Python:

```python
# Import the script
import wheat_predictor

import numpy as np
X_new = np.load("new_spectra.npy")

# Make predictions
y_pred = wheat_predictor.predict(X_new)
print(f"Predictions: {y_pred}")
```

### Direct BundleLoader Usage

For advanced use cases, load bundles directly with `BundleLoader`:

```python
from nirs4all.pipeline.bundle import BundleLoader

loader = BundleLoader("exports/model.n4a")

# Access metadata
print(f"Pipeline UID: {loader.metadata.pipeline_uid}")
print(f"Model step: {loader.metadata.model_step_index}")
print(f"Preprocessing: {loader.metadata.preprocessing_chain}")
print(f"Fold weights: {loader.fold_weights}")

# Make predictions
import numpy as np
X_new = np.random.randn(10, 100)
y_pred = loader.predict(X_new)
```

## Export Sources

Export works from `RunResult` or directly from `WorkspaceStore`:

```python
# From RunResult (most common)
result.export("model.n4a")

# From store chain ID
store.export_chain("chain_abc123", Path("model.n4a"))

# From bundle path (re-export)
preds = nirs4all.predict("old_model.n4a", data)
```

## Bundle Inspection

### Check Bundle Contents

```python
import zipfile
import json

with zipfile.ZipFile("exports/model.n4a", 'r') as zf:
    # List files
    print("Bundle contents:")
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f"  {name}: {info.file_size / 1024:.1f} KB")

    # Read manifest
    manifest = json.loads(zf.read('manifest.json'))
    print(f"\nChain ID: {manifest['chain_id']}")
    print(f"Model: {manifest['model_class']}")
    print(f"Preprocessing: {manifest['preprocessings']}")
    print(f"Exported at: {manifest['exported_at']}")
```

## Batch Export

Export multiple models from a training run:

```python
# Get top 3 predictions
top_preds = result.top(3)

# Export each
for i, row in enumerate(top_preds.iter_rows(named=True), 1):
    pred_id = row["prediction_id"]
    model = row["model_class"].split(".")[-1]
    result.export(f"exports/rank_{i}_{model}.n4a", prediction_id=pred_id)
```

## Deployment Patterns

### Docker Container

```dockerfile
FROM python:3.11-slim

# Install minimal dependencies
RUN pip install numpy joblib scikit-learn scipy

# Copy portable script
COPY wheat_predictor.n4a.py /app/predict.py

WORKDIR /app
ENTRYPOINT ["python", "predict.py"]
```

### AWS Lambda

```python
import json
import numpy as np
import wheat_predictor  # The .n4a.py script

def lambda_handler(event, context):
    # Parse input spectra
    X = np.array(event['spectra'])

    # Make prediction
    y_pred = wheat_predictor.predict(X)

    return {
        'statusCode': 200,
        'body': json.dumps({'predictions': y_pred.tolist()})
    }
```

### FastAPI Service

```python
from fastapi import FastAPI
import numpy as np
import wheat_predictor

app = FastAPI()

@app.post("/predict")
async def predict(spectra: list[list[float]]):
    X = np.array(spectra)
    y_pred = wheat_predictor.predict(X)
    return {"predictions": y_pred.tolist()}
```

## Retrain from Bundle

Bundles can be used as a starting point for retraining on new data:

```python
import nirs4all

# Retrain from exported bundle
result = nirs4all.retrain("exports/wheat_model.n4a", new_data, mode="transfer")
```

**Retrain modes:**
- `'full'`: Train everything from scratch (same pipeline structure)
- `'transfer'`: Use existing preprocessing artifacts, train new model
- `'finetune'`: Continue training existing model (deep learning only)

## Best Practices

1. **Version bundles**: Include version info in filenames
2. **Test before deployment**: Verify predictions match training validation
3. **Document dependencies**: Note required Python version and packages
4. **Store source metadata**: Keep the original prediction ID for traceability
5. **Use compression**: Enable compression for smaller bundle sizes

## Troubleshooting

### Bundle Too Large

- Use compression: `compress=True`
- Use `export_model()` for lightweight model-only export
- Consider `.n4a.py` for simpler models

### Portable Script Errors

- Ensure numpy, scipy, scikit-learn, and joblib are installed
- Check Python version compatibility (3.9+)
- Verify input data shape matches expected features
- **For deep learning models**: Use `.n4a` format instead

### Prediction Mismatch

- Verify same nirs4all version for export and load
- Check preprocessing is correctly replayed
- Compare fold weights between training and bundle

## See Also

- {doc}`/examples/index` - Examples including deployment workflows
- {doc}`/user_guide/pipelines/stacking` - Stacking with MetaModel for advanced ensembles
- {doc}`/reference/pipeline_syntax` - Pipeline configuration syntax
- {doc}`/getting_started/index` - Quick start guide

**Example files:**
- `examples/user/06_deployment/U22_export_bundle.py` - Complete export examples
- `examples/user/06_deployment/U21_save_load_predict.py` - Basic prediction workflows

```{seealso}
**Related Examples:**
- [U02: Export Bundle](../../../examples/user/06_deployment/U02_export_bundle.py) - Export to .n4a and .n4a.py bundles
- [U01: Save, Load, Predict](../../../examples/user/06_deployment/U01_save_load_predict.py) - Basic prediction workflow
```
