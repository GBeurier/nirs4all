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
- `manifest.json`: Bundle metadata and version info
- `pipeline.json`: Minimal pipeline configuration
- `trace.json`: Execution trace for deterministic replay
- `artifacts/`: Directory with model and transformer binaries
- `fold_weights.json`: CV fold weights (if applicable)

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
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(save_artifacts=True, verbose=0)

# Train your pipeline
predictions, _ = runner.run(pipeline_config, dataset_config)
best_pred = predictions.top(n=1, rank_partition="test")[0]

# Export to .n4a bundle
bundle_path = runner.export(
    source=best_pred,
    output_path="exports/wheat_model.n4a",
    format="n4a"
)
print(f"Bundle created: {bundle_path}")
```

### Export to Portable Script

```python
# Export to portable Python script
script_path = runner.export(
    source=best_pred,
    output_path="exports/wheat_predictor.n4a.py",
    format="n4a.py"
)
print(f"Script created: {script_path}")
```

## Export Options

The `export()` method accepts the following parameters:

```python
runner.export(
    source=best_pred,                  # Prediction source (dict, path, or model_id)
    output_path="exports/model.n4a",   # Output path
    format="n4a",                       # Format: "n4a" or "n4a.py"
    include_metadata=True,              # Include full metadata in bundle
    compress=True                       # Compress artifacts (n4a only)
)
```

**Parameters:**
- `source`: Prediction source - accepts prediction dict, folder path, or model ID
- `output_path`: Path for the output bundle file
- `format`: Bundle format, either `"n4a"` (default) or `"n4a.py"`
- `include_metadata`: Whether to include full metadata in the bundle (default: `True`)
- `compress`: Whether to compress artifacts using ZIP deflate (default: `True`)

## Lightweight Model Export

For cases where you only need the model binary (without preprocessing artifacts), use `export_model()`:

```python
# Export just the model artifact
model_path = runner.export_model(
    source=best_pred,
    output_path="exports/pls_model.joblib",
    fold=0  # Optional: specific fold (default: fold 0)
)

# Later, use in new pipeline
new_pipeline = [
    MinMaxScaler(),
    {"model": "exports/pls_model.joblib"}  # Load model from path
]
```

**Supported formats:**
- `.joblib`: Joblib serialization (recommended for sklearn)
- `.pkl`: Pickle serialization
- `.h5`, `.keras`: Keras models
- `.pt`: PyTorch models

## Using Exported Bundles

### Predict from .n4a Bundle

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.data import DatasetConfigs

runner = PipelineRunner()
new_data = DatasetConfigs({'X_test': 'path/to/new_spectra.csv'})

# Predict directly from bundle
y_pred, _ = runner.predict("exports/wheat_model.n4a", new_data)
print(f"Predictions: {y_pred[:5]}")
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

Export accepts the same source types as predict:

```python
# From prediction dict
runner.export(best_prediction, "model.n4a")

# From folder path (pipeline run directory)
runner.export("runs/2024-12-14_wheat/pipeline_abc123/", "model.n4a")

# From bundle path (re-export)
runner.export("old_model.n4a", "model_v2.n4a")
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
    print(f"\nPipeline UID: {manifest['pipeline_uid']}")
    print(f"Created: {manifest['created_at']}")
    print(f"Preprocessing: {manifest['preprocessing_chain']}")
```

## Batch Export

Export multiple models from a training run:

```python
# Get top 3 models
top_models = predictions.top(n=3, rank_partition="test")

# Export each
for i, pred in enumerate(top_models, 1):
    model_name = pred['model_name'].replace(" ", "_").lower()
    runner.export(pred, f"exports/rank_{i}_{model_name}.n4a")
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
# Load bundle and retrain on new data
retrained_preds, _ = runner.retrain(
    source="exports/wheat_model.n4a",
    dataset=new_training_data,
    mode='transfer',  # Use existing preprocessing, train new model
    dataset_name='new_calibration'
)
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
- {doc}`stacking` - Stacking with MetaModel for advanced ensembles
- {doc}`/reference/pipeline_syntax` - Pipeline configuration syntax
- {doc}`/getting_started/index` - Quick start guide

**Example files:**
- `examples/user/06_deployment/U22_export_bundle.py` - Complete export examples
- `examples/user/06_deployment/U21_save_load_predict.py` - Basic prediction workflows
