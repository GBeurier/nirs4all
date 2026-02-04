# Exporting Models

After training a pipeline, you can export trained models for deployment, sharing, or archival. nirs4all supports several export formats, from self-contained bundles to standalone Python scripts.

## Export Best Model

The simplest export path -- export the best model from a training run:

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
)

# Export best model as .n4a bundle
result.export("best_model.n4a")
```

This finds the best prediction (ranked by validation score), locates its chain in the workspace store, and packages the chain and all its artifacts into a `.n4a` ZIP file.

## Export a Specific Model

If you want to export a model other than the best, pass a `chain_id`:

```python
# Export a specific chain by ID
result.export("specific_model.n4a", chain_id="abc123-def456")
```

Or use a prediction entry from `result.top()`:

```python
# Get top 5, export the 3rd best
top5 = result.top(5)
result.export("third_best.n4a", source=top5[2])
```

## Export Formats

| Format | Extension | Use Case | Contains |
|--------|-----------|----------|----------|
| Bundle | `.n4a` | Standard deployment and sharing | Chain definition + all fitted artifacts in a ZIP |
| Python script | `.n4a.py` | Standalone prediction without nirs4all | Embedded base64 artifacts, runs independently |
| Pipeline config | `.json` | Re-run the same pipeline configuration | Expanded pipeline definition (no fitted artifacts) |
| Run metadata | `.yaml` | Archival and provenance tracking | Full run description with all pipelines and metrics |

### Bundle (.n4a)

The default and most common format. A `.n4a` file is a ZIP archive containing everything needed to reproduce predictions:

```python
result.export("model.n4a")                       # Default format
result.export("model.n4a", format="n4a")          # Explicit
```

### Python Script (.n4a.py)

A portable Python script with embedded artifacts. Runs without nirs4all installed:

```python
result.export("model.n4a.py", format="n4a.py")
```

Usage:

```bash
python model.n4a.py input_spectra.csv
```

### Pipeline Config (.json)

Export the pipeline configuration for re-running with different data:

```python
from pathlib import Path
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

store = WorkspaceStore(Path("workspace"))
store.export_pipeline_config("pipeline_id", Path("config.json"))
```

### Run Metadata (.yaml)

Export full run metadata for archival:

```python
store.export_run("run_id", Path("run_archive.yaml"))
```

## Bundle Anatomy

A `.n4a` bundle is a ZIP file with the following structure:

```
model.n4a (ZIP)
    manifest.json           # Bundle metadata
    chain.json              # Chain definition (steps, fold artifacts, shared artifacts)
    artifacts/
        abc123def456.joblib # Fitted model (fold 0)
        bcd234efg567.joblib # Fitted model (fold 1)
        cde345fgh678.joblib # Fitted scaler (shared)
```

### manifest.json

Contains the chain ID, model class, preprocessing summary, fold strategy, and export timestamp:

```json
{
    "chain_id": "abc123-def456-...",
    "model_class": "sklearn.cross_decomposition.PLSRegression",
    "model_step_idx": 2,
    "preprocessings": "MinMaxScaler",
    "fold_strategy": "per_fold",
    "exported_at": "2025-01-15T10:30:00+00:00"
}
```

### chain.json

Defines the ordered steps and maps fold/step indices to artifact filenames:

```json
{
    "steps": [
        {"step_idx": 0, "operator_class": "MinMaxScaler", "params": {}, "stateless": false},
        {"step_idx": 1, "operator_class": "PLSRegression", "params": {"n_components": 10}, "stateless": false}
    ],
    "model_step_idx": 1,
    "fold_artifacts": {
        "fold_0": "art_abc123",
        "fold_1": "art_bcd234"
    },
    "shared_artifacts": {
        "0": "art_cde345"
    }
}
```

## Sharing Models

`.n4a` bundles are self-contained and portable. To share a model:

1. Export the model: `result.export("model.n4a")`
2. Copy the `.n4a` file to the target machine
3. Predict on the target machine:

```python
import nirs4all

preds = nirs4all.predict(model="model.n4a", data=X_new)
```

No workspace, no DuckDB store, no artifacts directory needed on the target machine. The bundle contains everything.

## Loading Bundles as sklearn Pipelines

For integration with sklearn-compatible tools (SHAP, cross-validation, grid search), load a bundle as a `NIRSPipeline`:

```python
from nirs4all.sklearn import NIRSPipeline

# Load bundle as sklearn-compatible pipeline
model = NIRSPipeline.from_bundle("model.n4a")

# Use like any sklearn estimator
y_pred = model.predict(X_new)

# Works with SHAP
import shap
explainer = shap.Explainer(model, X_background)
shap_values = explainer(X_test)
```

For classification bundles, use `NIRSPipelineClassifier`:

```python
from nirs4all.sklearn import NIRSPipelineClassifier

model = NIRSPipelineClassifier.from_bundle("classifier.n4a")
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)
```

### Selecting a Fold

By default, `from_bundle` loads the first fold's model. To use a specific fold:

```python
model = NIRSPipeline.from_bundle("model.n4a", fold=2)
```

## Export from the Store

For programmatic exports (e.g., in scripts or the webapp), use the `WorkspaceStore` directly:

```python
from pathlib import Path
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

store = WorkspaceStore(Path("workspace"))

# Export a specific chain as bundle
store.export_chain("chain_id", Path("exports/model.n4a"))

# Export pipeline config
store.export_pipeline_config("pipeline_id", Path("exports/config.json"))

# Export full run metadata
store.export_run("run_id", Path("exports/run.yaml"))

# Export filtered predictions as Parquet
store.export_predictions_parquet(
    Path("exports/wheat_results.parquet"),
    dataset_name="wheat",
)

store.close()
```

## See Also

- [Making Predictions](making_predictions.md) -- How to predict from an exported bundle
- [Advanced Predictions](advanced_predictions.md) -- Transfer learning and retraining from bundles
- [Understanding Predictions](understanding_predictions.md) -- What chains are and how they work
