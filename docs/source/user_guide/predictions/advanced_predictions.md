# Advanced Predictions

This guide covers advanced prediction workflows: transfer learning, retraining, chain replay internals, SHAP explanations, and batch prediction patterns.

## Transfer Learning

Transfer learning reuses a trained model's preprocessing pipeline on new data, optionally replacing or fine-tuning the model itself. Use `nirs4all.retrain()` with the appropriate mode.

### Retrain Modes

| Mode | What Happens | When to Use |
|------|-------------|-------------|
| `"full"` | Retrain everything from scratch using the same pipeline structure | New domain with similar spectral features |
| `"transfer"` | Keep fitted preprocessing, train a new model on top | Same instrument, different product |
| `"finetune"` | Continue training the existing model with additional epochs | Neural networks, incremental learning |

### Full Retrain

Rebuilds the entire pipeline on new data. The pipeline structure is preserved, but all steps are re-fitted:

```python
import nirs4all

# Original training
original = nirs4all.run(pipeline, train_data)

# Full retrain on new data
retrained = nirs4all.retrain(
    source=original.best,
    data=new_train_data,
    mode="full",
)

print(f"Original RMSE: {original.best_rmse:.4f}")
print(f"Retrained RMSE: {retrained.best_rmse:.4f}")
```

### Transfer Mode

Keeps the fitted preprocessing transformers (scalers, SNV, etc.) and trains only the model step on new data:

```python
result = nirs4all.retrain(
    source="exports/wheat_model.n4a",
    data=corn_data,
    mode="transfer",
)
```

You can also replace the model entirely while keeping preprocessing:

```python
from sklearn.ensemble import RandomForestRegressor

result = nirs4all.retrain(
    source="exports/pls_model.n4a",
    data=new_data,
    mode="transfer",
    new_model=RandomForestRegressor(n_estimators=100),
)
```

### Fine-tuning

Continues training an existing model. Primarily useful for neural networks:

```python
result = nirs4all.retrain(
    source="exports/nn_model.n4a",
    data=new_data,
    mode="finetune",
    epochs=10,
    learning_rate=0.0001,
)
```

### Retrain from a Bundle

You can retrain from any `.n4a` bundle, not just from a `RunResult`:

```python
result = nirs4all.retrain(
    source="exports/wheat_model.n4a",
    data="new_wheat_data/",
    mode="full",
    verbose=2,
)

# Export the retrained model
result.export("exports/retrained_wheat.n4a")
```

### Retrain Output

`nirs4all.retrain()` returns a `RunResult`, just like `nirs4all.run()`. You can use all the same accessors:

```python
result = nirs4all.retrain(source, data, mode="full")

result.best_rmse         # Best RMSE of retrained model
result.top(5)            # Top 5 retrained predictions
result.export("new.n4a") # Export retrained model
```

## Chain Replay Internals

When you predict from a stored chain (via `chain_id` or from a bundle), nirs4all replays the chain step by step. Understanding this process helps diagnose issues and set expectations.

### Replay Process

1. **Load chain metadata** -- read the step definitions, fold artifacts, and shared artifacts
2. **For each preprocessing step** (in order):
   - If the step has a shared artifact: load the fitted transformer and call `transform(X)`
   - If the step is stateless (no artifact): skip it (e.g., a stateless SNV)
3. **At the model step**:
   - Load each fold's fitted model artifact
   - Call `model.predict(X_preprocessed)` for each fold
   - Average the fold predictions element-wise
4. **Return** the averaged predictions as a numpy array

```
Chain Replay:
    X_input
      |
      v
    [Step 0: MinMaxScaler] -- load shared artifact --> transform(X)
      |
      v
    [Step 1: SNV]          -- stateless            --> skip
      |
      v
    [Step 2: PLS model]   -- load fold artifacts   --> predict per fold --> average
      |
      v
    y_pred
```

### Store-Based Replay

For in-workspace prediction via chain ID:

```python
from pathlib import Path
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

store = WorkspaceStore(Path("workspace"))

# Replay a chain on new data
y_pred = store.replay_chain(chain_id="abc123", X=X_new)

store.close()
```

The `replay_chain()` method handles artifact loading, step execution, and fold averaging in a single call.

### Bundle-Based Replay

When predicting from a `.n4a` bundle, the same replay logic is used, but artifacts are loaded from the ZIP file instead of the workspace store. The `BundleLoader` extracts the chain definition and artifacts, then replays them identically.

## SHAP Explanations

`nirs4all.explain()` computes SHAP values for a trained model, revealing which features (wavelengths) contribute most to predictions.

### Basic Usage

```python
import nirs4all

explanation = nirs4all.explain(
    model="exports/wheat_model.n4a",
    data=X_test,
)

# Feature importance ranking
importance = explanation.get_feature_importance(top_n=10)
for feature, value in importance.items():
    print(f"{feature}: {value:.4f}")

# Top features (sorted by importance)
print(f"Top 5 features: {explanation.top_features[:5]}")
```

### From a RunResult

```python
result = nirs4all.run(pipeline, dataset)

explanation = nirs4all.explain(
    model=result.best,
    data=X_test,
)
```

### Explainer Types

nirs4all automatically selects the best SHAP explainer for your model, but you can override:

```python
# Tree-based models (Random Forest, XGBoost, LightGBM)
explanation = nirs4all.explain(
    model="model.n4a", data=X_test,
    explainer_type="tree",
)

# Linear models (PLS, Ridge, Lasso)
explanation = nirs4all.explain(
    model="model.n4a", data=X_test,
    explainer_type="linear",
)

# Any model (slower but universal)
explanation = nirs4all.explain(
    model="model.n4a", data=X_test,
    explainer_type="kernel",
)
```

### ExplainResult

The `ExplainResult` object provides:

```python
explanation.values           # Raw SHAP values (numpy array)
explanation.shape            # Shape: (n_samples, n_features)
explanation.mean_abs_shap    # Mean |SHAP| per feature
explanation.top_features     # Feature names sorted by importance
explanation.base_value       # Baseline prediction
explanation.visualizations   # Paths to generated plot files

# Per-sample explanation
sample_shap = explanation.get_sample_explanation(0)

# As DataFrame
df = explanation.to_dataframe()
df.to_csv("shap_values.csv")
```

### Using NIRSPipeline with SHAP Directly

For full control over SHAP computation, load the model as an sklearn pipeline:

```python
import shap
from nirs4all.sklearn import NIRSPipeline

model = NIRSPipeline.from_bundle("model.n4a")

explainer = shap.Explainer(model, X_background)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

## Batch Prediction Patterns

### Multiple Datasets

Predict across several datasets using the same model:

```python
import nirs4all
import numpy as np

datasets = {
    "site_A": np.load("site_a_spectra.npy"),
    "site_B": np.load("site_b_spectra.npy"),
    "site_C": np.load("site_c_spectra.npy"),
}

results = {}
for name, X in datasets.items():
    preds = nirs4all.predict(model="model.n4a", data=X)
    results[name] = preds.values
    print(f"{name}: {len(preds)} predictions")
```

### Multiple Models on Same Data

Compare predictions from different models on the same data:

```python
models = [
    "exports/pls_model.n4a",
    "exports/rf_model.n4a",
    "exports/svr_model.n4a",
]

for model_path in models:
    preds = nirs4all.predict(model=model_path, data=X_test)
    print(f"{preds.model_name}: mean={preds.values.mean():.4f}")
```

### Session-Based Batch Prediction

For repeated predictions with the same model, use a session to avoid re-loading:

```python
import nirs4all

with nirs4all.session(verbose=0) as s:
    # Train once
    result = nirs4all.run(pipeline, train_data, session=s)

    # Predict multiple datasets
    for data_path in ["data_jan/", "data_feb/", "data_mar/"]:
        preds = nirs4all.predict(model=result.best, data=data_path, session=s)
        print(f"{data_path}: {preds.shape}")
```

### Handling Different Wavelength Ranges

When datasets from different instruments have slightly different wavelength ranges, preprocess them to align before prediction:

```python
from nirs4all.operators.transforms import ResampleTransformer

# Resample to match training wavelengths
resampler = ResampleTransformer(target_wavelengths=training_wavelengths)
X_aligned = resampler.fit_transform(X_new)

preds = nirs4all.predict(model="model.n4a", data=X_aligned)
```

## See Also

- [Making Predictions](making_predictions.md) -- Basic prediction workflows
- [Exporting Models](exporting_models.md) -- Export formats and bundle structure
- [Analyzing Results](analyzing_results.md) -- Querying and visualizing results
- [Understanding Predictions](understanding_predictions.md) -- Core concepts
