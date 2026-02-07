# Making Predictions

This guide covers all the ways to predict on new data with trained nirs4all models. Whether you have a `RunResult` from a training session, an exported `.n4a` bundle, or a chain ID in your workspace, the `nirs4all.predict()` function handles them all.

## From a RunResult (Most Common)

After training a pipeline, the simplest path is to export the best model and predict from the bundle:

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# Train
result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
)

# Export best model
result.export("best_model.n4a")

# Predict on new data
preds = nirs4all.predict(model="best_model.n4a", data=X_new)
print(preds.values)  # numpy array of predictions
```

You can also predict directly from a prediction dictionary (the output of `result.best` or `result.top()`):

```python
# Predict using the best prediction entry
preds = nirs4all.predict(model=result.best, data=X_new)
```

## From an Exported Bundle (.n4a)

A `.n4a` bundle is a self-contained ZIP file that includes the chain definition and all fitted artifacts. It can be shared between machines without requiring the original workspace.

```python
import nirs4all

# Predict from a bundle file
preds = nirs4all.predict(model="exports/wheat_model.n4a", data=X_new)

print(f"Predictions shape: {preds.shape}")
print(f"Model: {preds.model_name}")
print(f"Values: {preds.values[:5]}")
```

Bundles are portable -- copy the `.n4a` file to another machine and predict without any workspace setup.

## From a Chain ID

If you are working within a workspace that has a DuckDB store, you can predict directly from a stored chain without exporting first:

```python
import nirs4all

# Predict using a chain ID from the workspace store
preds = nirs4all.predict(
    chain_id="abc123-def456",
    data=X_new,
    workspace_path="workspace",
)
```

This path replays the chain directly from the store, loading artifacts from the `artifacts/` directory. It avoids the overhead of exporting to a bundle first.

```{note}
The `model` and `chain_id` parameters are mutually exclusive. Provide one or the other, not both.
```

## From a Standalone Python Script (.n4a.py)

You can export a model as a self-contained Python script that embeds all artifacts. This script runs without nirs4all installed:

```python
# Export as standalone script
result.export("model.n4a.py", format="n4a.py")
```

Then run it from the command line:

```bash
python model.n4a.py input_spectra.csv
```

The script reads the input CSV, applies all preprocessing steps, runs the model, and prints predictions to stdout.

## Data Format Requirements

The `data` parameter accepts several formats:

| Format | Example | Notes |
|--------|---------|-------|
| NumPy array | `X_new` (shape: n_samples x n_features) | Most direct; features must match training |
| Tuple | `(X,)` or `(X, y)` | y is optional, used for evaluation |
| Dict | `{"X": X, "metadata": meta}` | For chain-based prediction |
| Path (string) | `"new_data/"` | Folder parsed by dataset loaders |
| SpectroDataset | `SpectroDataset(...)` | Direct dataset object |

### Feature Count

The number of features (columns) in the new data must match the number of features seen during training. If the training data had 256 wavelengths, the new data must also have 256 columns.

```python
# Check expected feature count from training
best = result.best
print(f"Expected features: {best.get('n_features')}")
```

### Wavelength Alignment

For spectroscopic data, the wavelengths of the new data should align with the training data. If you are using wavelength-aware operators (e.g., `CropTransformer`, `ResampleTransformer`), the wavelength arrays must be compatible.

## Cross-Validation Ensemble Averaging

When a model was trained with cross-validation (e.g., 5-fold CV), the chain contains 5 fitted model artifacts -- one per fold. During prediction, nirs4all:

1. Loads all fold models from the chain
2. Applies the shared preprocessing steps to `X_new`
3. Runs `model.predict(X_preprocessed)` for each fold model
4. Averages the fold predictions element-wise

This ensemble averaging typically improves prediction stability compared to using a single fold's model.

```
X_new --> Preprocessing (shared) --> fold_0 model --> y_pred_0
                                 --> fold_1 model --> y_pred_1  --> mean --> y_pred
                                 --> fold_2 model --> y_pred_2
```

## Preprocessing Replay

During prediction, the chain is replayed step by step:

1. Each preprocessing step loads its saved artifact (fitted scaler, transformer, etc.)
2. The step's `transform()` method is called on the data (never `fit()` -- the artifacts are already fitted)
3. The transformed data is passed to the next step
4. At the model step, `predict()` is called instead of `transform()`

This guarantees that the same preprocessing is applied to new data as was applied during training. You do not need to manually apply preprocessing -- it is handled automatically by the chain.

## Error Handling

### Feature Mismatch

If the new data has a different number of features than the training data:

```python
try:
    preds = nirs4all.predict(model="model.n4a", data=X_wrong_shape)
except ValueError as e:
    print(f"Feature mismatch: {e}")
    # Fix: ensure X_new has the correct number of columns
```

### Missing Bundle File

```python
try:
    preds = nirs4all.predict(model="missing_model.n4a", data=X_new)
except FileNotFoundError as e:
    print(f"Bundle not found: {e}")
    # Fix: check the file path or re-export the model
```

### Corrupt or Incomplete Bundle

If a bundle is missing artifact files:

```python
try:
    preds = nirs4all.predict(model="corrupt.n4a", data=X_new)
except (KeyError, FileNotFoundError) as e:
    print(f"Missing artifact: {e}")
    # Fix: re-export the model from the workspace
```

### Invalid Arguments

```python
# Both model and chain_id provided
try:
    preds = nirs4all.predict(model="model.n4a", chain_id="abc", data=X_new)
except ValueError as e:
    print(f"Error: {e}")
    # Fix: provide either model or chain_id, not both

# Neither model nor chain_id provided
try:
    preds = nirs4all.predict(data=X_new)
except ValueError as e:
    print(f"Error: {e}")
    # Fix: provide either model or chain_id
```

## PredictResult Output

`nirs4all.predict()` returns a `PredictResult` object:

```python
preds = nirs4all.predict(model="model.n4a", data=X_new)

# Access predictions
preds.values          # numpy array (alias for y_pred)
preds.y_pred          # numpy array of predicted values
preds.shape           # shape of prediction array
preds.model_name      # name of the model used

# Convert to other formats
preds.to_numpy()      # numpy array
preds.to_list()       # Python list
preds.to_dataframe()  # pandas DataFrame

# Check properties
len(preds)            # number of predictions
preds.is_multioutput  # True if multi-output prediction
preds.flatten()       # flattened 1D array
```

### PredictResult with Evaluation

When you provide both `X` and `y` (as a tuple), you can evaluate the predictions against ground truth:

```python
preds = nirs4all.predict(model="model.n4a", data=(X_test, y_test))

# The predicted values
print(preds.values)

# Additional metadata
print(preds.metadata)
```

## Complete Example

```python
import nirs4all
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Define pipeline with cross-validation
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

# Train
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    verbose=1,
)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R2:   {result.best_r2:.4f}")

# Export
result.export("wheat_model.n4a")

# Predict on new data
X_new = np.random.randn(20, result.best.get("n_features", 100))
preds = nirs4all.predict(model="wheat_model.n4a", data=X_new)

print(f"Predictions: {preds.shape}")
print(f"First 5: {preds.values[:5]}")

# Convert to DataFrame
df = preds.to_dataframe()
print(df.head())
```

## See Also

- [Understanding Predictions](understanding_predictions.md) -- Core concepts
- [Exporting Models](exporting_models.md) -- Export formats and bundle anatomy
- [Advanced Predictions](advanced_predictions.md) -- Transfer learning and batch prediction

```{seealso}
**Related Examples:**
- [U01: Save, Load, Predict](../../../examples/user/06_deployment/U01_save_load_predict.py) - Basic prediction workflow
- [U02: Export Bundle](../../../examples/user/06_deployment/U02_export_bundle.py) - Export and load models for prediction
```
