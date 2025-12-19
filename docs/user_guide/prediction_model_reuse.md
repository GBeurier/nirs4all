# Prediction and Model Reuse

This guide covers how to make predictions using trained models in nirs4all, including loading saved models and applying them to new datasets.

## Overview

After training a pipeline, you can use the trained models to make predictions on new data. nirs4all supports several prediction workflows:

1. **Direct prediction**: Use a trained model to predict new samples
2. **Model persistence**: Save and reload models for later use
3. **Cross-validation ensembles**: Combine predictions from multiple CV folds

## Basic Prediction Workflow

### Training a Model

First, train your pipeline:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Define pipeline
pipeline = [
    MinMaxScaler(),
    RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
]

# Train
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(
    PipelineConfigs(pipeline),
    DatasetConfigs(['path/to/training_data'])
)

# Get best model
best_prediction = predictions.top(n=1, rank_partition="test")[0]
print(f"Best model: {best_prediction['model_name']}")
print(f"RMSE: {best_prediction['rmse']:.4f}")
```

### Making Predictions

Use the `predict()` method with the best prediction entry:

```python
# Create predictor
predictor = PipelineRunner(save_files=False, verbose=0)

# Load new data
new_dataset = DatasetConfigs({
    'X_test': 'path/to/new_spectra.csv'
})

# Make predictions
y_pred, _ = predictor.predict(best_prediction, new_dataset, verbose=0)
print(f"Predictions: {y_pred[:5]}")
```

## Prediction Sources

The `predict()` method accepts various sources:

### 1. Prediction Dictionary (Most Common)

```python
# From Predictions object
best_prediction = predictions.top(n=1, rank_partition="test")[0]
y_pred, _ = runner.predict(best_prediction, new_data)
```

### 2. Model ID String

```python
# Using the prediction ID directly
model_id = best_prediction['id']
y_pred, _ = runner.predict(model_id, new_data)
```

### 3. Folder Path

```python
# From a pipeline folder
y_pred, _ = runner.predict("runs/2024-12-14_wheat/pipeline_abc123/", new_data)
```

### 4. Bundle File

```python
# From an exported bundle (see Export section)
y_pred, _ = runner.predict("exports/wheat_model.n4a", new_data)
```

### 5. Direct Model File

You can load a model directly from its binary file. This is useful when you have a pre-trained model saved externally or want to use models trained outside nirs4all.

```python
# From a sklearn/joblib model file
y_pred, _ = runner.predict("models/pls_wheat.joblib", new_data)

# From a pickle file
y_pred, _ = runner.predict("models/my_model.pkl", new_data)

# From a TensorFlow/Keras model
y_pred, _ = runner.predict("models/nn_model.h5", new_data)
y_pred, _ = runner.predict("models/nn_model.keras", new_data)

# From a PyTorch model
y_pred, _ = runner.predict("models/torch_model.pt", new_data)
y_pred, _ = runner.predict("models/checkpoint.pth", new_data)

# From a model folder (AutoGluon, TensorFlow SavedModel)
y_pred, _ = runner.predict("models/autogluon_model/", new_data)
y_pred, _ = runner.predict("models/tf_savedmodel/", new_data)
```

**Supported formats:**

| Extension | Framework | Notes |
|-----------|-----------|-------|
| `.joblib` | sklearn, XGBoost, LightGBM | Recommended for sklearn models |
| `.pkl` | Any (cloudpickle) | General purpose |
| `.h5`, `.hdf5` | TensorFlow/Keras | Legacy Keras format |
| `.keras` | TensorFlow/Keras | Modern Keras format |
| `.pt`, `.pth` | PyTorch | Full model or state dict |
| `.ckpt` | PyTorch | Checkpoint file |
| folder | AutoGluon, TensorFlow | SavedModel format |

**Important:** When using direct model files, no preprocessing artifacts are loaded. The input data should already be preprocessed appropriately for the model.

## Prediction Output

The `predict()` method returns:

```python
y_pred, predictions_obj = runner.predict(source, dataset)
```

- `y_pred`: numpy array of predictions (averaged across folds if CV)
- `predictions_obj`: Predictions object with full metadata

### Getting All Predictions

To get predictions from all models (not just the best):

```python
all_preds, predictions_obj = runner.predict(
    best_prediction,
    new_data,
    all_predictions=True,
    verbose=0
)

# Iterate over all predictions
for pred in predictions_obj.to_dicts():
    print(f"{pred['model_name']}: {pred['rmse']:.4f}")
```

## Cross-Validation Ensemble Predictions

When training with cross-validation, each fold produces a separate model. During prediction, nirs4all automatically:

1. Loads all fold models
2. Makes predictions with each
3. Combines predictions using weighted averaging

The weights are determined by validation performance:

```python
# Fold weights are stored in the prediction
fold_weights = best_prediction.get('fold_weights', {})
print(f"Fold weights: {fold_weights}")
# e.g., {0: 0.34, 1: 0.33, 2: 0.33}
```

## Data Format for Prediction

The new data must have the same number of features as the training data:

```python
# Check expected features
n_features = best_prediction['n_features']
print(f"Expected features: {n_features}")

# Supported formats
# 1. CSV file
new_data = DatasetConfigs({'X_test': 'spectra.csv'})

# 2. NumPy array
import numpy as np
X_new = np.random.randn(20, n_features)
new_data = DatasetConfigs({'test_x': X_new})

# 3. Dictionary
new_data = {'test_x': X_new}
```

## Preprocessing Replay

During prediction, nirs4all automatically replays the preprocessing steps:

1. Loads saved transformer artifacts (scalers, SNV, etc.)
2. Applies transforms in the same order as training
3. Feeds transformed data to the model

This ensures consistent preprocessing between training and prediction.

## Error Handling

Common prediction errors and solutions:

### Missing Model

```python
try:
    y_pred, _ = runner.predict(best_prediction, new_data)
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    # Re-train or check save_files=True during training
```

### Feature Mismatch

```python
try:
    y_pred, _ = runner.predict(best_prediction, new_data)
except ValueError as e:
    print(f"Feature mismatch: {e}")
    # Check new data has same number of features as training
```

### Missing Preprocessing Artifacts

```python
try:
    y_pred, _ = runner.predict(best_prediction, new_data)
except KeyError as e:
    print(f"Missing artifact: {e}")
    # Ensure all preprocessing steps were saved during training
```

## Using Model Files in Pipelines

You can include pre-trained models directly in your pipeline configuration. This is useful for:
- Transfer learning with pre-trained models
- Ensemble with external models
- Fine-tuning existing models

### Loading a Model in Pipeline

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# Use a pre-trained model file in pipeline
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": "models/pretrained_pls.joblib", "name": "pretrained_pls"}
]

runner = PipelineRunner()
predictions, _ = runner.run(pipeline, dataset)
```

### Supported Model Formats in Pipelines

The model path is resolved automatically:

```python
# sklearn/scikit-learn models
{"model": "models/pls.joblib"}
{"model": "models/ridge.pkl"}

# TensorFlow/Keras
{"model": "models/nn.h5"}
{"model": "models/nn.keras"}
{"model": "models/savedmodel_folder/"}  # SavedModel format

# PyTorch
{"model": "models/torch_model.pt"}
{"model": "models/checkpoint.pth"}
{"model": "models/checkpoint.ckpt"}

# AutoGluon
{"model": "models/autogluon_predictor/"}
```

### Fine-tuning a Pre-trained Model

```python
# Load and fine-tune a pre-trained model
runner = PipelineRunner()
predictions, _ = runner.retrain(
    source="models/pretrained_pls.joblib",
    dataset=new_training_data,
    mode='finetune'
)
```

### Transfer Learning

```python
# Use a model trained on one dataset for another
runner = PipelineRunner()
predictions, _ = runner.retrain(
    source="models/wheat_model.joblib",
    dataset=corn_dataset,
    mode='transfer'
)
```

## Best Practices

1. **Always use `save_files=True`** during training to persist models
2. **Verify feature dimensions** before prediction
3. **Use the same preprocessing** as training (automatic with nirs4all)
4. **Store prediction entries** for reproducibility
5. **Test predictions** against known validation data
6. **Match preprocessing** when using direct model files - no preprocessing is replayed

## See Also

- [Export and Bundles](export_bundles.md) - Export models for deployment
- [Retrain and Transfer](retrain_transfer.md) - Retrain models on new data
- [Migration Guide](migration_guide.md) - Upgrade from older versions
