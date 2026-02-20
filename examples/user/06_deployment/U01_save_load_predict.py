"""
U01 - Save, Load, and Predict: Model Persistence
=================================================

Save trained models and use them for prediction on new data.

This tutorial covers:

* Automatic model saving with PipelineRunner
* Prediction with prediction entries
* Prediction with model IDs
* Verifying prediction consistency

Prerequisites
-------------
Complete the model training examples first.

Next Steps
----------
See :ref:`U02_export_bundle` for portable model export.

Duration: ~4 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 Save Load Predict Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Why Model Persistence?
# =============================================================================
print("\n" + "=" * 60)
print("U01 - Save, Load, and Predict")
print("=" * 60)

print("""
Model persistence allows you to:

  📊 SAVE TRAINED MODELS
     - Automatic saving with save_artifacts=True
     - Models stored in workspace/runs/
     - Includes preprocessing pipeline

  📈 LOAD AND PREDICT
     - Predict on new data with trained models
     - Use prediction entries or model IDs
     - Full preprocessing pipeline applied

  📉 USE CASES
     ✓ Deploy models to production
     ✓ Batch prediction on new samples
     ✓ Share models with colleagues
     ✓ Compare predictions over time
""")

# =============================================================================
# Section 2: Training with Model Saving
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Training with Model Saving")
print("-" * 60)

print("""
Set save_artifacts=True to save trained models.
""")

# Define pipeline
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    FirstDerivative(),

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "U21_SaveLoad")
dataset_config = DatasetConfigs("sample_data/regression")

# Run with saving enabled
runner = PipelineRunner(
    save_artifacts=True,  # <-- Key setting!
    verbose=1
)
predictions, _run_info = runner.run(pipeline_config, dataset_config)

# Get best model
top_results = predictions.top(n=1, rank_partition="test")
assert isinstance(top_results, list)
best_prediction = top_results[0]
model_id = best_prediction['id']
model_name = best_prediction['model_name']

print(f"\nBest model: {model_name}")
print(f"Model ID: {model_id}")
test_mse = best_prediction.get('test_mse', best_prediction.get('mse'))
print(f"Test MSE: {test_mse:.4f}" if test_mse is not None else "Test MSE: (see detailed metrics)")

# Store reference predictions for verification
reference_predictions = best_prediction['y_pred'][:5].flatten()
print(f"Reference predictions (first 5): {reference_predictions}")

# =============================================================================
# Section 3: Prediction with Prediction Entry
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Prediction with Prediction Entry")
print("-" * 60)

print("""
Method 1: Use the prediction entry directly.
The entry contains all info needed to reconstruct the model.
""")

# Create predictor
predictor = PipelineRunner()

# Dataset for prediction (only X needed)
prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})

# Predict using the prediction entry
new_predictions_result, _ = predictor.predict(dict(best_prediction), prediction_dataset, verbose=0)
assert isinstance(new_predictions_result, np.ndarray)
new_predictions = new_predictions_result

print(f"New predictions shape: {new_predictions.shape}")
print(f"New predictions (first 5): {new_predictions[:5].flatten()}")

# Verify consistency
is_identical = np.allclose(new_predictions[:5].flatten(), reference_predictions)
print(f"Identical to training: {'YES ✓' if is_identical else 'NO ✗'}")

# =============================================================================
# Section 4: Prediction with Model ID
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Prediction with Model ID")
print("-" * 60)

print("""
Method 2: Use just the model ID string.
The system looks up the model from the artifact storage.
""")

# Create new predictor
predictor2 = PipelineRunner(save_artifacts=False, verbose=0)

# Predict using model ID
print(f"Using model ID: {model_id}")
id_predictions_result, _ = predictor2.predict(model_id, prediction_dataset, verbose=0)
assert isinstance(id_predictions_result, np.ndarray)
id_predictions = id_predictions_result

print(f"Predictions (first 5): {id_predictions[:5].flatten()}")

# Verify consistency
is_identical = np.allclose(id_predictions[:5].flatten(), reference_predictions)
print(f"Identical to training: {'YES ✓' if is_identical else 'NO ✗'}")

# =============================================================================
# Section 5: Prediction on New Data
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Prediction on New Data")
print("-" * 60)

print("""
Predict on completely new data (not in training set).
The full preprocessing pipeline is applied automatically.
""")

# Create synthetic "new" data
np.random.seed(123)
n_new = 10
n_features = 2151  # Must match training data feature count
X_new = np.random.randn(n_new, n_features)

# Predict using tuple input
new_data = DatasetConfigs({'X_test': X_new})
synthetic_predictions_result, _ = predictor.predict(model_id, new_data, verbose=0)
assert isinstance(synthetic_predictions_result, np.ndarray)
synthetic_predictions = synthetic_predictions_result

print(f"New data shape: {X_new.shape}")
print(f"Predictions shape: {synthetic_predictions.shape}")
print(f"Predictions: {synthetic_predictions.flatten()}")

# =============================================================================
# Section 6: Accessing Full Prediction Metadata
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Accessing Full Prediction Metadata")
print("-" * 60)

print("""
Use all_predictions=True to get full Predictions object.
""")

predictor3 = PipelineRunner(save_artifacts=False, verbose=0)
full_predictions, preds_obj = predictor3.predict(
    model_id,
    prediction_dataset,
    all_predictions=True,  # Get full metadata
    verbose=0
)

print(f"Number of prediction entries: {preds_obj.num_predictions}")

# Access prediction metadata via top()
top_preds = preds_obj.top(1)
assert isinstance(top_preds, list)
if top_preds:
    pred_entry = top_preds[0]
    print("\nPrediction metadata:")
    print(f"   Model: {pred_entry.get('model_name', 'N/A')}")
    print(f"   Preprocessings: {pred_entry.get('preprocessings', 'N/A')}")
    print(f"   Partition: {pred_entry.get('partition', 'N/A')}")
else:
    print("\nPrediction complete successfully")

# =============================================================================
# Section 7: Multiple Model Predictions
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Multiple Model Predictions")
print("-" * 60)

print("""
Compare predictions from different models.
""")

# Get top 3 models
top_models = predictions.top(3)
assert isinstance(top_models, list)

print("Comparing top 3 models on new data:")
for i, pred_entry in enumerate(top_models, 1):
    model_predictions_result, _ = predictor.predict(dict(pred_entry), prediction_dataset, verbose=0)
    assert isinstance(model_predictions_result, np.ndarray)
    print(f"   {i}. {pred_entry['model_name']}: {model_predictions_result[:3].flatten()}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Model Persistence Workflow:

  1. TRAIN WITH SAVING:
     runner = PipelineRunner(save_artifacts=True)
     predictions, _ = runner.run(pipeline_config, dataset_config)

  2. GET MODEL INFO:
     best = predictions.top(1)[0]
     model_id = best['id']

  3. PREDICT - Method A (Prediction Entry):
     predictor = PipelineRunner()
     new_preds, _ = predictor.predict(best_prediction, new_dataset)

  4. PREDICT - Method B (Model ID):
     new_preds, _ = predictor.predict(model_id, new_dataset)

  5. FULL METADATA:
     preds, preds_obj = predictor.predict(model_id, new_dataset,
                                          all_predictions=True)

Data Input Options:
  DatasetConfigs({'X_test': 'path/to/file.csv'})  # File path
  DatasetConfigs({'X_test': X_array})              # NumPy array

Model Storage:
  - Models saved in: workspace/runs/<run_id>/
  - Includes: preprocessing pipeline + trained model
  - Format: pickle/joblib

Next: U02_export_bundle.py - Export portable model bundles
""")
