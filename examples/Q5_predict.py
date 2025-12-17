#!/usr/bin/env python3
"""
Q4 Example - Model Persistence and Prediction Methods
====================================================
Demonstrates model training, saving, and three different prediction methods:
1. Prediction with a prediction entry
2. Prediction with a model ID
Shows how to reuse trained models for new data.
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.models.tensorflow.nicon import nicon

# Simple status symbols
CHECK = "[OK]"
CROSS = "[X]"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q5 Predict Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Build the pipeline with feature augmentation and model persistence
pipeline = [
    # Data preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},

    # Feature augmentation with preprocessing combinations
    {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()]},

    # Cross-validation setup
    RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),

    # Machine learning models
    {"model": PLSRegression(10), "name": "Q5_PLS_10"},
    # {"model": PLSRegression(20), "name": "Q4_PLS_20"},
    # {"model": GradientBoostingRegressor(n_estimators=20)},
    # nicon
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/regression'])

# Run pipeline with model saving enabled
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get best performing model for prediction testing
best_prediction = predictions.top(n=1, rank_partition="test")[0]
model_id = best_prediction['id']
fold_id = best_prediction['fold_id']
model_name = best_prediction['model_name']
step_idx = best_prediction['step_idx']
op_counter = best_prediction['op_counter']

print("=== Q5 - Model Persistence and Prediction Example ===")
print("--- Source Model ---")
print(f"Best model: {model_name} (id: {model_id}, fold: {fold_id})")
print(f"Dataset: {best_prediction['dataset_name']}, Partition: {best_prediction['partition']}")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print("Reference predictions:", reference_predictions)
print("-" * 80)

# Method 1: Predict using a prediction entry
print("--- Method 1: Predict with a prediction entry ---")

predictor = PipelineRunner()
prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})

# Make predictions using the best prediction entry
method1_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)
method1_array = method1_predictions[:5].flatten()
print("Method 1 predictions:", method1_array)
is_identical = np.allclose(method1_array, reference_predictions)
assert is_identical, "Method 1 predictions do not match reference!"
print(f"Method 1 identical to training: {f'{CHECK}YES' if is_identical else f'{CROSS}NO'}")

print("=" * 80)

# Method 2: Predict using a model ID
print("--- Method 2: Predict with a model ID ---")
predictor2 = PipelineRunner(save_files=False, verbose=0)
prediction_dataset2 = DatasetConfigs({
    'X_test': 'sample_data/regression/Xval.csv.gz',  # Same dataset as Method 1
})

print(f"Using model ID: [{model_id}]")
method2_predictions, _ = predictor2.predict(model_id, prediction_dataset2, verbose=0)
method2_array = method2_predictions[:5].flatten()
print("Method 2 predictions:", method2_array)
is_identical = np.allclose(method2_array, reference_predictions)
assert is_identical, "Method 2 predictions do not match reference!"
print(f"Method 2 identical to training: {f'{CHECK}YES' if is_identical else f'{CROSS}NO'}")

# Method 3: Predict using a model ID (same as Method 2, but showing the Predictions object return)
print("--- Method 3: Predict with a model ID and access full prediction metadata ---")
predictor3 = PipelineRunner(save_files=False, verbose=0)
prediction_dataset3 = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})
method3_predictions, method3_preds_obj = predictor3.predict(model_id, prediction_dataset3, all_predictions=True, verbose=0)

# When training uses aggregated folds (avg/w_avg), prediction mode only creates individual fold predictions.
# We need to find matching predictions by model_name, step_idx, and partition, then aggregate if needed.
test_preds = [
    pred for pred in method3_preds_obj.to_dicts()
    if pred['model_name'] == model_name and pred['step_idx'] == step_idx and pred['partition'] == 'test'
]

if test_preds:
    import json
    # Aggregate predictions from all folds (simple average for demonstration)
    all_fold_preds = []
    for pred in test_preds:
        pred_list = json.loads(pred['y_pred']) if isinstance(pred['y_pred'], str) else pred['y_pred']
        all_fold_preds.append(np.array(pred_list, dtype=float))

    # Compute average across folds (simple average - note: training uses weighted average)
    method3_full = np.mean(all_fold_preds, axis=0)
    method3_array = method3_full[:5].flatten()
    print(f"Method 3 predictions (averaged from {len(test_preds)} folds):", method3_array)
    # Use rtol=0.05 since simple avg differs slightly from weighted avg
    is_close = np.allclose(method3_array, reference_predictions, rtol=0.05)
    assert is_close, "Method 3 predictions differ too much from reference!"
    print(f"Method 3 close to training (rtol=0.05): {f'{CHECK}YES' if is_close else f'{CROSS}NO'}")
else:
    raise ValueError("No matching predictions found for Method 3")
