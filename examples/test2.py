#!/usr/bin/env python3
"""
Q4 Example - Model Persistence and Prediction Methods
====================================================
Demonstrates model training, saving, and three different prediction methods:
1. Prediction with a prediction entry
2. Prediction with a model ID
Shows how to reuse trained models for new data.
"""

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Build the pipeline with feature augmentation and model persistence
pipeline = [
    # Data preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},

    # Feature augmentation with preprocessing combinations
    {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()]},

    # Cross-validation setup
    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    # RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),

    # Machine learning models
    {"model": PLSRegression(10), "name": "Q4_PLS_10"},
    {"model": PLSRegression(20), "name": "Q4_PLS_20"},
    {"model": GradientBoostingRegressor(n_estimators=20)},
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/regression_2'])

# Run pipeline with model saving enabled
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get best performing model for prediction testing
best_prediction = predictions.top_k(1, partition="test")[0]
model_id = best_prediction['id']
fold_id = best_prediction['fold_id']

print("=== Q4 - Model Persistence and Prediction Example ===")
print("--- Source Model ---")
print(f"Best model: {best_prediction['model_name']} (id: {model_id})")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print("Reference predictions:", reference_predictions)
print("-" * 80)

# Method 1: Predict using a prediction entry
print("--- Method 1: Predict with a prediction entry ---")

predictor = PipelineRunner()
prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression_2/Xtest.csv'})

# Make predictions using the best prediction entry
method1_predictions, _ = predictor.predict(best_prediction, prediction_dataset, verbose=0)
method1_array = method1_predictions[:5].flatten()
print("Method 1 predictions:", method1_array)
is_identical = np.allclose(method1_array, reference_predictions)
print(f"Method 1 identical to training: {'✅ YES' if is_identical else '❌ NO'}")

print("=" * 80)
