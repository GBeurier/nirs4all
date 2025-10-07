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
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs, Predictions
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Build the pipeline with feature augmentation and model persistence
pipeline = [
    # Data preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},

    # Feature augmentation with preprocessing combinations
    {
        "feature_augmentation": {
            "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
            "size": [(5, 6), (1, 3)],
            "count": 1
        },
    },

    # # Cross-validation setup
    # RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),
    # # Machine learning models
    # {"model": PLSRegression(10), "name": "PLS_1"},
    # {"model": PLSRegression(20), "name": "PLS_2"},
    # # {"model": GradientBoostingRegressor(n_estimators=20)},
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/regression'])

# Run pipeline with model saving enabled
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Predictions.save_all_to_csv(predictions, path="test_", aggregate_partitions=True)

print("-" * 80)
print("#### ALL PREDICTIONS ####")
for pred in predictions.to_dicts():
    print(f"(id: {pred['id']}, {pred['model_name']}, {pred['metric']}: {pred['test_score']}) : {pred['val_score']}) : {pred['train_score']}), {pred['partition']} - {pred['fold_id']})")

print("-" * 80)
print("#### BEST PREDICTIONS ####")
best_prediction = predictions.top(1, aggregate_partitions=True)
for pred in best_prediction:
    print(pred.eval_score())
    print(f"(id: {pred['id']}, {pred['model_name']}, {pred['metric']}: {pred['test_score']}) : {pred['val_score']}) : {pred['train_score']}), {pred['partition']} - fold: {pred['fold_id']}")
    # print("Test RMSE in pred['test']['rmse']:", pred['test']['rmse'])
print("-" * 80)
