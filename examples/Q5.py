#!/usr/bin/env python3
import os
os.environ['DISABLE_EMOJIS'] = '1'

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
import shutil
from pathlib import Path
import numpy as np
from nirs4all.operators.models.cirad_tf import nicon
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},
    {
        "feature_augmentation":
        {
            "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
            "size": [(2, 3), (1, 3)],
            "count": 5
        }
    },
    ShuffleSplit(n_splits=3, test_size=.25, random_state=42),
    {"model": PLSRegression(10), "name": "Q5_PLS"},  # Added a name for easier identification
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"model": RandomForestRegressor(n_estimators=20)},
    {
        "model": nicon,
        "train_params": {
            "epochs": 100,
            "patience": 50,
            "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
        },
    },
]

pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/multi'])
runner = PipelineRunner(save_files=True, verbose=0)
run_predictions, _ = runner.run(pipeline_config, dataset_config)


# Get top models to verify the real model names are displayed correctly
best_count = 5
rank_metric = 'rmse'  # 'rmse', 'mae', 'r2'
top_n = run_predictions.top_k(best_count, rank_metric)
print(f"Top {best_count} models by {rank_metric}:")
for i, pred in enumerate(top_n):
    print(f"{i+1}. {Predictions.pred_short_string(pred, metrics=[rank_metric])} - {pred['preprocessings']}")

# TAB REPORT
analyzer = PredictionAnalyzer(run_predictions)  # Prétraitements dans le graphique
fig1 = analyzer.plot_top_k_comparison(k=best_count, metric='rmse')
# plt.savefig('test_top_k_models_Q1.png', dpi=150, bbox_inches='tight')

# TAB REPORT
fig2 = analyzer.plot_variable_heatmap(
    x_var="model_name",
    y_var="preprocessings",
)
plt.show()

# Best train model
reference_prediction = run_predictions.top_k(1, partition="test")[0]
prediction_id = reference_prediction['id']

print("=== Q5 - Example ===")
print("--- Source Model ---")
print(f"Best model: {reference_prediction['model_name']} (id: {prediction_id})")
reference_array = reference_prediction['y_pred'][:5].flatten()
print("Y reference:", reference_array)
print("-" * 120)
# ####################################################################

# Test prediction methods
print("=" * 120)
print("--- Predict with a model ID ---")
# Rebuild a pipeline runner and load dataset
predictor = PipelineRunner(save_files=False, verbose=0)  # No need to save files here
dataset_config = DatasetConfigs(['sample_data/multi'])
## Predict with the reference prediction entry
reference_id = reference_prediction['id']
print(f"Using model ID: [{reference_id}] in {reference_prediction['config_name']}")
method2_prediction, _ = predictor.predict(reference_id, dataset_config, verbose=0)
method2_array = method2_prediction[:5].flatten()
print("Y:", method2_array)
identical = np.allclose(method2_array, reference_array)
print(f"Method 2 identical to training: {'✅ YES' if identical else '❌ NO'}")
####################################################################
