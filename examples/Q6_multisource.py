#!/usr/bin/env python3
"""
Q5 Example - Multi-Source Regression with Model Reuse
====================================================
Demonstrates multi-source regression analysis using various models including
PLS, Random Forest, and neural networks. Shows model persistence and reuse.
"""

# Standard library imports
import os
import numpy as np
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
# from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.utils.emoji import CHECK, CROSS
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.linear_model import ElasticNet

# Disable emojis in output
os.environ['DISABLE_EMOJIS'] = '1'


# Build the pipeline for multi-target regression
pipeline = [
    # Data preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},

    # Feature augmentation with preprocessing combinations
    {
        "feature_augmentation": {
            "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
            "size": [(2, 3), (1, 3)],
            "count": 5
        }
    },

    # Cross-validation setup
    ShuffleSplit(n_splits=3),
    "fold_chart",

    # Machine learning models
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"model": PLSRegression(10), "name": "Q6_PLS_3"},
    # {"model": RandomForestRegressor(n_estimators=2)},
    ElasticNet(alpha=0.1, l1_ratio=0.5),
    {"model": PLSRegression(10), "name": "Q6_PLS_2"},

    # # Neural network model
    # {
    #     "model": nicon,
    #     "train_params": {
    #         "epochs": 100,
    #         "patience": 50,
    #         "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
    #     },
    # },
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/multi'])

# Run pipeline with model saving enabled
runner = PipelineRunner(save_files=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'

# Display top performing models
top_models = predictions.top_k(best_model_count, ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")

# Create visualizations
analyzer = PredictionAnalyzer(predictions)

# Plot comparison of top models
# fig1 = analyzer.plot_top_k_comparison(k=best_model_count, rank_metric='rmse')

# Plot heatmap: models vs preprocessing using NEW v2 method
# This properly ranks on val and displays test scores
fig2 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    rank_partition="val",
    display_metric="rmse",
    display_partition="test",
    aggregation='best'  # Options: 'best', 'mean', 'median'
)

# plt.show()

# Model reuse demonstration
best_prediction = predictions.top_k(1, partition="test")[0]
model_id = best_prediction['id']

print("\n=== Q6 - Multisource Model + Reuse Example ===")
print("--- Source Model ---")
print(f"Best model: {best_prediction['model_name']} (id: {model_id})")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print("Reference predictions:", reference_predictions)
print("-" * 80)

# Test model reuse with same dataset
print("=" * 80)
print("--- Predict with saved model ID ---")
predictor = PipelineRunner(save_files=False, verbose=0)
test_dataset_config = DatasetConfigs(['sample_data/multi'])

print(f"Using model ID: [{model_id}] in {best_prediction['config_name']}")
reuse_predictions, _ = predictor.predict(model_id, test_dataset_config, verbose=0)
reuse_array = reuse_predictions[:5].flatten()
print("Reuse predictions:", reuse_array)
is_identical = np.allclose(reuse_array, reference_predictions)
print(f"Model reuse identical to training: {f'{CHECK}YES' if is_identical else f'{CROSS}NO'}")

# # Create visualizations
analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k_comparison(k=best_model_count, rank_metric='rmse')

# Plot heatmap of model performance vs preprocessing
fig2 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="preprocessings",
    aggregation='best',  # Options: 'best', 'mean', 'median'
    rank_metric="rmse",
    rank_partition="val",
    display_metric="rmse",
    display_partition="test"
)

# Plot simplified heatmap without count display
fig3 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="preprocessings",
    aggregation='best',  # Show average instead of best
    show_counts=False,
    rank_metric="rmse",
    rank_partition="test",
    display_metric="rmse",
    display_partition="test"
)

# Plot candlestick chart for model performance distribution
fig4 = analyzer.plot_variable_candlestick(
    filters={"partition": "test"},
    variable="model_name",
)

fig5 = analyzer.plot_score_histogram(partition="test")

# plt.show()