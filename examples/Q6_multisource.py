#!/usr/bin/env python3
"""
Q5 Example - Multi-Source Regression with Model Reuse
====================================================
Demonstrates multi-source regression analysis using various models including
PLS, Random Forest, and neural networks. Shows model persistence and reuse.
"""

# Standard library imports
import argparse
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
# from nirs4all.operators.models.tensorflow.nicon import nicon
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
# Simple status symbols
CHECK = "[OK]"
CROSS = "[X]"
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.linear_model import ElasticNet

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q6 Multi-Source Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

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
            "pick": [2, 3],
            "then_pick": [1, 3],
            "count": 2
        }
    },

    # Cross-validation setup
    ShuffleSplit(n_splits=3),
    "fold_chart",

    # Machine learning models
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"model": PLSRegression(10), "name": "Q6_PLS_3"},
    # {"model": RandomForestRegressor(n_estimators=2)},
    # ElasticNet(alpha=0.1, l1_ratio=0.5),
    # {"model": PLSRegression(10), "name": "Q6_PLS_2"},

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
runner = PipelineRunner(save_artifacts=True, verbose=0)
predictions, _ = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'

# Display top performing models
top_models = predictions.top(n=best_model_count, rank_metric=ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")

# # Create visualizations
# analyzer = PredictionAnalyzer(predictions)

# # Plot comparison of top models
# # fig1 = analyzer.plot_top_k(k=best_model_count, rank_metric='rmse')

# # Plot heatmap: models vs preprocessing using NEW v2 method
# # This properly ranks on val and displays test scores
# fig2 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="preprocessings",
#     rank_metric="rmse",
#     rank_partition="val",
#     display_metric="rmse",
#     display_partition="test",
#     aggregation='best'  # Options: 'best', 'mean', 'median'
# )

# if args.show:
#     plt.show()

# Model reuse demonstration
best_prediction = predictions.top(n=1, rank_partition="test")[0]
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
predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
test_dataset_config = DatasetConfigs({'X_test': ['sample_data/multi/Xval_1.csv.gz', 'sample_data/multi/Xval_2.csv.gz', 'sample_data/multi/Xval_3.csv.gz']})

print(f"Using model ID: [{model_id}] in {best_prediction['config_name']}")
reuse_predictions, _ = predictor.predict(model_id, test_dataset_config, verbose=2)
reuse_array = reuse_predictions[:5].flatten()
print("Reuse predictions:", reuse_array)
is_identical = np.allclose(reuse_array, reference_predictions)
assert is_identical, "Method predictions do not match reference!"
print(f"Model reuse identical to training: {f'{CHECK}YES' if is_identical else f'{CROSS}NO'}")

# # # Create visualizations
# analyzer = PredictionAnalyzer(predictions)
# # Plot comparison of top models
# fig1 = analyzer.plot_top_k(k=best_model_count, rank_metric='rmse')

# # Plot heatmap of model performance vs preprocessing
# fig2 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="preprocessings",
#     aggregation='best',  # Options: 'best', 'mean', 'median'
#     rank_metric="rmse",
#     rank_partition="val",
#     display_metric="rmse",
#     display_partition="test"
# )

# # Plot simplified heatmap without count display
# fig3 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="preprocessings",
#     aggregation='best',  # Show average instead of best
#     show_counts=False,
#     rank_metric="rmse",
#     rank_partition="test",
#     display_metric="rmse",
#     display_partition="test"
# )

# # Plot candlestick chart for model performance distribution
# fig4 = analyzer.plot_candlestick(
#     variable="model_name",
#     partition="test"
# )

# fig5 = analyzer.plot_histogram(partition="test")

# if args.show:
#     plt.show()
