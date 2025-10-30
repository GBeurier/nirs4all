"""
Q2.5 Finetune Example - Hyperparameter Optimization with PLS Regression
==================================================================
Demonstrates automated hyperparameter tuning for PLS regression models using Optuna.
Includes preprocessing augmentation and comprehensive visualization of results.
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.models.tensorflow.nicon import nicon, customizable_nicon

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q3 Finetune Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Configuration variables
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/regression'

# Build the pipeline with hyperparameter optimization
pipeline = [
    "chart_2d",
    feature_scaler,
    {"y_processing": target_scaler},
    {"feature_augmentation": {"_or_": preprocessing_options, "size": [1, (1, 2)], "count": 5}},  # Generate preprocessing combinations
    cross_validation,
    {
        "model": PLSRegression(),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 10,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "grid",                       # "random", "grid", "bayes", "hyperband", "skopt", "tpe", "cmaes"
            "model_params": {
                'n_components': ('int', 1, 10),
            },
        }
    },
    {
        "model": customizable_nicon,
        "name": "PLS-Default",
        "finetune_params": {
            "n_trials": 10,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_2": [8, 16, 32, 64],
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 5,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 25,
            "verbose": 0
        }
    }
]

# Add standard PLS models for comparison
for n_components in range(1, 30, 3):
    model_config = {
        "name": f"PLS-{n_components}_components",
        "model": PLSRegression(n_components=n_components)
    }
    pipeline.append(model_config)

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1_finetune")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline with hyperparameter optimization
runner = PipelineRunner(save_files=False, verbose=0)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models (including finetuned ones)
top_models = predictions.top(n=best_model_count, rank_metric=ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")

# Create visualizations
analyzer = PredictionAnalyzer(predictions)

# Plot comparison of top models
fig1 = analyzer.plot_top_k(k=best_model_count, rank_metric='rmse')

# Plot heatmap of model performance vs preprocessing
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric='rmse'
)

# Plot simplified heatmap without count display
fig3 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric='rmse',
    show_counts=False
)

# Plot candlestick chart for model performance distribution
fig4 = analyzer.plot_candlestick(
    variable="model_name",
    partition="test"
)

if args.show:
    plt.show()
