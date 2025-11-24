"""
Q3 Example - Multi-Dataset Analysis with Feature Augmentation
============================================================
Demonstrates regression analysis across multiple datasets with comprehensive
preprocessing combinations and neural network models.
"""

# Standard library imports
import argparse
from matplotlib import pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.models.tensorflow.nicon import nicon
from nirs4all.operators.transforms import (
    Gaussian, SavitzkyGolay, StandardNormalVariate, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.utils.emoji import REFRESH

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q4 Multi-Datasets Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# Build the pipeline with feature augmentation
pipeline = [
    # Data preprocessing
    MinMaxScaler(feature_range=(0.1, 0.8)),

    # Feature augmentation with preprocessing combinations
    {"feature_augmentation": [
        MultiplicativeScatterCorrection,
        Gaussian,
        StandardNormalVariate,
        SavitzkyGolay,
        Haar,
        [MultiplicativeScatterCorrection, Gaussian],
        [MultiplicativeScatterCorrection, StandardNormalVariate],
        [MultiplicativeScatterCorrection, SavitzkyGolay],
        [MultiplicativeScatterCorrection, Haar],
    ]},
    "chart_2d",
    "y_chart",

    # Cross-validation and target processing
    ShuffleSplit(n_splits=3),
    {"y_processing": MinMaxScaler},

    # Machine learning models
    {"model": PLSRegression(15)},
    {"model": ElasticNet()},


    # Neural network model with enhanced training parameters
    # {
    #     "model": nicon,
    #     "train_params": {
    #         "epochs": 50,
    #         "patience": 50,
    #         "batch_size": 500,
    #         "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
    #     },
    # },
]



# Create pipeline configuration
pipeline_config = PipelineConfigs(pipeline, name="Q3")

# Multi-dataset configuration
data_paths = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']

dataset_config = DatasetConfigs(data_paths)



# Run the pipeline across multiple datasets
runner = PipelineRunner(save_files=True, verbose=0)
print(f"{REFRESH}Running pipeline with spinner enabled - watch for loading animations during model training!")
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analyze results per dataset
for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"Dataset: name={dataset_name}, number of predictions={len(dataset_prediction['run_predictions'])}")

    # Get the Predictions object from the dataset_prediction dictionary
    dataset_predictions = dataset_prediction['run_predictions']
    top_models = dataset_predictions.top(n=4, rank_metric='rmse')
    print("Top 4 models by RMSE:")
    for idx, model in enumerate(top_models):
        print(f"{idx+1}. {Predictions.pred_long_string(model, metrics=['rmse', 'r2', 'mae'])}")

    # Plot comparison for this dataset
    analyzer = PredictionAnalyzer(dataset_predictions)
    fig = analyzer.plot_top_k(k=5, rank_metric='rmse')

# Overall analysis across all datasets
analyzer = PredictionAnalyzer(predictions)

# Plot heatmap: models vs datasets
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
)

# Plot candlestick chart for model performance distribution
fig3 = analyzer.plot_candlestick(
    variable="model_name",
    display_partition="test"
)

if args.show:
    plt.show()
