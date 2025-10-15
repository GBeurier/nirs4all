"""
Q1 Example - Basic Regression Pipeline with PLS Models
=====================================================
Demonstrates NIRS regression analysis using PLS models with various preprocessing techniques.
Features automated hyperparameter tuning for n_components and comprehensive result visualization.
"""

# Standard library imports
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Configuration variables
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/regression'


# Build the pipeline
pipeline = [
    feature_scaler,
    {"y_processing": target_scaler},
    {"feature_augmentation": {"_or_": [Detrend, FirstDerivative, Gaussian, SavitzkyGolay, Haar], "size": 2, "count": 2}},  # Generate combinations of preprocessing techniques
    "chart_2d",
    cross_validation,

]

# Add PLS models with different numbers of components
for n_components in range(1, 30, 10):
    model_config = {
        "name": f"PLS-{n_components}_components",
        "model": PLSRegression(n_components=n_components)
    }
    pipeline.append(model_config)

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1")
dataset_config = DatasetConfigs(data_path)


# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models
top_models = predictions.top(best_model_count, ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")
top_models[0].save_to_csv("Q1_regression_best_model.csv")

# # Create visualizations
analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k_comparison(k=best_model_count, rank_metric='rmse')
fig2 = analyzer.plot_top_k_comparison(k=best_model_count, rank_metric='rmse', rank_partition='test')
fig3 = analyzer.plot_top_k_comparison(k=best_model_count, rank_metric='rmse', rank_partition='train')

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
