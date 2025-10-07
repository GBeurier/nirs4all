"""
Q2 Example - Multi-Model Comparison Pipeline
===========================================
Demonstrates comparison of various regression models including PLS, Random Forest,
Elastic Net, SVR, MLP, Gradient Boosting, and neural networks on NIRS data.
"""

# Standard library imports
from matplotlib import pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.operators.transformations import MultiplicativeScatterCorrection
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Build the pipeline with multiple model types
pipeline = [
    # Data preprocessing
    MinMaxScaler(feature_range=(0.1, 0.8)),
    MultiplicativeScatterCorrection,
    "3d_chart",
    "2d_chart",
    # Cross-validation setup
    ShuffleSplit(n_splits=3),
    "fold_chart",
    {"y_processing": MinMaxScaler},

    # Machine learning models
    {"model": PLSRegression(15)},
    {"model": PLSRegression(10)},
    {"model": RandomForestRegressor(n_estimators=100)},
    {"model": ElasticNet()},
    {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1), "name": "SVR_Custom_Model"},
    {"model": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500), "name": "MLP_Custom_Model"},
    {"model": GradientBoostingRegressor(n_estimators=100)},

    # Neural network model
    {
        "model": nicon,
        "train_params": {
            "epochs": 5,
            "patience": 50,
            "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
        },
    },
]

# Create pipeline configuration
pipeline_config = PipelineConfigs(pipeline, name="Q2")

# Dataset configuration
data_paths = ['sample_data/regression']
dataset_config = DatasetConfigs(data_paths)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=0, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
best_model_count = 3
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models
print(f"\nTop {best_model_count} models by {ranking_metric}:")
top_models = predictions.top_k(best_model_count, ranking_metric)
for idx, model in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(model, metrics=[ranking_metric])} - {model['preprocessings']}")

# Create visualizations
analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k_comparison(k=best_model_count, metric='rmse')

# Plot heatmap: models vs partitions
fig2 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="partition",
    aggregation="best",  # Options: 'best', 'mean', 'median'
)

# Plot heatmap: models vs datasets
fig3 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="dataset_name",
    rank_partition='val', # default
    rank_metric='rmse',  # default
    display_metric='rmse', # default
    display_partition='test', # default
    aggregation='best'  # default
)

# Plot heatmap: models vs datasets (all results)
fig4 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="dataset_name",
)

# Plot heatmap: models vs fold IDs
fig5 = analyzer.plot_heatmap_v2(
    x_var="model_name",
    y_var="fold_id",
)

plt.show()