"""
Q2 Example - Multi-Model Comparison Pipeline
===========================================
Demonstrates comparison of various regression models including PLS, Random Forest,
Elastic Net, SVR, MLP, Gradient Boosting, and neural networks on NIRS data.
"""

# Standard library imports
import argparse
from matplotlib import pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.models.tensorflow.nicon import thin_nicon
from nirs4all.operators.transforms import MultiplicativeScatterCorrection
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q2 Multi-Model Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

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
    {"model": PLSRegression(15), "name": "PLS_15"},
    {"model": PLSRegression(10), "name": "PLS_10"},
    {"model": PLSRegression(11), "name": "PLS_11"},
    {"model": PLSRegression(7), "name": "PLS_7"},
    {"model": PLSRegression(13), "name": "PLS_13"},
    # {"model": RandomForestRegressor(n_estimators=10)},
    # {"model": ElasticNet()},
    # {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1), "name": "SVR_Custom_Model"},
    # {"model": MLPRegressor(hidden_layer_sizes=(20,20), max_iter=50), "name": "MLP_Custom_Model"},
    # {"model": GradientBoostingRegressor(n_estimators=20)},
    {"model": AdaBoostRegressor(n_estimators=5), "name": "AdaBoost"},
]

if XGBRegressor:
    pipeline.append({"model": XGBRegressor(n_estimators=5, verbosity=0), "name": "XGBoost"})

if LGBMRegressor:
    pipeline.append({"model": LGBMRegressor(n_estimators=20, verbose=-1, verbosity=-1), "name": "LightGBM"})

if CatBoostRegressor:
    pipeline.append({"model": CatBoostRegressor(iterations=15, verbose=0, allow_writing_files=False), "name": "CatBoost"})

# Create pipeline configuration
pipeline_config = PipelineConfigs(pipeline, name="Q2")

# Dataset configuration
data_paths = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']
dataset_config = DatasetConfigs(data_paths)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=0, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
best_model_count = 3
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'


analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k(k=best_model_count, rank_metric='rmse')

for idx, dataset_name in enumerate(predictions_per_dataset):
    dataset_predictions = predictions_per_dataset[dataset_name]["run_predictions"]
    val_top_rmse = dataset_predictions.top(n=best_model_count, rank_metric='rmse', rank_partition='val', display_partition=['val', 'test'], display_metric=['mse', 'rmse', 'R2', 'mae'])
    test_top_rmse = dataset_predictions.top(n=best_model_count, rank_metric='rmse', rank_partition='test', display_partition=['val', 'test'], display_metric=['mse', 'rmse', 'R2', 'mae'])
    val_top_r2 = dataset_predictions.top(n=best_model_count, rank_metric='r2', rank_partition='val', display_partition=['val', 'test'], display_metric=['mse', 'rmse', 'R2', 'mae'])
    test_top_r2 = dataset_predictions.top(n=best_model_count, rank_metric='r2', rank_partition='test', display_partition=['val', 'test'], display_metric=['mse', 'rmse', 'R2', 'mae'])

    all_tops = [val_top_rmse, test_top_rmse, val_top_r2, test_top_r2]
    print(f"\nDataset: {dataset_name}")
    for top_models in all_tops:
        print("\nTop models:")
        for idx, model in enumerate(top_models):
            print(f"{idx+1}. {Predictions.pred_short_string(model, metrics=['mse', 'rmse', 'R2'], partition=['val', 'test'])}")

# Create visualizations

# Plot heatmap: models vs partitions
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    display_metric='mse',
)

# Plot heatmap: models vs datasets
fig3 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_partition='test',  # default
    rank_metric='rmse',   # default
    display_metric='mse',  # default
    display_partition='test',  # default
    rank_agg='best',  # default
    display_agg='best'  # default
)

# Plot heatmap: models vs datasets (all results)
fig4 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='r2',
    display_metric='r2',
)

# fig5 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="dataset_name",
#     rank_metric='rmse',
#     display_metric='r2',
# )

# fig6 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="dataset_name",
#     rank_metric='r2',
#     display_metric='rmse',
# )

# # Plot heatmap: models vs fold IDs
# fig7 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="fold_id",
#     display_metric='r2'
# )

# Plot heatmap: models vs fold IDs
fig8 = analyzer.plot_heatmap(
    x_var="dataset_name",
    y_var="fold_id",
    display_metric='r2'
)

# Plot heatmap: models vs fold IDs
fig8 = analyzer.plot_heatmap(
    x_var="dataset_name",
    y_var="preprocessings",
    display_metric='mse'
)

fig9 = analyzer.plot_candlestick(
    variable="model_name",
    metric='rmse',
)

fig10 = analyzer.plot_candlestick(
    variable="fold_id",
    metric='r2',
)

fig11 = analyzer.plot_histogram(
    metric='mae',
)

fig12 = analyzer.plot_histogram(
    metric='mape',
)

if args.show:
    plt.show()
