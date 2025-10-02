from matplotlib import pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


from nirs4all.dataset.predictions import Predictions
from nirs4all.operators.transformations import MultiplicativeScatterCorrection
from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer

pipeline = [
    MinMaxScaler(feature_range=(0.1, 0.8)),
    MultiplicativeScatterCorrection,
    "3d_chart",
    ShuffleSplit(n_splits=2),
    {"y_processing": MinMaxScaler},
    {"model": PLSRegression(15)},
    {"model": PLSRegression(10)},
    {"model": RandomForestRegressor(n_estimators=100)},
    {"model": ElasticNet()},
    {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1), "name": "SVR_Custom_Model"},
    {"model": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500), "name": "MLP_Custom_Model"},
    {"model": GradientBoostingRegressor(n_estimators=100)},
    {
        "model": nicon,
        "train_params": {
            "epochs": 5,
            "patience": 50,
            "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
        },
    },
]

# create pipeline config
pipeline_config = PipelineConfigs(pipeline, name="Q2")

path = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']
dataset_config = DatasetConfigs(path)

# Runner setup with spinner enabled (default is True, but let's be explicit)
runner = PipelineRunner(save_files=False, verbose=0)
predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

###############################################################################################################

# Get top models to verify the real model names are displayed correctly
best_count = 3
rank_metric = 'rmse'  # 'rmse', 'mae', 'r2'

for dataset_name, predict_dict in predictions_per_datasets.items():
    run_predictions = predict_dict['run_predictions']
    print(f"\nTop {best_count} models by {rank_metric} for dataset: {dataset_name}")
    top_10 = run_predictions.top_k(best_count, rank_metric)
    for i, model in enumerate(top_10):
        print(f"{i+1}. {Predictions.pred_short_string(model, metrics=[rank_metric])} - {model['preprocessings']}")

    analyzer = PredictionAnalyzer(run_predictions)
    fig1 = analyzer.plot_top_k_comparison(k=best_count, metric='rmse')
    # plt.savefig('test_top_k_models_Q1.png', dpi=150, bbox_inches='tight')
    plt.show()

analyzer = PredictionAnalyzer(predictions)
fig2 = analyzer.plot_variable_heatmap(
    filters={"partition": "test"},
    x_var="model_name",
    y_var="dataset_name",
    metric='rmse'
)
# # plt.savefig('test_heatmap2.png', dpi=300)
plt.show()