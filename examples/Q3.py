from matplotlib import pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar, MultiplicativeScatterCorrection
from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.dataset.predictions import Predictions

pipeline = [
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"feature_augmentation": [
        MultiplicativeScatterCorrection, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar,
        [MultiplicativeScatterCorrection, Gaussian],
        [MultiplicativeScatterCorrection, StandardNormalVariate],
        [MultiplicativeScatterCorrection, SavitzkyGolay],
        [MultiplicativeScatterCorrection, Haar],
    ]},
    ShuffleSplit(n_splits=3),
    {"y_processing": MinMaxScaler},
    {"model": PLSRegression(15)},
    {"model": ElasticNet()},
    {"model": GradientBoostingRegressor(n_estimators=100)},
    {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1)},
    {"model": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500)},
    {"model": GradientBoostingRegressor(n_estimators=100)},
    {"model": RandomForestRegressor(n_estimators=100)},
    {"model": Ridge(alpha=1.0)},
    {
        "model": nicon,
        "train_params": {
            "epochs": 50,
            "patience": 50,
            "batch_size": 500,
            "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
        },
    },
]

# create pipeline config
pipeline_config = PipelineConfigs(pipeline, name="Q3")

path = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']
dataset_config = DatasetConfigs(path)

# Runner setup with spinner enabled (default is True, but let's be explicit)
runner = PipelineRunner(save_files=False, verbose=0)
print("🔄 Running pipeline with spinner enabled - watch for loading animations during model training!")
run_predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)


###############################################################################################################


for name, dataset_prediction in datasets_predictions.items():
    print(f"Dataset: name={name}, number of predictions in the run={len(dataset_prediction['run_predictions'])}")

    # Get the Predictions object from the dataset_prediction dictionary
    predictions_obj = dataset_prediction['run_predictions']
    top_10 = predictions_obj.top_k(10, 'rmse')
    print("Top 10 models by RMSE:")
    for i, model in enumerate(top_10):
        print(f"{i+1}. {Predictions.pred_long_string(model, metrics=['rmse', 'r2', 'mae'])}")

    # Plot comparison with enhanced names (for readability in plots)
    analyzer = PredictionAnalyzer(predictions_obj)
    fig = analyzer.plot_top_k_comparison(k=10, metric='rmse')
    plt.show()

analyzer = PredictionAnalyzer(run_predictions)
fig1 = analyzer.plot_performance_matrix(metric='rmse', separate_avg=True)
fig1.suptitle('Performance Matrix - Normalized RMSE by Model and Dataset')
plt.show()

fig2 = analyzer.plot_candlestick_models(metric='rmse')
fig2.suptitle('RMSE Score Distribution by Dataset')
plt.show()

