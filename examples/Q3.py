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
    # {"model": GradientBoostingRegressor(n_estimators=100)},
    # {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1)},
    # {"model": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500)},
    # {"model": GradientBoostingRegressor(n_estimators=100)},
    # {"model": RandomForestRegressor(n_estimators=100)},
    {"model": Ridge(alpha=1.0)},
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

# create pipeline config
pipeline_config = PipelineConfigs(pipeline)

path = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']
dataset_config = DatasetConfigs(path)

# Runner setup with spinner enabled (default is True, but let's be explicit)
runner = PipelineRunner(save_files=False, verbose=0)
print("🔄 Running pipeline with spinner enabled - watch for loading animations during model training!")
run_predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)


###############################################################################################################


for name, dataset_prediction in datasets_predictions.items():
    print(f"Dataset: name={name}, number of predictions in the run={len(dataset_prediction['run_predictions'])}")
    analyzer = PredictionAnalyzer(dataset_prediction['run_predictions'])

    best_count = 5
    top_10 = analyzer.get_top_k(best_count, 'rmse')
    print(f"Top {best_count} models by RMSE:")
    for i, model in enumerate(top_10):
        print(f"{i+1}. Model: {model['enhanced_model_name']} | RMSE: {model['metrics']['rmse']:.6f} | MSE: {model['metrics']['mse']:.6f} | R²: {model['metrics']['r2']:.6f}  - Pipeline: {model['pipeline_info']['pipeline_name']}")

    # Plot comparison with enhanced names (for readability in plots)
    fig = analyzer.plot_top_k_comparison(k=best_count, metric='rmse', partition_type='test')
    plt.show()

analyzer = PredictionAnalyzer(run_predictions)
fig1 = analyzer.plot_performance_matrix(metric='rmse', normalize=True)
fig1.suptitle('Performance Matrix - Normalized RMSE by Model and Dataset')
plt.show()

fig2 = analyzer.plot_score_boxplots_by_dataset(metric='rmse')
fig2.suptitle('RMSE Score Distribution by Dataset')
plt.show()
