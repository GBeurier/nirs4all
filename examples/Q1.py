import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.operators.transformations import Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer


x_scaler = MinMaxScaler()  # StandardScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), LogTransform()
y_scaler = MinMaxScaler()
list_of_preprocessors = [Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection]
splitting_strategy = ShuffleSplit(n_splits=3, test_size=.25)
dataset_folder = 'sample_data/regression'

pipeline = [
    "chart_2d",
    x_scaler,
    {"y_processing": y_scaler},
    {"feature_augmentation": {"_or_": list_of_preprocessors, "size": [1, (1, 2)], "count": 10}},  # Generate all elements of size 1 and of order 1 or 2 (ie. "Gaussian", ["SavitzkyGolay", "Log"], etc.)
    splitting_strategy,
]

for i in range(5, 35, 2):
    model = {
        "name": f"PLS-{i}_cp",
        "model": PLSRegression(n_components=i)
    }
    pipeline.append(model)

pipeline_config = PipelineConfigs(pipeline, "Q1")
dataset_config = DatasetConfigs(dataset_folder)

# Create pipeline with verbose=1 to see debug output
runner = PipelineRunner(save_files=False, verbose=0)
predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

###############################################################################################################

# Get top models to verify the real model names are displayed correctly
best_count = 5
rank_metric = 'r2'  # 'rmse', 'mae', 'r2'
top_10 = predictions.top_k(best_count, rank_metric)
print(f"Top {best_count} models by {rank_metric}:")
for i, model in enumerate(top_10):
    print(f"{i+1}. {Predictions.pred_short_string(model, metrics=[rank_metric])} - {model['preprocessings']}")

# TAB REPORT
analyzer = PredictionAnalyzer(predictions) ## Prétraitements dans le graphique
fig1 = analyzer.plot_top_k_comparison(k=best_count, metric='rmse')
# plt.savefig('test_top_k_models_Q1.png', dpi=150, bbox_inches='tight')
plt.show()

fig2 = analyzer.plot_variable_heatmap(
    filters={"partition": "test"},
    x_var="model_name",
    y_var="preprocessings",
    metric='rmse'
)
# # plt.savefig('test_heatmap2.png', dpi=300)
plt.show()

fig3 = analyzer.plot_variable_candlestick(
    filters={"partition": "test"},
    variable="model_name",
    metric='rmse'
)
# plt.savefig('test_candlestick_models_Q1.png', dpi=150, bbox_inches='tight')
plt.show()