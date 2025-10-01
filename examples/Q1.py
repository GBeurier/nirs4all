import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.dataset import DatasetConfigs
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
    {"feature_augmentation": {"_or_": list_of_preprocessors, "size": [1, (1, 2)], "count": 5}},  # Generate all elements of size 1 and of order 1 or 2 (ie. "Gaussian", ["SavitzkyGolay", "Log"], etc.)
    PLSRegression(n_components=10),
    splitting_strategy,
]

for i in range(10, 30, 10):
    model = {
        "name": f"PLS-{i}_cp",
        "model": PLSRegression(n_components=i)
    }
    pipeline.append(model)

pipeline_config = PipelineConfigs(pipeline, "pipeline_Q1")
dataset_config = DatasetConfigs(dataset_folder)

# Create pipeline with verbose=1 to see debug output
runner = PipelineRunner(save_files=True, verbose=0)
run_predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)

###############################################################################################################

analyzer = PredictionAnalyzer(run_predictions)

# Get top models to verify the real model names are displayed correctly
best_count = 5
top_10 = analyzer.get_top_k(best_count, 'rmse')
print(f"Top {best_count} models by RMSE:")
for i, model in enumerate(top_10):
    print(f"{i+1}. Model: {model['enhanced_model_name']} | RMSE: {model['metrics']['rmse']:.6f} | MSE: {model['metrics']['mse']:.6f} | R²: {model['metrics']['r2']:.6f}  - Pipeline: {model['pipeline_info']['pipeline_name']}")

# Plot comparison with enhanced names (for readability in plots)
fig = analyzer.plot_top_k_comparison(k=best_count, metric='rmse', partition_type='test')
plt.show()