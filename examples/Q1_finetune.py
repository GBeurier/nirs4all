import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.operators.transformations import Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer

## TODO finetune
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

pipeline_config = PipelineConfigs(pipeline, "Q1")
dataset_config = DatasetConfigs(dataset_folder)

# Create pipeline with verbose=1 to see debug output
runner = PipelineRunner(save_files=False, verbose=0, random_state=42) ## SEED DANS PIPELINE RUNNER
predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)
# predictions du run, {
#     'regression': {
#         "global_predictions": global_dataset_predictions,
#         "run_predictions": run_dataset_predictions,
#         "dataset": dataset,
#         "dataset_name": dataset_name
#     }
# }



###############################################################################################################

# Get top models to verify the real model names are displayed correctly
best_count = 5
top_10 = predictions.top_k(best_count, 'rmse', partition="val")  ## partition VALIDATION PAR DEFAULT

print(f"Top {best_count} models by RMSE:")
for i, model in enumerate(top_10):
    print(f"{i+1}. {Predictions.pred_short_string(model, metrics=['r2', 'mae'])}") # TODO verif Model_name dans pred_short_string, Ajouter les prétraitements (noob level) dans pred_long_string
    ## Prétraitements, Components, Metrics


analyzer = PredictionAnalyzer(predictions) ## Prétraitements dans le graphique
# fig = analyzer.plot_top_k_comparison(k=best_count, metric='rmse')
# plt.show()
# fig1 = analyzer.plot_performance_matrix(metric='rmse', separate_avg=False) ## trier pas model_name pas model_classname (par pls_x), option best or average
# fig1.suptitle('Performance Matrix - Normalized RMSE by Model and Dataset')
# plt.show()

# -----------------------
### ordonnée pretraitements, abscisse model_name
#### Plot dédié ! avec aliasing des TransformerMixin
# -----------------------
### prédiction folds > either fold or avg or w-avg
# -------------------------------
### SEED DANS RUN
