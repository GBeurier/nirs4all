from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

from deprec.controllers.models import data
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
import json
import os
import shutil
from pathlib import Path


import numpy as np
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
# Clear old results to ensure fresh training with metadata
results_path = Path("./results")
if results_path.exists():
    shutil.rmtree(results_path)
    print("🧹 Cleared old results to ensure fresh training")

pipeline = [
    # Normalize the spectra reflectance
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},

    # Generate 5 version of feature augmentation combinations (3 elements with size 1 to 2, ie. [SG, [SNV, GS], Haar])
    # {
    #     "feature_augmentation": {
    #         "_or_": [
    #             Gaussian, StandardNormalVariate, SavitzkyGolay, Haar,
    #         ],
    #         "size": [3, (1,2)],
    #         "count": 5,
    #     }
    # },

    # Split the dataset in train and validation
    ShuffleSplit(n_splits=3, test_size=.25),

    # Normalize the y values
    {"model": PLSRegression(10)},
]

# create pipeline config
pipeline_config = PipelineConfigs(pipeline)

path = ['sample_data/regression']
dataset_config = DatasetConfigs(path)

# Runner setup with spinner enabled (default is True, but let's be explicit)
runner = PipelineRunner(save_files=False, verbose=0)
print("🔄 Running pipeline with spinner enabled - watch for loading animations during model training!")
run_predictions, datasets_predictions = runner.run(pipeline_config, dataset_config)

###############################################################################################################

analyzer = PredictionAnalyzer(run_predictions)
# Get top models to verify the real model names are displayed correctly
best_count = 5
top_10 = analyzer.get_top_k(best_count, 'rmse')
print(f"Top {best_count} models by RMSE:")
for i, model in enumerate(top_10):
    print(f"{i+1}. Model: {model['enhanced_model_name']} | RMSE: {model['metrics']['rmse']:.6f} | MSE: {model['metrics']['mse']:.6f} | R²: {model['metrics']['r2']:.6f}  - Pipeline: {model['pipeline_info']['pipeline_name']}")


#################################################################################################################

for dataset_name, dataset_prediction in datasets_predictions.items():
    print(f"Dataset: name={name}, number of predictions in the run={len(dataset_prediction['run_predictions'])}")
    analyzer = PredictionAnalyzer(dataset_prediction['run_predictions'])

    top_1 = analyzer.get_top_k(1, 'rmse')
    model_path = top_1[0]["model_path"]
    config_path = top_1[0]["config_path"]
    prediction_model = top_1[0]
    config_id = top_1[0]["config_id"]

    predicted_dataset = DatasetConfigs(['sample_data/regression_2'])
    predictions_from_model_path = PipelineRunner.predict(
        model_path,
        predicted_dataset
    )
    print(f"Predictions from model path: {model_path}")
    print(predictions_from_model_path)

    predictions_from_config_path = PipelineRunner.predict(
        config_path,
        predicted_dataset,
    )
    print(f"Predictions from config path: {config_path}")
    print(predictions_from_config_path)

    predictions_from_prediction_model = PipelineRunner.predict(
        prediction_model,
        predicted_dataset,
    )
    print(f"Predictions from prediction model: {prediction_model}")
    print(predictions_from_prediction_model)

    predictions_from_config_id = PipelineRunner.predict(
        config_id,
        predicted_dataset,
        top_best = 3, # Use top 3 models from that config to predict
    )
    print(f"Predictions from config id: {config_id}")
    print(predictions_from_config_id)
