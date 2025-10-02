from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

# from deprec.controllers.models import data  # Comment out problematic import
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
runner = PipelineRunner(save_files=True, verbose=0)  # CHANGED: Enable model saving for testing
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
    print(f"Dataset: name={dataset_name}, number of predictions in the run={len(dataset_prediction['run_predictions'])}")
    analyzer = PredictionAnalyzer(dataset_prediction['run_predictions'])

    top_models = analyzer.get_top_k(10, 'rmse')  # Get more models to find non-virtual ones

    # Use the best model (can be virtual or real)
    best_model = top_models[0]

    # Extract metadata fields (they're stored in metadata dict)
    metadata = best_model.get("metadata", {})
    model_path = metadata.get("model_path", "unknown")
    config_path = metadata.get("config_path", "unknown")
    config_id = metadata.get("config_id", "unknown")
    prediction_model = best_model

    print(f"Using best model: {best_model['enhanced_model_name']} - RMSE: {best_model['metrics']['rmse']:.4f}")
    print(f"Model path: {model_path}")

    predicted_dataset = DatasetConfigs(['sample_data/regression_2'])
    predictions_from_model_path = PipelineRunner.predict( ## Directly use the model_path to retrieve the model and predict
        model_path,
        predicted_dataset
    )
    print(f"Predictions from model path: {model_path}")
    print(predictions_from_model_path)

    predictions_from_config_path = PipelineRunner.predict( ## GO in results, search for config_path, take the best(s) model(s) and predict
        config_path,
        predicted_dataset,
        # top_best = 1, # DEFAULT VALUE
    )
    print(f"Predictions from config path: {config_path}")
    print(predictions_from_config_path)

    predictions_from_prediction_model = PipelineRunner.predict( ## Directly use the model_info and tags of prediction model to retrieve the model and predict
        prediction_model,
        predicted_dataset,
    )
    print(f"Predictions from prediction model: {prediction_model}")
    print(predictions_from_prediction_model)

    predictions_from_config_id = PipelineRunner.predict( ## GO in results, search for config_id, take the best(s) model(s) and predict
        config_id,
        predicted_dataset,
        top_best = 3, # Use top 3 models from that config to predict
    )
    print(f"Predictions from config id: {config_id}")
    print(predictions_from_config_id)

    predictions_from_dataset_name = PipelineRunner.predict( ## GO in results, search for dataset_name, take the best(s) config(s) and predict
        dataset_name,
        predicted_dataset,
        top_best = 3, # Use top 3 models from that config to predict
    )
    print(f"Predictions from dataset name: {dataset_name}")
    print(predictions_from_dataset_name)