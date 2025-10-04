#!/usr/bin/env python3
"""Simple Q4 Prediction Test - Run pipeline, then test 3 prediction methods"""

from sklearn.model_selection import ShuffleSplit, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
import numpy as np
from nirs4all.operators.models.cirad_tf import nicon
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler},
    {
        "feature_augmentation": {
            "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
            "size": [(2, 3), (1, 3)],
            "count": 5
        },
    },
    # ShuffleSplit(n_splits=3, test_size=.25, random_state=42),
    RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),
    {"model": PLSRegression(10), "name": "Q4_model1"},  # Added a name for easier identification
    {"model": PLSRegression(20), "name": "Q4_model2"},  # Added a name for easier identification
    # {"model": RandomForestRegressor(n_estimators=20)},
    {"model": GradientBoostingRegressor(n_estimators=20)},
    # {
    #     "model": nicon,
    #     "train_params": {
    #         "epochs": 100,
    #         "patience": 50,
    #         "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
    #     },
    # },
]

pipeline_config = PipelineConfigs(pipeline)
dataset_config = DatasetConfigs(['sample_data/regression'])
runner = PipelineRunner(save_files=True, verbose=0)
run_predictions, _ = runner.run(pipeline_config, dataset_config)

# Best train model
reference_prediction = run_predictions.top_k(1, partition="test")[0]
prediction_id = reference_prediction['id']

print("=== Q4 - Example ===")
print("--- Source Model ---")
print(f"Best model: {reference_prediction['model_name']} (id: {prediction_id})")
reference_array = reference_prediction['y_pred'][:5].flatten()
print("Y reference:", reference_array)
print("-" * 120)
# ####################################################################

# Test prediction methods
print("--- Method 1: Predict with a prediction entry ---")
## Rebuild a pipeline runner and load dataset
predictor = PipelineRunner(save_files=False, verbose=0)  # No need to save files here
prediction_dataset = DatasetConfigs({
    'X_test': 'sample_data/regression_2/Xval.csv.gz',
})
## Predict with the reference prediction entry
method1_prediction, _ = predictor.predict(reference_prediction, prediction_dataset, verbose=0)
method1_array = method1_prediction[:5].flatten()
print("Y:", method1_array)
identical = np.allclose(method1_array, reference_array)
print(f"Method 1 identical to training: {'✅ YES' if identical else '❌ NO'}")


####################################################################
print("=" * 120)
print("--- Method 2: Predict with a model ID ---")
## Rebuild a pipeline runner and load dataset
predictor = PipelineRunner(save_files=False, verbose=0)  # No need to save files here
prediction_dataset = DatasetConfigs({
    'X_test': 'sample_data/regression_2/Xval.csv.gz',
})
## Predict with the reference prediction entry
reference_id = reference_prediction['id']
print(f"Using model ID: [{reference_id}]")
method2_prediction, _ = predictor.predict(reference_id, prediction_dataset, verbose=0)
method2_array = method2_prediction[:5].flatten()
print("Y:", method2_array)
identical = np.allclose(method2_array, reference_array)
print(f"Method 2 identical to training: {'✅ YES' if identical else '❌ NO'}")
####################################################################
