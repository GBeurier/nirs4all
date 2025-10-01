from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

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

p_configs = PipelineConfigs(pipeline)

# path = ['../../sample_data/regression', '../../sample_data/classification', '../../sample_data/binary']
path = '../../sample_data/regression'
d_configs = DatasetConfigs(path)

# Train with explicit settings to ensure metadata is saved
runner = PipelineRunner(save_files=True, verbose=0)  # Set verbose=0 to reduce output
predictions, results = runner.run(p_configs, d_configs)

print(f"\n=== TRAINING METADATA CHECK ===")
print(f"Step binaries tracked: {len(runner.step_binaries)} steps")
print(f"Sample step binaries: {dict(list(runner.step_binaries.items())[:3])}")

visualizer = PredictionAnalyzer(predictions, dataset_name_override="dataset")
top_5 = visualizer.get_top_k(5, 'rmse') ##TODO get_top_1

# print(f"\n=== TOP 5 RESULTS ===")
# for i, model in enumerate(top_5, 1):
#     print(f"{i}. {model['path']} - RMSE: {model['rmse']:.6f}, R²: {model['r2']:.6f}, MAE: {model['mae']:.6f} {'✅' if has_metadata else '❌'}")

print(f"\n=== TESTING PREDICTION ===")
best_path = top_5[0]['path']
print(f"Using best model from: {best_path}")

try:
    predictions = PipelineRunner.predict(
        path=best_path,
        dataset=d_configs,
        # model=my_model,##TODO
        best_model=False,##TODO quand on veut prédire sur tous les modèles
        verbose=1
    )
    print("✅ Prediction successful!")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    # Show which step failed
    import traceback
    traceback.print_exc()