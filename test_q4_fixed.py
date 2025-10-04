#!/usr/bin/env python3
"""Simple Q4 Prediction Test - Run pipeline, then test 3 prediction methods"""

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transformations import Gaussian, SavitzkyGolay, StandardNormalVariate, Haar
import shutil
from pathlib import Path

def main():
    # Clear and run training
    if Path("./results").exists():
        shutil.rmtree("./results")

    pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler},
        {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay()]},  # No augmentation for simplicity
        ShuffleSplit(n_splits=2, test_size=.25, random_state=42),
        {"model": PLSRegression(10), "name": "MODEL_1"},  # Added a name for easier identification
        # {"model": PLSRegression(20), "name": "MODEL_2"},  # Added a name for easier identification
    ]

    pipeline_config = PipelineConfigs(pipeline)
    dataset_config = DatasetConfigs(['sample_data/regression'])
    runner = PipelineRunner(save_files=True, verbose=0)
    run_predictions, _ = runner.run(pipeline_config, dataset_config)

    # Best train model
    best_entry = run_predictions.top_k(1, filters={"partition": "test"})[0]
    prediction_id = best_entry['id']
    config_path = best_entry['config_path']
    print(f"Best model: {best_entry['model_name']} (id: {prediction_id})")
    train_preds = best_entry['y_pred'].flatten()
    print(f"First: {train_preds[0]:.6f}, Last: {train_preds[-1]:.6f}")

    ####################################################################

    # Test prediction methods
    predictor = PipelineRunner(save_files=False, verbose=0)  # No need to save files here
    prediction_dataset = DatasetConfigs(['sample_data/regression_2'])

    print("\n--- Method 1: Entry ---")
    predictions1, _ = predictor.predict(best_entry, prediction_dataset, verbose=0)
    preds1 = predictions1.top_k(1, filters={"partition": "test"})[0]['y_pred'].flatten()
    print(f"First: {preds1[0]:.6f}, Last: {preds1[-1]:.6f}")

    # print("\n--- Method 2: ID ---")
    # predictions2, _ = predictor.predict(prediction_id, prediction_dataset, verbose=0)
    # preds2 = predictions2.top_k(1)[0]['y_pred'].flatten()
    # print(f"First: {preds2[0]:.6f}, Last: {preds2[-1]:.6f}")

    # print("\n--- Method 3: Path ---")
    # predictions3, _ = predictor.predict(f"results/{config_path}", prediction_dataset, verbose=0)
    # preds3 = predictions3.top_k(1, filters={"partition": "test"})[0]['y_pred'].flatten()
    # print(f"First: {preds3[0]:.6f}, Last: {preds3[-1]:.6f}")

    # ########################################

    # import numpy as np
    # identical = np.allclose(preds1, preds2) and np.allclose(preds2, preds3) and np.allclose(preds1, train_preds)
    # print(f"\nAll methods identical: {'✅ YES' if identical else '❌ NO'}")

if __name__ == "__main__":
    main()