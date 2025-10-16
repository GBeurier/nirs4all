"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
# Disable emojis in output BEFORE importing anything (set to '1' to disable, '0' to enable)
os.environ['DISABLE_EMOJIS'] = '1'

import matplotlib.pyplot as plt

# Third-party imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer, Gaussian as Gauss,
    StandardNormalVariate as StdNorm, SavitzkyGolay as SavGol, Haar, MultiplicativeScatterCorrection as MSC
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.transformations import (
    Random_X_Operation,
    Rotate_Translate,
    Spline_Curve_Simplification,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    Spline_X_Perturbations,
    Spline_Smoothing,Rotate_Translate,
)
# Disable emojis in output (set to '1' to disable, '0' to enable)
os.environ['DISABLE_EMOJIS'] = '1'

# Configuration variables
feature_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FstDer, SndDer, Gauss,
    StdNorm, SavGol, Haar, MSC
]
split = SPXYSplitter(0.25)
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/classification'


import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


focus = "y"

chart = "fold_chart" if focus == "y" else "fold_Sample_ID"

pipeline = [
    chart,
    # {
    #     "sample_augmentation": {
    #         "transformers": [
    #             Rotate_Translate(p_range=2, y_factor=3),
    #         ],
    #         "count": 2
    #     }
    # },
    # chart,
    {
        "sample_augmentation": {
            "transformers": [
                Rotate_Translate
            ],
            "balance": "y",
            # "target_size": 100,
            # "max_factor": 0.8,
        }
    },
    chart,
    {"split": GroupKFold(n_splits=5), "group": "Sample_ID"},
    chart,
    # RandomForestClassifier(max_depth=30)
]
# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "q12")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# plt.show()