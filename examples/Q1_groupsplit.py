"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
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

# Disable emojis in output (set to '1' to disable, '0' to enable)
os.environ['DISABLE_EMOJIS'] = '0'

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


pipeline = [
    "fold_Sample_ID",
    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    # RandomForestClassifier(max_depth=30)
]
# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


pipeline_stratified = [
    "fold_Sample_ID",
    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    "fold_chart",
    # RandomForestClassifier(max_depth=30)
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline_stratified, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# plt.show()