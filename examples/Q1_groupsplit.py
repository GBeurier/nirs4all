"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold

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

data_path = 'sample_data/classification'

pipeline = [
    "fold_Sample_ID",
    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    # RandomForestClassifier(max_depth=30)
]

pipeline_config = PipelineConfigs(pipeline, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


pipeline_stratified = [
    "fold_Sample_ID",
    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_stratified, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)
