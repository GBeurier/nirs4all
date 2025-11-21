"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

import argparse

from sklearn.model_selection import ShuffleSplit, KFold, LeaveOneGroupOut, GroupKFold, StratifiedGroupKFold

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.operators.splitters import KennardStoneSplitter, SPXYSplitter, KMeansSplitter, SPlitSplitter, SystematicCircularSplitter, KBinsStratifiedSplitter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q1 GroupSplit Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

data_path = 'sample_data/classification'

pipeline = [
    "fold_Sample_ID",
    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    # RandomForestClassifier(max_depth=30)
]

pipeline_config = PipelineConfigs(pipeline, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


pipeline_stratified = [
    "fold_Sample_ID",
    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_stratified, "Q1_groupsplit")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)
