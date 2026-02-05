"""
Q2B Repetition-Based Group Example - Universal Group Support for Any Splitter
==============================================================================
Demonstrates how the `repetition` parameter in DatasetConfigs enables ANY
sklearn-compatible splitter (KFold, ShuffleSplit, StratifiedKFold, etc.)
to work with grouped samples automatically.

The repetition parameter:
1. Declares which column identifies sample repetitions
2. All splitters automatically respect repetition groups
3. Ensures all samples from the same group stay together in train/test
4. No need to specify grouping at each split step

Key benefits:
- Universal compatibility: Any sklearn splitter works with groups
- Prevents data leakage: Groups are never split across train/test
- Single declaration: Set once in DatasetConfigs, works everywhere

Compare with Q2_groupsplit.py which shows native group splitters (GroupKFold, etc.)
"""

import argparse

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q2B Force Group Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# =============================================================================
# Example 1: KFold with repetition
# =============================================================================
# KFold doesn't natively support groups, but repetition makes it work!

print("\n" + "=" * 70)
print("Example 1: KFold with repetition (non-group-aware splitter)")
print("=" * 70)

data_path = 'sample_data/classification'

pipeline_kfold = [
    "fold_Sample_ID",  # Visualize samples grouped by Sample_ID before split
    KFold(n_splits=3, shuffle=True, random_state=42),  # Auto-groups via repetition!
    "fold_Sample_ID",  # Visualize after split - groups are respected!
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_kfold, "repetition_kfold")
dataset_config = DatasetConfigs(data_path, repetition="Sample_ID")  # Wrap KFold with group-awareness
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Repetition with KFold completed - groups respected in all folds!")

# =============================================================================
# Example 2: ShuffleSplit with repetition
# =============================================================================
# ShuffleSplit explicitly ignores the 'groups' parameter in sklearn
# repetition fixes this!

print("\n" + "=" * 70)
print("Example 2: ShuffleSplit with repetition")
print("=" * 70)

pipeline_shuffle = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),  # Auto-groups via repetition!
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_shuffle, "repetition_shuffle")
dataset_config = DatasetConfigs(data_path, repetition="Sample_ID")
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Repetition with ShuffleSplit completed - groups respected!")

# =============================================================================
# Example 3: StratifiedKFold with repetition (classification)
# =============================================================================
# Combines stratification (balanced class distribution) with group-awareness!

print("\n" + "=" * 70)
print("Example 3: StratifiedKFold with repetition (stratified + grouped)")
print("=" * 70)

pipeline_stratified = [
    {"split": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
     "y_aggregation": "mode"},  # Use mode for classification targets
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_stratified, "repetition_stratified")
dataset_config = DatasetConfigs(data_path, repetition="Sample_ID")
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Repetition with StratifiedKFold completed - balanced class distribution per fold!")

# =============================================================================
# Comparison: repetition vs native GroupKFold
# =============================================================================
print("\n" + "=" * 70)
print("Comparison: repetition (KFold) vs native GroupKFold")
print("=" * 70)

# Using repetition with KFold
pipeline_rep = [
    "fold_Sample_ID",
    KFold(n_splits=3),  # Auto-groups via repetition!
    "fold_Sample_ID",
]

pipeline_config = PipelineConfigs(pipeline_rep, "repetition_comparison")
dataset_config = DatasetConfigs('sample_data/classification', repetition="Sample_ID")
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions_rep, _ = runner.run(pipeline_config, dataset_config)

# Using native GroupKFold
pipeline_native = [
    "fold_Sample_ID",
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"},
    "fold_Sample_ID",
]

pipeline_config = PipelineConfigs(pipeline_native, "native_group_comparison")
dataset_config = DatasetConfigs('sample_data/classification')
runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=1, plots_visible=args.plots)
predictions_native, _ = runner.run(pipeline_config, dataset_config)

print("\nBoth approaches ensure groups are never split across train/test!")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
