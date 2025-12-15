"""
Q2B Force Group Example - Universal Group Support for Any Splitter
===================================================================
Demonstrates the `force_group` mechanism that enables ANY sklearn-compatible
splitter (KFold, ShuffleSplit, StratifiedKFold, etc.) to work with grouped samples.

The force_group parameter:
1. Aggregates samples by group into "virtual samples"
2. Passes virtual samples to the splitter
3. Maps fold indices back to original sample indices
4. Ensures all samples from the same group stay together in train/test

Key benefits:
- Universal compatibility: Any sklearn splitter works with groups
- Prevents data leakage: Groups are never split across train/test
- Stratification support: Use force_group="y" for continuous target stratification

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
# Example 1: force_group with KFold
# =============================================================================
# KFold doesn't natively support groups, but force_group makes it work!

print("\n" + "=" * 70)
print("Example 1: force_group with KFold (non-group-aware splitter)")
print("=" * 70)

data_path = 'sample_data/classification'

pipeline_kfold = [
    "fold_Sample_ID",  # Visualize samples grouped by Sample_ID before split
    {
        "split": KFold(n_splits=3, shuffle=True, random_state=42),
        "force_group": "Sample_ID"  # Wrap KFold with group-awareness
    },
    "fold_Sample_ID",  # Visualize after split - groups are respected!
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_kfold, "force_group_kfold")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Force_group with KFold completed - groups respected in all folds!")

# =============================================================================
# Example 2: force_group with ShuffleSplit
# =============================================================================
# ShuffleSplit explicitly ignores the 'groups' parameter in sklearn
# force_group fixes this!

print("\n" + "=" * 70)
print("Example 2: force_group with ShuffleSplit")
print("=" * 70)

pipeline_shuffle = [
    {
        "split": ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        "force_group": "Sample_ID"
    },
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_shuffle, "force_group_shuffle")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Force_group with ShuffleSplit completed - groups respected!")

# =============================================================================
# Example 3: force_group with StratifiedKFold (classification)
# =============================================================================
# Combines stratification (balanced class distribution) with group-awareness!

print("\n" + "=" * 70)
print("Example 3: force_group with StratifiedKFold (stratified + grouped)")
print("=" * 70)

pipeline_stratified = [
    {
        "split": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        "force_group": "Sample_ID",
        "y_aggregation": "mode"  # Use mode for classification targets
    },
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_stratified, "force_group_stratified")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, _ = runner.run(pipeline_config, dataset_config)

print("Force_group with StratifiedKFold completed - balanced class distribution per fold!")

# =============================================================================
# Comparison: force_group vs native GroupKFold
# =============================================================================
print("\n" + "=" * 70)
print("Comparison: force_group (KFold) vs native GroupKFold")
print("=" * 70)

# Using force_group with KFold
pipeline_force = [
    "fold_Sample_ID",
    {"split": KFold(n_splits=3), "force_group": "Sample_ID"},
    "fold_Sample_ID",
]

pipeline_config = PipelineConfigs(pipeline_force, "force_group_comparison")
dataset_config = DatasetConfigs('sample_data/classification')
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions_force, _ = runner.run(pipeline_config, dataset_config)

# Using native GroupKFold
pipeline_native = [
    "fold_Sample_ID",
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"},
    "fold_Sample_ID",
]

pipeline_config = PipelineConfigs(pipeline_native, "native_group_comparison")
dataset_config = DatasetConfigs('sample_data/classification')
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions_native, _ = runner.run(pipeline_config, dataset_config)

print("\nBoth approaches ensure groups are never split across train/test!")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
