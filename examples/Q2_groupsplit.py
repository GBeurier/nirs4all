"""
Q2 GroupSplit Example - Native Group-Aware Splitters
=====================================================
Demonstrates group-based splitting using sklearn's native group splitters
(GroupKFold, StratifiedGroupKFold) with Sample_ID metadata.

Group splitting ensures that all samples from the same group (e.g., repeated
measurements of the same individual) stay together in train/test, preventing
data leakage.

Two approaches for group-aware splitting in nirs4all:

1. Native group splitters (this file):
   - Use `group` parameter with GroupKFold, StratifiedGroupKFold, etc.
   - Syntax: {"split": GroupKFold(n_splits=3), "group": "Sample_ID"}

2. Universal force_group (see Q2B_force_group.py):
   - Use `force_group` with ANY splitter (KFold, ShuffleSplit, StratifiedKFold)
   - Syntax: {"split": KFold(n_splits=3), "force_group": "Sample_ID"}
   - More flexible, works with all sklearn splitters
"""

import argparse

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q2 GroupSplit Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

# =============================================================================
# Example 1: GroupKFold with Sample_ID
# =============================================================================
# GroupKFold is sklearn's native group-aware K-fold cross-validator

print("\n" + "=" * 70)
print("Example 1: GroupKFold with Sample_ID")
print("=" * 70)

data_path = 'sample_data/classification'

pipeline = [
    "fold_Sample_ID",  # Visualize samples grouped by Sample_ID before split
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"},
    "fold_Sample_ID",  # Visualize after split - groups are respected!
    # RandomForestClassifier(max_depth=30)
]

pipeline_config = PipelineConfigs(pipeline, "Q2_groupkfold")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# =============================================================================
# Example 2: StratifiedGroupKFold - Groups + Stratification
# =============================================================================
# StratifiedGroupKFold combines group-awareness with stratified class distribution

print("\n" + "=" * 70)
print("Example 2: StratifiedGroupKFold (groups + stratification)")
print("=" * 70)

pipeline_stratified = [
    "fold_Sample_ID",
    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "Sample_ID"},
    "fold_Sample_ID",
    "fold_chart",
]

pipeline_config = PipelineConfigs(pipeline_stratified, "Q2_stratified_groupkfold")
dataset_config = DatasetConfigs(data_path)
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

print("\n" + "=" * 70)
print("TIP: For more flexibility, see Q2B_force_group.py which shows how to")
print("use force_group with ANY splitter (KFold, ShuffleSplit, etc.)")
print("=" * 70)
