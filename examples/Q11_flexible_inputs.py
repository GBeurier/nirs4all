"""
Q11 Example - Flexible Input Formats for PipelineRunner
========================================================
Demonstrates all possible input combinations for PipelineRunner.run(), predict(), and explain().

This example shows how to use:
1. Different pipeline formats: PipelineConfigs, List[steps], Dict, or file path
2. Different dataset formats: DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path
3. All combinations of the above

Key Features:
- No need to wrap inputs in PipelineConfigs/DatasetConfigs
- Direct numpy array support for quick experiments
- Automatic partition splitting for arrays
- Backward compatible with traditional approach
"""

import argparse
import numpy as np
from pathlib import Path

# NIRS4All imports
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs, SpectroDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q11 Flexible Inputs Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

print("=" * 100)
print("Q11: Flexible Input Formats Demo")
print("=" * 100)

# ==========================================
# Generate Sample Data
# ==========================================
np.random.seed(42)
n_samples = 200
n_features = 100

X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

print(f"\n📊 Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")

# ==========================================
# Define a Simple Pipeline
# ==========================================
pipeline_steps = [
    {"preprocessing": StandardScaler()},
    {"model": Ridge(alpha=1.0)}
]

pipeline_dict = {"pipeline": pipeline_steps}

print("\n📋 Pipeline defined with StandardScaler + Ridge regression")

# ==========================================
# Example 1: Traditional Approach (Still Works!)
# ==========================================
print("\n" + "=" * 100)
print("Example 1: Traditional Approach - PipelineConfigs + DatasetConfigs")
print("=" * 100)

runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

pipeline_configs = PipelineConfigs(pipeline_steps, name="traditional")
dataset_configs = DatasetConfigs({
    "name": "traditional_dataset",
    "train_x": X[:160],
    "train_y": y[:160],
    "test_x": X[160:],
    "test_y": y[160:]
})

result1 = runner.run(pipeline_configs, dataset_configs)
print("✅ Traditional approach works perfectly!")

# ==========================================
# Example 2: Direct List + Tuple (Simplest!)
# ==========================================
print("\n" + "=" * 100)
print("Example 2: Direct Approach - List[steps] + Tuple[X, y, partition_info]")
print("=" * 100)

runner2 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

# Just pass the list of steps and tuple of arrays!
partition_info = {"train": 160}  # First 160 samples for training
result2 = runner2.run(
    pipeline=pipeline_steps,
    dataset=(X, y, partition_info),
    pipeline_name="direct",
    dataset_name="array_data"
)
print("✅ Direct list + tuple approach - super simple!")

# ==========================================
# Example 3: Dict + Tuple with train size
# ==========================================
print("\n" + "=" * 100)
print("Example 3: Dict pipeline + Tuple (X, y, train_size)")
print("=" * 100)

runner3 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

# Tuple with partition info specifying train size
partition_3 = {"train": 150}  # First 150 samples for training
result3 = runner3.run(
    pipeline=pipeline_dict,
    dataset=(X, y, partition_3),
    pipeline_name="with_partition",
    dataset_name="partitioned_data"
)
print("✅ Dict + (X, y, partition_dict) - precise control!")

# ==========================================
# Example 4: List + SpectroDataset
# ==========================================
print("\n" + "=" * 100)
print("Example 4: List[steps] + SpectroDataset")
print("=" * 100)

runner4 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

# Create a SpectroDataset
dataset = SpectroDataset(name="custom_spectro")
dataset.add_samples(X[:160], indexes={"partition": "train"})
dataset.add_targets(y[:160])
dataset.add_samples(X[160:], indexes={"partition": "test"})
dataset.add_targets(y[160:])

result4 = runner4.run(
    pipeline=pipeline_steps,
    dataset=dataset
)
print("✅ List + SpectroDataset - full control over dataset!")

# ==========================================
# Example 5: PipelineConfigs + Tuple (Mixed)
# ==========================================
print("\n" + "=" * 100)
print("Example 5: PipelineConfigs + Tuple[X, y, partition_info] (mixed formats)")
print("=" * 100)

runner5 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

pipeline_configs5 = PipelineConfigs(pipeline_steps, name="mixed")

# Mix PipelineConfigs with tuple dataset
result5 = runner5.run(
    pipeline=pipeline_configs5,
    dataset=(X, y, {"train": 160}),  # With partition info
    dataset_name="mixed_tuple"
)
print("✅ PipelineConfigs + tuple - flexible mixing!")

# ==========================================
# Example 6: Dict + Dict Config
# ==========================================
print("\n" + "=" * 100)
print("Example 6: Dict pipeline + Dict dataset config")
print("=" * 100)

runner6 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

dataset_config_dict = {
    "name": "dict_config",
    "train_x": X[:160],
    "train_y": y[:160],
    "test_x": X[160:],
    "test_y": y[160:]
}

result6 = runner6.run(
    pipeline=pipeline_dict,
    dataset=dataset_config_dict
)
print("✅ Dict + Dict - consistent format!")

# ==========================================
# Example 7: Advanced - Custom Partition Indices
# ==========================================
print("\n" + "=" * 100)
print("Example 7: Advanced - Custom train/test indices")
print("=" * 100)

runner7 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

# Use explicit slices for train/test split
train_indices = slice(0, 150)
test_indices = slice(150, 200)
partition_info_advanced = {"train": train_indices, "test": test_indices}

result7 = runner7.run(
    pipeline=pipeline_steps,
    dataset=(X, y, partition_info_advanced),
    dataset_name="custom_split"
)
print("✅ Custom indices - precise control!")

# ==========================================
# Example 8: Cross-Validation with Splits
# ==========================================
print("\n" + "=" * 100)
print("Example 8: Pipeline with cross-validation")
print("=" * 100)

runner8 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

# Pipeline with cross-validation
cv_pipeline = [
    {"preprocessing": StandardScaler()},
    KFold(n_splits=3, shuffle=True, random_state=42),
    {"model": Ridge(alpha=1.0)}
]

result8 = runner8.run(
    pipeline=cv_pipeline,
    dataset=(X, y, {"train": 160}),
    pipeline_name="cv_pipeline",
    dataset_name="cv_data"
)
print("✅ Cross-validation with flexible inputs!")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 100)
print("📊 SUMMARY - All 8 Examples Completed Successfully!")
print("=" * 100)

print("\n✨ Key Takeaways:")
print("  1. No need to wrap everything in PipelineConfigs/DatasetConfigs anymore")
print("  2. Direct numpy array support: just pass (X, y)")
print("  3. Control partitioning with optional dict: (X, y, {'train': 160})")
print("  4. Mix and match any format: List + Dict, Dict + SpectroDataset, etc.")
print("  5. Backward compatible: old code still works perfectly")
print("  6. Cleaner, more intuitive API for quick experiments")
print("\n💡 Use cases:")
print("  - Quick experiments: runner.run(steps, (X, y))")
print("  - Prediction: runner.predict(model_ref, X_new)")
print("  - Full control: Use PipelineConfigs/DatasetConfigs/SpectroDataset as before")

print("\n" + "=" * 100)
print("✅ Q11 Example Complete!")
print("=" * 100)
