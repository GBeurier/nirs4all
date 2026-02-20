"""
U01 - Flexible Inputs: Different Data Format Options
=====================================================

Demonstrates all possible input formats for datasets and pipelines.

This tutorial covers:

* Direct numpy array input with ``(X, y)`` tuples
* Dictionary-based dataset configuration
* Partition info specification
* SpectroDataset object usage
* Flexible pipeline formats

Prerequisites
-------------
Complete the getting_started examples first.

Next Steps
----------
See :ref:`U02_multi_datasets` for analyzing multiple datasets.

Duration: ~2 minutes
Difficulty: ★☆☆☆☆
"""

# Standard library imports
import argparse
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Third-party imports
from sklearn.preprocessing import StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs, SpectroDataset
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 Flexible Inputs Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Generate Sample Data
# =============================================================================
print("\n" + "=" * 60)
print("U01 - Flexible Input Formats")
print("=" * 60)

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 200
n_features = 100

X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

print("\n📊 Generated synthetic data:")
print(f"   Samples: {n_samples}, Features: {n_features}")

# Define a simple pipeline
pipeline_steps = [
    {"preprocessing": StandardScaler()},
    {"model": Ridge(alpha=1.0)}
]

# =============================================================================
# Section 2: Traditional Approach (PipelineConfigs + DatasetConfigs)
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Traditional Approach")
print("-" * 60)

runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

pipeline_configs = PipelineConfigs(pipeline_steps, name="traditional")
dataset_configs = DatasetConfigs({
    "name": "traditional_dataset",
    "train_x": X[:160],
    "train_y": y[:160],
    "test_x": X[160:],
    "test_y": y[160:]
})

result1, _ = runner.run(pipeline_configs, dataset_configs)
print("   ✓ PipelineConfigs + DatasetConfigs works!")

# =============================================================================
# Section 3: Direct List + Tuple (Simplest!)
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Direct List + Tuple (Simplest)")
print("-" * 60)

runner2 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

# Just pass list of steps and tuple of (X, y, partition_info)
partition_info: dict[str, Any] = {"train": 160}  # First 160 samples for training
result2, _ = runner2.run(
    pipeline=pipeline_steps,
    dataset=(X, y, partition_info),  # type: ignore[arg-type]
    pipeline_name="direct",
    dataset_name="array_data"
)
print("   ✓ Direct list + tuple: pipeline_steps, (X, y, {'train': 160})")

# =============================================================================
# Section 4: Dictionary Dataset Configuration
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Dict Dataset Config")
print("-" * 60)

runner3 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

dataset_dict = {
    "name": "dict_config",
    "train_x": X[:160],
    "train_y": y[:160],
    "test_x": X[160:],
    "test_y": y[160:]
}

result3, _ = runner3.run(
    pipeline=pipeline_steps,
    dataset=dataset_dict
)
print("   ✓ Dict dataset: explicit train_x, train_y, test_x, test_y")

# =============================================================================
# Section 5: SpectroDataset Object
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: SpectroDataset Object")
print("-" * 60)

runner4 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

# Create SpectroDataset manually
dataset = SpectroDataset(name="custom_spectro")
dataset.add_samples(X[:160], indexes={"partition": "train"})
dataset.add_targets(y[:160])
dataset.add_samples(X[160:], indexes={"partition": "test"})
dataset.add_targets(y[160:])

result4, _ = runner4.run(
    pipeline=pipeline_steps,
    dataset=dataset
)
print("   ✓ SpectroDataset: full control over data structure")

# =============================================================================
# Section 6: Custom Partition Indices
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Custom Train/Test Indices")
print("-" * 60)

runner5 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

# Use explicit slices
partition_advanced = {
    "train": slice(0, 150),
    "test": slice(150, 200)
}

result5, _ = runner5.run(
    pipeline=pipeline_steps,
    dataset=(X, y, partition_advanced),  # type: ignore[arg-type]
    dataset_name="custom_split"
)
print("   ✓ Custom indices with slices: {'train': slice(0, 150), 'test': slice(150, 200)}")

# =============================================================================
# Section 7: Pipeline with Cross-Validation
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Pipeline with CV")
print("-" * 60)

runner6 = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

# Pipeline with embedded cross-validation
cv_pipeline = [
    {"preprocessing": StandardScaler()},
    KFold(n_splits=3, shuffle=True, random_state=42),
    {"model": Ridge(alpha=1.0)}
]

result6, _ = runner6.run(
    pipeline=cv_pipeline,
    dataset=(X, y, {"train": 160}),  # type: ignore[arg-type]
    pipeline_name="cv_pipeline"
)
print("   ✓ Pipeline with KFold cross-validation embedded")

# =============================================================================
# Section 8: Using nirs4all.run() Directly
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Using nirs4all.run()")
print("-" * 60)

result7 = nirs4all.run(
    pipeline=[
        StandardScaler(),
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": Ridge(alpha=1.0)}
    ],
    dataset=(X, y, {"train": 160}),  # type: ignore[arg-type]
    name="nirs4all_run",
    verbose=0
)
print("   ✓ nirs4all.run() with direct array input")
print(f"     Generated {result7.num_predictions} predictions")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Dataset Input Formats:

  1. Folder path (string)
     dataset="sample_data/regression"

  2. Tuple (X, y) or (X, y, partition_info)
     dataset=(X, y, {"train": 160})

  3. Dictionary config
     dataset={"train_x": X_train, "train_y": y_train, ...}

  4. DatasetConfigs object
     dataset=DatasetConfigs("path/to/data")

  5. SpectroDataset object
     dataset=SpectroDataset(name="my_data")

Pipeline Input Formats:

  1. List of steps
     pipeline=[MinMaxScaler(), PLSRegression()]

  2. PipelineConfigs object
     pipeline=PipelineConfigs(steps, name="my_pipeline")

Partition Info Options:

  {"train": 160}              # First 160 = train, rest = test
  {"train": slice(0, 150)}    # Explicit slice
  {"train": [0,1,2,...]}      # Explicit indices

Key Takeaways:
  • No need to wrap in PipelineConfigs/DatasetConfigs
  • Direct numpy array support for quick experiments
  • All formats work with nirs4all.run() and PipelineRunner.run()
  • Old code still works (backward compatible)

Next: U02_multi_datasets.py - Analyzing multiple datasets
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
