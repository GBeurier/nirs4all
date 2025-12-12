"""
Q31_outlier_branching.py - Outlier Branching Examples

This example demonstrates TWO outlier-based branching features of nirs4all:

1. **Outlier Excluder** (outlier_excluder): Creates branches where outliers are
   EXCLUDED from training but predictions are made on all samples. Useful for
   comparing "what if we removed outliers" scenarios.

2. **Sample Partitioner** (sample_partitioner): PARTITIONS samples into separate
   branches based on outlier status. Each branch contains a disjoint subset:
   - "outliers" branch: Contains ONLY the outlier samples
   - "inliers" branch: Contains ONLY the non-outlier samples
   Useful for training separate models for different data subsets.

Key differences:
- outlier_excluder: One model trained without outliers, predicts on all samples
- sample_partitioner: Two models, each trains and predicts only on its subset

Phase 7 of the branching feature implementation.
"""

# %%
# === Setup ===
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.visualization.predictions import PredictionAnalyzer

# %% [markdown]
# ## Part 1: Sample Partitioner (Separate Branches for Outliers/Inliers)
#
# The sample partitioner creates TWO branches with disjoint sample sets:
# - Branch "outliers": Contains ONLY samples detected as outliers
# - Branch "inliers": Contains ONLY samples NOT detected as outliers
#
# Each branch trains and predicts independently on its own subset.

# %%
# === Example 1: Basic Sample Partitioner Syntax ===
print("=" * 60)
print("Example 1: Sample Partitioner - Y Outlier Detection")
print("=" * 60)

pipeline_partition_y = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    SNV(),
    # Partition samples by Y outlier detection
    {"branch": {
        "by": "sample_partitioner",
        "filter": {"method": "y_outlier", "threshold": 1.5},  # IQR-based
    }},
    PLSRegression(n_components=5),
]

# Note: Replace with your actual dataset path
dataset_config = DatasetConfigs("sample_data/regression")
pipeline_config = PipelineConfigs(pipeline_partition_y)

runner = PipelineRunner(workspace_path="workspace")
predictions, pipelines = runner.run(pipeline_config, dataset_config)

print(f"\nTotal predictions: {len(predictions)}")
print(f"Branches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 2: Sample Partitioner with X Outliers
#
# You can also partition based on X (spectral) outlier detection methods.

# %%
# === Example 2: X Outlier Partitioning ===
print("\n" + "=" * 60)
print("Example 2: Sample Partitioner - X Outlier Detection")
print("=" * 60)

pipeline_partition_x = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    SNV(),
    # Partition samples by X outlier detection (Isolation Forest)
    {"branch": {
        "by": "sample_partitioner",
        "filter": {
            "method": "isolation_forest",
            "contamination": 0.10,  # Expect 10% outliers
        },
    }},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_partition_x)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nBranches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Example 3: Custom Branch Names
#
# You can provide custom names for the two partition branches.

# %%
# === Example 3: Custom Branch Names ===
print("\n" + "=" * 60)
print("Example 3: Custom Branch Names")
print("=" * 60)

pipeline_custom_names = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    SNV(),
    {"branch": {
        "by": "sample_partitioner",
        "filter": {
            "method": "y_outlier",
            "threshold": 2.0,
            "branch_names": ["extreme_values", "normal_range"],
        },
    }},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_custom_names)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nCustom branch names: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## Part 2: Outlier Excluder (Exclude Outliers from Training)
#
# The outlier excluder is different - it creates branches where outliers
# are EXCLUDED from training, but predictions are made on ALL samples.
# This is useful for comparing "baseline vs outlier-removed" scenarios.

# %%
# === Example 4: Outlier Excluder Comparison ===
print("\n" + "=" * 60)
print("Example 4: Outlier Excluder - Exclusion Strategies")
print("=" * 60)

pipeline_excluder = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    SNV(),
    # Branch by outlier exclusion strategy
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,  # Baseline - no exclusion
            {"method": "isolation_forest", "contamination": 0.05},
            {"method": "mahalanobis", "threshold": 3.0},
        ],
    }},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_excluder)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nBranches: {predictions.get_unique_values('branch_name')}")

# %% [markdown]
# ## When to Use Each Approach
#
# **Use Sample Partitioner when:**
# - You want separate models for different data subsets
# - You suspect outliers have fundamentally different patterns
# - You want to analyze outlier samples independently
# - You need to understand model behavior on outliers vs. normal samples
#
# **Use Outlier Excluder when:**
# - You want to compare "with outliers" vs "without outliers" training
# - You still need predictions on all samples
# - You're deciding whether to remove outliers from your pipeline
# - You're comparing different outlier detection methods

# %%
# === Example 5: Combined Approach ===
print("\n" + "=" * 60)
print("Example 5: Nested Preprocessing + Sample Partitioner")
print("=" * 60)

from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

pipeline_nested = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    MinMaxScaler(),
    # First: preprocessing branch
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
    }},
    # Second: sample partitioner (nested)
    {"branch": {
        "by": "sample_partitioner",
        "filter": {"method": "y_outlier", "threshold": 1.5},
    }},
    PLSRegression(n_components=5),
]

pipeline_config = PipelineConfigs(pipeline_nested)
predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nCombined branches: {predictions.get_unique_values('branch_name')}")
# Expected: snv_y_outliers, snv_y_inliers, msc_y_outliers, msc_y_inliers

# %% [markdown]
# ## Summary
#
# **Sample Partitioner** (`sample_partitioner`):
# - Creates 2 branches: outliers and inliers
# - Each branch trains and predicts on its own subset
# - Samples are PARTITIONED (disjoint sets)
# - Syntax: `{"branch": {"by": "sample_partitioner", "filter": {...}}}`
#
# **Outlier Excluder** (`outlier_excluder`):
# - Creates N branches based on exclusion strategies
# - Each branch excludes different samples from training
# - Predictions are made on ALL samples
# - Syntax: `{"branch": {"by": "outlier_excluder", "strategies": [...]}}`
#
# Both features support:
# - Y-based outlier detection (IQR)
# - X-based outlier detection (Isolation Forest, Mahalanobis, LOF, etc.)
# - Nesting with other branch types
# - Full integration with prediction storage and visualization

print("\n" + "=" * 60)
print("Outlier branching examples completed!")
print("=" * 60)
