"""
D03 - Repetition Transform: Handling Repeated Measurements
===========================================================

rep_to_sources and rep_to_pp convert repeated measurements
into multi-source format or preprocessing variations.

This tutorial covers:

* Repeated measurements problem
* rep_to_sources: Convert to multi-source format
* rep_to_pp: Create preprocessing variations
* Aggregation strategies
* Integration with pipelines

Prerequisites
-------------
- 01_branching/D04_merge_sources for multi-source concepts

Next Steps
----------
See 06_internals/D01_session_workflow for session management.

Duration: ~4 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D03 Repetition Transform Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D03 - Repetition Transform: Handling Repeated Measurements")
print("=" * 60)

print("""
Spectral data often has repeated measurements:
  - Multiple scans per sample (technical replicates)
  - Sequential measurements (time series)
  - Different angles/positions

Data format:
  Original: (n_samples, n_wavelengths × n_repetitions)

Options to handle:
  1. rep_to_sources: Treat each repetition as a source
  2. rep_to_pp: Create preprocessing variations
  3. Aggregate: Mean/median across repetitions
""")


# =============================================================================
# Section 1: The Repeated Measurements Problem
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: The Repeated Measurements Problem")
print("-" * 60)

print("""
Common scenario:
  - 3 scans per sample
  - 200 wavelengths each
  - Data shape: (100 samples, 600 features)

Challenge: How to use the repetition information?
""")

# Simulate repeated measurement data
np.random.seed(42)
n_samples = 100
n_wavelengths = 200
n_repetitions = 3

# Simulated data with repetitions concatenated
X_repeated = np.random.randn(n_samples, n_wavelengths * n_repetitions)

# Add some structure
base_signal = np.random.randn(n_samples, n_wavelengths)
for i in range(n_repetitions):
    start = i * n_wavelengths
    end = (i + 1) * n_wavelengths
    # Each rep is base + noise
    X_repeated[:, start:end] = base_signal + np.random.randn(n_samples, n_wavelengths) * 0.1

print(f"Simulated data shape: {X_repeated.shape}")
print(f"  = {n_samples} samples × ({n_wavelengths} wavelengths × {n_repetitions} reps)")


# =============================================================================
# Section 2: rep_to_sources - Convert to Multi-Source
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: rep_to_sources - Convert to Multi-Source")
print("-" * 60)

print("""
rep_to_sources splits repetitions into separate sources:

    {"rep_to_sources": {
        "n_repetitions": 3,
        "n_features_per_rep": 200,
    }}

Before: (100, 600) - single matrix
After:  {"rep_0": (100, 200), "rep_1": (100, 200), "rep_2": (100, 200)}

Can then use source_branch for per-repetition processing.
""")

pipeline_rep_sources = [
    {"rep_to_sources": {
        "n_repetitions": 3,
        "n_features_per_rep": 200,
    }},
    {"source_branch": {
        "rep_0": [SNV()],
        "rep_1": [SNV()],
        "rep_2": [SNV()],
    }},
    {"merge_sources": "concat"},
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    PLSRegression(n_components=5),
]

print("Pipeline with rep_to_sources:")
print("  1. Split into 3 sources (one per rep)")
print("  2. SNV each source independently")
print("  3. Merge sources back together")


# =============================================================================
# Section 3: rep_to_pp - Preprocessing Variations
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: rep_to_pp - Preprocessing Variations")
print("-" * 60)

print("""
rep_to_pp creates preprocessing variations from repetitions:

    {"rep_to_pp": {
        "n_repetitions": 3,
        "aggregation": "mean",  # How to combine
        "preprocessing": SNV(),  # Apply before aggregation
    }}

Options:
  - aggregation: "mean", "median", "std", "concat"
  - preprocessing: Applied to each rep before combining
""")

pipeline_rep_pp = [
    {"rep_to_pp": {
        "n_repetitions": 3,
        "n_features_per_rep": 200,
        "aggregation": "mean",
        "preprocessing": SNV(),
    }},
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    PLSRegression(n_components=5),
]

print("Pipeline with rep_to_pp:")
print("  1. Split into 3 repetitions")
print("  2. Apply SNV to each")
print("  3. Average across repetitions")
print("  Result: (100, 200) - single clean spectrum")


# =============================================================================
# Section 4: Aggregation Strategies
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Aggregation Strategies")
print("-" * 60)

print("""
Different aggregation methods:

┌────────────┬─────────────────────────────────────────┐
│ Method     │ Description                             │
├────────────┼─────────────────────────────────────────┤
│ mean       │ Average (reduces noise)                 │
│ median     │ Robust average (handles outliers)       │
│ std        │ Variability (for quality assessment)    │
│ concat     │ Keep all (for multi-source)             │
│ first      │ First repetition only                   │
│ best       │ Best by quality metric                  │
└────────────┴─────────────────────────────────────────┘
""")

# Compare aggregation methods
aggregation_methods = ["mean", "median", "std"]

for method in aggregation_methods:
    # Reshape data
    X_reshaped = X_repeated.reshape(n_samples, n_repetitions, n_wavelengths)

    if method == "mean":
        X_agg = X_reshaped.mean(axis=1)
    elif method == "median":
        X_agg = np.median(X_reshaped, axis=1)
    elif method == "std":
        X_agg = X_reshaped.std(axis=1)

    print(f"  {method}: shape = {X_agg.shape}, mean = {X_agg.mean():.4f}")


# =============================================================================
# Section 5: Quality-Based Selection
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Quality-Based Selection")
print("-" * 60)

print("""
Select best repetition based on quality metric:

    {"rep_to_pp": {
        "n_repetitions": 3,
        "selection": "best",
        "quality_metric": "variance",  # Lower variance = less noise
    }}

Automatically picks the cleanest repetition per sample.
""")


# =============================================================================
# Section 6: Combining with Source Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Combining with Source Branching")
print("-" * 60)

print("""
Complex workflow: per-rep preprocessing then fusion

    1. rep_to_sources: Split repetitions
    2. source_branch: Per-rep preprocessing
    3. merge_sources: Combine results
    4. Model training
""")

pipeline_complex = [
    # Split repetitions into sources
    {"rep_to_sources": {
        "n_repetitions": 3,
        "n_features_per_rep": 200,
    }},

    # Different preprocessing per repetition
    {"source_branch": {
        "rep_0": [SNV(), FirstDerivative()],   # First rep: aggressive preprocessing
        "rep_1": [SNV()],                      # Second rep: moderate
        "rep_2": [MSC()],                      # Third rep: different approach
    }},

    # Merge all sources
    {"merge_sources": {
        "mode": "concat",
        "weights": {"rep_0": 1.0, "rep_1": 1.0, "rep_2": 1.0}
    }},

    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    PLSRegression(n_components=10),
]

print("Complex repetition handling pipeline defined")


# =============================================================================
# Section 7: Time Series Repetitions
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Time Series Repetitions")
print("-" * 60)

print("""
For sequential measurements (time series):

    {"rep_to_pp": {
        "n_repetitions": 3,
        "time_aware": True,
        "features": ["current", "delta", "trend"],
    }}

Features:
  - current: Latest measurement
  - delta: Change from previous
  - trend: Linear trend across repetitions
""")


# =============================================================================
# Section 8: Practical Recommendations
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Practical Recommendations")
print("-" * 60)

print("""
📋 When to use each approach:

┌─────────────────────┬─────────────────────────────────────┐
│ Scenario            │ Recommended Approach                │
├─────────────────────┼─────────────────────────────────────┤
│ Technical           │ rep_to_pp with mean aggregation     │
│ replicates          │ (reduces measurement noise)         │
├─────────────────────┼─────────────────────────────────────┤
│ Different           │ rep_to_sources with source_branch   │
│ conditions          │ (per-condition preprocessing)       │
├─────────────────────┼─────────────────────────────────────┤
│ Quality varies      │ rep_to_pp with "best" selection     │
│                     │ (pick cleanest measurement)         │
├─────────────────────┼─────────────────────────────────────┤
│ All info needed     │ rep_to_sources with concat merge    │
│                     │ (keep all repetitions)              │
└─────────────────────┴─────────────────────────────────────┘
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. rep_to_sources converts to multi-source format
2. rep_to_pp applies preprocessing and aggregates
3. Aggregation: mean, median, std, concat, best
4. Quality-based selection for noisy data
5. Combine with source_branch for per-rep control
6. Time-aware features for sequential data

Key syntax:
    {"rep_to_sources": {
        "n_repetitions": N,
        "n_features_per_rep": M
    }}

    {"rep_to_pp": {
        "n_repetitions": N,
        "aggregation": "mean"|"median"|"std"|"concat",
        "preprocessing": transform
    }}

Next: 06_internals/D01_session_workflow.py - Session management
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
