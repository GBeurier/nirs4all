"""
D03 - Merge Basics: Combining Branch Outputs
=============================================

After branching, use ``merge`` to combine outputs and continue with
a single pipeline path.

This tutorial covers:

* Feature merging: concatenate features from branches
* Prediction merging: collect OOF predictions for stacking
* Mixed merging: combine features and predictions selectively
* Per-branch selection and aggregation

Prerequisites
-------------
- D01_branching_basics for understanding branches

Next Steps
----------
See D04_merge_sources for multi-source data handling.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SavitzkyGolay
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D03 Merge Basics Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D03 - Merge Basics: Combining Branch Outputs")
print("=" * 60)

print("""
After branching, the pipeline has N parallel contexts. The ``merge`` step
combines these into a single path for further processing.

Merge modes:
  "features"    - Concatenate feature matrices horizontally
  "predictions" - Collect OOF predictions (for stacking)
  {...}         - Mixed selection with dict syntax

Important: ``merge`` ALWAYS exits branch mode.
""")


# =============================================================================
# Section 1: Feature Merging
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Feature Merging")
print("-" * 60)

print("""
Feature merging concatenates X matrices from all branches horizontally.
Use case: Combine multiple preprocessing views into one feature space.

     Before merge:              After merge:
     ┌────────────┐
     │ Branch 0   │  shape: (n, p)
     ├────────────┤                   ┌─────────────────────────────┐
     │ Branch 1   │  shape: (n, p) →  │ Merged X                    │
     ├────────────┤                   └─────────────────────────────┘
     │ Branch 2   │  shape: (n, p)              shape: (n, 3*p)
     └────────────┘
""")

pipeline_feature_merge = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": "features"},  # Concatenate X from all branches
    PLSRegression(n_components=5),  # Train on merged features
]

result_feature = nirs4all.run(
    pipeline=pipeline_feature_merge,
    dataset="sample_data/regression",
    name="FeatureMerge",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions after feature merge: {result_feature.num_predictions}")
# After merge, we're back to single-path (no branch column)
print("Branch mode: exited (single pipeline path)")


# =============================================================================
# Section 2: Prediction Merging (Stacking Level 1)
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Prediction Merging (Stacking)")
print("-" * 60)

print("""
Prediction merging collects OOF (Out-Of-Fold) predictions from branches.
This is the foundation for stacking:

     Before merge:              After merge:
     ┌────────────┐
     │ Branch 0   │  pred: (n,)
     ├────────────┤                   ┌──────────────────────┐
     │ Branch 1   │  pred: (n,)  →    │ Meta X               │
     ├────────────┤                   └──────────────────────┘
     │ Branch 2   │  pred: (n,)              shape: (n, 3)
     └────────────┘

The predictions become features for a meta-model (Level 2).
""")

pipeline_prediction_merge = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=5)],
        "ridge": [MSC(), Ridge(alpha=1.0)],
        "rf": [FirstDerivative(), RandomForestRegressor(n_estimators=50, random_state=42)],
    }},
    {"merge": "predictions"},  # Collect OOF predictions
    Ridge(alpha=0.1),  # Meta-model trains on stacked predictions
]

result_prediction = nirs4all.run(
    pipeline=pipeline_prediction_merge,
    dataset="sample_data/regression",
    name="PredictionMerge",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMeta-model predictions: {result_prediction.num_predictions}")
print("This is basic 2-level stacking!")


# =============================================================================
# Section 3: Mixed Merging
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Mixed Merging")
print("-" * 60)

print("""
Mixed merging selects features from some branches and predictions from others:

    {"merge": {"features": [0, 1], "predictions": [2]}}

Branch 0-1 contribute X features, Branch 2 contributes predictions.
""")

pipeline_mixed_merge = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": [
        [SNV()],  # Branch 0
        [MSC()],  # Branch 1
        [FirstDerivative(), PLSRegression(n_components=5)],  # Branch 2
    ]},
    # Features from branches 0-1, predictions from branch 2
    {"merge": {"features": [0, 1], "predictions": [2]}},
    Ridge(alpha=0.1),
]

result_mixed = nirs4all.run(
    pipeline=pipeline_mixed_merge,
    dataset="sample_data/regression",
    name="MixedMerge",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMixed merge predictions: {result_mixed.num_predictions}")


# =============================================================================
# Section 4: Per-Branch Selection
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Per-Branch Selection")
print("-" * 60)

print("""
Select specific branches by index:

    {"merge": {"features": [0, 2]}}  # Select branches 0 and 2

Unselected branches are discarded.
""")

pipeline_selection = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": [
        [SNV()],      # Branch 0: snv
        [MSC()],      # Branch 1: msc
        [SavitzkyGolay(window_length=11, polyorder=2)],  # Branch 2: savgol
        [FirstDerivative()],  # Branch 3: derivative
    ]},
    # Only use branches 0 (snv) and 3 (derivative), discard msc and savgol
    {"merge": {"features": [0, 3]}},
    PLSRegression(n_components=5),
]

result_selection = nirs4all.run(
    pipeline=pipeline_selection,
    dataset="sample_data/regression",
    name="BranchSelection",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nSelected branch merge: {result_selection.num_predictions}")


# =============================================================================
# Section 5: Aggregation Instead of Concatenation
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Aggregation Instead of Concatenation")
print("-" * 60)

print("""
Instead of concatenating features, aggregate them:

    {"merge": {"features": "all", "aggregation": "mean"}}

Aggregation options: "mean", "median", "std", "min", "max"
""")

pipeline_aggregation = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},
    # Average features across branches instead of concatenating
    {"merge": {"features": "all", "aggregation": "mean"}},
    PLSRegression(n_components=5),
]

result_aggregation = nirs4all.run(
    pipeline=pipeline_aggregation,
    dataset="sample_data/regression",
    name="FeatureAggregation",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nAggregated feature merge: {result_aggregation.num_predictions}")


# =============================================================================
# Section 6: Nested Branching with Sequential Merges
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Nested Branching with Sequential Merges")
print("-" * 60)

print("""
Multiple branch-merge cycles enable complex architectures:

    branch → process → merge → branch → process → merge

Each merge exits the current branch level.
""")

pipeline_nested = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # First branching: preprocessing exploration
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
    }},
    {"merge": "features"},  # Exit first branch level

    # Second branching: model comparison
    {"branch": {
        "pls": [PLSRegression(n_components=5)],
        "ridge": [Ridge(alpha=1.0)],
    }},
    {"merge": "predictions"},  # Stack model predictions

    # Final meta-model
    Ridge(alpha=0.1),
]

result_nested = nirs4all.run(
    pipeline=pipeline_nested,
    dataset="sample_data/regression",
    name="NestedBranching",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nNested branching predictions: {result_nested.num_predictions}")


# =============================================================================
# Section 7: Merge with Original Features
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Merge with Original Features")
print("-" * 60)

print("""
Combine branch features with original (pre-branch) features:

    {"merge": {"features": "all", "include_original": True}}

Useful for preserving raw spectral data alongside preprocessed views.
""")

pipeline_with_original = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "snv": [SNV()],
        "derivative": [FirstDerivative()],
    }},
    {"merge": {"features": "all", "include_original": True}},
    PLSRegression(n_components=5),
]

result_with_original = nirs4all.run(
    pipeline=pipeline_with_original,
    dataset="sample_data/regression",
    name="MergeWithOriginal",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMerge with original: {result_with_original.num_predictions}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. "features" merge: horizontal concatenation of X matrices
2. "predictions" merge: collect OOF predictions for stacking
3. Mixed merge: select features/predictions per branch
4. Per-branch selection: filter branches by name or index
5. Aggregation: mean/median/etc. instead of concatenation
6. Nested branching: multiple branch-merge cycles
7. include_original: preserve pre-branch features

Key concepts:
- merge ALWAYS exits branch mode
- Prediction merge reconstructs OOF to prevent data leakage
- Use dict syntax for fine-grained control

Next: D04_merge_sources.py - Multi-source data handling
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
