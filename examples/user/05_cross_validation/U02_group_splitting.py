"""
U02 - Group Splitting: Handling Grouped/Clustered Data
=======================================================

Prevent data leakage when samples are grouped.

This tutorial covers:

* Why group splitting matters
* GroupKFold and StratifiedGroupKFold
* Automatic group-awareness via repetition
* Visualizing group assignments

Prerequisites
-------------
Complete :ref:`U01_cv_strategies` first.

Next Steps
----------
See :ref:`U03_sample_filtering` for sample selection.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
)

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 Group Splitting Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Group Splitting?
# =============================================================================
print("\n" + "=" * 60)
print("U02 - Group Splitting")
print("=" * 60)

print("""
Group splitting prevents DATA LEAKAGE in clustered data.

Common scenarios requiring group splitting:

  📊 REPEATED MEASUREMENTS
     Multiple spectra from the same sample
     → Keep all measurements together

  📈 BIOLOGICAL SAMPLES
     Multiple measurements per patient/animal
     → Never split patient across train/test

  📉 TEMPORAL REPLICATES
     Measurements at different times from same source
     → Group by source identifier

Without proper group splitting:
  ❌ Model memorizes sample patterns
  ❌ Overly optimistic performance estimates
  ❌ Poor generalization to new samples
""")


# =============================================================================
# Section 2: GroupKFold - Native Group Splitter
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: GroupKFold - Native Group Splitter")
print("-" * 60)

print("""
GroupKFold is sklearn's group-aware K-fold splitter.
All samples from the same group stay together.
""")

pipeline_groupkfold = [
    # Visualize samples by Sample_ID before split
    "fold_Sample_ID",

    # GroupKFold with group parameter
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"},

    # Visualize after split - groups respected!
    "fold_Sample_ID",

    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_groupkfold = nirs4all.run(
    pipeline=pipeline_groupkfold,
    dataset="sample_data/classification",
    name="GroupKFold",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_groupkfold.best_rmse) * 100 if not np.isnan(result_groupkfold.best_rmse) else float('nan')
print(f"\nGroupKFold - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nGroupKFold - (see detailed metrics)")
print("Note: Groups (Sample_ID) are never split across train/test!")


# =============================================================================
# Section 3: StratifiedGroupKFold - Groups + Stratification
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: StratifiedGroupKFold - Groups + Stratification")
print("-" * 60)

print("""
StratifiedGroupKFold combines:
  - Group awareness (groups stay together)
  - Stratification (class balance preserved)
""")

pipeline_strat_group = [
    "fold_Sample_ID",

    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
     "group": "Sample_ID"},

    "fold_chart",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_strat_group = nirs4all.run(
    pipeline=pipeline_strat_group,
    dataset="sample_data/classification",
    name="StratifiedGroupKFold",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_strat_group.best_rmse) * 100 if not np.isnan(result_strat_group.best_rmse) else float('nan')
print(f"\nStratifiedGroupKFold - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nStratifiedGroupKFold - (see detailed metrics)")
print("Note: Groups respected AND class proportions preserved!")


# =============================================================================
# Section 4: Automatic Group Support via repetition
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Automatic Group Support via repetition")
print("-" * 60)

print("""
With repetition defined in DatasetConfigs, ANY splitter becomes group-aware!
Works with KFold, ShuffleSplit, StratifiedKFold, etc.

How it works:
  1. Define repetition column once in DatasetConfigs
  2. All splitters automatically respect repetition groups
  3. Use group_by for additional grouping columns
""")

# KFold with repetition - automatic group awareness
pipeline_auto_group = [
    "fold_Sample_ID",

    # KFold automatically respects repetition groups!
    KFold(n_splits=3, shuffle=True, random_state=42),

    "fold_Sample_ID",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_auto = nirs4all.run(
    pipeline=pipeline_auto_group,
    dataset=DatasetConfigs("sample_data/classification", repetition="Sample_ID"),
    name="AutoGroup_KFold",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_auto.best_rmse) * 100 if not np.isnan(result_auto.best_rmse) else float('nan')
print(f"\nKFold + repetition - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nKFold + repetition - (see detailed metrics)")


# =============================================================================
# Section 5: ShuffleSplit with repetition
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: ShuffleSplit with repetition")
print("-" * 60)

print("""
ShuffleSplit explicitly ignores groups in sklearn.
With repetition defined, nirs4all automatically makes it group-aware!
""")

pipeline_shuffle_group = [
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),

    "fold_chart",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_shuffle_group = nirs4all.run(
    pipeline=pipeline_shuffle_group,
    dataset=DatasetConfigs("sample_data/classification", repetition="Sample_ID"),
    name="AutoGroup_Shuffle",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_shuffle_group.best_rmse) * 100 if not np.isnan(result_shuffle_group.best_rmse) else float('nan')
print(f"\nShuffleSplit + repetition - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nShuffleSplit + repetition - (see detailed metrics)")


# =============================================================================
# Section 6: StratifiedKFold with repetition
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: StratifiedKFold with repetition")
print("-" * 60)

print("""
Combine stratification with group awareness using repetition.
Use y_aggregation to specify how to aggregate targets within groups.
""")

pipeline_strat_auto = [
    {"split": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
     "y_aggregation": "mode"},  # Use mode for classification targets

    "fold_chart",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_strat_auto = nirs4all.run(
    pipeline=pipeline_strat_auto,
    dataset=DatasetConfigs("sample_data/classification", repetition="Sample_ID"),
    name="AutoGroup_Stratified",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_strat_auto.best_rmse) * 100 if not np.isnan(result_strat_auto.best_rmse) else float('nan')
print(f"\nStratifiedKFold + repetition - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nStratifiedKFold + repetition - (see detailed metrics)")


# =============================================================================
# Section 7: Comparison Without vs With Group Splitting
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Comparison - No Groups vs With Groups")
print("-" * 60)

print("""
Comparing performance WITH and WITHOUT group-aware splitting.
Without group splitting: often overly optimistic!
""")

# WITHOUT group splitting (leakage possible)
pipeline_no_group = [
    StandardNormalVariate(),
    KFold(n_splits=3, shuffle=True, random_state=42),  # No group awareness
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_no_group = nirs4all.run(
    pipeline=pipeline_no_group,
    dataset="sample_data/classification",
    name="NoGroup",
    verbose=0
)

# WITH group splitting via repetition
pipeline_with_group = [
    StandardNormalVariate(),
    KFold(n_splits=3, shuffle=True, random_state=42),
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_with_group = nirs4all.run(
    pipeline=pipeline_with_group,
    dataset=DatasetConfigs("sample_data/classification", repetition="Sample_ID"),
    name="WithGroup",
    verbose=0
)

acc_no_group = (1 - result_no_group.best_rmse) * 100 if not np.isnan(result_no_group.best_rmse) else float('nan')
acc_with_group = (1 - result_with_group.best_rmse) * 100 if not np.isnan(result_with_group.best_rmse) else float('nan')

print(f"\nResults comparison:")
print(f"   WITHOUT group splitting: {acc_no_group:.1f}% (may be optimistic)" if not np.isnan(acc_no_group) else "   WITHOUT group splitting: (see detailed metrics)")
print(f"   WITH group splitting:    {acc_with_group:.1f}% (realistic)" if not np.isnan(acc_with_group) else "   WITH group splitting:    (see detailed metrics)")

if acc_no_group > acc_with_group:
    print("\n   ⚠️  Without groups appears better - likely due to data leakage!")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Group Splitting Methods:

  RECOMMENDED: Use repetition in DatasetConfigs (auto-groups all splitters):
    DatasetConfigs("path", repetition="Sample_ID")
    # Then ANY splitter respects groups automatically:
    KFold(n_splits=3)
    ShuffleSplit(n_splits=5)
    StratifiedKFold(n_splits=3)

  NATIVE GROUP SPLITTERS (still work with explicit "group" parameter):
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"}
    {"split": StratifiedGroupKFold(n_splits=3), "group": "Sample_ID"}

  ADDITIONAL GROUPING (combine with repetition):
    {"split": KFold(n_splits=3), "group_by": ["Year", "Location"]}
    # Groups by (Sample_ID, Year, Location) tuples

  OPT-OUT of repetition grouping:
    {"split": KFold(n_splits=3), "ignore_repetition": True}

y_aggregation Options (for stratified splitters):
  "mode"   - Most common value (classification)
  "mean"   - Average value (regression)

When to Use Group Splitting:
  ✓ Repeated measurements per sample
  ✓ Multiple spectra per individual
  ✓ Technical replicates
  ✓ Batch effects by sample source

Visualization:
  "fold_Sample_ID" - Visualize samples by Sample_ID column
  "fold_chart"     - Visualize train/test split

Next: U03_sample_filtering.py - Filter samples in pipeline
""")
