"""
U18 - Group Splitting: Handling Grouped/Clustered Data
=======================================================

Prevent data leakage when samples are grouped.

This tutorial covers:

* Why group splitting matters
* GroupKFold and StratifiedGroupKFold
* force_group for any splitter
* Visualizing group assignments

Prerequisites
-------------
Complete :ref:`U17_cv_strategies` first.

Next Steps
----------
See :ref:`U19_sample_filtering` for sample selection.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
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
from nirs4all.operators.transforms import StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U18 Group Splitting Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Group Splitting?
# =============================================================================
print("\n" + "=" * 60)
print("U18 - Group Splitting")
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

accuracy = (1 - result_groupkfold.best_rmse) * 100
print(f"\nGroupKFold - Accuracy: {accuracy:.1f}%")
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

accuracy = (1 - result_strat_group.best_rmse) * 100
print(f"\nStratifiedGroupKFold - Accuracy: {accuracy:.1f}%")
print("Note: Groups respected AND class proportions preserved!")


# =============================================================================
# Section 4: force_group - Universal Group Support
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: force_group - Universal Group Support")
print("-" * 60)

print("""
force_group makes ANY sklearn splitter group-aware!
Works with KFold, ShuffleSplit, StratifiedKFold, etc.

How it works:
  1. Aggregates samples into "virtual groups"
  2. Passes virtual samples to the splitter
  3. Maps indices back to original samples
""")

# KFold with force_group
pipeline_force_kfold = [
    "fold_Sample_ID",

    # KFold doesn't natively support groups
    # force_group makes it work!
    {"split": KFold(n_splits=3, shuffle=True, random_state=42),
     "force_group": "Sample_ID"},

    "fold_Sample_ID",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_force = nirs4all.run(
    pipeline=pipeline_force_kfold,
    dataset="sample_data/classification",
    name="ForceGroup_KFold",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_force.best_rmse) * 100
print(f"\nKFold + force_group - Accuracy: {accuracy:.1f}%")


# =============================================================================
# Section 5: force_group with ShuffleSplit
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: force_group with ShuffleSplit")
print("-" * 60)

print("""
ShuffleSplit explicitly ignores groups in sklearn.
force_group fixes this!
""")

pipeline_shuffle_group = [
    {"split": ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
     "force_group": "Sample_ID"},

    "fold_chart",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_shuffle_group = nirs4all.run(
    pipeline=pipeline_shuffle_group,
    dataset="sample_data/classification",
    name="ForceGroup_Shuffle",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_shuffle_group.best_rmse) * 100
print(f"\nShuffleSplit + force_group - Accuracy: {accuracy:.1f}%")


# =============================================================================
# Section 6: force_group with StratifiedKFold
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: force_group with StratifiedKFold")
print("-" * 60)

print("""
Combine stratification with group awareness using force_group.
Use y_aggregation to specify how to aggregate targets within groups.
""")

pipeline_strat_force = [
    {"split": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
     "force_group": "Sample_ID",
     "y_aggregation": "mode"},  # Use mode for classification targets

    "fold_chart",
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_strat_force = nirs4all.run(
    pipeline=pipeline_strat_force,
    dataset="sample_data/classification",
    name="ForceGroup_Stratified",
    verbose=1,
    plots_visible=args.plots
)

accuracy = (1 - result_strat_force.best_rmse) * 100
print(f"\nStratifiedKFold + force_group - Accuracy: {accuracy:.1f}%")


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

# WITH group splitting
pipeline_with_group = [
    StandardNormalVariate(),
    {"split": KFold(n_splits=3, shuffle=True, random_state=42),
     "force_group": "Sample_ID"},
    {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
]

result_with_group = nirs4all.run(
    pipeline=pipeline_with_group,
    dataset="sample_data/classification",
    name="WithGroup",
    verbose=0
)

acc_no_group = (1 - result_no_group.best_rmse) * 100
acc_with_group = (1 - result_with_group.best_rmse) * 100

print(f"\nResults comparison:")
print(f"   WITHOUT group splitting: {acc_no_group:.1f}% (may be optimistic)")
print(f"   WITH group splitting:    {acc_with_group:.1f}% (realistic)")

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

  NATIVE GROUP SPLITTERS (use "group" parameter):
    {"split": GroupKFold(n_splits=3), "group": "Sample_ID"}
    {"split": StratifiedGroupKFold(n_splits=3), "group": "Sample_ID"}

  UNIVERSAL force_group (works with ANY splitter):
    {"split": KFold(n_splits=3), "force_group": "Sample_ID"}
    {"split": ShuffleSplit(n_splits=5), "force_group": "Sample_ID"}
    {"split": StratifiedKFold(n_splits=3), "force_group": "Sample_ID",
     "y_aggregation": "mode"}

Group Column Options:
  "Sample_ID"    - Metadata column name
  "y"            - Group by target values (stratification)

y_aggregation Options (for force_group with stratified):
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

Next: U19_sample_filtering.py - Filter samples in pipeline
""")
