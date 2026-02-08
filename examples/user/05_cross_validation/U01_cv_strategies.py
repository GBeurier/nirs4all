"""
U01 - Cross-Validation Strategies: Choosing the Right CV Method
================================================================

Select appropriate cross-validation for your data structure.

This tutorial covers:

* Standard CV: KFold, ShuffleSplit, RepeatedKFold
* Stratified CV for classification
* Time-series CV for temporal data
* Leave-One-Out and custom splits

Prerequisites
-------------
Understanding of model evaluation concepts.

Next Steps
----------
See :ref:`U02_group_splitting` for grouped data.

Duration: ~4 minutes
Difficulty: ★★☆☆☆
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
    RepeatedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    LeaveOneOut,
    TimeSeriesSplit,
)
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U01 CV Strategies Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Overview of CV Strategies
# =============================================================================
print("\n" + "=" * 60)
print("U01 - Cross-Validation Strategies")
print("=" * 60)

print("""
Cross-validation estimates model performance on unseen data.
Choose the right CV strategy based on your data structure:

  📊 STANDARD CV (i.i.d. data)
     KFold           - K non-overlapping folds
     ShuffleSplit    - Random train/test splits
     RepeatedKFold   - KFold repeated multiple times

  📈 STRATIFIED CV (classification, imbalanced)
     StratifiedKFold          - Preserve class proportions
     StratifiedShuffleSplit   - Stratified random splits

  📉 TIME-SERIES CV (temporal data)
     TimeSeriesSplit - Expanding window, no look-ahead

  🔍 EXHAUSTIVE CV (small datasets)
     LeaveOneOut     - Leave one sample out each fold
""")


# =============================================================================
# Section 2: KFold - Standard K-Fold
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: KFold - Standard K-Fold")
print("-" * 60)

print("""
KFold divides data into K non-overlapping folds.
Each fold is used once as validation.
""")

pipeline_kfold = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # 5-fold cross-validation
    KFold(n_splits=5, shuffle=True, random_state=42),

    {"model": PLSRegression(n_components=10)},
]

result_kfold = nirs4all.run(
    pipeline=pipeline_kfold,
    dataset="sample_data/regression",
    name="KFold",
    verbose=1
)

print(f"\nKFold (5 splits) - RMSE: {result_kfold.best_score:.4f}")


# =============================================================================
# Section 3: ShuffleSplit - Random Splits
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: ShuffleSplit - Random Splits")
print("-" * 60)

print("""
ShuffleSplit creates random train/test splits.
Flexible: control test_size and number of splits independently.
""")

pipeline_shuffle = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # 10 random splits with 25% test
    ShuffleSplit(n_splits=10, test_size=0.25, random_state=42),

    {"model": PLSRegression(n_components=10)},
]

result_shuffle = nirs4all.run(
    pipeline=pipeline_shuffle,
    dataset="sample_data/regression",
    name="ShuffleSplit",
    verbose=1
)

print(f"\nShuffleSplit (10 splits, 25% test) - RMSE: {result_shuffle.best_score:.4f}")


# =============================================================================
# Section 4: RepeatedKFold - Multiple Repetitions
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: RepeatedKFold - Multiple Repetitions")
print("-" * 60)

print("""
RepeatedKFold repeats K-fold CV multiple times with different shuffles.
More robust estimates, especially for small datasets.
""")

pipeline_repeated = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # 5-fold repeated 3 times = 15 total folds
    RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),

    {"model": PLSRegression(n_components=10)},
]

result_repeated = nirs4all.run(
    pipeline=pipeline_repeated,
    dataset="sample_data/regression",
    name="RepeatedKFold",
    verbose=1
)

print(f"\nRepeatedKFold (5×3 = 15 folds) - RMSE: {result_repeated.best_score:.4f}")


# =============================================================================
# Section 5: StratifiedKFold - Classification
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: StratifiedKFold - Classification")
print("-" * 60)

print("""
StratifiedKFold preserves class proportions in each fold.
Essential for imbalanced classification datasets.
""")

# Create synthetic balanced classification data for demo
np.random.seed(42)
X_classif = np.random.randn(60, 100)  # 60 samples, 100 features
y_classif = np.array([0]*20 + [1]*20 + [2]*20)  # 3 classes, 20 each

pipeline_stratified = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # Stratified 3-fold
    StratifiedKFold(n_splits=3, shuffle=True, random_state=42),

    {"model": RandomForestClassifier(n_estimators=5, random_state=42)},
]

result_stratified = nirs4all.run(
    pipeline=pipeline_stratified,
    dataset=(X_classif, y_classif),
    name="StratifiedKFold",
    verbose=1
)

accuracy = (1 - result_stratified.best_score) * 100 if not np.isnan(result_stratified.best_score) else float('nan')
print(f"\nStratifiedKFold - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nStratifiedKFold - (see detailed metrics)")


# =============================================================================
# Section 6: StratifiedShuffleSplit
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: StratifiedShuffleSplit")
print("-" * 60)

print("""
Combines stratification with random splitting.
Flexible test_size while preserving class balance.
""")

# Reuse balanced synthetic data from Section 5
pipeline_strat_shuffle = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # Stratified random splits
    StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42),

    {"model": RandomForestClassifier(n_estimators=5, random_state=42)},
]

result_strat_shuffle = nirs4all.run(
    pipeline=pipeline_strat_shuffle,
    dataset=(X_classif, y_classif),
    name="StratShuffleSplit",
    verbose=1
)

accuracy = (1 - result_strat_shuffle.best_score) * 100 if not np.isnan(result_strat_shuffle.best_score) else float('nan')
print(f"\nStratifiedShuffleSplit - Accuracy: {accuracy:.1f}%" if not np.isnan(accuracy) else "\nStratifiedShuffleSplit - (see detailed metrics)")


# =============================================================================
# Section 7: TimeSeriesSplit
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: TimeSeriesSplit")
print("-" * 60)

print("""
TimeSeriesSplit for temporal/sequential data.
Uses expanding window: train on past, test on future.
Prevents data leakage from future to past.
""")

pipeline_timeseries = [
    MinMaxScaler(),
    StandardNormalVariate(),

    # Time series 5-fold
    TimeSeriesSplit(n_splits=5),

    {"model": PLSRegression(n_components=10)},
]

result_timeseries = nirs4all.run(
    pipeline=pipeline_timeseries,
    dataset="sample_data/regression",
    name="TimeSeriesSplit",
    verbose=1
)

print(f"\nTimeSeriesSplit - RMSE: {result_timeseries.best_score:.4f}")
print("Note: For truly temporal data, use TimeSeriesSplit to avoid look-ahead bias.")


# =============================================================================
# Section 8: LeaveOneOut
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: LeaveOneOut")
print("-" * 60)

print("""
LeaveOneOut leaves exactly one sample out per fold.
N samples = N folds. Exhaustive but slow for large datasets.
""")

# Use small dataset for LOO demo
import numpy as np
np.random.seed(42)
X_small = np.random.randn(30, 100)
y_small = np.random.randn(30)

pipeline_loo = [
    MinMaxScaler(),

    # Leave-one-out
    LeaveOneOut(),

    {"model": PLSRegression(n_components=5)},
]

result_loo = nirs4all.run(
    pipeline=pipeline_loo,
    dataset=(X_small, y_small),
    name="LeaveOneOut",
    verbose=1
)

print(f"\nLeaveOneOut (30 folds) - RMSE: {result_loo.best_score:.4f}")


# =============================================================================
# Section 9: Comparing CV Strategies
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Comparing CV Strategies")
print("-" * 60)

# Run same model with different CV
cv_strategies = [
    ("KFold-3", KFold(n_splits=3, shuffle=True, random_state=42)),
    ("KFold-5", KFold(n_splits=5, shuffle=True, random_state=42)),
    ("KFold-10", KFold(n_splits=10, shuffle=True, random_state=42)),
    ("ShuffleSplit-5", ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)),
    ("ShuffleSplit-10", ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)),
]

print("\nComparing CV strategies on same data:")
for name, cv in cv_strategies:
    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        cv,
        {"model": PLSRegression(n_components=10)},
    ]
    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        name=name,
        verbose=0
    )
    print(f"   {name:20s}: RMSE={result.best_score:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
CV Strategy Selection Guide:

  ┌────────────────────────┬───────────────────────────────────┐
  │ Strategy               │ When to Use                       │
  ├────────────────────────┼───────────────────────────────────┤
  │ KFold                  │ Standard regression, large data   │
  │ ShuffleSplit           │ Flexible test size, many repeats  │
  │ RepeatedKFold          │ Small data, need robust estimate  │
  │ StratifiedKFold        │ Classification, class imbalance   │
  │ StratifiedShuffleSplit │ Classification + flexible splits  │
  │ TimeSeriesSplit        │ Temporal/sequential data          │
  │ LeaveOneOut            │ Very small datasets               │
  └────────────────────────┴───────────────────────────────────┘

Common Parameters:
  n_splits    - Number of folds/splits
  shuffle     - Shuffle data before splitting (KFold)
  test_size   - Fraction for test set (ShuffleSplit)
  n_repeats   - Number of repetitions (RepeatedKFold)
  random_state - Reproducibility seed

Rules of Thumb:
  • 5-10 folds for moderate datasets
  • Stratified CV for classification
  • ShuffleSplit when you need specific test size
  • TimeSeriesSplit for temporal dependencies
  • More splits = more reliable estimate, slower runtime

Next: U02_group_splitting.py - Handle grouped/clustered data
""")
