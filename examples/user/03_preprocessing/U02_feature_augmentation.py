"""
U02 - Feature Augmentation: Automated Preprocessing Exploration
================================================================

Explore multiple preprocessing combinations automatically.

This tutorial covers:

* feature_augmentation keyword for preprocessing variants
* Three action modes: extend, add, replace
* Building preprocessing search spaces
* Comparing variants with the new API

Prerequisites
-------------
Complete :ref:`U01_preprocessing_basics` first.

Next Steps
----------
See :ref:`U03_sample_augmentation` for data augmentation.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection,
    Detrend,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay,
    Gaussian,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 Feature Augmentation Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: What is Feature Augmentation?
# =============================================================================
print("\n" + "=" * 60)
print("U02 - Feature Augmentation")
print("=" * 60)

print("""
Feature Augmentation automatically explores preprocessing variants.
Instead of manually defining each preprocessing pipeline, you specify
a set of transformations and the system generates all combinations.

Three action modes control HOW variants are generated:

  📊 EXTEND (linear growth)
     Add new processings to the set independently
     [SNV, Detrend] → raw_SNV, raw_Detrend (2 variants)

  📈 ADD (multiplicative, keep originals)
     Chain on all existing + keep originals
     MinMax → ADD [SNV] → MinMax, MinMax_SNV (2 variants)

  📉 REPLACE (multiplicative, discard originals)
     Chain on all existing, discard originals
     MinMax → REPLACE [SNV] → MinMax_SNV (1 variant)
""")


# =============================================================================
# Section 2: EXTEND Mode - Independent Preprocessing Options
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: EXTEND Mode - Independent Options")
print("-" * 60)

print("""
EXTEND adds each preprocessing independently, no chaining.
Good for exploring different preprocessing approaches.
""")

pipeline_extend = [
    # Start with scaling
    MinMaxScaler(),

    # Add SNV, FirstDerivative as independent alternatives
    {"feature_augmentation": [StandardNormalVariate, FirstDerivative], "action": "extend"},

    # Cross-validation and model
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

result_extend = nirs4all.run(
    pipeline=pipeline_extend,
    dataset="sample_data/regression",
    name="ExtendMode",
    verbose=1
)

print(f"\nNumber of variants explored: {result_extend.num_predictions}")
print(f"Best Score (MSE): {result_extend.best_score:.4f}")

# Show top results with display_metrics
print("\nTop preprocessing variants:")
for pred in result_extend.top(3, display_metrics=['rmse']):
    print(f"   {pred.get('preprocessings', 'N/A')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 3: ADD Mode - Chain While Keeping Originals
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: ADD Mode - Chain + Keep Originals")
print("-" * 60)

print("""
ADD chains each operation on existing processings AND keeps originals.
Good for ablation studies where you need baselines for comparison.
""")

pipeline_add = [
    # Base preprocessing
    StandardNormalVariate(),

    # Chain derivative AND keep original SNV for comparison
    {"feature_augmentation": [FirstDerivative], "action": "add"},

    # Cross-validation and model
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

result_add = nirs4all.run(
    pipeline=pipeline_add,
    dataset="sample_data/regression",
    name="AddMode",
    verbose=1
)

print(f"\nNumber of variants: {result_add.num_predictions}")
print("Variants include: SNV alone, SNV + FirstDerivative")

for pred in result_add.top(5, display_metrics=['rmse']):
    print(f"   {pred.get('preprocessings', 'N/A')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 4: REPLACE Mode - Pure Chaining
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: REPLACE Mode - Pure Chaining")
print("-" * 60)

print("""
REPLACE chains operations on existing processings, discarding originals.
Good for multi-stage preprocessing pipelines without intermediate bloat.
""")

pipeline_replace = [
    # Stage 1: Smoothing (extend to explore options)
    {"feature_augmentation": [Gaussian(sigma=2), SavitzkyGolay(deriv=0)], "action": "extend"},

    # Stage 2: Force derivative on all (replace - no smoothing-only variants)
    {"feature_augmentation": [FirstDerivative], "action": "replace"},

    # Stage 3: Force SNV on all (replace - clean final chain)
    {"feature_augmentation": [StandardNormalVariate], "action": "replace"},

    # Cross-validation and model
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=5)},
]

result_replace = nirs4all.run(
    pipeline=pipeline_replace,
    dataset="sample_data/regression",
    name="ReplaceMode",
    verbose=1
)

print(f"\nNumber of variants: {result_replace.num_predictions}")
print("Both variants end with: _FirstDerivative_SNV")

for pred in result_replace.top(5, display_metrics=['rmse']):
    print(f"   {pred.get('preprocessings', 'N/A')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 5: Comprehensive Preprocessing Search
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Comprehensive Preprocessing Search")
print("-" * 60)

print("""
Combine extend and add to explore a large preprocessing space efficiently.
""")

pipeline_search = [
    # Stage 1: Scatter correction options (extend)
    {"feature_augmentation": [
        StandardNormalVariate,
        MultiplicativeScatterCorrection,
        Detrend,
    ], "action": "extend"},

    # Stage 2: Add derivative options (add - keep scatter-only as baseline)
    {"feature_augmentation": [
        FirstDerivative,
        SecondDerivative,
    ], "action": "add"},

    # Cross-validation and model
    ShuffleSplit(n_splits=2, random_state=42),
    {"model": PLSRegression(n_components=10)},
]

result_search = nirs4all.run(
    pipeline=pipeline_search,
    dataset="sample_data/regression",
    name="Search",
    verbose=1
)

print(f"\nTotal variants explored: {result_search.num_predictions}")
print(f"Best Score (MSE): {result_search.best_score:.4f}")

# Show all results ranked
print("\nAll preprocessing variants (ranked by RMSE):")
for i, pred in enumerate(result_search.top(20, display_metrics=['rmse']), 1):
    print(f"   {i}. {pred.get('preprocessings', 'N/A')}: RMSE={pred.get('rmse', 0):.4f}")


# =============================================================================
# Section 6: Comparing Action Modes
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Action Mode Comparison")
print("-" * 60)

# Same transforms, different modes
transforms = [FirstDerivative, SecondDerivative]

# EXTEND: Independent alternatives
pipeline_demo_extend = [
    MinMaxScaler(),
    {"feature_augmentation": transforms, "action": "extend"},
    ShuffleSplit(n_splits=1),
    {"model": PLSRegression(n_components=5)},
]

# ADD: Chain + keep original
pipeline_demo_add = [
    MinMaxScaler(),
    {"feature_augmentation": transforms, "action": "add"},
    ShuffleSplit(n_splits=1),
    {"model": PLSRegression(n_components=5)},
]

# REPLACE: Chain, discard original
pipeline_demo_replace = [
    MinMaxScaler(),
    {"feature_augmentation": transforms, "action": "replace"},
    ShuffleSplit(n_splits=1),
    {"model": PLSRegression(n_components=5)},
]

res_ext = nirs4all.run(pipeline=pipeline_demo_extend, dataset="sample_data/regression", name="ext", verbose=0)
res_add = nirs4all.run(pipeline=pipeline_demo_add, dataset="sample_data/regression", name="add", verbose=0)
res_rep = nirs4all.run(pipeline=pipeline_demo_replace, dataset="sample_data/regression", name="rep", verbose=0)

print(f"""
Given: MinMaxScaler → [FirstDerivative, SecondDerivative]

Mode      Variants  What's Included
--------  --------  ---------------
EXTEND       {res_ext.num_predictions}      raw_MinMax, raw_FirstDeriv, raw_SecondDeriv
ADD          {res_add.num_predictions}      MinMax, MinMax_FirstDeriv, MinMax_SecondDeriv
REPLACE      {res_rep.num_predictions}      MinMax_FirstDeriv, MinMax_SecondDeriv (no MinMax-only)
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Feature Augmentation Action Modes:

┌──────────┬────────────────────────────────┬─────────────────────────┐
│ Mode     │ Behavior                       │ Use Case                │
├──────────┼────────────────────────────────┼─────────────────────────┤
│ extend   │ Add new options independently  │ Explore alternatives    │
│          │ Linear growth: N base + M new  │ No chaining needed      │
├──────────┼────────────────────────────────┼─────────────────────────┤
│ add      │ Chain on all existing          │ Ablation studies        │
│          │ Keep originals + chained       │ Need baselines          │
├──────────┼────────────────────────────────┼─────────────────────────┤
│ replace  │ Chain on all existing          │ Multi-stage pipelines   │
│          │ Discard originals              │ Clean final chains      │
└──────────┴────────────────────────────────┴─────────────────────────┘

Common Patterns:

  # Explore scatter correction options
  {"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"}

  # Add derivatives while keeping baseline
  {"feature_augmentation": [FirstDerivative], "action": "add"}

  # Multi-stage: smoothing → derivative → normalization
  {"feature_augmentation": [Gaussian], "action": "replace"}
  {"feature_augmentation": [FirstDerivative], "action": "replace"}
  {"feature_augmentation": [SNV], "action": "replace"}

Next: U03_sample_augmentation.py - Data augmentation techniques
""")
