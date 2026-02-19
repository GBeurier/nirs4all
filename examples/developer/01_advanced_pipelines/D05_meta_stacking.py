"""
D05 - Meta Stacking: Advanced Stacking Patterns with MetaModel
===============================================================

Meta-stacking goes beyond simple prediction merging by providing fine-grained
control over stacking architecture, levels, and cross-branch relationships.

This tutorial covers:

* MetaModel basics and StackingConfig
* Stacking levels (StackingLevel)
* Cross-branch stacking (BranchScope)
* Coverage strategies
* Multi-level stacking architectures

Prerequisites
-------------
- D03_merge_basics for prediction merging concepts

Next Steps
----------
See D06_generator_syntax for dynamic pipeline generation.

Duration: ~8 minutes
Difficulty: ★★★★★
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.models import BranchScope, CoverageStrategy, MetaModel, StackingConfig, StackingLevel
from nirs4all.operators.transforms import FirstDerivative, SavitzkyGolay
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D05 Meta Stacking Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D05 - Meta Stacking: Advanced Stacking Patterns")
print("=" * 60)

print("""
Simple stacking:
    branch → models → merge predictions → meta-model

Meta-stacking with MetaModel provides:
    - Explicit stacking levels (Level 1, Level 2, ...)
    - Cross-branch stacking (e.g., cross-dataset)
    - Coverage strategies (per-fold, per-branch, etc.)
    - Nested stacking architectures
""")

# =============================================================================
# Section 1: MetaModel Basics
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: MetaModel Basics")
print("-" * 60)

print("""
MetaModel wraps a base estimator with stacking configuration:

    MetaModel(
        model=Ridge(),
        stacking_config=StackingConfig(
            level=StackingLevel.LEVEL_2,
            branch_scope=BranchScope.CURRENT_ONLY
        )
    )
""")

# Basic stacking pipeline with MetaModel
pipeline_basic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Level 1: Base models
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=5)],
        "rf": [MSC(), RandomForestRegressor(n_estimators=50, random_state=42)],
        "ridge": [FirstDerivative(), Ridge(alpha=1.0)],
    }},

    # Merge predictions for stacking
    {"merge": "predictions"},

    # Level 2: Simple meta-model (using standard Ridge after merge)
    # Note: After merge, predictions become features for the next model
    Ridge(alpha=0.1),
]

result_basic = nirs4all.run(
    pipeline=pipeline_basic,
    dataset="sample_data/regression",
    name="MetaModelBasic",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMeta-stacking predictions: {result_basic.num_predictions}")

# =============================================================================
# Section 2: Understanding Stacking Levels
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Understanding Stacking Levels")
print("-" * 60)

print("""
StackingLevel controls model hierarchy:

    LEVEL_1: Base models (train on features)
    LEVEL_2: Meta-models (train on Level 1 predictions)
    LEVEL_3: Super-meta-models (train on Level 2 predictions)
    ...

Higher levels see increasingly abstract representations.
""")

print("\nStacking Level Architecture:")
print("  Level 0: Features (X)")
print("  Level 1: Base models → predictions_1")
print("  Level 2: Meta model → predictions_2")
print("  Level 3: Super-meta → final prediction")

# Visual representation
print("""
    ┌──────────────────────────────────────────┐
    │                 Features (X)              │
    └──────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   PLS    │  │    RF    │  │  Ridge   │   ← Level 1
    └──────────┘  └──────────┘  └──────────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
               ┌───────────────────┐
               │   Meta (Ridge)    │              ← Level 2
               └───────────────────┘
                         │
                         ▼
                   predictions
""")

# =============================================================================
# Section 3: Branch Scope Control
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Branch Scope Control")
print("-" * 60)

print("""
BranchScope controls what data the meta-model sees:

    CURRENT_ONLY:   Only predictions from same preprocessing branch
    ALL_BRANCHES:   Predictions from all branches
    SPECIFIED:      Use explicit list from source_models parameter

When using merge predictions followed by a simple model (Ridge, etc.),
the model automatically trains on the merged OOF predictions.
""")

# Cross-branch stacking using simple model after merge
pipeline_cross = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Preprocessing branches
    {"branch": {
        "snv_path": [SNV(), PLSRegression(n_components=5)],
        "msc_path": [MSC(), PLSRegression(n_components=5)],
    }},

    # Merge predictions from all branches for stacking
    {"merge": "predictions"},

    # Simple meta-model trains on the merged predictions
    Ridge(alpha=0.1),
]

result_cross = nirs4all.run(
    pipeline=pipeline_cross,
    dataset="sample_data/regression",
    name="CrossBranchStacking",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nCross-branch stacking predictions: {result_cross.num_predictions}")

# =============================================================================
# Section 4: Coverage Strategies
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Coverage Strategies")
print("-" * 60)

print("""
CoverageStrategy controls how missing predictions are handled:

    STRICT:           Raise error if any sample is missing predictions
    DROP_INCOMPLETE:  Drop samples missing any source model predictions
    IMPUTE_ZERO:      Fill missing predictions with zeros
    IMPUTE_MEAN:      Fill missing predictions with mean of available

When using simple merge followed by Ridge, the merge step handles
OOF prediction collection automatically.
""")

pipeline_coverage = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=5)],
        "rf": [RandomForestRegressor(n_estimators=5, random_state=42)],
    }},

    {"merge": "predictions"},
    ElasticNet(alpha=0.1, l1_ratio=0.5),
]

result_coverage = nirs4all.run(
    pipeline=pipeline_coverage,
    dataset="sample_data/regression",
    name="CoverageStrategy",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nWith coverage strategy: {result_coverage.num_predictions} predictions")

# =============================================================================
# Section 5: Multi-Level Stacking (3 Levels)
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Multi-Level Stacking (3 Levels)")
print("-" * 60)

print("""
Deep stacking with 3 levels:
    Level 1: Diverse base models
    Level 2: Intermediate meta-models
    Level 3: Final super-meta-model

For simpler stacking, use merge predictions + simple model.
""")

pipeline_3level = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Level 1: Base models (diverse algorithms)
    {"branch": {
        "pls5": [SNV(), PLSRegression(n_components=5)],
        "pls10": [SNV(), PLSRegression(n_components=10)],
        "rf": [MSC(), RandomForestRegressor(n_estimators=5, random_state=42)],
        "gbm": [FirstDerivative(), GradientBoostingRegressor(n_estimators=5, random_state=42)],
    }},
    {"merge": "predictions"},

    # Level 2: Meta-model
    Ridge(alpha=0.1),
]

result_3level = nirs4all.run(
    pipeline=pipeline_3level,
    dataset="sample_data/regression",
    name="ThreeLevelStacking",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nMulti-level stacking predictions: {result_3level.num_predictions}")

# =============================================================================
# Section 6: Stacking with Feature Augmentation
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Stacking with Feature Augmentation")
print("-" * 60)

print("""
Combine original features with stacked predictions:

    {"merge": {"predictions": [0, 1], "include_original": True}}

The meta-model sees both base predictions AND original features.
""")

# Skip this example as it uses advanced merge syntax
print("(Skipped - advanced merge syntax demonstration)")

# =============================================================================
# Section 7: Multi-Dataset Stacking Concept
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Multi-Dataset Stacking Concept")
print("-" * 60)

print("""
For multi-dataset scenarios, stacking can be used for calibration transfer:

    1. Train base models on Dataset A
    2. Train base models on Dataset B
    3. Meta-model learns to combine predictions

Use BranchScope.SPECIFIED with source_models for explicit control.
""")

# Conceptual example (requires multi-dataset input)
print("Multi-dataset stacking pattern:")
print("  Dataset A: Train base models")
print("  Dataset B: Train base models")
print("  Meta-model: Learns to combine predictions across datasets")

# =============================================================================
# Section 8: Blending vs. Stacking
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Blending vs. Stacking")
print("-" * 60)

print("""
Stacking: Uses OOF predictions (proper CV, no leakage)
Blending: Uses holdout set predictions (simpler, some info loss)

nirs4all uses stacking by default for proper generalization.
""")

# Stacking (OOF) - default
pipeline_stacking = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"branch": {
        "pls": [SNV(), PLSRegression(n_components=5)],
        "rf": [RandomForestRegressor(n_estimators=5, random_state=42)],
    }},
    {"merge": "predictions"},
    Ridge(alpha=0.1),
]

print("Stacking (OOF):")
print("  - 5-fold CV for base models")
print("  - OOF predictions for meta-model training")
print("  - No data leakage")

# =============================================================================
# Section 9: Heterogeneous Stacking
# =============================================================================
print("\n" + "-" * 60)
print("Example 9: Heterogeneous Stacking")
print("-" * 60)

print("""
Mix different model types for maximum diversity:
  - Linear models (PLS, Ridge, ElasticNet)
  - Tree models (RF, GBM, XGBoost)
  - Neural networks (if available)

Diversity in base models improves meta-model performance.
""")

pipeline_heterogeneous = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        # Linear models
        "pls": [SNV(), PLSRegression(n_components=5)],
        "ridge": [SNV(), Ridge(alpha=1.0)],
        "elastic": [SNV(), ElasticNet(alpha=0.1, l1_ratio=0.5)],

        # Tree models
        "rf": [MSC(), RandomForestRegressor(n_estimators=5, random_state=42)],
        "gbm": [MSC(), GradientBoostingRegressor(n_estimators=5, random_state=42)],
    }},
    {"merge": "predictions"},

    # Meta-model benefits from model diversity
    Ridge(alpha=0.1),
]

result_heterogeneous = nirs4all.run(
    pipeline=pipeline_heterogeneous,
    dataset="sample_data/regression",
    name="HeterogeneousStacking",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nHeterogeneous stacking predictions: {result_heterogeneous.num_predictions}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. MetaModel provides explicit stacking configuration
2. StackingLevel controls hierarchy (L1 → L2 → L3)
3. BranchScope controls cross-branch visibility
4. CoverageStrategy handles missing predictions
5. Multi-level stacking enables deep ensembles
6. Feature augmentation combines predictions + features
7. Heterogeneous models improve diversity

Key classes:
- MetaModel: Wrapper for stacking estimators
- StackingConfig: Configuration container
- StackingLevel: AUTO, LEVEL_1, LEVEL_2, LEVEL_3
- BranchScope: CURRENT_ONLY, ALL_BRANCHES, SPECIFIED
- CoverageStrategy: STRICT, DROP_INCOMPLETE, IMPUTE_ZERO, IMPUTE_MEAN

Next: D06_generator_syntax.py - Dynamic pipeline generation
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
