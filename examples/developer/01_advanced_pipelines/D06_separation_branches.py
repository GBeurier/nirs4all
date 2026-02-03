"""
D06 - Separation Branches: Different Samples to Different Branches
==================================================================

Separation branches route DIFFERENT samples to different branches
based on some criterion, then reassemble them with concat merge.

This is different from duplication branches where the SAME samples
go to all branches with different preprocessing.

This tutorial covers:

* by_tag branching - Branch by tag values
* by_metadata branching - Branch by metadata column
* by_filter branching - Branch by filter result
* by_source branching - Branch by feature source
* Concat merge for reassembly

Branch Types:
  - **Duplication branches**: Same samples, different preprocessing (default)
  - **Separation branches**: Different samples, parallel processing

Prerequisites
-------------
- D01_branching_basics for duplication branching
- U05_tagging_analysis for tagging concepts

Next Steps
----------
See D07_value_mapping for user-friendly value mapping syntax.

Duration: ~8 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative
)
from nirs4all.operators.filters import YOutlierFilter

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D06 Separation Branches Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D06 - Separation Branches")
print("=" * 60)

print("""
Separation branches route DIFFERENT samples to different branches:

  DUPLICATION (default):  All samples → [Branch A, Branch B]
  SEPARATION:             Samples split → Branch A gets some, Branch B gets others

Separation branch types:
  by_tag:      Route by tag values (from prior tag/exclude)
  by_metadata: Route by metadata column values
  by_filter:   Route by filter result (pass/fail)
  by_source:   Route by feature source (multi-source data)

IMPORTANT: Separation branches require "concat" merge to reassemble samples.
""")


# =============================================================================
# Section 1: by_tag Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: by_tag Branching")
print("-" * 60)

print("""
Branch samples based on a prior tag:

    {"tag": YOutlierFilter()}  # Creates tag "y_outlier_iqr"
    {"branch": {"by_tag": "y_outlier_iqr", "steps": {...}}}
    {"merge": "concat"}

Samples tagged as outliers go to one branch, others to another.
""")

pipeline_by_tag = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # First, tag the outliers
    {"tag": YOutlierFilter(method="iqr", threshold=1.5)},

    # Branch by tag: outliers get different preprocessing
    {"branch": {
        "by_tag": "y_outlier_iqr",
        "steps": {
            True: [SNV(), FirstDerivative()],  # Outliers: aggressive preprocessing
            False: [SNV()],                     # Normal: standard preprocessing
        }
    }},
    {"merge": "concat"},  # Reassemble in original order

    {"model": PLSRegression(n_components=5)},
]

print("""Pipeline structure:
  1. Tag Y outliers
  2. Branch: outliers → SNV+Deriv, normal → SNV
  3. Concat merge: reassemble
  4. Single PLS model on all samples
""")


# =============================================================================
# Section 2: by_metadata Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: by_metadata Branching")
print("-" * 60)

print("""
Branch samples based on metadata column values:

    {"branch": {
        "by_metadata": "instrument",
        "steps": {
            "NIR_500": [SNV()],
            "NIR_700": [MSC()],
        }
    }}
    {"merge": "concat"}

Each unique metadata value creates a branch with its samples.
""")

pipeline_by_metadata = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Branch by metadata column
    {"branch": {
        "by_metadata": "farm",  # Assuming dataset has 'farm' column
        "steps": [SNV()],       # Same preprocessing for all farms
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Pipeline structure:
  1. Branch by "farm" metadata
  2. Each farm gets its own branch
  3. All use same preprocessing (SNV)
  4. Concat merge: reassemble
  5. Single model trained on all samples
""")


# =============================================================================
# Section 3: by_metadata with Per-Value Steps
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: by_metadata with Per-Value Steps")
print("-" * 60)

print("""
Different preprocessing for different metadata values:

    {"branch": {
        "by_metadata": "instrument",
        "steps": {
            "portable": [SNV(), FirstDerivative()],  # Aggressive
            "benchtop": [SNV()],                      # Minimal
        }
    }}
""")

pipeline_per_value = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Per-value preprocessing
    {"branch": {
        "by_metadata": "instrument",
        "steps": {
            "portable": [SNV(), FirstDerivative()],  # Portable: aggressive
            "benchtop": [SNV()],                      # Benchtop: minimal
        }
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Per-value preprocessing:
  portable → SNV → FirstDerivative (aggressive)
  benchtop → SNV (minimal)
""")


# =============================================================================
# Section 4: by_filter Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: by_filter Branching")
print("-" * 60)

print("""
Branch based on filter result (pass/fail):

    {"branch": {
        "by_filter": YOutlierFilter(method="iqr"),
        "steps": {
            "pass": [SNV()],      # Samples passing filter
            "fail": [MSC()],      # Samples failing filter (outliers)
        }
    }}

This is similar to by_tag but creates the tag inline.
""")

pipeline_by_filter = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Branch by filter result
    {"branch": {
        "by_filter": YOutlierFilter(method="iqr", threshold=1.5),
        "steps": {
            "pass": [SNV()],                    # Non-outliers
            "fail": [SNV(), FirstDerivative()], # Outliers: extra processing
        }
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""by_filter creates two branches:
  "pass": samples NOT flagged by filter
  "fail": samples flagged by filter
""")


# =============================================================================
# Section 5: by_source Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: by_source Branching")
print("-" * 60)

print("""
For multi-source datasets, branch by feature source:

    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV(), FirstDerivative()],
            "markers": [StandardScaler()],
        }
    }}
    {"merge": {"sources": "concat"}}

Each source gets its own preprocessing, then features are merged.
""")

pipeline_by_source = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Per-source preprocessing
    {"branch": {
        "by_source": True,
        "steps": {
            "source_0": [MinMaxScaler(), SNV()],
            "source_1": [MinMaxScaler(), MSC()],
        }
    }},
    {"merge": {"sources": "concat"}},  # Merge sources horizontally

    {"model": PLSRegression(n_components=5)},
]

print("""by_source:
  source_0 → MinMaxScaler → SNV
  source_1 → MinMaxScaler → MSC
  merge sources → concatenate features
""")


# =============================================================================
# Section 6: Per-Branch Model Training
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Per-Branch Model Training")
print("-" * 60)

print("""
Include models inside separation branches for per-group models:

    {"branch": {
        "by_metadata": "farm",
        "steps": [SNV(), PLSRegression(n_components=5)],
    }}
    {"merge": "concat"}

Each farm gets its own model. Predictions are concatenated.
""")

pipeline_per_branch_model = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Model inside branch = separate model per group
    {"branch": {
        "by_metadata": "farm",
        "steps": [SNV(), {"model": PLSRegression(n_components=5)}],
    }},
    {"merge": "concat"},  # Concatenate predictions
]

print("""Per-branch models:
  - Each farm trains its own PLS model
  - Predictions from all farms concatenated
  - Useful when groups have different calibrations
""")


# =============================================================================
# Section 7: Combining Separation and Duplication
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Combining Separation and Duplication Branches")
print("-" * 60)

print("""
Combine separation and duplication branches:

    # Separation: per-metadata preprocessing
    {"branch": {"by_metadata": "instrument", ...}}
    {"merge": "concat"}

    # Duplication: model comparison
    {"branch": {"pls": [...], "rf": [...]}}
""")

pipeline_combined = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Separation: per-instrument preprocessing
    {"branch": {
        "by_metadata": "instrument",
        "steps": {
            "portable": [SNV(), FirstDerivative()],
            "benchtop": [SNV()],
        }
    }},
    {"merge": "concat"},

    # Duplication: compare models
    {"branch": {
        "pls5": [PLSRegression(n_components=5)],
        "pls10": [PLSRegression(n_components=10)],
    }},
]

print("""Combined workflow:
  1. Separation branch: per-instrument preprocessing
  2. Concat merge: reassemble samples
  3. Duplication branch: compare PLS variants
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Separation branches route DIFFERENT samples to different branches
2. by_tag: Branch by prior tag values
3. by_metadata: Branch by metadata column
4. by_filter: Branch by filter pass/fail
5. by_source: Branch by feature source (multi-source)
6. Always use "concat" merge to reassemble samples
7. Can include models for per-group training

Syntax Reference:
  by_tag:      {"branch": {"by_tag": "tag_name", "steps": {...}}}
  by_metadata: {"branch": {"by_metadata": "column", "steps": {...}}}
  by_filter:   {"branch": {"by_filter": Filter(), "steps": {...}}}
  by_source:   {"branch": {"by_source": True, "steps": {...}}}

Merge types:
  {"merge": "concat"}           # Reassemble samples (separation)
  {"merge": {"sources": "concat"}}  # Merge source features (by_source)

Key difference from duplication:
  Duplication: Same samples → different preprocessing → compare results
  Separation:  Different samples → parallel processing → reassemble

Next: D07_value_mapping.py - User-friendly value mapping syntax
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
