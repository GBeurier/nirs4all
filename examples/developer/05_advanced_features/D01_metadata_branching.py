"""
D01 - Metadata Branching: Partitioning by Metadata
===================================================

The ``by_metadata`` branch mode creates branches based on sample metadata,
enabling group-specific processing or stratified analysis.

This is a SEPARATION branch type: different samples go to different branches
based on their metadata values.

This tutorial covers:

* by_metadata branch basics
* Partitioning by categorical metadata
* Value mapping for grouping
* Combining with regular branching
* Concat merge for reassembly

Syntax:
  {"branch": {"by_metadata": "column_name", "steps": {...}}}

Prerequisites
-------------
- 01_branching/D01_branching_basics for branching concepts

Next Steps
----------
See D02_concat_transform for feature concatenation.
See D06_separation_branches for comprehensive separation branch examples.
See D07_value_mapping for advanced value mapping syntax.

Duration: ~4 minutes
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D01 Metadata Branching Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D01 - Metadata Branching: Partitioning by Metadata")
print("=" * 60)

print("""
by_metadata branching creates branches based on sample attributes:

  Use cases:
  - Different preprocessing per instrument
  - Separate models per sample type
  - Stratified analysis by metadata field

  Syntax (v2):
    {"branch": {
        "by_metadata": "instrument",  # Metadata column to branch by
        "steps": {                     # Per-value preprocessing
            "InstrumentA": [SNV()],
            "InstrumentB": [MSC()],
        }
    }}
    {"merge": "concat"}  # Reassemble samples in original order

This is a SEPARATION branch - different samples go to different branches.
""")


# =============================================================================
# Section 1: Basic Metadata Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic Metadata Branching")
print("-" * 60)

print("""
Branch samples by categorical metadata:

    {"branch": {
        "by_metadata": "variety",
        "steps": {
            "variety_A": [preprocessing_for_A],
            "variety_B": [preprocessing_for_B],
        }
    }}
    {"merge": "concat"}  # Reassemble samples

Samples are routed to their matching branch based on metadata value.
""")

pipeline_basic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_metadata": "variety",
        "steps": {
            "variety_A": [SNV()],
            "variety_B": [MSC()],
            "variety_C": [FirstDerivative()],
        }
    }},
    {"merge": "concat"},  # Reassemble samples from all branches
    PLSRegression(n_components=5),
]

print("Pipeline with by_metadata branch:")
print("  - Samples routed by 'variety' column value")
print("  - Each variety gets different preprocessing")
print("  - Samples reassembled via concat merge")
print("  - Single model trained on all samples")


# =============================================================================
# Section 2: Auto-Discovery of Metadata Values
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Auto-Discovery of Metadata Values")
print("-" * 60)

print("""
When no explicit "steps" dict is provided, branches are auto-created
for each unique metadata value:

    {"branch": {
        "by_metadata": "instrument",
        "steps": [SNV()],  # Same preprocessing for all values
    }}

All unique values of "instrument" column create separate branches.
Each branch applies the same preprocessing steps.
""")

pipeline_auto = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_metadata": "instrument",
        "steps": [SNV()],  # Applied to all instrument branches
    }},
    {"merge": "concat"},
    PLSRegression(n_components=5),
]

print("Auto-discovery mode:")
print("  - Branches created for each unique 'instrument' value")
print("  - Same preprocessing applied to all branches")
print("  - Useful when you don't know all possible values")


# =============================================================================
# Section 3: Value Mapping for Grouping
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Value Mapping for Grouping")
print("-" * 60)

print("""
Use "values" to group metadata values into named branches:

    {"branch": {
        "by_metadata": "instrument",
        "values": {
            "portable": ["NIR_100", "NIR_200"],    # Group portable devices
            "benchtop": ["NIR_500", "NIR_700"],    # Group benchtop devices
        },
        "steps": {
            "portable": [SNV(), FirstDerivative()],  # Aggressive preprocessing
            "benchtop": [SNV()],                     # Minimal preprocessing
        }
    }}

Value mapping groups multiple metadata values into logical branches.
See D07_value_mapping.py for advanced value mapping syntax.
""")

pipeline_grouped = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_metadata": "instrument",
        "values": {
            "portable": ["NIR_100", "NIR_200"],
            "benchtop": ["NIR_500", "NIR_700"],
        },
        "steps": {
            "portable": [SNV(), FirstDerivative()],
            "benchtop": [SNV()],
        }
    }},
    {"merge": "concat"},
    PLSRegression(n_components=5),
]

print("Value mapping:")
print("  - Multiple instruments grouped into 'portable' and 'benchtop'")
print("  - Each group gets appropriate preprocessing")


# =============================================================================
# Section 4: Per-Branch Model Training
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Per-Branch Model Training")
print("-" * 60)

print("""
Models inside branches train on each partition separately:

    {"branch": {
        "by_metadata": "farm",
        "steps": [SNV(), PLSRegression(n_components=5)],  # Model in branch
    }}
    {"merge": "concat"}  # Collect predictions

Each farm gets its own PLS model trained on its data.
This is useful when different groups have fundamentally different relationships.
""")

pipeline_per_branch_model = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_metadata": "farm",
        "steps": [SNV(), PLSRegression(n_components=5)],  # Separate model per farm
    }},
    {"merge": "concat"},  # Collect predictions from all farms
]

print("Per-branch model training:")
print("  - Each farm gets its own PLS model")
print("  - Predictions merged back in sample order")


# =============================================================================
# Section 5: Minimum Samples Per Branch
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Minimum Samples Per Branch")
print("-" * 60)

print("""
Use min_samples to skip branches with too few samples:

    {"branch": {
        "by_metadata": "variety",
        "min_samples": 10,  # Skip varieties with < 10 samples
        "steps": [SNV()],
    }}

Branches with insufficient samples are skipped to avoid training issues.
""")

pipeline_min_samples = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_metadata": "variety",
        "min_samples": 10,  # Skip varieties with fewer than 10 samples
        "steps": [SNV()],
    }},
    {"merge": "concat"},
    PLSRegression(n_components=5),
]

print("Minimum samples option:")
print("  - Varieties with < 10 samples are skipped")
print("  - Prevents training on tiny partitions")


# =============================================================================
# Section 6: Combining with Duplication Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Combining with Duplication Branching")
print("-" * 60)

print("""
Combine metadata branching (separation) with preprocessing branching (duplication):

    # First: preprocess by metadata (separation)
    {"branch": {"by_metadata": "instrument", "steps": {...}}}
    {"merge": "concat"}

    # Then: compare models (duplication)
    {"branch": {"pls": [PLS()], "ridge": [Ridge()]}}
""")

pipeline_combined = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # First: per-instrument preprocessing (separation branch)
    {"branch": {
        "by_metadata": "instrument",
        "steps": {
            "NIR_500": [SNV()],
            "NIR_700": [MSC()],
        }
    }},
    {"merge": "concat"},  # Reassemble samples

    # Then: compare models (duplication branch)
    {"branch": {
        "pls5": [PLSRegression(n_components=5)],
        "pls10": [PLSRegression(n_components=10)],
    }},
]

print("Combined branching:")
print("  1. by_metadata: per-instrument preprocessing (separation)")
print("  2. concat merge: reassemble samples")
print("  3. Regular branch: compare PLS components (duplication)")


# =============================================================================
# Section 7: Other Separation Branch Types
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Other Separation Branch Types")
print("-" * 60)

print("""
by_metadata is one of several separation branch types:

  by_tag:       Branch by tag values (from outlier detection, etc.)
  by_metadata:  Branch by metadata column (this tutorial)
  by_filter:    Branch by filter result (pass/fail)
  by_source:    Branch by feature source (for multi-source data)

Example - by_tag branching:
    {"tag": YOutlierFilter()}           # First create a tag
    {"branch": {"by_tag": "y_outlier_iqr", "steps": {...}}}

Example - by_filter branching:
    {"branch": {"by_filter": YOutlierFilter(), "steps": {...}}}
""")


# =============================================================================
# Section 8: Use Cases for Metadata Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Use Cases for Metadata Branching")
print("-" * 60)

print("""
📋 Common Use Cases:

┌─────────────────────┬─────────────────────────────────────┐
│ Scenario            │ Branch By                           │
├─────────────────────┼─────────────────────────────────────┤
│ Multi-instrument    │ by_metadata: "instrument"           │
│ calibration         │ Different preprocessing per device  │
├─────────────────────┼─────────────────────────────────────┤
│ Batch correction    │ by_metadata: "batch"                │
│                     │ Batch-specific normalization        │
├─────────────────────┼─────────────────────────────────────┤
│ Sample type         │ by_metadata: "variety"              │
│ specific models     │ Type-specific models                │
├─────────────────────┼─────────────────────────────────────┤
│ Farm/site specific  │ by_metadata: "farm"                 │
│                     │ Site-specific calibration           │
├─────────────────────┼─────────────────────────────────────┤
│ Season adjustment   │ by_metadata: "season"               │
│                     │ Season-specific preprocessing       │
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
1. by_metadata is a SEPARATION branch - different samples to different branches
2. Auto-discovery: branches created for each unique value
3. Value mapping: group metadata values into logical branches
4. Per-branch models: train separate models per partition
5. min_samples option: skip partitions with too few samples
6. Combining with duplication branches

Key syntax (v2):
    {"branch": {
        "by_metadata": "column_name",
        "steps": {...},       # Per-value OR shared preprocessing
        "values": {...},      # Optional: group values
        "min_samples": 10,    # Optional: minimum partition size
    }}
    {"merge": "concat"}       # Reassemble samples in order

Separation branch types:
    by_tag       - Branch by tag values
    by_metadata  - Branch by metadata column (this tutorial)
    by_filter    - Branch by filter result
    by_source    - Branch by feature source

Next: D02_concat_transform.py - Feature concatenation
      D06_separation_branches.py - Comprehensive separation branch examples
      D07_value_mapping.py - Advanced value mapping syntax
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
