"""
D17 - Metadata Branching: Partitioning by Metadata
===================================================

metadata_partitioner creates branches based on sample metadata,
enabling group-specific processing or stratified analysis.

This tutorial covers:

* metadata_partitioner basics
* Partitioning by categorical metadata
* Partitioning by numeric ranges
* Custom partition functions
* Combining with regular branching

Prerequisites
-------------
- D01_branching_basics for branching concepts

Next Steps
----------
See D18_concat_transform for feature concatenation.

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
parser = argparse.ArgumentParser(description='D17 Metadata Branching Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D17 - Metadata Branching: Partitioning by Metadata")
print("=" * 60)

print("""
metadata_partitioner creates branches based on sample attributes:

  Use cases:
  - Different preprocessing per instrument
  - Separate models per sample type
  - Stratified analysis by metadata field

  Syntax:
    {"metadata_partitioner": {
        "column": "instrument",     # Metadata column
        "branches": {
            "InstrumentA": [SNV()],
            "InstrumentB": [MSC()],
        }
    }}
""")


# =============================================================================
# Section 1: Basic Metadata Partitioning
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic Metadata Partitioning")
print("-" * 60)

print("""
Partition samples by categorical metadata:

    {"metadata_partitioner": {
        "column": "variety",
        "branches": {
            "variety_A": [preprocessing_for_A],
            "variety_B": [preprocessing_for_B],
        }
    }}

Samples are routed to their matching branch.
""")

pipeline_basic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"metadata_partitioner": {
        "column": "variety",
        "branches": {
            "variety_A": [SNV()],
            "variety_B": [MSC()],
            "variety_C": [FirstDerivative()],
        }
    }},
    PLSRegression(n_components=5),
]

print("Pipeline with metadata_partitioner:")
print("  - Samples partitioned by 'variety' column")
print("  - Each variety gets different preprocessing")
print("  - Model trained per partition")


# =============================================================================
# Section 2: Default Branch for Unmatched Samples
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Default Branch for Unmatched Samples")
print("-" * 60)

print("""
Handle samples that don't match any specified value:

    {"metadata_partitioner": {
        "column": "instrument",
        "branches": {
            "known_instrument": [specific_preprocessing],
        },
        "default": [fallback_preprocessing],  # For unknown instruments
    }}
""")

pipeline_default = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"metadata_partitioner": {
        "column": "instrument",
        "branches": {
            "NIR_500": [SNV()],
            "NIR_700": [MSC()],
        },
        "default": [StandardScaler()],  # Fallback for unknown instruments
    }},
    PLSRegression(n_components=5),
]

print("With default branch:")
print("  - Known instruments get specific preprocessing")
print("  - Unknown instruments use default preprocessing")


# =============================================================================
# Section 3: Numeric Range Partitioning
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Numeric Range Partitioning")
print("-" * 60)

print("""
Partition by numeric ranges:

    {"metadata_partitioner": {
        "column": "temperature",
        "ranges": {
            "cold": [-50, 10],    # -50 to 10°C
            "normal": [10, 30],   # 10 to 30°C
            "hot": [30, 100],     # 30 to 100°C
        },
        "branches": {
            "cold": [cold_preprocessing],
            "normal": [normal_preprocessing],
            "hot": [hot_preprocessing],
        }
    }}
""")

pipeline_ranges = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"metadata_partitioner": {
        "column": "temperature",
        "ranges": {
            "low": [0, 20],
            "medium": [20, 40],
            "high": [40, 100],
        },
        "branches": {
            "low": [SNV()],
            "medium": [SNV(), FirstDerivative()],
            "high": [MSC(), FirstDerivative()],
        }
    }},
    PLSRegression(n_components=5),
]

print("Numeric range partitioning:")
print("  - Samples routed by temperature value")
print("  - Each range gets appropriate preprocessing")


# =============================================================================
# Section 4: Custom Partition Functions
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Custom Partition Functions")
print("-" * 60)

print("""
Use functions for complex partitioning logic:

    def is_outlier(row):
        return row['concentration'] > 1000

    {"metadata_partitioner": {
        "function": is_outlier,
        "branches": {
            True: [outlier_preprocessing],
            False: [normal_preprocessing],
        }
    }}
""")

def partition_by_quality(metadata_row):
    """Partition samples by quality score."""
    score = metadata_row.get('quality_score', 0)
    if score >= 90:
        return 'high_quality'
    elif score >= 70:
        return 'medium_quality'
    else:
        return 'low_quality'

print("Custom partition function defined:")
print("  - partition_by_quality(row) returns quality category")
print("  - Each category routed to appropriate branch")


# =============================================================================
# Section 5: Multi-Column Partitioning
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Multi-Column Partitioning")
print("-" * 60)

print("""
Partition based on multiple metadata columns:

    {"metadata_partitioner": {
        "columns": ["instrument", "operator"],
        "branches": {
            ("NIR_500", "Alice"): [preproc_1],
            ("NIR_500", "Bob"): [preproc_2],
            ("NIR_700", "Alice"): [preproc_3],
        }
    }}
""")

pipeline_multi = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"metadata_partitioner": {
        "columns": ["instrument", "batch"],
        "branches": {
            ("Instrument_A", "Batch_1"): [SNV()],
            ("Instrument_A", "Batch_2"): [SNV(), FirstDerivative()],
            ("Instrument_B", "Batch_1"): [MSC()],
        },
        "default": [StandardScaler()],
    }},
    PLSRegression(n_components=5),
]

print("Multi-column partitioning:")
print("  - Combinations of instrument + batch")
print("  - Fine-grained control per subgroup")


# =============================================================================
# Section 6: Combining with Regular Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Combining with Regular Branching")
print("-" * 60)

print("""
Nest metadata_partitioner inside regular branches:

    {"branch": {
        "model_A": [
            {"metadata_partitioner": {...}},
            ModelA()
        ],
        "model_B": [
            {"metadata_partitioner": {...}},
            ModelB()
        ]
    }}
""")

pipeline_combined = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "pls_path": [
            {"metadata_partitioner": {
                "column": "instrument",
                "branches": {
                    "NIR_500": [SNV()],
                    "NIR_700": [MSC()],
                },
                "default": [StandardScaler()],
            }},
            PLSRegression(n_components=5),
        ],
        "simple_path": [
            SNV(),
            PLSRegression(n_components=10),
        ],
    }},
]

print("Combined branching:")
print("  - pls_path: metadata-aware preprocessing")
print("  - simple_path: uniform preprocessing")


# =============================================================================
# Section 7: Partition Statistics
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Partition Statistics")
print("-" * 60)

print("""
Analyze partition distribution:

    result = nirs4all.run(pipeline, dataset)

    # Get partition info from predictions
    partitions = result.predictions.get_unique_values('metadata_partition')
    for p in partitions:
        subset = result.predictions.filter(metadata_partition=p)
        print(f"{p}: {len(subset)} samples")
""")


# =============================================================================
# Section 8: Use Cases for Metadata Partitioning
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Use Cases for Metadata Partitioning")
print("-" * 60)

print("""
📋 Common Use Cases:

┌─────────────────────┬─────────────────────────────────────┐
│ Scenario            │ Partition By                        │
├─────────────────────┼─────────────────────────────────────┤
│ Multi-instrument    │ "instrument" column                 │
│ calibration         │ Different preprocessing per device  │
├─────────────────────┼─────────────────────────────────────┤
│ Batch correction    │ "batch" or "date" column            │
│                     │ Batch-specific normalization        │
├─────────────────────┼─────────────────────────────────────┤
│ Sample type         │ "variety" or "material" column      │
│ specific models     │ Type-specific models                │
├─────────────────────┼─────────────────────────────────────┤
│ Quality control     │ "quality_score" ranges              │
│                     │ Different handling for outliers     │
├─────────────────────┼─────────────────────────────────────┤
│ Environmental       │ "temperature", "humidity" ranges    │
│ compensation        │ Environment-specific corrections    │
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
1. metadata_partitioner routes samples by metadata
2. Categorical partitioning by column values
3. Numeric range partitioning
4. Custom functions for complex logic
5. Multi-column combinations
6. Default branch for unmatched samples
7. Combining with regular branching

Key syntax:
    {"metadata_partitioner": {
        "column": "column_name",          # or "columns": [...]
        "branches": {
            "value1": [preprocessing1],
            "value2": [preprocessing2],
        },
        "default": [fallback],            # Optional
        "ranges": {...},                  # For numeric
        "function": custom_fn,            # For complex logic
    }}

Next: D18_concat_transform.py - Feature concatenation
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
