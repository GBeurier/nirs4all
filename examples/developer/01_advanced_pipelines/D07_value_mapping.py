"""
D07 - Value Mapping: User-Friendly Grouping Syntax
==================================================

Value mapping allows grouping multiple metadata values into logical branches
with user-friendly names.

This tutorial covers:

* Basic value mapping syntax
* Grouping multiple values
* Default handling for unmapped values
* Combining value mapping with per-group steps
* Practical examples

Prerequisites
-------------
- D06_separation_branches for separation branch concepts

Next Steps
----------
See D01_metadata_branching for comprehensive metadata branching.

Duration: ~5 minutes
Difficulty: ★★★★☆
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
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D07 Value Mapping Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D07 - Value Mapping: User-Friendly Grouping Syntax")
print("=" * 60)

print("""
Value mapping groups metadata values into logical branches:

  Without mapping (raw values):
    {"by_metadata": "instrument"}
    → Creates branches: "NIR_100", "NIR_200", "NIR_500", "NIR_700"

  With mapping (grouped values):
    {"by_metadata": "instrument", "values": {"portable": [...], "benchtop": [...]}}
    → Creates branches: "portable", "benchtop"

Benefits:
  - Clearer branch names
  - Group similar values together
  - Different preprocessing per group
  - Better organization and documentation
""")


# =============================================================================
# Section 1: Basic Value Mapping
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic Value Mapping")
print("-" * 60)

print("""
Map metadata values to user-friendly group names:

    {"branch": {
        "by_metadata": "instrument",
        "values": {
            "portable": ["NIR_100", "NIR_200"],
            "benchtop": ["NIR_500", "NIR_700"],
        },
        "steps": [SNV()],  # Same for all groups
    }}
""")

pipeline_basic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        "by_metadata": "instrument",
        "values": {
            "portable": ["NIR_100", "NIR_200"],
            "benchtop": ["NIR_500", "NIR_700"],
        },
        "steps": [SNV()],
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Value mapping:
  "portable" branch: samples with instrument=NIR_100 OR NIR_200
  "benchtop" branch: samples with instrument=NIR_500 OR NIR_700
""")


# =============================================================================
# Section 2: Per-Group Steps with Value Mapping
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Per-Group Steps with Value Mapping")
print("-" * 60)

print("""
Combine value mapping with per-group preprocessing:

    {"branch": {
        "by_metadata": "instrument",
        "values": {
            "portable": ["NIR_100", "NIR_200"],
            "benchtop": ["NIR_500", "NIR_700"],
        },
        "steps": {
            "portable": [SNV(), FirstDerivative()],  # Aggressive
            "benchtop": [SNV()],                      # Minimal
        }
    }}
""")

pipeline_per_group = [
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

    {"model": PLSRegression(n_components=5)},
]

print("""Per-group preprocessing:
  portable devices → SNV → FirstDerivative (more processing)
  benchtop devices → SNV only (stable instruments)
""")


# =============================================================================
# Section 3: Quality-Based Grouping
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Quality-Based Grouping")
print("-" * 60)

print("""
Group samples by quality categories:

    {"branch": {
        "by_metadata": "quality_grade",
        "values": {
            "high_quality": ["A", "A+"],
            "medium_quality": ["B", "B+"],
            "low_quality": ["C", "D"],
        },
        "steps": {...}
    }}
""")

pipeline_quality = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        "by_metadata": "quality_grade",
        "values": {
            "high_quality": ["A", "A+"],
            "standard": ["B", "B+", "C"],
        },
        "steps": {
            "high_quality": [SNV()],              # Minimal processing
            "standard": [SNV(), FirstDerivative()], # More processing
        }
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Quality grouping:
  high_quality (A, A+) → minimal preprocessing
  standard (B, B+, C) → more aggressive preprocessing
""")


# =============================================================================
# Section 4: Geographic/Farm Grouping
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Geographic/Farm Grouping")
print("-" * 60)

print("""
Group farms by region:

    {"branch": {
        "by_metadata": "farm",
        "values": {
            "north_region": ["Farm_A", "Farm_B", "Farm_C"],
            "south_region": ["Farm_D", "Farm_E"],
        },
        "steps": {...}
    }}
""")

pipeline_geographic = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        "by_metadata": "farm",
        "values": {
            "north_region": ["Farm_A", "Farm_B", "Farm_C"],
            "south_region": ["Farm_D", "Farm_E"],
        },
        "steps": [SNV()],
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Geographic grouping:
  north_region: Farm_A, Farm_B, Farm_C
  south_region: Farm_D, Farm_E
""")


# =============================================================================
# Section 5: Season Grouping
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Season Grouping")
print("-" * 60)

print("""
Group months into seasons:

    {"branch": {
        "by_metadata": "harvest_month",
        "values": {
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
            "winter": [12, 1, 2],
        },
        "steps": {...}
    }}
""")

pipeline_season = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        "by_metadata": "harvest_month",
        "values": {
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
            "winter": [12, 1, 2],
        },
        "steps": {
            "spring": [SNV()],
            "summer": [SNV(), FirstDerivative()],
            "fall": [SNV()],
            "winter": [MSC()],
        }
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Season grouping:
  spring (Mar-May) → SNV
  summer (Jun-Aug) → SNV + Derivative
  fall (Sep-Nov) → SNV
  winter (Dec-Feb) → MSC
""")


# =============================================================================
# Section 6: Instrument Type + Year Grouping
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Complex Grouping Patterns")
print("-" * 60)

print("""
Complex grouping patterns are possible:

    {"branch": {
        "by_metadata": "instrument_id",
        "values": {
            "old_portable": ["P001", "P002"],      # Old portables
            "new_portable": ["P010", "P011"],      # New portables
            "lab_primary": ["L001"],                # Primary lab
            "lab_backup": ["L002", "L003"],        # Backup labs
        },
        "steps": {...}
    }}
""")

pipeline_complex = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    {"branch": {
        "by_metadata": "instrument_id",
        "values": {
            "old_portable": ["P001", "P002"],
            "new_portable": ["P010", "P011"],
            "lab_equipment": ["L001", "L002", "L003"],
        },
        "steps": {
            "old_portable": [SNV(), FirstDerivative(), MSC()],  # Heavy processing
            "new_portable": [SNV(), FirstDerivative()],          # Medium
            "lab_equipment": [SNV()],                             # Minimal
        }
    }},
    {"merge": "concat"},

    {"model": PLSRegression(n_components=5)},
]

print("""Complex grouping:
  old_portable (P001, P002) → heavy preprocessing
  new_portable (P010, P011) → medium preprocessing
  lab_equipment (L001, L002, L003) → minimal preprocessing
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Value mapping groups metadata values into logical branches
2. Use "values" dict to define groupings
3. Group names become branch names
4. Combine with "steps" dict for per-group preprocessing
5. Useful for instruments, quality grades, regions, seasons, etc.

Syntax:
    {"branch": {
        "by_metadata": "column_name",
        "values": {
            "group_a": ["value1", "value2"],
            "group_b": ["value3", "value4"],
        },
        "steps": {
            "group_a": [preprocessing_a],
            "group_b": [preprocessing_b],
        }
        # OR: "steps": [shared_preprocessing] for all groups
    }}
    {"merge": "concat"}

Common use cases:
  - Instrument type grouping (portable vs benchtop)
  - Quality grade grouping (A/B grades, good/bad)
  - Geographic grouping (regions, farms)
  - Temporal grouping (seasons, years)
  - Equipment age grouping (old vs new)

Benefits:
  - Clear, readable branch names
  - Logical organization
  - Easy to modify groupings
  - Self-documenting pipelines

Next: D01_metadata_branching.py - Full metadata branching reference
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
