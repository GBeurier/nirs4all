"""
D04 - Merge Sources: Multi-Source Data Handling
================================================

When working with multiple data sources (e.g., NIR + Raman, or multiple
spectrometers), nirs4all provides ``by_source`` branching and source merge
for per-source processing.

This tutorial covers:

* Multi-source data loading
* by_source branching for per-source preprocessing
* Source merging with unified merge syntax
* Hybrid source and regular branching

Syntax:
  - Branch by source: {"branch": {"by_source": True, "steps": {...}}}
  - Merge sources: {"merge": {"sources": "concat"}}

Prerequisites
-------------
- D03_merge_basics for understanding merge operations

Next Steps
----------
See D05_meta_stacking for advanced stacking patterns.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SavitzkyGolay
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D04 Merge Sources Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Introduction
# =============================================================================
print("\n" + "=" * 60)
print("D04 - Merge Sources: Multi-Source Data Handling")
print("=" * 60)

print("""
Multi-source scenarios in spectroscopy:
  - NIR + Raman spectroscopy
  - Multiple spectrometers (e.g., portable vs. benchtop)
  - Spectral + metadata features

nirs4all handles this with source-aware branching (v2 syntax):

    {"branch": {"by_source": True, "steps": {...}}}  - Per-source processing
    {"merge": {"sources": "concat"}}                 - Combine processed sources

This is a type of SEPARATION branch: each source is processed independently,
then samples are reassembled during merge.
""")


# =============================================================================
# Section 1: Loading Multi-Source Data
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Multi-Source Data Loading")
print("-" * 60)

print("""
Multi-source data can be loaded by specifying multiple paths or using
a structured directory format. Each source has its own feature matrix.
""")

# For demonstration, we'll use the same data twice to simulate multi-source
# In real use, you'd have: {"NIR": "path/to/nir", "Raman": "path/to/raman"}

pipeline_simple = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    PLSRegression(n_components=5),
]

# Single source (baseline)
result_single = nirs4all.run(
    pipeline=pipeline_simple,
    dataset="sample_data/regression",
    name="SingleSource",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nSingle source predictions: {result_single.num_predictions}")


# =============================================================================
# Section 2: Source Branching Basics
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Source Branching Basics")
print("-" * 60)

print("""
Use by_source branching to apply different preprocessing per data source:

    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV(), FirstDerivative()],
            "Raman": [MSC(), SavitzkyGolay()],
            "markers": [StandardScaler()],
        }
    }}

Each source name maps to its pipeline steps. Sources not listed get no processing.
""")

# Multi-source branching configuration using v2 syntax
# Note: This requires multi-source dataset support
pipeline_source_branch = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_source": True,
        "steps": {
            "main": [SNV(), MinMaxScaler()],  # Main spectral source
            "auxiliary": [MSC(), StandardScaler()],  # Could be another source
        }
    }},
    PLSRegression(n_components=5),
]

print("Pipeline with by_source branch:")
print("  - main source: SNV → MinMaxScaler")
print("  - auxiliary source: MSC → StandardScaler")
print("\nNote: Requires multi-source dataset for full functionality")


# =============================================================================
# Section 3: Merge Sources Operations
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Merge Sources Operations")
print("-" * 60)

print("""
After by_source branching, use the unified merge syntax to combine sources:

    {"merge": {"sources": "concat"}}  - Horizontal concatenation
    {"merge": {"sources": "stack"}}   - 3D stacking for CNNs
    {"merge": {"sources": "dict"}}    - Keep as dict for multi-input models

This is the v2 unified syntax. The merge keyword now handles both
branch merging AND source merging through the "sources" key.
""")

# Conceptual pipeline showing source merge with unified syntax
pipeline_merge_sources = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_source": True,
        "steps": {
            "nir": [SNV(), FirstDerivative()],
            "raman": [MSC(), SavitzkyGolay(window_length=11, polyorder=2)],
        }
    }},
    {"merge": {"sources": "concat"}},  # Combine horizontally (unified syntax)
    PLSRegression(n_components=10),
]

print("Merge modes:")
print("  concat  - shape: (n, p_nir + p_raman)")
print("  stack   - shape: (n, 2, max_p) - for 2D convolutions")
print("  dict    - keep as dict for multi-input neural networks")


# =============================================================================
# Section 4: Source-Specific Feature Selection
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Source-Specific Feature Selection")
print("-" * 60)

print("""
Each source can have its own feature selection within the by_source branch.
This allows tuning dimensionality per source type.
""")

pipeline_selection = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_source": True,
        "steps": {
            "spectra": [
                SNV(),
                VarianceThreshold(threshold=0.01),  # Remove low-variance wavelengths
            ],
            "metadata": [
                StandardScaler(),
                # No feature selection - keep all metadata
            ],
        }
    }},
    {"merge": {"sources": "concat"}},
    PLSRegression(n_components=5),
]

print("Source-specific processing:")
print("  spectra  - SNV → VarianceThreshold")
print("  metadata - StandardScaler only")


# =============================================================================
# Section 5: Hybrid Source and Regular Branching
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Hybrid Source and Regular Branching")
print("-" * 60)

print("""
Combine source branching with regular branching for complex experiments:

    1. by_source branch processes each data source (separation branch)
    2. merge sources combines them
    3. Regular branch tests different models (duplication branch)
""")

pipeline_hybrid = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # First: Per-source preprocessing (separation branch)
    {"branch": {
        "by_source": True,
        "steps": {
            "nir": [SNV()],
            "raman": [MSC()],
        }
    }},
    {"merge": {"sources": "concat"}},

    # Then: Compare models on merged features (duplication branch)
    {"branch": {
        "pls": [PLSRegression(n_components=5)],
        "pls_more": [PLSRegression(n_components=10)],
    }},
]

print("Hybrid pipeline:")
print("  1. by_source branch: NIR→SNV, Raman→MSC")
print("  2. merge sources: concatenate")
print("  3. duplication branch: compare PLS components")


# =============================================================================
# Section 6: Source Weights in Merging
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Source Weights in Merging")
print("-" * 60)

print("""
Weight sources differently during merging:

    {"merge": {"sources": {"mode": "concat", "weights": {"nir": 1.0, "raman": 0.5}}}}

This scales source contributions before combining.
""")

pipeline_weighted = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_source": True,
        "steps": {
            "nir": [SNV()],
            "raman": [MSC()],
        }
    }},
    {"merge": {
        "sources": {
            "mode": "concat",
            "weights": {"nir": 1.0, "raman": 0.5}  # Weight Raman less
        }
    }},
    PLSRegression(n_components=5),
]

print("Weighted merge:")
print("  NIR features × 1.0")
print("  Raman features × 0.5")


# =============================================================================
# Section 7: Selective Source Merging
# =============================================================================
print("\n" + "-" * 60)
print("Example 7: Selective Source Merging")
print("-" * 60)

print("""
Select specific sources to include in merge:

    {"merge": {"sources": {"select": ["nir", "markers"], "mode": "concat"}}}

Unselected sources are discarded.
""")

pipeline_selective = [
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"branch": {
        "by_source": True,
        "steps": {
            "nir": [SNV()],
            "raman": [MSC()],
            "markers": [StandardScaler()],
        }
    }},
    {"merge": {
        "sources": {
            "select": ["nir", "markers"],  # Exclude raman
            "mode": "concat"
        }
    }},
    PLSRegression(n_components=5),
]

print("Selective merge:")
print("  Include: nir, markers")
print("  Exclude: raman")


# =============================================================================
# Section 8: Practical Multi-Source Workflow
# =============================================================================
print("\n" + "-" * 60)
print("Example 8: Practical Multi-Source Workflow")
print("-" * 60)

print("""
Complete workflow for multi-instrument fusion:

    1. Load multi-source data (NIR portable + NIR benchtop)
    2. Apply device-specific calibration (by_source branch)
    3. Harmonize preprocessing
    4. Merge and model
""")

# Simulated practical pipeline
pipeline_practical = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),

    # Device-specific preprocessing
    {"branch": {
        "by_source": True,
        "steps": {
            "portable": [
                # Portable needs more aggressive preprocessing
                SNV(),
                SavitzkyGolay(window_length=11, polyorder=2),
                FirstDerivative(),
            ],
            "benchtop": [
                # Benchtop is more stable
                SNV(),
                FirstDerivative(),
            ],
        }
    }},

    # Merge with equal weight
    {"merge": {"sources": "concat"}},

    # Final model
    PLSRegression(n_components=10),
]

print("Multi-instrument fusion pipeline:")
print("  portable: SNV → SavGol → 1st Deriv (aggressive)")
print("  benchtop: SNV → 1st Deriv (stable)")
print("  merge: concatenate sources")
print("  model: PLS(n=10)")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. by_source branching for per-source preprocessing (separation branch)
2. Unified merge syntax: {"merge": {"sources": "concat"}}
3. Merge modes: concat, stack, dict
4. Weighted merging scales source contributions
5. Selective merging filters sources
6. Hybrid workflows combine source and regular branching

Syntax Reference (v2):
  Branch:  {"branch": {"by_source": True, "steps": {...}}}
  Merge:   {"merge": {"sources": "concat"}}

Use cases:
- Multi-spectrometer fusion (NIR + Raman)
- Portable vs. benchtop harmonization
- Spectral + metadata combination
- Multi-instrument calibration transfer

Next: D05_meta_stacking.py - Advanced stacking patterns with MetaModel
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
