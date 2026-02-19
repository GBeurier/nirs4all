"""
U06 - Exclusion Strategies: Comparing Different Exclusion Methods
=================================================================

Compare different exclusion strategies for outlier handling.

This tutorial covers:

* Single vs. multiple filter exclusion
* Exclusion modes: "any" vs "all"
* Y-based vs X-based exclusion
* Combining exclusion with tagging
* Best practices for exclusion

Prerequisites
-------------
Complete :ref:`U03_sample_filtering` and :ref:`U05_tagging_analysis` first.

Next Steps
----------
See :ref:`D06_separation_branches` for advanced branching by filter results.

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
from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U06 Exclusion Strategies Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Exclusion Basics
# =============================================================================
print("\n" + "=" * 60)
print("U06 - Exclusion Strategies")
print("=" * 60)

print("""
The ``exclude`` keyword removes samples from training:

    {"exclude": Filter()}                         # Single filter
    {"exclude": [Filter1(), Filter2()], "mode": "any"}  # Multiple filters

Key points:
  - Excluded samples are NOT used for training
  - A tag is still created for analysis
  - Does NOT apply during prediction (all samples predicted)
""")

# =============================================================================
# Section 2: Single Filter Exclusion
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Single Filter Exclusion")
print("-" * 60)

print("""
Simplest exclusion: one filter, one criterion.

    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)}
""")

pipeline_single = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Single filter exclusion
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_single = nirs4all.run(
    pipeline=pipeline_single,
    dataset="sample_data/regression",
    name="SingleFilter",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nSingle filter RMSE: {result_single.best_rmse:.4f}")

# =============================================================================
# Section 3: Multiple Filters - "any" Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Multiple Filters - 'any' Mode (Strict)")
print("-" * 60)

print("""
"any" mode: Exclude if ANY filter flags the sample (strict).

    {"exclude": [Filter1(), Filter2()], "mode": "any"}

Use when you want to remove samples with ANY outlier characteristic.
""")

pipeline_any = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Exclude if ANY filter flags the sample
    {
        "exclude": [
            YOutlierFilter(method="iqr", threshold=1.5),
            XOutlierFilter(method="pca_leverage"),
        ],
        "mode": "any",  # Exclude if Y OR X is outlier
    },

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_any = nirs4all.run(
    pipeline=pipeline_any,
    dataset="sample_data/regression",
    name="AnyMode",
    verbose=1,
    plots_visible=args.plots
)

print(f"\n'any' mode RMSE: {result_any.best_rmse:.4f}")

# =============================================================================
# Section 4: Multiple Filters - "all" Mode
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Multiple Filters - 'all' Mode (Lenient)")
print("-" * 60)

print("""
"all" mode: Exclude only if ALL filters agree (lenient).

    {"exclude": [Filter1(), Filter2()], "mode": "all"}

Use when you want to remove only samples that are outliers
by MULTIPLE criteria (high confidence outliers).
""")

pipeline_all = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Exclude only if ALL filters flag the sample
    {
        "exclude": [
            YOutlierFilter(method="iqr", threshold=1.5),
            XOutlierFilter(method="pca_leverage"),
        ],
        "mode": "all",  # Exclude only if Y AND X are outliers
    },

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_all = nirs4all.run(
    pipeline=pipeline_all,
    dataset="sample_data/regression",
    name="AllMode",
    verbose=1,
    plots_visible=args.plots
)

print(f"\n'all' mode RMSE: {result_all.best_rmse:.4f}")

# =============================================================================
# Section 5: Y-Based vs X-Based Exclusion
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Y-Based vs X-Based Exclusion")
print("-" * 60)

print("""
Compare Y-based (target outliers) vs X-based (spectral outliers):

  Y-based: Remove samples with extreme target values
  X-based: Remove samples with unusual spectral patterns
""")

# Y-based exclusion only
pipeline_y = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
    SNV(),
    {"model": PLSRegression(n_components=5)},
]

# X-based exclusion only
pipeline_x = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"exclude": XOutlierFilter(method="pca_leverage")},
    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_y = nirs4all.run(
    pipeline=pipeline_y,
    dataset="sample_data/regression",
    name="YExclusion",
    verbose=0,
    plots_visible=args.plots
)

result_x = nirs4all.run(
    pipeline=pipeline_x,
    dataset="sample_data/regression",
    name="XExclusion",
    verbose=0,
    plots_visible=args.plots
)

print(f"\nY-based exclusion RMSE: {result_y.best_rmse:.4f}")
print(f"X-based exclusion RMSE: {result_x.best_rmse:.4f}")

# =============================================================================
# Section 6: Combined Tag + Exclude
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Combined Tag + Exclude")
print("-" * 60)

print("""
Tag some outliers for analysis, exclude only the extreme ones:

    {"tag": YOutlierFilter(method="iqr", threshold=1.5)}   # Tag mild outliers
    {"exclude": YOutlierFilter(method="iqr", threshold=3)} # Remove extreme

This allows analyzing the impact of mild outliers while removing extremes.
""")

pipeline_combined = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Tag mild Y outliers (for analysis)
    {"tag": YOutlierFilter(method="iqr", threshold=1.5, tag_name="mild_y_outlier")},

    # Tag X outliers (for analysis)
    {"tag": XOutlierFilter(method="pca_leverage", tag_name="x_outlier")},

    # Exclude only extreme Y outliers
    {"exclude": YOutlierFilter(method="iqr", threshold=3.0)},

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_combined = nirs4all.run(
    pipeline=pipeline_combined,
    dataset="sample_data/regression",
    name="CombinedTagExclude",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nCombined approach RMSE: {result_combined.best_rmse:.4f}")

# =============================================================================
# Section 7: Strategy Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 6: Strategy Comparison Summary")
print("-" * 60)

# No exclusion baseline
pipeline_none = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_none = nirs4all.run(
    pipeline=pipeline_none,
    dataset="sample_data/regression",
    name="NoExclusion",
    verbose=0,
    plots_visible=args.plots
)

print("""
Strategy Comparison:
""")
print(f"  No exclusion:          RMSE = {result_none.best_rmse:.4f}")
print(f"  Single Y filter:       RMSE = {result_single.best_rmse:.4f}")
print(f"  Any mode (Y or X):     RMSE = {result_any.best_rmse:.4f}")
print(f"  All mode (Y and X):    RMSE = {result_all.best_rmse:.4f}")
print(f"  Y-only:                RMSE = {result_y.best_rmse:.4f}")
print(f"  X-only:                RMSE = {result_x.best_rmse:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. Single filter: Simple, focused exclusion
2. "any" mode: Strict - exclude if ANY filter flags
3. "all" mode: Lenient - exclude only if ALL agree
4. Y-based vs X-based serve different purposes
5. Combine tag + exclude for comprehensive analysis

Syntax:
  {"exclude": Filter()}                            # Single filter
  {"exclude": [F1(), F2()], "mode": "any"}         # Any filter (strict)
  {"exclude": [F1(), F2()], "mode": "all"}         # All filters (lenient)

Decision guide:
  - Start with no exclusion as baseline
  - Try Y-based first (target outliers often problematic)
  - Add X-based if spectral anomalies exist
  - Use "all" mode to be conservative
  - Use "any" mode for thorough cleaning

Best practices:
  1. Always compare with no-exclusion baseline
  2. Use conservative thresholds initially
  3. Tag before excluding to analyze impact
  4. Document exclusion criteria

Next: D06_separation_branches.py - Branch by filter results
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
