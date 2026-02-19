"""
U05 - Tagging Analysis: Tag Samples Without Removal
===================================================

Tag samples for analysis without removing them from training.

This tutorial covers:

* The ``tag`` keyword for sample tagging
* Difference between tag and exclude
* Analyzing tagged samples
* Visualizing tag distributions
* Using tags for stratified analysis

Prerequisites
-------------
Complete :ref:`U03_sample_filtering` first.

Next Steps
----------
See :ref:`U06_exclusion_strategies` for exclusion comparison.

Duration: ~4 minutes
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
from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter
from nirs4all.operators.transforms import StandardNormalVariate as SNV

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U05 Tagging Analysis Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Why Tagging?
# =============================================================================
print("\n" + "=" * 60)
print("U05 - Tagging Analysis")
print("=" * 60)

print("""
Tagging marks samples for analysis WITHOUT removing them from training.

  ``tag``:     Mark samples, keep them in training
  ``exclude``: Mark samples AND remove from training

When to use tagging:
  - Investigate outlier impact on model performance
  - Compare model behavior on normal vs. tagged samples
  - Create reports showing outlier characteristics
  - Stratify analysis by sample categories
""")

# =============================================================================
# Section 2: Basic Tagging
# =============================================================================
print("\n" + "-" * 60)
print("Example 1: Basic Tagging with Filters")
print("-" * 60)

print("""
Use the ``tag`` keyword with any filter:

    {"tag": YOutlierFilter(method="iqr", threshold=1.5)}

This creates a tag column (e.g., "y_outlier_iqr") in predictions.
Tagged samples are STILL used for training.
""")

pipeline_tagged = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Tag Y outliers without removing
    {"tag": YOutlierFilter(method="iqr", threshold=1.5)},

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_tagged = nirs4all.run(
    pipeline=pipeline_tagged,
    dataset="sample_data/regression",
    name="TaggedPipeline",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nPredictions: {result_tagged.num_predictions}")

# =============================================================================
# Section 3: Multiple Tags
# =============================================================================
print("\n" + "-" * 60)
print("Example 2: Multiple Tags")
print("-" * 60)

print("""
Apply multiple tags to categorize samples:

    {"tag": YOutlierFilter(method="iqr", tag_name="y_iqr_outlier")}
    {"tag": XOutlierFilter(method="pca_leverage", tag_name="x_pca_outlier")}

Each tag creates a separate column in predictions.
""")

pipeline_multi_tag = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Tag Y outliers
    {"tag": YOutlierFilter(method="iqr", threshold=1.5, tag_name="y_iqr_outlier")},

    # Tag X outliers (PCA leverage-based)
    {"tag": XOutlierFilter(method="pca_leverage", tag_name="x_pca_outlier")},

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_multi = nirs4all.run(
    pipeline=pipeline_multi_tag,
    dataset="sample_data/regression",
    name="MultiTagged",
    verbose=1,
    plots_visible=args.plots
)

print("\nMultiple tags applied")

# =============================================================================
# Section 4: Tag vs Exclude Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Example 3: Tag vs Exclude Comparison")
print("-" * 60)

print("""
Compare model performance:
  - With outliers included (tag only)
  - With outliers removed (exclude)

This helps determine if outliers negatively impact the model.
""")

# Pipeline with tagging only (outliers included)
pipeline_with_outliers = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"tag": YOutlierFilter(method="iqr", threshold=1.5)},  # Tag but keep
    SNV(),
    {"model": PLSRegression(n_components=5)},
]

# Pipeline with exclusion (outliers removed)
pipeline_without_outliers = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},  # Remove
    SNV(),
    {"model": PLSRegression(n_components=5)},
]

result_with = nirs4all.run(
    pipeline=pipeline_with_outliers,
    dataset="sample_data/regression",
    name="WithOutliers",
    verbose=0,
    plots_visible=args.plots
)

result_without = nirs4all.run(
    pipeline=pipeline_without_outliers,
    dataset="sample_data/regression",
    name="WithoutOutliers",
    verbose=0,
    plots_visible=args.plots
)

print(f"\nWith outliers (tag only):    RMSE = {result_with.best_rmse:.4f}")
print(f"Without outliers (exclude):  RMSE = {result_without.best_rmse:.4f}")

if result_without.best_rmse < result_with.best_rmse:
    improvement = (result_with.best_rmse - result_without.best_rmse) / result_with.best_rmse * 100
    print(f"Excluding outliers improved RMSE by {improvement:.1f}%")
else:
    print("Excluding outliers did not improve performance")

# =============================================================================
# Section 5: Tagging for Stratified Analysis
# =============================================================================
print("\n" + "-" * 60)
print("Example 4: Tagging for Stratified Analysis")
print("-" * 60)

print("""
Use tags to analyze model performance on different sample groups.

Example use cases:
  - Compare prediction errors for normal vs. outlier samples
  - Stratify by metadata (instrument, batch, etc.)
  - Identify problematic sample categories
""")

# =============================================================================
# Section 6: Custom Tag Names
# =============================================================================
print("\n" + "-" * 60)
print("Example 5: Custom Tag Names")
print("-" * 60)

print("""
Use descriptive tag names for better analysis:

    {"tag": YOutlierFilter(
        method="iqr",
        threshold=1.5,
        tag_name="extreme_concentration"  # Custom name
    )}
""")

pipeline_custom = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),

    # Custom tag name
    {"tag": YOutlierFilter(
        method="iqr",
        threshold=1.5,
        tag_name="extreme_concentration"
    )},

    SNV(),
    {"model": PLSRegression(n_components=5)},
]

print("Custom tag 'extreme_concentration' will appear in predictions")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
What we learned:
1. ``tag`` marks samples WITHOUT removing them from training
2. Multiple tags can be applied for multi-dimensional analysis
3. Compare tag vs exclude to assess outlier impact
4. Custom tag names improve analysis readability

Syntax:
  {"tag": Filter()}                          # Simple tagging
  {"tag": Filter(tag_name="custom_name")}    # Custom tag name

Key differences:
  tag:     Mark only, samples used in training
  exclude: Mark AND remove from training

Use cases:
  - Outlier impact analysis
  - Stratified performance reporting
  - Sample categorization
  - Quality control documentation

Next: U06_exclusion_strategies.py - Compare exclusion methods
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
