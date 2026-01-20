"""
U03 - Sample Exclusion: Outlier Detection and Quality Control
==============================================================

Exclude outliers and poor-quality samples from training.

This tutorial covers:

* Y-based outlier filtering (IQR, Z-score, MAD)
* X-based outlier filtering (PCA, Mahalanobis)
* Spectral quality checks
* Pipeline integration with the ``exclude`` keyword
* Composite filters

Prerequisites
-------------
Complete :ref:`U01_cv_strategies` first.

Next Steps
----------
See :ref:`U04_aggregation` for prediction aggregation.
See :ref:`U05_tagging_analysis` for tagging without exclusion.

Duration: ~5 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.operators.filters import YOutlierFilter
from nirs4all.operators.filters.base import CompositeFilter

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U03 Sample Filtering Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Sample Exclusion?
# =============================================================================
print("\n" + "=" * 60)
print("U03 - Sample Exclusion")
print("=" * 60)

print("""
Sample exclusion removes outliers and poor-quality samples from training:

  📊 Y-BASED FILTERING (target outliers)
     IQR      - Interquartile Range method (robust)
     Z-score  - Standard deviations from mean
     MAD      - Median Absolute Deviation (most robust)
     Percentile - Fixed percentage in tails

  📈 X-BASED FILTERING (spectral outliers)
     PCA residual - Large reconstruction error
     PCA leverage - High T² in reduced space
     Mahalanobis  - Distance from center

  📉 QUALITY FILTERING
     NaN ratio    - Too many missing values
     Zero ratio   - Flat/dead spectra
     Variance     - Constant spectra
     Value range  - Saturated/clipped values

Benefits:
  ✓ Remove measurement errors
  ✓ Improve model robustness
  ✓ More realistic performance estimates
""")


# =============================================================================
# Section 2: Y-Based Outlier Filtering
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Y-Based Outlier Filtering")
print("-" * 60)

print("""
Filter samples with extreme target values.
The IQR method is robust to outliers.
""")

# Create dataset with outliers
np.random.seed(42)
n_normal = 50
X_normal = np.random.rand(n_normal, 100)
y_normal = np.random.normal(50, 5, n_normal)

# Add outliers
n_outliers = 5
X_outliers = np.random.rand(n_outliers, 100)
y_outliers = np.array([150, -50, 175, -75, 200])

X = np.vstack([X_normal, X_outliers])
y = np.concatenate([y_normal, y_outliers])

print(f"Dataset: {len(X)} samples ({n_normal} normal + {n_outliers} outliers)")
print(f"Y range: [{y.min():.1f}, {y.max():.1f}]")

# Test different methods
methods = [
    ("IQR", YOutlierFilter(method="iqr", threshold=1.5)),
    ("Z-score", YOutlierFilter(method="zscore", threshold=3.0)),
    ("MAD", YOutlierFilter(method="mad", threshold=3.5)),
]

print("\nFiltering results:")
for name, filter_obj in methods:
    filter_obj.fit(X, y)
    mask = filter_obj.get_mask(X, y)
    excluded = (~mask).sum()
    outliers_caught = (~mask[-n_outliers:]).sum()
    print(f"   {name:10s}: Excluded {excluded}, Outliers caught: {outliers_caught}/{n_outliers}")


# =============================================================================
# Section 3: Pipeline Integration
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Pipeline Integration")
print("-" * 60)

print("""
Use the ``exclude`` keyword to integrate filtering into pipelines.
Exclusion happens before model training, on training data only.

The ``exclude`` keyword:
  - Fits the filter on training data
  - Marks matching samples as excluded (they won't be used for training)
  - Creates a tag (e.g., "excluded_y_outlier_iqr") for analysis
  - Does NOT apply during prediction (all prediction samples are used)
""")

pipeline_filtered = [
    # Show Y distribution before exclusion
    "chart_y",

    # Apply exclusion - simple syntax with single filter
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},

    # Show Y distribution after exclusion
    "chart_y",

    StandardNormalVariate(),
    KFold(n_splits=3),
    {"model": PLSRegression(n_components=5)},
]

result_filtered = nirs4all.run(
    pipeline=pipeline_filtered,
    dataset="sample_data/regression",
    name="Filtered",
    verbose=1,
    plots_visible=args.plots
)

print(f"\nFiltered pipeline RMSE: {result_filtered.best_rmse:.4f}")


# =============================================================================
# Section 4: Composite Filters
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Composite Filters")
print("-" * 60)

print("""
Combine multiple filters with different modes:
  "any" - Exclude if ANY filter flags the sample (stricter, default)
  "all" - Exclude only if ALL filters agree (lenient)

With ``exclude``, you can pass a list of filters:
  {"exclude": [Filter1(), Filter2()], "mode": "any"}
""")

# Create filters
filter_iqr = YOutlierFilter(method="iqr", threshold=1.5, reason="iqr")
filter_zscore = YOutlierFilter(method="zscore", threshold=3.0, reason="zscore")

# Composite with "any" mode
composite_any = CompositeFilter(
    filters=[filter_iqr, filter_zscore],
    mode="any"
)

# Composite with "all" mode
composite_all = CompositeFilter(
    filters=[filter_iqr, filter_zscore],
    mode="all"
)

# Test on our outlier dataset
composite_any.fit(X, y)
composite_all.fit(X, y)

mask_any = composite_any.get_mask(X, y)
mask_all = composite_all.get_mask(X, y)

print(f"Mode 'any' (stricter):  Excluded {(~mask_any).sum()}")
print(f"Mode 'all' (lenient):   Excluded {(~mask_all).sum()}")


# =============================================================================
# Section 5: Comparing Filtering Methods
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Comparing Filtering Methods")
print("-" * 60)

print("""
Compare different filtering methods on the same data.
""")

thresholds = [
    ("IQR 1.0 (strict)", YOutlierFilter(method="iqr", threshold=1.0)),
    ("IQR 1.5 (standard)", YOutlierFilter(method="iqr", threshold=1.5)),
    ("IQR 3.0 (lenient)", YOutlierFilter(method="iqr", threshold=3.0)),
    ("Z-score 2.0", YOutlierFilter(method="zscore", threshold=2.0)),
    ("Z-score 3.0", YOutlierFilter(method="zscore", threshold=3.0)),
    ("MAD 3.5", YOutlierFilter(method="mad", threshold=3.5)),
]

print("\nExclusion comparison:")
for name, filter_obj in thresholds:
    filter_obj.fit(X, y)
    mask = filter_obj.get_mask(X, y)
    excluded = (~mask).sum()
    rate = 100 * excluded / len(X)
    print(f"   {name:20s}: {excluded:2d} excluded ({rate:.1f}%)")


# =============================================================================
# Section 6: Visualizing Exclusions
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Visualizing Exclusions")
print("-" * 60)

print("""
Use exclusion_chart and chart options to visualize filtered samples.
""")

pipeline_visual = [
    # Apply exclusion
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},

    # Exclusion chart: PCA-based visualization of excluded samples
    {"exclusion_chart": {"color_by": "status"}},   # Color by included/excluded
    {"exclusion_chart": {"color_by": "y"}},        # Color by target value

    # Regular charts with excluded samples shown
    {"chart_y": {"include_excluded": True, "highlight_excluded": True}},
    {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},

    StandardNormalVariate(),
    KFold(n_splits=3),
    {"model": PLSRegression(n_components=5)},
]

result_visual = nirs4all.run(
    pipeline=pipeline_visual,
    dataset="sample_data/regression",
    name="VisualFilter",
    verbose=1,
    plots_visible=args.plots
)

print("Charts generated (use --plots to view)")


# =============================================================================
# Section 7: Effect on Model Performance
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Effect on Model Performance")
print("-" * 60)

print("""
Compare model performance with and without filtering.
""")

# Without filtering
pipeline_no_filter = [
    StandardNormalVariate(),
    KFold(n_splits=3),
    {"model": PLSRegression(n_components=10)},
]

# With exclusion
pipeline_with_filter = [
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
    StandardNormalVariate(),
    KFold(n_splits=3),
    {"model": PLSRegression(n_components=10)},
]

result_no = nirs4all.run(
    pipeline=pipeline_no_filter,
    dataset="sample_data/regression",
    name="NoFilter",
    verbose=0
)

result_yes = nirs4all.run(
    pipeline=pipeline_with_filter,
    dataset="sample_data/regression",
    name="WithFilter",
    verbose=0
)

print(f"\nWithout filtering: RMSE = {result_no.best_rmse:.4f}")
print(f"With filtering:    RMSE = {result_yes.best_rmse:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Sample Exclusion Options:

  Y-BASED FILTERS:
    YOutlierFilter(method="iqr", threshold=1.5)
    YOutlierFilter(method="zscore", threshold=3.0)
    YOutlierFilter(method="mad", threshold=3.5)
    YOutlierFilter(method="percentile", lower_percentile=5, upper_percentile=95)

  PIPELINE INTEGRATION (exclude keyword):
    # Single filter - simple syntax
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)}

    # Multiple filters with mode
    {"exclude": [Filter1(), Filter2()], "mode": "any"}

    Note: exclude always removes samples from training.
    Use {"tag": Filter()} if you only want to tag without removing.

  COMPOSITE FILTERS:
    CompositeFilter(
        filters=[filter1, filter2],
        mode="any"  # Exclude if ANY flags
    )

  VISUALIZATION:
    {"exclusion_chart": {"color_by": "status"}}
    {"chart_y": {"include_excluded": True, "highlight_excluded": True}}

Best Practices:
  1. Start with conservative thresholds
  2. Combine Y and X filters for thorough cleaning
  3. Document exclusion reasons via filter's tag_name parameter
  4. Compare model performance with/without exclusion
  5. Use {"tag": ...} to analyze outliers without removing them

Next: U04_aggregation.py - Aggregate predictions across folds
      U05_tagging_analysis.py - Tag samples without exclusion
""")
