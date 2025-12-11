"""
Q28: Sample Filtering Example
=============================
Demonstrates the sample filtering capabilities for removing outliers and
poor-quality samples from training datasets.

Sample filtering is a non-destructive operation that marks samples as excluded
in the indexer, allowing:
- Easy inspection of what was excluded
- Reversible exclusion (samples can be re-included)
- Audit trail of filtering decisions

Scenarios covered:
1. Y-based Outlier Filtering (IQR method)
2. Y-based Outlier Filtering (Z-score method)
3. Y-based Outlier Filtering (Percentile method)
4. Y-based Outlier Filtering (MAD method - robust)
5. Pipeline Integration with sample_filter keyword
6. Composite Filters (combining multiple filters)
7. Filter Statistics and Reporting
8. Exclusion Chart Visualization (Phase 4)
9. Filtering Report Generator (Phase 4)
10. Charts with include_excluded Parameter (Phase 4)
11. Edge Case Handling (Phase 5)
12. X-based Outlier Filtering (Phase 5)
13. Spectral Quality Filtering (Phase 5)
14. Best Practices (Phase 5)
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'

import argparse
import numpy as np

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters import YOutlierFilter, FilteringReportGenerator
from nirs4all.operators.filters.base import CompositeFilter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q28 Sample Filtering Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


def create_dataset_with_outliers():
    """Create a synthetic dataset with known outliers for demonstration."""
    np.random.seed(42)

    # Normal samples (50 samples)
    n_normal = 50
    X_normal = np.random.rand(n_normal, 100)
    y_normal = np.random.normal(50, 5, n_normal)  # Mean=50, Std=5

    # Outlier samples (5 samples with extreme y values)
    n_outliers = 5
    X_outliers = np.random.rand(n_outliers, 100)
    y_outliers = np.array([150, -50, 175, -75, 200])  # Extreme values

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.concatenate([y_normal, y_outliers])

    # Create dataset
    dataset = SpectroDataset("demo_outliers")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)

    return dataset, n_normal, n_outliers


def scenario_1_iqr_filtering():
    """Scenario 1: IQR-based outlier filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: IQR-based Outlier Filtering")
    print("=" * 70)
    print("""
The IQR (Interquartile Range) method identifies outliers based on the
spread of the middle 50% of data. Samples outside Q1 - 1.5*IQR and
Q3 + 1.5*IQR are considered outliers.

This method is robust to extreme outliers in the data.
""")

    dataset, n_normal, n_outliers = create_dataset_with_outliers()

    # Create IQR filter
    filter_obj = YOutlierFilter(method="iqr", threshold=1.5)

    # Get training data
    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d", include_augmented=False)
    y = dataset.y(selector, include_augmented=False)
    sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

    print(f"Total samples: {len(X)}")
    print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Y mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Fit and get mask
    filter_obj.fit(X, y)
    mask = filter_obj.get_mask(X, y)

    print(f"\nFilter bounds: [{filter_obj.lower_bound_:.2f}, {filter_obj.upper_bound_:.2f}]")
    print(f"Samples kept: {mask.sum()}")
    print(f"Samples excluded: {(~mask).sum()}")

    # Get filter statistics
    stats = filter_obj.get_filter_stats(X, y)
    print(f"\nExclusion rate: {stats['exclusion_rate']:.1%}")


def scenario_2_zscore_filtering():
    """Scenario 2: Z-score based outlier filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Z-score Based Outlier Filtering")
    print("=" * 70)
    print("""
The Z-score method identifies outliers based on standard deviations from
the mean. Samples with |z-score| > threshold are considered outliers.

Common threshold values:
- 2.0: ~5% of normal data excluded
- 3.0: ~0.3% of normal data excluded (3-sigma rule)
""")

    dataset, _, _ = create_dataset_with_outliers()

    # Compare different thresholds
    for threshold in [2.0, 3.0]:
        filter_obj = YOutlierFilter(method="zscore", threshold=threshold)

        selector = {"partition": "train"}
        X = dataset.x(selector, layout="2d")
        y = dataset.y(selector)

        filter_obj.fit(X, y)
        stats = filter_obj.get_filter_stats(X, y)

        print(f"\nThreshold: {threshold}")
        print(f"  Bounds: [{filter_obj.lower_bound_:.2f}, {filter_obj.upper_bound_:.2f}]")
        print(f"  Excluded: {stats['n_excluded']} ({stats['exclusion_rate']:.1%})")


def scenario_3_percentile_filtering():
    """Scenario 3: Percentile-based outlier filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Percentile-Based Outlier Filtering")
    print("=" * 70)
    print("""
The percentile method excludes samples in the tails of the distribution.
For example, 1st and 99th percentile bounds exclude the extreme 2%.

This method is useful when you want to exclude a specific proportion
of extreme values regardless of the distribution shape.
""")

    dataset, _, _ = create_dataset_with_outliers()

    filter_obj = YOutlierFilter(
        method="percentile",
        lower_percentile=5.0,
        upper_percentile=95.0
    )

    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d")
    y = dataset.y(selector)

    filter_obj.fit(X, y)
    stats = filter_obj.get_filter_stats(X, y)

    print(f"Percentile range: [{stats['lower_percentile']}%, {stats['upper_percentile']}%]")
    print(f"Bounds: [{filter_obj.lower_bound_:.2f}, {filter_obj.upper_bound_:.2f}]")
    print(f"Excluded: {stats['n_excluded']} ({stats['exclusion_rate']:.1%})")


def scenario_4_mad_filtering():
    """Scenario 4: MAD (Median Absolute Deviation) based filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: MAD-Based Outlier Filtering (Robust)")
    print("=" * 70)
    print("""
The MAD (Median Absolute Deviation) method is more robust to outliers
than z-score because it uses median instead of mean.

MAD = median(|x - median(x)|)

This method is recommended when you have extreme outliers that could
distort mean/std-based methods.
""")

    dataset, _, _ = create_dataset_with_outliers()

    filter_obj = YOutlierFilter(method="mad", threshold=3.5)

    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d")
    y = dataset.y(selector)

    filter_obj.fit(X, y)
    stats = filter_obj.get_filter_stats(X, y)

    print(f"Excluded: {stats['n_excluded']} ({stats['exclusion_rate']:.1%})")
    print("Threshold: 3.5 (scaled MADs from median)")
    print(f"Center (median): {filter_obj.center_:.2f}")


def scenario_5_pipeline_integration():
    """Scenario 5: Pipeline integration with sample_filter keyword."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: Pipeline Integration")
    print("=" * 70)
    print("""
Sample filtering can be integrated directly into pipelines using the
'sample_filter' keyword. This allows automatic filtering before model
training.

The filter:
1. Fits on training data only
2. Marks outliers as excluded in the indexer
3. Excluded samples are automatically skipped in subsequent steps
""")

    # Use a real dataset
    dataset_config = DatasetConfigs("sample_data/regression_2")

    # Pipeline with sample filtering
    pipeline = [
        "chart_y",  # Show y distribution before filtering
        {
            "sample_filter": {
                "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
                "mode": "any",
                "report": True,  # Print filtering report
            }
        },
        "chart_y",  # Show y distribution after filtering
        "snv",
        {"split": KFold(n_splits=3)},
        {"model": PLSRegression(n_components=5)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="filtered_regression")
    runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)

    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"Note: Pipeline example requires proper dataset setup: {e}")


def scenario_6_composite_filters():
    """Scenario 6: Combining multiple filters."""
    print("\n" + "=" * 70)
    print("SCENARIO 6: Composite Filters")
    print("=" * 70)
    print("""
Multiple filters can be combined using CompositeFilter with two modes:

- "any": Exclude if ANY filter flags the sample (intersection of kept samples)
- "all": Exclude only if ALL filters flag the sample (union of kept samples)

This allows flexible multi-criteria filtering.
""")

    dataset, _, _ = create_dataset_with_outliers()

    # Create individual filters
    filter_iqr = YOutlierFilter(method="iqr", threshold=1.5, reason="iqr")
    filter_zscore = YOutlierFilter(method="zscore", threshold=3.0, reason="zscore")

    # Create composites
    composite_any = CompositeFilter(
        filters=[filter_iqr, filter_zscore],
        mode="any"
    )
    composite_all = CompositeFilter(
        filters=[filter_iqr, filter_zscore],
        mode="all"
    )

    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d")
    y = dataset.y(selector)

    # Fit and compare
    composite_any.fit(X, y)
    composite_all.fit(X, y)

    stats_any = composite_any.get_filter_stats(X, y)
    stats_all = composite_all.get_filter_stats(X, y)

    print("Mode 'any' (stricter - exclude if ANY flags):")
    print(f"  Excluded: {stats_any['n_excluded']}")
    print(f"  Exclusion rate: {stats_any['exclusion_rate']:.1%}")

    print("\nMode 'all' (lenient - exclude only if ALL flag):")
    print(f"  Excluded: {stats_all['n_excluded']}")
    print(f"  Exclusion rate: {stats_all['exclusion_rate']:.1%}")


def scenario_7_exclusion_tracking():
    """Scenario 7: Tracking and managing exclusions."""
    print("\n" + "=" * 70)
    print("SCENARIO 7: Exclusion Tracking and Management")
    print("=" * 70)
    print("""
Exclusions are tracked in the indexer, allowing:
- Inspection of excluded samples
- Summary statistics by reason
- Reverting exclusions (re-including samples)
""")

    dataset, _, _ = create_dataset_with_outliers()

    # Mark some samples as excluded
    filter_obj = YOutlierFilter(method="iqr", threshold=1.5)

    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d")
    y = dataset.y(selector)
    sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

    filter_obj.fit(X, y)
    mask = filter_obj.get_mask(X, y)
    exclude_indices = sample_indices[~mask].tolist()

    # Mark as excluded
    n_excluded = dataset._indexer.mark_excluded(
        exclude_indices,
        reason="iqr_outlier"
    )
    print(f"Marked {n_excluded} samples as excluded")

    # Get exclusion summary
    summary = dataset._indexer.get_exclusion_summary()
    print("\nExclusion Summary:")
    print(f"  Total excluded: {summary['total_excluded']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Exclusion rate: {summary['exclusion_rate']:.1%}")
    print(f"  By reason: {summary['by_reason']}")

    # View excluded samples
    excluded_df = dataset._indexer.get_excluded_samples()
    print(f"\nExcluded samples DataFrame:\n{excluded_df}")

    # Demonstrate re-including samples
    n_reset = dataset._indexer.reset_exclusions()
    print(f"\nReset {n_reset} exclusions")

    summary_after = dataset._indexer.get_exclusion_summary()
    print(f"Exclusions after reset: {summary_after['total_excluded']}")


def scenario_8_exclusion_chart():
    """Scenario 8: Visualizing excluded samples with exclusion_chart."""
    print("\n" + "=" * 70)
    print("SCENARIO 8: Exclusion Chart Visualization")
    print("=" * 70)
    print("""
The 'exclusion_chart' controller creates a 2D scatter plot using PCA
to visualize which samples have been excluded.

Features:
- PCA-based 2D/3D projection of feature space
- Color coding by: status, target value (y), or exclusion reason
- Shows excluded samples with distinct markers
- Useful for understanding filtering decisions
""")

    # Use a real dataset
    dataset_config = DatasetConfigs("sample_data/regression_2")

    # Pipeline with filtering and exclusion visualization
    pipeline = [
        "chart_2d",  # Show spectra before filtering
        {
            "sample_filter": {
                "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
                "report": True,
            }
        },
        # Visualize exclusions with different color modes
        {"exclusion_chart": {"color_by": "status"}},  # Color by included/excluded
        {"exclusion_chart": {"color_by": "y"}},       # Color by target value
        {"exclusion_chart": {"color_by": "reason"}},  # Color by exclusion reason
        "snv",
        {"split": KFold(n_splits=3)},
        {"model": PLSRegression(n_components=5)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="exclusion_visualization")
    runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)

    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print("\nExclusion chart example completed!")
    except Exception as e:
        print(f"Note: Exclusion chart example requires proper dataset setup: {e}")


def scenario_9_filtering_report():
    """Scenario 9: Using the FilteringReportGenerator."""
    print("\n" + "=" * 70)
    print("SCENARIO 9: Filtering Report Generator")
    print("=" * 70)
    print("""
The FilteringReportGenerator creates comprehensive reports about
sample filtering operations, including:
- Per-filter statistics and breakdown
- Combined exclusion results
- JSON export for auditing
- Comparison between multiple filters
""")

    dataset, n_normal, n_outliers = create_dataset_with_outliers()

    # Create filters
    filter_iqr = YOutlierFilter(method="iqr", threshold=1.5, reason="iqr_outlier")
    filter_zscore = YOutlierFilter(method="zscore", threshold=3.0, reason="zscore_outlier")

    # Get training data
    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d", include_augmented=False)
    y = dataset.y(selector, include_augmented=False)
    sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

    # Create report generator
    report_gen = FilteringReportGenerator(dataset)

    # Generate report for multiple filters
    report = report_gen.create_report(
        filters=[filter_iqr, filter_zscore],
        X=X,
        y=y,
        sample_indices=sample_indices,
        mode="any",  # Exclude if ANY filter flags
        partition="train",
        cascade_to_augmented=False,  # No augmented samples in this example
        dry_run=True,  # Don't actually mark samples
    )

    # Print the report
    report.print_report(verbose=2)

    # Get JSON export
    print("\nJSON Export (truncated):")
    json_report = report.to_json()
    print(json_report[:500] + "...")

    # Compare filters
    print("\n--- Filter Comparison ---")
    comparison = report_gen.compare_filters(
        filters=[filter_iqr, filter_zscore],
        X=X,
        y=y,
    )
    print(f"Total samples: {comparison['n_samples']}")
    print("Individual filter results:")
    for name, stats in comparison["individual"].items():
        print(f"  {name}: {stats['n_excluded']} excluded ({stats['exclusion_rate']:.1%})")
    if comparison["overlap"]:
        print("Overlap analysis:")
        print(f"  Excluded by all filters: {comparison['overlap']['excluded_by_all']}")
        print(f"  Excluded by any filter: {comparison['overlap']['excluded_by_any']}")
        print(f"  Unique exclusions: {comparison['unique_exclusions']}")


def scenario_10_charts_with_excluded():
    """Scenario 10: Using include_excluded parameter in charts."""
    print("\n" + "=" * 70)
    print("SCENARIO 10: Charts with include_excluded Parameter")
    print("=" * 70)
    print("""
Existing chart controllers (chart_2d, chart_3d, chart_y) now support
include_excluded and highlight_excluded parameters:

- include_excluded: Include excluded samples in the visualization
- highlight_excluded: Show excluded samples with distinct style (red/dashed)

This allows visualizing the effect of filtering on your data.
""")

    # Use a real dataset
    dataset_config = DatasetConfigs("sample_data/regression_2")

    # Pipeline demonstrating chart options with excluded samples
    pipeline = [
        # Before filtering
        "chart_y",

        # Apply filter
        {
            "sample_filter": {
                "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
                "report": True,
            }
        },

        # After filtering - default behavior (excluded not shown)
        "chart_y",

        # After filtering - show excluded with highlighting
        {"chart_y": {"include_excluded": True, "highlight_excluded": True}},

        # 2D spectra chart with excluded samples highlighted
        {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},

        "snv",
        {"split": KFold(n_splits=3)},
        {"model": PLSRegression(n_components=5)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="charts_with_excluded")
    runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)

    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print("\nCharts with include_excluded example completed!")
    except Exception as e:
        print(f"Note: Charts example requires proper dataset setup: {e}")


def main():
    """Run all scenarios."""
    print("=" * 70)
    print("NIRS4ALL - Sample Filtering Examples")
    print("=" * 70)
    print("""
This example demonstrates sample filtering capabilities for removing
outliers and poor-quality samples from training datasets.

Sample filtering is NON-DESTRUCTIVE - samples are marked as excluded
but the underlying data remains intact.
""")

    scenario_1_iqr_filtering()
    scenario_2_zscore_filtering()
    scenario_3_percentile_filtering()
    scenario_4_mad_filtering()
    scenario_5_pipeline_integration()
    scenario_6_composite_filters()
    scenario_7_exclusion_tracking()
    scenario_8_exclusion_chart()
    scenario_9_filtering_report()
    scenario_10_charts_with_excluded()
    scenario_11_edge_cases()
    scenario_12_x_outlier_filtering()
    scenario_13_spectral_quality()
    scenario_14_best_practices()

    print("\n" + "=" * 70)
    print("All scenarios completed!")
    print("=" * 70)


def scenario_11_edge_cases():
    """Scenario 11: Edge case handling."""
    print("\n" + "=" * 70)
    print("SCENARIO 11: Edge Case Handling")
    print("=" * 70)
    print("""
Sample filtering handles edge cases gracefully:
- Empty datasets
- Single sample datasets
- All samples excluded
- NaN/Inf values in data
""")

    # Edge case 1: Small dataset
    print("\n--- Edge Case 1: Small Dataset ---")
    np.random.seed(42)
    X_small = np.random.rand(3, 100)
    y_small = np.array([50, 52, 200])  # One obvious outlier

    filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
    filter_obj.fit(X_small, y_small)
    mask = filter_obj.get_mask(X_small, y_small)
    print(f"Small dataset (3 samples): Kept {mask.sum()}, Excluded {(~mask).sum()}")

    # Edge case 2: Single sample
    print("\n--- Edge Case 2: Single Sample ---")
    X_single = np.random.rand(1, 100)
    y_single = np.array([50])

    filter_single = YOutlierFilter(method="iqr", threshold=1.5)
    filter_single.fit(X_single, y_single)
    mask_single = filter_single.get_mask(X_single, y_single)
    print(f"Single sample: Kept {mask_single.sum()} (expected: 1)")

    # Edge case 3: Constant y values
    print("\n--- Edge Case 3: Constant Y Values ---")
    X_const = np.random.rand(10, 100)
    y_const = np.ones(10) * 50  # All same value

    filter_const = YOutlierFilter(method="zscore", threshold=3.0)
    filter_const.fit(X_const, y_const)
    mask_const = filter_const.get_mask(X_const, y_const)
    print(f"Constant y values (10 samples): Kept {mask_const.sum()}")

    # Edge case 4: Data with NaN
    print("\n--- Edge Case 4: Data with NaN Values ---")
    X_nan = np.random.rand(10, 100)
    y_nan = np.array([50, 52, 48, np.nan, 51, 49, 150, 47, np.nan, 53])

    filter_nan = YOutlierFilter(method="iqr", threshold=1.5)
    filter_nan.fit(X_nan, y_nan)
    mask_nan = filter_nan.get_mask(X_nan, y_nan)
    print(f"Data with 2 NaN values (10 samples): Kept {mask_nan.sum()}")
    print(f"  NaN samples excluded: {(~mask_nan)[[3, 8]].all()}")

    print("\nEdge cases handled gracefully!")


def scenario_12_x_outlier_filtering():
    """Scenario 12: X-based outlier filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 12: X-Based Outlier Filtering")
    print("=" * 70)
    print("""
X-based filters detect outliers in the feature (spectral) space:
- Mahalanobis distance: Distance from center in feature space
- PCA residual: Q-statistic from PCA reconstruction error
- PCA leverage: Hotelling's T² in reduced space
""")

    from nirs4all.operators.filters import XOutlierFilter

    np.random.seed(42)

    # Create normal spectra
    n_samples = 50
    n_features = 200
    X_normal = np.random.rand(n_samples, n_features)

    # Add some spectral outliers (shifted baseline, noise)
    X_outliers = np.zeros((5, n_features))
    X_outliers[0] = np.random.rand(n_features) + 2.0  # Shifted baseline
    X_outliers[1] = np.random.rand(n_features) * 3.0  # High amplitude
    X_outliers[2] = np.random.randn(n_features)  # Random noise (can be negative)
    X_outliers[3] = np.random.rand(n_features)
    X_outliers[3][50:150] = 0  # Flat region
    X_outliers[4] = np.random.rand(n_features) + np.sin(np.linspace(0, 10, n_features))

    X = np.vstack([X_normal, X_outliers])

    # Test different X-based methods
    methods = ["mahalanobis", "pca_residual", "pca_leverage"]

    print(f"\nTotal samples: {len(X)} ({n_samples} normal + 5 outliers)")

    for method in methods:
        print(f"\n--- Method: {method} ---")

        filter_x = XOutlierFilter(method=method, n_components=10)
        filter_x.fit(X)
        mask = filter_x.get_mask(X)

        stats = filter_x.get_filter_stats(X)
        print(f"Excluded: {stats['n_excluded']} samples ({stats['exclusion_rate']:.1%})")

        # Check if outliers were detected
        outlier_detected = (~mask[-5:]).sum()
        print(f"Synthetic outliers detected: {outlier_detected}/5")


def scenario_13_spectral_quality():
    """Scenario 13: Spectral quality filtering."""
    print("\n" + "=" * 70)
    print("SCENARIO 13: Spectral Quality Filtering")
    print("=" * 70)
    print("""
SpectralQualityFilter checks for data quality issues:
- NaN ratio: Too many missing values
- Zero ratio: Too many zero values (flat spectra)
- Variance: Very low variance (constant spectra)
- Value range: Values outside expected range (saturation)
""")

    from nirs4all.operators.filters import SpectralQualityFilter

    np.random.seed(42)

    # Create normal spectra
    n_features = 200
    X_good = np.random.rand(40, n_features) * 2  # Values in [0, 2]

    # Create problematic spectra
    X_bad = []

    # High NaN ratio
    x_nan = np.random.rand(n_features) * 2
    x_nan[:40] = np.nan  # 20% NaN
    X_bad.append(x_nan)

    # High zero ratio
    x_zero = np.random.rand(n_features) * 2
    x_zero[50:150] = 0  # 50% zeros
    X_bad.append(x_zero)

    # Very low variance (almost flat)
    X_bad.append(np.ones(n_features) * 1.5 + np.random.rand(n_features) * 1e-9)

    # Saturated spectrum (values > 4)
    X_bad.append(np.random.rand(n_features) * 2 + 3.5)

    # Values below minimum
    X_bad.append(np.random.rand(n_features) * 2 - 1.5)

    X = np.vstack([X_good, X_bad])
    print(f"\nTotal samples: {len(X)} ({len(X_good)} good + {len(X_bad)} problematic)")

    # Create quality filter with specific thresholds
    filter_quality = SpectralQualityFilter(
        max_nan_ratio=0.15,
        max_zero_ratio=0.4,
        min_variance=1e-6,
        max_value=4.0,
        min_value=-0.5,
    )

    mask = filter_quality.get_mask(X)
    stats = filter_quality.get_filter_stats(X)

    print("\nResults:")
    print(f"Samples kept: {stats['n_kept']}")
    print(f"Samples excluded: {stats['n_excluded']}")
    print("\nFailure breakdown:")
    for check, count in stats['failure_counts'].items():
        if count > 0:
            print(f"  - {check}: {count} samples")

    # Get detailed breakdown
    breakdown = filter_quality.get_quality_breakdown(X)
    print("\nPer-check pass rates:")
    for check, passes in breakdown.items():
        if check != "passes_all":
            print(f"  - {check}: {passes.sum()}/{len(X)} passed")


def scenario_14_best_practices():
    """Scenario 14: Best practices demonstration."""
    print("\n" + "=" * 70)
    print("SCENARIO 14: Best Practices")
    print("=" * 70)
    print("""
Best practices for sample filtering:
1. Start with conservative thresholds
2. Use dry runs to preview filtering effects
3. Combine multiple filter types
4. Document exclusion reasons
5. Consider filter order in pipeline
""")

    from nirs4all.operators.filters import (
        YOutlierFilter, XOutlierFilter, SpectralQualityFilter
    )
    from nirs4all.operators.filters.base import CompositeFilter

    dataset, n_normal, n_outliers = create_dataset_with_outliers()

    selector = {"partition": "train"}
    X = dataset.x(selector, layout="2d")
    y = dataset.y(selector)
    sample_indices = dataset._indexer.x_indices(selector, include_augmented=False)

    # Best Practice 1: Start conservative
    print("\n--- Best Practice 1: Start Conservative ---")
    filter_lenient = YOutlierFilter(method="iqr", threshold=3.0)  # Very lenient
    filter_standard = YOutlierFilter(method="iqr", threshold=1.5)  # Standard
    filter_strict = YOutlierFilter(method="iqr", threshold=1.0)   # Strict

    for label, f in [("Lenient (3.0)", filter_lenient),
                     ("Standard (1.5)", filter_standard),
                     ("Strict (1.0)", filter_strict)]:
        f.fit(X, y)
        stats = f.get_filter_stats(X, y)
        print(f"  {label}: {stats['n_excluded']} excluded ({stats['exclusion_rate']:.1%})")

    # Best Practice 2: Use dry runs
    print("\n--- Best Practice 2: Dry Run Preview ---")
    report_gen = FilteringReportGenerator(dataset)
    dry_run_report = report_gen.create_report(
        filters=[YOutlierFilter(method="iqr", threshold=1.5)],
        X=X,
        y=y,
        sample_indices=sample_indices,
        dry_run=True,  # Don't actually exclude
    )
    print(f"  Dry run: Would exclude {dry_run_report.n_final_excluded} samples")
    print(f"  Dataset unchanged: {dataset._indexer.get_exclusion_summary()['total_excluded']} excluded")

    # Best Practice 3: Combine multiple filter types
    print("\n--- Best Practice 3: Combined Filtering ---")
    combined = CompositeFilter(
        filters=[
            YOutlierFilter(method="iqr", threshold=1.5, reason="y_iqr"),
            YOutlierFilter(method="mad", threshold=3.5, reason="y_mad"),
        ],
        mode="any"
    )
    combined.fit(X, y)
    combined_stats = combined.get_filter_stats(X, y)
    print(f"  Combined (any mode): {combined_stats['n_excluded']} excluded")
    print("  Per-filter breakdown:")
    for fs in combined_stats['filter_breakdown']:
        print(f"    - {fs['reason']}: {fs['n_excluded']} excluded")

    # Best Practice 4: Clear documentation
    print("\n--- Best Practice 4: Document Exclusions ---")
    # Apply filter and mark with clear reason
    filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
    filter_obj.fit(X, y)
    mask = filter_obj.get_mask(X, y)
    exclude_idx = sample_indices[~mask].tolist()

    dataset._indexer.mark_excluded(
        exclude_idx,
        reason="y_outlier_iqr_1.5_phase5_demo"  # Clear, searchable reason
    )

    summary = dataset._indexer.get_exclusion_summary()
    print(f"  Documented exclusions: {summary['by_reason']}")

    # Clean up
    dataset._indexer.reset_exclusions()

    print("\nBest practices demonstration completed!")


if __name__ == "__main__":
    main()
