"""
Generate All In-Pipeline Charts
===============================

This example generates samples of ALL in-pipeline chart types available in nirs4all.
It is designed to showcase the visualization capabilities for documentation purposes.

Charts generated:
1. Spectra charts (chart_2d, chart_3d)
2. Spectral distribution (spectral_distribution)
3. Fold charts (fold_chart)
4. Target charts (y_chart)
5. Augmentation charts (augment_chart, augment_details_chart)
6. Exclusion charts (exclusion_chart)

Output: All charts are saved to workspace/examples_output/all_charts/
"""

# Standard library imports
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# NIRS4All imports
import nirs4all
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    FirstDerivative,
    Detrend,
    GaussianAdditiveNoise,
    WavelengthShift,
)
from nirs4all.operators.filters import XOutlierFilter

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "workspace" / "examples_output" / "all_charts"
REGRESSION_DATA = Path(__file__).parent.parent / "sample_data" / "regression"
CLASSIFICATION_DATA = Path(__file__).parent.parent / "sample_data" / "classification"


def run_regression_charts():
    """Generate charts for regression task."""
    print("\n" + "=" * 60)
    print("REGRESSION CHARTS")
    print("=" * 60)

    # Pipeline with all chart types for regression
    pipeline = [
        # ============================================
        # SECTION 1: Raw Data Visualization
        # ============================================
        "chart_2d",                          # 2D spectra (raw)
        "chart_3d",                          # 3D spectra (raw)
        "spectral_distribution",             # Spectral envelope (raw)
        "y_chart",                           # Y distribution (before split)

        # ============================================
        # SECTION 2: Preprocessing
        # ============================================
        StandardNormalVariate(),
        FirstDerivative(),
        "chart_2d",                          # 2D spectra (after preprocessing)
        "spectral_distribution",             # Envelope (after preprocessing)

        # ============================================
        # SECTION 3: Outlier Exclusion
        # ============================================
        {"sample_filter": {
            "filters": [XOutlierFilter(method='isolation_forest', contamination=0.05)],
        }},
        {"exclusion_chart": {                # Exclusion visualization
            "color_by": "status",
            "n_components": 2,
        }},

        # ============================================
        # SECTION 4: Augmentation
        # ============================================
        {"sample_augmentation": {
            "transformers": [
                GaussianAdditiveNoise(sigma=0.005),
                WavelengthShift(),
            ],
            "count": 2,
        }},
        "augment_chart",                     # Overlay of augmented samples
        "augment_details_chart",             # Detailed grid view

        # ============================================
        # SECTION 5: Cross-Validation
        # ============================================
        ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
        "fold_chart",                        # Fold distribution (y-colored)
        "y_chart",                           # Y histogram per fold
        "spectral_distribution",             # Envelope per fold

        # ============================================
        # SECTION 6: Model
        # ============================================
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=1,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(OUTPUT_DIR / "regression"),
    )

    predictions, per_dataset = runner.run(
        PipelineConfigs(pipeline, "AllChartsRegression"),
        DatasetConfigs(str(REGRESSION_DATA)),
    )

    print(f"Regression charts saved to: {OUTPUT_DIR / 'regression'}")
    return predictions


def run_classification_charts():
    """Generate charts for classification task."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION CHARTS")
    print("=" * 60)

    # Pipeline with charts for classification
    pipeline = [
        # Raw data
        "chart_2d",
        "y_chart",
        "spectral_distribution",

        # Preprocessing
        StandardNormalVariate(),
        Detrend(),
        "chart_2d",

        # Cross-validation (stratified for classification)
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        "fold_chart",                        # Discrete colors for classes
        "y_chart",                           # Class distribution per fold

        # Model
        {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
    ]

    runner = PipelineRunner(
        verbose=1,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(OUTPUT_DIR / "classification"),
    )

    predictions, per_dataset = runner.run(
        PipelineConfigs(pipeline, "AllChartsClassification"),
        DatasetConfigs(str(CLASSIFICATION_DATA)),
    )

    print(f"Classification charts saved to: {OUTPUT_DIR / 'classification'}")
    return predictions


def run_chart_options_demo():
    """Demonstrate chart options (dict syntax)."""
    print("\n" + "=" * 60)
    print("CHART OPTIONS DEMO")
    print("=" * 60)

    pipeline = [
        # Y chart with different layouts
        {"y_chart": {"layout": "standard"}},

        # Preprocessing
        StandardNormalVariate(),

        # 2D chart with exclusion highlighting
        {"chart_2d": {
            "include_excluded": False,
        }},

        # Exclusion with different color modes
        {"sample_filter": {
            "filters": [XOutlierFilter(method='isolation_forest', contamination=0.1)],
        }},
        {"exclusion_chart": {"color_by": "y", "n_components": 2}},

        # Cross-validation
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {"y_chart": {"layout": "stacked"}},
        {"y_chart": {"layout": "staggered"}},

        # Model
        {"model": PLSRegression(n_components=8)},
    ]

    runner = PipelineRunner(
        verbose=1,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(OUTPUT_DIR / "options_demo"),
    )

    predictions, per_dataset = runner.run(
        PipelineConfigs(pipeline, "ChartOptionsDemo"),
        DatasetConfigs(str(REGRESSION_DATA)),
    )

    print(f"Options demo charts saved to: {OUTPUT_DIR / 'options_demo'}")
    return predictions


def main():
    """Run all chart generation examples."""
    print("\n" + "#" * 60)
    print("# NIRS4ALL - Generate All In-Pipeline Charts")
    print("#" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all chart types
    run_regression_charts()
    run_classification_charts()
    run_chart_options_demo()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
    print("\nGenerated chart types:")
    print("  - chart_2d: 2D spectra visualization")
    print("  - chart_3d: 3D spectra visualization")
    print("  - spectral_distribution: Envelope plots (min/max/mean/IQR)")
    print("  - fold_chart: CV fold distribution")
    print("  - y_chart: Target value histograms")
    print("  - augment_chart: Augmentation overlay")
    print("  - augment_details_chart: Augmentation details grid")
    print("  - exclusion_chart: Excluded samples PCA scatter")


if __name__ == "__main__":
    main()
