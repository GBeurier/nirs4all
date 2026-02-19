"""
Generate Documentation Images
=============================

This script generates sample images for ALL visualization types in nirs4all
and saves them to docs/source/assets/ for inclusion in RTD documentation.

Generated images:
1. In-Pipeline Charts: chart_2d, chart_3d, spectral_distribution, fold_chart, y_chart, augment_chart, exclusion_chart
2. PredictionAnalyzer: plot_top_k, plot_confusion_matrix, plot_histogram, plot_heatmap, plot_candlestick
3. ShapAnalyzer: plot_spectral_importance, plot_beeswarm_binned, plot_waterfall_binned
4. PipelineDiagram: pipeline structure visualization with branching

Output: docs/source/assets/
"""

import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.filters import XOutlierFilter
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    Gaussian,
    GaussianAdditiveNoise,
    LinearBaselineDrift,
    MultiplicativeScatterCorrection,
    Rotate_Translate,
    SavitzkyGolay,
    SecondDerivative,
    StandardNormalVariate,
    WavelengthShift,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.pipeline_diagram import PipelineDiagram
from nirs4all.visualization.predictions import PredictionAnalyzer

# Paths
SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
ASSETS_DIR = REPO_ROOT / "docs" / "source" / "assets"
SAMPLE_DATA = EXAMPLES_DIR / "sample_data"
WORKSPACE_DIR = EXAMPLES_DIR / "workspace" / "doc_images"

# Use regression_2 which has proper wavelength headers
DATA_WITH_WAVELENGTHS = SAMPLE_DATA / "regression_2"

# Ensure directories exist
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, name: str, dpi: int = 150):
    """Save figure to assets directory."""
    output_path = ASSETS_DIR / f"{name}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")

def copy_chart_from_artifacts(artifacts_dir: Path, chart_keyword: str, dest_name: str) -> bool:
    """Copy chart from artifacts directory to assets."""
    if not artifacts_dir.exists():
        return False

    for run_dir in artifacts_dir.iterdir():
        if run_dir.is_dir():
            for img_file in run_dir.glob("*.png"):
                if chart_keyword.lower() in img_file.stem.lower():
                    dest_path = ASSETS_DIR / f"{dest_name}.png"
                    shutil.copy(img_file, dest_path)
                    print(f"  ✓ Copied: {dest_name}.png")
                    return True
    return False

# =============================================================================
# SECTION 1: In-Pipeline Charts - Basic (with proper wavelengths)
# =============================================================================
def generate_basic_charts():
    """Generate basic chart_2d, chart_3d with proper wavelength axes."""
    print("\n" + "=" * 60)
    print("GENERATING BASIC SPECTRA CHARTS (with proper wavelengths)")
    print("=" * 60)

    # Simple pipeline with raw spectra charts
    pipeline = [
        "chart_2d",
        "chart_3d",
        "spectral_distribution",
        "y_chart",
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        "fold_chart",
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "basic"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "BasicCharts"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Copy charts from artifacts
    artifacts_dir = WORKSPACE_DIR / "basic" / "runs" / "regression_2"
    if artifacts_dir.exists():
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                for img_file in run_dir.glob("*.png"):
                    name = img_file.stem.lower()
                    if "2d_chart" in name or "2d chart" in name.lower():
                        shutil.copy(img_file, ASSETS_DIR / "chart_2d.png")
                        print("  ✓ Copied: chart_2d.png")
                    elif "3d_chart" in name or "3d chart" in name.lower():
                        shutil.copy(img_file, ASSETS_DIR / "chart_3d.png")
                        print("  ✓ Copied: chart_3d.png")
                    elif "spectral_distribution" in name:
                        shutil.copy(img_file, ASSETS_DIR / "spectral_distribution.png")
                        print("  ✓ Copied: spectral_distribution.png")
                    elif "y_distribution" in name and "fold" not in name:
                        shutil.copy(img_file, ASSETS_DIR / "y_chart.png")
                        print("  ✓ Copied: y_chart.png")
                    elif "fold" in name:
                        shutil.copy(img_file, ASSETS_DIR / "fold_chart.png")
                        print("  ✓ Copied: fold_chart.png")
                break

    return predictions

# =============================================================================
# SECTION 2: Multiple Preprocessing Chart (using feature_augmentation)
# =============================================================================
def generate_multiple_preprocessing_chart():
    """Generate chart_2d with multiple preprocessing views using feature_augmentation."""
    print("\n" + "=" * 60)
    print("GENERATING MULTIPLE PREPROCESSING CHART (feature_augmentation)")
    print("=" * 60)

    # Use feature_augmentation to create multiple preprocessing views
    # This generates a single chart_2d with multiple subplots (one per preprocessing)
    pipeline = [
        # Feature augmentation creates multiple preprocessing "views"
        {"feature_augmentation": [
            StandardNormalVariate,
            FirstDerivative,
            SavitzkyGolay,
        ], "action": "extend"},

        # chart_2d will show all processings as separate subplots
        "chart_2d",

        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "multi_preproc"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "MultiPreprocessing"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Copy the chart
    artifacts_dir = WORKSPACE_DIR / "multi_preproc" / "runs" / "regression_2"
    if artifacts_dir.exists():
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and not run_dir.name.startswith("_"):
                for img_file in run_dir.glob("*.png"):
                    if "2d" in img_file.stem.lower():
                        shutil.copy(img_file, ASSETS_DIR / "chart_2d_preprocessed.png")
                        print("  ✓ Copied: chart_2d_preprocessed.png")
                        break
                break

# =============================================================================
# SECTION 3: Augmentation Charts (single and multiple)
# =============================================================================
def generate_augmentation_charts():
    """Generate augmentation charts with VISIBLE differences using higher intensity transforms."""
    print("\n" + "=" * 60)
    print("GENERATING AUGMENTATION CHARTS (with visible differences)")
    print("=" * 60)

    # Import additional augmenters for more visible effects

    # Single augmentation with MORE VISIBLE transform (Rotate_Translate + higher noise)
    pipeline_single = [
        StandardNormalVariate(),
        {"sample_augmentation": {
            "transformers": [
                Rotate_Translate(p_range=5, y_factor=8),  # More visible rotation/translation
            ],
            "count": 3,
        }},
        "augment_chart",
        "augment_details_chart",
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "augment_single"),
    )

    runner.run(
        PipelineConfigs(pipeline_single, "SingleAugment"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Copy single augmentation charts
    artifacts_dir = WORKSPACE_DIR / "augment_single" / "runs" / "regression_2"
    if artifacts_dir.exists():
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and not run_dir.name.startswith("_"):
                for img_file in run_dir.glob("*.png"):
                    name = img_file.stem.lower()
                    if "details" in name:
                        shutil.copy(img_file, ASSETS_DIR / "augment_details_chart.png")
                        print("  ✓ Copied: augment_details_chart.png")
                    elif "augmentation" in name:
                        shutil.copy(img_file, ASSETS_DIR / "augment_chart.png")
                        print("  ✓ Copied: augment_chart.png")
                break

    # Multiple augmentations with VISIBLE differences
    pipeline_multi = [
        StandardNormalVariate(),
        {"sample_augmentation": {
            "transformers": [
                Rotate_Translate(p_range=5, y_factor=8),       # Visible rotation + translation
                LinearBaselineDrift(offset_range=(-0.1, 0.1), slope_range=(-0.005, 0.005)),  # Visible baseline drift
                GaussianAdditiveNoise(sigma=0.02),              # More visible noise
            ],
            "count": 2,
        }},
        "augment_chart",
        "augment_details_chart",
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]

    runner2 = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "augment_multi"),
    )

    runner2.run(
        PipelineConfigs(pipeline_multi, "MultiAugment"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Copy multi augmentation charts
    artifacts_dir = WORKSPACE_DIR / "augment_multi" / "runs" / "regression_2"
    if artifacts_dir.exists():
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and not run_dir.name.startswith("_"):
                for img_file in run_dir.glob("*.png"):
                    name = img_file.stem.lower()
                    if "details" in name:
                        shutil.copy(img_file, ASSETS_DIR / "augment_multi_details_chart.png")
                        print("  ✓ Copied: augment_multi_details_chart.png")
                    elif "augmentation" in name:
                        shutil.copy(img_file, ASSETS_DIR / "augment_multi_chart.png")
                        print("  ✓ Copied: augment_multi_chart.png")
                break

# =============================================================================
# SECTION 4: Exclusion Charts (with chart_2d style)
# =============================================================================
def generate_exclusion_charts():
    """Generate exclusion chart showing excluded samples with MORE VISIBLE exclusions."""
    print("\n" + "=" * 60)
    print("GENERATING EXCLUSION CHARTS (with visible exclusions)")
    print("=" * 60)

    # Use higher contamination (20%) to make exclusions more visible
    pipeline = [
        StandardNormalVariate(),
        "chart_2d",  # Before exclusion
        {"sample_filter": {
            "filters": [XOutlierFilter(method='isolation_forest', contamination=0.2)],  # 20% exclusion
        }},
        "exclusion_chart",
        "chart_2d",  # After exclusion - shows filtered data
        ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "exclusion"),
    )

    runner.run(
        PipelineConfigs(pipeline, "ExclusionDemo"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Copy charts - we need to get the right order
    # First chart_2d is before exclusion, second is after
    artifacts_dir = WORKSPACE_DIR / "exclusion" / "runs" / "regression_2"
    if artifacts_dir.exists():
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and not run_dir.name.startswith("_"):
                chart_2d_files = []
                exclusion_file = None

                for img_file in sorted(run_dir.glob("*.png")):
                    name = img_file.stem.lower()
                    if "exclusion" in name:
                        exclusion_file = img_file
                    elif "2d" in name:
                        chart_2d_files.append(img_file)

                # Copy exclusion chart
                if exclusion_file:
                    shutil.copy(exclusion_file, ASSETS_DIR / "exclusion_chart.png")
                    print("  ✓ Copied: exclusion_chart.png")

                # Copy the second chart_2d (after exclusion) for the "with exclusion" view
                if len(chart_2d_files) >= 2:
                    shutil.copy(chart_2d_files[1], ASSETS_DIR / "chart_2d_with_exclusion.png")
                    print("  ✓ Copied: chart_2d_with_exclusion.png")
                elif chart_2d_files:
                    shutil.copy(chart_2d_files[0], ASSETS_DIR / "chart_2d_with_exclusion.png")
                    print("  ✓ Copied: chart_2d_with_exclusion.png (single)")
                break

# =============================================================================
# SECTION 5: Pipeline Diagram with Branching
# =============================================================================
def generate_pipeline_diagram():
    """Generate pipeline diagram with branching structure."""
    print("\n" + "=" * 60)
    print("GENERATING PIPELINE DIAGRAM (with branching)")
    print("=" * 60)

    # Complex pipeline with branching
    pipeline = [
        StandardNormalVariate(),
        {"branch": [
            [FirstDerivative(), PLSRegression(n_components=10)],
            [SavitzkyGolay(), PLSRegression(n_components=15)],
        ]},
        {"merge": "predictions"},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"model": RandomForestRegressor(n_estimators=10, random_state=42)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "diagram"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "BranchingPipeline"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Create diagram from trace
    print("  Generating pipeline_diagram...")
    try:
        trace = runner.last_execution_trace
        if trace:
            diagram = PipelineDiagram.from_trace(trace)
            fig = diagram.render(show_shapes=True, title="Pipeline with Branching")
            if fig:
                save_figure(fig, "pipeline_diagram", dpi=200)
        else:
            print("    ⚠ No execution trace available")
    except Exception as e:
        print(f"    ⚠ Skipped pipeline_diagram: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# SECTION 6: PredictionAnalyzer Charts
# =============================================================================
def generate_prediction_analyzer_charts():
    """Generate PredictionAnalyzer chart samples."""
    print("\n" + "=" * 60)
    print("GENERATING PREDICTION ANALYZER CHARTS")
    print("=" * 60)

    # Regression pipeline
    pipeline = [
        StandardNormalVariate(),
        FirstDerivative(),
        ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "pred_analyzer"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "PredAnalyzer"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    analyzer = PredictionAnalyzer(predictions)

    # plot_top_k
    print("  Generating plot_top_k...")
    try:
        fig = analyzer.plot_top_k(k=3, rank_metric='rmse', rank_partition='val')
        if fig:
            save_figure(fig, "plot_top_k")
    except Exception as e:
        print(f"    ⚠ Skipped plot_top_k: {e}")

    # plot_histogram
    print("  Generating plot_histogram...")
    try:
        fig = analyzer.plot_histogram(display_metric='rmse', display_partition='test')
        if fig:
            save_figure(fig, "plot_histogram")
    except Exception as e:
        print(f"    ⚠ Skipped plot_histogram: {e}")

    print("  (heatmap.png and candlestick.png already exist in assets)")

    return predictions

def generate_classification_charts():
    """Generate classification-specific charts."""
    print("\n" + "=" * 60)
    print("GENERATING CLASSIFICATION CHARTS")
    print("=" * 60)

    pipeline = [
        StandardNormalVariate(),
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        {"model": RandomForestClassifier(n_estimators=50, random_state=42)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "classification"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "ClassificationCharts"),
        DatasetConfigs(str(SAMPLE_DATA / "classification")),
    )

    analyzer = PredictionAnalyzer(predictions)

    # plot_confusion_matrix
    print("  Generating plot_confusion_matrix...")
    try:
        fig = analyzer.plot_confusion_matrix(k=1, rank_partition='val')
        if fig:
            save_figure(fig, "plot_confusion_matrix")
    except Exception as e:
        print(f"    ⚠ Skipped plot_confusion_matrix: {e}")

# =============================================================================
# SECTION 7: SHAP Charts
# =============================================================================
def generate_shap_charts():
    """Generate SHAP visualization samples."""
    print("\n" + "=" * 60)
    print("GENERATING SHAP CHARTS")
    print("=" * 60)

    try:
        import shap
        print("  SHAP library available")
    except ImportError:
        print("  ⚠ SHAP not available, skipping SHAP charts")
        return

    # Train a model for SHAP
    pipeline = [
        StandardNormalVariate(),
        {"model": PLSRegression(n_components=10)},
    ]

    runner = PipelineRunner(
        verbose=0,
        plots_visible=False,
        save_artifacts=True,
        workspace_path=str(WORKSPACE_DIR / "shap"),
    )

    predictions, _ = runner.run(
        PipelineConfigs(pipeline, "ShapExample"),
        DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
    )

    # Get best prediction
    best_prediction = predictions.top(n=1, rank_metric='rmse', rank_partition="test")[0]

    # Run SHAP analysis
    print("  Running SHAP analysis...")
    shap_params = {
        'n_samples': 50,
        'explainer_type': 'auto',
        'visualizations': ['spectral', 'waterfall', 'beeswarm'],
        'bin_size': 10,
        'bin_stride': 5,
    }

    try:
        shap_results, output_dir = runner.explain(
            best_prediction,
            DatasetConfigs(str(DATA_WITH_WAVELENGTHS)),
            shap_params=shap_params,
            plots_visible=False
        )

        print(f"  SHAP output directory: {output_dir}")

        # Copy generated images
        if output_dir and Path(output_dir).exists():
            for img_file in Path(output_dir).glob("*.png"):
                name = img_file.stem.lower()
                if "spectral" in name:
                    shutil.copy(img_file, ASSETS_DIR / "shap_spectral.png")
                    print("  ✓ Copied: shap_spectral.png")
                elif "waterfall" in name:
                    shutil.copy(img_file, ASSETS_DIR / "shap_waterfall.png")
                    print("  ✓ Copied: shap_waterfall.png")
                elif "beeswarm" in name:
                    shutil.copy(img_file, ASSETS_DIR / "shap_beeswarm.png")
                    print("  ✓ Copied: shap_beeswarm.png")

    except Exception as e:
        print(f"  ⚠ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Generate all documentation images."""
    print("\n" + "#" * 60)
    print("# NIRS4ALL - Generate Documentation Images")
    print("#" * 60)
    print(f"\nOutput directory: {ASSETS_DIR}")
    print(f"Data source: {DATA_WITH_WAVELENGTHS}")

    # Generate all chart types
    generate_basic_charts()
    generate_multiple_preprocessing_chart()
    generate_augmentation_charts()
    generate_exclusion_charts()
    generate_pipeline_diagram()
    generate_prediction_analyzer_charts()
    generate_classification_charts()
    generate_shap_charts()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated images in: {ASSETS_DIR}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(ASSETS_DIR.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
