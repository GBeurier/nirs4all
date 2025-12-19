#!/usr/bin/env python3
"""
Q32 Example - Export Trained Pipelines as Standalone Bundles
=============================================================
Demonstrates exporting trained pipelines to standalone prediction bundles
(.n4a) that can be used for deployment, sharing, or archival.

Key Features:
1. Export to .n4a bundle format (ZIP with artifacts)
2. Export to portable Python script (.n4a.py) for deployment
3. Load and predict from bundles
4. Bundle inspection and metadata

Phase 6 Implementation:
    This example showcases the bundle export feature that enables
    creating self-contained prediction packages without requiring
    the original workspace or full nirs4all installation.

Use Cases:
    - Production deployment: Lightweight container/lambda deployment
    - Sharing: Send a model to a colleague
    - Archival: Reproducible predictions years later
    - Edge deployment: Embedded systems, mobile

Note on Model Compatibility:
    - .n4a bundle format: Works with ALL model types (sklearn, PyTorch, JAX, TensorFlow)
    - .n4a.py portable script: Only works with sklearn-compatible models
      Deep learning models (PyTorch, JAX, TensorFlow) require the full .n4a format
      because they need their framework installed to reconstruct the model.
"""

# Standard library imports
import argparse
import os
import sys
import tempfile
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    Gaussian,
)

# Simple status symbols
CHECK = "[OK]"
CROSS = "[X]"
ROCKET = ">"
PACKAGE = "[PKG]"
SPARKLE = "*"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q32 Export Bundle Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"{ROCKET} {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# Example 1: Train a Pipeline and Export to .n4a Bundle
# =============================================================================


def example_1_train_and_export():
    """
    Demonstrate training a pipeline and exporting to .n4a bundle.

    The .n4a format is a ZIP archive containing:
    - manifest.json: Bundle metadata and version info
    - pipeline.json: Minimal pipeline configuration
    - trace.json: Execution trace for deterministic replay
    - artifacts/: Directory with serialized model and transformer artifacts
    - fold_weights.json: CV fold weights (if applicable)
    """
    print_section("Example 1: Train Pipeline and Export to .n4a Bundle")

    # Build pipeline with preprocessing and model
    pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay()]},
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=10), "name": "PLS_10"},
    ]

    # Create configuration objects
    pipeline_config = PipelineConfigs(pipeline, "export_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    # Train the pipeline
    print_subsection("Training Pipeline")
    runner = PipelineRunner(save_artifacts=True, verbose=0)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Get best prediction for export
    best_prediction = predictions.top(n=1, rank_partition="test")[0]
    print(f"Best model: {best_prediction['model_name']}")
    print(f"Test RMSE: {best_prediction['rmse']:.4f}")
    print(f"Pipeline UID: {best_prediction['pipeline_uid'][:16]}...")

    # Export to .n4a bundle
    print_subsection("Exporting to .n4a Bundle")

    # Create exports directory if it doesn't exist
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)

    bundle_path = runner.export(
        source=best_prediction,
        output_path="exports/wheat_pls_model.n4a",
        format="n4a",
        include_metadata=True,
        compress=True
    )

    print(f"{CHECK} Bundle exported to: {bundle_path}")
    print(f"   Size: {bundle_path.stat().st_size / 1024:.1f} KB")

    return best_prediction, bundle_path


# =============================================================================
# Example 2: Export to Portable Python Script
# =============================================================================


def example_2_export_portable_script(best_prediction):
    """
    Demonstrate exporting to a portable Python script.

    The .n4a.py format is a single Python file with:
    - Embedded artifacts (base64 encoded)
    - Standalone predict() function
    - No nirs4all dependency (only numpy, joblib required)

    This is ideal for:
    - Lightweight deployment
    - Sharing with users who don't have nirs4all
    - Edge/embedded systems
    """
    print_section("Example 2: Export to Portable Python Script")

    runner = PipelineRunner()

    # Export to portable script
    script_path = runner.export(
        source=best_prediction,
        output_path="exports/wheat_predictor.n4a.py",
        format="n4a.py",
        include_metadata=True
    )

    print(f"{CHECK} Portable script exported to: {script_path}")
    print(f"   Size: {script_path.stat().st_size / 1024:.1f} KB")

    # Show first few lines of the script
    print_subsection("Script Preview")
    with open(script_path, 'r') as f:
        lines = f.readlines()[:25]
        for line in lines:
            print(f"   {line.rstrip()}")
    print("   ...")

    return script_path


# =============================================================================
# Example 3: Load and Predict from Bundle
# =============================================================================


def example_3_predict_from_bundle(bundle_path, best_prediction):
    """
    Demonstrate loading a bundle and making predictions.

    Bundles can be loaded and used for prediction without
    needing the original workspace or training run.
    """
    print_section("Example 3: Load and Predict from Bundle")

    # Create a new runner (simulating a deployment scenario)
    predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

    # Load new data for prediction
    prediction_dataset = DatasetConfigs({
        'X_test': 'sample_data/regression/Xval.csv.gz'
    })

    # Predict from bundle
    print_subsection("Predicting from Bundle")
    print(f"Loading bundle: {bundle_path}")

    bundle_predictions, _ = predictor.predict(
        prediction_obj=str(bundle_path),
        dataset=prediction_dataset,
        verbose=0
    )

    print(f"{CHECK} Predictions generated: {len(bundle_predictions)} samples")

    # Compare with original predictions
    print_subsection("Comparing with Original Model")

    original_predictions, _ = predictor.predict(
        prediction_obj=best_prediction,
        dataset=prediction_dataset,
        verbose=0
    )

    # Verify predictions match
    if np.allclose(bundle_predictions, original_predictions, rtol=1e-5):
        print(f"{CHECK} Bundle predictions match original: YES")
    else:
        print(f"{CROSS} Bundle predictions match original: NO")
        diff = np.abs(bundle_predictions - original_predictions).max()
        print(f"   Max difference: {diff}")

    print(f"\nFirst 5 predictions (bundle): {bundle_predictions[:5].flatten()}")
    print(f"First 5 predictions (original): {original_predictions[:5].flatten()}")

    return bundle_predictions


# =============================================================================
# Example 4: Bundle Inspection
# =============================================================================


def example_4_inspect_bundle(bundle_path):
    """
    Demonstrate inspecting bundle contents and metadata.

    The BundleLoader class provides methods to inspect bundle contents
    without fully loading all artifacts.
    """
    print_section("Example 4: Bundle Inspection")

    import zipfile
    import json

    print_subsection("Bundle Contents")
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        print("Files in bundle:")
        for name in zf.namelist():
            info = zf.getinfo(name)
            size_kb = info.file_size / 1024
            print(f"   {name}: {size_kb:.1f} KB")

    print_subsection("Bundle Metadata")
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        manifest = json.loads(zf.read('manifest.json'))
        print(f"Format version: {manifest.get('bundle_format_version', 'unknown')}")
        print(f"nirs4all version: {manifest.get('nirs4all_version', 'unknown')}")
        print(f"Created: {manifest.get('created_at', 'unknown')}")
        print(f"Pipeline UID: {manifest.get('pipeline_uid', 'unknown')}")
        print(f"Model step: {manifest.get('model_step_index', 'unknown')}")
        print(f"Preprocessing: {manifest.get('preprocessing_chain', 'unknown')}")

    # Use BundleLoader for programmatic access
    print_subsection("Programmatic Inspection")
    from nirs4all.pipeline.bundle import BundleLoader

    loader = BundleLoader(bundle_path)
    print(f"Loader: {loader}")
    print(f"Metadata: {loader.metadata}")
    print(f"Fold weights: {loader.fold_weights}")

    return loader


# =============================================================================
# Example 5: Export from Different Sources
# =============================================================================


def example_5_export_from_different_sources():
    """
    Demonstrate exporting from different prediction sources.

    The export() method accepts various sources:
    - prediction dict: From Predictions object
    - folder path: Path to pipeline directory
    - Run object: Best prediction from a Run
    """
    print_section("Example 5: Export from Different Sources")

    # First, train a simple pipeline
    pipeline = [
        MinMaxScaler(),
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=5), "name": "simple_pls"},
    ]

    pipeline_config = PipelineConfigs(pipeline, "source_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    best_pred = predictions.top(n=1, rank_partition="test")[0]

    # Method 1: Export from prediction dict
    print_subsection("Method 1: Export from Prediction Dict")
    bundle1 = runner.export(best_pred, "exports/from_dict.n4a")
    print(f"{CHECK} Exported: {bundle1}")

    # Method 2: Export from folder path
    print_subsection("Method 2: Export from Folder Path")
    run_dir = runner.current_run_dir
    if run_dir:
        pipeline_uid = best_pred.get('pipeline_uid', '')
        folder_path = run_dir / pipeline_uid
        if folder_path.exists():
            bundle2 = runner.export(str(folder_path), "exports/from_folder.n4a")
            print(f"{CHECK} Exported: {bundle2}")
        else:
            print(f"Pipeline folder not found: {folder_path}")
    else:
        print("Run directory not available")

    print_subsection("Export Summary")
    print(f"Both methods produce equivalent bundles that can be used")
    print(f"for prediction without the original workspace.")


# =============================================================================
# Example 6: Batch Export Multiple Models
# =============================================================================


def example_6_batch_export():
    """
    Demonstrate batch exporting multiple models from a run.

    Useful for archiving all top-performing models or creating
    a model zoo for deployment.
    """
    print_section("Example 6: Batch Export Multiple Models")

    # Train pipeline with multiple models
    pipeline = [
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        StandardNormalVariate(),
        RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
        {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        {"model": PLSRegression(n_components=10), "name": "PLS_10"},
        {"model": PLSRegression(n_components=15), "name": "PLS_15"},
    ]

    pipeline_config = PipelineConfigs(pipeline, "batch_export_demo")
    dataset_config = DatasetConfigs(['sample_data/regression'])

    runner = PipelineRunner(save_artifacts=True, verbose=0)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Export top 3 models
    print_subsection("Exporting Top 3 Models")
    top_models = predictions.top(n=3, rank_partition="test")

    exports_dir = Path("exports/model_zoo")
    exports_dir.mkdir(parents=True, exist_ok=True)

    for i, pred in enumerate(top_models, 1):
        model_name = pred['model_name'].replace(" ", "_").lower()
        bundle_name = f"rank_{i}_{model_name}.n4a"
        bundle_path = runner.export(
            pred,
            exports_dir / bundle_name,
            format="n4a"
        )
        print(f"   {i}. {pred['model_name']} -> {bundle_path.name}")
        print(f"      RMSE: {pred['rmse']:.4f}")

    print(f"\n{CHECK} Exported {len(top_models)} models to {exports_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print(f"# {PACKAGE} Q32 - Export Trained Pipelines as Standalone Bundles")
    print("#" * 70)

    # Example 1: Train and export to .n4a
    best_prediction, bundle_path = example_1_train_and_export()

    # Example 2: Export to portable script
    script_path = example_2_export_portable_script(best_prediction)

    # Example 3: Predict from bundle
    bundle_predictions = example_3_predict_from_bundle(bundle_path, best_prediction)

    # Example 4: Inspect bundle
    loader = example_4_inspect_bundle(bundle_path)

    # Example 5: Export from different sources
    example_5_export_from_different_sources()

    # Example 6: Batch export
    example_6_batch_export()

    print("\n" + "=" * 70)
    print(f"{SPARKLE} All examples completed successfully!")
    print("=" * 70)
    print("\nBundle files created in ./exports/")
    print("These can be shared and used for prediction without the original workspace.")
