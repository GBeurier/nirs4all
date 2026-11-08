"""
U02 - Export Bundle: Model Deployment
===============================================

Export trained models to bundles for deployment, sharing, and archival.
Python-host .n4a payloads require a trusted producer and remain nonportable.

This tutorial covers:

* Export to .n4a bundle format
* Export to portable Python script (.n4a.py)
* Load and predict from bundles
* Bundle inspection and metadata

Prerequisites
-------------
Complete :ref:`U01_save_load_predict` first.

Next Steps
----------
See :ref:`U03_workspace_management` for session handling.

Duration: ~5 minutes
Difficulty: ★★☆☆☆
"""

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.dagml.host_container import open_n4a_source

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 Export Bundle Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Why Export Bundles?
# =============================================================================
print("\n" + "=" * 60)
print("U02 - Export Bundle: Model Deployment")
print("=" * 60)

print("""
Export bundles solve key deployment challenges:

  📦 .n4a BUNDLE FORMAT (Core V1 host container)
     - Self-contained prediction package
     - Includes preprocessing + model artifacts
     - Works with ALL model types

  🐍 .n4a.py PORTABLE SCRIPT
     - Single Python file
     - No nirs4all dependency needed
     - Only works with sklearn models

  ✨ USE CASES
     ✓ Production deployment (containers, lambda)
     ✓ Share models with colleagues
     ✓ Long-term archival
     ✓ Edge/embedded deployment
""")

# =============================================================================
# Section 2: Train a Pipeline for Export
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Train a Pipeline for Export")
print("-" * 60)

# Build pipeline with preprocessing and model
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [StandardNormalVariate(), SavitzkyGolay()]},

    RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),

    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "U22_Export")
dataset_config = DatasetConfigs("sample_data/regression")

# Train the pipeline
print("Training pipeline...")
runner = PipelineRunner(save_artifacts=True, verbose=0)
predictions, _run_info = runner.run(pipeline_config, dataset_config)

# Select the export candidate from CV validation only, excluding refit/final rows.
top_results = predictions.top(n=1, rank_partition="val", score_scope="cv")
assert isinstance(top_results, list)
best_prediction = top_results[0]
print(f"\nBest model selected by CV validation: {best_prediction['model_name']}")
# Display held-out performance after selection for assessment only.
test_mse = best_prediction.get('test_mse', best_prediction.get('mse'))
print(f"Held-out test MSE (assessment only): {test_mse:.4f}" if test_mse is not None else "Held-out test MSE: (see detailed metrics; assessment only)")
print(f"Pipeline UID: {best_prediction['pipeline_uid'][:16]}...")

# =============================================================================
# Section 3: Export to .n4a Bundle
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Export to .n4a Bundle")
print("-" * 60)

print("""
The .n4a Python-host profile uses Core V1 storage:
  - manifest.json: Core inventory (nirs4all.n4a.container.v1)
  - host/bundle_manifest.json: SDK bundle metadata
  - pipeline.json: Pipeline configuration
  - trace.json: Execution trace
  - artifacts/: Serialized models and transformers

Owner: nirs4all.python-host; required feature: nirs4all-python-host@1.
Storage is uncompressed zip-stored.v1; there is no compress option.
Core verifies storage; SDK semantic checks and verified private extraction
precede trusted joblib loading. Checksums do not make pickle safe.
Old SDK ZIP inputs require re-export; Python-host payloads are nonportable.
""")

# Create exports directory
exports_dir = Path("exports")
exports_dir.mkdir(exist_ok=True)

# Export to .n4a bundle
bundle_path = runner.export(
    source=dict(best_prediction),
    output_path="exports/wheat_model.n4a",
    format="n4a",
    include_metadata=True
)

print(f"✓ Bundle exported: {bundle_path}")
print(f"  Size: {bundle_path.stat().st_size / 1024:.1f} KB")

# =============================================================================
# Section 4: Export to Portable Python Script
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Export to Portable Python Script")
print("-" * 60)

print("""
The .n4a.py format is ideal for lightweight deployment:
  - Single Python file with embedded artifacts
  - Standalone predict() function
  - Only requires numpy and joblib

Note: Only works with sklearn-compatible models!
""")

# Export to portable script
script_path = runner.export(
    source=dict(best_prediction),
    output_path="exports/wheat_predictor.n4a.py",
    format="n4a.py",
    include_metadata=True
)

print(f"✓ Portable script exported: {script_path}")
print(f"  Size: {script_path.stat().st_size / 1024:.1f} KB")

# Show script preview
print("\nScript preview (first 15 lines):")
with open(script_path) as f:
    lines = f.readlines()[:15]
    for line in lines:
        print(f"  {line.rstrip()}")
print("  ...")

# =============================================================================
# Section 5: Predict from Bundle
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Predict from Bundle")
print("-" * 60)

print("""
Load a bundle and make predictions without the original workspace.
""")

# Create a new predictor (simulating deployment)
predictor = PipelineRunner(save_artifacts=False, verbose=0)

# Load new data
prediction_dataset = DatasetConfigs({
    'X_test': 'sample_data/regression/Xval.csv.gz'
})

# Predict from bundle
print(f"Loading bundle: {bundle_path}")
bundle_predictions_result, _preds1 = predictor.predict(
    prediction_obj=str(bundle_path),
    dataset=prediction_dataset,
    verbose=0
)
assert isinstance(bundle_predictions_result, np.ndarray)
bundle_predictions = bundle_predictions_result

print(f"✓ Predictions generated: {len(bundle_predictions)} samples")
print(f"  First 5: {bundle_predictions[:5].flatten()}")

# Verify against original
original_predictions_result, _preds2 = predictor.predict(
    prediction_obj=dict(best_prediction),
    dataset=prediction_dataset,
    verbose=0
)
assert isinstance(original_predictions_result, np.ndarray)
original_predictions = original_predictions_result

bundle_flat = np.asarray(bundle_predictions).flatten()
original_flat = np.asarray(original_predictions).flatten()

if np.allclose(bundle_flat, original_flat, rtol=1e-5):
    print("✓ Bundle predictions match original: YES")
else:
    diff = np.abs(bundle_flat - original_flat).max()
    print(f"✗ Max difference: {diff}")

# =============================================================================
# Section 6: Inspect Bundle Contents
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Inspect Bundle Contents")
print("-" * 60)

print("""
Inspect bundle contents and metadata programmatically.
""")

# One verified source serves inventory, SDK metadata and the borrowed loader.
with open_n4a_source(bundle_path) as source:
    container = source.require_host()
    print("Bundle contents (declared order):")
    for name in container.namelist():
        print(f"  {name}: {container.size(name) / 1024:.1f} KB")

    print("\nBundle metadata (host/bundle_manifest.json):")
    manifest = container.bundle_manifest
    print(f"  Format version: {manifest.get('bundle_format_version', 'N/A')}")
    print(f"  nirs4all version: {manifest.get('nirs4all_version', 'N/A')}")
    print(f"  Created: {manifest.get('created_at', 'N/A')}")

    # This loader borrows source; the outer context owns and closes it.
    print("\nProgrammatic inspection:")
    with BundleLoader(bundle_path, source=source) as loader:
        print(f"  Metadata: {loader.metadata}")

# =============================================================================
# Section 7: Batch Export Multiple Models
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Batch Export Multiple Models")
print("-" * 60)

print("""
Export multiple models to create a "model zoo" for comparison.
""")

# Train pipeline with multiple models
pipeline_multi = [
    MinMaxScaler(),
    StandardNormalVariate(),
    RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
    {"model": PLSRegression(n_components=5), "name": "PLS-5"},
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
    {"model": PLSRegression(n_components=15), "name": "PLS-15"},
]

pipeline_config_multi = PipelineConfigs(pipeline_multi, "U22_BatchExport")
runner_multi = PipelineRunner(save_artifacts=True, verbose=0)
predictions_multi, _run_info_multi = runner_multi.run(pipeline_config_multi, dataset_config)

# Export the top 3 CV validation selections; held-out scores do not rank them.
print("\nExporting top 3 models selected by CV validation:")
top_models = predictions_multi.top(3, rank_partition="val", score_scope="cv")
assert isinstance(top_models, list)

model_zoo = Path("exports/model_zoo")
model_zoo.mkdir(parents=True, exist_ok=True)

for i, pred in enumerate(top_models, 1):
    model_name = pred['model_name'].replace(" ", "_").lower()
    bundle_name = f"rank_{i}_{model_name}.n4a"
    bundle = runner_multi.export(dict(pred), model_zoo / bundle_name)
    print(f"  {i}. {pred['model_name']} -> {bundle.name}")
    print(f"     Held-out test RMSE (assessment only): {pred.get('rmse', 0):.4f}")

print(f"\n✓ Exported {len(top_models)} models to {model_zoo}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Export Bundle Workflow:

  1. TRAIN WITH ARTIFACTS:
     runner = PipelineRunner(save_artifacts=True)
     predictions, _ = runner.run(config, dataset)

  2. EXPORT TO .n4a BUNDLE:
     bundle = runner.export(
         source=best_prediction,
         output_path="model.n4a",
         format="n4a"
     )

  3. EXPORT TO PORTABLE SCRIPT:
     script = runner.export(
         source=best_prediction,
         output_path="model.n4a.py",
         format="n4a.py"
     )

  4. PREDICT FROM BUNDLE:
     predictor = PipelineRunner()
     preds, _ = predictor.predict(
         str(bundle_path),
         new_dataset
     )

  5. INSPECT BUNDLE:
     from nirs4all.pipeline.bundle import BundleLoader
     with BundleLoader(bundle_path) as loader:
         print(loader.metadata)

Format Comparison:
  .n4a:      All model types, requires nirs4all
  .n4a.py:   sklearn only, standalone (numpy + joblib)

Next: U03_workspace_management.py - Session and workspace handling
""")
