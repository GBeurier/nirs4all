#!/usr/bin/env python3
"""
Workspace Integration Quick Test - Simplified Version
=====================================================

This is a simplified version of the full integration test that runs faster
for quick validation of the workspace architecture.

Uses: 1 model, 2 datasets, 1-fold CV
"""

import os
import sys
from pathlib import Path
import shutil

# Set UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.operators.transformations import MultiplicativeScatterCorrection
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.workspace import WorkspaceManager, LibraryManager

# Disable emojis
os.environ['DISABLE_EMOJIS'] = '1'

print("="*80)
print("WORKSPACE QUICK INTEGRATION TEST")
print("="*80)
print()

# Initialize workspace
print("[STEP 1] Initialize Workspace")
workspace_path = Path("workspace_quick_test")
if workspace_path.exists():
    shutil.rmtree(workspace_path)

# Note: Workspace initialization is now automatic in PipelineRunner
# We just need to ensure the path exists for validation
print(f"  [OK] Workspace path: {workspace_path}")
print()

# Build pipeline
print("[STEP 2] Build Pipeline")
example_dir = Path(__file__).parent
pipeline = [
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"feature_augmentation": [MultiplicativeScatterCorrection]},
    ShuffleSplit(n_splits=1, random_state=42),
    {"y_processing": MinMaxScaler},
    {"model": PLSRegression(10), "name": "PLS_10"},
]
# Build pipeline
print("[STEP 2] Build Pipeline")
example_dir = Path(__file__).parent
pipeline = [
    MinMaxScaler(feature_range=(0.1, 0.8)),
    {"feature_augmentation": [MultiplicativeScatterCorrection]},
    ShuffleSplit(n_splits=1, random_state=42),
    {"y_processing": MinMaxScaler},
    {"model": PLSRegression(10), "name": "PLS_10"},
]
pipeline_config = PipelineConfigs(pipeline, name="quick_test_pipeline")
print("  [OK] Pipeline configured")
print()

# Configure datasets
print("[STEP 3] Configure Datasets")
data_paths = [
    str(example_dir / 'sample_data' / 'regression'),
    str(example_dir / 'sample_data' / 'regression_2'),
]
dataset_config = DatasetConfigs(data_paths)
print(f"  [OK] {len(data_paths)} datasets")
print()

# Run pipeline
print("[STEP 4] Run Pipeline")
runner = PipelineRunner(workspace_path=workspace_path, save_files=True, verbose=0, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)
print(f"  [OK] {predictions.num_predictions} predictions generated")
print()

# Analyze results
print("[STEP 5] Analyze Results")
top_models = predictions.top(n=2, rank_metric='rmse', rank_partition="val", display_partition="test")
print(f"  [OK] Top 2 models extracted")
best_model = top_models[0]
print(f"      Best: {best_model['model_name']} (test_rmse: {best_model.get('test_rmse', 'N/A')})")
print()

# Validate file structure
print("[STEP 6] Validate Files")
# Check in workspace runs directory
runs_dir = workspace_path / "runs"
if not runs_dir.exists():
    print("  [FAIL] workspace/runs/ not found")
    sys.exit(1)

# Find all pipeline directories in runs
import time
recent_time = time.time() - 300  # Last 5 minutes
pipeline_dirs = []
manifest_dirs = []

for run_dir in runs_dir.glob("*/"):
    if run_dir.is_dir():
        # Check ManifestManager structure: pipelines/UID/
        pipelines_mgr_dir = run_dir / "pipelines"
        if pipelines_mgr_dir.exists():
            for uid_dir in pipelines_mgr_dir.glob("*/"):
                if uid_dir.is_dir() and uid_dir.stat().st_mtime > recent_time:
                    manifest_dirs.append(uid_dir)

        # Check SimulationSaver structure: dataset_name/ (not pipelines/datasets/artifacts)
        for dataset_dir in run_dir.glob("*/"):
            if dataset_dir.is_dir() and dataset_dir.name not in ['pipelines', 'datasets', 'artifacts']:
                if dataset_dir.stat().st_mtime > recent_time:
                    pipeline_dirs.append(dataset_dir)

print(f"  [OK] Found {len(manifest_dirs)} manifest directories (recent)")
print(f"  [OK] Found {len(pipeline_dirs)} dataset directories (recent)")

# Check ManifestManager files
manifest_file_counts = {'manifest.yaml': 0}
for mdir in manifest_dirs:
    if (mdir / 'manifest.yaml').exists():
        manifest_file_counts['manifest.yaml'] += 1

# Check SimulationSaver files
saver_file_counts = {'predictions.json': 0}
for pdir in pipeline_dirs:
    if (pdir / 'predictions.json').exists():
        saver_file_counts['predictions.json'] += 1

print(f"  [OK] ManifestManager files: {manifest_file_counts}")
print(f"  [OK] SimulationSaver files: {saver_file_counts}")
print()

# Test Predictions API
print("[STEP 7] Test Predictions API")
filtered = predictions.filter_predictions(partition="test")
datasets = predictions.get_datasets()
models = predictions.get_models()
print(f"  [OK] filter_predictions: {len(filtered)} results")
print(f"  [OK] get_datasets: {datasets}")
print(f"  [OK] get_models: {models}")
print()

# Test catalog
print("[STEP 8] Test Catalog")
catalog_dir = workspace_path / "catalog"

# Archive one prediction - use manifest directory if available
pred_archive = Predictions()
source_dir = manifest_dirs[0] if manifest_dirs else None

if source_dir and source_dir.exists():
    metrics = {
        "dataset_name": best_model['dataset_name'],
        "test_score": best_model.get('test_rmse', 0.0),
        "model_type": best_model['model_name']
    }
    pred_id = pred_archive.archive_to_catalog(catalog_dir, source_dir, metrics)
    print(f"  [OK] Archived: {pred_id[:8]}...")

    # Test catalog queries
    catalog_pred = Predictions.load_from_parquet(catalog_dir)
    best_catalog = catalog_pred.query_best(metric="test_score", n=1, ascending=True)
    print(f"  [OK] load_from_parquet: {catalog_pred._df.height} predictions")
    print(f"  [OK] query_best: {best_catalog.height} results")
else:
    print("  [SKIP] No manifest directory found")
print()

# Test library
print("[STEP 9] Test Library")
library = LibraryManager(workspace_path / "library")

template_config = {
    "preprocessing": [{"name": "MinMaxScaler"}],
    "model": {"name": "PLSRegression", "n_components": 10}
}
template_path = library.save_template(template_config, "quick_template", "Quick test template")
templates = library.list_templates()
print(f"  [OK] save_template: {template_path.name}")
print(f"  [OK] list_templates: {len(templates)} templates")
print()

# Test model prediction
print("[STEP 10] Test Model Prediction")
predictor = PipelineRunner(save_files=False, verbose=0)
prediction_dataset = DatasetConfigs([str(example_dir / 'sample_data' / 'regression' / 'Xval.csv.gz')])
try:
    new_predictions, pred_objects = predictor.predict(
        best_model,
        prediction_dataset,
        all_predictions=True,
        verbose=0
    )
    print(f"  [OK] predict: shape {new_predictions.shape}")
except Exception as e:
    print(f"  [SKIP] predict error: {e}")
    new_predictions = None
print()

# Final summary
print("="*80)
print("VALIDATION SUMMARY")
print("="*80)

tests = {
    "Workspace initialization": workspace_path.exists(),
    "Run creation": (workspace_path / "runs").exists() and len(list((workspace_path / "runs").iterdir())) > 0,
    "Pipeline execution": predictions.num_predictions > 0,
    "File structure": len(manifest_dirs) > 0 and len(pipeline_dirs) > 0,
    "Predictions API": len(filtered) > 0,
    "Catalog archiving": catalog_dir.exists(),
    "Library management": len(templates) > 0,
    "Model prediction": new_predictions is not None and (isinstance(new_predictions, type(None)) or new_predictions.shape[0] > 0),
}

all_passed = all(tests.values())
for test_name, passed in tests.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {test_name}")

print()
if all_passed:
    print("[SUCCESS] All tests passed!")
    print(f"\nWorkspace: {workspace_path}")
    print(f"Predictions: {predictions.num_predictions}")
    print(f"Pipeline dirs: {len(pipeline_dirs)}")
    sys.exit(0)
else:
    print("[ERROR] Some tests failed")
    sys.exit(1)
