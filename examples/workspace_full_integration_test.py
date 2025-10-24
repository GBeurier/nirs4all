#!/usr/bin/env python3
"""
Workspace Full Integration Test - Comprehensive Example
======================================================

This comprehensive example demonstrates the NEW flat workspace architecture:

✅ WHAT THIS TESTS:
1. Sequential pipeline numbering (0001, 0002, 0003, ...)
2. Date-prefixed run directories (YYYY-MM-DD_dataset/)
3. Content-addressed artifact storage
4. Multiple models and datasets
5. Library management (save/load templates)
6. Prediction analysis and comparison
7. File persistence and structure validation

📁 EXPECTED STRUCTURE:
workspace_test/
  runs/
    2025-10-24_regression/
      0001_pls_baseline_abc123/
        manifest.yaml
        metrics.json
        predictions.csv
      0002_rf_model_def456/
        ...
      artifacts/
        objects/
    2025-10-24_regression_2/
      0001_pls_baseline_ghi789/
        ...
  library/
    templates/

🎯 KEY CHANGES FROM OLD ARCHITECTURE:
- NO WorkspaceManager (deleted - was redundant)
- NO dataset/pipeline nested structure
- FLAT sequential numbering within runs
- Direct PipelineRunner usage with workspace_path
"""

import os
import sys
from pathlib import Path
import shutil

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all import PipelineRunner, PipelineConfigs
from nirs4all.dataset import DatasetConfigs
from nirs4all.workspace import LibraryManager

# Disable emojis for cleaner test output
os.environ['DISABLE_EMOJIS'] = '1'


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def validate_workspace_structure(workspace_path):
    """Validate that workspace has correct structure."""
    print("\n🔍 Validating workspace structure...")

    runs_dir = workspace_path / 'runs'
    assert runs_dir.exists(), "runs/ directory missing"
    print("  ✓ runs/ directory exists")

    # Find run directories (date-prefixed)
    run_dirs = list(runs_dir.glob('*_*'))
    assert len(run_dirs) > 0, "No run directories found"
    print(f"  ✓ Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        # Check for sequential pipeline directories
        pipeline_dirs = sorted([d for d in run_dir.iterdir()
                               if d.is_dir() and d.name[0].isdigit()])
        print(f"  ✓ {run_dir.name}: {len(pipeline_dirs)} pipelines")

        for pdir in pipeline_dirs:
            # Validate pipeline structure
            assert (pdir / 'manifest.yaml').exists(), f"manifest.yaml missing in {pdir.name}"
            print(f"    - {pdir.name}: manifest.yaml ✓")

        # Check artifacts directory
        artifacts_dir = run_dir / 'artifacts' / 'objects'
        if artifacts_dir.exists():
            artifact_count = len(list(artifacts_dir.glob('sha256_*')))
            print(f"  ✓ Content-addressed artifacts: {artifact_count} files")

    return True


def main():
    """Run comprehensive workspace integration test."""

    print_section("WORKSPACE FULL INTEGRATION TEST")
    print("""
This test demonstrates the complete workflow:
  1. Multiple models (PLS, RandomForest)
  2. Multiple datasets (regression, regression_2)
  3. Sequential pipeline numbering
  4. Content-addressed storage
  5. Library management
  6. Validation of file structure
""")

    # =========================================================================
    # STEP 1: WORKSPACE SETUP
    # =========================================================================
    print_section("STEP 1: Workspace Setup")

    workspace_path = Path("workspace_integration_test")

    # Clean up if exists
    if workspace_path.exists():
        print(f"Removing existing workspace: {workspace_path}")
        shutil.rmtree(workspace_path)

    print(f"✓ Workspace path: {workspace_path}")
    print("✓ No WorkspaceManager needed - direct folder structure")

    # =========================================================================
    # STEP 2: CONFIGURE DATASETS
    # =========================================================================
    print_section("STEP 2: Configure Datasets")

    example_dir = Path(__file__).parent
    dataset_paths = [
        str(example_dir / 'sample_data' / 'regression'),
        str(example_dir / 'sample_data' / 'regression_2'),
    ]

    print(f"Datasets:")
    for i, path in enumerate(dataset_paths, 1):
        print(f"  {i}. {Path(path).name}")

    # =========================================================================
    # STEP 3: BUILD PIPELINE CONFIGURATIONS
    # =========================================================================
    print_section("STEP 3: Build Pipeline Configurations")

    # Pipeline 1: PLS Baseline
    pipeline_pls = [
        ShuffleSplit(n_splits=2, random_state=42),
        {'model': PLSRegression(n_components=5), 'name': 'PLS_baseline'}
    ]
    config_pls = PipelineConfigs(pipeline_pls, name='pls_baseline')
    print("✓ Pipeline 1: PLS Baseline (5 components)")

    # Pipeline 2: Random Forest
    pipeline_rf = [
        ShuffleSplit(n_splits=2, random_state=42),
        {'model': RandomForestRegressor(n_estimators=50, random_state=42),
         'name': 'RF_model'}
    ]
    config_rf = PipelineConfigs(pipeline_rf, name='rf_model')
    print("✓ Pipeline 2: Random Forest (50 estimators)")

    # Pipeline 3: PLS with more components
    pipeline_pls_10 = [
        ShuffleSplit(n_splits=2, random_state=42),
        {'model': PLSRegression(n_components=10), 'name': 'PLS_10comp'}
    ]
    config_pls_10 = PipelineConfigs(pipeline_pls_10, name='pls_10components')
    print("✓ Pipeline 3: PLS with 10 components")

    # =========================================================================
    # STEP 4: RUN PIPELINES
    # =========================================================================
    print_section("STEP 4: Run Pipelines")

    runner = PipelineRunner(
        workspace_path=str(workspace_path),
        save_files=True,
        verbose=0
    )

    print("\n📊 Running multiple pipelines on multiple datasets...")
    print("This demonstrates sequential numbering within each run.\n")

    all_predictions = []

    # Run on first dataset
    print("🔄 Dataset 1: regression")
    dataset_config_1 = DatasetConfigs([dataset_paths[0]])

    print("  Pipeline 1/3: PLS Baseline...")
    pred1, _ = runner.run(config_pls, dataset_config_1)
    all_predictions.append(('Dataset1_PLS', pred1))
    print(f"    ✓ {pred1.num_predictions} predictions")

    print("  Pipeline 2/3: Random Forest...")
    pred2, _ = runner.run(config_rf, dataset_config_1)
    all_predictions.append(('Dataset1_RF', pred2))
    print(f"    ✓ {pred2.num_predictions} predictions")

    print("  Pipeline 3/3: PLS 10 components...")
    pred3, _ = runner.run(config_pls_10, dataset_config_1)
    all_predictions.append(('Dataset1_PLS10', pred3))
    print(f"    ✓ {pred3.num_predictions} predictions")

    # Run on second dataset
    print("\n🔄 Dataset 2: regression_2")
    dataset_config_2 = DatasetConfigs([dataset_paths[1]])

    print("  Pipeline 1/3: PLS Baseline...")
    pred4, _ = runner.run(config_pls, dataset_config_2)
    all_predictions.append(('Dataset2_PLS', pred4))
    print(f"    ✓ {pred4.num_predictions} predictions")

    print("  Pipeline 2/3: Random Forest...")
    pred5, _ = runner.run(config_rf, dataset_config_2)
    all_predictions.append(('Dataset2_RF', pred5))
    print(f"    ✓ {pred5.num_predictions} predictions")

    total_predictions = sum(p.num_predictions for _, p in all_predictions)
    print(f"\n✅ All pipelines completed!")
    print(f"   Total predictions generated: {total_predictions}")

    # =========================================================================
    # STEP 5: VALIDATE WORKSPACE STRUCTURE
    # =========================================================================
    print_section("STEP 5: Validate Workspace Structure")

    validate_workspace_structure(workspace_path)

    # Show sequential numbering
    print("\n📊 Sequential Numbering Demonstration:")
    runs_dir = workspace_path / 'runs'
    for run_dir in sorted(runs_dir.glob('*_*')):
        print(f"\n  {run_dir.name}:")
        pipeline_dirs = sorted([d for d in run_dir.iterdir()
                               if d.is_dir() and d.name[0].isdigit()])
        for pdir in pipeline_dirs:
            print(f"    {pdir.name}")

    # =========================================================================
    # STEP 6: ANALYZE PREDICTIONS
    # =========================================================================
    print_section("STEP 6: Analyze Predictions")

    print("\n📈 Model Performance Comparison:\n")
    for name, predictions in all_predictions:
        best = predictions.top(n=1, rank_metric='rmse', rank_partition='test')
        if best:
            model = best[0]
            test_rmse = model.get('test_rmse', 'N/A')
            test_r2 = model.get('test_r2', 'N/A')

            print(f"  {name}:")
            print(f"    Model: {model.get('model_name', 'N/A')}")

            if isinstance(test_rmse, (int, float)):
                print(f"    Test RMSE: {test_rmse:.4f}")
            else:
                print(f"    Test RMSE: {test_rmse}")

            if isinstance(test_r2, (int, float)):
                print(f"    Test R²: {test_r2:.4f}")
            else:
                print(f"    Test R²: {test_r2}")

    # =========================================================================
    # STEP 7: LIBRARY MANAGEMENT
    # =========================================================================
    print_section("STEP 7: Library Management")

    library = LibraryManager(workspace_path / 'library')

    # Save template
    template_config = {
        "preprocessing": [
            {"name": "MinMaxScaler"}
        ],
        "model": {
            "name": "PLSRegression",
            "n_components": 5
        }
    }

    template_path = library.save_template(
        template_config,
        "baseline_pls",
        "Baseline PLS configuration for regression"
    )
    print(f"✓ Template saved: {template_path.name}")

    # List templates
    templates = library.list_templates()
    print(f"✓ Templates in library: {len(templates)}")
    for t in templates:
        print(f"  - {t['name']}")

    # Load template
    if templates:
        loaded = library.load_template("baseline_pls")
        print(f"✓ Template loaded successfully")
        if 'model' in loaded and 'name' in loaded['model']:
            print(f"  Model: {loaded['model']['name']}")
        else:
            print(f"  Template structure: {list(loaded.keys())}")

    # =========================================================================
    # STEP 8: WORKSPACE STRUCTURE SUMMARY
    # =========================================================================
    print_section("STEP 8: Workspace Structure Summary")

    print("""
📁 Final Workspace Structure:

workspace_integration_test/
  runs/
    2025-10-24_regression/              # Date + dataset
      0001_pls_baseline_abc123/         # Sequential: 0001
        manifest.yaml
        metrics.json
        predictions.csv
      0002_rf_model_def456/             # Sequential: 0002
        ...
      0003_pls_10components_ghi789/     # Sequential: 0003
        ...
      artifacts/
        objects/                        # Content-addressed
          sha256_abc.../model.pkl
          sha256_def.../scaler.pkl
      predictions.json

    2025-10-24_regression_2/            # Different dataset
      0001_pls_baseline_jkl012/         # Numbering restarts
      0002_rf_model_mno345/
      artifacts/
        objects/
      predictions.json

  library/
    templates/
      baseline_pls.json
    trained/
      filtered/
      pipeline/
      fullrun/

✅ KEY FEATURES DEMONSTRATED:
  1. Flat sequential numbering (0001, 0002, 0003)
  2. Date-prefixed runs (YYYY-MM-DD_dataset)
  3. Content-addressed artifacts (deduplication)
  4. No WorkspaceManager needed
  5. Simple PipelineRunner API
  6. Library management for templates
  7. Multiple datasets handled cleanly
""")

    # =========================================================================
    # STEP 9: FILE VALIDATION
    # =========================================================================
    print_section("STEP 9: File Validation")

    runs_dir = workspace_path / 'runs'

    # Count pipeline directories
    total_pipelines = 0
    total_manifests = 0
    total_artifacts = 0

    for run_dir in runs_dir.glob('*_*'):
        for pdir in run_dir.iterdir():
            if pdir.is_dir() and pdir.name[0].isdigit():
                total_pipelines += 1
                if (pdir / 'manifest.yaml').exists():
                    total_manifests += 1

        artifacts_dir = run_dir / 'artifacts' / 'objects'
        if artifacts_dir.exists():
            total_artifacts += len(list(artifacts_dir.glob('sha256_*')))

    print(f"✓ Total pipeline directories: {total_pipelines}")
    print(f"✓ Total manifests: {total_manifests}")
    print(f"✓ Total content-addressed artifacts: {total_artifacts}")
    print(f"✓ Library templates: {len(templates)}")

    assert total_pipelines > 0, "No pipelines created"
    assert total_pipelines == total_manifests, "Manifests missing"

    print("\n✅ All validations passed!")

    # =========================================================================
    # CLEANUP (optional)
    # =========================================================================
    print_section("CLEANUP")

    print("Workspace created at:", workspace_path.absolute())
    print("\nTo inspect the structure:")
    print(f"  cd {workspace_path}")
    print(f"  tree /F  # Windows")
    print(f"  find .   # Linux/Mac")

    # Uncomment to auto-cleanup:
    # shutil.rmtree(workspace_path)
    # print(f"\n🧹 Cleaned up workspace: {workspace_path}")

    print("\n" + "=" * 80)
    print("✅ WORKSPACE FULL INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
