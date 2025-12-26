"""
Workspace Export, Library, and Global Predictions Example

This example demonstrates the simplified nirs4all API for workspace management:
1. Running pipelines and generating predictions
2. Exporting best results with ONE CALL: runner.export_best_for_dataset()
3. Saving to library with automatic n_features extraction
4. Using global predictions database (workspace/dataset_name.json)
5. Browsing and querying all predictions across runs

Key nirs4all APIs demonstrated:
- PipelineRunner.export_best_for_dataset(): ONE CALL exports best results!
- LibraryManager: save_template(), save_filtered(), save_pipeline_full() with n_features
- Predictions.load_from_file_cls(): Global predictions database access
- Clean filenames: No redundant date/time prefixes
"""

import argparse
import shutil
from pathlib import Path
import json
from datetime import datetime

from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.workspace.library_manager import LibraryManager

# Simple status symbols
DISK = "[D]"
TROPHY = "[1]"
SEARCH = "[?]"
ROCKET = ">"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q14 Workspace Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

def main():
    print("=" * 80)
    print("  WORKSPACE EXPORT, LIBRARY & GLOBAL PREDICTIONS EXAMPLE")
    print("=" * 80)

    # Clean up previous example
    workspace_path = Path("workspace_export_example")
    if workspace_path.exists():
        print(f"Removing existing workspace: {workspace_path}")
        shutil.rmtree(workspace_path)

    # ============================================================================
    # STEP 1: Run Multiple Pipelines
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 1: Running Multiple Pipelines")
    print("=" * 80)

    # Get demo dataset path (using sample_data from examples)
    dataset_path = Path(__file__).parent / 'sample_data' / 'regression'
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        print("This example requires the sample_data directory.")
        print("Skipping pipeline execution...")
        return

    # Configure pipelines
    pipeline1 = [
        {"class": "sklearn.model_selection._split.ShuffleSplit", "params": {"n_splits": 2, "random_state": 42}},
        {"model": {"class": "sklearn.cross_decomposition._pls.PLSRegression", "params": {"n_components": 5}}, "name": "PLS_baseline"}
    ]

    pipeline2 = [
        {"class": "sklearn.model_selection._split.ShuffleSplit", "params": {"n_splits": 2, "random_state": 42}},
        {"model": {"class": "sklearn.cross_decomposition._pls.PLSRegression", "params": {"n_components": 10}}, "name": "PLS_optimized"}
    ]

    # Create configs
    pipelines_config1 = PipelineConfigs(pipeline1, name='pls_baseline')
    pipelines_config2 = PipelineConfigs(pipeline2, name='pls_optimized')
    dataset_config = DatasetConfigs([str(dataset_path)])

    # Run pipelines
    runner = PipelineRunner(workspace_path=workspace_path, verbose=0)
    run_predictions1, datasets_predictions1 = runner.run(pipelines_config1, dataset_config)
    run_predictions2, datasets_predictions2 = runner.run(pipelines_config2, dataset_config)

    total_predictions = run_predictions1.num_predictions + run_predictions2.num_predictions
    print(f"\n✅ Completed! Generated {total_predictions} predictions")

    # ============================================================================
    # STEP 2: Explore Workspace Structure
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 2: Workspace Structure")
    print("=" * 80)

    print("\n📁 Current workspace structure:")
    print(f"\n{workspace_path}/")

    # Show runs directory
    runs_dir = workspace_path / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            print(f"  📅 {run_dir.name}/")

            # Show pipelines
            pipeline_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
            for pipeline_dir in sorted(pipeline_dirs):
                print(f"      🔢 {pipeline_dir.name}/")

                # Show files in pipeline
                files = list(pipeline_dir.glob("*.json")) + list(pipeline_dir.glob("*.csv"))
                for f in sorted(files)[:3]:  # Show first 3 files
                    print(f"          📄 {f.name}")

            # Show _binaries
            binaries_dir = run_dir / "_binaries"
            if binaries_dir.exists():
                print(f"      {DISK}_binaries/")
                binary_files = list(binaries_dir.iterdir())
                for bf in sorted(binary_files)[:3]:  # Show first 3
                    print(f"          📦 {bf.name}")

            # Show best predictions in run root
            best_preds = list(run_dir.glob("*Best_prediction*.csv"))
            if best_preds:
                print(f"      {TROPHY}Best predictions (run root):")
                for bp in sorted(best_preds):
                    print(f"          📊 {bp.name}")

    # ============================================================================
    # STEP 3: Global Predictions Database
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 3: Global Predictions Database")
    print("=" * 80)

    print("\n📊 Global predictions are stored at workspace root:")
    print(f"    {workspace_path}/regression.json")

    # Load global predictions
    global_predictions_path = workspace_path / "regression.json"
    global_preds = None  # Initialize variable
    if global_predictions_path.exists():
        global_preds = Predictions.load_from_file_cls(global_predictions_path)
        print(f"\n✓ Loaded {global_preds.num_predictions} predictions from database")

        # Show best prediction
        if global_preds.num_predictions > 0:
            best = global_preds.get_best(ascending=True)  # Regression: lower is better
            print(f"\n{TROPHY}Best prediction overall:")
            print(f"    Model: {best['model_name']}")
            print(f"    Config: {best['config_name']}")
            print(f"    Val Score: {best.get('val_score', 'N/A')}")
            print(f"    Test Score: {best.get('test_score', 'N/A')}")
            print(f"    ID: {best['id']}")

    # ============================================================================
    # STEP 4: Export Best Results Using nirs4all API (ONE CALL!)
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 4: Export Best Results (Simple One-Call API)")
    print("=" * 80)

    # Export best results for the dataset with ONE function call
    export_dir = runner.export_best_for_dataset("regression", mode="predictions")

    if export_dir:
        print(f"\n✅ Exported best results to: {export_dir}")
        print(f"\n📁 Exported files:")
        for f in sorted(export_dir.iterdir()):
            if not f.name.startswith('_'):
                print(f"  📄 {f.name}")
    else:
        print("\n⚠️  No results to export")

    # ============================================================================
    # STEP 5: Library Management Using nirs4all API
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 5: Library Management Using nirs4all API")
    print("=" * 80)

    # Initialize LibraryManager
    library_dir = workspace_path / "library"
    library = LibraryManager(library_dir)

    print(f"\n📚 Initialized LibraryManager at: {library_dir}")

    # Save pipeline as template (config only, no binaries)
    template_config = {
        "steps": [
            {"class": "sklearn.model_selection._split.ShuffleSplit", "params": {"n_splits": 2, "random_state": 42}},
            {"model": {"class": "sklearn.cross_decomposition._pls.PLSRegression", "params": {"n_components": 5}}, "name": "PLS_baseline"}
        ]
    }

    template_path = library.save_template(
        pipeline_config=template_config,
        name="pls_baseline_template",
        description="Standard PLS regression with 5 components - reusable template"
    )
    print(f"\n✅ Saved template using LibraryManager: {template_path.name}")

    if global_preds and global_preds.num_predictions > 0:
        best = global_preds.get_best(ascending=True)

        # Find the source pipeline and run directories
        config_name = best['config_name']
        source_pipeline_dir = None
        source_run_dir = None

        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                for pipeline_dir in run_dir.iterdir():
                    if pipeline_dir.is_dir() and config_name in pipeline_dir.name and not pipeline_dir.name.startswith('_'):
                        source_pipeline_dir = pipeline_dir
                        source_run_dir = run_dir
                        break
                if source_pipeline_dir:
                    break

        if source_pipeline_dir and source_run_dir:
            # Save filtered pipeline (config + metrics only)
            filtered_path = library.save_filtered(
                pipeline_dir=source_pipeline_dir,
                name=f"best_{best['model_name']}_filtered",
                description=f"Best {best['model_name']} configuration and metrics (Score: {best.get('test_score', 'N/A'):.4f})"
            )
            print(f"✅ Saved filtered pipeline: {filtered_path.name}")

            # Save full pipeline (config + binaries for deployment)
            full_path = library.save_pipeline_full(
                run_dir=source_run_dir,
                pipeline_dir=source_pipeline_dir,
                name=f"best_{best['model_name']}_full",
                description=f"Best {best['model_name']} - complete trained model for deployment"
            )
            print(f"✅ Saved full pipeline: {full_path.name}")

            # Save entire run (for complete experiment archiving)
            fullrun_path = library.save_fullrun(
                run_dir=source_run_dir,
                name=f"complete_experiment_{runs_dir.name}",
                description="Complete experimental run with all pipelines and results"
            )
            print(f"✅ Saved full run: {fullrun_path.name}")

    # Display library contents using LibraryManager API
    print(f"\n📚 Library contents (with n_features metadata):")

    templates = library.list_templates()
    print(f"\n📄 Templates ({len(templates)}):")
    for template in templates:
        print(f"  • {template['name']}: {template.get('description', 'No description')}")

    filtered = library.list_filtered()
    print(f"\n{SEARCH}Filtered pipelines ({len(filtered)}):")
    for filt in filtered:
        n_features = filt.get('n_features', 'N/A')
        print(f"  • {filt['name']}: {filt.get('description', 'No description')}")
        print(f"      n_features: {n_features}")

    pipelines = library.list_pipelines()
    print(f"\n{ROCKET}Full pipelines ({len(pipelines)}):")
    for pipeline in pipelines:
        n_features = pipeline.get('n_features', 'N/A')
        print(f"  • {pipeline['name']}: {pipeline.get('description', 'No description')}")
        print(f"      n_features: {n_features} (for compatibility checking)")

    fullruns = library.list_fullruns()
    print(f"\n📦 Full runs ({len(fullruns)}):")
    for fullrun in fullruns:
        print(f"  • {fullrun['name']}: {fullrun.get('description', 'No description')}")

    print(f"\n📁 Final library structure:")
    print(f"  {library_dir.name}/")
    print(f"    templates/        # Reusable pipeline configurations")
    print(f"    trained/")
    print(f"      filtered/       # Config + metrics only")
    print(f"      pipeline/       # Full trained models")
    print(f"      fullrun/        # Complete experiment archives")

    # ============================================================================
    # STEP 6: Query Global Predictions
    # ============================================================================
    print("\n" + "=" * 80)
    print("  STEP 6: Querying Global Predictions")
    print("=" * 80)

    if global_preds and global_preds.num_predictions > 0:
        dataset_name = "regression"  # We know this from our setup
        print(f"\n📊 All predictions for dataset '{dataset_name}':")

        for pred_dict in global_preds.to_dicts():
            print(f"  • {pred_dict['model_name']} ({pred_dict['config_name']})")
            val = pred_dict.get('val_score', 'N/A')
            test = pred_dict.get('test_score', 'N/A')
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            test_str = f"{test:.4f}" if isinstance(test, float) else str(test)
            print(f"      Val: {val_str} | Test: {test_str} | ID: {pred_dict['id']}")

        # Filter by model
        print(f"\n{SEARCH}Filtering predictions by model name:")
        for pred_dict in global_preds.to_dicts():
            if "PLS" in pred_dict['model_name']:
                val = pred_dict.get('val_score', 'N/A')
                test = pred_dict.get('test_score', 'N/A')
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                test_str = f"{test:.4f}" if isinstance(test, float) else str(test)
                print(f"  • {pred_dict['model_name']}: Val={val_str}, Test={test_str}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print(f"""
✅ Workspace structure created with nirs4all API:

{workspace_path}/
├── runs/                       # All experimental runs
│   └── <dataset>/             # Dataset folder (no date prefix)
│       ├── 0001_name_hash/    # Sequential numbered pipelines
│       │   ├── pipeline.json
│       │   ├── Report_best_<pipeline_id>_<model>_<pred_id>.csv
│       │   └── folds_*.csv
│       ├── _binaries/         # Shared binaries (created only when needed)
│       └── best_<pipeline_folder>.csv  # Best prediction (replaced on better score)
│
├── exports/                    # Best results per dataset (ONE CALL!)
│   └── <dataset_name>/        # runner.export_best_for_dataset()
│       ├── <model>_predictions.csv
│       ├── <model>_pipeline.json
│       ├── <model>_summary.json
│       └── <model>_*.png  # Charts
│
├── library/                    # Managed by LibraryManager
│   ├── templates/             # Pipeline configs (save_template)
│   └── trained/               # Managed trained models
│       ├── filtered/          # Config + metrics (save_filtered)
│       │   └── <name>/
│       │       ├── pipeline.json
│       │       └── library_metadata.json  # includes n_features!
│       ├── pipeline/          # Full models (save_pipeline_full)
│       │   └── <name>/
│       │       ├── pipeline.json
│       │       ├── library_metadata.json  # includes n_features!
│       │       └── _binaries/
│       └── fullrun/           # Complete experiments (save_fullrun)
│
└── <dataset_name>.json        # Global predictions database (at workspace root)

Key nirs4all API Features Demonstrated:
• runner.export_best_for_dataset(): ONE CALL to export best results!
• LibraryManager: Automatically extracts and stores n_features in metadata
• Dataset-centric runs: Simple folder structure without date prefixes
• Best prediction replacement: Only one best_*.csv per dataset (replaced on better score)
""")

    print(f"\n📁 Workspace created at: {workspace_path.absolute()}")
    print("\n🔧 Simple API Usage:")
    print("  # Export best results (predictions, config, charts)")
    print("  runner.export_best_for_dataset('dataset_name', mode='predictions')")
    print()
    print("  # Save to library with automatic n_features extraction")
    print("  library.save_pipeline_full(run_dir, pipeline_dir, 'my_model')")
    print("  # -> library_metadata.json includes n_features for compatibility!")
    print("\nTo clean up: shutil.rmtree('workspace_export_example')")


if __name__ == "__main__":
    main()
