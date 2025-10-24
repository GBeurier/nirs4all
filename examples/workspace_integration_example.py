"""
Workspace Architecture - Integration Example

Demonstrates end-to-end usage of the new workspace architecture.
"""

from pathlib import Path
from nirs4all.workspace import WorkspaceManager, LibraryManager
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.dataset.predictions import Predictions


def example_full_workflow():
    """Complete workspace workflow example."""

    # ========== STEP 1: Initialize Workspace ==========
    print("=" * 60)
    print("STEP 1: Initialize Workspace")
    print("=" * 60)

    workspace_root = Path("example_workspace")
    workspace = WorkspaceManager(workspace_root)
    workspace.initialize_workspace()

    print(f"✓ Workspace initialized at: {workspace_root}")
    print(f"  - runs/")
    print(f"  - exports/")
    print(f"  - library/")
    print(f"  - catalog/")
    print()

    # ========== STEP 2: Create Runs ==========
    print("=" * 60)
    print("STEP 2: Create Multiple Runs")
    print("=" * 60)

    # Run 1: Default naming
    run1 = workspace.create_run("wheat_sample1")
    print(f"✓ Run 1: {run1.run_dir.name}")

    # Run 2: Custom naming
    run2 = workspace.create_run("wheat_sample1", run_name="baseline_experiment")
    print(f"✓ Run 2: {run2.run_dir.name}")

    # Run 3: Different dataset
    run3 = workspace.create_run("corn_sample1", run_name="comparison_study")
    print(f"✓ Run 3: {run3.run_dir.name}")
    print()

    # ========== STEP 3: Register Pipelines in Run ==========
    print("=" * 60)
    print("STEP 3: Register Pipelines in Run")
    print("=" * 60)

    saver = SimulationSaver()

    # Pipeline 1: Default naming
    pipeline1_dir = saver.register_workspace(
        workspace_root=workspace_root,
        dataset_name="wheat_sample1",
        pipeline_hash="abc123def456",
        run_name="baseline_experiment"
    )
    print(f"✓ Pipeline 1: {pipeline1_dir.name}")

    # Pipeline 2: Custom naming
    pipeline2_dir = saver.register_workspace(
        workspace_root=workspace_root,
        dataset_name="wheat_sample1",
        pipeline_hash="ghi789jkl012",
        run_name="baseline_experiment",
        pipeline_name="optimized_pls"
    )
    print(f"✓ Pipeline 2: {pipeline2_dir.name}")

    # Pipeline 3: Sequential numbering
    pipeline3_dir = saver.register_workspace(
        workspace_root=workspace_root,
        dataset_name="wheat_sample1",
        pipeline_hash="mno345pqr678",
        run_name="baseline_experiment"
    )
    print(f"✓ Pipeline 3: {pipeline3_dir.name}")
    print()

    # ========== STEP 4: Archive to Catalog ==========
    print("=" * 60)
    print("STEP 4: Archive Predictions to Catalog")
    print("=" * 60)

    catalog_dir = workspace_root / "catalog"

    # Create sample predictions (in real usage, these come from actual model runs)
    pred = Predictions()

    # Archive pipeline 1
    # Note: In real usage, read from pipeline_dir / "predictions.csv"
    # Here we simulate the metrics
    pred_id1 = "demo_pred_001"  # pred.archive_to_catalog() would return UUID
    print(f"✓ Archived pipeline 1 predictions: {pred_id1}")

    pred_id2 = "demo_pred_002"
    print(f"✓ Archived pipeline 2 predictions: {pred_id2}")

    pred_id3 = "demo_pred_003"
    print(f"✓ Archived pipeline 3 predictions: {pred_id3}")
    print()

    # ========== STEP 5: Query Catalog ==========
    print("=" * 60)
    print("STEP 5: Query Catalog")
    print("=" * 60)

    # Note: This requires actual Parquet files to exist
    # In real usage after archiving predictions:
    """
    catalog_pred = Predictions.load_from_parquet(catalog_dir)

    # Find best pipelines
    best = catalog_pred.query_best(metric="test_score", n=5)
    print("Top 5 pipelines by test_score:")
    print(best)

    # Filter by criteria
    good_pipelines = catalog_pred.filter_by_criteria(
        dataset_name="wheat_sample1",
        metric_thresholds={"test_score": 0.50}
    )
    print(f"\\nPipelines with test_score >= 0.50: {good_pipelines.height}")

    # Get summary stats
    stats = catalog_pred.get_summary_stats(metric="test_score")
    print(f"\\nTest Score Statistics:")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    """
    print("✓ Catalog query methods available:")
    print("  - query_best(metric, n)")
    print("  - filter_by_criteria(dataset, date_range, thresholds)")
    print("  - compare_across_datasets(pipeline_hash, metric)")
    print("  - list_runs(dataset_name)")
    print("  - get_summary_stats(metric)")
    print()

    # ========== STEP 6: Export Best Models ==========
    print("=" * 60)
    print("STEP 6: Export Best Models")
    print("=" * 60)

    exports_dir = workspace_root / "exports"

    # Export full pipeline
    """
    export_path = saver.export_pipeline_full(
        pipeline_dir=pipeline2_dir,
        exports_dir=exports_dir,
        dataset_name="wheat_sample1",
        run_date="20241024",
        custom_name="production_model_v1"
    )
    print(f"✓ Exported full pipeline: {export_path.name}")
    """

    # Export best predictions CSV
    """
    pred_csv = pipeline2_dir / "predictions.csv"
    if pred_csv.exists():
        export_csv = saver.export_best_prediction(
            predictions_file=pred_csv,
            exports_dir=exports_dir,
            dataset_name="wheat_sample1",
            run_date="20241024",
            pipeline_id="0002_optimized_pls",
            custom_name="best_baseline"
        )
        print(f"✓ Exported predictions: {export_csv.name}")
    """
    print("✓ Export methods available:")
    print("  - export_pipeline_full() -> exports/full_pipelines/")
    print("  - export_best_prediction() -> exports/best_predictions/")
    print()

    # ========== STEP 7: Library Management ==========
    print("=" * 60)
    print("STEP 7: Library Management")
    print("=" * 60)

    library = LibraryManager(workspace_root / "library")

    # Save template (config only)
    template_config = {
        "preprocessing": [
            {"name": "StandardScaler"},
            {"name": "SNV"}
        ],
        "model": {
            "name": "PLSRegression",
            "n_components": 5
        }
    }
    template_path = library.save_template(
        template_config,
        "baseline_pls_template",
        "Baseline PLS configuration for wheat"
    )
    print(f"✓ Saved template: {template_path.name}")

    # Save filtered (config + metrics only)
    """
    if pipeline2_dir.exists():
        filtered_path = library.save_filtered(
            pipeline2_dir,
            "wheat_baseline_v1",
            "First baseline experiment on wheat"
        )
        print(f"✓ Saved filtered: {filtered_path.name}")
    """

    # Save full pipeline (with binaries)
    """
    if pipeline2_dir.exists() and run2_dir.exists():
        pipeline_path = library.save_pipeline_full(
            run2_dir,
            pipeline2_dir,
            "wheat_production_model",
            "Production-ready model for wheat dataset"
        )
        print(f"✓ Saved full pipeline: {pipeline_path.name}")
    """

    # Save complete run
    """
    if run2_dir.exists():
        fullrun_path = library.save_fullrun(
            run2_dir,
            "wheat_baseline_complete",
            "Complete baseline experiment with all pipelines"
        )
        print(f"✓ Saved full run: {fullrun_path.name}")
    """
    print("✓ Library save methods available:")
    print("  - save_template() -> library/templates/")
    print("  - save_filtered() -> library/trained/filtered/")
    print("  - save_pipeline_full() -> library/trained/pipeline/")
    print("  - save_fullrun() -> library/trained/fullrun/")
    print()

    # List saved items
    templates = library.list_templates()
    print(f"✓ Templates in library: {len(templates)}")
    for t in templates:
        print(f"  - {t['name']}: {t.get('description', 'No description')}")
    print()

    # ========== STEP 8: List Runs ==========
    print("=" * 60)
    print("STEP 8: List All Runs")
    print("=" * 60)

    runs = workspace.list_runs()
    print(f"✓ Total runs: {len(runs)}")
    for run_info in runs:
        print(f"  - {run_info['name']}")
        print(f"    Dataset: {run_info['dataset']}")
        print(f"    Date: {run_info['date']}")
        if run_info.get('custom_name'):
            print(f"    Custom name: {run_info['custom_name']}")
    print()

    print("=" * 60)
    print("✓ Workspace workflow complete!")
    print("=" * 60)
    print()
    print("Workspace structure:")
    print(f"{workspace_root}/")
    print("  runs/")
    print("    baseline_experiment_wheat_sample1/")
    print("      0001_abc123def456/")
    print("      0002_optimized_pls/")
    print("      0003_mno345pqr678/")
    print("      _binaries/")
    print("  exports/")
    print("    full_pipelines/")
    print("    best_predictions/")
    print("  library/")
    print("    templates/")
    print("      baseline_pls_template.json")
    print("    trained/")
    print("      filtered/")
    print("      pipeline/")
    print("      fullrun/")
    print("  catalog/")
    print("    predictions_meta.parquet")
    print("    predictions_data.parquet")


def example_query_catalog():
    """Example of querying the catalog."""

    print("\n" + "=" * 60)
    print("CATALOG QUERY EXAMPLES")
    print("=" * 60 + "\n")

    catalog_dir = Path("example_workspace/catalog")

    # Load catalog
    print("Loading catalog...")
    """
    pred = Predictions.load_from_parquet(catalog_dir)
    print(f"✓ Loaded {pred._df.height} predictions from catalog\\n")

    # Example 1: Find best models
    print("1. Top 10 models by test_score:")
    best = pred.query_best(metric="test_score", n=10)
    print(best)
    print()

    # Example 2: Filter by dataset and threshold
    print("2. Wheat models with test_score >= 0.45:")
    filtered = pred.filter_by_criteria(
        dataset_name="wheat_sample1",
        metric_thresholds={"test_score": 0.45}
    )
    print(f"   Found {filtered.height} models")
    print()

    # Example 3: Compare across datasets
    print("3. Compare pipeline abc123 across datasets:")
    comparison = pred.compare_across_datasets("abc123", metric="test_score")
    print(comparison)
    print()

    # Example 4: Summary statistics
    print("4. Test score statistics:")
    stats = pred.get_summary_stats(metric="test_score")
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
    print()

    # Example 5: List all runs
    print("5. List all runs:")
    runs = pred.list_runs()
    print(runs)
    """

    print("Catalog query methods:")
    print("  ✓ query_best()")
    print("  ✓ filter_by_criteria()")
    print("  ✓ compare_across_datasets()")
    print("  ✓ list_runs()")
    print("  ✓ get_summary_stats()")


if __name__ == "__main__":
    # Run full workflow example
    example_full_workflow()

    # Uncomment to run query examples (requires actual catalog data)
    # example_query_catalog()
