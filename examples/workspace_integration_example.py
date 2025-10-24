"""
Workspace Integration Example - Flat Sequential Architecture

This example demonstrates the new simplified workspace architecture:
- Flat sequential pipeline numbering (0001_hash, 0002_hash, ...)
- Date-prefixed run directories (YYYY-MM-DD_dataset/)
- Content-addressed artifact storage
- No WorkspaceManager - simpler direct approach

Structure:
workspace/
  runs/
    2025-10-24_regression/
      0001_pls_baseline_abc123/
        manifest.yaml
        metrics.json
        predictions.csv
        outputs/
      0002_svm_test_def456/
        ...
      artifacts/
        objects/          # Content-addressed storage
    2025-10-24_classification/
      0001_randomforest_ghi789/
        ...
  library/
    templates/
    trained/
"""

from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# Import the main components
from nirs4all import PipelineRunner, PipelineConfigs
from nirs4all.dataset import DatasetConfigs
from nirs4all.workspace import LibraryManager


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_basic_pipeline():
    """
    Example 1: Basic Pipeline Execution

    Shows how to run a simple pipeline and understand the workspace structure.
    """
    print_section("EXAMPLE 1: Basic Pipeline Execution")

    # Configure a simple pipeline
    pipeline = [
        ShuffleSplit(n_splits=2, random_state=42),
        {'model': PLSRegression(n_components=5), 'name': 'PLS_baseline'}
    ]
    pipeline_config = PipelineConfigs(pipeline, name='pls_baseline')

    # Configure dataset
    dataset_path = Path(__file__).parent / 'sample_data' / 'regression'
    dataset_config = DatasetConfigs([str(dataset_path)])

    # Run pipeline with workspace
    print("\n📝 Running pipeline...")
    runner = PipelineRunner(
        workspace_path='demo_workspace',
        save_files=True,
        verbose=1
    )

    predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

    print(f"\n✅ Pipeline completed!")
    print(f"   Generated {predictions.num_predictions} predictions")

    # Show workspace structure
    print("\n📁 Workspace structure created:")
    workspace_path = Path('demo_workspace')
    show_workspace_structure(workspace_path)

    return workspace_path


def example_multiple_pipelines():
    """
    Example 2: Multiple Pipelines with Sequential Numbering

    Demonstrates how multiple pipelines get sequential numbers (0001, 0002, 0003).
    """
    print_section("EXAMPLE 2: Multiple Pipelines - Sequential Numbering")

    dataset_path = Path(__file__).parent / 'sample_data' / 'regression'
    dataset_config = DatasetConfigs([str(dataset_path)])

    runner = PipelineRunner(
        workspace_path='demo_workspace',
        save_files=True,
        verbose=0
    )

    # Pipeline 1: PLS with 3 components
    print("\n🔄 Running pipeline 1: PLS(3)...")
    pipeline1 = [
        ShuffleSplit(n_splits=1, random_state=42),
        {'model': PLSRegression(n_components=3), 'name': 'PLS_3comp'}
    ]
    config1 = PipelineConfigs(pipeline1, name='pls_3components')
    runner.run(config1, dataset_config)

    # Pipeline 2: PLS with 5 components
    print("🔄 Running pipeline 2: PLS(5)...")
    pipeline2 = [
        ShuffleSplit(n_splits=1, random_state=42),
        {'model': PLSRegression(n_components=5), 'name': 'PLS_5comp'}
    ]
    config2 = PipelineConfigs(pipeline2, name='pls_5components')
    runner.run(config2, dataset_config)

    # Pipeline 3: PLS with 10 components
    print("🔄 Running pipeline 3: PLS(10)...")
    pipeline3 = [
        ShuffleSplit(n_splits=1, random_state=42),
        {'model': PLSRegression(n_components=10), 'name': 'PLS_10comp'}
    ]
    config3 = PipelineConfigs(pipeline3, name='pls_10components')
    runner.run(config3, dataset_config)

    print("\n✅ All pipelines completed!")

    # Show the sequential numbering
    print("\n📊 Sequential pipeline numbering:")
    workspace_path = Path('demo_workspace')
    run_dirs = list((workspace_path / 'runs').glob('*_regression'))
    if run_dirs:
        run_dir = run_dirs[0]
        pipeline_dirs = sorted([d for d in run_dir.iterdir()
                               if d.is_dir() and d.name[0].isdigit()])

        for i, pdir in enumerate(pipeline_dirs, 1):
            print(f"   Pipeline {i}: {pdir.name}")
            manifest_file = pdir / 'manifest.yaml'
            if manifest_file.exists():
                import yaml
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                    print(f"      Name: {manifest.get('name', 'N/A')}")
                    print(f"      Dataset: {manifest.get('dataset', 'N/A')}")


def example_library_management():
    """
    Example 3: Library Management

    Shows how to save pipeline configurations and trained models to library.
    """
    print_section("EXAMPLE 3: Library Management")

    workspace_path = Path('demo_workspace')
    library = LibraryManager(workspace_path / 'library')

    # Save a template (configuration only)
    print("\n💾 Saving pipeline template...")
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
        "Baseline PLS configuration for regression tasks"
    )
    print(f"   ✓ Template saved: {template_path.name}")

    # List all templates
    print("\n📚 Templates in library:")
    templates = library.list_templates()
    for template in templates:
        print(f"   - {template['name']}")
        if 'description' in template:
            print(f"     Description: {template['description']}")

    # Load a template
    if templates:
        print(f"\n📖 Loading template: {templates[0]['name']}")
        loaded_config = library.load_template(templates[0]['name'])
        print("   ✓ Template loaded successfully")
        print(f"   Model: {loaded_config.get('model', {}).get('name', 'N/A')}")


def example_understanding_structure():
    """
    Example 4: Understanding the Workspace Structure

    Detailed explanation of each directory and file.
    """
    print_section("EXAMPLE 4: Understanding the Workspace Structure")

    print("""
📁 Workspace Structure Explained:

workspace/
├── runs/                           # All pipeline runs
│   ├── 2025-10-24_regression/     # Run directory (date + dataset)
│   │   ├── 0001_pls_baseline_abc123/  # Pipeline 1 (sequential number + name + hash)
│   │   │   ├── manifest.yaml      # Pipeline metadata (config, artifacts, predictions)
│   │   │   ├── metrics.json       # Performance metrics
│   │   │   ├── predictions.csv    # Model predictions
│   │   │   ├── pipeline.json      # Pipeline configuration
│   │   │   └── outputs/           # Additional outputs (plots, etc.)
│   │   ├── 0002_pls_test_def456/  # Pipeline 2
│   │   ├── 0003_svm_ghi789/       # Pipeline 3
│   │   ├── artifacts/             # Shared artifacts for this run
│   │   │   └── objects/           # Content-addressed storage (SHA256)
│   │   │       ├── sha256_abc.../model.pkl
│   │   │       └── sha256_def.../scaler.pkl
│   │   └── predictions.json       # Global predictions for this run
│   └── 2025-10-24_classification/ # Another run
│       └── ...
└── library/                        # Saved pipelines and templates
    ├── templates/                  # Pipeline configurations (JSON)
    │   └── baseline_pls.json
    └── trained/                    # Trained models
        ├── filtered/               # Config + metrics only
        ├── pipeline/               # Complete pipelines with binaries
        └── fullrun/                # Full run archives

KEY CONCEPTS:

1. Sequential Numbering:
   - Pipelines are numbered 0001, 0002, 0003, ... within each run
   - Makes it easy to identify execution order
   - Format: NNNN_name_hash (e.g., 0001_pls_baseline_abc123)

2. Content-Addressed Storage:
   - Artifacts (models, scalers) stored by SHA256 hash
   - Deduplication: identical objects stored once
   - Located in: runs/DATE_DATASET/artifacts/objects/

3. Manifest Files:
   - manifest.yaml: Complete pipeline metadata
   - Lists all artifacts, predictions, configuration
   - Used for reproducibility and loading

4. Flat Structure:
   - No nested dataset/pipeline directories
   - All pipelines flat in run directory
   - Dataset name in manifest, not directory structure
""")


def show_workspace_structure(workspace_path, max_depth=3):
    """
    Display the workspace directory structure.

    Args:
        workspace_path: Path to workspace
        max_depth: Maximum depth to display
    """
    def print_tree(path, prefix="", depth=0):
        if depth > max_depth:
            return

        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "

                # Show file/dir name with emoji
                if entry.is_dir():
                    if entry.name.startswith('20'):  # Date folder
                        name = f"📅 {entry.name}/"
                    elif entry.name[0:4].isdigit():  # Numbered pipeline
                        name = f"🔢 {entry.name}/"
                    elif entry.name == 'objects':
                        name = f"💾 {entry.name}/"
                    else:
                        name = f"📁 {entry.name}/"
                else:
                    if entry.suffix == '.yaml':
                        name = f"📄 {entry.name}"
                    elif entry.suffix == '.json':
                        name = f"📋 {entry.name}"
                    elif entry.suffix == '.csv':
                        name = f"📊 {entry.name}"
                    else:
                        name = f"📄 {entry.name}"

                print(f"{prefix}{current_prefix}{name}")

                if entry.is_dir():
                    print_tree(entry, prefix + next_prefix, depth + 1)
        except PermissionError:
            pass

    print(f"\n{workspace_path}/")
    print_tree(workspace_path)


def cleanup_demo_workspace():
    """Clean up the demo workspace."""
    import shutil
    workspace_path = Path('demo_workspace')
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
        print(f"\n🧹 Cleaned up demo workspace: {workspace_path}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         NIRS4ALL - Workspace Integration Examples                   ║
║         Flat Sequential Architecture (v3.2)                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Clean up any existing demo workspace
    cleanup_demo_workspace()

    try:
        # Run examples
        example_basic_pipeline()

        example_multiple_pipelines()

        example_library_management()

        example_understanding_structure()

        print_section("SUMMARY")
        print("""
✅ Examples completed successfully!

The new workspace architecture provides:
  • Flat sequential pipeline numbering (0001, 0002, ...)
  • Clear run organization by date and dataset
  • Content-addressed artifact storage (deduplication)
  • Simple library management for templates and trained models
  • No complex WorkspaceManager - direct folder operations

Key takeaways:
  1. Use PipelineRunner with workspace_path
  2. Pipelines get sequential numbers automatically
  3. All artifacts deduplicated via content addressing
  4. Library management separate from runs

For more examples, see:
  • workspace_full_integration_test.py - Comprehensive test
  • examples/Q*.py - Specific use cases
""")

    finally:
        # Uncomment to keep demo workspace for inspection
        # pass
        cleanup_demo_workspace()
