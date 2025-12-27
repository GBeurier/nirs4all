# Workspace Architecture

**Version**: 3.3  
**Status**: Implemented

This document describes the nirs4all workspace directory structure and file organization.

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Shallow structure** | Maximum 3 levels deep for easy navigation |
| **Sequential numbering** | Pipelines numbered `0001`, `0002`, etc. for clear execution order |
| **Dataset-centric runs** | All pipelines for a dataset in one folder |
| **Fast access** | Best results accessible via `best_<pipeline>.csv` per dataset |
| **Content-addressed binaries** | Deduplication within runs via `_binaries/` |
| **Library flexibility** | Three types: filtered, pipeline, fullrun |

---

## Directory Structure

```
workspace/
│
├── runs/                                   # Experimental runs
│   │
│   ├── wheat_sample1/                      # Dataset name (no date prefix)
│   │   │
│   │   ├── best_0042_pls_baseline_x9y8z7.csv   # Best prediction (auto-updated)
│   │   │
│   │   ├── _binaries/                      # Shared artifacts (lazy creation)
│   │   │   ├── transformer_MinMaxScaler_abc123.joblib
│   │   │   └── model_PLSRegression_def456.joblib
│   │   │
│   │   ├── 0001_pls_baseline_a1b2c3/       # Pipeline: number + name + hash
│   │   │   ├── pipeline.json               # Pipeline configuration
│   │   │   ├── manifest.yaml               # V3 manifest with artifacts
│   │   │   ├── metrics.json                # Train/val/test metrics
│   │   │   └── folds_*.csv                 # Fold predictions
│   │   │
│   │   ├── 0002_b2c3d4/                    # Pipeline without custom name
│   │   └── 0003_c3d4e5/
│   │
│   └── corn_samples/                       # Another dataset
│       ├── best_0088_svm_opt_m5n6o7.csv
│       ├── _binaries/
│       └── 0001_svm_opt_x1y2z3/
│
├── binaries/                               # Centralized artifact storage (V3)
│   ├── wheat_sample1/                      # Per-dataset binaries
│   │   ├── model_PLSRegression_abc123.joblib
│   │   └── transformer_StandardScaler_def456.joblib
│   └── corn_samples/
│
├── exports/                                # Best results (fast access)
│   │
│   ├── wheat_sample1/                      # Dataset-based exports
│   │   ├── PLSRegression_predictions.csv
│   │   ├── PLSRegression_pipeline.json
│   │   └── PLSRegression_summary.json
│   │
│   ├── best_predictions/                   # Quick access to predictions only
│   │   ├── wheat_sample1_0042_x9y8z7.csv
│   │   └── corn_samples_0088_m5n6o7.csv
│   │
│   └── session_reports/
│       └── wheat_sample1.html
│
├── library/                                # Reusable pipelines
│   │
│   ├── templates/                          # Pipeline configs (no binaries)
│   │   ├── baseline_pls.json
│   │   └── optimized_svm.json
│   │
│   └── trained/                            # Trained pipelines (3 types)
│       │
│       ├── filtered/                       # Config + metrics only
│       │   └── wheat_quality_v1/
│       │
│       ├── pipeline/                       # Config + all binaries
│       │   └── wheat_quality_v1/
│       │
│       └── fullrun/                        # Everything + training data
│           └── wheat_quality_v1/
│
└── catalog/                                # Prediction index (permanent)
    ├── predictions_meta.parquet            # Fast queries (no arrays)
    ├── predictions_data.parquet            # Arrays (on-demand)
    └── archives/
        └── best_predictions/
```

---

## Key File Formats

### pipeline.json

```json
{
  "id": "0042_x9y8z7",
  "hash": "x9y8z7",
  "created_at": "2024-10-23T10:45:30Z",
  "status": "completed",
  "steps": [
    {
      "step": 0,
      "operator": "StandardScaler",
      "class": "sklearn.preprocessing.StandardScaler",
      "params": {"with_mean": true, "with_std": true}
    },
    {
      "step": 1,
      "operator": "PLSRegression",
      "class": "sklearn.cross_decomposition.PLSRegression",
      "params": {"n_components": 5}
    }
  ],
  "artifacts": [
    {
      "step": 0,
      "name": "StandardScaler",
      "hash": "abc123",
      "path": "../_binaries/transformer_StandardScaler_abc123.joblib"
    }
  ]
}
```

### metrics.json

```json
{
  "train": {"rmse": 0.32, "r2": 0.95, "mae": 0.25},
  "val": {"rmse": 0.38, "r2": 0.92, "mae": 0.31},
  "test": {"rmse": 0.42, "r2": 0.90, "mae": 0.34},
  "cross_validation": {
    "folds": 5,
    "mean_rmse": 0.40,
    "std_rmse": 0.05,
    "fold_results": [
      {"fold": 1, "rmse": 0.38},
      {"fold": 2, "rmse": 0.45}
    ]
  }
}
```

### manifest.yaml (V3)

```yaml
schema_version: "3.0"
pipeline_id: "0042_x9y8z7"
dataset: wheat_sample1
n_features: 1024

artifacts:
  items:
    - artifact_id: "0042_x9y8z7$abc123def456:all"
      chain_path: "s1.StandardScaler"
      artifact_type: transformer
      class_name: StandardScaler
      path: transformer_StandardScaler_abc123.joblib
      content_hash: "sha256:abc123..."
      version: 3
```

---

## Naming Conventions

### Pipeline IDs

Format: `NNNN_[name_]hash`

- `NNNN`: 4-digit sequential number (0001, 0002, ...)
- `name`: Optional custom name (lowercase, underscores)
- `hash`: 6-character config hash

Examples:
- `0001_a1b2c3` (no custom name)
- `0042_pls_baseline_x9y8z7` (with custom name)

### Artifact Filenames

Format: `{type}_{class}_{short_hash}.{ext}`

Examples:
- `model_PLSRegression_abc123def456.joblib`
- `transformer_StandardScaler_def456789012.joblib`
- `encoder_LabelEncoder_ghi789012345.joblib`

---

## Library Types

| Type | Contents | Use Case | Size |
|------|----------|----------|------|
| **templates/** | Config JSON only | Share pipeline recipes | Small |
| **filtered/** | Config + metrics | Track experiments | Small |
| **pipeline/** | Config + all binaries | Deploy/retrain | Medium |
| **fullrun/** | Everything + data | Full reproducibility | Large |

---

## API Classes

### SimulationSaver

Main class for managing pipeline output storage.

```python
from nirs4all.pipeline.storage.io import SimulationSaver

saver = SimulationSaver(
    base_path=runs_dir,
    save_artifacts=True,
    save_charts=True
)

# Register pipeline
saver.register("0001_abc123")

# Save files
saver.save_json("metrics.json", metrics_dict)
saver.save_output(step_number=1, name="chart", data=png_bytes, extension=".png")

# Export best results
saver.export_best_for_dataset("wheat_sample1", workspace_path, runs_dir)
```

### LibraryManager

Manage saved pipeline templates and trained models.

```python
from nirs4all.workspace import LibraryManager

library = LibraryManager(workspace / "library")

# Save template
library.save_template(pipeline_config, "baseline_pls", "PLS baseline")

# Save trained pipeline
library.save_pipeline_full(run_dir, pipeline_dir, "wheat_quality_v1")

# List and load
templates = library.list_templates()
config = library.load_template("baseline_pls")
```

### PipelineLibrary

Alternative library manager with category support.

```python
from nirs4all.pipeline.storage.library import PipelineLibrary

library = PipelineLibrary(workspace_path)

# Save with category and tags
library.save_template(
    pipeline_config,
    name="optimized_pls",
    category="regression",
    tags=["nirs", "pls", "optimized"],
    metrics={"rmse": 0.42}
)

# Search templates
templates = library.list_templates(category="regression", tags=["pls"])
```

---

## Common Workflows

### 1. Training Session

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(
    workspace="./workspace",
    save_artifacts=True
)

# Run pipeline - creates workspace/runs/{dataset}/0001_hash/
predictions, per_dataset = runner.run(pipeline, dataset)
```

### 2. Export Best Model

```python
from nirs4all.workspace import LibraryManager

# After training, save best to library
library = LibraryManager(workspace / "library")
library.save_pipeline_full(
    run_dir=runs_dir / "wheat_sample1",
    pipeline_dir=runs_dir / "wheat_sample1" / "0042_x9y8z7",
    name="wheat_quality_prod_v1"
)
```

### 3. Load and Predict

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(workspace="./workspace")
predictions = runner.predict(
    source="library/trained/pipeline/wheat_quality_prod_v1",
    data=new_samples
)
```

### 4. Cleanup Old Runs

```bash
# Delete runs older than 30 days (catalog preserves best results)
find workspace/runs -mtime +30 -type d -name "20*" -exec rm -rf {} +
```

---

## See Also

- [Storage API](./storage.md) - Artifact storage reference
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration
- [CLI Reference](/reference/cli) - Command-line interface
