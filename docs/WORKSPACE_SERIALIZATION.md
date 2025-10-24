# Workspace Architecture & Serialization

## Overview

nirs4all uses a structured workspace approach to organize pipelines, trained models, predictions, and exports. The system employs content-addressed storage with automatic deduplication for efficient artifact management.

## Workspace Structure

```
workspace/
├── runs/
│   └── YYYY-MM-DD_dataset/                 # Run directory per dataset
│       ├── 0001_name_hash/                 # Sequential pipeline directories
│       │   ├── manifest.yaml               # Pipeline metadata & artifact registry
│       │   ├── pipeline.json               # Pipeline configuration
│       │   ├── Report_best_<pipeline>_<model>_<id>.csv  # Report (no date prefix!)
│       │   └── folds_*.csv
│       ├── 0002_name_hash/
│       ├── _binaries/                      # Shared binary artifacts (models, scalers)
│       │   ├── PLSRegression_a3f2e1.joblib
│       │   ├── StandardScaler_b4c9d2.joblib
│       │   └── LogisticRegression_c5e8f3.pkl
│       └── Best_prediction_<pipeline>_<model>_<id>.csv  # Best prediction (run root)
├── exports/                                # Best results per dataset (ONE CALL!)
│   └── dataset_name/                       # Organized by dataset
│       ├── YYYY-MM-DD_<model>_predictions.csv      # Includes run date
│       ├── YYYY-MM-DD_<model>_pipeline.json
│       ├── YYYY-MM-DD_<model>_summary.json
│       └── YYYY-MM-DD_<model>_chart.png            # Charts if available
├── library/                                # Reusable pipeline templates and models
│   ├── templates/
│   │   └── pls_baseline.json
│   └── trained/
│       ├── filtered/                       # Config + metrics only
│       │   └── model_name/
│       │       ├── pipeline.json
│       │       └── library_metadata.json   # Includes n_features!
│       ├── pipeline/                       # Full trained models
│       │   └── model_name/
│       │       ├── pipeline.json
│       │       ├── library_metadata.json   # Includes n_features!
│       │       └── _binaries/
│       └── fullrun/                        # Complete experiments
│           └── experiment_name/
└── dataset_name.json                       # Global predictions database per dataset
```

## Pipeline Organization

### Sequential Numbering
Pipelines are automatically numbered sequentially within each run directory:
- Format: `<sequence>_<name>_<hash>`
- Example: `0001_pls_baseline_a3f2e1`
- Hash represents the pipeline configuration fingerprint (no duplication)

### Manifest System
Each pipeline directory contains a `manifest.yaml` with:
- Pipeline metadata (UID, name, dataset, creation time)
- Artifact registry (models, scalers, transformers)
- Prediction results and metrics
- Execution history and parameters

## Serialization System

### Content-Addressed Storage
Binary artifacts (models, scalers, arrays) are stored using content-addressed storage:

**Key Features:**
- **Deduplication**: Identical objects share the same file (detected via SHA-256 hash)
- **Flat structure**: Files stored directly in `_binaries/` (no subdirectory sharding)
- **Meaningful names**: `ClassName_hash.ext` format (e.g., `PLSRegression_a3f2e1.joblib`)
- **Format detection**: Automatic selection of optimal serialization (joblib, pickle, numpy)

**Example:**
```python
from nirs4all.utils.serializer import persist, load

# Persist object
artifact_meta = persist(model, artifacts_dir, "my_model")
# Creates: _binaries/PLSRegression_a3f2e1.joblib

# Load object
loaded_model = load(artifact_meta, results_dir)
```

### Supported Formats
| Type | Format | Extension |
|------|--------|-----------|
| scikit-learn models | joblib | `.joblib` |
| Generic Python objects | pickle | `.pkl` |
| NumPy arrays | numpy | `.npy` |
| Pandas DataFrames | pickle | `.pkl` |

### Deduplication Strategy
1. Compute SHA-256 hash of serialized object
2. Check if artifact with same hash exists
3. If exists: reuse existing file, update metadata only
4. If new: create new file with `ClassName_hash.ext` naming

## Global Predictions Database

Each dataset maintains a JSON database at the workspace root:

**File**: `workspace/dataset_name.json`

**Structure:**
```json
[
  {
    "pipeline_uid": "0001_pls_baseline_a3f2e1",
    "dataset": "dataset_name",
    "task": "regression",
    "metrics": {
      "r2": 0.89,
      "rmse": 0.23
    },
    "timestamp": "2024-10-25T14:30:00",
    "run_dir": "runs/2024-10-25_dataset"
  }
]
```

**Usage:**
- Query best performing pipelines across all runs
- Compare performance across different configurations
- Track model evolution over time
- Filter by metrics, task type, or date range

## Library Management

### Library Metadata Structure

Each saved pipeline in the library includes a `library_metadata.json` file with compatibility information:

```json
{
  "name": "best_model",
  "description": "Best performing model for deployment",
  "saved_at": "2025-10-25T14:30:00",
  "type": "pipeline",  // "template", "filtered", "pipeline", or "fullrun"
  "source": "workspace/runs/2025-10-25_dataset/0001_name_hash",
  "n_features": 2151   // Extracted automatically for compatibility!
}
```

### LibraryManager API

```python
from nirs4all.workspace import LibraryManager

library = LibraryManager(library_dir)

# Save template (config only)
library.save_template(pipeline_config, "my_template", "Description")

# Save filtered (config + metrics, includes n_features)
library.save_filtered(pipeline_dir, "my_filtered", "Description")

# Save full pipeline (config + binaries, includes n_features)
library.save_pipeline_full(run_dir, pipeline_dir, "my_model", "Description")

# Save complete run
library.save_fullrun(run_dir, "my_experiment", "Description")

# List and load
templates = library.list_templates()
pipelines = library.list_pipelines()  # Each includes n_features!
```

### Export API

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(workspace_path="workspace")

# Export best results for a dataset (ONE CALL!)
export_dir = runner.export_best_for_dataset(
    dataset_name="my_dataset",
    mode="predictions"  // "predictions", "template", "trained", or "full"
)

# Creates:
# exports/my_dataset/
#   ├── YYYY-MM-DD_<model>_predictions.csv
#   ├── YYYY-MM-DD_<model>_pipeline.json
#   ├── YYYY-MM-DD_<model>_summary.json
#   └── YYYY-MM-DD_<model>_*.png  (charts if available)
```

## For Developers

### Filename Conventions

**In Pipeline Directories:**
- `Report_best_<pipeline_id>_<model>_<pred_id>.csv` - Performance report
- `folds_*.csv` - Cross-validation fold information
- `pipeline.json` - Pipeline configuration
- No date/time prefixes (redundant with run directory name)

**In Run Root:**
- `Best_prediction_<pipeline_id>_<model>_<pred_id>.csv` - Best prediction for run

**In Exports (with run date):**
- `YYYY-MM-DD_<model>_predictions.csv` - Exported predictions
- `YYYY-MM-DD_<model>_pipeline.json` - Pipeline configuration
- `YYYY-MM-DD_<model>_summary.json` - Metadata summary

### Serializer Module (`nirs4all.utils.serializer`)

**Core Functions:**
```python
persist(obj, artifacts_dir, custom_name=None) -> ArtifactMeta
load(artifact_meta, results_dir) -> Any
compute_hash(obj) -> str
detect_framework(obj) -> str
is_serializable(obj) -> bool
```

**ArtifactMeta Structure:**
```python
{
    "hash": "sha256:abc123...",      # Content hash
    "name": "custom_name",            # User-provided name
    "path": "ClassName_hash.ext",     # Relative path in _binaries/
    "format": "joblib|pickle|numpy",  # Serialization format
    "size": 12345,                    # File size in bytes
    "saved_at": "2024-10-25T14:30:00",
    "step": 0                         # Pipeline step index
}
```

### Binary Loader (`nirs4all.pipeline.binary_loader`)

Lazy loading of artifacts with caching:

```python
from nirs4all.pipeline.binary_loader import BinaryLoader

# Initialize from manifest
loader = BinaryLoader.from_manifest(manifest, results_dir)

# Load artifacts for specific pipeline step
binaries = loader.get_step_binaries("0")  # Returns [(name, obj), ...]

# Cache management
loader.clear_cache()
info = loader.get_cache_info()  # Get cache statistics
```

### Backward Compatibility

The serializer supports loading old sharded artifacts:
```python
# Old format: artifacts/objects/ab/abc123.joblib
# New format: _binaries/ClassName_abc123.joblib

# load() automatically detects and handles both formats
```

## Best Practices

**For Users:**
- Use descriptive custom names when creating pipelines
- **Export best results with ONE CALL**: `runner.export_best_for_dataset('dataset_name')`
- Library automatically extracts n_features for compatibility checking
- Leverage the global predictions database for performance queries
- Organize exports by dataset (automatic with new API)

**Simple Export Workflow:**
```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.workspace import LibraryManager

# After running pipelines
runner = PipelineRunner(workspace_path="workspace")
# ... run pipelines ...

# Export best results (ONE CALL!) - creates exports/dataset_name/
runner.export_best_for_dataset('my_dataset', mode='predictions')

# Save to library with automatic n_features extraction
library = LibraryManager(workspace_path / "library")
library.save_pipeline_full(run_dir, pipeline_dir, "best_model")
# -> Automatically extracts and stores n_features in library_metadata.json!
```

**For Developers:**
- Always use `persist()` for saving binary artifacts (automatic deduplication)
- Use `BinaryLoader` for lazy loading (better memory efficiency)
- Use `PipelineRunner.export_best_for_dataset()` for user-facing exports
- `LibraryManager._extract_n_features()` handles n_features extraction automatically
- Implement `__repr__` and `__eq__` for custom objects (better deduplication)
- Check `is_serializable()` before attempting to persist complex objects

## Migration Notes

If upgrading from older versions:
1. Old artifacts in `artifacts/objects/<hash[:2]>/<hash>` are still loadable
2. New artifacts use flat structure in `_binaries/`
3. Global predictions moved from `runs/<date>/predictions.json` to `workspace/<dataset>.json`
4. Pipeline outputs no longer nested in `outputs/` subdirectory
