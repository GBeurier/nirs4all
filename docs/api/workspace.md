# Workspace Architecture - User-Friendly Design

**Version**: 3.3 (Updated)
**Date**: December 19, 2025
**Status**: Implemented

---

## Executive Summary

This architecture prioritizes **user-friendliness** and **practical daily use** with shallow folder structures, sequential numbering, and fast access to best results.

### Key Principles

✅ **Shallow structure** - Maximum 3 levels deep
✅ **Sequential numbering** - Easy browsing, clear execution order
✅ **Dataset-centric runs** - All pipelines for a dataset in one folder (no date prefix)
✅ **Fast access** - Best results as single `best_<pipeline>.csv` file per dataset
✅ **No broken links** - Catalog stores copies, not references
✅ **Split Parquet storage** - Metadata separate from arrays for fast filtering
✅ **Content-addressed binaries** - Deduplication within runs (lazy creation in `_binaries/`)
✅ **Library flexibility** - Three types: filtered, full pipeline, full run

### Integration with Existing Code

This architecture **extends** existing nirs4all components:

- **ManifestManager** (`pipeline/manifest_manager.py`) - Extended for sequential numbering per run
- **SimulationSaver** (`pipeline/io.py`) - Updated to `runs/<dataset>/NNNN_hash/` structure with export methods
- **Predictions** (`dataset/predictions.py`) - Extended with split Parquet storage and catalog/query methods
- **PipelineRunner** (`pipeline/runner.py`) - Minimal changes, uses new workspace coordination

**No backward compatibility** - clean implementation for new workflows.

---

## Complete Workspace Structure

```
workspace/
│
├── runs/                                          # Experimental runs
│   │
│   ├── wheat_sample1/                            # Dataset name only (no date prefix)
│   │   │
│   │   ├── best_0042_pls_baseline_x9y8z7.csv    # Best prediction (replaced on better score)
│   │   │
│   │   ├── _binaries/                            # Shared artifacts (created only when needed)
│   │   │   ├── scaler_abc123.pkl                 # Custom name if provided
│   │   │   ├── PLSRegression_def456.pkl          # No custom name provided
│   │   │   └── svm_classifier_ghi789.pkl
│   │   │
│   │   ├── 0001_pls_baseline_a1b2c3/             # Pipeline: number + custom name + hash
│   │   │   ├── pipeline.json
│   │   │   ├── manifest.yaml
│   │   │   ├── metrics.json
│   │   │   └── folds_*.csv
│   │   │
│   │   ├── 0002_b2c3d4/                          # No custom name
│   │   ├── 0003_c3d4e5/
│   │   └── ...                                   # Up to 150+ pipelines
│   │
│   ├── corn_samples/                             # Another dataset
│   │   ├── best_0088_svm_opt_m5n6o7.csv
│   │   ├── _binaries/
│   │   ├── 0001_svm_opt_x1y2z3/
│   │   └── ...
│   │
│   └── wheat_sample2/                            # Different dataset
│       └── ...
│
├── exports/                                       # Best results (fast access)
│   │
│   ├── wheat_sample1/                            # Dataset-based exports
│   │   ├── PLSRegression_predictions.csv
│   │   ├── PLSRegression_pipeline.json
│   │   └── PLSRegression_summary.json
│   │
│   ├── best_predictions/                         # Quick access to just predictions
│   │   ├── wheat_sample1_0042_x9y8z7.csv
│   │   └── corn_samples_0088_m5n6o7.csv
│   │
│   └── session_reports/                          # HTML summaries
│       └── wheat_sample1.html
│
├── library/                                       # Reusable pipelines
│   │
│   ├── templates/                                 # Pipeline configs (no binaries)
│   │   ├── baseline_pls.json
│   │   ├── optimized_svm.json
│   │   └── neural_network_v1.json
│   │
│   └── trained/                                   # Trained pipelines (3 types)
│       │
│       ├── filtered/                              # Best model + essential preprocessing
│       │   ├── wheat_quality_v1.zip              # Minimal, production-ready
│       │   ├── wheat_quality_v2.zip
│       │   └── corn_moisture_v1.zip
│       │
│       ├── pipeline/                              # All preprocessing + all models
│       │   ├── wheat_quality_v1.zip              # Full experiment, reproducible
│       │   └── corn_moisture_v1.zip
│       │
│       └── fullrun/                               # Everything + training data
│           ├── wheat_quality_v1.zip              # Complete reproducibility
│           └── corn_moisture_v1.zip
│
└── catalog/                                       # Permanent storage (survives deletions)
    │
    ├── predictions_meta.parquet                   # Metadata: dataset, config, metrics (fast queries)
    ├── predictions_data.parquet                   # Arrays: y_true, y_pred, indices (loaded on-demand)
    │
    ├── predictions_meta.parquet                   # Metadata for fast filtering
    │   # Columns: prediction_id, dataset, run_date, pipeline_id, test_score, train_score, etc.
    │   # Example:
    │   # | prediction_id | dataset       | run_date   | pipeline_id | test_score | train_score |
    │   # | uuid-1234...  | wheat_sample1 | 2024-10-23 | 0042_x9...  | 0.45       | 0.32        |
    │
    ├── predictions_data.parquet                   # Arrays (loaded when needed)
    │   # Columns: prediction_id, y_true, y_pred, sample_indices, partition_y_true, partition_y_pred
    │   # Linked to metadata via prediction_id
    │   # Arrays stored as List[float] or List[int]
    │
    ├── reports/                                   # Generated reports
    │   ├── global_performance.html                # Cross-dataset comparison
    │   └── dataset_history/
    │       ├── wheat_sample1.html                 # Performance over time
    │       └── corn_samples.html
    │
    └── archives/                                  # Permanent copies (NOT links!)
        │
        ├── filtered/                              # Filtered pipelines
        │   ├── wheat_sample1_2024-10-23_0042_x9y8z7.zip
        │   └── corn_samples_2024-10-23_0088_m5n6o7.zip
        │
        ├── pipeline/                              # Full pipelines
        │   ├── wheat_sample1_2024-10-23_0042_x9y8z7.zip
        │   └── corn_samples_2024-10-23_0088_m5n6o7.zip
        │
        └── best_predictions/                      # Just predictions (lightweight)
            ├── wheat_sample1_2024-10-23_0042_x9y8z7.csv
            ├── wheat_sample1_2024-10-25_0015_a3b4c5.csv
            └── corn_samples_2024-10-23_0088_m5n6o7.csv
```
│   │   │   │
│   │   │   ├── wheat_sample1/             # Dataset name
│   │   │   │   │
│   │   │   │   ├── dataset_info.json      # Dataset metadata
│   │   │   │   │   {
│   │   │   │   │     "name": "wheat_sample1",
│   │   │   │   │     "samples": {"train": 150, "test": 50},
│   │   │   │   │     "features": 1024,
│   │   │   │   │     "targets": ["protein", "moisture"],
│   │   │   │   │     "loaded_at": "2024-10-23T10:30:16Z"
│   │   │   │   │   }
│   │   │   │   │
│   │   │   │   └── pipelines/             # All pipelines for this dataset
│   │   │   │       │
│   │   │   │       ├── baseline_pls_a1b2c3/      # Pipeline ID (name + hash)
│   │   │   │       │   │
│   │   │   │       │   ├── pipeline.json          # Complete pipeline definition
│   │   │   │       │   │   {
│   │   │   │       │   │     "id": "baseline_pls_a1b2c3",
│   │   │   │       │   │     "name": "baseline_pls",
│   │   │   │       │   │     "created_at": "2024-10-23T10:30:30Z",
│   │   │   │       │   │     "status": "completed",
│   │   │   │       │   │     "steps": [...],
│   │   │   │       │   │     "artifacts": [
│   │   │   │       │   │       {"step": 0, "name": "scaler", "path": "../../binaries/StandardScaler_abc123.pkl"},
│   │   │   │       │   │       {"step": 2, "name": "model", "path": "../../binaries/PLSRegression_ghi789.pkl"}
│   │   │   │       │   │     ]
│   │   │   │       │   │   }
│   │   │   │       │   │
│   │   │   │       │   ├── metrics.json           # All metrics (train/val/test)
│   │   │   │       │   │   {
│   │   │   │       │   │     "train": {"rmse": 0.32, "r2": 0.95, "mae": 0.25},
│   │   │   │       │   │     "val": {"rmse": 0.41, "r2": 0.91, "mae": 0.33},
│   │   │   │       │   │     "test": {"rmse": 0.45, "r2": 0.89, "mae": 0.36},
│   │   │   │       │   │     "cross_val": {"mean_rmse": 0.43, "std_rmse": 0.08}
│   │   │   │       │   │   }
│   │   │   │       │   │
│   │   │   │       │   ├── predictions.csv        # Predictions (y_true, y_pred, partition)
│   │   │   │       │   │   y_true,y_pred,partition,sample_id
│   │   │   │       │   │   12.5,12.3,train,sample_001
│   │   │   │       │   │   14.2,14.5,test,sample_150
│   │   │   │       │   │   ...
│   │   │   │       │   │
│   │   │   │       │   └── outputs/              # Visualizations & other outputs
│   │   │   │       │       ├── predictions_plot.png
│   │   │   │       │       ├── residuals_plot.png
│   │   │   │       │       ├── feature_importance.png
│   │   │   │       │       └── confusion_matrix.png  # For classification
│   │   │   │       │
│   │   │   │       ├── optimized_pls_b2c3d4/     # Another pipeline
│   │   │   │       │   ├── pipeline.json
│   │   │   │       │   ├── metrics.json
│   │   │   │       │   ├── predictions.csv
│   │   │   │       │   └── outputs/
│   │   │   │       │
│   │   │   │       └── failed_experiment_x1y2z3/ # Failed pipeline (kept for learning)
│   │   │   │           ├── pipeline.json
│   │   │   │           ├── error.log
│   │   │   │           └── partial_outputs/
│   │   │   │
│   │   │   └── wheat_sample2/             # Another dataset in same session
│   │   │       ├── dataset_info.json
│   │   │       └── pipelines/
│   │   │           ├── baseline_pls_c3d4e5/
│   │   │           └── optimized_svm_x9y8z7/
│   │   │
│   │   └── exports/                       # Best results from this session
│   │       ├── wheat_sample1_best.csv     # Best predictions
│   │       ├── wheat_sample1_report.json  # Detailed report
│   │       ├── wheat_sample2_best.csv
│   │       └── session_summary.md         # Human-readable summary
│   │
│   ├── 2024-10-24_exploratory-tests/      # Another session
│   │   ├── run_config.json
│   │   ├── run_summary.json
│   │   ├── run.log
│   │   ├── binaries/
│   │   ├── datasets/
│   │   └── exports/
│   │
│   └── 2024-10-25_production-validation/
│       └── ...
│
├── library/                               # Reusable pipeline library
│   │
│   ├── templates/                         # Generic pipeline templates
│   │   ├── basic_pls_regression.json     # Portable config (no binaries)
│   │   ├── svm_classification.yaml
│   │   └── neural_network_baseline.json
│   │
│   ├── trained/                           # Saved trained pipelines
│   │   ├── wheat_quality_v1.zip          # With binaries (ready to predict)
│   │   │   # Contains: pipeline.json + all .pkl/.keras files
│   │   ├── corn_moisture_v2.zip
│   │   └── protein_predictor_production.zip
│   │
│   └── experiments/                       # Full experiments (with data)
│       ├── wheat_full_study.zip          # Pipeline + binaries + training data
│       └── comparative_analysis.zip      # For reproducibility
│
└── catalog/                               # Global prediction index
    │
    ├── predictions.parquet          # Parquet file
    │   # Contains: session, dataset, pipeline, metrics, path
    │
    └── datasets/                         # Per-dataset tracking
        │
        ├── wheat_sample1/
        │   ├── index.json                # All predictions for this dataset
        │   │   {
        │   │     "dataset_name": "wheat_sample1",
        │   │     "predictions": [
        │   │       {
        │   │         "session": "2024-10-23_wheat-quality-study",
        │   │         "pipeline_id": "baseline_pls_a1b2c3",
        │   │         "pipeline_name": "baseline_pls",
        │   │         "created_at": "2024-10-23T10:31:22Z",
        │   │         "metrics": {"test_rmse": 0.45, "test_r2": 0.89},
        │   │         "path": "runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3/"
        │   │       },
        │   │       {
        │   │         "session": "2024-10-24_exploratory-tests",
        │   │         "pipeline_id": "optimized_svm_x9y8z7",
        │   │         ...
        │   │       }
        │   │     ],
        │   │     "best_model": {
        │   │       "pipeline_id": "baseline_pls_a1b2c3",
        │   │       "metric": "test_rmse",
        │   │       "value": 0.45
        │   │     }
        │   │   }
        │   │
        │   └── best_model_link.json      # Quick reference to best
        │       {
        │         "dataset": "wheat_sample1",
        │         "best_pipeline_id": "baseline_pls_a1b2c3",
        │         "best_session": "2024-10-23_wheat-quality-study",
        │         "metric": "test_rmse",
        │         "value": 0.45,
        │         "path": "runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3/"
        │       }
        │
        ├── wheat_sample2/
        │   └── ...
        │
        └── corn_samples/
            └── ...
```

---

## File Formats

### Split Parquet Storage (Catalog)

Predictions are stored in **two separate Parquet files** for optimal performance:

#### `predictions_meta.parquet` (Metadata - Fast Queries)
```
Columns:
- prediction_id (UUID from ManifestManager)
- dataset_name
- run_date
- pipeline_id
- config_name
- config_hash
- model_name
- task_type
- metrics (train_score, val_score, test_score)
- n_samples, n_features
- created_at
```

**Usage**: Fast filtering, finding best models, comparing configurations (no arrays loaded)

#### `predictions_data.parquet` (Arrays - On-Demand Loading)
```
Columns:
- prediction_id (links to metadata)
- y_true (array)
- y_pred (array)
- sample_indices (array)
- partition (array or single value)
- weights (optional array)
```

**Usage**: Plotting, detailed analysis, residuals calculation (explicit load)

**Why Split Storage?**
- ✅ 10-100x faster metadata queries (200 KB vs 20 MB)
- ✅ Arrays loaded only when needed
- ✅ Most workflows query metadata first, load data second
- ✅ Better memory efficiency for large result sets

---

## Design Principles

### 1. **Shallow Structure** (Max 3 Levels)

✅ **User-friendly browsing**:
```
runs/2024-10-23_wheat_sample1/0001_a1b2c3/predictions.csv
└── Only 4 levels deep!
```

Benefits:
- Quick navigation
- No deep folder diving
- Easy to understand at a glance

### 2. **Sequential Numbering** (Human-Readable)

✅ **Clear execution order**:
```
0001_a1b2c3/    # First pipeline
0002_b2c3d4/    # Second pipeline
0042_x9y8z7/    # 42nd pipeline (maybe the best one!)
```

Benefits:
- Easy to browse in order
- Number shows when it ran
- Hash ensures uniqueness
- No distinctive names needed

### 3. **Hidden Binaries** (_binaries/)

✅ **Clean user interface**:
```
runs/2024-10-23_wheat_sample1/
├── _binaries/              # Hidden with underscore
├── 0001_a1b2c3/           # User focuses here
├── 0002_b2c3d4/
└── 0003_c3d4e5/
```

Benefits:
- Underscore convention = "internal"
- Keeps focus on pipeline results
- Still accessible when needed
- Content-addressed deduplication

### 4. **Fast Access** (best_predictions/)

✅ **Dedicated folder for quick access**:
```
exports/best_predictions/
├── wheat_sample1_2024-10-23_0042_x9y8z7.csv    # Just open and view!
├── wheat_sample1_2024-10-25_0015_a3b4c5.csv
└── corn_samples_2024-10-23_0088_m5n6o7.csv
```

Benefits:
- One folder with all best predictions
- No need to open pipeline folders
- CSV files ready for Excel/analysis
- Lightweight (just predictions, no models)

### 5. **Permanent Catalog** (Copies, Not Links)

✅ **Survives run deletion**:
```python
# Copy to catalog (not link)
shutil.copy(
    "runs/2024-10-23_wheat_sample1/0042_x9y8z7/predictions.csv",
    "catalog/archives/best_predictions/wheat_sample1_2024-10-23_0042_x9y8z7.csv"
)

# Now safe to delete run
rm -rf runs/2024-10-23_wheat_sample1/
# Catalog still has the predictions!
```

Benefits:
- No broken links
- Permanent storage
- Safe to cleanup old runs
- Data never lost

### 6. **Library Flexibility** (3 Types)

✅ **Clear distinction**:

| Type | Contains | Use Case | Size |
|------|----------|----------|------|
| **filtered/** | Best model + essential preprocessing | Production | Small |
| **pipeline/** | All preprocessing + all models | Reproduction | Medium |
| **fullrun/** | Everything + training data | Full reproducibility | Large |

Benefits:
- Choose what you need
- Clear naming
- Flexible storage
- Efficient space usage

### 7. **Parquet Database** (Fast Queries)

✅ **Single file for all predictions**:
✅ **Single file for all predictions**:
```python
import polars as pl

df = pl.read_parquet("catalog/predictions.parquet")

# Best ever for wheat_sample1
best = df.filter(pl.col("dataset") == "wheat_sample1").sort("rmse").head(1)

# All runs from October
october = df.filter(pl.col("date").dt.month() == 10)

# Compare all models
comparison = df.group_by("dataset").agg(pl.col("rmse").min())
```

Benefits:
- Fast queries with polars/pandas
- Single file (no folders per dataset)
- Compressed storage
- Easy analytics

---

## File Format Specifications

### run_config.json

```json
{
  "session_name": "wheat_sample1",
  "created_at": "2024-10-23T10:30:00Z",
  "created_by": "username",
  "nirs4all_version": "0.6.0",
  "python_version": "3.11.5",
  "description": "Grid search for wheat quality prediction"
}
```

### run_summary.json

```json
{
  "session_name": "wheat_sample1",
  "status": "completed",
  "started_at": "2024-10-23T10:30:00Z",
  "completed_at": "2024-10-23T12:15:47Z",
  "duration_seconds": 6347,

  "statistics": {
    "total_pipelines": 150,
    "successful": 148,
    "failed": 2,
    "best_pipeline": "0042_x9y8z7",
    "best_rmse": 0.42,
    "artifact_count": 45,
    "artifact_size_mb": 12.3
  },

  "errors": [
    {
      "pipeline": "0089_a1b2c3",
      "error": "ValueError: Invalid hyperparameter",
      "timestamp": "2024-10-23T11:45:12Z"
    }
  ]
}
```

### pipeline.json

```json
{
  "id": "0042_x9y8z7",
  "hash": "x9y8z7",
  "created_at": "2024-10-23T10:45:30Z",
  "completed_at": "2024-10-23T10:46:36Z",
  "duration_seconds": 66,
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
      "path": "../_binaries/StandardScaler_abc123.pkl",
      "size_kb": 12
    },
    {
      "step": 1,
      "name": "PLSRegression",
      "hash": "x9y8z7",
      "path": "../_binaries/PLSRegression_x9y8z7.pkl",
      "size_kb": 850
    }
  ]
}
```

### metrics.json

```json
{
  "train": {
    "rmse": 0.32,
    "mae": 0.25,
    "r2": 0.95,
    "samples": 150
  },

  "val": {
    "rmse": 0.38,
    "mae": 0.31,
    "r2": 0.92,
    "samples": 30
  },

  "test": {
    "rmse": 0.42,
    "mae": 0.34,
    "r2": 0.90,
    "samples": 50
  },

  "cross_validation": {
    "folds": 5,
    "mean_rmse": 0.40,
    "std_rmse": 0.05,
    "fold_results": [
      {"fold": 1, "rmse": 0.38},
      {"fold": 2, "rmse": 0.45},
      {"fold": 3, "rmse": 0.39},
      {"fold": 4, "rmse": 0.37},
      {"fold": 5, "rmse": 0.41}
    ]
  }
}
```

### predictions.parquet Schema

```python
schema = {
    "dataset": pl.Utf8,           # "wheat_sample1"
    "session": pl.Utf8,           # "2024-10-23_wheat_sample1"
    "pipeline": pl.Utf8,          # "0042_x9y8z7"
    "date": pl.Date,              # 2024-10-23
    "rmse": pl.Float64,
    "r2": pl.Float64,
    "mae": pl.Float64,
    "n_samples": pl.Int64,
    "duration_sec": pl.Float64
}
```

---

## Workflows

### Workflow 1: Training 150 Pipelines

```python
session = SessionExecutor("2024-10-23_wheat_sample1")

for i, config in enumerate(grid_search_configs):  # 150 configs
    pipeline_hash = compute_hash(config)[:6]
    pipeline_id = f"{i+1:04d}_{pipeline_hash}"    # 0001_a1b2c3

    session.run_pipeline(pipeline_id, config)
    # Creates: runs/2024-10-23_wheat_sample1/0001_a1b2c3/

# Find best
best = session.get_best_pipeline()  # Based on test RMSE
print(f"Best: {best.id} with RMSE={best.metrics['test']['rmse']}")

# Export best
session.export_best()
# Creates: exports/wheat_sample1_2024-10-23_0042_x9y8z7/
# Also:    exports/best_predictions/wheat_sample1_2024-10-23_0042_x9y8z7.csv
```

**Result**: 150 pipelines trained, best exported for fast access.

### Workflow 2: Browsing Results (Fast Access)

```bash
# Open best predictions immediately
cd exports/best_predictions/
open wheat_sample1_2024-10-23_0042_x9y8z7.csv    # Excel/viewer

# See full best results
cd ../wheat_sample1_2024-10-23_0042_x9y8z7/
open chart_predictions.png                       # View chart
cat metrics.json                                 # See scores
```

**No digging through 150 folders!**

### Workflow 3: Saving to Library

```python
library = LibraryManager("workspace/library")

# Production deployment (minimal)
library.save_filtered(
    from_run="2024-10-23_wheat_sample1",
    pipeline="0042_x9y8z7",
    name="wheat_quality_prod_v1"
)
# → library/trained/filtered/wheat_quality_prod_v1.zip
# Contains: StandardScaler + PLS model only

# Experiment reproduction (all models)
library.save_pipeline(
    from_run="2024-10-23_wheat_sample1",
    pipeline="0042_x9y8z7",
    name="wheat_quality_exp_v1"
)
# → library/trained/pipeline/wheat_quality_exp_v1.zip
# Contains: All preprocessing + all 150 models tried

# Complete reproducibility (with data)
library.save_fullrun(
    from_run="2024-10-23_wheat_sample1",
    pipeline="0042_x9y8z7",
    name="wheat_quality_full_v1",
    include_data=True
)
# → library/trained/fullrun/wheat_quality_full_v1.zip
# Contains: Everything + wheat_sample1 training data
```

### Workflow 4: Archiving to Catalog

```python
catalog = CatalogManager("workspace/catalog")

# Archive best predictions (permanent, lightweight)
catalog.archive_predictions(
    from_export="wheat_sample1_2024-10-23_0042_x9y8z7"
)
# → catalog/archives/best_predictions/wheat_sample1_2024-10-23_0042_x9y8z7.csv

# Archive full pipeline (if important)
catalog.archive_pipeline(
    from_export="wheat_sample1_2024-10-23_0042_x9y8z7"
)
# → catalog/archives/pipeline/wheat_sample1_2024-10-23_0042_x9y8z7.zip

# Update predictions database
catalog.update_database(
    dataset="wheat_sample1",
    session="2024-10-23_wheat_sample1",
    pipeline="0042_x9y8z7",
    metrics={"rmse": 0.42, "r2": 0.90}
)
# → Updates catalog/predictions.parquet

# Now safe to delete old run!
rm -rf runs/2024-10-23_wheat_sample1/
# Catalog has permanent copies
```

### Workflow 5: Querying Best Models

```python
import polars as pl

df = pl.read_parquet("catalog/predictions.parquet")

# Best ever for wheat_sample1
best = df.filter(
    pl.col("dataset") == "wheat_sample1"
).sort("rmse").head(1)

print(f"Best model: {best['pipeline'][0]}")
print(f"From session: {best['session'][0]}")
print(f"RMSE: {best['rmse'][0]:.3f}")

# Load from catalog
predictions_path = f"catalog/archives/best_predictions/{best['dataset'][0]}_{best['session'][0].split('_')[0]}_" f"{best['pipeline'][0]}.csv"
predictions = pl.read_csv(predictions_path)
```

### Workflow 6: Cleanup

```bash
# Delete old runs (> 30 days)
find workspace/runs -name "2024-09-*" -type d -exec rm -rf {} +

# Catalog still has:
# - predictions.parquet (all metrics)
# - archives/best_predictions/*.csv (best predictions)
# - archives/pipeline/*.zip (important pipelines)

# No data lost!
```

---

## Benefits Summary

| Feature | Benefit |
|---------|---------|
| **Shallow structure** | Max 4 levels, easy navigation |
| **Sequential numbering** | Clear order, easy browsing |
| **_binaries/ folder** | Hidden, clean interface |
| **best_predictions/** | Fast access to just CSVs |
| **Permanent catalog** | Survives run deletion |
| **3 library types** | Flexible storage options |
| **Parquet database** | Fast queries, single file |
| **No broken links** | Catalog stores copies |
| **User-friendly** | Designed for daily use |

---

## Comparison with Previous Design

| Aspect | v3.0 (Old) | v3.2 (Final) | Why Changed |
|--------|------------|--------------|-------------|
| **Structure depth** | 6 levels | 3 levels | Too deep |
| **Pipeline naming** | name_hash | 0001_hash | No distinctive names |
| **Binaries location** | binaries/ | _binaries/ | Hide from user |
| **Best predictions** | In pipeline folders | Dedicated folder | Fast access |
| **Library types** | Mixed | filtered/pipeline/fullrun | Clear distinction |
| **Catalog** | Links | Copies | No broken links |
| **Dataset folders** | Per-dataset subfolders | Flat in run | Simpler |

---

## Implementation Notes

### Counter Management

```python
def get_next_pipeline_number(run_dir: Path) -> int:
    """Simple counter: count existing pipelines."""
    existing = [d for d in run_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_")]
    return len(existing) + 1

# Usage
pipeline_num = get_next_pipeline_number(run_dir)
pipeline_id = f"{pipeline_num:04d}_{hash[:6]}"
```

### Export Best Predictions

```python
def export_best_predictions(run_dir: Path, exports_dir: Path):
    """Export best pipeline predictions to dedicated folder."""
    best_pipeline = find_best_pipeline(run_dir)

    # Full export
    export_path = exports_dir / f"{dataset}_{session}_{best_pipeline.id}"
    shutil.copytree(best_pipeline.path, export_path)

    # Predictions only
    pred_export = exports_dir / "best_predictions"
    pred_export.mkdir(exist_ok=True)
    shutil.copy(
        best_pipeline.path / "predictions.csv",
        pred_export / f"{dataset}_{session}_{best_pipeline.id}.csv"
    )
```

### Archive to Catalog

```python
def archive_to_catalog(export_path: Path, catalog_dir: Path):
    """Permanent copy to catalog."""
    export_name = export_path.name

    # Archive predictions (lightweight)
    shutil.copy(
        export_path / "predictions.csv",
        catalog_dir / "archives" / "best_predictions" / f"{export_name}.csv"
    )

    # Optionally archive full pipeline
    shutil.make_archive(
        str(catalog_dir / "archives" / "pipeline" / export_name),
        "zip",
        export_path
    )
```

---

## Success Criteria

✅ **Complete when:**
1. Maximum 3 folder levels deep
2. Sequential numbering (0001, 0002, ...) working
3. _binaries/ folder with underscore prefix
4. best_predictions/ folder in exports and archives
5. Library has filtered/, pipeline/, fullrun/ subdirectories
6. Catalog stores copies (not links)
7. predictions.parquet database functional
8. Fast access to best results verified
9. User can navigate easily without documentation
10. Cleanup doesn't break anything

---

**Status**: ✅ Approved - Ready for Implementation
**Next**: Update roadmap document
````
# Two pipelines use identical StandardScaler
pipeline1: StandardScaler(with_mean=True, with_std=True)
pipeline2: StandardScaler(with_mean=True, with_std=True)

# Both reference same binary:
binaries/StandardScaler_abc123.pkl

# Storage: 1 file, not 2!
```

### 4. No Counter Management 🚫

**Concept**: Use pipeline IDs instead of sequential numbers.

**Benefits**:
- No state tracking ("what's the max counter?")
- No race conditions (parallel execution)
- Unique by content (pipeline hash)
- Human-readable (includes name)

**Anti-pattern (avoided)**:
```python
# Complex state tracking
if pipeline_exists:
    counter = max(existing_counters) + 1
else:
    counter = 1
filename = f"predictions_{counter}_model.csv"
```

**Our approach**:
```python
# Simple, stateless
pipeline_id = f"{pipeline_name}_{hash(pipeline_config)[:6]}"
filename = f"predictions.csv"  # Inside pipeline folder
```

### 5. Clear Separation of Concerns 🎭

**Concept**: runs/ (experiments) vs library/ (saved) vs catalog/ (index)

**Benefits**:
- Experiments are temporary (can delete old runs)
- Library is permanent (curated pipelines)
- Catalog provides global view
- No confusion about what to keep

**Example**:
```
runs/          # "Working directory" - experiments in progress
library/       # "Production" - proven pipelines to reuse
catalog/       # "Index" - what exists, where is it, what's best
```

### 6. Provenance Tracking 🔍

**Concept**: Clear path from session → dataset → pipeline → results

**Benefits**:
- "How did I create this prediction?" → Check pipeline folder
- "What was the best model for this dataset?" → Check catalog
- "Which session produced these results?" → Path includes session name
- Full reproducibility

**Example**:
```
Path: runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3/predictions.csv

Tells us:
- Session: wheat-quality-study (Oct 23, 2024)
- Dataset: wheat_sample1
- Pipeline: baseline_pls (ID: a1b2c3)
- Result: predictions.csv
```

---

## Workflows

### Workflow 1: Training Session

**User Goal**: Train multiple models on wheat quality dataset

**Steps**:

1. **Start session**:
```bash
nirs4all run --session wheat-quality-study \
             --datasets wheat_sample1 wheat_sample2 \
             --pipelines baseline_pls optimized_svm random_forest
```

2. **System creates**:
```
runs/2024-10-23_wheat-quality-study/
├── run_config.json
├── run.log
├── binaries/
└── datasets/
    ├── wheat_sample1/
    └── wheat_sample2/
```

3. **For each dataset × pipeline combination**:
   - Creates `datasets/<dataset>/pipelines/<pipeline_id>/`
   - Trains model
   - Saves artifacts to `binaries/` (deduplicated)
   - Saves metrics to `metrics.json`
   - Saves predictions to `predictions.csv`
   - Generates visualizations in `outputs/`

4. **At end**:
   - Aggregates results in `run_summary.json`
   - Exports best models to `exports/`
   - Updates catalog: `catalog/datasets/<dataset>/index.json`

**Result**: Clean, organized session with full provenance.

---

### Workflow 2: Reusing Pipeline

**User Goal**: Apply proven pipeline to new dataset

**Steps**:

1. **Find good pipeline**:
```bash
nirs4all catalog list-pipelines wheat_sample1 --sort test_rmse
# Shows: baseline_pls_a1b2c3 (RMSE: 0.45)
```

2. **Export to library** (optional):
```bash
nirs4all library save \
  --from runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3 \
  --name wheat_quality_baseline \
  --include-binaries
```

Creates: `library/trained/wheat_quality_baseline.zip`

3. **Use on new dataset**:
```bash
nirs4all run --session new-samples-test \
             --datasets new_wheat_samples \
             --pipeline library/trained/wheat_quality_baseline.zip
```

**Result**: Pipeline reused without duplication, new predictions created.

---

### Workflow 3: Finding Best Model

**User Goal**: "What's the best model I've trained for wheat_sample1?"

**Steps**:

1. **Check catalog**:
```bash
nirs4all catalog best wheat_sample1
```

2. **System reads**:
```
catalog/datasets/wheat_sample1/best_model_link.json
```

3. **Returns**:
```
Best Model for wheat_sample1:
  Pipeline: baseline_pls_a1b2c3
  Session: 2024-10-23_wheat-quality-study
  Test RMSE: 0.45
  Test R²: 0.89
  Path: runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3/
```

4. **User can**:
   - Load model for prediction
   - View all metrics
   - See visualizations
   - Export to library

**Result**: Instant access to best model, full context available.

---

### Workflow 4: Cleanup

**User Goal**: Clean up old experiments, keep library

**Steps**:

1. **Review old runs**:
```bash
nirs4all runs list --older-than 30-days
# Shows:
# 2024-09-15_initial-tests (45 days old, 2.1 GB)
# 2024-09-20_exploratory (38 days old, 1.8 GB)
```

2. **Check if in library**:
```bash
nirs4all library list
# Shows: wheat_quality_baseline (from 2024-10-23_wheat-quality-study)
```

3. **Safe delete** (old runs not in library):
```bash
nirs4all runs delete 2024-09-15_initial-tests --confirm
# Deletes entire run folder
# Catalog updated automatically
```

4. **OR: Archive instead**:
```bash
nirs4all runs archive 2024-09-15_initial-tests
# Creates: archives/2024-09-15_initial-tests.tar.gz
# Deletes original
```

**Result**: Workspace stays clean, important pipelines preserved in library.

---

### Workflow 5: Sharing Results

**User Goal**: Send best model to collaborator

**Steps**:

1. **Export with data** (full reproducibility):
```bash
nirs4all export \
  --session 2024-10-23_wheat-quality-study \
  --dataset wheat_sample1 \
  --pipeline baseline_pls_a1b2c3 \
  --include binaries \
  --include data \
  --output wheat_quality_model_full.zip
```

2. **OR: Export just model** (for prediction):
```bash
nirs4all export \
  --session 2024-10-23_wheat-quality-study \
  --dataset wheat_sample1 \
  --pipeline baseline_pls_a1b2c3 \
  --include binaries \
  --output wheat_quality_model.zip
```

3. **Collaborator uses**:
```bash
nirs4all predict \
  --model wheat_quality_model.zip \
  --data new_samples.csv \
  --output predictions_new.csv
```

**Result**: Easy sharing, full reproducibility optional.

---

## File Format Specifications

### run_config.json

```json
{
  "session_name": "wheat-quality-study",
  "created_at": "2024-10-23T10:30:00Z",
  "created_by": "username",
  "nirs4all_version": "0.6.0",
  "python_version": "3.11.5",

  "datasets": [
    {
      "name": "wheat_sample1",
      "source": "data/wheat_sample1.csv",
      "type": "regression",
      "loaded_at": "2024-10-23T10:30:16Z"
    }
  ],

  "pipelines": [
    {
      "name": "baseline_pls",
      "config_file": "configs/baseline_pls.json",
      "status": "pending"
    }
  ],

  "global_params": {
    "random_seed": 42,
    "n_jobs": -1,
    "verbose": 1
  },

  "description": "Testing quality prediction models for wheat samples"
}
```

### run_summary.json

```json
{
  "session_name": "wheat-quality-study",
  "status": "completed",
  "started_at": "2024-10-23T10:30:00Z",
  "completed_at": "2024-10-23T11:00:47Z",
  "duration_seconds": 1847,

  "statistics": {
    "total_pipelines": 8,
    "successful": 7,
    "failed": 1,
    "total_models_trained": 42,
    "artifact_count": 15,
    "artifact_size_mb": 3.2
  },

  "datasets": {
    "wheat_sample1": {
      "pipelines_run": 4,
      "pipelines_successful": 4,
      "best_pipeline": "baseline_pls_a1b2c3",
      "best_metric": "test_rmse",
      "best_value": 0.45
    },
    "wheat_sample2": {
      "pipelines_run": 4,
      "pipelines_successful": 3,
      "best_pipeline": "optimized_svm_x9y8z7",
      "best_metric": "test_rmse",
      "best_value": 0.52
    }
  },

  "errors": [
    {
      "pipeline": "failed_experiment_x1y2z3",
      "dataset": "wheat_sample1",
      "error": "ValueError: Invalid hyperparameter range",
      "timestamp": "2024-10-23T10:45:12Z"
    }
  ]
}
```

### pipeline.json

```json
{
  "id": "baseline_pls_a1b2c3",
  "name": "baseline_pls",
  "created_at": "2024-10-23T10:30:30Z",
  "completed_at": "2024-10-23T10:31:36Z",
  "duration_seconds": 66,
  "status": "completed",

  "dataset": {
    "name": "wheat_sample1",
    "samples": {"train": 150, "val": 30, "test": 50},
    "features": 1024,
    "targets": ["protein", "moisture"]
  },

  "steps": [
    {
      "step_index": 0,
      "operator": "StandardScaler",
      "class": "sklearn.preprocessing._data.StandardScaler",
      "params": {"with_mean": true, "with_std": true},
      "fitted": true
    },
    {
      "step_index": 1,
      "operator": "PLSRegression",
      "class": "sklearn.cross_decomposition._pls.PLSRegression",
      "params": {"n_components": 5},
      "fitted": true
    }
  ],

  "artifacts": [
    {
      "step": 0,
      "name": "scaler",
      "operator": "StandardScaler",
      "hash": "abc123",
      "path": "../../binaries/StandardScaler_abc123.pkl",
      "size_kb": 12
    },
    {
      "step": 1,
      "name": "model",
      "operator": "PLSRegression",
      "hash": "ghi789",
      "path": "../../binaries/PLSRegression_ghi789.pkl",
      "size_kb": 850
    }
  ],

  "hyperparameters": {
    "n_components": 5,
    "max_iter": 500
  }
}
```

### metrics.json

```json
{
  "train": {
    "rmse": 0.32,
    "mae": 0.25,
    "r2": 0.95,
    "mse": 0.10,
    "samples": 150
  },

  "val": {
    "rmse": 0.41,
    "mae": 0.33,
    "r2": 0.91,
    "mse": 0.17,
    "samples": 30
  },

  "test": {
    "rmse": 0.45,
    "mae": 0.36,
    "r2": 0.89,
    "mse": 0.20,
    "samples": 50
  },

  "cross_validation": {
    "folds": 5,
    "mean_rmse": 0.43,
    "std_rmse": 0.08,
    "mean_r2": 0.90,
    "std_r2": 0.04,
    "fold_results": [
      {"fold": 1, "rmse": 0.38, "r2": 0.92},
      {"fold": 2, "rmse": 0.51, "r2": 0.87},
      {"fold": 3, "rmse": 0.42, "r2": 0.91},
      {"fold": 4, "rmse": 0.39, "r2": 0.93},
      {"fold": 5, "rmse": 0.45, "r2": 0.88}
    ]
  },

  "feature_importance": {
    "top_features": [
      {"index": 512, "importance": 0.23},
      {"index": 345, "importance": 0.18},
      {"index": 678, "importance": 0.15}
    ]
  }
}
```

### catalog/datasets/<dataset>/index.json

```json
{
  "dataset_name": "wheat_sample1",
  "created_at": "2024-10-23T11:00:50Z",
  "last_updated": "2024-10-24T15:30:22Z",
  "total_predictions": 12,

  "predictions": [
    {
      "session": "2024-10-23_wheat-quality-study",
      "pipeline_id": "baseline_pls_a1b2c3",
      "pipeline_name": "baseline_pls",
      "created_at": "2024-10-23T10:31:36Z",
      "metrics": {
        "test_rmse": 0.45,
        "test_r2": 0.89,
        "test_mae": 0.36
      },
      "path": "runs/2024-10-23_wheat-quality-study/datasets/wheat_sample1/pipelines/baseline_pls_a1b2c3/",
      "in_library": true,
      "library_name": "wheat_quality_baseline"
    },
    {
      "session": "2024-10-24_exploratory-tests",
      "pipeline_id": "optimized_svm_x9y8z7",
      "pipeline_name": "optimized_svm",
      "created_at": "2024-10-24T14:22:10Z",
      "metrics": {
        "test_rmse": 0.52,
        "test_r2": 0.85,
        "test_mae": 0.41
      },
      "path": "runs/2024-10-24_exploratory-tests/datasets/wheat_sample1/pipelines/optimized_svm_x9y8z7/",
      "in_library": false
    }
  ],

  "best_models": {
    "by_rmse": {
      "pipeline_id": "baseline_pls_a1b2c3",
      "metric": "test_rmse",
      "value": 0.45,
      "session": "2024-10-23_wheat-quality-study"
    },
    "by_r2": {
      "pipeline_id": "baseline_pls_a1b2c3",
      "metric": "test_r2",
      "value": 0.89,
      "session": "2024-10-23_wheat-quality-study"
    }
  }
}
```

---

## Comparison with Current Architecture

### Current (serialization_refactoring branch)

```
results/
├── artifacts/objects/<hash[:2]>/<hash>.<ext>  # Global artifacts
├── pipelines/<uid>/manifest.yaml              # Pipeline-centric
└── datasets/<name>/index.yaml                 # Dataset index
```

**Issues**:
- Pipeline-centric (not session-centric)
- Global artifacts (deduplication across all time)
- UID-based (not human-readable)
- No session concept
- No library management

### New (workspace architecture)

```
workspace/
├── runs/<date>_<session>/                     # Session-centric
│   ├── binaries/                              # Per-run artifacts
│   └── datasets/<name>/pipelines/<id>/        # Hierarchical
├── library/                                   # Reusable pipelines
└── catalog/datasets/<name>/                   # Global index
```

**Improvements**:
- Session-centric (natural workflow)
- Per-run artifacts (simpler cleanup)
- Dataset hierarchy (clearer organization)
- Human-readable names (date + session name)
- Library for reuse
- Catalog for global view

---

## Migration Strategy

### Phase 1: Parallel Structure (Weeks 1-2)

- Keep existing `results/` structure
- Add new `workspace/` structure
- New runs go to `workspace/`
- Old runs stay in `results/`

### Phase 2: Migration Tool (Week 3)

```bash
nirs4all migrate --from results/ --to workspace/
```

**Tool converts**:
- `pipelines/<uid>/` → `runs/<session>/datasets/<dataset>/pipelines/<id>/`
- `artifacts/objects/` → `runs/<session>/binaries/`
- Creates `catalog/` from all predictions

### Phase 3: Deprecation (Week 4+)

- Mark `results/` as deprecated
- All new features use `workspace/`
- Eventually remove `results/` support

---

## Benefits Summary

| Feature | Current Architecture | New Architecture | Improvement |
|---------|---------------------|------------------|-------------|
| Organization | Pipeline-centric | Session-centric | Better workflow alignment |
| Hierarchy | Flat (UID-based) | Hierarchical (dataset/pipeline) | Clearer structure |
| Naming | UUIDs | Date + descriptive names | Human-readable |
| Deduplication | Global (all time) | Per-run | Simpler cleanup |
| Reusability | Manual copying | Library system | Easy reuse |
| Best model tracking | Manual search | Catalog with best links | Instant access |
| Cleanup | Complex (global deps) | Simple (delete runs) | Easy maintenance |
| Provenance | UID mapping | Clear path structure | Immediate understanding |
| Multi-dataset | Separate pipelines | Single session | Natural grouping |

---

## Next Steps

1. **Review & Approve**: Get user feedback on this architecture
2. **Prototype**: Implement basic structure (Phase 1)
3. **Integration**: Update PipelineRunner for new structure (Phase 2-3)
4. **Testing**: Verify with real workflows (Phase 4)
5. **Migration**: Create migration tool for existing data (Phase 5)
6. **Documentation**: Update all docs and examples (Phase 6)

---

**Status**: Ready for implementation pending user approval ✅
