# Workspace Architecture

**Version**: 4.0
**Status**: Implemented

This document describes the nirs4all workspace structure based on the DuckDB storage backend.

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Database-first** | All structured data in a single DuckDB file (`store.duckdb`) |
| **Flat artifacts** | Binary artifacts in content-addressed flat directory |
| **Export on demand** | Human-readable files produced only by explicit export operations |
| **Chain as first-class entity** | The preprocessing-to-model chain is stored, not reconstructed |
| **No folder hierarchy** | No nested `runs/` directories, no YAML manifests, no `pipeline.json` files |

---

## Directory Structure

```
workspace/
├── store.duckdb                        # All structured data (DuckDB database)
│                                        #   Tables: runs, pipelines, chains,
│                                        #   predictions, prediction_arrays,
│                                        #   artifacts, logs
│
├── artifacts/                           # Flat content-addressed binary storage
│   ├── ab/                              # 2-char shard prefix
│   │   └── abc123def456.joblib          # Fitted model/transformer
│   ├── cd/
│   │   └── cde789012345.joblib
│   └── ...
│
└── exports/                             # User-triggered exports (on demand)
    ├── model.n4a                        # Exported bundle
    ├── predictions.parquet              # Exported predictions
    └── run_summary.yaml                 # Exported run metadata
```

---

## DuckDB Schema (7 tables)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `runs` | Experiment sessions | `run_id`, `name`, `status`, `config`, `datasets` |
| `pipelines` | Expanded pipeline configs | `pipeline_id`, `run_id`, `expanded_config`, `dataset_name` |
| `chains` | Preprocessing-to-model chains | `chain_id`, `pipeline_id`, `steps`, `fold_artifacts`, `shared_artifacts` |
| `predictions` | Scalar prediction scores | `prediction_id`, `pipeline_id`, `chain_id`, `val_score`, `test_score` |
| `prediction_arrays` | Dense y_true/y_pred arrays | `prediction_id`, `y_true DOUBLE[]`, `y_pred DOUBLE[]` |
| `artifacts` | Artifact metadata & ref counts | `artifact_id`, `artifact_path`, `content_hash`, `ref_count` |
| `logs` | Structured execution logs | `log_id`, `pipeline_id`, `step_idx`, `event`, `duration_ms` |

---

## API Classes

### WorkspaceStore

Central class for all workspace persistence. Replaces the legacy ManifestManager, SimulationSaver, PipelineWriter, PredictionStorage, and ArrayRegistry.

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Run lifecycle
run_id = store.begin_run(name="experiment_1", config={...}, datasets=[...])
pipeline_id = store.begin_pipeline(run_id, name="0001_pls", ...)
chain_id = store.save_chain(pipeline_id, steps=[...], ...)
pred_id = store.save_prediction(pipeline_id, chain_id, ...)
store.complete_pipeline(pipeline_id, best_val=0.12, ...)
store.complete_run(run_id, summary={...})

# Queries
top = store.top_predictions(n=5, metric="val_score")
runs = store.list_runs(status="completed")
preds = store.query_predictions(dataset_name="wheat", partition="val")

# Chain replay (in-workspace prediction)
y_pred = store.replay_chain(chain_id="abc123", X=X_new)

# Export on demand
store.export_chain("abc123", Path("model.n4a"))
store.export_predictions_parquet(Path("results.parquet"), dataset_name="wheat")
store.export_run(run_id, Path("run_summary.yaml"))

# Cleanup
store.delete_run(run_id, delete_artifacts=True)
store.gc_artifacts()
store.vacuum()
```

### PipelineLibrary

Manage reusable pipeline templates with category support.

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
config = library.load_template("optimized_pls")
```

---

## Common Workflows

### 1. Training Session

```python
import nirs4all

result = nirs4all.run(
    pipeline=[MinMaxScaler(), PLSRegression(10)],
    dataset="sample_data/regression",
    verbose=1,
)
# All metadata, predictions, and artifacts are stored in store.duckdb + artifacts/
```

### 2. Export Best Model

```python
# Export best model as a portable bundle
result.export("model.n4a")

# Or export specific prediction's model
result.export("model.n4a", prediction_id="abc123")
```

### 3. Predict from Bundle

```python
import nirs4all

# Predict from exported bundle (no workspace needed)
preds = nirs4all.predict("model.n4a", new_data)
```

### 4. Query Predictions Across Runs

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Top models across all datasets
top = store.top_predictions(20, metric="val_score", group_by="model_class")

# Filter by dataset and partition
wheat_preds = store.query_predictions(dataset_name="wheat", partition="test")

# Export filtered predictions
store.export_predictions_parquet(Path("wheat_results.parquet"), dataset_name="wheat")
```

### 5. Delete Old Runs

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Delete a run and cascade to all related data
store.delete_run(run_id, delete_artifacts=True)

# Reclaim disk space
store.vacuum()
```

---

## See Also

- [Storage API](./storage.md) - WorkspaceStore API reference
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration
- [CLI Reference](/reference/cli) - Command-line interface
