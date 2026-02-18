# Workspace Architecture

**Version**: 5.0
**Status**: Implemented

This document describes the nirs4all workspace directory structure, which uses a hybrid DuckDB + Parquet storage backend.

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Hybrid storage** | Structured metadata in DuckDB, dense arrays in Parquet sidecar files |
| **Content-addressed artifacts** | Binary deduplication via SHA-256 hashing in flat `artifacts/` directory |
| **Export on demand** | No export files written during training — only on explicit `export()` calls |
| **Zero-copy queries** | DuckDB returns Polars DataFrames via Arrow transfer |
| **Library flexibility** | Pipeline templates stored as JSON with category and tag support |

---

## Directory Structure

```
workspace/
├── store.duckdb                        # Structured metadata (7 DuckDB tables)
├── arrays/                              # Prediction arrays (Parquet sidecar files)
│   ├── wheat.parquet                    # All arrays for dataset "wheat"
│   └── corn.parquet                     # All arrays for dataset "corn"
├── artifacts/                          # Flat content-addressed binary storage
│   ├── ab/abc123def456.joblib          # Sharded by first 2 chars of hash
│   └── cd/cde789012345.joblib
├── exports/                            # User-triggered exports (on demand)
│   ├── wheat_model.n4a                 # Bundle exports
│   └── results.parquet                 # Prediction exports
└── library/                            # Reusable pipeline templates
    └── templates/
        ├── baseline_pls.json
        └── optimized_svm.json
```

---

## DuckDB Tables

| Table | Purpose |
|-------|---------|
| `runs` | Experiment sessions (name, config, datasets, status) |
| `pipelines` | Individual pipeline executions within a run |
| `chains` | Preprocessing-to-model step sequences with artifact references |
| `predictions` | Per-fold, per-partition prediction scores and metadata |
| `artifacts` | Content-addressed artifact registry with ref_count |
| `logs` | Structured execution logs per pipeline step |
| `projects` | Project grouping for runs |

## Parquet Array Storage

Dense prediction arrays (y_true, y_pred, y_proba, sample_indices, weights) are stored in per-dataset Parquet files under `arrays/`, managed by `ArrayStore`. Each Parquet file uses Zstd compression and embeds lightweight metadata columns for self-describing portability.

---

## API Classes

### WorkspaceStore

Central class for all workspace persistence.

```python
from nirs4all.pipeline.storage import WorkspaceStore

store = WorkspaceStore(workspace_path)

# Run lifecycle
run_id = store.begin_run(name="experiment_1", config={...}, datasets=[...])
pipeline_id = store.begin_pipeline(run_id, name="0001_pls", ...)
chain_id = store.save_chain(pipeline_id, steps=[...], ...)
pred_id = store.save_prediction(pipeline_id, chain_id, ...)
store.complete_pipeline(pipeline_id, best_val=0.95, ...)
store.complete_run(run_id, summary={...})

# Queries (return polars.DataFrame)
top = store.top_predictions(n=5, metric="val_score")
preds = store.query_predictions(dataset_name="wheat", partition="val")
runs = store.list_runs(status="completed")

# Chain replay
y_pred = store.replay_chain(chain_id, X_new)

# Export
store.export_chain(chain_id, Path("model.n4a"))

# Cleanup
store.delete_run(run_id)
store.gc_artifacts()
store.close()
```

### PipelineLibrary

Manage reusable pipeline templates with category support.

```python
from nirs4all.pipeline.storage import PipelineLibrary

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

## See Also

- [Workspace Reference](/reference/workspace) - Full workspace architecture details
- [Storage API Reference](/reference/storage) - Complete WorkspaceStore API
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration
