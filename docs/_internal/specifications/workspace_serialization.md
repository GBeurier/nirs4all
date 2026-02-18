# Workspace Architecture & Serialization

## Overview

nirs4all uses a hybrid DuckDB + Parquet workspace architecture. Structured metadata lives in DuckDB (`store.duckdb`), dense prediction arrays are stored in per-dataset Parquet sidecar files (`arrays/`), and binary artifacts (fitted models, transformers) are stored in a flat content-addressed directory.

## Workspace Structure

```
workspace/
├── store.duckdb                        # Structured metadata (7 DuckDB tables)
├── arrays/                              # Prediction arrays (Parquet sidecar files)
│   ├── wheat.parquet                    # All arrays for dataset "wheat"
│   └── corn.parquet                     # All arrays for dataset "corn"
├── artifacts/                           # Flat content-addressed binary storage
│   ├── ab/abc123def456.joblib
│   └── cd/cde789012345.joblib
└── exports/                             # User-triggered exports (on demand)
```

## DuckDB Schema

### runs
```sql
CREATE TABLE runs (
    run_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'running',
    config JSON,
    datasets JSON,
    summary JSON,
    error VARCHAR,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
```

### pipelines
```sql
CREATE TABLE pipelines (
    pipeline_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'running',
    expanded_config JSON,
    generator_choices JSON,
    dataset_name VARCHAR NOT NULL,
    dataset_hash VARCHAR,
    best_val DOUBLE,
    best_test DOUBLE,
    metric VARCHAR,
    duration_ms INTEGER,
    error VARCHAR,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
```

### chains
```sql
CREATE TABLE chains (
    chain_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL,
    steps JSON NOT NULL,
    model_step_idx INTEGER NOT NULL,
    model_class VARCHAR NOT NULL,
    preprocessings VARCHAR,
    fold_strategy VARCHAR,
    fold_artifacts JSON,
    shared_artifacts JSON,
    branch_path JSON,
    source_index INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### predictions
```sql
CREATE TABLE predictions (
    prediction_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL,
    chain_id VARCHAR,
    dataset_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    model_class VARCHAR NOT NULL,
    fold_id VARCHAR,
    partition VARCHAR NOT NULL,
    val_score DOUBLE,
    test_score DOUBLE,
    train_score DOUBLE,
    metric VARCHAR,
    task_type VARCHAR,
    n_samples INTEGER,
    n_features INTEGER,
    scores JSON,
    best_params JSON,
    preprocessings VARCHAR,
    branch_id INTEGER,
    branch_name VARCHAR,
    exclusion_count INTEGER DEFAULT 0,
    exclusion_rate DOUBLE DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### Parquet Array Storage (ArrayStore)

Dense prediction arrays are stored in per-dataset Parquet files under `arrays/`, managed by `ArrayStore`:

```
arrays/
├── wheat.parquet     # y_true, y_pred, y_proba, sample_indices, weights
└── corn.parquet      # Zstd-compressed, one row per prediction
```

Each Parquet file contains columns: `prediction_id`, `dataset_name`, `model_name`, `fold_id`, `partition`, `metric`, `val_score`, `task_type`, `y_true` (list), `y_pred` (list), `y_proba` (list), `sample_indices` (list), `weights` (list).

### projects
```sql
CREATE TABLE projects (
    project_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description VARCHAR,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### artifacts
```sql
CREATE TABLE artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    artifact_path VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL UNIQUE,
    operator_class VARCHAR,
    artifact_type VARCHAR,
    format VARCHAR,
    size_bytes BIGINT,
    ref_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### logs
```sql
CREATE TABLE logs (
    log_id VARCHAR PRIMARY KEY,
    pipeline_id VARCHAR NOT NULL,
    step_idx INTEGER NOT NULL,
    operator_class VARCHAR,
    event VARCHAR NOT NULL,
    duration_ms INTEGER,
    message VARCHAR,
    details JSON,
    level VARCHAR DEFAULT 'info',
    timestamp TIMESTAMPTZ DEFAULT now()
);
```

## Key Indexes

```sql
CREATE INDEX idx_predictions_val_score ON predictions(val_score);
CREATE INDEX idx_predictions_dataset ON predictions(dataset_name);
CREATE INDEX idx_predictions_pipeline ON predictions(pipeline_id);
CREATE INDEX idx_pipelines_run ON pipelines(run_id);
CREATE INDEX idx_chains_pipeline ON chains(pipeline_id);
CREATE INDEX idx_logs_pipeline ON logs(pipeline_id);
CREATE INDEX idx_artifacts_hash ON artifacts(content_hash);
```

## Serialization System

### Content-Addressed Artifact Storage

Binary artifacts are stored using content-addressed storage with SHA-256 hashing:

1. Serialize object to bytes (joblib or pickle)
2. Compute SHA-256 hash of the bytes
3. Check if artifact with same hash exists in store
4. If exists: increment ref_count, return existing artifact_id
5. If new: write to `artifacts/{hash[:2]}/{hash}.{ext}`, insert row

### Supported Formats

| Type | Format | Extension |
|------|--------|-----------|
| scikit-learn models | joblib | `.joblib` |
| Generic Python objects | pickle | `.pkl` |
| Cloud-pickled objects | cloudpickle | `.pkl` |

### Deduplication

Identical objects share the same artifact file. The `ref_count` column tracks how many chains reference each artifact. When a chain is deleted, ref_counts are decremented. `gc_artifacts()` removes files with `ref_count == 0`.

## Chain Structure

A chain captures the complete preprocessing-to-model path:

```json
{
    "chain_id": "abc123",
    "steps": [
        {"step_idx": 0, "operator_class": "MinMaxScaler", "params": {}, "artifact_id": "art_001", "stateless": false},
        {"step_idx": 1, "operator_class": "SNV", "params": {}, "artifact_id": null, "stateless": true},
        {"step_idx": 2, "operator_class": "PLSRegression", "params": {"n_components": 10}, "artifact_id": null, "stateless": false}
    ],
    "model_step_idx": 2,
    "fold_artifacts": {"fold_0": "art_002", "fold_1": "art_003"},
    "shared_artifacts": {"0": "art_001"}
}
```

## Export Formats

Exports are produced on demand from the store:

| Operation | Method | Output |
|-----------|--------|--------|
| Export chain | `store.export_chain(chain_id, path)` | `.n4a` ZIP bundle |
| Export config | `store.export_pipeline_config(pipeline_id, path)` | `.json` file |
| Export run | `store.export_run(run_id, path)` | `.yaml` file |
| Export predictions | `store.export_predictions_parquet(path, **filters)` | `.parquet` file |

## API: WorkspaceStore

See [Storage API Reference](../source/reference/storage.md) for the complete API.

### Key Methods

```python
from nirs4all.pipeline.storage import WorkspaceStore, ArrayStore

store = WorkspaceStore(workspace_path)

# Run lifecycle
run_id = store.begin_run(name, config, datasets)
pipeline_id = store.begin_pipeline(run_id, name, config, ...)
chain_id = store.save_chain(pipeline_id, steps, ...)
pred_id = store.save_prediction(pipeline_id, chain_id, ...)
store.complete_pipeline(pipeline_id, best_val, best_test, metric, duration_ms)
store.complete_run(run_id, summary)

# Prediction arrays (via ArrayStore)
array_store = ArrayStore(workspace_path / "arrays")
array_store.save_batch([{"prediction_id": pred_id, "dataset_name": "wheat", ...}])

# Queries (return polars.DataFrame)
store.list_runs(status="completed")
store.top_predictions(n=10, metric="val_score")
store.query_predictions(dataset_name="wheat", partition="val")

# Chain replay
y_pred = store.replay_chain(chain_id, X_new)

# Export
store.export_chain(chain_id, Path("model.n4a"))

# Cleanup
store.delete_run(run_id)
store.gc_artifacts()
store.vacuum()
store.close()
```
