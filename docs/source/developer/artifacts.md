# Artifacts & Storage

This guide covers the artifact storage system and workspace structure in nirs4all.

## Overview

The storage system is centered on a DuckDB-backed `WorkspaceStore` that provides:

- **Single database** -- all structured data in `store.duckdb` (runs, pipelines, chains, predictions, logs)
- **Content-addressed artifacts** -- binary deduplication via SHA-256 hashing in flat `artifacts/` directory
- **Chain-based replay** -- in-workspace prediction by replaying stored chains
- **Export on demand** -- no files written during training except `store.duckdb` and artifact binaries

## Workspace Structure

```
workspace/
├── store.duckdb                   # All structured data (7 tables)
├── artifacts/                     # Flat content-addressed binary storage
│   ├── ab/abc123def456.joblib     # Sharded by first 2 chars of hash
│   └── cd/cde789012345.joblib
├── exports/                       # User-triggered exports (on demand)
│   ├── wheat_model.n4a            # Bundle exports
│   └── results.parquet            # Prediction exports
└── library/                       # Reusable pipeline templates
    └── templates/
        └── baseline_pls.json
```

## DuckDB Tables

| Table | Purpose |
|-------|---------|
| `runs` | Experiment sessions (name, config, datasets, status) |
| `pipelines` | Individual pipeline executions within a run |
| `chains` | Preprocessing-to-model step sequences with artifact references |
| `predictions` | Per-fold, per-partition prediction scores and metadata |
| `prediction_arrays` | Dense arrays (y_true, y_pred, y_proba) |
| `artifacts` | Content-addressed artifact registry with ref_count |
| `logs` | Structured execution logs per pipeline step |

## Using the WorkspaceStore

The `WorkspaceStore` is the central class for all workspace persistence:

```python
from pathlib import Path
from nirs4all.pipeline.storage import WorkspaceStore

# Initialize (creates store.duckdb and artifacts/ if they don't exist)
store = WorkspaceStore(Path("./workspace"))

# Run lifecycle
run_id = store.begin_run("experiment_1", config={...}, datasets=[...])
pipeline_id = store.begin_pipeline(run_id, name="0001_pls", ...)
chain_id = store.save_chain(pipeline_id, steps=[...], ...)
pred_id = store.save_prediction(pipeline_id, chain_id, ...)
store.complete_pipeline(pipeline_id, best_val=0.95, ...)
store.complete_run(run_id, summary={"total_pipelines": 5})

# Queries (return polars.DataFrame)
top = store.top_predictions(n=5, metric="val_score")
preds = store.query_predictions(dataset_name="wheat", partition="val")
runs = store.list_runs(status="completed")

# Chain replay (in-workspace prediction)
y_pred = store.replay_chain(chain_id, X_new)

# Export (on demand)
store.export_chain(chain_id, Path("model.n4a"))

# Cleanup
store.delete_run(run_id)
store.gc_artifacts()
store.close()
```

## Artifact Storage

### Saving Artifacts

Artifacts are persisted through the `WorkspaceStore.save_artifact()` method:

```python
# Save a fitted model
artifact_id = store.save_artifact(
    obj=trained_model,
    operator_class="sklearn.cross_decomposition.PLSRegression",
    artifact_type="model",
    format="joblib"
)

# Save a fitted transformer
artifact_id = store.save_artifact(
    obj=fitted_scaler,
    operator_class="sklearn.preprocessing.StandardScaler",
    artifact_type="transformer",
    format="joblib"
)
```

### Content-Addressed Deduplication

When an artifact is saved, its binary content is SHA-256 hashed. If an identical artifact already exists (same content hash), the existing entry is reused and its `ref_count` incremented. This provides automatic deduplication across pipelines and runs.

```python
# Same fitted scaler saved twice -> same artifact_id returned
id1 = store.save_artifact(scaler, "StandardScaler", "transformer", "joblib")
id2 = store.save_artifact(scaler, "StandardScaler", "transformer", "joblib")
assert id1 == id2  # Content-addressed deduplication
```

### Loading Artifacts

```python
# Load by artifact ID
model = store.load_artifact(artifact_id)

# Get filesystem path (for external tools or bundle building)
path = store.get_artifact_path(artifact_id)
```

## Chain Management

A **chain** captures the complete, ordered sequence of steps (transformers and model) executed during training, with references to fitted artifacts for each fold. Chains are the unit of export and replay.

### Building Chains from Execution Traces

The `ChainBuilder` converts an `ExecutionTrace` into the chain dict format:

```python
from nirs4all.pipeline.storage import ChainBuilder

builder = ChainBuilder(trace, artifact_registry)
chain_data = builder.build()
chain_id = store.save_chain(pipeline_id=pipeline_id, **chain_data)
```

### Chain Replay

Replay a stored chain on new data to produce predictions:

```python
# In-workspace prediction
y_pred = store.replay_chain(chain_id, X_new)

# With wavelength-aware operators
y_pred = store.replay_chain(chain_id, X_new, wavelengths=wavelengths)
```

The replay loads each step's artifact, applies transformations in order, and averages predictions across fold models.

## Library Management

### Save Pipeline Templates

```python
from nirs4all.pipeline.storage import PipelineLibrary

library = PipelineLibrary(workspace_path)

# Save config-only template with category and tags
library.save_template(
    pipeline_config=pipeline_dict,
    name="baseline_pls",
    category="regression",
    description="PLS baseline with SNV preprocessing",
    tags=["nirs", "pls"],
)
```

### Load and Reuse

```python
# List templates
templates = library.list_templates(category="regression")
for t in templates:
    print(f"{t['name']}: {t['description']}")

# Load template
config = library.load_template("baseline_pls")

# Use in pipeline
runner = PipelineRunner(workspace="./workspace")
predictions = runner.run(config, new_dataset)
```

## Cleanup Utilities

```python
# Garbage-collect unreferenced artifacts (ref_count = 0)
removed = store.gc_artifacts()
print(f"Removed {removed} orphaned artifact files")

# Delete a run and all descendant data
rows_deleted = store.delete_run(run_id)
print(f"Deleted {rows_deleted} rows")

# Reclaim disk space after large deletions
store.vacuum()
```

## Export Operations

All exports are user-triggered (on demand):

```python
# Export chain as .n4a bundle (self-contained ZIP)
store.export_chain(chain_id, Path("exports/model.n4a"))

# Export pipeline configuration as JSON
store.export_pipeline_config(pipeline_id, Path("exports/config.json"))

# Export run metadata as YAML
store.export_run(run_id, Path("exports/run.yaml"))

# Export predictions as Parquet
store.export_predictions_parquet(
    Path("exports/results.parquet"),
    dataset_name="wheat",
    partition="val"
)
```

## Best Practices

1. **Let the store handle deduplication** -- identical artifacts (same content hash) automatically share the same file via `ref_count` tracking.

2. **Use chain replay for in-workspace prediction** -- `store.replay_chain()` loads artifacts and applies transformations without exporting to `.n4a` bundles.

3. **Export on demand** -- the workspace only produces `store.duckdb` and `artifacts/` during training. Use the export methods to create shareable files.

4. **Clean up periodically** -- use `gc_artifacts()` after deleting runs or pipelines to reclaim disk space from unreferenced artifact files.

5. **Close the store when done** -- call `store.close()` to release the DuckDB connection.

## See Also

- {doc}`/reference/storage` - Storage API reference
- {doc}`/reference/workspace` - Workspace architecture
- {doc}`architecture` - Pipeline architecture overview
- {doc}`/reference/pipeline_syntax` - Pipeline configuration syntax
