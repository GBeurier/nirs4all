# Storage API Reference

**Version**: 5.0
**Status**: Implemented

This document provides the API reference for the nirs4all storage system, which uses a hybrid DuckDB + Parquet architecture.

## Module: `nirs4all.pipeline.storage`

### WorkspaceStore

```python
from nirs4all.pipeline.storage import WorkspaceStore

class WorkspaceStore:
    """Database-backed workspace storage.

    Central storage facade for all workspace data: runs, pipelines, chains,
    predictions, artifacts, and structured execution logs.

    Manages three on-disk resources:
    - store.duckdb: DuckDB database with 7 tables (runs, pipelines, chains,
      predictions, artifacts, logs, projects)
    - arrays/: Per-dataset Parquet files for prediction arrays (via ArrayStore)
    - artifacts/: A flat, content-addressed directory for binary artifacts
    """
```

#### Constructor

```python
def __init__(self, workspace_path: Path) -> None:
    """
    Initialize the workspace store.

    Creates store.duckdb and artifacts/ directory if they don't exist.
    Schema is created automatically on first use.

    Args:
        workspace_path: Root directory of the workspace.
    """
```

---

### Run Lifecycle

```python
def begin_run(self, name: str, config: Any, datasets: list[dict]) -> str:
    """Create a new run and return its unique identifier.

    Args:
        name: Human-readable name (e.g. "protein_sweep").
        config: Serializable run-level configuration (stored as JSON).
        datasets: List of dataset metadata dicts (name, path, hash, etc.).

    Returns:
        A unique run identifier (UUID-based string).
    """

def complete_run(self, run_id: str, summary: dict) -> None:
    """Mark a run as successfully completed.

    Args:
        run_id: Identifier returned by begin_run().
        summary: Free-form summary dictionary (total pipelines, best score, etc.).
    """

def fail_run(self, run_id: str, error: str) -> None:
    """Mark a run as failed.

    Args:
        run_id: Identifier returned by begin_run().
        error: Human-readable error description or traceback excerpt.
    """
```

---

### Pipeline Lifecycle

```python
def begin_pipeline(
    self,
    run_id: str,
    name: str,
    expanded_config: Any,
    generator_choices: list,
    dataset_name: str,
    dataset_hash: str,
) -> str:
    """Register a new pipeline execution under a run.

    Args:
        run_id: Parent run identifier.
        name: Pipeline name (e.g. "0001_pls_abc123").
        expanded_config: Fully expanded pipeline configuration (after generators).
        generator_choices: List of generator choices that produced this pipeline.
        dataset_name: Name of the dataset being processed.
        dataset_hash: Content hash of the dataset at execution time.

    Returns:
        A unique pipeline identifier (UUID-based string).
    """

def complete_pipeline(
    self,
    pipeline_id: str,
    best_val: float,
    best_test: float,
    metric: str,
    duration_ms: int,
) -> None:
    """Mark a pipeline execution as successfully completed.

    Args:
        pipeline_id: Identifier returned by begin_pipeline().
        best_val: Best validation score achieved.
        best_test: Corresponding test score for the best validation model.
        metric: Name of the metric (e.g. "rmse").
        duration_ms: Total execution time in milliseconds.
    """

def fail_pipeline(self, pipeline_id: str, error: str) -> None:
    """Mark a pipeline as failed and roll back its data.

    Predictions, chains, and logs for this pipeline are removed.
    Artifacts whose ref_count drops to zero become GC candidates.

    Args:
        pipeline_id: Identifier returned by begin_pipeline().
        error: Human-readable error description.
    """
```

---

### Chain Management

```python
def save_chain(
    self,
    pipeline_id: str,
    steps: list[dict],
    model_step_idx: int,
    model_class: str,
    preprocessings: str,
    fold_strategy: str,
    fold_artifacts: dict,
    shared_artifacts: dict,
    branch_path: list[int] | None = None,
    source_index: int | None = None,
) -> str:
    """Store a preprocessing-to-model chain.

    A chain captures the complete, ordered sequence of steps (transformers
    and model) executed during training, with references to fitted artifacts
    for each fold.  Chains are the unit of export and replay.

    Args:
        pipeline_id: Parent pipeline identifier.
        steps: Ordered list of step descriptors. Each dict contains:
            {
                "step_idx": int,
                "operator_class": str,
                "params": dict,
                "artifact_id": str | None,
                "stateless": bool,
            }
        model_step_idx: Index (within steps) of the model step.
        model_class: Fully qualified class name of the model.
        preprocessings: Short display string (e.g. "SNV>Detr>MinMax").
        fold_strategy: "per_fold" or "shared".
        fold_artifacts: Mapping from fold id to artifact id.
            Example: {"fold_0": "art_abc123", "fold_1": "art_def456"}.
        shared_artifacts: Mapping from step index to artifact id.
            Example: {"0": "art_scaler_abc"}.
        branch_path: Branch indices for branching pipelines (None if non-branching).
        source_index: Source index for multi-source pipelines (None if single-source).

    Returns:
        A unique chain identifier (UUID-based string).
    """

def get_chain(self, chain_id: str) -> dict | None:
    """Retrieve a chain by its identifier.

    Returns:
        Dictionary with all chain fields (steps, model_step_idx,
        fold_artifacts, shared_artifacts, etc.), or None.
    """

def get_chains_for_pipeline(self, pipeline_id: str) -> polars.DataFrame:
    """List all chains belonging to a pipeline.

    Returns:
        polars.DataFrame with columns: chain_id, model_class,
        preprocessings, branch_path, source_index.
    """
```

---

### Prediction Storage

Prediction scalar scores are stored in DuckDB. Dense arrays (y_true, y_pred, etc.) are stored in per-dataset Parquet sidecar files via `ArrayStore`.

```python
def save_prediction(
    self,
    pipeline_id: str,
    chain_id: str,
    dataset_name: str,
    model_name: str,
    model_class: str,
    fold_id: str,
    partition: str,
    val_score: float,
    test_score: float,
    train_score: float,
    metric: str,
    task_type: str,
    n_samples: int,
    n_features: int,
    scores: dict,
    best_params: dict,
    branch_id: int | None,
    branch_name: str | None,
    exclusion_count: int,
    exclusion_rate: float,
    preprocessings: str = "",
) -> str:
    """Store a single prediction record (scalar scores in DuckDB).

    Arrays are stored separately via ArrayStore in Parquet sidecar files.

    Args:
        pipeline_id: Parent pipeline identifier.
        chain_id: Chain that produced this prediction.
        dataset_name: Name of the dataset.
        model_name: Short model name (e.g. "PLSRegression").
        model_class: Fully qualified model class name.
        fold_id: Fold identifier (e.g. "fold_0", "avg").
        partition: Data partition ("train", "val", "test").
        val_score: Validation score (primary ranking metric).
        test_score: Test score.
        train_score: Training score.
        metric: Name of the metric (e.g. "rmse", "r2").
        task_type: "regression" or "classification".
        n_samples: Number of samples in this partition.
        n_features: Number of features.
        scores: Nested dict of all computed scores per partition.
        best_params: Best hyperparameters found.
        branch_id: Branch index (0-based) or None.
        branch_name: Human-readable branch name or None.
        exclusion_count: Number of samples excluded by outlier filters.
        exclusion_rate: Fraction of samples excluded (0.0 - 1.0).
        preprocessings: Short display string for preprocessing chain.

    Returns:
        A unique prediction identifier (UUID-based string).
    """
```

---

### ArrayStore

```python
from nirs4all.pipeline.storage import ArrayStore
```

Parquet-backed storage for prediction arrays. Arrays live under `workspace/arrays/`, one `.parquet` file per dataset, with Zstd compression.

```python
class ArrayStore:
    def __init__(self, base_dir: Path) -> None:
        """Initialize. Creates arrays/ subdirectory automatically."""

    def save_batch(self, records: list[dict]) -> None:
        """Append prediction arrays for a batch of records."""

    def load(self, prediction_id: str, dataset_name: str) -> dict | None:
        """Load arrays for a single prediction."""

    def load_batch(self, prediction_ids: list[str], dataset_name: str) -> list[dict]:
        """Load arrays for multiple predictions from the same dataset."""

    def delete(self, prediction_ids: list[str]) -> None:
        """Mark predictions for deletion (tombstone)."""

    def compact(self, dataset_name: str | None = None) -> int:
        """Rewrite Parquet files, removing tombstoned rows."""
```

---

### Artifact Storage

```python
def save_artifact(
    self,
    obj: Any,
    operator_class: str,
    artifact_type: str,
    format: str,
) -> str:
    """Persist a binary artifact (fitted model or transformer).

    Uses content-addressed storage with SHA-256 hashing and
    reference counting for deduplication.

    Args:
        obj: Python object to persist (e.g. fitted StandardScaler).
        operator_class: Fully qualified class name of the operator.
        artifact_type: Category ("model", "transformer", "scaler", etc.).
        format: Serialization format ("joblib", "cloudpickle", etc.).

    Returns:
        Artifact identifier. If the content already existed, the same
        identifier is returned (deduplication).
    """

def load_artifact(self, artifact_id: str) -> Any:
    """Load a binary artifact from disk.

    Args:
        artifact_id: Identifier returned by save_artifact().

    Returns:
        Deserialized Python object.

    Raises:
        KeyError: If artifact_id is unknown.
        FileNotFoundError: If artifact file is missing.
    """

def get_artifact_path(self, artifact_id: str) -> Path:
    """Return the filesystem path of a stored artifact.

    Args:
        artifact_id: Identifier returned by save_artifact().

    Returns:
        Absolute path to the artifact file.
    """
```

---

### Structured Logging

```python
def log_step(
    self,
    pipeline_id: str,
    step_idx: int,
    operator_class: str,
    event: str,
    duration_ms: int | None = None,
    message: str | None = None,
    details: dict | None = None,
    level: str = "info",
) -> None:
    """Record a structured log entry for a pipeline step.

    Args:
        pipeline_id: Pipeline the step belongs to.
        step_idx: Zero-based step index in the pipeline.
        operator_class: Fully qualified class name of the operator.
        event: Event name ("start", "end", "skip", "warning", "error").
        duration_ms: Step execution time in milliseconds.
        message: Optional human-readable message.
        details: Optional structured details (stored as JSON).
        level: Log level ("debug", "info", "warning", "error").
    """
```

---

### Queries -- Runs

```python
def get_run(self, run_id: str) -> dict | None:
    """Retrieve a single run record.

    Returns:
        Dictionary with run_id, name, status, config, datasets, summary,
        created_at, completed_at, error. Or None if not found.
    """

def list_runs(
    self,
    status: str | None = None,
    dataset: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> polars.DataFrame:
    """List runs with optional filtering and pagination.

    Args:
        status: Filter by status ("running", "completed", "failed").
        dataset: Filter by dataset name in the datasets list.
        limit: Maximum rows.
        offset: Rows to skip.

    Returns:
        polars.DataFrame ordered by created_at descending.
    """
```

---

### Queries -- Pipelines

```python
def get_pipeline(self, pipeline_id: str) -> dict | None:
    """Retrieve a single pipeline record.

    Returns:
        Dictionary with pipeline fields (pipeline_id, run_id, name, status,
        expanded_config, generator_choices, dataset_name, dataset_hash,
        best_val, best_test, metric, duration_ms, created_at, completed_at,
        error). Or None if not found.
    """

def list_pipelines(
    self,
    run_id: str | None = None,
    dataset_name: str | None = None,
) -> polars.DataFrame:
    """List pipelines with optional filtering.

    Args:
        run_id: Filter by parent run.
        dataset_name: Filter by dataset name.

    Returns:
        polars.DataFrame ordered by created_at descending.
    """
```

---

### Queries -- Predictions

```python
def get_prediction(
    self,
    prediction_id: str,
    load_arrays: bool = False,
) -> dict | None:
    """Retrieve a single prediction record.

    Args:
        prediction_id: Prediction identifier.
        load_arrays: If True, includes y_true, y_pred, y_proba,
            sample_indices, weights as numpy arrays.

    Returns:
        Prediction dictionary or None.
    """

def query_predictions(
    self,
    dataset_name: str | None = None,
    model_class: str | None = None,
    partition: str | None = None,
    fold_id: str | None = None,
    branch_id: int | None = None,
    pipeline_id: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> polars.DataFrame:
    """Query predictions with flexible filtering.

    All filters are combined with AND semantics.

    Returns:
        polars.DataFrame (arrays not included; use get_prediction()
        with load_arrays=True for those).
    """

def top_predictions(
    self,
    n: int,
    metric: str = "val_score",
    ascending: bool = True,
    partition: str = "val",
    dataset_name: str | None = None,
    group_by: str | None = None,
) -> polars.DataFrame:
    """Return the top-N predictions ranked by a score column.

    Args:
        n: Number of top predictions.
        metric: Column to rank by ("val_score", "test_score", "train_score").
        ascending: True for error metrics (RMSE), False for R2.
        partition: Only consider this partition.
        dataset_name: Optional dataset filter.
        group_by: Optional grouping column (e.g. "model_class").

    Returns:
        polars.DataFrame with top predictions.
    """
```

---

### Queries -- Logs

```python
def get_pipeline_log(self, pipeline_id: str) -> polars.DataFrame:
    """Retrieve all log entries for a pipeline.

    Returns:
        polars.DataFrame ordered by (step_idx, timestamp).
    """

def get_run_log_summary(self, run_id: str) -> polars.DataFrame:
    """Aggregate log entries across all pipelines of a run.

    Returns:
        polars.DataFrame with per-pipeline duration, step counts,
        warning/error counts.
    """
```

---

### Export Operations

```python
def export_chain(
    self,
    chain_id: str,
    output_path: Path,
    format: str = "n4a",
) -> Path:
    """Export a chain as a standalone prediction bundle (.n4a).

    Builds a self-contained ZIP archive with manifest.json, chain.json,
    and all referenced artifact files.

    Args:
        chain_id: Chain to export.
        output_path: Destination file path.
        format: Export format ("n4a").

    Returns:
        Resolved output path.
    """

def export_pipeline_config(
    self,
    pipeline_id: str,
    output_path: Path,
) -> Path:
    """Export a pipeline's expanded configuration as JSON.

    Args:
        pipeline_id: Pipeline to export.
        output_path: Destination .json file path.

    Returns:
        Resolved output path.
    """

def export_run(
    self,
    run_id: str,
    output_path: Path,
) -> Path:
    """Export full run metadata (run + pipelines + chains) as YAML.

    Does not include binary artifacts or prediction arrays.

    Args:
        run_id: Run to export.
        output_path: Destination .yaml file path.

    Returns:
        Resolved output path.
    """

def export_predictions_parquet(
    self,
    output_path: Path,
    **filters: Any,
) -> Path:
    """Export prediction records to a Parquet file.

    Filters use the same keyword arguments as query_predictions().

    Args:
        output_path: Destination .parquet file path.
        **filters: Optional filters (dataset_name, model_class, etc.).

    Returns:
        Resolved output path.
    """
```

---

### Deletion and Cleanup

```python
def delete_run(self, run_id: str, delete_artifacts: bool = True) -> int:
    """Delete a run and all its descendant data.

    Cascades to pipelines, chains, predictions, arrays (via ArrayStore),
    and log entries.

    Args:
        run_id: Run to delete.
        delete_artifacts: Whether to remove orphaned artifact files.

    Returns:
        Total rows deleted across all tables.
    """

def delete_prediction(self, prediction_id: str) -> bool:
    """Delete a single prediction and its associated arrays.

    Returns:
        True if prediction existed and was deleted, False otherwise.
    """

def gc_artifacts(self) -> int:
    """Garbage-collect unreferenced artifacts.

    Removes artifact files from disk with ref_count = 0 and
    deletes their rows from the artifacts table.

    Returns:
        Number of artifact files removed.
    """

def vacuum(self) -> None:
    """Reclaim unused space in the DuckDB database file."""

def close(self) -> None:
    """Close the database connection. Safe to call multiple times."""
```

---

### Chain Replay

```python
def replay_chain(
    self,
    chain_id: str,
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a stored chain on new data to produce predictions.

    Loads each step's artifact, applies the transformation in order,
    and averages predictions across fold models.

    This is the primary in-workspace prediction path. For
    out-of-workspace prediction, export to .n4a first.

    Args:
        chain_id: Chain to replay.
        X: Input feature matrix (n_samples x n_features).
        wavelengths: Optional wavelength array for wavelength-aware operators.

    Returns:
        Predicted values as a 1-D numpy.ndarray of shape (n_samples,).

    Raises:
        KeyError: If the chain does not exist.
        RuntimeError: If the chain has no model step.
    """
```

---

## Module: `nirs4all.pipeline.storage.chain_builder`

### ChainBuilder

```python
from nirs4all.pipeline.storage import ChainBuilder

class ChainBuilder:
    """Converts an ExecutionTrace into the chain dict for WorkspaceStore.save_chain().

    Extracts the ordered sequence of non-skipped steps, identifies the model
    step, collects fold and shared artifact IDs, and produces a chain descriptor
    ready for DuckDB persistence.

    Args:
        trace: Finalized ExecutionTrace from TraceRecorder.
        artifact_registry: ArtifactRegistry that holds artifact records
            produced during this pipeline execution.
    """
```

#### Methods

```python
def build(self) -> dict:
    """Build the chain dict from the execution trace.

    Returns:
        Dictionary with keys matching WorkspaceStore.save_chain() parameters:
        steps, model_step_idx, model_class, preprocessings, fold_strategy,
        fold_artifacts, shared_artifacts, branch_path, source_index.
    """
```

**Usage:**
```python
from nirs4all.pipeline.storage import ChainBuilder, WorkspaceStore

builder = ChainBuilder(trace, artifact_registry)
chain_data = builder.build()
chain_id = store.save_chain(pipeline_id=pipeline_id, **chain_data)
```

---

## Module: `nirs4all.pipeline.storage.chain_replay`

### replay_chain (standalone function)

```python
from nirs4all.pipeline.storage import replay_chain

def replay_chain(
    store: WorkspaceStore,
    chain_id: str,
    X: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a chain on new data using a WorkspaceStore instance.

    Convenience wrapper that delegates to WorkspaceStore.replay_chain().

    Args:
        store: WorkspaceStore instance.
        chain_id: Chain to replay.
        X: Input feature matrix.
        wavelengths: Optional wavelength array.

    Returns:
        Predicted values as a 1-D numpy.ndarray.
    """
```

---

## DuckDB Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `runs` | Experiment sessions | run_id, name, status, config, datasets, summary |
| `pipelines` | Individual pipeline executions | pipeline_id, run_id, name, expanded_config, best_val |
| `chains` | Preprocessing-to-model step sequences | chain_id, pipeline_id, steps, fold_artifacts, shared_artifacts |
| `predictions` | Per-fold, per-partition scores | prediction_id, pipeline_id, chain_id, val_score, test_score |
| `artifacts` | Content-addressed artifact registry | artifact_id, artifact_path, content_hash, ref_count |
| `logs` | Structured execution logs per step | log_id, pipeline_id, step_idx, event, duration_ms |
| `projects` | Project grouping for runs | project_id, name, description, color |

Dense prediction arrays (y_true, y_pred, y_proba, sample_indices, weights) are stored in per-dataset Parquet sidecar files under `arrays/`, managed by `ArrayStore`. Legacy workspaces with a `prediction_arrays` DuckDB table are auto-migrated on first access.

---

## Content-Addressed Artifact Storage

Artifacts are stored in a flat directory structure using SHA-256 content hashing:

```
workspace/
  artifacts/
    ab/abc123def456.joblib    # Sharded by first 2 chars of hash
    cd/cde789012345.joblib
```

**Deduplication:** If two different chains produce an identical fitted object (same binary content), only one file is stored. The `artifacts` table tracks `ref_count` -- when a chain references an existing artifact, the count is incremented. When a chain is deleted, counts are decremented and `gc_artifacts()` removes files with `ref_count = 0`.

---

## See Also

- [Workspace Architecture](./workspace.md) - Workspace directory structure
- [Workspace Reference](/reference/workspace) - Full workspace architecture details
- [Storage Reference](/reference/storage) - Complete WorkspaceStore API reference
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration reference
