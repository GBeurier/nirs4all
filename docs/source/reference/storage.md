# Storage API Reference

This document provides the API reference for the nirs4all storage system, centered on the DuckDB-backed `WorkspaceStore`.

## Module: `nirs4all.pipeline.storage`

### WorkspaceStore

```python
class WorkspaceStore:
    """Database-backed workspace storage.

    Central storage facade for all workspace data: runs, pipelines, chains,
    predictions, prediction arrays, artifacts, and structured execution logs.

    The store manages two on-disk resources:
    - store.duckdb: A single DuckDB database with 7 tables
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

#### Run Lifecycle

```python
def begin_run(self, name: str, config: Any, datasets: list[dict]) -> str:
    """Create a new run and return its unique identifier.

    Args:
        name: Human-readable name for the run.
        config: Serializable run-level configuration (stored as JSON).
        datasets: List of dataset metadata dicts (name, path, hash, etc.).

    Returns:
        A unique run identifier (UUID string).
    """

def complete_run(self, run_id: str, summary: dict) -> None:
    """Mark a run as successfully completed."""

def fail_run(self, run_id: str, error: str) -> None:
    """Mark a run as failed with an error message."""
```

---

#### Pipeline Lifecycle

```python
def begin_pipeline(
    self, run_id: str, name: str, expanded_config: Any,
    generator_choices: list, dataset_name: str, dataset_hash: str,
) -> str:
    """Register a new pipeline execution under a run.

    Args:
        run_id: Parent run identifier.
        name: Pipeline name (e.g., "0001_pls_abc123").
        expanded_config: Fully resolved pipeline configuration (JSON).
        generator_choices: List of generator choices for this expansion.
        dataset_name: Name of the dataset being processed.
        dataset_hash: Content hash of the dataset.

    Returns:
        A unique pipeline identifier (UUID string).
    """

def complete_pipeline(
    self, pipeline_id: str, best_val: float, best_test: float,
    metric: str, duration_ms: int,
) -> None:
    """Mark a pipeline execution as completed with scores."""

def fail_pipeline(self, pipeline_id: str, error: str) -> None:
    """Mark a pipeline execution as failed, rolling back associated data."""
```

---

#### Chain Management

A **chain** captures the complete, ordered sequence of preprocessing steps and model that were executed during training, together with references to fitted artifacts for each fold. Chains are the unit of export and replay.

```python
def save_chain(
    self, pipeline_id: str, steps: list[dict], model_step_idx: int,
    model_class: str, preprocessings: str, fold_strategy: str,
    fold_artifacts: dict, shared_artifacts: dict,
    branch_path: list[int] | None = None,
    source_index: int | None = None,
) -> str:
    """Store a preprocessing-to-model chain.

    Args:
        pipeline_id: Parent pipeline identifier.
        steps: Ordered list of step descriptors, each containing:
            step_idx, operator_class, params, artifact_id, stateless.
        model_step_idx: Index of the model step within steps.
        model_class: Fully qualified model class name.
        preprocessings: Display string (e.g., "SNV>Detr>MinMax").
        fold_strategy: CV fold strategy ("per_fold" or "shared").
        fold_artifacts: Mapping from fold ID to model artifact ID.
        shared_artifacts: Mapping from step index to shared artifact ID.
        branch_path: Branch indices for branching pipelines.
        source_index: Source index for multi-source pipelines.

    Returns:
        A unique chain identifier (UUID string).
    """

def get_chain(self, chain_id: str) -> dict | None:
    """Retrieve a chain by its identifier."""

def get_chains_for_pipeline(self, pipeline_id: str) -> pl.DataFrame:
    """List all chains belonging to a pipeline."""
```

---

#### Prediction Storage

```python
def save_prediction(
    self, pipeline_id: str, chain_id: str, dataset_name: str,
    model_name: str, model_class: str, fold_id: str, partition: str,
    val_score: float, test_score: float, train_score: float,
    metric: str, task_type: str, n_samples: int, n_features: int,
    scores: dict, best_params: dict,
    branch_id: int | None, branch_name: str | None,
    exclusion_count: int, exclusion_rate: float,
    preprocessings: str = "",
) -> str:
    """Store a single prediction record.

    Returns:
        A unique prediction identifier (UUID string).
    """

def save_prediction_arrays(
    self, prediction_id: str, y_true: np.ndarray | None,
    y_pred: np.ndarray | None, y_proba: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> None:
    """Store dense arrays as native DOUBLE[] columns in DuckDB."""
```

---

#### Artifact Storage

Binary artifacts (fitted models, transformers) are stored in the `artifacts/` directory using content-addressed storage with automatic deduplication.

```python
def save_artifact(
    self, obj: Any, operator_class: str,
    artifact_type: str, format: str,
) -> str:
    """Persist a binary artifact with content-addressed deduplication.

    Args:
        obj: Python object to persist (e.g., fitted StandardScaler).
        operator_class: Fully qualified class name.
        artifact_type: Category ("model", "transformer", "scaler").
        format: Serialization format ("joblib", "cloudpickle").

    Returns:
        Artifact identifier (same ID returned for duplicate content).
    """

def load_artifact(self, artifact_id: str) -> Any:
    """Load a binary artifact from disk."""

def get_artifact_path(self, artifact_id: str) -> Path:
    """Return the filesystem path of a stored artifact."""
```

---

#### Structured Logging

```python
def log_step(
    self, pipeline_id: str, step_idx: int, operator_class: str,
    event: str, duration_ms: int | None = None,
    message: str | None = None, details: dict | None = None,
    level: str = "info",
) -> None:
    """Record a structured log entry for a pipeline step."""
```

---

#### Queries

```python
def get_run(self, run_id: str) -> dict | None:
    """Retrieve a single run record."""

def list_runs(
    self, status: str | None = None, dataset: str | None = None,
    limit: int = 100, offset: int = 0,
) -> pl.DataFrame:
    """List runs with optional filtering and pagination."""

def get_pipeline(self, pipeline_id: str) -> dict | None:
    """Retrieve a single pipeline record."""

def list_pipelines(
    self, run_id: str | None = None,
    dataset_name: str | None = None,
) -> pl.DataFrame:
    """List pipelines with optional filtering."""

def get_prediction(
    self, prediction_id: str, load_arrays: bool = False,
) -> dict | None:
    """Retrieve a prediction (optionally with y_true/y_pred arrays)."""

def query_predictions(
    self, dataset_name: str | None = None,
    model_class: str | None = None, partition: str | None = None,
    fold_id: str | None = None, branch_id: int | None = None,
    pipeline_id: str | None = None, run_id: str | None = None,
    limit: int | None = None, offset: int = 0,
) -> pl.DataFrame:
    """Query predictions with flexible filtering (AND semantics)."""

def top_predictions(
    self, n: int, metric: str = "val_score", ascending: bool = True,
    partition: str = "val", dataset_name: str | None = None,
    group_by: str | None = None,
) -> pl.DataFrame:
    """Return the top-N predictions ranked by a score column."""

def get_pipeline_log(self, pipeline_id: str) -> pl.DataFrame:
    """Retrieve all log entries for a pipeline."""

def get_run_log_summary(self, run_id: str) -> pl.DataFrame:
    """Aggregate log entries across all pipelines of a run."""
```

---

#### Export Operations

Exports produce files on demand from the store. No files are written during training except `store.duckdb` and artifact binaries.

```python
def export_chain(
    self, chain_id: str, output_path: Path, format: str = "n4a",
) -> Path:
    """Export a chain as a standalone .n4a bundle or .n4a.py script."""

def export_pipeline_config(
    self, pipeline_id: str, output_path: Path,
) -> Path:
    """Export a pipeline's expanded configuration as JSON."""

def export_run(self, run_id: str, output_path: Path) -> Path:
    """Export full run metadata (run + pipelines + chains) as YAML."""

def export_predictions_parquet(
    self, output_path: Path, **filters,
) -> Path:
    """Export prediction records to a Parquet file."""
```

---

#### Chain Replay

```python
def replay_chain(
    self, chain_id: str, X: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a stored chain on new data to produce predictions.

    Loads each step's artifact, applies transformations in order,
    and for the model step loads all fold models and averages.

    Args:
        chain_id: Chain to replay.
        X: Input feature matrix (n_samples x n_features).
        wavelengths: Optional wavelength array for wavelength-aware ops.

    Returns:
        Predicted values as 1-D numpy array.
    """
```

---

#### Deletion and Cleanup

```python
def delete_run(self, run_id: str, delete_artifacts: bool = True) -> int:
    """Delete a run and cascade to all descendant data.

    Returns:
        Total number of database rows deleted.
    """

def delete_prediction(self, prediction_id: str) -> bool:
    """Delete a single prediction and its arrays."""

def gc_artifacts(self) -> int:
    """Garbage-collect unreferenced artifacts (ref_count == 0)."""

def vacuum(self) -> None:
    """Reclaim unused space in the DuckDB database file."""

def close(self) -> None:
    """Close the database connection."""
```

---

### ChainBuilder

```python
from nirs4all.pipeline.storage import ChainBuilder
```

Converts an `ExecutionTrace` into a chain dict suitable for `store.save_chain()`. Used internally by the pipeline execution engine.

---

### replay_chain

```python
from nirs4all.pipeline.storage import replay_chain
```

Standalone function for replaying chains from store data. Used by the prediction API.

---

### PipelineLibrary

```python
from nirs4all.pipeline.storage import PipelineLibrary
```

Manages reusable pipeline templates with category and tag support. Templates are stored as JSON files in the workspace library directory.

---

## See Also

- [Workspace Architecture](./workspace.md) - Workspace directory structure
- [Pipeline Syntax](/reference/pipeline_syntax) - Pipeline configuration reference
