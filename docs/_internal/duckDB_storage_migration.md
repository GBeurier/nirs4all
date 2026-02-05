# DuckDB Storage Refactoring — Detailed Roadmap

**Decision**: Proposal 2 (DuckDB Columnar Store)
**Scope**: Replace file-hierarchy storage with DuckDB database-first architecture
**Strategy**: Clean cut — implement new, delete old, no backward compatibility

---

## Objectives & Goals

This refactoring replaces the current file-hierarchy storage (YAML manifests, split Parquet files, nested directory trees) with a single DuckDB database. The motivations are:

### Easier Access to Predictions & Metadata

The current system scatters prediction data across per-dataset `.meta.parquet` files, requiring filesystem globbing and multi-file joins to answer simple questions like "what are my top 10 models across all datasets?" A single DuckDB store makes all predictions queryable with a single SQL/Polars call, with full filtering on any column (dataset, model, partition, branch, fold, metric).

### Simpler Workspace Structure

The current workspace creates deep directory trees: `workspace/runs/<dataset>/0001_xxx/manifest.yaml`, `workspace/binaries/<dataset>/`, per-dataset Parquet files, etc. The new structure is flat: `store.duckdb` + `artifacts/`. No folders to manage, no naming conventions to maintain, no directory scanning to discover runs.

### Ease of Management

Run deletion, prediction cleanup, and artifact garbage collection become database operations with cascade semantics. No more orphaned files, no more partial cleanup after interrupted runs, no more manual directory pruning.

### Easier Export, Sharing, Transfer & Retraining

Exporting a model, a set of predictions, or a full run becomes an on-demand operation from the store rather than copying directory trees. Partial exports (e.g., "export only predictions for dataset X with RMSE < 0.5") become trivial filters. The chain-as-first-class-entity design means any trained model can be exported, replayed, or retrained from its chain ID without reconstructing the preprocessing path.

### Performance

DuckDB provides columnar storage with vectorized query execution. Filtering 100K+ predictions by any combination of fields is orders of magnitude faster than loading and filtering Parquet files. Insertion is transactional — no partial writes on failure. Array storage uses native `DOUBLE[]` columns instead of serialized blobs, enabling zero-copy Arrow transfer to Polars.

### Foundation for Future Features

A structured store enables features that are impractical with file hierarchies: prediction retention policies, run comparison dashboards, cross-dataset model rankings, structured execution logs, and workspace-level analytics.

---

## Guiding Principles

1. **Clean cut**: No migration tooling, no backward compatibility layer, no dual-backend switches. The old system is replaced and deleted.
2. **API-first**: Design the public `WorkspaceStore` API before implementing internals. Consumers adapt early.
3. **Chain as first-class entity**: The preprocessing→model chain is stored, not reconstructed.
4. **Files are exports**: No folder hierarchy. `store.duckdb` + `artifacts/` directory only.
5. **Delete as you go**: Legacy code is deleted in the same phase that replaces it. No dead code accumulates.

---

## Phase 0 — API Surface Design

**Goal**: Define the complete public interface of `WorkspaceStore` before writing any implementation. This is the contract that all consumers (PipelineRunner, Predictions, RunResult, webapp, examples) will program against.

### Files to Create

**`nirs4all/pipeline/storage/workspace_store.py`** — The central API class:

```python
class WorkspaceStore:
    """Database-backed workspace storage.

    Replaces: ManifestManager, SimulationSaver, PipelineWriter,
              PredictionStorage, ArrayRegistry, WorkspaceExporter.

    All metadata, configs, logs, chains, and predictions are stored in the database.
    Binary artifacts are stored in a flat content-addressed directory.
    Files are produced only by explicit export operations.
    """

    def __init__(self, workspace_path: Path): ...

    # --- Run lifecycle ---
    def begin_run(self, name: str, config: Any, datasets: list[dict]) -> str: ...
    def complete_run(self, run_id: str, summary: dict) -> None: ...
    def fail_run(self, run_id: str, error: str) -> None: ...

    # --- Pipeline lifecycle ---
    def begin_pipeline(self, run_id: str, name: str, expanded_config: Any,
                       generator_choices: list, dataset_name: str,
                       dataset_hash: str) -> str: ...
    def complete_pipeline(self, pipeline_id: str, best_val: float,
                          best_test: float, metric: str, duration_ms: int) -> None: ...
    def fail_pipeline(self, pipeline_id: str, error: str) -> None: ...

    # --- Chain management ---
    def save_chain(self, pipeline_id: str, steps: list[dict],
                   model_step_idx: int, model_class: str,
                   preprocessings: str, fold_strategy: str,
                   fold_artifacts: dict, shared_artifacts: dict,
                   branch_path: list[int] | None = None,
                   source_index: int | None = None) -> str: ...
    def get_chain(self, chain_id: str) -> dict: ...
    def get_chains_for_pipeline(self, pipeline_id: str) -> pl.DataFrame: ...

    # --- Prediction storage ---
    def save_prediction(self, pipeline_id: str, chain_id: str,
                        dataset_name: str, model_name: str, model_class: str,
                        fold_id: str, partition: str,
                        val_score: float, test_score: float, train_score: float,
                        metric: str, task_type: str,
                        n_samples: int, n_features: int,
                        scores: dict, best_params: dict,
                        branch_id: int | None, branch_name: str | None,
                        exclusion_count: int, exclusion_rate: float) -> str: ...
    def save_prediction_arrays(self, prediction_id: str,
                               y_true: np.ndarray | None,
                               y_pred: np.ndarray | None,
                               y_proba: np.ndarray | None,
                               sample_indices: np.ndarray | None,
                               weights: np.ndarray | None) -> None: ...

    # --- Artifact storage ---
    def save_artifact(self, obj: Any, operator_class: str,
                      artifact_type: str, format: str) -> str: ...
    def load_artifact(self, artifact_id: str) -> Any: ...
    def get_artifact_path(self, artifact_id: str) -> Path: ...

    # --- Structured logging ---
    def log_step(self, pipeline_id: str, step_idx: int,
                 operator_class: str, event: str,
                 duration_ms: int | None = None,
                 message: str | None = None,
                 details: dict | None = None,
                 level: str = "info") -> None: ...

    # --- Queries ---
    def get_run(self, run_id: str) -> dict | None: ...
    def list_runs(self, status: str | None = None,
                  dataset: str | None = None,
                  limit: int = 100, offset: int = 0) -> pl.DataFrame: ...
    def get_pipeline(self, pipeline_id: str) -> dict | None: ...
    def list_pipelines(self, run_id: str | None = None,
                       dataset_name: str | None = None) -> pl.DataFrame: ...
    def get_prediction(self, prediction_id: str,
                       load_arrays: bool = False) -> dict | None: ...
    def query_predictions(self, dataset_name: str | None = None,
                          model_class: str | None = None,
                          partition: str | None = None,
                          fold_id: str | None = None,
                          branch_id: int | None = None,
                          limit: int | None = None,
                          offset: int = 0) -> pl.DataFrame: ...
    def top_predictions(self, n: int, metric: str = "val_score",
                        ascending: bool = True,
                        partition: str = "val",
                        dataset_name: str | None = None,
                        group_by: str | None = None) -> pl.DataFrame: ...
    def get_pipeline_log(self, pipeline_id: str) -> pl.DataFrame: ...
    def get_run_log_summary(self, run_id: str) -> pl.DataFrame: ...

    # --- Export operations (produce files on demand) ---
    def export_chain(self, chain_id: str, output_path: Path,
                     format: str = "n4a") -> Path: ...
    def export_pipeline_config(self, pipeline_id: str,
                               output_path: Path) -> Path: ...
    def export_run(self, run_id: str, output_path: Path) -> Path: ...
    def export_predictions_parquet(self, output_path: Path,
                                   **filters) -> Path: ...

    # --- Deletion & cleanup ---
    def delete_run(self, run_id: str, delete_artifacts: bool = True) -> int: ...
    def delete_prediction(self, prediction_id: str) -> bool: ...
    def gc_artifacts(self) -> int: ...
    def vacuum(self) -> None: ...
```

**`nirs4all/pipeline/storage/store_protocol.py`** — Protocol for backend abstraction (if we ever want to swap DuckDB for something else):

```python
class WorkspaceStoreProtocol(Protocol):
    """Minimal protocol for workspace storage backends."""
    def begin_run(self, name: str, config: Any, datasets: list[dict]) -> str: ...
    def save_prediction(self, ...) -> str: ...
    def save_artifact(self, obj: Any, ...) -> str: ...
    def top_predictions(self, n: int, ...) -> pl.DataFrame: ...
    def export_chain(self, chain_id: str, output_path: Path, ...) -> Path: ...
```

### Files to Create for Tests

**`tests/unit/pipeline/storage/test_workspace_store_api.py`** — API contract tests (interface-level, can run against any backend).

### Deliverable

A reviewed, approved API surface. No implementation yet. All consumers know what methods they will call.

### Dependencies

None. This is the first phase.

---

## Phase 1 — DuckDB Core Implementation

**Goal**: Implement `WorkspaceStore` backed by DuckDB. All CRUD operations, schema creation, queries. Fully tested in isolation.

### Files to Create

| File | Purpose |
|------|---------|
| `nirs4all/pipeline/storage/workspace_store.py` | Full implementation of WorkspaceStore |
| `nirs4all/pipeline/storage/store_schema.py` | DuckDB schema DDL |
| `nirs4all/pipeline/storage/store_queries.py` | Reusable query builders |
| `tests/unit/pipeline/storage/test_workspace_store.py` | Unit tests: CRUD, queries, edge cases |
| `tests/unit/pipeline/storage/test_store_schema.py` | Schema creation, validation |

### Schema (DuckDB DDL)

Defined in `store_schema.py`:

```sql
-- 7 tables, no folder hierarchy
CREATE TABLE IF NOT EXISTS runs (...)
CREATE TABLE IF NOT EXISTS pipelines (...)
CREATE TABLE IF NOT EXISTS chains (...)
CREATE TABLE IF NOT EXISTS predictions (...)
CREATE TABLE IF NOT EXISTS prediction_arrays (...)
CREATE TABLE IF NOT EXISTS artifacts (...)
CREATE TABLE IF NOT EXISTS logs (...)
-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_val_score ON predictions(val_score);
-- etc.
```

(Full DDL as specified in Proposal 2 from the design document.)

### Key Implementation Details

**Connection management:**
- Single DuckDB connection per `WorkspaceStore` instance
- `store.duckdb` file in workspace root
- WAL mode for concurrent read/write
- Foreign keys enabled

**Polars integration:**
- All query methods return `pl.DataFrame` via DuckDB's `.pl()` zero-copy Arrow transfer
- `save_prediction_arrays()` stores arrays as native `DOUBLE[]` (no BLOB serialization)

**Artifact storage:**
- Binary files in `workspace/artifacts/{hash[:2]}/{hash}.{ext}`
- `artifacts` table stores metadata only (path, hash, type, format, ref_count)
- Content-addressed deduplication: same binary = same `artifact_id`
- `ref_count` incremented on reuse, decremented on chain deletion
- `gc_artifacts()` removes files with `ref_count == 0`

**Transaction model:**
- `begin_run()` starts implicit DuckDB transaction
- `complete_pipeline()` commits after each pipeline
- `fail_pipeline()` rolls back current pipeline's data
- `complete_run()` / `fail_run()` commits final state

### Test Requirements

| Test | What it validates |
|------|-------------------|
| `test_run_lifecycle` | begin → complete → query |
| `test_run_failure` | begin → fail → verify no leaked data |
| `test_pipeline_crud` | Create, query, complete, fail |
| `test_chain_save_load` | Save chain with fold_artifacts, shared_artifacts; load by ID |
| `test_prediction_save_query` | Save 100 predictions; query by dataset, model, partition |
| `test_prediction_arrays` | Save y_true/y_pred as DOUBLE[]; load; verify numpy roundtrip |
| `test_top_predictions` | Top-N with group_by, ascending/descending |
| `test_artifact_dedup` | Save same binary twice → same artifact_id, ref_count=2 |
| `test_artifact_gc` | Delete chain → ref_count decrements → gc removes file |
| `test_log_step` | Log events; query per pipeline; aggregate per run |
| `test_delete_cascade` | Delete run → pipelines, chains, predictions, arrays deleted |
| `test_export_chain_n4a` | Export chain → valid .n4a ZIP with manifest + artifacts |
| `test_export_pipeline_config` | Export pipeline config → valid JSON |
| `test_export_run` | Export run → valid YAML with all pipelines |
| `test_concurrent_read_write` | Two threads: one writing, one reading → no corruption |
| `test_empty_workspace` | All queries return empty DataFrames on fresh store |
| `test_schema_creation` | Create store from scratch; verify all tables exist |

### Dependencies

Phase 0 (API approved).

---

## Phase 2 — Replace Pipeline Execution Storage

**Goal**: Replace `ManifestManager`, `SimulationSaver`, `PipelineWriter` with `WorkspaceStore` calls in the execution engine. Delete all replaced code immediately.

### Files to Modify

| File | Changes |
|------|---------|
| `pipeline/execution/orchestrator.py` | Replace `ManifestManager` + `SimulationSaver` with `WorkspaceStore` calls |
| `pipeline/execution/executor.py` | Replace `self.saver.persist_artifact()` and `self.manifest_manager.append_artifacts()` with `store.save_artifact()` |
| `pipeline/execution/builder.py` | Construct `WorkspaceStore` instead of legacy components |
| `pipeline/runner.py` | Construct `WorkspaceStore`; remove legacy component construction |
| `pipeline/config/context.py` | Add `WorkspaceStore` reference to `RuntimeContext` |
| `api/run.py` | Use `WorkspaceStore` (no backend flag) |

### Detailed Change Map for `PipelineOrchestrator.execute()`

Current write points → new equivalents:

| Current Write | Current Code | New DuckDB Equivalent |
|---------------|-------------|----------------------|
| Create `workspace/runs/<dataset>/` | `current_run_dir.mkdir()` | Not needed (no directory) |
| Create `ManifestManager` | `ManifestManager(current_run_dir)` | `store.begin_run(...)` already called |
| Create pipeline directory | `manifest_manager.create_pipeline(...)` | `store.begin_pipeline(...)` |
| Save `pipeline.json` | `saver.save_json("pipeline.json", steps)` | Stored in `pipelines.expanded_config` column |
| Persist artifacts | `saver.persist_artifact(...)` | `store.save_artifact(...)` |
| Append artifacts to manifest | `manifest_manager.append_artifacts(...)` | Part of `store.save_chain(...)` at end |
| Save execution trace | `manifest_manager.save_execution_trace(...)` | Chain built from trace via `store.save_chain(...)` |
| Save outputs (charts) | `saver.save_output(...)` | `store.save_output(...)` → `workspace/exports/charts/` (if enabled) |
| Save predictions to Parquet | `predictions.save_to_file(path)` | `store.save_prediction(...)` per prediction |
| Save best CSV | `Predictions.save_predictions_to_csv(...)` | Export-on-demand: `store.export_predictions_parquet(...)` |

### Chain Construction

The chain is built from the `TraceRecorder` output at the end of each pipeline execution:

```python
# After executor.execute() completes:
trace = trace_recorder.finalize(...)

# Convert trace to chain (new logic):
chain_steps = []
for step in trace.execution_steps:
    if step.execution_mode != StepExecutionMode.SKIP:
        chain_steps.append({
            "step_idx": step.step_index,
            "operator_class": step.operator_class,
            "params": step.operator_params,
            "artifact_id": step.artifacts.primary_artifact_id,
            "stateless": not step.artifacts.artifact_ids,
        })

chain_id = store.save_chain(
    pipeline_id=pipeline_id,
    steps=chain_steps,
    model_step_idx=trace.model_step_index,
    model_class=trace.model_class,
    preprocessings=trace.preprocessing_chain,
    fold_strategy="per_fold",
    fold_artifacts=trace.fold_artifact_ids,
    shared_artifacts=trace.shared_artifact_ids,
)
```

### Files to Create

| File | Purpose |
|------|---------|
| `pipeline/storage/chain_builder.py` | Convert `ExecutionTrace` → chain dict for `store.save_chain()` |
| `tests/integration/storage/test_duckdb_pipeline.py` | Run full pipeline with WorkspaceStore, verify store contents |

### Files to Delete

| File | Reason |
|------|--------|
| `pipeline/storage/manifest_manager.py` | Replaced by `WorkspaceStore` |
| `pipeline/storage/io.py` (SimulationSaver) | Replaced by `WorkspaceStore` |
| `pipeline/storage/io_writer.py` (PipelineWriter) | Replaced by `WorkspaceStore` |
| `tests/unit/pipeline/storage/test_manifest_manager.py` | No more ManifestManager |
| `tests/unit/pipeline/storage/test_manifest_v2.py` | No more manifests |

### Test Requirements

| Test | What it validates |
|------|-------------------|
| `test_basic_pipeline` | `nirs4all.run(pipeline, dataset)` produces valid RunResult |
| `test_no_file_hierarchy` | After run: no `runs/` folder, no `manifest.yaml`, no `pipeline.json` |
| `test_artifacts_flat` | Artifacts saved in `workspace/artifacts/` flat structure |
| `test_chain_from_trace` | Chain built from ExecutionTrace has correct steps, artifacts, model_step_idx |
| `test_generator_pipelines` | Generator-expanded pipelines → N pipeline rows in store with distinct configs |
| `test_branching_pipeline` | Branching pipeline → chains per branch with correct branch_path |
| `test_stacking_pipeline` | Stacking → chains reference meta-model and base model artifacts |

### Dependencies

Phase 1 (WorkspaceStore implementation).

---

## Phase 3 — Predictions System Rewrite

**Goal**: Rewrite the `Predictions` facade to be store-backed only. Delete `PredictionStorage`, `ArrayRegistry`, `PredictionSerializer`, `PredictionIndexer`, `PredictionRanker`, `PartitionAggregator`, `CatalogQueryEngine`, and `PREDICTION_SCHEMA`.

### Strategy

The `Predictions` class is the main facade used by:
- `RunResult` (via `.predictions` property)
- `PipelineOrchestrator` (accumulates predictions during run)
- `PredictionAnalyzer` (visualization)
- Webapp backend (predictions endpoint)

The class is rewritten to work exclusively with `WorkspaceStore`. No dual-backend logic.

### Predictions Rewrite

```python
class Predictions:
    def __init__(self, store: WorkspaceStore):
        self._store = store
        self._buffer: list[dict] = []  # In-memory during execution

    def add_prediction(self, **kwargs) -> str:
        pred_id = str(uuid4())
        self._buffer.append({"prediction_id": pred_id, **kwargs})
        return pred_id

    def flush(self, pipeline_id: str):
        """Flush buffer to store (called at pipeline end)."""
        for pred in self._buffer:
            self._store.save_prediction(**pred)
        self._buffer.clear()

    def top(self, n, **kwargs):
        return self._store.top_predictions(n, **kwargs)

    def filter(self, **kwargs):
        return self._store.query_predictions(**kwargs)
```

### RunResult Rewrite

```python
class RunResult:
    def __init__(self, predictions, per_dataset, store):
        self._store = store

    def export(self, path, prediction_id=None, format="n4a"):
        if prediction_id is None:
            best = self._store.top_predictions(1).row(0, named=True)
            prediction_id = best["prediction_id"]
        pred = self._store.get_prediction(prediction_id)
        return self._store.export_chain(pred["chain_id"], Path(path), format)
```

### Files to Modify

| File | Changes |
|------|---------|
| `data/predictions.py` | Rewrite: store-backed only, remove all Parquet/legacy logic |
| `api/result.py` | Rewrite: `RunResult` uses store only, remove runner-based export path |

### Files to Delete

| File | Reason |
|------|--------|
| `data/_predictions/storage.py` (PredictionStorage) | Replaced by store predictions table |
| `data/_predictions/serializer.py` (PredictionSerializer) | No longer needed |
| `data/_predictions/indexer.py` (PredictionIndexer) | Replaced by store queries |
| `data/_predictions/ranker.py` (PredictionRanker) | Replaced by store `top_predictions()` |
| `data/_predictions/aggregator.py` (PartitionAggregator) | Replaced by store queries |
| `data/_predictions/query.py` (CatalogQueryEngine) | Replaced by store queries |
| `data/_predictions/array_registry.py` (ArrayRegistry) | Replaced by store `prediction_arrays` table |
| `data/_predictions/schemas.py` (PREDICTION_SCHEMA) | Replaced by `store_schema.py` |
| `tests/unit/data/predictions/test_array_registry.py` | No more ArrayRegistry |

### Files to Create

| File | Purpose |
|------|---------|
| `tests/unit/data/test_predictions_store.py` | Test Predictions in store-backed mode |

### Test Requirements

| Test | What it validates |
|------|-------------------|
| `test_predictions_buffer_flush` | Add 100 predictions → flush → verify all in store |
| `test_predictions_top` | `top(5)` returns correct ranking from store |
| `test_predictions_filter` | `filter(dataset="wheat")` → correct results from store |
| `test_predictions_arrays_roundtrip` | Save y_true/y_pred arrays → load → numpy equality |
| `test_result_export` | `result.export("model.n4a")` produces valid bundle from store chain |
| `test_result_best_score` | `result.best_rmse` correct from store-backed predictions |

### Dependencies

Phase 2 (store wired into execution engine).

---

## Phase 4 — Export & Import System

**Goal**: Complete the export-on-demand system. Replace `WorkspaceExporter` and `LibraryManager` with store-based exports. Delete replaced code.

### Export Operations

| Operation | Method | Produces |
|-----------|--------|----------|
| Export best model | `store.export_chain(chain_id, "model.n4a")` | .n4a ZIP bundle |
| Export pipeline config | `store.export_pipeline_config(pipeline_id, "config.json")` | JSON file |
| Export run metadata | `store.export_run(run_id, "run.yaml")` | YAML file |
| Export predictions | `store.export_predictions_parquet(path, dataset="wheat")` | Parquet file |
| Export portable script | `store.export_chain(chain_id, "model.n4a.py", format="n4a.py")` | Python script |

### Import / Load Operations

The load path is unchanged: `.n4a` bundles are self-contained ZIP files. `BundleLoader` reads them independently of the workspace store.

```python
# These stay the same:
preds = nirs4all.predict("model.n4a", data)      # BundleLoader unchanged
session = nirs4all.load_session("config.json")    # Config loading unchanged
```

### Chain Replay

`store.replay_chain(chain_id, X)` is the new primary prediction path for stored models:

```python
def replay_chain(self, chain_id: str, X: np.ndarray) -> np.ndarray:
    chain = self.get_chain(chain_id)
    steps = json.loads(chain["steps"])
    fold_artifacts = json.loads(chain["fold_artifacts"])
    shared_artifacts = json.loads(chain["shared_artifacts"])

    # Load and apply each step in order
    X_current = X.copy()
    for step in steps:
        idx = step["step_idx"]
        if idx == chain["model_step_idx"]:
            # Model step: load all fold models, predict, average
            fold_preds = []
            for fold_id, artifact_id in fold_artifacts.items():
                model = self.load_artifact(artifact_id)
                fold_preds.append(model.predict(X_current))
            return np.mean(fold_preds, axis=0)
        elif idx in shared_artifacts:
            transformer = self.load_artifact(shared_artifacts[str(idx)])
            X_current = transformer.transform(X_current)
        elif step.get("stateless"):
            cls = _import_class(step["operator_class"])
            X_current = cls(**step["params"]).transform(X_current)
    raise RuntimeError("Chain has no model step")
```

### Files to Modify

| File | Changes |
|------|---------|
| `pipeline/bundle/generator.py` | Store-based export path (load chain from store, build bundle) |
| `workspace/library_manager.py` | Rewrite to use store (templates saved as runs with special tag) |
| `api/predict.py` | Add `chain_id` as prediction source (alongside .n4a, folder, dict) |
| `pipeline/resolver.py` | Rewrite to store-only resolution (chain_id → ResolvedPrediction) |

### Files to Delete

| File | Reason |
|------|--------|
| `pipeline/storage/io_exporter.py` (WorkspaceExporter) | Replaced by `WorkspaceStore.export_*()` |
| `pipeline/storage/io_resolver.py` (TargetResolver/PredictionResolver) | Replaced by `WorkspaceStore.get_prediction()` |

### Files to Create

| File | Purpose |
|------|---------|
| `pipeline/storage/chain_replay.py` | `replay_chain()` implementation |
| `tests/unit/pipeline/storage/test_export_chain.py` | Export chain → valid .n4a bundle |
| `tests/unit/pipeline/storage/test_chain_replay.py` | Replay chain on new data → correct predictions |
| `tests/integration/storage/test_export_roundtrip.py` | Run → store → export .n4a → predict from .n4a → compare |

### Test Requirements

| Test | What it validates |
|------|-------------------|
| `test_export_chain_n4a` | Chain → .n4a has manifest.json, chain.json, artifacts/ |
| `test_export_chain_n4a_py` | Chain → .n4a.py is valid Python with embedded artifacts |
| `test_export_pipeline_config` | Pipeline config → valid JSON matching expanded_config |
| `test_export_run_yaml` | Run → YAML with all pipelines, chains, metrics |
| `test_export_predictions_parquet` | Filtered predictions → valid Parquet readable by Polars |
| `test_replay_chain_simple` | MinMaxScaler→PLS chain replayed on new X → same predictions as original |
| `test_replay_chain_branching` | Branching chain replayed correctly |
| `test_predict_from_chain_id` | `nirs4all.predict(chain_id=..., data=X)` works |
| `test_export_import_roundtrip` | Export .n4a → predict from .n4a → predictions match original |

### Dependencies

Phase 3 (predictions system).

---

## Phase 5 — Webapp Integration

**Goal**: Update the webapp backend to use `WorkspaceStore` for run/prediction discovery and queries. Replace filesystem globbing with store queries.

### Files to Modify

| File | Changes |
|------|---------|
| `nirs4all-webapp/api/workspace_manager.py` | `WorkspaceScanner.discover_runs()` → `store.list_runs()` |
| `nirs4all-webapp/api/workspace.py` | Runs/predictions endpoints query store instead of globbing |
| `nirs4all-webapp/api/training.py` | Training job writes to DuckDB store |
| `nirs4all-webapp/api/predictions.py` | Prediction endpoints query store |
| `nirs4all-webapp/api/nirs4all_adapter.py` | Use `WorkspaceStore` directly |

### Endpoint Changes

| Endpoint | Current | New |
|----------|---------|-----|
| `GET /workspaces/{id}/runs` | Glob filesystem + parse YAML | `store.list_runs()` |
| `GET /workspaces/{id}/predictions/summary` | Read Parquet footer | `store.top_predictions(10)` + count query |
| `GET /workspaces/{id}/predictions/data` | Load Parquet + filter | `store.query_predictions(**filters)` |
| `POST /training/start` | Create job → nirs4all.run() | Same, store is default |
| `DELETE /runs/{id}` | Not implemented | `store.delete_run(run_id)` |
| `GET /runs/{id}/log` | Not implemented | `store.get_run_log_summary(run_id)` |

### Files to Create

| File | Purpose |
|------|---------|
| `nirs4all-webapp/api/store_adapter.py` | Adapter between webapp endpoints and `WorkspaceStore` |
| `nirs4all-webapp/tests/test_store_integration.py` | Test webapp endpoints with DuckDB store |

### Test Requirements

| Test | What it validates |
|------|-------------------|
| `test_webapp_list_runs` | `/workspaces/{id}/runs` returns runs from store |
| `test_webapp_predictions_summary` | `/workspaces/{id}/predictions/summary` fast response from store |
| `test_webapp_predictions_data` | `/workspaces/{id}/predictions/data` returns paginated results |
| `test_webapp_delete_run` | `DELETE /runs/{id}` removes run + cascades |
| `test_webapp_training_duckdb` | Training job writes to DuckDB store |

### Dependencies

Phase 4 (export system).

---

## Phase 6 — Predictions Documentation Overhaul

**Goal**: Consolidate, restructure, and rewrite the predictions documentation into a single authoritative guide. Predictions are a fundamental user workflow and deserve first-class documentation, not scattered fragments across design docs, API references, and deployment guides.

### Problem Statement

Prediction documentation is currently fragmented across **10+ files** in different formats, audiences, and levels of detail:

| Current File | Problem |
|--------------|---------|
| `docs/source/reference/predictions_api.md` | API reference only; no workflow guidance, outdated (PredictionResultsList) |
| `docs/source/user_guide/deployment/prediction_model_reuse.md` | Buried under "deployment"; mixes prediction with export/transfer/retrain |
| `docs/_internal/specifications/prediction_reload_design.md` | 1,680 lines of internal design spec; not user-facing |
| `docs/design/run_predictions_storage_redesign.md` | Storage internals; irrelevant to users |
| `examples/predictions.ipynb` | Good but orphaned — not referenced from docs |
| `examples/user/06_deployment/U01_save_load_predict.py` | Hidden under deployment folder |
| `examples/legacy/Q5_predict.py` | Legacy example, should be deleted |
| `nirs4all-webapp/docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md` | Webapp-internal concepts doc |
| Multiple RST files in `docs/source/api/` | Auto-generated, no narrative guidance |

A user looking for "how do I make predictions with nirs4all?" has no single entry point. They must piece together information from API docs, deployment guides, internal specs, and scattered examples.

### Target Documentation Structure

After this phase, prediction documentation will be reorganized into a clear hierarchy under `docs/source/user_guide/predictions/`:

```
docs/source/user_guide/predictions/
├── index.md                        # Overview & navigation
├── understanding_predictions.md    # Core concepts
├── making_predictions.md           # Practical prediction workflows
├── analyzing_results.md            # Working with RunResult & querying
├── exporting_models.md             # Export, bundle, share
└── advanced_predictions.md         # Transfer, retrain, chain replay
```

### 6.1 — `index.md`: Predictions Overview

The entry point for all prediction documentation. Short (1 page), links to sub-pages.

**Contents:**
- What predictions are in nirs4all (a prediction = model + preprocessing chain + fold strategy + scores + arrays)
- The prediction lifecycle: train → store → query → export → predict on new data
- Navigation to sub-sections
- Quick-start snippet: train, get best model, predict on new data (5 lines)

### 6.2 — `understanding_predictions.md`: Core Concepts

Explains the mental model. No code-heavy examples — conceptual clarity.

**Contents:**

- **What is stored per prediction**: Every prediction entry records the model class, preprocessing chain, fold ID, partition (train/val/test), all scores (RMSE, R², MAE, etc.), y_true/y_pred arrays, sample indices, branch info, exclusion stats, and a link to the trained chain
- **Chains**: A chain is the complete preprocessing→model path as executed during training. It includes every fitted transformer and model artifact, stored by fold. Chains are the unit of export and replay
- **Partitions**: train, val, test — what each means, how they relate to cross-validation folds
- **Scores vs. arrays**: Scalar scores (val_score, test_score) for ranking; full arrays (y_true, y_pred, y_proba) for visualization and detailed analysis
- **Prediction lifecycle diagram**: Train → Predictions stored in workspace → Query/filter/rank → Export chain → Predict on new data
- **Workspace storage**: Predictions live in `store.duckdb` alongside runs, pipelines, chains, and artifacts. No filesystem hierarchy

### 6.3 — `making_predictions.md`: Practical Workflows

The main "how-to" document. Code-heavy, task-oriented.

**Contents:**

- **From a RunResult** (most common path):
  ```python
  result = nirs4all.run(pipeline, dataset)
  preds = nirs4all.predict(result, new_data)
  ```

- **From an exported bundle**:
  ```python
  result.export("model.n4a")
  preds = nirs4all.predict("model.n4a", new_data)
  ```

- **From a chain ID** (new with DuckDB store):
  ```python
  preds = nirs4all.predict(chain_id="abc123", data=new_data)
  ```

- **From a standalone Python script**:
  ```python
  result.export("model.n4a.py", format="n4a.py")
  # Run standalone: python model.n4a.py input.csv
  ```

- **Data format requirements**: What shape/format new_data must be (numpy array, CSV path, SpectroDataset), wavelength alignment, feature count matching
- **Cross-validation ensemble**: How fold models are averaged during prediction
- **Preprocessing replay**: How the chain re-applies all preprocessing steps in order
- **Error handling**: Common errors (feature mismatch, missing wavelengths, corrupt bundle) and how to fix them
- **Prediction output**: PredictResult object, accessing y_pred, confidence, metadata

### 6.4 — `analyzing_results.md`: Working with Results

How to query, filter, rank, and visualize predictions after a run.

**Contents:**

- **RunResult API**:
  ```python
  result.best_rmse                    # Best validation RMSE
  result.best_r2                      # Best validation R²
  result.best_accuracy                # Best validation accuracy (classification)
  result.top(10)                      # Top 10 predictions by val_score
  result.filter(dataset="wheat")      # Filter by dataset
  result.filter(model_class="PLS*")   # Filter by model pattern
  result.get_datasets()               # List all datasets in results
  result.get_models()                 # List all model classes
  ```

- **Store-level queries** (cross-run analysis):
  ```python
  store = WorkspaceStore(workspace_path)
  store.top_predictions(20, metric="val_score", group_by="model_class")
  store.query_predictions(dataset_name="wheat", partition="test")
  store.list_runs(status="completed")
  ```

- **Prediction fields reference**: Complete table of all fields stored per prediction (model_name, model_class, dataset_name, fold_id, partition, val_score, test_score, train_score, metric, task_type, n_samples, n_features, scores, best_params, branch_id, branch_name, exclusion_count, exclusion_rate, created_at)

- **Visualization with PredictionAnalyzer**:
  - Loading predictions from workspace
  - Actual vs. predicted plots
  - Residual analysis
  - Top-K model comparison charts
  - Confusion matrices (classification)
  - Heatmaps, candlestick plots, histograms
  - Multi-dataset comparison
  - Model filtering and renaming for publication-ready charts

- **Exporting prediction data**:
  ```python
  store.export_predictions_parquet("results.parquet", dataset_name="wheat")
  ```

### 6.5 — `exporting_models.md`: Export, Bundle, Share

How to get models out of the workspace for deployment, sharing, or archival.

**Contents:**

- **Export best model**:
  ```python
  result.export("best_model.n4a")
  ```

- **Export specific prediction's model**:
  ```python
  result.export("model.n4a", prediction_id="abc123")
  ```

- **Export formats**:
  | Format | Extension | Use Case |
  |--------|-----------|----------|
  | Bundle | `.n4a` | Standard: self-contained ZIP with chain + artifacts |
  | Python script | `.n4a.py` | Standalone: embedded artifacts, no nirs4all dependency |
  | Pipeline config | `.json` | Re-run: pipeline definition for `nirs4all.run()` |
  | Run metadata | `.yaml` | Archival: full run description |

- **Bundle anatomy**: What's inside a `.n4a` file (manifest.json, chain.json, artifacts/)
- **Sharing models**: Copy `.n4a` to another machine, predict without workspace
- **Loading bundles**:
  ```python
  from nirs4all.sklearn import NIRSPipeline
  model = NIRSPipeline.from_bundle("model.n4a")  # sklearn-compatible
  ```

### 6.6 — `advanced_predictions.md`: Transfer, Retrain, Chain Replay

Advanced workflows for users who need more than basic predict.

**Contents:**

- **Transfer learning**:
  ```python
  result = nirs4all.retrain("model.n4a", new_data, mode="transfer")
  ```
  Modes: FULL (retrain all), TRANSFER (freeze base), FINETUNE (partial)

- **Retraining from chain ID**:
  ```python
  result = nirs4all.retrain(chain_id="abc123", data=new_data, mode="finetune")
  ```

- **Chain replay internals**: How `replay_chain()` works — step-by-step artifact loading, transformer application, fold model averaging

- **SHAP explanations**:
  ```python
  explanation = nirs4all.explain("model.n4a", data)
  explanation.plot()  # Feature importance visualization
  ```

- **Confidence intervals**: Bootstrap, jackknife, ensemble-based prediction intervals

- **Batch prediction patterns**: Predicting across multiple datasets, handling different wavelength ranges, multi-source prediction

### Files to Delete

| File | Reason |
|------|--------|
| `examples/legacy/Q5_predict.py` | Legacy example, replaced by updated user examples |
| `docs/design/run_predictions_storage_redesign.md` | Obsolete: the DuckDB migration replaces this design |
| `docs/_internal/specifications/prediction_reload_design.md` | Internals absorbed into updated architecture docs |

### Files to Modify

| File | Changes |
|------|---------|
| `docs/source/reference/predictions_api.md` | Rewrite: reference the new store-backed API, link to user guide |
| `docs/source/user_guide/deployment/prediction_model_reuse.md` | Delete or redirect: content moved to `predictions/` section |
| `examples/predictions.ipynb` | Update: use store-backed queries, reference new docs |
| `examples/user/06_deployment/U01_save_load_predict.py` | Update: use store-based workflow |
| `examples/user/06_deployment/U02_export_bundle.py` | Update: export from RunResult using store chain |
| All RST API docs in `docs/source/api/` | Regenerate: new classes, removed classes |

### Files to Create

| File | Purpose |
|------|---------|
| `docs/source/user_guide/predictions/index.md` | Overview & navigation |
| `docs/source/user_guide/predictions/understanding_predictions.md` | Core concepts |
| `docs/source/user_guide/predictions/making_predictions.md` | Practical workflows |
| `docs/source/user_guide/predictions/analyzing_results.md` | Querying, filtering, visualization |
| `docs/source/user_guide/predictions/exporting_models.md` | Export, bundle, share |
| `docs/source/user_guide/predictions/advanced_predictions.md` | Transfer, retrain, chain replay |

### Quality Criteria

- A new user can go from "I just ran `nirs4all.run()`" to "I predicted on new data and exported my model" by reading `making_predictions.md` alone
- Every code snippet in the docs runs without modification (tested in CI)
- No internal implementation details leak into user-facing docs
- The `predictions/` section is reachable from the main docs sidebar and from `nirs4all.predict.__doc__`
- All old prediction doc locations redirect or link to the new canonical location

### Dependencies

Phase 5 (webapp integration — all code changes complete before rewriting docs).

---

## Phase 7 — General Documentation, Examples & Tests Update

**Goal**: Update all remaining documentation, examples, and tests to reflect the new storage architecture. Remove references to manifests, pipeline.json, folder hierarchy. Clean up any remaining legacy files.

### Documentation Updates

| Document | Changes |
|----------|---------|
| `docs/source/reference/workspace.md` | Rewrite: new workspace structure (store.duckdb + artifacts/), no folder hierarchy |
| `docs/source/reference/storage.md` | Rewrite: DuckDB store architecture, WorkspaceStore API |
| `docs/source/user_guide/deployment/export_bundles.md` | Update: export from store (chain-based), not from filesystem |
| `docs/source/user_guide/deployment/retrain_transfer.md` | Update: retrain from chain_id |
| `docs/_internal/specifications/workspace_serialization.md` | Rewrite: DuckDB schema, no YAML manifests |
| `CLAUDE.md` | Update workspace structure, storage commands |
| `nirs4all-webapp/docs/_internals/CONCEPTS_RUN_RESULTS_PRED.md` | Rewrite: store-backed concepts |

### Example Updates

| Example | Changes |
|---------|---------|
| `examples/user/06_deployment/U03_workspace_management.py` | Rewrite: show store.duckdb, list_runs(), export operations |
| `examples/developer/06_internals/D01_session_workflow.py` | Update: session with store |
| `examples/legacy/Q14_workspace.py` | Delete |
| `examples/legacy/Q32_export_bundle.py` | Delete |

### Test Cleanup

All storage/prediction-related test files need review. Key categories:

| Category | Action |
|----------|--------|
| Storage unit tests (10 files) | Rewrite to test `WorkspaceStore` |
| Bundle tests (1 file) | Update to export from store chain |
| Predictions tests (5 files) | Already rewritten in Phase 3; verify completeness |
| Workspace tests (3 files) | Rewrite for store-based workspace |
| Integration artifact tests (5 files) | Update artifact paths (flat directory) |
| Integration pipeline tests (47 files) | Should pass unchanged if API is stable |
| Integration API tests (1 file) | Update for new storage |

### CLAUDE.md Updates

The workspace structure section in `CLAUDE.md` must be updated:

**Current:**
```
workspace/
  runs/<dataset>/0001_xxx/manifest.yaml
  binaries/<dataset>/
  exports/<dataset>/
  library/templates/
<dataset>.meta.parquet
```

**New:**
```
workspace/
  store.duckdb                    # All metadata, configs, logs, chains, predictions
  artifacts/                      # Flat content-addressed binaries
    ab/abc123.joblib
  exports/                        # User-triggered exports (on demand)
```

### Final Verification

After all updates, run the full test suite:

```bash
pytest tests/                        # All tests pass
cd examples && ./run.sh -q           # All examples pass
npm run test                         # Webapp tests pass (from nirs4all-webapp/)
```

### Dependencies

Phase 6 (predictions documentation complete).

---

## Verification Strategy

### Per-Phase Verification

Each phase has its own test suite. Before proceeding to the next phase:

1. All new tests pass
2. All existing tests pass (no regression)
3. `cd examples && ./run.sh -q` passes (quick examples)

### End-to-End Verification Checklist

After Phase 7, run the full verification:

| Verification | Command | Expected |
|--------------|---------|----------|
| All unit tests | `pytest tests/unit/` | Pass |
| All integration tests | `pytest tests/integration/` | Pass |
| sklearn-only tests | `pytest -m sklearn` | Pass |
| Quick examples | `cd examples && ./run.sh -q` | All pass |
| User examples | `cd examples && ./run.sh -c user` | All pass |
| Developer examples | `cd examples && ./run.sh -c developer` | All pass |
| Webapp tests | `cd nirs4all-webapp && npm run test` | Pass |
| Node validation | `cd nirs4all-webapp && npm run validate:nodes` | Pass |
| Lint | `ruff check .` | No errors |
| Type check | `mypy .` | No new errors |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| DuckDB version breaks storage format | Pin DuckDB version in `pyproject.toml`; add version check on store open |
| Performance regression on large workloads | Benchmark suite testing DuckDB on 100K predictions |
| Concurrent webapp + training writes | DuckDB WAL mode; add integration test with two threads |
| Users rely on filesystem layout | Document breaking change clearly in release notes and changelog |
| Bundle format changes | Bundle format (.n4a) is independent of store — no change needed |
| `duckdb` dependency too large | It's ~50MB; acceptable for a data science library; document in install notes |

---

## Phase Summary

| Phase | Goal | Key Deliverables | Depends On |
|-------|------|------------------|------------|
| **0** | API Surface Design | `WorkspaceStore` interface, protocol | — |
| **1** | DuckDB Implementation | Full CRUD, queries, schema, unit tests | Phase 0 |
| **2** | Replace Execution Storage | WorkspaceStore in PipelineRunner, chain builder, delete old storage code | Phase 1 |
| **3** | Predictions Rewrite | Store-backed Predictions, RunResult, delete old predictions code | Phase 2 |
| **4** | Export & Import | Export chain/pipeline/run, chain replay, delete old export code | Phase 3 |
| **5** | Webapp Integration | Store-backed endpoints | Phase 4 |
| **6** | Predictions Documentation | Consolidated predictions guide (6 documents) | Phase 5 |
| **7** | General Docs & Cleanup | All remaining docs, examples, tests updated | Phase 6 |

---

## File Inventory

### New Files (to create)

| File | Phase |
|------|-------|
| `nirs4all/pipeline/storage/workspace_store.py` | 0-1 |
| `nirs4all/pipeline/storage/store_protocol.py` | 0 |
| `nirs4all/pipeline/storage/store_schema.py` | 1 |
| `nirs4all/pipeline/storage/store_queries.py` | 1 |
| `nirs4all/pipeline/storage/chain_builder.py` | 2 |
| `nirs4all/pipeline/storage/chain_replay.py` | 4 |
| `tests/unit/pipeline/storage/test_workspace_store_api.py` | 0 |
| `tests/unit/pipeline/storage/test_workspace_store.py` | 1 |
| `tests/unit/pipeline/storage/test_store_schema.py` | 1 |
| `tests/unit/data/test_predictions_store.py` | 3 |
| `tests/unit/pipeline/storage/test_export_chain.py` | 4 |
| `tests/unit/pipeline/storage/test_chain_replay.py` | 4 |
| `tests/integration/storage/test_duckdb_pipeline.py` | 2 |
| `tests/integration/storage/test_export_roundtrip.py` | 4 |
| `nirs4all-webapp/api/store_adapter.py` | 5 |
| `nirs4all-webapp/tests/test_store_integration.py` | 5 |
| `docs/source/user_guide/predictions/index.md` | 6 |
| `docs/source/user_guide/predictions/understanding_predictions.md` | 6 |
| `docs/source/user_guide/predictions/making_predictions.md` | 6 |
| `docs/source/user_guide/predictions/analyzing_results.md` | 6 |
| `docs/source/user_guide/predictions/exporting_models.md` | 6 |
| `docs/source/user_guide/predictions/advanced_predictions.md` | 6 |

### Files to Modify (by phase)

| Phase | Files |
|-------|-------|
| 2 | `orchestrator.py`, `executor.py`, `builder.py`, `runner.py`, `context.py`, `api/run.py` |
| 3 | `predictions.py`, `api/result.py` |
| 4 | `bundle/generator.py`, `library_manager.py`, `api/predict.py`, `resolver.py` |
| 5 | `webapp/workspace_manager.py`, `webapp/workspace.py`, `webapp/training.py`, `webapp/predictions.py`, `webapp/nirs4all_adapter.py` |
| 6 | `predictions_api.md`, `prediction_model_reuse.md`, `predictions.ipynb`, examples |
| 7 | `workspace.md`, `storage.md`, `CLAUDE.md`, remaining docs + examples, 83 test files reviewed |

### Files to Delete (by phase)

| File | Phase | Replaced By |
|------|-------|-------------|
| `pipeline/storage/manifest_manager.py` | 2 | `workspace_store.py` |
| `pipeline/storage/io.py` | 2 | `workspace_store.py` |
| `pipeline/storage/io_writer.py` | 2 | `workspace_store.py` |
| `tests/unit/pipeline/storage/test_manifest_manager.py` | 2 | `test_workspace_store.py` |
| `tests/unit/pipeline/storage/test_manifest_v2.py` | 2 | `test_workspace_store.py` |
| `data/_predictions/storage.py` | 3 | `workspace_store.py` |
| `data/_predictions/serializer.py` | 3 | `workspace_store.py` |
| `data/_predictions/indexer.py` | 3 | `workspace_store.py` |
| `data/_predictions/ranker.py` | 3 | `workspace_store.py` |
| `data/_predictions/aggregator.py` | 3 | `workspace_store.py` |
| `data/_predictions/query.py` | 3 | `workspace_store.py` |
| `data/_predictions/array_registry.py` | 3 | `workspace_store.py` |
| `data/_predictions/schemas.py` | 3 | `store_schema.py` |
| `tests/unit/data/predictions/test_array_registry.py` | 3 | `test_predictions_store.py` |
| `pipeline/storage/io_exporter.py` | 4 | `workspace_store.py` |
| `pipeline/storage/io_resolver.py` | 4 | `workspace_store.py` |
| `examples/legacy/Q5_predict.py` | 6 | New predictions docs |
| `docs/design/run_predictions_storage_redesign.md` | 6 | Obsolete |
| `docs/_internal/specifications/prediction_reload_design.md` | 6 | Absorbed into architecture |
| `examples/legacy/Q14_workspace.py` | 7 | Updated workspace examples |
| `examples/legacy/Q32_export_bundle.py` | 7 | Updated export examples |
