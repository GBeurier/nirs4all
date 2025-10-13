# Pipeline & Utils Review

## Analysis

### Processing flow
- Pipeline definitions originate from `PipelineConfigs` (`nirs4all/pipeline/config.py:13`), which loads YAML/JSON, merges `*_params` keys, serializes classes/functions via `serialize_component`, and optionally expands combinatorial specs with `_or_`/`_range_` through `expand_spec` (`nirs4all/pipeline/generator.py`).
- Expanded step dictionaries are consumed by `PipelineRunner` (`nirs4all/pipeline/runner.py:21`), which loops over dataset configurations, instantiates controllers from `CONTROLLER_REGISTRY`, executes operations, and accumulates `Predictions`, history, and on-disk artifacts via `SimulationSaver`.
- Execution tracking is handled by `PipelineHistory` (`nirs4all/pipeline/history.py`), storing `PipelineExecution`/`StepExecution` dataclasses, bundling logs, fitted operations, and optional dataset snapshots.
- File and binary management relies on `SimulationSaver` and `BinaryLoader` (`nirs4all/pipeline/io.py`, `binary_loader.py`), which serialize metadata to JSON and load pickled artifacts when replaying predictions.
- Supporting utilities in `nirs4all/utils` (e.g., `model_utils.py`, `tab_report_manager.py`, `shap_analyzer.py`, `spinner.py`) provide task-type heuristics, metric calculations, reporting, progress visualization, and SHAP analysis, and are invoked directly by runner/controllers.

### Pain points and risks
- **Monolithic runner**: `PipelineRunner` manages orchestration, IO, prediction aggregation, reporting, visualization hooks, SHAP preparation, and random seed initialization in ~600 LOC, with pervasive mutable state (`self.step_number`, `self.prediction_metadata`, `self._capture_model`, etc.). Failures in one concern (e.g., TabReport) can cascade because responsibilities are tightly coupled.
- **Config expansion semantics**: `expand_spec` (`nirs4all/pipeline/generator.py`) intermixes random sampling (`random.sample`) with deterministic combinatorics, lacks reproducibility controls, and treats lists, dicts, and `_or_` values inconsistently. There is no schema validation or type safety before serialization.
- **Serialization hazards**: `serialize_component`/`deserialize_component` instantiate classes/functions dynamically via `importlib`, attach `_runtime_instance`, and attempt to infer constructor parameter types using reflection. Instantiation side effects (network calls, GPU allocation) occur at deserialization time, and missing modules silently return raw strings.
- **Artifact management gaps**: `SimulationSaver` writes metadata with assumptions (e.g., `filepath.existed_before` attribute) and scatters prediction saving logic (`save_file`, `save_json`, `save_binary`). `BinaryLoader` references `_cache` without initialization and trusts metadata paths without checksums or versioning.
- **Observability**: Colorized `print` statements across runner/history (e.g., `print(f"\033[94m🚀 Starting...")`) bypass `logging`, complicating integration with services. There is no structured event stream for step start/stop, errors, or artifacts.
- **Duplicate metric logic**: `ModelUtils` and `TabReportManager` re-implement scoring already available in `nirs4all/dataset/evaluator.py`, leading to mismatch in metric names, task detection, and output formatting.
- **Utils coupling**: `ShapAnalyzer`, `TabReportManager`, `spinner`, and `backend_utils` expose free functions instead of cohesive services, making dependency inversion difficult and increasing surface area for side effects during pipeline execution.

## Recommendations

### 1. Formalize pipeline definitions
- Introduce typed descriptors to replace ad-hoc nested dicts:
  ```python
  @dataclass(frozen=True)
  class StepSpec:
      id: str
      kind: StepKind
      component: ComponentRef
      params: Mapping[str, Any]
      children: tuple["StepSpec", ...] = ()

  @dataclass(frozen=True)
  class PipelinePlan:
      name: str
      steps: tuple[StepSpec, ...]
      metadata: Mapping[str, Any] = field(default_factory=dict)
  ```
- Build a `PipelineSpecParser` that validates loaded YAML/JSON against a schema (pydantic or voluptuous), resolves `_or_`/`_range_` into typed expansion nodes, and emits reproducible plans (seeded randomness managed centrally).
- Replace `serialize_component` with a registry-driven approach:
  ```python
  class ComponentRegistry(Protocol):
      def resolve(self, ref: ComponentRef, *, runtime: bool = False) -> Any
      def to_ref(self, obj: Any) -> ComponentRef
  ```
  enabling explicit whitelisting of creatable classes/functions and preventing arbitrary import execution.

### 2. Modular execution engine
- Factor `PipelineRunner` into testable units:
  - `ExecutionContext`: immutable snapshot with dataset handle, partition, random state, and artifact sink.
  - `StepExecutor` (Strategy pattern) per `StepKind`, responsible only for translating `StepSpec` into controller calls.
  - `PipelineEngine` orchestrating step order, error policies, parallelism, and emitting structured events (`StepStarted`, `StepCompleted`, `StepFailed`).
  - `RunResult` aggregating predictions, metrics, and artifact manifests rather than direct disk writes.
- Example interface sketch:
  ```python
  class PipelineEngine:
      def run(self, plan: PipelinePlan, datasets: Iterable[DatasetView], *, options: RunOptions) -> RunResult:
          ...

  class StepExecutor(Protocol):
      def execute(self, spec: StepSpec, ctx: ExecutionContext) -> ExecutionContext:
          ...
  ```
- Provide dependency injection for utilities (`ModelMetrics`, `ReportWriter`, `ShapService`) so runner orchestration remains agnostic to concrete implementations.

### 3. Deterministic spec expansion
- Replace `expand_spec`/`count_combinations` with a composable expander:
  ```python
  class SpecExpander:
      def __init__(self, *, rng: Random, max_combinations: int) -> None: ...
      def expand(self, node: SpecNode) -> Iterator[PipelinePlan]: ...
  ```
  where `SpecNode` captures `_or_`, `_range_`, and list semantics explicitly. Enforce sorted order and seed control to guarantee reproducible runs (especially for grid vs sampled searches).
- Emit diagnostics (e.g., tree depth, estimated combinations) before expansion to allow early guardrails.

### 4. Artifact & binary management
- Introduce an `ArtifactStore` abstraction to mediate all file outputs:
  ```python
  class ArtifactStore(Protocol):
      def start_run(self, dataset_name: str, plan: PipelinePlan) -> ArtifactSession: ...

  class ArtifactSession(Protocol):
      def write_blob(self, artifact: ArtifactPayload) -> ArtifactRef: ...
      def finalize(self, manifest: RunManifest) -> None: ...
  ```
- Represent binaries with metadata-rich dataclasses (`BinaryArtifact` with checksum, created_at, controller_id), and version manifests as Parquet/JSON with schema validation.
- Fix `BinaryLoader` by introducing an internal cache (`functools.lru_cache` or explicit dict keyed by artifact hash) and verifying presence + checksum before unpickling. Provide streaming loaders for large models to avoid memory spikes.

### 5. Serialization & dependency control
- Replace dynamic instantiation with declarative component references:
  ```python
  @dataclass(frozen=True)
  class ComponentRef:
      namespace: str  # e.g. "sklearn.preprocessing"
      name: str       # e.g. "StandardScaler"
      factory: ComponentFactory  # optional callable
  ```
- Maintain a `ComponentCatalog` sourced from `CONTROLLER_REGISTRY` and known operator registries, preventing arbitrary module execution. Use entry points or plugin discovery for extensibility.
- Persist runtime state via explicit artifact types (e.g., `FittedModel`, `TransformerState`) rather than embedding `_runtime_instance` into serialized configs.

### 6. Observability and control plane
- Replace `print` calls with structured logging (`logging.Logger` injected per component) and optional event hooks (WebSocket, CLI spinner).
- Emit execution events to `PipelineHistory` through a pub/sub channel, enabling alternative sinks (database, telemetry).
- Standardize status codes (`RUNNING`, `COMPLETED`, `SKIPPED`, `FAILED`) and include controller identifiers, duration, and artifact references in every event.

### 7. Utilities alignment
- Derive `ModelUtils` and `TabReportManager` from the shared evaluator API (`nirs4all/dataset/evaluator.py`). Establish a `MetricRegistry`:
  ```python
  class MetricRegistry:
      def register(self, metric: Metric): ...
      def compute(self, metric_id: str, y_true, y_pred, task: TaskType) -> float: ...
  ```
- Replace heuristic task detection with `TaskType` propagation from `SpectroDataset` and controllers. Utilities should accept `TaskType` rather than re-detecting from `y_true`.
- Consolidate SHAP tooling into a `ShapService` that consumes `ExecutionContext`, ensuring consistent data extraction and artifact publishing.
- Rework spinner/progress indicators to subscribe to execution events rather than being toggled ad-hoc within runner methods.

## Implementation roadmap

1. **Schema & validation**
   - Draft pipeline configuration schema and publish developer documentation.
   - Add regression tests for spec loading, expansion counts, and serialization round-trips.
   - Restrict `PipelineConfigs` to producing `PipelinePlan` instances only after validation.

2. **Execution engine refactor**
   - Implement `PipelineEngine`, `ExecutionContext`, and `StepExecutor` abstractions.
   - Port existing controllers to the new executor interface, leaving legacy runner as adapter during transition.
   - Instrument engine with structured logging and event emission.

3. **Artifact store & binary loader**
   - Build `ArtifactStore` backed by hierarchical directories (dataset/plan/run) with manifest files (JSON/Parquet) that include checksums and artifact metadata.
   - Replace `SimulationSaver`/`BinaryLoader` with the new APIs, ensuring compatibility with prediction replay and SHAP workflows.
   - Provide migration script to convert existing `simulations` directories into the new manifest format.

4. **Serialization overhaul**
   - Implement `ComponentRegistry` and safe deserialization pipeline. Remove `_runtime_instance` leakage from persisted configs.
   - Provide tooling to register custom components/operators explicitly (e.g., entry points or config files).
   - Introduce compatibility layer to read legacy serialized configs and map them into `ComponentRef`.

5. **Utility consolidation**
   - Refactor `ModelUtils`, `TabReportManager`, and evaluator usage to rely on a common `MetricRegistry`.
   - Centralize task-type handling and scoring; remove divergent implementations scattered across utils.
   - Encapsulate SHAP/report generators as services injected via runner options.

6. **Observability & UX**
   - Implement event-driven progress reporting (CLI spinner, dashboards) consuming the engine’s event bus.
   - Update `PipelineHistory` to store normalized event records and artifact manifests. Provide export tools for dashboards.
   - Replace ANSI-colored prints with logger-based formatting configurable by the host application.

7. **Transition & QA**
   - Run legacy and new engines side-by-side on benchmark pipelines, comparing predictions, artifacts, and execution logs.
   - Add stress tests for spec expansion bounds, artifact persistence/replay, and multi-dataset runs.
   - Document migration steps for existing pipelines, including seed management for deterministic configuration runs.

Executing this plan yields a pipeline subsystem with explicit contracts, deterministic configuration, modular execution, and aligned utilities—improving reliability and maintainability while keeping room for extension.

