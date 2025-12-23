# Document 1: Pipeline Big Picture

**Date**: December 2025
**Author**: Architecture Assessment
**Status**: Analysis Complete

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Payload Model](#payload-model)
3. [Dataset Structure & Indexing](#dataset-structure--indexing)
4. [Views and Layouts](#views-and-layouts)
5. [Operator Execution Patterns](#operator-execution-patterns)
6. [Prediction Generation & Persistence](#prediction-generation--persistence)
7. [Pipeline Organization & Execution](#pipeline-organization--execution)
8. [Generator Behavior](#generator-behavior)
9. [DAG Compatibility Analysis](#dag-compatibility-analysis)

---

## Current Architecture Overview

### Module Structure

```
nirs4all/
├── pipeline/
│   ├── runner.py              # PipelineRunner (public entry point)
│   ├── config/
│   │   ├── pipeline_config.py # PipelineConfigs (step parsing, generation)
│   │   ├── context.py         # ExecutionContext, RuntimeContext, DataSelector
│   │   ├── generator.py       # expand_spec, _or_, _range_ expansion
│   │   └── component_serialization.py
│   ├── execution/
│   │   ├── orchestrator.py    # PipelineOrchestrator (multi-pipeline/dataset)
│   │   ├── executor.py        # PipelineExecutor (single pipeline execution)
│   │   ├── builder.py         # ExecutorBuilder
│   │   └── result.py          # StepOutput, StepResult, ArtifactMeta
│   ├── steps/
│   │   ├── parser.py          # StepParser, ParsedStep
│   │   ├── router.py          # ControllerRouter (registry dispatch)
│   │   └── step_runner.py     # StepRunner (single step execution)
│   ├── storage/
│   │   ├── artifacts/         # ArtifactRegistry, ArtifactRecord
│   │   ├── manifest_manager.py
│   │   └── io.py              # SimulationSaver
│   └── trace/
│       ├── execution_trace.py # ExecutionTrace, ExecutionStep
│       ├── recorder.py        # TraceRecorder
│       └── extractor.py       # TraceBasedExtractor, MinimalPipeline
├── controllers/
│   ├── registry.py            # @register_controller, CONTROLLER_REGISTRY
│   ├── controller.py          # OperatorController (ABC)
│   ├── data/                  # branch, merge, feature_augmentation, etc.
│   ├── transforms/            # TransformerMixin controllers
│   ├── models/                # Model training controllers
│   ├── splitters/             # Cross-validation controllers
│   ├── charts/                # Visualization controllers
│   └── flow/                  # condition, scope, sequential
├── data/
│   ├── dataset.py             # SpectroDataset (main facade)
│   ├── features.py            # Features (multi-source coordinator)
│   ├── targets.py             # Targets (y transformation chain)
│   ├── predictions.py         # Predictions (storage/ranking facade)
│   ├── indexer.py             # Indexer (sample tracking)
│   ├── metadata.py            # Metadata (Polars-backed)
│   └── _features/
│       └── feature_source.py  # FeatureSource (per-source storage)
└── operators/
    ├── transforms/            # NIRS-specific transformers (SNV, MSC, etc.)
    ├── models/                # Pre-built models (nicon, decon)
    └── splitters/             # Data splitting (KS, SPXY)
```

### Key Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `PipelineRunner` | Public API entry point, delegates to `PipelineOrchestrator` |
| `PipelineOrchestrator` | Multi-dataset/pipeline coordination, workspace management |
| `PipelineExecutor` | Single pipeline execution, step iteration, branching |
| `StepRunner` | Single step dispatch: parse → route → execute |
| `StepParser` | Normalize diverse step syntaxes to `ParsedStep` |
| `ControllerRouter` | Match step to controller via registry priority |
| `OperatorController` | Abstract base for all step handlers |
| `SpectroDataset` | Multi-source feature/target/metadata facade |
| `Predictions` | Prediction storage, ranking, OOF reconstruction |
| `TraceRecorder` | Record execution path for deterministic replay |

---

## Payload Model

The pipeline manipulates a conceptual "payload" composed of four elements:

### 1. Dataset (`SpectroDataset`)

```
SpectroDataset
├── _features (Features)
│   └── sources: List[FeatureSource]
│       └── data: np.ndarray (samples × processings × features)
│       └── processing_ids: List[str]
│       └── headers: List[str]
├── _targets (Targets)
│   └── versions: Dict[str, np.ndarray]  # "numeric" → "scaled" → ...
│   └── transformers: Dict[str, TransformerMixin]
├── _metadata (Metadata)
│   └── df: pl.DataFrame
├── _indexer (Indexer)
│   └── df: pl.DataFrame  # partition, group, origin, augmentation, etc.
└── _folds: List[Tuple[train_idx, val_idx]]
```

### 2. Predictions (`Predictions`)

```
Predictions
├── _storage (PredictionStorage)
│   └── _df: pl.DataFrame  # Metadata columns
│   └── _array_registry: Dict[id, np.ndarray]  # y_true, y_pred, y_proba
├── _indexer (PredictionIndexer)
├── _ranker (PredictionRanker)
├── _aggregator (PartitionAggregator)
└── _query (CatalogQueryEngine)
```

Key fields per prediction entry:
- `dataset_name`, `config_name`, `pipeline_uid`
- `model_name`, `model_classname`, `fold_id`
- `partition` (train/val/test), `step_idx`, `op_counter`
- `y_true`, `y_pred`, `y_proba` (stored via array registry)
- `val_score`, `test_score`, `metric`
- `branch_id`, `branch_path`, `branch_name`
- `model_artifact_id`, `trace_id`

### 3. Artifacts

Artifacts are persisted binary objects (fitted transformers, trained models):

```python
@dataclass
class ArtifactRecord:
    artifact_id: str          # "s1.MinMaxScaler$abc123:all"
    step_index: int
    operator_class: str
    file_path: Path
    format: str               # "joblib", "keras", "torch"
    branch_path: List[int]
    fold_id: Optional[int]
    chain_path: str           # "s1.MinMaxScaler>s3.PLS"
    source_index: Optional[int]
```

### 4. Context (`ExecutionContext` + `RuntimeContext`)

```python
@dataclass
class ExecutionContext:
    selector: DataSelector      # partition, processing, layout, branch_path
    state: PipelineState        # y_processing, step_number, mode
    metadata: StepMetadata      # keyword, ephemeral flags
    custom: Dict[str, Any]      # branch_contexts, in_branch_mode
    aggregate_column: Optional[str]

@dataclass
class RuntimeContext:
    saver: SimulationSaver
    manifest_manager: ManifestManager
    artifact_loader: ArtifactLoader
    artifact_provider: ArtifactProvider
    artifact_registry: ArtifactRegistry
    step_runner: StepRunner
    trace_recorder: TraceRecorder
    pipeline_uid: str
    step_number: int
    operation_count: int
    # ... (counters, explainer, etc.)
```

---

## Dataset Structure & Indexing

### Multi-Source Architecture

```
SpectroDataset
└── Features
    └── sources[0]: FeatureSource  # e.g., "NIR spectra"
    │   └── data: (N, P₀, F₀)      # N samples, P₀ processings, F₀ features
    │   └── processing_ids: ["raw", "SNV", "SNV>SG"]
    └── sources[1]: FeatureSource  # e.g., "markers"
        └── data: (N, P₁, F₁)
        └── processing_ids: ["raw", "MinMax"]
```

### Indexer Structure

The `Indexer` maintains a Polars DataFrame tracking sample provenance:

| Column | Type | Description |
|--------|------|-------------|
| `_id` | int | Unique sample ID |
| `partition` | str | "train", "test", "val" |
| `group` | str/int | Optional grouping key |
| `origin` | int | Parent sample ID (for augmented) |
| `augmentation` | str | Augmentation method name |
| `excluded` | bool | Outlier exclusion flag |

### Target Transformation Chain

```
Targets.versions:
  "raw" → (100,)                    # Original y values
  "numeric" → (100,)                # After label encoding (classification)
  "scaled_MinMaxScaler_001" → (100,) # After y_processing

Targets.transformers:
  "scaled_MinMaxScaler_001" → MinMaxScaler(fitted)
```

Enables inverse transform for prediction output.

---

## Views and Layouts

### SpectroDataset.x() Layouts

```python
# 2D layout (default): flatten processings, optionally concat sources
X = dataset.x(selector, layout="2d", concat_source=True)
# Shape: (N, sum(P_i * F_i))

# 2D layout per-source
X_list = dataset.x(selector, layout="2d", concat_source=False)
# Returns: [arr₀, arr₁, ...] each (N, P_i * F_i)

# 3D layout: keep processing dimension
X = dataset.x(selector, layout="3d", concat_source=False)
# Returns: [arr₀, ...] each (N, P_i, F_i)
```

### DataSelector (View Specification)

```python
@dataclass
class DataSelector:
    partition: str = "all"
    processing: List[List[str]] = [["raw"]]  # Per-source
    layout: str = "2d"
    concat_source: bool = True
    fold_id: Optional[int] = None
    include_augmented: bool = False
    branch_id: Optional[int] = None
    branch_path: List[int] = []
    branch_name: Optional[str] = None
```

Controllers update `selector.processing` as transformations are applied.

---

## Operator Execution Patterns

### Controller Dispatch

```
StepRunner.execute(step)
  └─ StepParser.parse(step) → ParsedStep
  └─ ControllerRouter.route(parsed_step)
      └─ for controller in CONTROLLER_REGISTRY (sorted by priority):
           if controller.matches(step, operator, keyword):
               return controller
  └─ controller.execute(...)
```

### Controller Base Class

```python
class OperatorController(ABC):
    priority: int = 100  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool: ...

    @classmethod
    def use_multi_source(cls) -> bool: ...

    @classmethod
    def supports_prediction_mode(cls) -> bool: ...

    def execute(self, step_info, dataset, context, runtime_context,
                source, mode, loaded_binaries, prediction_store
    ) -> Tuple[ExecutionContext, StepOutput]: ...
```

### Sklearn Transformer Pattern

From `TransformController`:

```python
def execute(...):
    for src in sources:
        X = dataset.x(selector, include_excluded=True)

        if mode == "train":
            transformer.fit(X)
            X_new = transformer.transform(X)
            # Register artifact
        else:
            transformer = loaded_binaries[...]
            X_new = transformer.transform(X)

        dataset.update_features(old_proc, X_new, new_proc, source=src)
        context.selector.processing[src].append(new_proc)

    return context, StepOutput(artifacts=[...])
```

### TensorFlow/PyTorch Model Pattern

From `BaseModelController`:

```python
def execute(...):
    for fold_idx, (train_idx, val_idx) in enumerate(dataset.folds):
        X_train = dataset.x({"partition": "train"})
        y_train = dataset.y({"partition": "train"})

        if mode == "train":
            model = self.build_model(n_features=X_train.shape[1])  # DYNAMIC
            model.fit(X_train, y_train)
            # Save artifact per fold
        else:
            model = loaded_binaries[fold_idx]

        y_pred = model.predict(X_val)
        prediction_store.add_prediction(...)

    return context, StepOutput(artifacts=[...])
```

**Key insight**: TF/Keras model `build()` happens at runtime after feature dimension is known.

---

## Prediction Generation & Persistence

### Training Flow

```
ModelController.execute(mode="train")
  └─ For each fold:
      └─ Build/train model
      └─ model.predict(X_val) → y_pred_val
      └─ model.predict(X_test) → y_pred_test
      └─ prediction_store.add_prediction(
           partition="val", fold_id=fold_idx,
           y_true, y_pred, sample_indices,
           model_name, step_idx, branch_id, ...
         )
      └─ Save model artifact
  └─ Aggregate folds → add averaged predictions
```

### OOF Reconstruction (for Stacking)

```python
# TrainingSetReconstructor: Reconstruct OOF predictions for meta-model training
reconstructor = TrainingSetReconstructor(prediction_store, source_model_names)
result = reconstructor.reconstruct(dataset, context)
# result.X_train_meta: (N_train, N_models) - OOF predictions
# result.X_test_meta: (N_test, N_models) - averaged test predictions
```

### Prediction Persistence

```
Predictions.save_to_file("predictions.meta.parquet")
  └─ predictions_meta.parquet  # Metadata columns
  └─ predictions_arrays.parquet  # y_true, y_pred, y_proba (binary)
```

---

## Pipeline Organization & Execution

### Current Execution Model

```
PipelineOrchestrator.execute(pipeline, dataset)
  └─ Normalize inputs to PipelineConfigs, DatasetConfigs
  └─ For each dataset config:
      └─ Create ArtifactRegistry, ExecutorBuilder
      └─ For each pipeline config (after generation):
          └─ executor.execute(steps, dataset, context, ...)
              └─ For each step in steps:
                  └─ step_runner.execute(step, ...)
                      └─ Check branch_contexts → dispatch per-branch
                      └─ Parse → Route → Controller.execute()
```

### Branching Mechanics

**Branch Creation** (`BranchController`):
```python
def execute(...):
    branch_defs = self._parse_branch_definitions(step_info)
    branch_contexts = []

    for branch_id, branch_def in enumerate(branch_defs):
        branch_context = initial_context.copy()
        branch_context.selector.branch_path = [branch_id]

        # Snapshot features before branch
        # Execute branch steps
        for substep in branch_def["steps"]:
            result = runtime_context.step_runner.execute(substep, ...)
            branch_context = result.updated_context

        # Snapshot features after branch
        branch_contexts.append({
            "branch_id": branch_id,
            "context": branch_context,
            "features_snapshot": ...
        })

    context.custom["branch_contexts"] = branch_contexts
    context.custom["in_branch_mode"] = True
```

**Post-Branch Execution** (`PipelineExecutor._execute_steps`):
```python
for step in steps:
    branch_contexts = context.custom.get("branch_contexts", [])
    is_branch_step = isinstance(step, dict) and "branch" in step
    is_merge_step = isinstance(step, dict) and "merge" in step

    if branch_contexts and not is_branch_step and not is_merge_step:
        # Execute step on EACH branch
        for branch_info in branch_contexts:
            # Restore features snapshot
            # Execute step with branch context
    else:
        # Normal single-context execution
```

**Branch Merge** (`MergeController`):
```python
def execute(...):
    # Collect features and/or predictions from branches
    merged_features = self._collect_features(branch_contexts, ...)
    merged_predictions = self._collect_predictions(prediction_store, ...)

    # Store merged output
    dataset.add_merged_features(concatenated)

    # EXIT BRANCH MODE
    context.custom["branch_contexts"] = []
    context.custom["in_branch_mode"] = False
```

---

## Generator Behavior

### Current Design

Generators (`_or_`, `_range_`, etc.) expand **before execution**:

```python
class PipelineConfigs:
    def __init__(self, definition, max_generation_count=10000):
        self.steps = self._load_steps(definition)
        self.steps = serialize_component(self.steps)

        if self._has_gen_keys(self.steps):
            count = count_combinations(self.steps)
            expanded = expand_spec_with_choices(self.steps)
            self.steps = [config for config, choices in expanded]
            self.generator_choices = [choices for config, choices in expanded]
        else:
            self.steps = [self.steps]
```

**Exception**: Generators inside `branch` are handled at runtime by `BranchController`:

```python
# In BranchController._parse_branch_definitions:
if is_generator_node(raw_def):
    return self._expand_generator_branches(raw_def)
    # {"_or_": [SNV, MSC, D1]} → 3 runtime branches
```

### Generator Keywords

| Keyword | Behavior | Scope |
|---------|----------|-------|
| `_or_` | Choose from alternatives | Pre-execution OR branch runtime |
| `_range_` | Numeric sequence | Pre-execution OR branch runtime |
| `_log_range_` | Logarithmic sequence | Pre-execution |
| `_grid_` | Cartesian product | Pre-execution |
| `_sample_` | Statistical sampling | Pre-execution |
| `pick` | Combinations | Pre-execution |
| `arrange` | Permutations | Pre-execution |
| `count` | Limit variants | Pre-execution |

---

## DAG Compatibility Analysis

### What Maps Naturally to Nodes/Edges

| Current Concept | DAG Mapping |
|-----------------|-------------|
| Pipeline step | Node |
| Context flow | Edge (data dependency) |
| Controller | Node executor |
| Dataset | Payload on edges |
| Branch | Fork node (1 → N edges) |
| Merge | Join node (N → 1 edges) |
| Generator in branch | Dynamic node creation |

### What Is Currently Sequential But Could Be DAG

1. **Cross-source operations**: Steps applied to multiple sources could parallelize
2. **Independent branches**: Currently sequential iteration, could parallel
3. **Fold-level training**: Currently sequential, could parallelize per-fold
4. **Multi-dataset runs**: Already conceptually parallel (separate contexts)

### Runtime-Only Construction Requirements

```
⚠️ CRITICAL: These patterns REQUIRE dynamic DAG expansion

1. Feature Dimension Discovery
   └─ TensorFlow model.build(input_shape) requires knowing X.shape[1]
   └─ AutoML feature selection changes dimensions
   └─ PCA n_components="mle" determines dim at fit time

2. Branching from Generator
   └─ {"branch": {"_or_": [A, B, C]}} expands at runtime
   └─ Number of branches unknown until step executes

3. Condition-Based Flow
   └─ {"condition": {"if": metric > threshold, "then": ...}}
   └─ Requires runtime evaluation

4. Auto-Operators
   └─ AutoSelectPreprocessing chooses transform at runtime
   └─ Decision feeds into subsequent steps
```

### Current Assumptions That Would Break in Static DAG

1. **Shared mutable dataset**: All controllers modify the same `SpectroDataset`
   - Feature updates are in-place
   - Branches snapshot/restore features

2. **Sequential context propagation**: `ExecutionContext` flows step-to-step
   - Processing chains accumulate
   - Y processing name updates

3. **Branch context storage**: `context.custom["branch_contexts"]` holds state
   - Post-branch steps iterate over this list
   - Merge reads and clears it

4. **Prediction store accumulation**: Single `Predictions` instance collects all
   - Models from different branches write to same store
   - OOF reconstruction reads across branches

### Gaps Identified

| Gap | Description | Severity |
|-----|-------------|----------|
| No explicit data dependencies | Steps assume sequential order | Medium |
| No parallel execution infrastructure | Single-threaded, no task scheduler | Low |
| Feature mutations not tracked | Updates happen in-place | High |
| Branch state in dict, not graph | `custom["branch_contexts"]` is ad-hoc | Medium |
| Generator expansion location split | Some pre-run, some at runtime | Low |

---

## Summary

The current architecture is **highly compatible** with a DAG transformation:

1. **Controller pattern** is already node-like (stateless, receives context)
2. **Context immutability** (via `copy()`) supports edge semantics
3. **Artifact registry** provides addressable state storage
4. **Trace recording** already captures execution graph

**Key migration challenges**:

1. Replace in-place dataset mutations with immutable transforms
2. Formalize branch/merge as graph fork/join
3. Support dynamic node insertion for runtime-determined shapes
4. Maintain linear syntax compatibility (implicit ancestor = previous node)

**Recommendation**: Pursue incremental DAG refactoring, not wholesale rewrite. The existing abstractions (`ExecutionContext`, `StepOutput`, controller dispatch) can evolve to support DAG semantics.
