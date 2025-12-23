# Document 1 — Pipeline Big Picture

**Version**: 1.0.0
**Date**: December 2025
**Status**: Analysis Document

---

## Table of Contents

1. [Current Pipeline Architecture](#current-pipeline-architecture)
2. [Payload Model](#payload-model)
3. [Dataset Structure & Indexing](#dataset-structure--indexing)
4. [Views and Layouts](#views-and-layouts)
5. [Operator Execution Patterns](#operator-execution-patterns)
6. [Prediction Generation and Persistence](#prediction-generation-and-persistence)
7. [Pipeline Execution Flow](#pipeline-execution-flow)
8. [Generator Behavior](#generator-behavior)
9. [DAG Compatibility Analysis](#dag-compatibility-analysis)

---

## Current Pipeline Architecture

### Module Map

```
nirs4all/
├── pipeline/
│   ├── runner.py              # PipelineRunner - main entry point for users
│   ├── config/
│   │   ├── pipeline_config.py # PipelineConfigs - normalizes/validates/expands pipelines
│   │   ├── context.py         # ExecutionContext, DataSelector, PipelineState, RuntimeContext
│   │   ├── generator.py       # expand_spec, _or_, _range_ keywords → multiple configs
│   │   └── component_serialization.py  # serialize_component → JSON-safe format
│   ├── execution/
│   │   ├── orchestrator.py    # PipelineOrchestrator - runs N pipelines × M datasets
│   │   ├── executor.py        # PipelineExecutor - runs a single pipeline on one dataset
│   │   ├── builder.py         # ExecutorBuilder - factory for executor creation
│   │   └── result.py          # StepResult, StepOutput, ArtifactMeta
│   ├── steps/
│   │   ├── step_runner.py     # StepRunner - executes one step via controller dispatch
│   │   ├── parser.py          # StepParser, ParsedStep - normalizes step configs
│   │   └── router.py          # ControllerRouter - matches step → controller
│   ├── storage/
│   │   ├── io.py              # SimulationSaver - file I/O for runs
│   │   ├── manifest_manager.py # ManifestManager - tracks pipelines/artifacts
│   │   └── artifacts/         # ArtifactRegistry, ArtifactLoader, ArtifactRecord
│   ├── bundle/                # Export/import trained pipelines (.n4a)
│   └── trace/                 # TraceRecorder, ExecutionTrace - replay support
├── controllers/
│   ├── controller.py          # OperatorController abstract base class
│   ├── registry.py            # @register_controller decorator, priority dispatch
│   ├── transforms/            # TransformController, YProcessingController
│   ├── models/                # BaseModelController, MetaModelController
│   ├── splitters/             # SplitterController - CV splits (KFold, etc.)
│   ├── data/                  # BranchController, MergeController, source_branch, etc.
│   └── flow/                  # SequentialController, ScopeController, ConditionController
├── data/
│   ├── dataset.py             # SpectroDataset - main facade
│   ├── config.py              # DatasetConfigs - normalizes dataset inputs
│   ├── predictions.py         # Predictions - stores/ranks model outputs
│   ├── features.py            # Features - internal feature storage
│   ├── targets.py             # Targets - internal target storage
│   ├── indexer.py             # Indexer - sample indexing (partition, fold, origin)
│   └── metadata.py            # Metadata - sample-level metadata
├── operators/
│   ├── transforms/            # SNV, MSC, FirstDerivative, etc. (TransformerMixin)
│   ├── augmentation/          # AddNoise, Shift, etc.
│   ├── models/                # nicon, decon, meta-model configs
│   └── splitters/             # KennardStone, SPXY
└── visualization/             # PredictionAnalyzer, charts
```

### Execution Flow Summary

```
┌─────────────────┐
│ PipelineRunner  │  User-facing API: runner.run(pipeline, dataset)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ PipelineOrchestrator │  Coordinates N pipelines × M datasets
│                      │  Creates workspace, aggregates predictions
└──────────┬───────────┘
           │  for each (pipeline, dataset)
           ▼
┌──────────────────┐
│ PipelineExecutor │  Executes one pipeline on one dataset
│                  │  Manages step iteration, branch contexts
└────────┬─────────┘
         │  for each step
         ▼
┌────────────┐
│ StepRunner │  Parses step, routes to controller
└─────┬──────┘
      │
      ▼
┌────────────────────┐
│ OperatorController │  Implements step-specific logic
│ (via registry)     │  Returns (context, StepOutput)
└────────────────────┘
```

---

## Payload Model

The pipeline manipulates a **moving payload** that consists of four components:

### 1. Dataset (`SpectroDataset`)
- **Location**: `nirs4all/data/dataset.py`
- **Role**: Holds features (X), targets (Y), metadata, sample indexing
- **Key methods**: `x(selector, layout)`, `y(selector)`, `add_samples()`, `add_processing()`

### 2. Predictions (`Predictions`)
- **Location**: `nirs4all/data/predictions.py`
- **Role**: Stores model predictions with provenance (model, fold, partition, branch)
- **Key methods**: `add_prediction()`, `filter_predictions()`, `top()`, `get_best()`
- **Stored fields**: `y_true`, `y_pred`, `y_proba`, `sample_indices`, `fold_id`, `branch_id`, `step_idx`, scores

### 3. Artifacts
- **Location**: `nirs4all/pipeline/storage/artifacts/`
- **Role**: Persisted binary objects (fitted transformers, trained models)
- **System**: `ArtifactRegistry.register()` → `ArtifactRecord` → saved to disk
- **Identification**: Deterministic artifact_id based on `step:fold:branch_path:source`

### 4. Context (`ExecutionContext` + `RuntimeContext`)
- **Location**: `nirs4all/pipeline/config/context.py`
- **Role**: Carries state through pipeline execution
- **Components**:
  - `DataSelector`: partition, processing chains, layout, branch_id, branch_path
  - `PipelineState`: y_processing version, step_number, mode (train/predict)
  - `StepMetadata`: ephemeral flags for controller coordination
  - `RuntimeContext`: infrastructure references (saver, manifest_manager, artifact_registry, step_runner)

### Payload Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         PAYLOAD                                   │
├──────────────────────────────────────────────────────────────────┤
│  Dataset         │  Predictions     │  Artifacts    │  Context   │
│  ─────────       │  ───────────     │  ─────────    │  ───────   │
│  • X (features)  │  • y_pred        │  • Fitted     │  • Selector│
│  • Y (targets)   │  • y_true        │    transformers│  • State   │
│  • Metadata      │  • Scores        │  • Trained    │  • Metadata│
│  • Folds         │  • Fold/branch   │    models     │  • Custom  │
│  • Sources       │    provenance    │  • Splitter   │  • Branch  │
│  • Processings   │  • Sample indices│    states     │    contexts│
└──────────────────────────────────────────────────────────────────┘
       ▲                   ▲                  ▲             ▲
       │                   │                  │             │
       └───────────────────┴──────────────────┴─────────────┘
                    Controllers read/write these
```

---

## Dataset Structure & Indexing

### SpectroDataset Internal Architecture

```
SpectroDataset
├── _indexer (Indexer)
│   └── sample_records: List[Dict]  # partition, fold_id, origin, excluded, etc.
├── _features (Features)
│   └── sources: List[FeatureSource]  # one per data source
│       └── processings: List[Processing]  # transformation chains
│           └── data: np.ndarray  # shape (n_samples, n_features)
├── _targets (Targets)
│   └── processings: Dict[str, np.ndarray]  # "numeric", "scaled_MinMaxScaler", etc.
├── _metadata (Metadata)
│   └── columns: Dict[str, List]  # ID, Group, custom columns
├── _folds: List[Tuple[train_indices, val_indices]]
├── _signal_types: List[SignalType]  # per-source signal type
└── _aggregate_column: Optional[str]  # for prediction aggregation
```

### Multi-Dimensional Indexing

**Sources** (data provenance):
- Multiple sensors/modalities (e.g., NIR, Raman, markers)
- Each source has independent feature headers/units
- Accessed via `dataset.n_sources`, `dataset.signal_type(source_idx)`

**Transformations/Processings** (per source):
- Transformation chains: `["raw"]`, `["raw", "SNV"]`, `["raw", "SNV", "D1"]`
- New processings added via `dataset.add_processing(source, name, data)`
- Selected via `context.selector.processing = [["raw", "SNV"]]`

**Repetitions** (samples with same target):
- Multiple measurements per physical sample
- Tracked via `origin` in indexer (augmented samples point to origin)
- Aggregation: `dataset.aggregate = "ID"` or `"y"`

### Target Transformation Chain

Targets follow a similar transformation pattern:

```python
dataset._targets.processings = {
    "numeric": np.array([...]),               # Original
    "scaled_MinMaxScaler": np.array([...]),   # After y_processing
    "encoded_LabelEncoder_001": np.array([...])  # Classification
}
```

The chain is tracked in `context.state.y_processing` to enable inverse transforms.

---

## Views and Layouts

### Layout Modes

| Layout | Shape | Use Case |
|--------|-------|----------|
| `"2d"` | `(samples, features)` | Standard ML models (sklearn, XGBoost) |
| `"3d"` | `(samples, processings, features)` | CNNs, attention models |
| `"native"` | `List[np.ndarray]` per source | Multi-head models |

### View Construction

```python
# 2D layout with source concatenation
X = dataset.x(selector, layout="2d", concat_source=True)
# Shape: (n_samples, sum(features_per_source))

# 3D layout without concatenation
X = dataset.x(selector, layout="3d", concat_source=False)
# Returns: [array(samples, procs, feats_src0), array(samples, procs, feats_src1), ...]

# Native layout (for multi-input models)
X = dataset.x(selector, layout="native")
# Returns: {"NIR": array, "Raman": array, "markers": array}
```

### Selector-Based Filtering

The `DataSelector` drives all data access:

```python
selector = DataSelector(
    partition="train",           # "train", "test", "val", "all"
    processing=[["raw", "SNV"]], # Per-source processing chains
    layout="2d",
    fold_id=0,                   # For CV iteration
    include_augmented=True,      # Include augmented samples?
    include_excluded=False,      # Include outlier-excluded samples?
    branch_id=1,                 # Branch context (0-indexed)
    branch_path=[0, 1],          # Nested branch path
)
X = dataset.x(selector)
```

---

## Operator Execution Patterns

### Controller Pattern

All step logic is implemented via controllers:

```python
@register_controller
class TransformController(OperatorController):
    priority = 100  # Lower = higher priority in matching

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        # Match if operator is TransformerMixin
        return isinstance(operator, TransformerMixin)

    def execute(self, step_info, dataset, context, runtime_context, ...):
        # 1. Get features
        X = dataset.x(context.selector)

        # 2. Fit/transform
        operator.fit(X, y)
        X_transformed = operator.transform(X)

        # 3. Update dataset
        dataset.add_processing(source, name, X_transformed)

        # 4. Register artifact
        artifact = runtime_context.artifact_registry.register(
            step_index=runtime_context.step_number,
            operator=operator,
            fold_id=None,
            ...
        )

        # 5. Return updated context and artifacts
        return context, StepOutput(artifacts=[artifact])
```

### Framework-Specific Patterns

**sklearn TransformerMixin**:
- `TransformController` - standard transformers
- `YProcessingController` - target transformers
- `FeatureSelectionController` - feature selection

**sklearn Model**:
- `BaseModelController` - handles CV loop internally
- Creates N fold artifacts + averaged predictions
- Registers predictions in `prediction_store`

**TensorFlow/PyTorch/JAX**:
- `TFModelController`, `TorchModelController`, `JaxModelController`
- Dynamic model building (shape known only at runtime)
- Checkpoint saving/loading

### Serialization Boundaries

All operators pass through `serialize_component()` before execution:

```python
# Input (Python objects)
pipeline = [MinMaxScaler(), PLSRegression(n_components=10)]

# After serialization
[
    {"class": "sklearn.preprocessing._data.MinMaxScaler", "params": {}},
    {"class": "sklearn.cross_decomposition._pls.PLSRegression", "params": {"n_components": 10}}
]
```

This enables:
- YAML/JSON pipeline definitions
- Reproducible pipeline hashing
- Artifact identification

---

## Prediction Generation and Persistence

### Prediction Flow

```
┌──────────────────┐
│ BaseModelController │
└────────┬─────────┘
         │ For each fold:
         ▼
┌─────────────────────────────────────────┐
│ 1. Get train/val indices from dataset   │
│ 2. Train model on train partition       │
│ 3. Predict on val partition (OOF)       │
│ 4. Predict on test partition            │
│ 5. Add predictions to prediction_store  │
│ 6. Register model artifact              │
└─────────────────────────────────────────┘
```

### Prediction Record Fields

```python
prediction_store.add_prediction(
    dataset_name="wheat",
    model_name="PLSRegression",
    model_classname="sklearn.cross_decomposition._pls.PLSRegression",
    pipeline_uid="abc123",
    step_idx=3,
    fold_id=0,
    branch_id=1,
    branch_name="snv_pls",
    partition="val",          # "train", "val", "test"
    sample_indices=[...],     # Which samples
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_proba=np.array([...]),  # For classification
    val_score=0.85,
    test_score=0.88,
    metric="rmse",
    scores={"val": {"rmse": 0.15, "r2": 0.92}},
    model_artifact_id="0001:3:0:b0",  # For model loading
    trace_id="...",           # For execution replay
)
```

### OOF Reconstruction

For stacking/merging, predictions are reconstructed from OOF (out-of-fold) values:

```python
# TrainingSetReconstructor in controllers/models/stacking.py
reconstructor = TrainingSetReconstructor(
    prediction_store=prediction_store,
    source_model_names=["PLS", "RF"],
)
result = reconstructor.reconstruct(dataset, context)
# result.X_train_meta: OOF predictions for train samples
# result.X_test_meta: Averaged test predictions
```

This prevents data leakage when merged predictions are used for downstream training.

---

## Pipeline Execution Flow

### Single Pipeline Execution (PipelineExecutor)

```python
# In executor.py
def _execute_steps(self, steps, dataset, context, runtime_context, ...):
    for step in steps:
        self.step_number += 1
        context = context.with_step_number(self.step_number)

        # Check for branch mode
        branch_contexts = context.custom.get("branch_contexts", [])

        if branch_contexts and not is_branch_step and not is_merge_step:
            # Execute on each branch
            context = self._execute_step_on_branches(...)
        else:
            # Normal execution
            context = self._execute_single_step(...)

    return context
```

### Branch Mode Execution

When a `branch` step is encountered:

1. **BranchController.execute()**:
   - Parses branch definitions
   - For each branch: creates isolated context copy
   - Executes branch substeps sequentially
   - Stores branch contexts in `context.custom["branch_contexts"]`

2. **Post-branch steps**:
   - Executor detects `branch_contexts` is non-empty
   - Each step is executed N times (once per branch)
   - Branch-specific artifacts are tagged with `branch_id`

3. **MergeController.execute()**:
   - Collects features/predictions from branches
   - Concatenates into unified dataset
   - Clears `branch_contexts` (exits branch mode)

### Fold Iteration

For cross-validation, the model controller handles fold iteration internally:

```python
# In BaseModelController
for fold_idx, (train_indices, val_indices) in enumerate(folds):
    # Get partition data
    X_train = dataset.x(context.with_partition("train"))
    y_train = dataset.y(context.with_partition("train"))

    # Train
    model.fit(X_train, y_train)

    # Predict val (OOF)
    X_val = dataset.x(context.with_fold(fold_idx))
    y_pred_val = model.predict(X_val)

    # Store predictions
    prediction_store.add_prediction(fold_id=fold_idx, partition="val", ...)

    # Register artifact
    artifact_registry.register(step_index, model, fold_id=fold_idx, ...)
```

---

## Generator Behavior

### Current Generator System

The generator expands pipeline specifications into concrete variants **before** execution:

```python
# In pipeline_config.py
class PipelineConfigs:
    def __init__(self, definition, ...):
        self.steps = self._load_steps(definition)
        self.steps = serialize_component(self.steps)

        if self._has_gen_keys(self.steps):
            # Expand _or_, _range_, etc.
            expanded = expand_spec_with_choices(self.steps)
            self.steps = [config for config, choices in expanded]
            self.generator_choices = [choices for config, choices in expanded]
        else:
            self.steps = [self.steps]
```

### Generator Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `_or_` | Choice between alternatives | `{"_or_": [SNV(), MSC()]}` → 2 configs |
| `_range_` | Numeric sequence | `{"_range_": [5, 15, 5]}` → [5, 10, 15] |
| `_log_range_` | Logarithmic sequence | `{"_log_range_": [0.001, 1, 4]}` |
| `_grid_` | Cartesian product | `{"_grid_": {"x": [1, 2], "y": ["A", "B"]}}` |
| `pick` | Combinations | `{"_or_": [A, B, C], "pick": 2}` → C(3,2) |
| `count` | Limit variants | `{"_or_": [...], "count": 5}` |

### Branch vs Generator

Currently, **generator expansion happens statically** at `PipelineConfigs` initialization. Branch expansion happens **dynamically** at runtime via `BranchController`.

**Key insight for DAG**: Generator syntax inside branches is handled by `BranchController`:

```python
# In branch.py
def _parse_branch_definitions(self, step_info):
    raw_def = step_info.original_step.get("branch")

    # Handle generator syntax within branch
    if isinstance(raw_def, dict) and is_generator_node(raw_def):
        return self._expand_generator_branches(raw_def)
```

This means generators within branches become **runtime branches**, not separate pipelines.

---

## DAG Compatibility Analysis

### What Maps Naturally to Nodes/Edges

| Current Concept | DAG Mapping | Notes |
|-----------------|-------------|-------|
| Pipeline step | Node | 1:1 mapping |
| Step output → next step input | Edge | Implicit today, explicit in DAG |
| Branch | Fork node | One input, N outputs |
| Merge | Join node | N inputs, one output |
| Fold iteration | Sub-graph per fold | Or: fold models as branches |
| Source branch | Fork by data provenance | Orthogonal to pipeline branches |

### Currently Sequential but Could Be DAG

1. **Feature augmentation**: `{"feature_augmentation": [SNV(), D1()]}`
   - Currently: sequential loop adding each transformer
   - DAG: parallel nodes, merge at end

2. **Multi-source processing**: `{"source_branch": {...}}`
   - Currently: sequential per-source
   - DAG: parallel per-source, merge

3. **Cross-validation folds**:
   - Currently: loop inside model controller
   - DAG: N parallel fold nodes → merge (average) node

4. **Generator-based branches**: `{"branch": {"_or_": [A, B, C]}}`
   - Currently: runtime branch expansion
   - DAG: explicit fork → N parallel paths → merge

### Runtime-Only Construction Requirements

**Critical**: Some nodes cannot be pre-defined because their structure depends on earlier transformations:

1. **TensorFlow/PyTorch model building**:
   - Input shape unknown until preprocessing completes
   - Build function called with actual `input_shape` at runtime
   - **DAG implication**: Model node is a "placeholder" until input shape is available

2. **Auto-operators** (e.g., `AutoPreprocessor`):
   - Selects transformer based on data characteristics
   - Injects selected transformer into pipeline
   - **DAG implication**: Dynamic node insertion

3. **Feature selection with threshold**:
   - Number of output features unknown until after `fit()`
   - Downstream nodes depend on this shape
   - **DAG implication**: Edge cardinality determined at runtime

4. **Conditional branches**:
   - `{"condition": {"if": expr, "then": [...], "else": [...]}}`
   - Branch selection based on runtime evaluation
   - **DAG implication**: Conditional edge activation

### Current Assumptions That Would Break in DAG

1. **Sequential step numbering**: `step_number = 1, 2, 3, ...`
   - DAG: Nodes may execute in parallel; need topological ordering
   - **Solution**: Use node ID, preserve step_number for artifact compatibility

2. **Single "current" context**: `context = context.with_...`
   - DAG: Multiple concurrent contexts (one per active branch)
   - **Solution**: Context per edge, merge node combines contexts

3. **Implicit data flow**: Next step gets previous step's dataset
   - DAG: Explicit edge defines which node's output feeds which node's input
   - **Solution**: Edge metadata specifies source slot → target slot

4. **Branch contexts in custom dict**: `context.custom["branch_contexts"]`
   - DAG: Branch contexts are first-class parallel execution paths
   - **Solution**: DAG executor maintains active branches natively

5. **Fold iteration inside controller**: Model controller loops over folds
   - DAG: Could model each fold as a parallel sub-graph
   - **Solution**: Hybrid - keep fold iteration for backward compat, expose as branches optionally

### Compatibility Summary

| Aspect | Compatibility | Migration Effort |
|--------|---------------|------------------|
| Controller pattern | ✅ High | Minimal - controllers become node executors |
| Operator serialization | ✅ High | No change - already JSON-serializable |
| Prediction store | ✅ High | No change - branch/fold provenance already tracked |
| Artifact system | ✅ High | Minor - add node_id to artifact metadata |
| Dataset views | ✅ High | No change - selector-based access preserved |
| Generator syntax | ⚠️ Medium | Move to runtime branch expansion |
| Branch/merge | ⚠️ Medium | Formalize as fork/join nodes |
| Runtime shape changes | ⚠️ Medium | Support placeholder nodes with deferred configuration |
| Sequential step numbering | ⚠️ Medium | Add node_id, preserve step_number for compat |

---

## Conclusion

The current nirs4all architecture is **highly compatible** with a DAG transformation because:

1. **Controller pattern** already separates step logic from execution orchestration
2. **Payload components** (dataset, predictions, artifacts, context) are well-isolated
3. **Branch/merge** semantics exist and can be formalized as fork/join
4. **Prediction provenance** already tracks fold, branch, step information
5. **Serialization** is complete for all operators

Key challenges for DAG migration:
1. Supporting **dynamic graph expansion** for runtime-only construction (TF models, auto-operators)
2. Formalizing **fold-as-branches** while preserving backward compatibility
3. Moving generator expansion from static (pre-execution) to dynamic (in-DAG) where beneficial
4. Maintaining the **linear syntax** that avoids explicit ancestor declarations
