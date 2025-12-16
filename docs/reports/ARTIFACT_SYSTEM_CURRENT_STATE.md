# Artifact System - Current State Analysis

**Date**: December 15, 2025
**Author**: Technical Analysis
**Status**: Comprehensive Analysis - Foundation for Redesign

---

## 1. Executive Summary

This document provides an exhaustive analysis of the current artifact management system in nirs4all. It serves as the foundation for designing a unified, once-and-for-all artifact system that handles all edge cases correctly.

### Current Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Basic model persistence | ✅ Working | Single models, single source |
| Transformer persistence | ✅ Working | X transformers per processing |
| Y-processing persistence | ✅ Working | Encoders and scalers |
| Fold-specific models | ✅ Working | CV with fold averaging |
| Branching (training) | ⚠️ Partial | Works but trace incomplete |
| Branching (reload) | ❌ Broken | Artifact matching fails |
| Multi-source (training) | ✅ Working | Multiple X arrays |
| Multi-source (reload) | ❌ Broken | Operation counter mismatch |
| Meta-model stacking | ⚠️ Partial | Signature issues |
| Nested branches | ❌ Broken | Branch path not tracked |
| Subpipeline models | ⚠️ Partial | Q15 JAX fails |
| Bundle export/import | ⚠️ Partial | Depends on correct IDs |

### Root Cause Summary

The fundamental problem is **incomplete execution path tracking**:
1. Branch substeps are not recorded individually in the execution trace
2. Artifact IDs don't fully encode the execution path for reload disambiguation
3. Operation counters diverge between training and prediction modes
4. No concept of "operator chain" for deterministic replay

---

## 2. Architecture Overview

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARTIFACT SYSTEM V2                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │   ArtifactRegistry   │    │    ArtifactLoader    │                       │
│  │   (Training Mode)    │◄──►│   (Predict Mode)     │                       │
│  └──────────┬───────────┘    └───────────┬──────────┘                       │
│             │                            │                                  │
│             ▼                            ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    Centralized Binaries Storage                  │       │
│  │                    workspace/binaries/<dataset>/                 │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │   ExecutionTrace     │    │    MinimalPipeline   │                       │
│  │   (Recording)        │───►│    (Replay)          │                       │
│  └──────────────────────┘    └──────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Files and Locations

| Component | File | Role |
|-----------|------|------|
| `ArtifactRegistry` | `pipeline/storage/artifacts/artifact_registry.py` | Training: ID generation, persistence, deduplication |
| `ArtifactLoader` | `pipeline/storage/artifacts/artifact_loader.py` | Prediction: Load from manifest, cache |
| `ArtifactRecord` | `pipeline/storage/artifacts/types.py` | Data structure for artifact metadata |
| `artifact_persistence` | `pipeline/storage/artifacts/artifact_persistence.py` | Serialization/deserialization |
| `utils` | `pipeline/storage/artifacts/utils.py` | ID generation, parsing, hashing |
| `ExecutionTrace` | `pipeline/trace/execution_trace.py` | Records execution path |
| `TraceRecorder` | `pipeline/trace/recorder.py` | Records trace during training |
| `TraceBasedExtractor` | `pipeline/trace/extractor.py` | Extracts minimal pipeline |
| `MinimalPredictor` | `pipeline/minimal_predictor.py` | Executes minimal pipeline |

---

## 3. Artifact ID System

### 3.1 Current Format

```
{pipeline_id}:{branch_path}:{step_index}.{sub_index}:{fold_id}

Examples:
- "0001_pls:3:all"           # Step 3, no branch, shared across folds
- "0001_pls:0:3:0"           # Branch 0, step 3, fold 0
- "0001_pls:0:2:3:all"       # Branch [0,2], step 3, shared
- "0001_pls:3.1:all"         # Step 3, sub_index 1, shared
```

### 3.2 ID Generation (Training)

```python
# In ArtifactRegistry.generate_id()
def generate_id(
    pipeline_id: str,
    branch_path: List[int],
    step_index: int,
    fold_id: Optional[int] = None,
    sub_index: Optional[int] = None
) -> str:
    parts = [pipeline_id]
    for branch_idx in branch_path:
        parts.append(str(branch_idx))

    if sub_index is not None:
        parts.append(f"{step_index}.{sub_index}")
    else:
        parts.append(str(step_index))

    fold_str = str(fold_id) if fold_id is not None else "all"
    parts.append(fold_str)

    return ":".join(parts)
```

### 3.3 ID Parsing (Prediction)

```python
# In utils.parse_artifact_id()
def parse_artifact_id(artifact_id: str) -> Tuple[str, List[int], int, Optional[int], Optional[int]]:
    parts = artifact_id.split(":")
    pipeline_id = parts[0]
    fold_str = parts[-1]
    fold_id = None if fold_str == "all" else int(fold_str)

    step_str = parts[-2]
    sub_index = None
    if "." in step_str:
        step_part, sub_part = step_str.split(".", 1)
        step_index = int(step_part)
        sub_index = int(sub_part)
    else:
        step_index = int(step_str)

    branch_path = []
    if len(parts) > 3:
        branch_parts = parts[1:-2]
        branch_path = [int(b) for b in branch_parts]

    return pipeline_id, branch_path, step_index, fold_id, sub_index
```

### 3.4 Problems with Current ID System

1. **Ambiguous for nested structures**: Subpipeline models (e.g., `[model1, model2]`) use `sub_index` but branch substeps don't
2. **No chain concept**: Can't reconstruct the full path of operators that led to this artifact
3. **Context-dependent interpretation**: Same ID means different things in different pipeline structures
4. **Source index missing**: Multi-source transformers need source identification

---

## 4. Execution Trace System

### 4.1 Purpose

The ExecutionTrace records the exact sequence of operations during training, enabling deterministic replay during prediction.

### 4.2 Data Structures

```python
@dataclass
class StepArtifacts:
    artifact_ids: List[str]           # All artifacts at this step
    primary_artifact_id: Optional[str] # Main artifact (model)
    fold_artifact_ids: Dict[int, str]  # Per-fold artifacts
    metadata: Dict[str, Any]

@dataclass
class ExecutionStep:
    step_index: int
    operator_type: str              # "transform", "model", "branch", etc.
    operator_class: str             # "PLSRegression", "SNV", etc.
    operator_config: Dict[str, Any]
    execution_mode: StepExecutionMode
    artifacts: StepArtifacts
    branch_path: List[int]          # Current branch context
    branch_name: str
    duration_ms: float
    metadata: Dict[str, Any]

@dataclass
class ExecutionTrace:
    trace_id: str
    pipeline_uid: str
    steps: List[ExecutionStep]
    model_step_index: Optional[int]
    fold_weights: Optional[Dict[int, float]]
    preprocessing_chain: str
    metadata: Dict[str, Any]
```

### 4.3 Recording Process

```python
# In TraceRecorder
class TraceRecorder:
    def start_step(self, step_index, operator_type, operator_class, operator_config, branch_path):
        step = ExecutionStep(
            step_index=step_index,
            operator_type=operator_type,
            operator_class=operator_class,
            operator_config=operator_config,
            branch_path=branch_path or []
        )
        self._current_step = step

    def record_artifact(self, artifact_id, is_primary=False, fold_id=None):
        if self._current_step:
            self._current_step.artifacts.add_artifact(artifact_id, is_primary)
            if fold_id is not None:
                self._current_step.artifacts.add_fold_artifact(fold_id, artifact_id)

    def end_step(self):
        self._trace.add_step(self._current_step)
        self._current_step = None
```

### 4.4 Problems with Trace Recording

**Critical Issue: Branch substeps not individually recorded**

```
During Training:
  Step 1: MinMaxScaler     → recorded as step 1
  Step 2: ShuffleSplit     → recorded as step 2
  Step 3: Branch           → recorded as step 3 (ALL substep artifacts lumped here)
    ├─ SNV (branch_0)      → NOT recorded as separate step
    └─ SavGol (branch_1)   → NOT recorded as separate step
  Step 4: PLSRegression    → recorded as step 4

The trace shows step 3 with artifact_ids: [
    "0001:0:3.1:all",  # SNV branch 0
    "0001:0:3.2:all",  # SNV branch 0 source 1
    "0001:0:3.3:all",  # SNV branch 0 source 2
    "0001:1:3.4:all",  # SavGol branch 1
    "0001:1:3.5:all",  # SavGol branch 1 source 1
    "0001:1:3.6:all",  # SavGol branch 1 source 2
]

But the step's branch_path is [] (empty) - making filtering impossible!
```

---

## 5. Controller Artifact Handling

### 5.1 BaseModelController._persist_model()

```python
def _persist_model(
    self,
    runtime_context: 'RuntimeContext',
    model: Any,
    model_id: str,
    branch_id: Optional[int] = None,
    branch_name: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
    fold_id: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    custom_name: Optional[str] = None
) -> 'ArtifactMeta':
    # Use artifact registry if available (v2 system)
    if runtime_context.artifact_registry is not None:
        registry = runtime_context.artifact_registry
        pipeline_id = runtime_context.saver.pipeline_id
        step_index = runtime_context.step_number

        bp = branch_path or ([branch_id] if branch_id is not None else [])
        sub_index = runtime_context.substep_number if runtime_context.substep_number >= 0 else None

        artifact_id = registry.generate_id(
            pipeline_id=pipeline_id,
            branch_path=bp,
            step_index=step_index,
            fold_id=fold_id,
            sub_index=sub_index
        )

        record = registry.register(
            obj=model,
            artifact_id=artifact_id,
            artifact_type=ArtifactType.MODEL,
            params=params or {},
            format_hint=format_hint,
            custom_name=custom_name
        )

        # Record in execution trace
        runtime_context.record_step_artifact(
            artifact_id=artifact_id,
            is_primary=(fold_id is None),
            fold_id=fold_id,
            metadata={"class_name": model.__class__.__name__}
        )

        return record
```

### 5.2 TransformerMixinController._persist_transformer()

```python
def _persist_transformer(
    self,
    runtime_context: 'RuntimeContext',
    transformer: Any,
    name: str,
    context: ExecutionContext
) -> Any:
    if runtime_context.artifact_registry is not None:
        registry = runtime_context.artifact_registry
        pipeline_id = runtime_context.saver.pipeline_id
        step_index = runtime_context.step_number
        branch_path = context.selector.branch_path or []

        # Extract operation counter from name for sub_index
        sub_index = None
        if "_" in name:
            try:
                sub_index = int(name.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                pass

        artifact_id = registry.generate_id(
            pipeline_id=pipeline_id,
            branch_path=branch_path,
            step_index=step_index,
            fold_id=None,  # Transformers shared across folds
            sub_index=sub_index
        )

        record = registry.register(
            obj=transformer,
            artifact_id=artifact_id,
            artifact_type=ArtifactType.TRANSFORMER,
            format_hint='sklearn'
        )

        runtime_context.record_step_artifact(
            artifact_id=artifact_id,
            is_primary=False,
            fold_id=None,
            metadata={"class_name": transformer.__class__.__name__, "name": name}
        )

        return record
```

### 5.3 BranchController.execute() - The Problem Source

```python
def execute(
    self,
    step_info: "ParsedStep",
    dataset: "SpectroDataset",
    context: "ExecutionContext",
    runtime_context: "RuntimeContext",
    ...
) -> Tuple["ExecutionContext", StepOutput]:

    branch_defs = self._parse_branch_definitions(step_info)

    for idx, branch_def in enumerate(branch_defs):
        branch_id = idx
        branch_name = branch_def.get("name", f"branch_{branch_id}")
        branch_steps = branch_def.get("steps", [])

        # Create branch context
        branch_context = initial_context.copy()
        new_branch_path = parent_branch_path + [branch_id]
        branch_context.selector = branch_context.selector.with_branch(
            branch_id=branch_id,
            branch_name=branch_name,
            branch_path=new_branch_path
        )

        # Execute branch steps
        for substep in branch_steps:
            if runtime_context.step_runner:
                runtime_context.substep_number += 1  # ← substep tracking
                result = runtime_context.step_runner.execute(
                    step=substep,
                    dataset=dataset,
                    context=branch_context,
                    runtime_context=runtime_context,
                    ...
                )
                branch_context = result.updated_context
                all_artifacts.extend(result.artifacts)  # ← Artifacts collected here

    # Problem: artifacts are returned but trace recorded as ONE step
    return result_context, StepOutput(artifacts=all_artifacts, ...)
```

**The branch controller executes substeps but:**
1. All substep artifacts are lumped under the parent branch step
2. Trace shows branch step with `branch_path: []` (parent context)
3. Individual substep execution is not recorded separately

---

## 6. Prediction Mode Flow

### 6.1 Minimal Pipeline Extraction

```python
# In TraceBasedExtractor.extract()
def extract(self, trace: ExecutionTrace, full_pipeline: List[Any]) -> MinimalPipeline:
    minimal = MinimalPipeline(
        trace_id=trace.trace_id,
        model_step_index=trace.model_step_index,
        fold_weights=trace.fold_weights,
        ...
    )

    trace_steps = trace.get_steps_up_to_model()

    for exec_step in trace_steps:
        # Skip skipped steps
        if exec_step.execution_mode == StepExecutionMode.SKIP:
            continue

        # Get config from full pipeline
        step_config = full_pipeline[exec_step.step_index - 1]

        minimal_step = MinimalPipelineStep(
            step_index=exec_step.step_index,
            step_config=step_config,
            execution_mode=StepExecutionMode.PREDICT,
            artifacts=exec_step.artifacts,  # ← All branch artifacts here
            branch_path=list(exec_step.branch_path),  # ← Often empty!
            ...
        )

        minimal.steps.append(minimal_step)
        minimal.artifact_map[exec_step.step_index] = exec_step.artifacts

    return minimal
```

### 6.2 MinimalArtifactProvider

```python
class MinimalArtifactProvider(ArtifactProvider):
    def get_artifacts_for_step(
        self,
        step_index: int,
        branch_path: Optional[List[int]] = None,
        branch_id: Optional[int] = None
    ) -> List[Tuple[str, Any]]:

        step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
        if not step_artifacts:
            return []

        # Determine target branch
        target_branch = None
        if branch_path and len(branch_path) > 0:
            target_branch = branch_path[0]
        elif branch_id is not None:
            target_branch = branch_id

        results = []
        for artifact_id in step_artifacts.artifact_ids:
            # Filter by branch using artifact_id parsing
            if target_branch is not None:
                artifact_branch = self._parse_branch_from_artifact_id(artifact_id)
                if artifact_branch is not None and artifact_branch != target_branch:
                    continue  # Skip wrong branch

            obj = self._load_artifact(artifact_id)
            if obj is not None:
                operator_name = self._derive_operator_name(obj, artifact_id, step_index)
                results.append((operator_name, obj))

        return results
```

### 6.3 Controller Artifact Loading (Prediction)

```python
# In TransformerMixinController.execute() - prediction mode
if mode == "predict" or mode == "explain":
    transformer = None

    # Try artifact_provider first (Phase 4 approach)
    if runtime_context.artifact_provider is not None:
        step_index = runtime_context.step_number
        step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
            step_index,
            branch_path=context.selector.branch_path
        )
        if step_artifacts:
            artifacts_dict = dict(step_artifacts)
            transformer = artifacts_dict.get(new_operator_name)
            if transformer is None:
                # Fallback: search by class name
                transformer = _find_transformer_by_class(
                    operator_name, artifacts_dict, transformer_load_index
                )

    # Fallback: loaded_binaries (legacy)
    if transformer is None and loaded_binaries:
        binaries_dict = dict(loaded_binaries)
        transformer = binaries_dict.get(new_operator_name)
        if transformer is None:
            transformer = _find_transformer_by_class(
                operator_name, binaries_dict, transformer_load_index
            )

    if transformer is None:
        raise ValueError(f"Binary for {new_operator_name} not found")
```

---

## 7. Known Edge Cases and Failures

### 7.1 Q15 JAX Models - Subpipeline with Multiple Models

**Pipeline Structure:**
```python
pipeline = [
    StandardScaler,
    ShuffleSplit(n_splits=2),
    [  # Subpipeline with 2 models
        {"model": JaxMLPRegressor(features=[64, 32]), ...},
        {"model": nicon_jax, ...},
    ]
]
```

**Problem:** Both models execute at the same step_number, differentiated only by substep_number.
When reloading, the artifact_id parsing must correctly identify which model to load.

**Current Handling:**
- `sub_index` in artifact_id: `0001:3.0:0` vs `0001:3.1:0`
- `MinimalArtifactProvider.target_sub_index` filters by sub_index

**Failure Point:** When `model_artifact_id` is missing from prediction record (e.g., for avg/w_avg),
falls back to name-based matching which can be ambiguous.

### 7.2 Branching + Multi-source Reload

**Pipeline Structure:**
```python
pipeline = [
    MinMaxScaler(),  # Per-source transformer
    ShuffleSplit(n_splits=2),
    {"branch": [
        [SNV()],       # Branch 0: SNV per source
        [SavGol()],    # Branch 1: SavGol per source
    ]},
    PLSRegression(n_components=5),
]
```

**Training Creates:**
- Step 1: 3 MinMaxScaler artifacts (one per source)
- Step 3: 6 transformer artifacts (2 branches × 3 sources)
- Step 4: 4 PLS artifacts (2 branches × 2 folds)

**Reload Problem:**
When reloading branch 0, needs to load only:
- Step 1: All 3 MinMaxScaler (shared)
- Step 3: Only 3 SNV (branch 0)
- Step 4: Only 2 PLS (branch 0)

But trace step 3 has `branch_path: []`, so filtering fails.

### 7.3 Nested Branches

**Pipeline Structure:**
```python
pipeline = [
    {"branch": [
        [SNV(), {"branch": [  # Nested branch
            [PCA(10)],
            [PCA(20)],
        ]}],
        [MSC()],
    ]},
    PLSRegression(),
]
```

**Problem:** Branch path `[0, 0]` vs `[0, 1]` not properly tracked through nested branches.

### 7.4 Meta-Model with Branches

**Pipeline Structure:**
```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression()],
        [MSC(), RandomForest()],
    ]},
    MetaModel(Ridge(), source_models="all"),
]
```

**Problem:** MetaModel must collect predictions from both branches, requiring cross-branch artifact access.

### 7.5 Loaded Bundle Artifacts

When a pipeline uses a pre-trained bundle:
```python
pipeline = [
    load_bundle("wheat_pls_model"),  # Pre-trained
    AdditionalProcessing(),
]
```

**Problem:** Bundle artifacts have different pipeline_id, need to be merged into current trace.

---

## 8. Artifact Types and Their Behavior

### 8.1 ArtifactType Enum

```python
class ArtifactType(str, Enum):
    MODEL = "model"           # Trained ML models
    TRANSFORMER = "transformer"  # Fitted preprocessors
    SPLITTER = "splitter"     # Train/test split config
    ENCODER = "encoder"       # Y-scalers, label encoders
    META_MODEL = "meta_model" # Stacking meta-models
```

### 8.2 Type-Specific Behavior

| Type | Fold-Specific | Branch-Specific | Source-Specific | Dependencies |
|------|---------------|-----------------|-----------------|--------------|
| MODEL | Yes (per-fold) | Yes | No | None |
| TRANSFORMER | No (shared) | Yes | Yes | None |
| SPLITTER | No | No | No | None |
| ENCODER | No (shared) | Yes | No | None |
| META_MODEL | Yes (per-fold) | Yes | No | Source models |

---

## 9. Operation Counter System

### 9.1 Current Implementation

```python
# In RuntimeContext
class RuntimeContext:
    operation_count: int = 0

    def next_op(self) -> int:
        self.operation_count += 1
        return self.operation_count
```

### 9.2 Usage Patterns

**Transformers:** Name includes operation counter for uniqueness
```python
new_operator_name = f"{operator_name}_{runtime_context.next_op()}"
# e.g., "MinMaxScaler_1", "MinMaxScaler_2", "MinMaxScaler_3"
```

**Models:** Used for prediction record identification
```python
op_counter = runner.next_op()
prediction_data['op_counter'] = op_counter
```

### 9.3 The Counter Divergence Problem

**Training:**
```
Step 1: MinMaxScaler → op 1, 2, 3 (3 sources)
Step 2: ShuffleSplit → no ops
Step 3: Branch 0: SNV → op 4, 5, 6; Branch 1: SavGol → op 7, 8, 9
Step 4: Branch 0: PLS → op 10, 11; Branch 1: PLS → op 12, 13
```

**Prediction (branch 0 only):**
```
Step 1: MinMaxScaler → op 1, 2, 3
Step 2: ShuffleSplit → no ops
Step 3: SNV → op 4, 5, 6 ← SAME (correct)
Step 4: PLS → op 7, 8 ← WRONG! (expected 10, 11)
```

The counter doesn't skip branch 1 operations, causing name mismatch.

---

## 10. Manifest Storage Format

### 10.1 Artifacts Section (v2)

```yaml
artifacts:
  schema_version: "2.0"
  items:
    - artifact_id: "0001_pls:1.1:all"
      content_hash: "sha256:abc123..."
      path: "transformer_MinMaxScaler_abc123.joblib"
      pipeline_id: "0001_pls"
      branch_path: []
      step_index: 1
      fold_id: null
      artifact_type: "transformer"
      class_name: "MinMaxScaler"
      custom_name: ""
      depends_on: []
      format: "joblib"
      format_version: "sklearn==1.5.0"
      nirs4all_version: "0.9.0"
      size_bytes: 1234
      created_at: "2025-12-15T10:00:00Z"
      params: {}
      version: 1
```

### 10.2 Execution Traces Section

```yaml
execution_traces:
  abc123:
    trace_id: "abc123"
    pipeline_uid: "0001_pls_abc123"
    created_at: "2025-12-15T10:00:00Z"
    model_step_index: 4
    fold_weights:
      0: 0.5
      1: 0.5
    preprocessing_chain: "MinMaxScaler>SNV"
    steps:
      - step_index: 1
        operator_type: "transform"
        operator_class: "MinMaxScaler"
        execution_mode: "train"
        branch_path: []
        artifacts:
          artifact_ids: ["0001:1.1:all", "0001:1.2:all", "0001:1.3:all"]
          primary_artifact_id: null
          fold_artifact_ids: {}
```

---

## 11. Dependency Graph

### 11.1 Current Implementation

```python
class DependencyGraph:
    def __init__(self):
        self._dependencies: Dict[str, List[str]] = {}  # artifact -> dependencies
        self._dependents: Dict[str, List[str]] = {}    # artifact -> dependents

    def add_dependency(self, artifact_id: str, depends_on: str):
        if artifact_id not in self._dependencies:
            self._dependencies[artifact_id] = []
        if depends_on not in self._dependencies[artifact_id]:
            self._dependencies[artifact_id].append(depends_on)

        # Reverse mapping
        if depends_on not in self._dependents:
            self._dependents[depends_on] = []
        if artifact_id not in self._dependents[depends_on]:
            self._dependents[depends_on].append(artifact_id)

    def resolve_dependencies(self, artifact_id: str) -> List[str]:
        """Topologically sorted dependencies (dependencies before dependents)."""
        ...
```

### 11.2 Usage

Currently only used for:
1. Meta-model source model references
2. Validation that source models exist before meta-model registration

**Not Used For:**
- Transformer → Model dependencies
- Step ordering constraints
- Chain reconstruction

---

## 12. Summary of Issues

### 12.1 Architectural Issues

1. **Trace step granularity too coarse**: Branch substeps not recorded individually
2. **No operator chain concept**: Can't reconstruct the path to an artifact
3. **Operation counter not branch-aware**: Diverges between train and predict
4. **Source index missing from ID**: Multi-source disambiguation fails

### 12.2 Implementation Issues

1. **Branch controller doesn't record substeps**: All artifacts lumped under parent
2. **Artifact filtering relies on ID parsing**: Fragile, context-dependent
3. **Name-based fallback matching**: Ambiguous for custom names
4. **Missing cross-branch references**: MetaModel can't find branch predictions

### 12.3 Data Model Issues

1. **ArtifactRecord lacks chain info**: No parent/predecessor tracking
2. **ExecutionStep lacks substep info**: Can't distinguish branch contents
3. **Manifest doesn't capture full graph**: Only lists artifacts, not relationships

---

## 13. What Works Well

Despite the issues, several aspects work correctly:

1. **Content-addressed deduplication**: Same content → same file
2. **LRU caching in loader**: Efficient repeated access
3. **Cleanup utilities**: Orphan detection and deletion
4. **Simple pipelines**: Single branch, single source, single model
5. **CV fold handling**: Per-fold artifacts and averaging
6. **Y-processing artifacts**: Encoders and scalers tracked correctly
7. **Type classification**: Clear distinction between model/transformer/encoder

---

## 14. Conclusion

The current artifact system has solid foundations but breaks down when:
- Multiple branches exist (trace doesn't track substeps)
- Multiple sources exist with branches (operation counter diverges)
- Subpipelines contain multiple models (sub_index handling incomplete)
- Meta-models need cross-branch access (no unified prediction access)
- Bundles are imported (different pipeline_id namespace)

The solution requires:
1. **Operator chain tracking**: Each artifact knows its full path
2. **Proper substep recording**: Branch substeps as individual trace entries
3. **Source-aware IDs**: Include source index for multi-source
4. **Unified artifact graph**: All relationships explicitly tracked
5. **Deterministic replay**: Same trace → identical execution

See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for the proposed redesign.
