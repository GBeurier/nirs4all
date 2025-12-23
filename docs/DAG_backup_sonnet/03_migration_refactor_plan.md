# Document 3: Adaptation / Refactoring Plan

**Date**: December 2025
**Author**: Migration Planning
**Status**: Implementation Roadmap

---

## Table of Contents

1. [Executive Summary](#executive-summary)
<<<<<<< Updated upstream
2. [Component Migration Analysis](#component-migration-analysis)
3. [Migration Phases](#migration-phases)
4. [Compatibility Strategy](#compatibility-strategy)
5. [Success Criteria](#success-criteria)
6. [Risk Assessment](#risk-assessment)
=======
2. [Key Design Insights](#key-design-insights)
3. [Component Migration Analysis](#component-migration-analysis)
4. [Primitive Implementation Order](#primitive-implementation-order)
5. [Migration Phases](#migration-phases)
6. [Compatibility Strategy](#compatibility-strategy)
7. [Success Criteria](#success-criteria)
8. [Risk Assessment](#risk-assessment)
>>>>>>> Stashed changes

---

## Executive Summary

This document outlines the most efficient migration path from the current sequential pipeline execution to the DAG-based model specified in Document 2. The strategy prioritizes:

1. **Incremental adoption** - Each phase delivers working software
2. **Backward compatibility** - Existing pipelines continue to work
3. **Minimal disruption** - Refactor abstractions, not implementations
4. **Test-driven** - Each phase validated by existing + new tests

**Estimated effort**: 6 phases over ~12-16 weeks

---

<<<<<<< Updated upstream
=======
## Key Design Insights

### 1. Generators = Pre-Runtime Branches

**Insight**: All generator syntax can be expanded to DAG branches before execution.

| Generator Location | Current Behavior | DAG Equivalent |
|--------------------|------------------|----------------|
| Top-level | Creates N separate pipelines | N separate DAG executions |
| Inside `branch` | Expands at runtime | N branches (pre-computed) |
| In `feature_augmentation` | Expands operator list | Fork → apply each → Join(concat) |
| In `sample_augmentation` | Expands operator list | Apply each, add samples |
| In `model` params | N/A | N models → Join(select_best) |
| In `model` list | N/A | N models → Join(select_best) |

**Migration Impact**: Generator expansion logic remains in `PipelineConfigs` and `BranchController`. No runtime node creation needed for generators.

### 2. OOF Safety is Automatic

**Insight**: Once folds are assigned, all data access is automatically OOF-safe.

**Current Implementation** (already works):
- `assign_folds()` sets up train/val partitions per fold
- Controllers request data by partition
- Predictions are stored with fold_id
- OOF reconstruction uses held-out fold predictions

**Migration Impact**: No changes needed. OOF is an invariant of the current design.

### 3. Folds = Implicit Fork/Join

**Insight**: Model training can be modeled as Fork(N folds) → Train each → Join(aggregate).

```
Current ModelController:
  for fold in folds:
    train(fold)
    predict(fold)
  aggregate_predictions()

DAG Equivalent:
  FORK(folds) → [FoldNode_0, FoldNode_1, ...] → JOIN(aggregate_folds)
```

**Migration Impact**:
- Add `VirtualModel` concept for fold aggregation
- Model artifacts include per-fold models + virtual model metadata
- Prediction mode uses virtual model aggregation

### 4. Virtual Model = Aggregated Folds

**Insight**: The "final model" is actually a virtual model that aggregates fold predictions.

```python
# New artifact structure
artifacts = {
    "fold_0": Model,
    "fold_1": Model,
    "fold_2": Model,
    "virtual": VirtualModelMeta(
        strategy="weighted_mean",
        weights={0: 0.35, 1: 0.40, 2: 0.25}  # By validation score
    )
}

# Prediction mode
def predict(X):
    preds = [fold_model.predict(X) for fold_model in fold_models]
    return aggregate(preds, virtual.strategy, virtual.weights)
```

**Migration Impact**: Add `VirtualModel` dataclass. Update artifact serialization. Update prediction mode to use virtual model.

---

>>>>>>> Stashed changes
## Component Migration Analysis

### 1. PipelineRunner / Orchestrator

**Current Responsibility**: Entry point, multi-pipeline/dataset coordination

**Migration Solution**:

```python
# Current signature
class PipelineRunner:
    def run(self, config: PipelineConfigs, datasets: DatasetConfigs, ...) -> Tuple[Predictions, Dict]

# Proposed changes (minimal)
class PipelineRunner:
    def run(self, config: PipelineConfigs, datasets: DatasetConfigs,
            execution_mode: str = "sequential",  # NEW: "sequential" | "dag"
            ...) -> Tuple[Predictions, Dict]

    def _run_sequential(self, ...):
        """Original implementation - calls PipelineExecutor."""
        return self._orchestrator.execute(...)

    def _run_dag(self, ...):
        """New DAG path - calls DAGBuilder + DAGExecutor."""
        dag = DAGBuilder(steps).build()
        executor = DAGExecutor(dag, runtime)
        return executor.execute(initial_payload)
```

**Critiques**:

1. **Critique**: Adding `execution_mode` parameter increases API surface
   - **Mitigation**: Default to "sequential", DAG opt-in. Remove sequential once stable.

2. **Critique**: Two code paths doubles maintenance
   - **Mitigation**: Share StepRunner, controllers, artifacts. Only execution scheduling differs.

3. **Critique**: Users must choose mode
   - **Mitigation**: Auto-detect: if pipeline uses advanced DAG features (conditional, parallel), use DAG mode.

**Final Approach**: Add internal `_run_dag()` path, controlled by `execution_mode` with auto-detection. Orchestrator delegates to appropriate executor.

---

### 2. Controllers

**Current Responsibility**: Execute single step, return updated context + artifacts

**Migration Solution**:

```python
# Current signature (unchanged)
class OperatorController(ABC):
    def execute(
        self,
        step_info: ParsedStep,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Predictions] = None
    ) -> Tuple[ExecutionContext, StepOutput]: ...

# NEW: Wrapper for DAG execution
class DAGNodeAdapter:
    """Adapts existing controller to DAGNode interface."""

    def __init__(self, controller: OperatorController, step_info: ParsedStep):
        self.controller = controller
        self.step_info = step_info

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        # Convert Payload → ExecutionContext
        context = payload_in.to_execution_context()
        dataset = payload_in.get_dataset()

        # Call existing controller
        result_context, output = self.controller.execute(
            step_info=self.step_info,
            dataset=dataset,
            context=context,
            runtime_context=runtime,
            mode=payload_in.mode,
            prediction_store=payload_in.prediction_store
        )

        # Convert result → Payload
        return payload_in.with_updates(result_context, output)
```

**Critiques**:

1. **Critique**: Adapter adds indirection/overhead
   - **Mitigation**: Adapter is thin (< 20 lines). Controllers remain unchanged.

2. **Critique**: Controllers still mutate dataset
   - **Mitigation**: Phase 2 introduces DatasetView; controllers transition gradually.

3. **Critique**: Some controllers (Branch, Merge) have complex return types
   - **Mitigation**: Create specialized adapters: ForkNodeAdapter, JoinNodeAdapter.

**Final Approach**: Keep controllers unchanged. Introduce `DAGNodeAdapter` to bridge controller interface to DAGNode. Specialized adapters for Branch/Merge.

---

### 3. Dataset (SpectroDataset)

**Current Responsibility**: Mutable container for features, targets, metadata

**Migration Solution**:

```python
# Phase 1: Add append-only processing registration (non-breaking)
class SpectroDataset:
    def register_processing(
        self,
        source: int,
        processing_name: str,
        features: np.ndarray,
        parent_processing: Optional[str] = None
    ) -> str:
        """Append-only registration of new processing.

        Returns the full processing chain name.
        Existing update_features() calls this internally.
        """
        chain_name = f"{parent_processing}>{processing_name}" if parent_processing else processing_name
        self._features.sources[source].append_processing(chain_name, features)
        return chain_name

# Phase 2: Add DatasetView for immutable access
@dataclass(frozen=True)
class DatasetView:
    dataset_id: str                      # Reference to shared dataset
    processing: Tuple[Tuple[str, ...]]   # Immutable per-source
    partition: str
    sample_mask: Optional[Tuple[int, ...]]

    def materialize(self, dataset: SpectroDataset) -> np.ndarray:
        """Convert view to actual array."""
        selector = self.to_selector()
        return dataset.x(selector)
```

**Critiques**:

1. **Critique**: Two access patterns (direct vs view) is confusing
   - **Mitigation**: Document clearly. Views are for DAG mode, direct for backward compat.

2. **Critique**: Append-only storage increases memory
   - **Mitigation**: Only store transformed arrays, not copies. Add optional GC.

3. **Critique**: Processing chain names can get long
   - **Mitigation**: Use hash-based shortnames internally, full names for debugging.

**Final Approach**:
- Phase 1: Add `register_processing()` as thin wrapper around existing mutation
- Phase 2: Add `DatasetView` for DAG mode
- No changes to existing `update_features()` callers

---

### 4. Operator Execution (StepRunner)

**Current Responsibility**: Parse step, route to controller, execute

**Migration Solution**:

```python
# StepRunner remains unchanged - it's already "node-like"

# NEW: DAGBuilder creates nodes from steps
class DAGBuilder:
    def __init__(self, steps: List[Any], parser: StepParser, router: ControllerRouter):
        self.steps = steps
        self.parser = parser
        self.router = router

    def build(self) -> DAG:
        """Convert linear steps to DAG structure."""
        dag = DAG()
        prev_node_id = "START"

        for step_idx, step in enumerate(self.steps):
            parsed = self.parser.parse(step)
            controller = self.router.route(parsed, step)

            if parsed.keyword == "branch":
                node = self._create_fork_node(step_idx, parsed, controller, prev_node_id)
            elif parsed.keyword in ("merge", "merge_sources", "merge_predictions"):
                node = self._create_join_node(step_idx, parsed, controller, prev_node_id)
            else:
                node = self._create_operator_node(step_idx, parsed, controller, prev_node_id)

            dag.add_node(node)
            prev_node_id = node.node_id

        dag.finalize(prev_node_id)  # Connect to END
        return dag
```

**Critiques**:

1. **Critique**: Duplicates parsing logic from StepRunner
   - **Mitigation**: DAGBuilder uses StepRunner's parser/router directly.

2. **Critique**: Linear syntax doesn't expose parallel opportunities
   - **Mitigation**: DAGBuilder can analyze steps for implicit parallelism (multi-source).

3. **Critique**: Branch/merge detection is fragile
   - **Mitigation**: Use controller.matches() or add node_type() method to controllers.

**Final Approach**: DAGBuilder reuses StepParser and ControllerRouter. StepRunner unchanged (used inside node adapters).

---

### 5. Predictions

**Current Responsibility**: Store, query, rank predictions

**Migration Solution**:

```python
# Predictions class unchanged - already shared/append-only

# NEW: Add DAG-aware query methods
class Predictions:
    def get_by_node_id(self, node_id: str, partition: str = "val") -> List[Dict]:
        """Query predictions by DAG node ID."""
        # node_id format: "step_3_branch_0"
        parts = node_id.split("_")
        step_idx = int(parts[1])
        branch_id = int(parts[3]) if len(parts) > 3 else None

        return self.filter_predictions(
            step_idx=step_idx,
            branch_id=branch_id,
            partition=partition
        )

    def get_by_chain(self, chain_path: str, partition: str = "val") -> List[Dict]:
        """Query predictions by operator chain path."""
        # chain_path format: "s1.MinMax>s3.SNV>s5.PLS"
        return self.filter_predictions(
            preprocessings=chain_path,  # Already supported!
            partition=partition
        )
```

**Critiques**:

1. **Critique**: node_id parsing is brittle
   - **Mitigation**: Use structured node IDs or store node_id in prediction record.

2. **Critique**: Chain path query may be slow (string matching)
   - **Mitigation**: Add index on preprocessings column. Already Polars, indexing is fast.

3. **Critique**: Adding fields to prediction records increases storage
   - **Mitigation**: node_id and chain_path are optional, computed on query.

**Final Approach**: Add optional query methods. Predictions core unchanged.

---

### 6. Generator

**Current Responsibility**: Expand `_or_`, `_range_` into pipeline variants

**Migration Solution**:

<<<<<<< Updated upstream
```python
# Current: expand_spec() called in PipelineConfigs.__init__

# NEW: Split behavior by location
class PipelineConfigs:
    def __init__(self, definition, ...):
        # Only expand TOP-LEVEL generators (outside branch)
        if self._has_gen_keys(self.steps, skip_branch=True):
            self.steps = expand_spec(self.steps)

        # Generators INSIDE branch are kept for runtime expansion
        # BranchController already handles this

# DAG mode: generators create dynamic nodes
class GeneratorNode(DAGNode):
    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        expanded = expand_spec(self.generator_spec)
        child_nodes = []

        for variant in expanded:
            child = runtime.dag.create_dynamic_node(variant, self.node_id)
            child_nodes.append(child)

        return [node.execute(payload_in, runtime) for node in child_nodes]
```

**Critiques**:

1. **Critique**: Top-level vs branch-level distinction is confusing
   - **Mitigation**: Document clearly. Top-level = separate pipelines, branch-level = DAG branches.

2. **Critique**: Dynamic node creation complicates DAG analysis
   - **Mitigation**: Mark dynamic nodes. Static analysis uses bounds (count parameter).

3. **Critique**: Memory for large generator spaces
   - **Mitigation**: Use `expand_spec_iter()` for lazy expansion when count > threshold.

**Final Approach**: Keep current split behavior. DAG mode uses GeneratorNode for in-branch expansion.
=======
**Insight**: Generators don't need runtime expansion. They can be fully expanded before DAG construction.

```python
# Current behavior (already correct):
# - Top-level generators → separate pipeline runs
# - In-branch generators → BranchController expands at parse time

# Unified rule:
# ALL generators → expand to branches before execution

class DAGBuilder:
    def build(self, steps):
        for step in steps:
            if is_generator_step(step):
                # Expand generator to explicit branches
                expanded = expand_generator_to_branches(step)
                # Continue with expanded form
```

**Generator Expansion Rules**:

| Context | Generator | Expansion | Implicit Merge |
|---------|-----------|-----------|----------------|
| `feature_augmentation` | `{"_or_": [A,B,C], "size": 2}` | All combinations applied | Concat features |
| `sample_augmentation` | `{"_or_": [A,B]}` | All operators applied | Add samples |
| `model` params | `{"_range_": [5,15,5]}` | Sequential models | Select best |
| `model` list | `[A, B, C]` | Sequential models | Select best |
| `branch` content | `{"_or_": [A,B,C]}` | N branches | Explicit merge |

**Critiques**:

1. **Critique**: Pre-expansion may create large DAGs
   - **Mitigation**: `count` parameter limits expansion. Already enforced.

2. **Critique**: Implicit merges need to be made explicit
   - **Mitigation**: Controller semantics define the merge (documented per controller).

3. **Critique**: Different controllers handle generators differently
   - **Mitigation**: Formalize: generators = operator lists, controllers define how lists are processed.

**Final Approach**: Generators expand to explicit structures before execution. No dynamic node creation for generators.
>>>>>>> Stashed changes

---

### 7. Serialization

**Current Responsibility**: Serialize configs, manifests, artifacts

**Migration Solution**:

```python
# Existing serialization unchanged

# NEW: DAG serialization for replay
@dataclass
class SerializedDAG:
    """Serializable DAG representation."""

    nodes: List[Dict[str, Any]]  # Each node's config
    edges: List[Tuple[str, str]]  # (from_id, to_id)
    execution_order: List[str]   # Topological order

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SerializedDAG":
        return cls(**data)

# Integration with existing manifest
class ManifestManager:
    def save_dag(self, dag: DAG, manifest_path: Path):
        """Save DAG structure alongside existing manifest."""
        dag_data = dag.serialize()

        # Existing manifest format extended
        manifest = self.load_manifest(manifest_path)
        manifest["dag"] = dag_data.to_dict()
        self._write_manifest(manifest_path, manifest)
```

**Critiques**:

1. **Critique**: DAG in manifest increases file size
   - **Mitigation**: DAG is compact (~1KB per node). Optional field.

2. **Critique**: ExecutionTrace already captures similar info
   - **Mitigation**: Merge: DAG = static structure, Trace = dynamic execution. Trace references DAG nodes.

3. **Critique**: Two serialization formats (manifest + trace)
   - **Mitigation**: Trace is primary for replay. DAG optional for visualization.

**Final Approach**: DAG serialization optional extension to manifest. ExecutionTrace remains primary replay mechanism.

---

### 8. Charts/Logging

**Current Responsibility**: Visualization, progress tracking

**Migration Solution**:

```python
# ChartController unchanged - receives Payload, produces chart

# NEW: DAG visualization
class DAGVisualizer:
    """Generate visual representation of DAG."""

    def to_mermaid(self, dag: DAG) -> str:
        """Generate Mermaid diagram syntax."""
        lines = ["graph TD"]
        for node in dag.nodes.values():
            for parent_id in node.parents:
                lines.append(f"    {parent_id} --> {node.node_id}")
            lines.append(f"    {node.node_id}[{node.operator_class}]")
        return "\n".join(lines)

    def to_ascii(self, dag: DAG) -> str:
        """Generate ASCII art representation."""
        # ... tree-like ASCII output

# Integration with logging
class DAGExecutor:
    def _on_node_complete(self, node: DAGNode):
        logger.info(f"✓ Node {node.node_id}: {node.operator_class}")
        if self.verbose >= 2:
            logger.debug(f"  Input shape: {node.payload_in.shape}")
            logger.debug(f"  Output shape: {node.payload_out.shape}")
```

**Critiques**:

1. **Critique**: ASCII visualization hard for large DAGs
   - **Mitigation**: Add collapsing, show only active path.

2. **Critique**: Progress tracking different from sequential
   - **Mitigation**: Show node count: "Step 5/12 (node_3a)"

3. **Critique**: Chart controllers need dataset access
   - **Mitigation**: ChartController receives Payload.materialize() as before.

**Final Approach**: Add DAGVisualizer utility. ChartController unchanged. Logging adds node context.

---

<<<<<<< Updated upstream
## Migration Phases

### Phase 0: Preparation (2 weeks)

**Goal**: Create foundation without breaking anything

**Tasks**:
1. Add `DatasetView` dataclass (frozen, no behavior yet)
2. Add `Payload` dataclass (immutable snapshot)
3. Add `DAGNode`, `DAG` dataclasses (structure only)
4. Write unit tests for new dataclasses
5. No changes to execution path

**Deliverables**:
- `nirs4all/pipeline/dag/` package
- `dag/node.py`, `dag/payload.py`, `dag/view.py`
- 100% test coverage on new code

**Validation**:
- All existing tests pass
- No behavior change

---

### Phase 1: DAGBuilder (3 weeks)

**Goal**: Convert linear pipeline to DAG structure

**Tasks**:
1. Implement `DAGBuilder.build()` - creates DAG from steps
2. Implement `DAGNodeAdapter` - wraps controllers
3. Implement `ForkNodeAdapter`, `JoinNodeAdapter` for branch/merge
4. Add `DAG.serialize()` / `DAG.deserialize()`
5. Add Mermaid visualization

**Deliverables**:
- `dag/builder.py`
- `dag/adapters.py`
- `dag/serialization.py`
- `dag/visualizer.py`

**Validation**:
- Can build DAG from any existing example pipeline
- DAG visualization matches expected structure
- Serialization round-trip works

---

### Phase 2: DAGExecutor (Sequential) (3 weeks)

**Goal**: Execute DAG in topological order (sequential, matching current behavior)

**Tasks**:
1. Implement `DAGExecutor.execute()` - topological traversal
2. Implement `Payload.to_execution_context()` conversion
3. Implement `Payload.with_updates()` for result propagation
4. Add `execution_mode="dag"` to PipelineRunner
5. Extensive comparison testing: same results as sequential mode

**Deliverables**:
- `dag/executor.py`
- Updated `pipeline/runner.py`
- Comparison test suite

**Validation**:
- All examples produce identical results in both modes
- All unit tests pass
- Performance within 10% of sequential

---

### Phase 3: Dynamic Nodes (2 weeks)

**Goal**: Support runtime node creation

**Tasks**:
1. Implement `DAGExecutor.create_dynamic_node()`
2. Implement `GeneratorNode` for in-branch generators
3. Implement `FoldNode` for per-fold expansion
4. Add `is_dynamic` flag to trace recording

**Deliverables**:
- Updated `dag/executor.py`
- `dag/dynamic_nodes.py`
- Test suite for dynamic expansion

**Validation**:
- Generator-in-branch examples work
- TensorFlow/PyTorch models build correctly
- Dynamic nodes appear in trace

---

### Phase 4: Immutable Dataset Access (3 weeks)

**Goal**: Controllers use DatasetView for reads

**Tasks**:
1. Add `SpectroDataset.register_processing()` (append-only)
2. Add `DatasetView.materialize()`
3. Add `DataAccessor` for controller use
4. Migrate controllers to use accessor (gradual)
5. Deprecate direct dataset mutation in DAG mode

**Deliverables**:
- Updated `data/dataset.py`
- `dag/accessor.py`
- Migration guide for custom controllers

**Validation**:
- Controllers work with accessor
- No correctness regression
- Memory usage acceptable
=======
## Primitive Implementation Order

Based on Document 2's axioms, here is the **dependency-ordered** implementation sequence:

### Tier 1: Foundation (No Dependencies)

These primitives have no dependencies on other primitives:

| Primitive | Implementation | Exists Today? |
|-----------|---------------|---------------|
| `materialize(view)` | `SpectroDataset.x()`, `SpectroDataset.y()` | ✅ Yes |
| `store_artifact(obj, meta)` | `ArtifactRegistry.register()` | ✅ Yes |
| `store_predictions(preds, meta)` | `Predictions.add_prediction()` | ✅ Yes |
| `filter_samples(view, mask)` | `DataSelector.with_partition()` | ✅ Yes |

### Tier 2: Data Modification (Depends on Tier 1)

| Primitive | Implementation | Exists Today? |
|-----------|---------------|---------------|
| `register_processing(ds, src, name, X)` | Add to `FeatureSource` | ⚠️ Partial |
| `add_samples(ds, X, y, origin)` | `SpectroDataset.add_samples()` | ✅ Yes |
| `assign_folds(ds, splitter)` | `SplitterController` logic | ✅ Yes |

### Tier 3: Operators (Depends on Tier 1-2)

| Primitive | Implementation | Exists Today? |
|-----------|---------------|---------------|
| `apply_transform(X, op)` | `operator.transform(X)` | ✅ Yes |
| `fit_transform(view, op)` | `operator.fit(X_train).transform(X)` | ✅ Yes |
| `fit_transform_target(view, op)` | `YProcessingController` logic | ✅ Yes |

### Tier 4: Model Training (Depends on Tier 1-3)

| Primitive | Implementation | Exists Today? |
|-----------|---------------|---------------|
| `train_fold(view, model, fold_id)` | `ModelController` per-fold logic | ✅ Yes |
| `aggregate_folds(preds, strategy)` | **NEW**: `VirtualModel` | ❌ No |

### Tier 5: Flow Control (Depends on Tier 1-4)

| Primitive | Implementation | Exists Today? |
|-----------|---------------|---------------|
| `fork(view, N, by)` | `BranchController` snapshot logic | ✅ Yes |
| `join(views, mode, agg)` | `MergeController` merge logic | ✅ Yes |
| `select_best(store, criteria)` | `PredictionRanker.rank()` | ✅ Yes |

### Implementation Summary

| Status | Count | Action |
|--------|-------|--------|
| ✅ Exists | 13 | Formalize as explicit primitives, add tests |
| ⚠️ Partial | 1 | Extend `register_processing()` for append-only |
| ❌ New | 1 | Implement `VirtualModel` for fold aggregation |

### VirtualModel Implementation

```python
@dataclass
class VirtualModel:
    """Aggregated fold models for prediction."""

    fold_artifact_ids: List[str]  # References to fold model artifacts
    aggregation: str              # "mean", "weighted_mean", "vote", "all"
    weights: Optional[Dict[int, float]] = None  # Per-fold weights

    @classmethod
    def from_fold_predictions(cls, pred_store, model_name, metric="rmse"):
        """Create from prediction store with automatic weighting."""
        fold_preds = pred_store.filter_predictions(
            model_name=model_name, partition="val"
        )

        # Compute weights from validation scores
        weights = {}
        for pred in fold_preds:
            fold_id = pred["fold_id"]
            score = pred["val_score"]
            # Inverse for "lower is better" metrics
            weights[fold_id] = 1.0 / (score + 1e-10) if metric in LOWER_BETTER else score

        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return cls(
            fold_artifact_ids=[p["model_artifact_id"] for p in fold_preds],
            aggregation="weighted_mean",
            weights=weights
        )

    def predict(self, X, artifact_loader):
        """Aggregate predictions from fold models."""
        fold_preds = []
        for artifact_id in self.fold_artifact_ids:
            model = artifact_loader.load(artifact_id)
            fold_preds.append(model.predict(X))

        if self.aggregation == "mean":
            return np.mean(fold_preds, axis=0)
        elif self.aggregation == "weighted_mean":
            weight_list = [self.weights[i] for i in range(len(fold_preds))]
            return np.average(fold_preds, axis=0, weights=weight_list)
        elif self.aggregation == "vote":
            return mode(fold_preds, axis=0)
        elif self.aggregation == "all":
            return np.stack(fold_preds, axis=-1)
```

---

## Migration Phases

### Phase 0: Primitives Formalization (2 weeks)

**Goal**: Create explicit primitive functions from existing code

**Tasks**:
1. Extract `materialize()` as standalone function wrapping `dataset.x()`
2. Extract `fit_transform()` as standalone function
3. Extract `assign_folds()` as standalone function
4. Add `register_processing()` with append-only semantics
5. Create `VirtualModel` dataclass for fold aggregation
6. Write unit tests for each primitive

**Deliverables**:
- `nirs4all/pipeline/primitives/` package
- `primitives/data.py` (materialize, register_processing, add_samples)
- `primitives/operators.py` (fit_transform, apply_transform)
- `primitives/models.py` (train_fold, VirtualModel)
- `primitives/flow.py` (fork, join, select_best)
- 100% test coverage on primitives

**Validation**:
- All existing tests pass
- Primitives are drop-in replacements for existing logic

---

### Phase 1: VirtualModel Integration (2 weeks)

**Goal**: Integrate VirtualModel into ModelController

**Tasks**:
1. Create `VirtualModel` with aggregation strategies (mean, weighted_mean, vote)
2. Update `BaseModelController` to create VirtualModel after fold training
3. Store VirtualModel metadata in artifact registry
4. Update prediction mode to use VirtualModel.predict()
5. Update manifest serialization for VirtualModel

**Deliverables**:
- `models/virtual_model.py`
- Updated `controllers/models/base_model.py`
- Updated `pipeline/storage/artifacts/`

**Validation**:
- Model training produces VirtualModel artifacts
- Prediction mode correctly aggregates fold predictions
- Weighted averaging uses validation scores

---

### Phase 2: DAG Data Structures (2 weeks)

**Goal**: Create DAG representation without changing execution

**Tasks**:
1. Add `DatasetView` dataclass (frozen, immutable)
2. Add `Payload` dataclass (carries view + context)
3. Add `DAGNode`, `DAG` dataclasses
4. Implement `DAGBuilder.build()` - converts pipeline to DAG
5. Add Mermaid/ASCII visualization

**Deliverables**:
- `nirs4all/pipeline/dag/` package
- `dag/node.py`, `dag/payload.py`, `dag/view.py`
- `dag/builder.py`, `dag/visualizer.py`

**Validation**:
- Can build DAG from any example pipeline
- DAG visualization matches expected structure
- No execution yet - structure only

---

### Phase 3: DAGExecutor (3 weeks)

**Goal**: Execute DAG using primitives

**Tasks**:
1. Implement `DAGExecutor.execute()` with topological traversal
2. Implement `Payload.to_execution_context()` conversion
3. Wire primitives into node execution
4. Handle Fork/Join nodes for branches
5. Add `execution_mode="dag"` to PipelineRunner
6. Extensive comparison testing

**Deliverables**:
- `dag/executor.py`
- Updated `pipeline/runner.py`
- Comparison test suite

**Validation**:
- All examples produce identical results in both modes
- OOF safety maintained
- Performance within 10% of sequential

---

### Phase 4: Generator Unification (2 weeks)

**Goal**: All generators expand to explicit DAG structures before execution

**Tasks**:
1. Unify generator expansion: all contexts use same expansion logic
2. Document implicit merges per controller type
3. Ensure generators in branches become explicit Fork/Join
4. Update BranchController to use DAGBuilder for generator expansion
5. Test complex nested generator scenarios

**Deliverables**:
- Updated `controllers/data/branch.py`
- `dag/generator_expansion.py`
- Documentation of expansion rules
- Test suite for generator → DAG conversion

**Validation**:
- All generator examples work identically
- No dynamic node creation needed
- Expansion is deterministic and predictable

**Implicit Merge Rules (Documented)**:

| Controller | Generator In | Implicit Merge |
|------------|--------------|----------------|
| `feature_augmentation` | `_or_` with `size` | Concat features (all applied) |
| `sample_augmentation` | `_or_` | All applied (no merge - samples added) |
| `concat_transform` | `_or_` | Concat features (parallel transforms) |
| `model` | `_range_` params | Select best by validation |
| `model` | List of models | Select best by validation |
| `branch` | `_or_` | Explicit merge required |
>>>>>>> Stashed changes

---

### Phase 5: Parallel Execution (Optional) (2 weeks)

**Goal**: Execute independent nodes in parallel

**Tasks**:
1. Add `DAGExecutor.execute_parallel()` using ThreadPoolExecutor
2. Identify parallelizable node groups (independent branches, folds)
3. Add `parallel=True` option to PipelineRunner
4. Benchmark parallel vs sequential

**Deliverables**:
- `dag/parallel_executor.py`
- Benchmark results

**Validation**:
- Parallel execution produces identical results
- Speedup for multi-branch/fold pipelines
- Thread-safe dataset access

---

### Phase 6: Deprecation & Cleanup (2 weeks)

**Goal**: Clean up, document, default to DAG mode

**Tasks**:
1. Make DAG mode default (`execution_mode="dag"`)
2. Add deprecation warnings for sequential-only features
3. Update all documentation
4. Remove legacy code (after deprecation period)
5. Performance optimization pass

**Deliverables**:
- Updated documentation
- Deprecation warnings
- Version 1.0 release notes

**Validation**:
- All examples work with DAG mode
- No regressions
- Documentation complete

---

## Compatibility Strategy

### Pipeline Syntax Compatibility

**Guarantee**: All existing pipeline YAML/Python definitions work unchanged

```python
# This works today and after migration:
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
runner = PipelineRunner()
predictions, results = runner.run(PipelineConfigs(pipeline), DatasetConfigs(data))
```

### API Compatibility

| API | Status | Notes |
|-----|--------|-------|
| `PipelineRunner.run()` | ✅ Unchanged | execution_mode param added |
| `PipelineConfigs` | ✅ Unchanged | Still expands generators |
| `DatasetConfigs` | ✅ Unchanged | No changes |
| `SpectroDataset` | ✅ Unchanged | New methods added, none removed |
| `Predictions` | ✅ Unchanged | New query methods added |
| All controllers | ✅ Unchanged | Wrapped by adapters |
| All operators | ✅ Unchanged | No changes |

### Custom Controller Compatibility

```python
# Existing custom controllers work without changes:
@register_controller
class MyCustomController(OperatorController):
    def execute(self, step_info, dataset, context, ...):
        # This code works in both modes
        X = dataset.x(context.selector)
        X_new = self.transform(X)
        dataset.update_features(..., X_new, ...)
        return context, StepOutput(artifacts=[...])
```

In DAG mode, `DAGNodeAdapter` wraps this controller transparently.

### Deprecation Timeline

| Phase | Deprecated | Removed |
|-------|------------|---------|
| Phase 2 | - | - |
| Phase 4 | Direct dataset mutation in controllers | - |
| Phase 6 | `execution_mode="sequential"` | - |
| v2.0 | - | `execution_mode="sequential"` |

---

## Success Criteria

### Phase Completion Criteria

| Phase | Criterion |
|-------|-----------|
| 0 | New dataclasses exist, all tests pass |
| 1 | DAG builds from all examples, visualization works |
| 2 | DAG execution matches sequential for all examples |
| 3 | Dynamic nodes work (generators, TF build) |
| 4 | Controllers use accessor, no behavior change |
| 5 | Parallel execution faster for multi-branch |
| 6 | DAG mode is default, docs complete |

### Test Coverage Requirements

```
Phase 0: 100% coverage on new dataclasses
Phase 1: 95% coverage on builder/adapters
Phase 2: 90% coverage on executor, comparison tests pass
Phase 3: 90% coverage on dynamic nodes
Phase 4: 80% coverage on accessor migration
Phase 5: 80% coverage on parallel execution
Phase 6: Maintain overall 85% coverage
```

### Performance Requirements

| Metric | Target |
|--------|--------|
| DAG build time | < 100ms for 50-step pipeline |
| DAG execution overhead | < 5% vs sequential |
| Memory overhead | < 20% increase |
| Parallel speedup | > 2x for 4+ branches |

### Acceptance Criteria

1. **All 20+ examples run successfully** with DAG mode
2. **Predictions are identical** between modes (within numerical precision)
3. **Artifacts are interchangeable** (train sequential, predict DAG or vice versa)
4. **ExecutionTrace captures DAG structure** for replay
5. **Custom controllers work** without modification
6. **Documentation is complete** with migration guide

---

## Risk Assessment

### High Risk

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes to controller API | Low | High | Adapter pattern isolates controllers |
| Performance regression | Medium | Medium | Benchmark at each phase, optimize |
| Memory explosion from immutability | Medium | Medium | Append-only, not full copies |

### Medium Risk

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Complex debugging in DAG mode | Medium | Medium | DAG visualization, detailed trace |
| Thread safety issues in parallel | Medium | Medium | Extensive concurrent testing |
| Generator expansion edge cases | Low | Medium | Comprehensive generator tests |

### Low Risk

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Documentation lag | Medium | Low | Docs updated per phase |
| User confusion with two modes | Low | Low | Clear messaging, deprecation |

---

## Summary

The migration to DAG-based execution follows these principles:

1. **Adapter, not rewrite**: Controllers wrapped, not reimplemented
2. **Incremental delivery**: Each phase is independently valuable
3. **Backward compatible**: Existing code keeps working
4. **Test-driven**: Extensive validation at each phase
5. **Performance-conscious**: Benchmarked throughout

The estimated timeline of 12-16 weeks allows for careful implementation with minimal risk to existing functionality. The final system will support:

- Linear syntax with implicit DAG structure
- Dynamic node creation for runtime decisions
- Parallel execution for independent branches
- Deterministic replay from traces
- Full backward compatibility during transition

This migration unlocks future capabilities (distributed execution, advanced caching, partial replay) while preserving nirs4all's pragmatic, domain-specific design.
