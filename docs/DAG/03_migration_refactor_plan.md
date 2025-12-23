# Document 3 — Adaptation / Refactoring Plan

**Version**: 1.0.0
**Date**: December 2025
**Status**: Migration Plan

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Component-by-Component Migration](#component-by-component-migration)
3. [Migration Phases](#migration-phases)
4. [Compatibility Layer](#compatibility-layer)
5. [Deprecation Schedule](#deprecation-schedule)
6. [Success Criteria](#success-criteria)

---

## Executive Summary

### Migration Philosophy

The migration follows a **surgical approach**:
1. Add the DAG layer as a parallel execution path
2. Gradually migrate components while maintaining backward compatibility
3. Deprecate legacy paths once DAG mode is stable
4. Remove deprecated code only after sufficient testing

### Effort Estimation

| Phase | Duration | Risk | Deliverable |
|-------|----------|------|-------------|
| Phase 1: Foundations | 2 weeks | Low | Primitives, Payload, DAGGraph |
| Phase 2: Builder | 2 weeks | Medium | DAGBuilder, linear→DAG conversion |
| Phase 3: Executor | 3 weeks | Medium | DAGExecutor, scheduler |
| Phase 4: Controllers | 2 weeks | Low | Controller adaptation, mode flag |
| Phase 5: Integration | 2 weeks | Medium | PipelineRunner integration |
| Phase 6: Generator | 1 week | Low | Generator→branches |
| Phase 7: Stabilization | 2 weeks | Medium | Bug fixes, edge cases |

**Total**: ~14 weeks for full migration

---

## Component-by-Component Migration

### 1. Primitives Layer

#### Proposed Solution

Create `nirs4all/pipeline/dag/primitives.py`:

```python
# Signature examples
def get_features(payload: Payload, selector: DataSelector) -> np.ndarray: ...
def set_features(payload: Payload, source: int, proc_name: str, data: np.ndarray) -> Payload: ...
def add_prediction(payload: Payload, record: Dict) -> Payload: ...
def fork(payload: Payload, n: int) -> List[Payload]: ...
def join(payloads: List[Payload], strategy: JoinStrategy) -> Payload: ...
def reconstruct_oof(payload: Payload, models: List[str], partition: str) -> np.ndarray: ...
def register_artifact(payload: Payload, obj: Any, meta: Dict) -> Tuple[Payload, str]: ...
```

#### Critiques

1. **Critique**: Primitives duplicate existing dataset/prediction methods
   - **Mitigation**: Primitives are thin wrappers that delegate to existing methods. They provide a consistent interface for DAG execution without duplicating logic.

2. **Critique**: Immutable payload creates memory overhead from copying
   - **Mitigation**: Use shallow copies—dataset is shared (mutations visible to all branches), only context/predictions are copied. For large datasets, this is negligible.

3. **Critique**: Too many primitives may complicate controller implementation
   - **Mitigation**: Group primitives into categories (dataset, fold, prediction, flow). Controllers only import what they need. Provide helper functions for common patterns.

#### Chosen Approach

- Implement primitives as a thin facade over existing dataset/prediction methods
- Payload uses shallow copy for dataset reference, deep copy for context/predictions
- Primitives module exposes categories via submodules: `primitives.dataset.get_features()`, `primitives.flow.fork()`

---

### 2. DAGBuilder

#### Proposed Solution

Create `nirs4all/pipeline/dag/builder.py`:

```python
class DAGBuilder:
    def build(self, steps: List[Any], dataset: SpectroDataset) -> DAGGraph:
        """Convert linear pipeline to DAG."""

    def _create_fork_node(self, step: Dict, node_id: str) -> ForkNode: ...
    def _create_join_node(self, step: Dict, node_id: str) -> JoinNode: ...
    def _create_operator_node(self, step: Any, node_id: str) -> OperatorNode: ...
    def _build_branch_subgraph(self, graph: DAGGraph, parent: str, steps: List) -> str: ...
```

#### Critiques

1. **Critique**: Linear→DAG conversion may not handle all edge cases (nested branches, conditional steps)
   - **Mitigation**: Start with core cases (branch/merge, splitter, model). Add edge case handling iteratively. Maintain comprehensive test suite with complex pipeline examples.

2. **Critique**: Node ID generation may conflict with existing step_number-based artifact IDs
   - **Mitigation**: Node IDs use a compatible scheme: `s{step_index}` for linear steps, `s{step_index}.b{branch}` for branches. Artifact system uses node_id but extracts step_index for backward compat.

3. **Critique**: Graph construction requires full step list upfront, preventing streaming
   - **Mitigation**: Not a problem for nirs4all use cases—pipelines are small (10-50 steps). Streaming construction would complicate dynamic expansion.

#### Chosen Approach

- Node ID scheme: `s{step_index}[.b{branch}][.ss{substep}][.f{fold}]`
- Handle core patterns first: linear, branch, merge, source_branch, merge_sources
- Add ConditionController handling in Phase 5

---

### 3. DAGExecutor

#### Proposed Solution

Create `nirs4all/pipeline/dag/executor.py`:

```python
class DAGExecutor:
    def __init__(self, mode: str = "train", ...): ...

    def execute(self, graph: DAGGraph, initial_payload: Payload) -> Payload:
        """Execute DAG and return final payload."""

    def _execute_node(self, node: DAGNode, inputs: List[Payload]) -> Union[Payload, List[Payload]]:
        """Execute single node based on type."""

    def _expand_node(self, graph: DAGGraph, node: DAGNode, inputs: List[Payload]) -> None:
        """Handle dynamic node expansion."""

class TopologicalScheduler:
    def get_ready_nodes(self) -> List[str]:
        """Return nodes with all inputs satisfied."""

    def mark_complete(self, node_id: str) -> None: ...
    def is_complete(self) -> bool: ...
```

#### Critiques

1. **Critique**: Topological scheduling doesn't support parallelism
   - **Mitigation**: Initially, execute sequentially (simpler, deterministic). Add optional parallel execution in a later phase using thread pool for CPU-bound or process pool for GPU-bound nodes.

2. **Critique**: Dynamic expansion modifies graph during execution, breaking invariants
   - **Mitigation**: Dynamic expansion only adds nodes/edges, never removes. Scheduler handles new nodes by recalculating ready set. Mark expanded nodes as EXPANDED to prevent re-expansion.

3. **Critique**: Payload propagation through edges may have edge cases for multi-input nodes
   - **Mitigation**: Define clear edge slot semantics. JOIN nodes receive ordered list based on slot names. Add validation that all expected inputs are present before execution.

#### Chosen Approach

- Sequential execution initially for simplicity and determinism
- Scheduler recalculates ready nodes after dynamic expansion
- Edge slots use deterministic naming: `branch_0`, `branch_1`, `fold_0`, etc.

---

### 4. Controller Adaptation

#### Proposed Solution

Add DAG mode detection to controllers:

```python
class OperatorController(ABC):
    @abstractmethod
    def execute(self, step_info, dataset, context, runtime_context, ...):
        """Execute step. In DAG mode, receives Payload instead of dataset+context."""
        pass

    def execute_dag(self, node: DAGNode, input_payload: Payload, runtime_context: RuntimeContext) -> Payload:
        """DAG-native execution. Default delegates to execute()."""
        # Convert payload to legacy format
        dataset = input_payload.dataset
        context = input_payload.context

        # Call legacy execute
        updated_context, output = self.execute(step_info, dataset, context, runtime_context, ...)

        # Convert back to payload
        return input_payload.with_context(updated_context)
```

#### Critiques

1. **Critique**: Dual interface (execute vs execute_dag) complicates controller development
   - **Mitigation**: Default `execute_dag()` implementation wraps `execute()`. Controllers only override `execute_dag()` if they need DAG-specific behavior. Most controllers work unchanged.

2. **Critique**: Payload↔legacy conversion may lose information
   - **Mitigation**: Payload contains all legacy fields (dataset, context, predictions, artifacts). Conversion is lossless. Add validation in tests.

3. **Critique**: Controllers that modify dataset in-place will affect all branches
   - **Mitigation**: This is intentional—dataset is shared for efficiency. Branches that need isolation (like different preprocessing) use `add_processing()` which creates new processing chains. Document this behavior clearly.

#### Chosen Approach

- Add `execute_dag()` with default delegation to `execute()`
- Controllers opt-in to DAG-native behavior by overriding `execute_dag()`
- Priority controllers to migrate: `TransformController`, `BaseModelController`, `BranchController`, `MergeController`

---

### 5. Generator Migration

#### Proposed Solution

Move generator expansion from static (PipelineConfigs) to dynamic (DAGBuilder):

```python
# In pipeline_config.py - detect generator syntax
class PipelineConfigs:
    def __init__(self, definition, ...):
        if self._has_gen_keys(self.steps):
            if self._generator_at_top_level():
                # Keep static expansion for top-level generators
                self.steps = expand_spec(self.steps)
            else:
                # Mark for DAG-time expansion (branch generators)
                self.steps = [self.steps]  # Single pipeline with branch generators

# In DAGBuilder - expand at fork creation
def _create_fork_node(self, step: Dict, node_id: str) -> ForkNode:
    branch_def = step.get("branch")

    if is_generator_node(branch_def):
        # Expand generator to branches at DAG build time
        expanded = expand_spec(branch_def)
        return ForkNode(
            node_id=node_id,
            fork_type=ForkType.GENERATOR,
            branch_specs=[{"steps": [e]} for e in expanded],
            generator_spec=branch_def,  # Keep original for serialization
        )
    else:
        return ForkNode(
            node_id=node_id,
            fork_type=ForkType.BRANCH,
            branch_specs=self._parse_branch_specs(branch_def),
        )
```

#### Critiques

1. **Critique**: Mixing static and dynamic expansion is confusing
   - **Mitigation**: Clear rule: top-level generators expand statically (N pipelines), branch-level generators expand dynamically (N branches). Document with examples.

2. **Critique**: Large generator expansions (1000+ variants) create huge graphs
   - **Mitigation**: Apply `count` limiter at expansion time. Implement streaming expansion for very large spaces. Log warnings for >100 branches.

3. **Critique**: Generator choices tracking (`generator_choices`) is lost in DAG mode
   - **Mitigation**: Store `generator_spec` in ForkNode. Extract choices from selected branch path for serialization. Add `get_generator_choices(branch_path)` method.

#### Chosen Approach

- Top-level generators remain static (multiple pipelines)
- Branch-level generators expand at DAGBuilder time (multiple branches)
- Store original generator spec for serialization/replay

---

### 6. Predictions Store Integration

#### Proposed Solution

Minimal changes—Predictions already has provenance tracking:

```python
# No changes to Predictions class needed

# Ensure DAG execution uses existing add_prediction() with proper provenance
def add_prediction_from_dag(payload: Payload, pred_data: Dict):
    """Helper to add prediction with DAG provenance."""
    payload.predictions.add_prediction(
        dataset_name=payload.dataset.name,
        pipeline_uid=payload.context.custom.get("pipeline_uid"),
        step_idx=payload.source_node_id.split(".")[0][1:],  # Extract step index
        branch_id=extract_branch_id(payload.branch_path),
        branch_name=extract_branch_name(payload.context),
        fold_id=payload.fold_id,
        **pred_data
    )
```

#### Critiques

1. **Critique**: Node ID differs from step_idx in predictions
   - **Mitigation**: Extract step_idx from node_id (`s3.b0` → step_idx=3). Predictions continue to use step_idx for backward compat.

2. **Critique**: Predictions from parallel fold branches may have ordering issues
   - **Mitigation**: Predictions are stored with fold_id. Ordering is handled by sort on retrieval. No execution-order dependency.

3. **Critique**: Branch path tracking may not match between DAG and legacy modes
   - **Mitigation**: DAG uses same branch_path format as legacy (`[0, 1]` for nested branches). Ensure BranchController and DAGBuilder produce identical paths.

#### Chosen Approach

- No changes to Predictions class
- Helper function ensures consistent provenance extraction from DAG payload
- Validate branch_path equivalence in integration tests

---

### 7. Artifact System Integration

#### Proposed Solution

Extend `ArtifactRecord` with optional node_id:

```python
@dataclass
class ArtifactRecord:
    artifact_id: str
    step_index: int
    fold_id: Optional[int]
    branch_path: List[int]
    # New field
    node_id: Optional[str] = None  # e.g., "s3.b0.f2"

    @classmethod
    def from_dag_node(cls, node: DAGNode, fold_id: Optional[int], ...):
        return cls(
            artifact_id=generate_id(node.node_id, fold_id),
            step_index=extract_step_index(node.node_id),
            fold_id=fold_id,
            branch_path=extract_branch_path(node.node_id),
            node_id=node.node_id,
        )
```

#### Critiques

1. **Critique**: Adding node_id changes artifact schema, may break existing artifacts
   - **Mitigation**: node_id is optional with default None. Existing artifacts continue to work. node_id only populated for DAG-mode artifacts.

2. **Critique**: Artifact loading by step_index may match wrong artifacts in DAG mode
   - **Mitigation**: DAG mode uses node_id for precise matching. Legacy mode continues using step_index+branch_path. Add fallback logic for mixed scenarios.

3. **Critique**: artifact_id generation differs between modes
   - **Mitigation**: Use consistent format: `{pipeline}:{step}:{fold}:b{branch_path}`. DAG mode just has more specific step identifiers.

#### Chosen Approach

- Add optional `node_id` field to `ArtifactRecord`
- Generate `artifact_id` consistently between modes
- Use node_id for DAG-mode loading, step_index for legacy fallback

---

### 8. Charts/Visualization Integration

#### Proposed Solution

No changes needed—charts operate on final predictions and dataset:

```python
# Charts receive dataset and predictions after execution
# No dependency on DAG vs legacy execution

# Existing code continues to work:
analyzer = PredictionAnalyzer(predictions, dataset)
analyzer.plot_scatter(partition="test")
```

#### Critiques

1. **Critique**: DAG execution trace visualization may need new charts
   - **Mitigation**: Add optional DAG visualization module in Phase 5. Not blocking for core migration.

2. **Critique**: Per-branch charts may need branch_path filtering
   - **Mitigation**: Predictions already support branch filtering. Charts work as-is with `predictions.filter(branch_id=0)`.

3. **Critique**: Charts inside pipeline (e.g., chart steps) need DAG adaptation
   - **Mitigation**: ChartController follows same pattern as other controllers—add `execute_dag()` with default delegation.

#### Chosen Approach

- No changes for post-execution visualization
- ChartController gets standard `execute_dag()` default implementation
- Add DAG visualization in later phase (optional)

---

## Migration Phases

### Phase 1: Foundations (2 weeks)

**Deliverables**:
- `nirs4all/pipeline/dag/__init__.py`
- `nirs4all/pipeline/dag/primitives.py` - All primitive operations
- `nirs4all/pipeline/dag/payload.py` - Payload dataclass
- `nirs4all/pipeline/dag/graph.py` - DAGGraph, DAGNode, DAGEdge
- Unit tests for primitives and payload

**Exit Criteria**:
- All primitives have unit tests
- Payload round-trip (create → serialize → deserialize) works
- DAGGraph supports add/remove nodes/edges, topological sort

### Phase 2: Builder (2 weeks)

**Deliverables**:
- `nirs4all/pipeline/dag/builder.py` - DAGBuilder
- `nirs4all/pipeline/dag/node_types.py` - NodeType enum, typed nodes
- Tests for linear→DAG conversion

**Exit Criteria**:
- Linear pipeline (no branches) converts correctly
- Branch/merge converts correctly
- Nested branches convert correctly
- Node IDs match expected scheme

### Phase 3: Executor (3 weeks)

**Deliverables**:
- `nirs4all/pipeline/dag/executor.py` - DAGExecutor
- `nirs4all/pipeline/dag/scheduler.py` - TopologicalScheduler
- Integration tests with real controllers

**Exit Criteria**:
- Simple pipeline executes correctly via DAG
- Branch/merge pipeline executes correctly
- Predictions match legacy execution
- Artifacts saved with correct IDs

### Phase 4: Controller Adaptation (2 weeks)

**Deliverables**:
- Updated `OperatorController` base class with `execute_dag()`
- Adapted `TransformController`
- Adapted `BaseModelController`
- Adapted `BranchController`, `MergeController`

**Exit Criteria**:
- Core controllers work in DAG mode
- Legacy mode still works (no regression)
- Fold iteration works as expected

### Phase 5: Integration (2 weeks)

**Deliverables**:
- Updated `PipelineRunner` with DAG mode flag
- Updated `PipelineExecutor` to use DAG when enabled
- End-to-end tests with all Q*.py examples

**Exit Criteria**:
- All examples pass in both legacy and DAG mode
- Predictions identical between modes (within floating-point tolerance)
- Artifacts loadable for prediction in DAG mode

### Phase 6: Generator Migration (1 week)

**Deliverables**:
- Updated `PipelineConfigs` to detect branch-level generators
- Updated `DAGBuilder` to expand generators to branches
- Tests for generator→branches conversion

**Exit Criteria**:
- `{"branch": {"_or_": [A, B, C]}}` expands to 3 branches
- `{"_or_": [A, B]}` at top level still creates 2 pipelines
- Generator choices trackable for serialization

### Phase 7: Stabilization (2 weeks)

**Deliverables**:
- Bug fixes from integration testing
- Documentation updates
- Performance benchmarks
- Migration guide for custom controllers

**Exit Criteria**:
- No regressions in CI
- Performance within 10% of legacy mode
- Documentation complete

---

## Compatibility Layer

### Legacy Mode Preservation

```python
class PipelineRunner:
    def __init__(self, ..., use_dag: bool = False):
        self.use_dag = use_dag

    def run(self, pipeline, dataset, ...):
        if self.use_dag:
            return self._run_dag_mode(pipeline, dataset, ...)
        else:
            return self._run_legacy_mode(pipeline, dataset, ...)
```

### Controller Compatibility

```python
class OperatorController(ABC):
    def execute_dag(self, node, payload, runtime_context) -> Payload:
        """DAG execution. Default wraps legacy execute()."""
        # Extract legacy format
        step_info = ParsedStep.from_node(node)
        dataset = payload.dataset
        context = payload.context

        # Call legacy
        updated_context, output = self.execute(
            step_info, dataset, context, runtime_context, ...
        )

        # Wrap result
        return payload.with_context(updated_context).with_artifacts(output.artifacts)
```

### Artifact Loading Compatibility

```python
class ArtifactLoader:
    def load_by_id(self, artifact_id: str):
        # Try new format first (with node_id)
        if self._is_dag_format(artifact_id):
            return self._load_dag_artifact(artifact_id)
        else:
            return self._load_legacy_artifact(artifact_id)

    def get_step_binaries(self, step_index: int, branch_path: List[int] = None):
        # Works for both modes
        return self._find_artifacts(step_index=step_index, branch_path=branch_path)
```

---

## Deprecation Schedule

### Phase A: Soft Deprecation (Post Phase 5)

- Add deprecation warnings to legacy mode
- Document DAG mode as recommended
- Keep legacy mode fully functional

```python
if not self.use_dag:
    warnings.warn(
        "Legacy pipeline execution is deprecated. "
        "Use PipelineRunner(use_dag=True) for improved performance. "
        "Legacy mode will be removed in v2.0.",
        DeprecationWarning
    )
```

### Phase B: Hard Deprecation (v1.x → v2.0)

- Default `use_dag=True`
- Legacy mode emits louder warnings
- Announce removal in release notes

### Phase C: Removal (v2.0)

- Remove `_run_legacy_mode()`
- Remove legacy execute paths from PipelineExecutor
- Simplify controller base class (remove execute, keep execute_dag)
- Update all documentation

---

## Success Criteria

### Functional Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| All examples pass | Run `examples/run.sh` in DAG mode |
| Predictions match | Compare predictions between modes |
| Artifacts loadable | Round-trip train→predict test |
| Branch/merge works | Stacking example produces same results |
| Generator→branches | New test case with branch generators |

### Performance Criteria

| Criterion | Target |
|-----------|--------|
| Execution time | Within 10% of legacy mode |
| Memory usage | Within 20% of legacy mode |
| Startup time | No regression |

### Compatibility Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Legacy controllers work | Run unmodified Q*.py examples |
| Artifacts from legacy mode load | Cross-mode prediction test |
| Custom controllers work | Test with user-contributed controllers |

### Test Coverage

| Component | Minimum Coverage |
|-----------|------------------|
| Primitives | 95% |
| DAGBuilder | 90% |
| DAGExecutor | 90% |
| Scheduler | 85% |
| Controller adapters | 80% |

---

## Summary

This migration plan provides:

1. **Incremental approach**: Each phase is independently testable
2. **Backward compatibility**: Legacy mode preserved throughout
3. **Clear exit criteria**: Each phase has measurable deliverables
4. **Minimal disruption**: Controllers adapt via default delegation
5. **Reversibility**: Can pause migration at any phase boundary

The key to success is maintaining the **parallel execution paths** until DAG mode is proven stable, then gradually deprecating legacy mode.

### Immediate Next Steps

1. Create `nirs4all/pipeline/dag/` package structure
2. Implement `Payload` dataclass
3. Implement core primitives (`get_features`, `set_features`, `fork`, `join`)
4. Write unit tests for primitives
5. Begin DAGGraph implementation

### Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Controller incompatibility | Default delegation preserves behavior |
| Artifact ID mismatch | Consistent format, fallback logic |
| Performance regression | Benchmark at each phase |
| Scope creep | Strict phase boundaries, no feature additions |
