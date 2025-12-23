# Document 2 — DAG Pipeline Specification

**Version**: 1.0.0
**Date**: December 2025
**Status**: Design Specification

---

## Table of Contents

1. [Objectives](#objectives)
2. [Core Concepts](#core-concepts)
3. [DAG Architecture](#dag-architecture)
4. [Primitives](#primitives)
5. [Node Types](#node-types)
6. [Execution Engine](#execution-engine)
7. [Serialization & Caching](#serialization--caching)
8. [Branch/Merge Semantics](#branchmerge-semantics)
9. [Fold-as-Branches](#fold-as-branches)
10. [Rationale](#rationale)
11. [Examples](#examples)

---

## Objectives

### Primary Goals

1. **Preserve existing code**: Reuse serialization, artifacts, prediction store, dataset representation, operator wrappers with minimal modification
2. **DAG design that is nirs4all-shaped**: Simple, pragmatic, domain-specific—not a generic workflow engine
3. **Smaller set of primitives**: Axiomatize operations so controllers become compositions of well-defined primitives
4. **Runtime decisions**: Support auto-operators, best-model selection, dynamic feature dimensions
5. **Clear separation**: *What* is done (primitives/operators) vs *how* execution flows (DAG engine)
6. **Reproducibility**: Save/replay/retrain any pipeline, path, or component deterministically

### Non-Goals

- Build a "wandb-like platform" or heavy workflow engine
- Replace Polars/NumPy data stack
- Major rewrites unless game-changing

### Linear Syntax Preservation

The user-facing syntax remains linear (list of steps). The DAG is the **internal execution model**, not the user API.

```python
# User writes this (unchanged)
pipeline = [
    {"branch": [
        [SNV(), PLS(10)],
        [MSC(), RF()]
    ]},
    MinMaxScaler(),  # Implicitly cloned to both branches
    {"merge": "predictions"},
    Ridge()
]
```

The DAG engine interprets this as:

```
         ┌── SNV ─── PLS ─── MinMax ──┐
Input ───┤                            ├─── Merge ─── Ridge ─── Output
         └── MSC ─── RF ─── MinMax ───┘
```

---

## Core Concepts

### Payload

The DAG operates on a **Payload** that flows through edges:

```python
@dataclass
class Payload:
    """Immutable snapshot of pipeline state at a point in the DAG."""
    dataset: SpectroDataset       # Feature/target data (reference, not copy)
    predictions: Predictions      # Accumulated predictions
    context: ExecutionContext     # Selector, state, metadata
    artifacts: List[ArtifactRef]  # References to saved artifacts

    # Provenance tracking
    source_node_id: str           # Which node produced this payload
    edge_id: str                  # Which edge carried this payload
    fold_id: Optional[int]        # Active fold (if in fold branch)
    branch_path: List[int]        # Active branch path [0, 1] = branch 1 inside branch 0
```

**Immutability principle**: Payloads are shallow-immutable. Dataset is shared (mutations apply to all edges), but context/predictions are copied on fork.

### Nodes

Nodes are the execution units:

```python
@dataclass
class DAGNode:
    """A node in the execution DAG."""
    node_id: str                  # Unique identifier (e.g., "s1", "s2.b0.ss1")
    step_index: int               # For artifact compatibility (1-based)
    node_type: NodeType           # OPERATOR, FORK, JOIN, VIRTUAL
    controller: Optional[OperatorController]  # Executes the node
    operator: Any                 # The operator instance (if applicable)
    config: Dict[str, Any]        # Node configuration

    # Topology
    input_edges: List[str]        # Edge IDs feeding into this node
    output_edges: List[str]       # Edge IDs produced by this node

    # Runtime state
    status: NodeStatus            # PENDING, READY, RUNNING, COMPLETED, FAILED
    result: Optional[Payload]     # Output after execution
```

### Edges

Edges carry payloads between nodes:

```python
@dataclass
class DAGEdge:
    """An edge connecting two nodes."""
    edge_id: str
    source_node: str
    target_node: str
    slot: str                     # "default", "branch_0", "fold_2", etc.
    payload: Optional[Payload]    # Set after source node completes
```

---

## DAG Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DAG EXECUTION ENGINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ DAG Builder │────▶│  DAG Graph  │────▶│ DAG Executor│               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│        ▲                    │                   │                       │
│        │                    │                   ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │   Pipeline  │     │    Nodes    │     │  Scheduler  │               │
│  │   Config    │     │   + Edges   │     │             │               │
│  └─────────────┘     └─────────────┘     └──────┬──────┘               │
│                                                  │                       │
│                                          ┌───────┴───────┐               │
│                                          ▼               ▼               │
│                                    ┌──────────┐   ┌──────────┐          │
│                                    │ Workers  │   │ Workers  │          │
│                                    │ (fold 0) │   │ (fold 1) │          │
│                                    └──────────┘   └──────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │                                        ▲
         ▼                                        │
┌─────────────────┐                      ┌─────────────────┐
│   Controllers   │                      │    Primitives   │
│  (unchanged)    │◀─────────────────────│    (new layer)  │
└─────────────────┘                      └─────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `DAGBuilder` | Converts linear pipeline → DAG | `nirs4all/pipeline/dag/builder.py` |
| `DAGGraph` | Holds nodes + edges, topological queries | `nirs4all/pipeline/dag/graph.py` |
| `DAGExecutor` | Runs the DAG, handles scheduling | `nirs4all/pipeline/dag/executor.py` |
| `Scheduler` | Determines execution order, parallelism | `nirs4all/pipeline/dag/scheduler.py` |
| `Primitives` | Atomic operations on payload | `nirs4all/pipeline/dag/primitives.py` |
| `Controllers` | Execute nodes via primitives | `nirs4all/controllers/` (existing) |

---

## Primitives

### Design Philosophy

Primitives are the **axiomatic operations** on the payload. Controllers compose primitives to implement higher-level behavior.

**Key insight from feedback**: Once folds exist, all data wrangling is by default OOF-safe. The primitives must respect this.

### Primitive Categories

#### Dataset Primitives

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `get_features` | `(payload, selector) → ndarray` | Extract features with view/filter |
| `get_targets` | `(payload, selector) → ndarray` | Extract targets with view/filter |
| `set_features` | `(payload, source, proc_name, data) → payload` | Add/update processing |
| `set_targets` | `(payload, proc_name, data) → payload` | Add/update target processing |
| `filter_samples` | `(payload, predicate) → payload` | Filter by X/Y/metadata |
| `get_metadata` | `(payload, column) → ndarray` | Extract metadata column |
| `set_metadata` | `(payload, column, data) → payload` | Add/update metadata |

#### Fold Primitives

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `assign_folds` | `(payload, splitter) → payload` | Create fold assignments |
| `get_fold_indices` | `(payload, fold_id, partition) → indices` | Get train/val indices for fold |
| `iter_folds` | `(payload) → Iterator[(fold_id, train_idx, val_idx)]` | Iterate over folds |

#### Prediction Primitives

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `add_prediction` | `(payload, pred_record) → payload` | Add prediction with provenance |
| `get_predictions` | `(payload, filter) → List[Prediction]` | Query predictions |
| `reconstruct_oof` | `(payload, model_names, partition) → ndarray` | OOF reconstruction |
| `average_predictions` | `(payload, model_name, strategy) → ndarray` | Average across folds |

#### Artifact Primitives

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `register_artifact` | `(payload, obj, meta) → (payload, artifact_id)` | Save artifact |
| `load_artifact` | `(artifact_id) → obj` | Load artifact by ID |
| `get_fold_artifacts` | `(step_idx, branch_path) → List[(fold_id, obj)]` | Get fold-specific artifacts |

#### Flow Primitives

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `fork` | `(payload, n_branches) → List[payload]` | Clone payload N times |
| `join` | `(List[payload], strategy) → payload` | Combine payloads |
| `fork_by_source` | `(payload) → Dict[source_name, payload]` | Fork per data source |
| `fork_by_fold` | `(payload) → List[(fold_id, payload)]` | Fork per fold |

### Controller as Primitive Composition

Example: `TransformController` as primitives:

```python
class TransformController(OperatorController):
    def execute(self, step_info, dataset, context, runtime_context, ...):
        # Compose primitives
        X = primitives.get_features(payload, context.selector)

        if mode == "train":
            operator.fit(X)
            artifact_id = primitives.register_artifact(payload, operator, {...})
        else:
            operator = primitives.load_artifact(artifact_id)

        X_transformed = operator.transform(X)

        payload = primitives.set_features(payload, source, proc_name, X_transformed)
        return payload
```

Example: `BaseModelController` with fold-as-branches:

```python
class BaseModelController(OperatorController):
    def execute(self, step_info, dataset, context, runtime_context, ...):
        # Fork by folds
        fold_payloads = primitives.fork_by_fold(payload)

        results = []
        for fold_id, fold_payload in fold_payloads:
            # Train partition
            X_train = primitives.get_features(fold_payload, selector.with_partition("train"))
            y_train = primitives.get_targets(fold_payload, selector.with_partition("train"))

            model = clone(operator)
            model.fit(X_train, y_train)

            # Val predictions (OOF)
            X_val = primitives.get_features(fold_payload, selector.with_fold(fold_id))
            y_pred_val = model.predict(X_val)

            fold_payload = primitives.add_prediction(fold_payload, {
                "partition": "val",
                "fold_id": fold_id,
                "y_pred": y_pred_val,
                ...
            })

            # Register fold artifact
            primitives.register_artifact(fold_payload, model, {"fold_id": fold_id})

            results.append(fold_payload)

        # Join fold results (average predictions, merge artifacts)
        payload = primitives.join(results, strategy="prediction_average")
        return payload
```

---

## Node Types

### NodeType Enum

```python
class NodeType(Enum):
    OPERATOR = "operator"     # Standard controller execution
    FORK = "fork"             # Branch/source_branch/fold_fork
    JOIN = "join"             # Merge/fold_merge
    VIRTUAL = "virtual"       # No-op, for graph structure (e.g., entry/exit)
    SUBGRAPH = "subgraph"     # Nested DAG (for complex branches)
```

### Node Specifications

#### OPERATOR Node

Standard nodes that execute a controller:

```python
@dataclass
class OperatorNode(DAGNode):
    controller: OperatorController
    operator: Any
    requires_fit: bool        # True for transformers/models in train mode
    supports_predict: bool    # From controller.supports_prediction_mode()
```

#### FORK Node

Creates multiple output edges:

```python
@dataclass
class ForkNode(DAGNode):
    fork_type: ForkType       # BRANCH, SOURCE, FOLD, GENERATOR
    branch_specs: List[Dict]  # Branch definitions

    # For FOLD fork
    n_folds: Optional[int]

    # For GENERATOR fork (moved from static expansion)
    generator_spec: Optional[Dict]  # {"_or_": [...], "count": 5}
```

#### JOIN Node

Combines multiple input edges:

```python
@dataclass
class JoinNode(DAGNode):
    join_type: JoinType       # FEATURES, PREDICTIONS, MIXED, FOLD_AVERAGE
    merge_config: MergeConfig # From existing MergeController

    # For prediction merge
    oof_safe: bool = True     # Enforce OOF reconstruction
    aggregation: AggregationStrategy = AggregationStrategy.SEPARATE

    # For fold merge
    averaging: str = "weighted"  # "mean", "weighted", "stacking"
```

---

## Execution Engine

### DAGBuilder: Linear → DAG Conversion

The builder converts linear pipeline syntax to DAG:

```python
class DAGBuilder:
    def build(self, steps: List[Any], dataset: SpectroDataset) -> DAGGraph:
        graph = DAGGraph()
        current_node = graph.add_node(NodeType.VIRTUAL, "entry")
        active_branches = [current_node]  # Stack of active branch contexts

        for step_idx, step in enumerate(steps, 1):
            node_id = f"s{step_idx}"

            if self._is_branch_step(step):
                # Create FORK node
                fork_node = self._create_fork_node(step, node_id)
                graph.add_edge(current_node, fork_node)

                # Process each branch
                branch_exit_nodes = []
                for branch_idx, branch_def in enumerate(step["branch"]):
                    branch_entry = f"{node_id}.b{branch_idx}"
                    self._build_branch_subgraph(graph, fork_node, branch_def, branch_entry)
                    branch_exit_nodes.append(graph.get_exit_node(branch_entry))

                active_branches = branch_exit_nodes
                current_node = None  # No single current node

            elif self._is_merge_step(step):
                # Create JOIN node
                join_node = self._create_join_node(step, node_id)
                for branch_exit in active_branches:
                    graph.add_edge(branch_exit, join_node)

                current_node = join_node
                active_branches = [join_node]

            else:
                # Standard operator node
                if active_branches and len(active_branches) > 1:
                    # Clone node for each branch
                    for branch_idx, branch_node in enumerate(active_branches):
                        cloned_id = f"{node_id}.b{branch_idx}"
                        op_node = self._create_operator_node(step, cloned_id)
                        graph.add_edge(branch_node, op_node)
                        active_branches[branch_idx] = op_node
                else:
                    op_node = self._create_operator_node(step, node_id)
                    graph.add_edge(current_node or active_branches[0], op_node)
                    current_node = op_node
                    active_branches = [op_node]

        # Connect to exit
        exit_node = graph.add_node(NodeType.VIRTUAL, "exit")
        for node in active_branches:
            graph.add_edge(node, exit_node)

        return graph
```

### DAGExecutor: Execution with Dynamic Expansion

```python
class DAGExecutor:
    def execute(self, graph: DAGGraph, initial_payload: Payload) -> Payload:
        scheduler = TopologicalScheduler(graph)

        # Initialize entry node
        graph.set_payload("entry", initial_payload)

        while not scheduler.is_complete():
            # Get ready nodes (all inputs available)
            ready_nodes = scheduler.get_ready_nodes()

            for node_id in ready_nodes:
                node = graph.get_node(node_id)
                input_payloads = graph.get_input_payloads(node_id)

                # Handle dynamic expansion
                if node.requires_dynamic_build():
                    self._expand_node(graph, node, input_payloads)
                    continue

                # Execute node
                output = self._execute_node(node, input_payloads)

                # Distribute output to edges
                graph.set_output(node_id, output)
                scheduler.mark_complete(node_id)

        return graph.get_payload("exit")

    def _execute_node(self, node: DAGNode, inputs: List[Payload]) -> Union[Payload, List[Payload]]:
        if node.node_type == NodeType.FORK:
            return self._execute_fork(node, inputs[0])
        elif node.node_type == NodeType.JOIN:
            return self._execute_join(node, inputs)
        elif node.node_type == NodeType.OPERATOR:
            return self._execute_operator(node, inputs[0])
        elif node.node_type == NodeType.VIRTUAL:
            return inputs[0] if inputs else Payload()

    def _execute_fork(self, node: ForkNode, payload: Payload) -> List[Payload]:
        if node.fork_type == ForkType.BRANCH:
            return primitives.fork(payload, len(node.branch_specs))
        elif node.fork_type == ForkType.FOLD:
            return [p for _, p in primitives.fork_by_fold(payload)]
        elif node.fork_type == ForkType.GENERATOR:
            # Expand generator at runtime
            expanded = expand_spec(node.generator_spec)
            return primitives.fork(payload, len(expanded))
```

### Dynamic Node Expansion

For runtime-only construction (TF models, auto-operators):

```python
def _expand_node(self, graph: DAGGraph, node: DAGNode, inputs: List[Payload]) -> None:
    """Expand a placeholder node based on runtime information."""

    if node.config.get("dynamic_build"):
        # Get input shape from payload
        payload = inputs[0]
        X = primitives.get_features(payload, payload.context.selector)
        input_shape = X.shape[1:]

        # Build model with actual shape
        model = node.operator.build(input_shape)
        node.operator = model
        node.config["dynamic_build"] = False

    elif node.config.get("auto_select"):
        # Auto-operator: select based on data
        payload = inputs[0]
        selected = node.operator.select(payload.dataset)

        # Replace node with selected operator
        node.operator = selected
        node.controller = ControllerRouter().route(ParsedStep(selected))
        node.config["auto_select"] = False
```

---

## Serialization & Caching

### Node/Edge Serialization

Reuse existing `serialize_component` for operators:

```python
def serialize_graph(graph: DAGGraph) -> Dict:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "step_index": node.step_index,
                "node_type": node.node_type.value,
                "operator": serialize_component(node.operator) if node.operator else None,
                "config": node.config,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source_node,
                "target": edge.target_node,
                "slot": edge.slot,
            }
            for edge in graph.edges.values()
        ],
    }
```

### Artifact Identification

Extend existing artifact ID scheme to include node topology:

```python
# Current: "{pipeline_uid}:{step_index}:{fold_id}:b{branch_path}"
# DAG:     "{pipeline_uid}:{node_id}:{fold_id}"

# Examples:
# "abc123:s1:all"           # Transformer at step 1, no fold
# "abc123:s3.b0:0"          # Model at step 3, branch 0, fold 0
# "abc123:s3.b0.ss1:0"      # Substep 1 in branch 0, fold 0
```

### Caching / Hash Stability

Cache key for a node depends on:
1. Node configuration hash
2. Input payload hash (dataset fingerprint + context state)

```python
def node_cache_key(node: DAGNode, input_payload: Payload) -> str:
    config_hash = hash_dict(serialize_component(node.operator))
    payload_hash = input_payload.fingerprint()
    return f"{node.node_id}:{config_hash}:{payload_hash}"
```

For deterministic replay:
- `ExecutionTrace` stores per-step hashes
- Replay validates hashes match before using cached artifacts

---

## Branch/Merge Semantics

### Fork Semantics

| Fork Type | Trigger | Output |
|-----------|---------|--------|
| `BRANCH` | `{"branch": [...]}` | N payloads, one per branch |
| `SOURCE` | `{"source_branch": {...}}` | N payloads, one per source |
| `FOLD` | Model controller internal | N payloads, one per fold |
| `GENERATOR` | `{"branch": {"_or_": [...]}}` | N payloads per generator expansion |

**Fork operation**:
```python
def fork(payload: Payload, n: int) -> List[Payload]:
    """Clone payload N times with distinct branch paths."""
    results = []
    for i in range(n):
        cloned = Payload(
            dataset=payload.dataset,  # Shared reference
            predictions=Predictions(),  # Fresh per branch
            context=payload.context.copy().with_branch(branch_id=i),
            artifacts=[],
            branch_path=payload.branch_path + [i],
        )
        results.append(cloned)
    return results
```

### Join Semantics

| Join Type | Trigger | Strategy |
|-----------|---------|----------|
| `FEATURES` | `{"merge": "features"}` | Horizontal concat |
| `PREDICTIONS` | `{"merge": "predictions"}` | OOF reconstruct + concat |
| `MIXED` | `{"merge": {"features": [0], "predictions": [1]}}` | Per-branch strategy |
| `FOLD_AVERAGE` | End of model controller | Weighted average |

**Join operation** (features):
```python
def join_features(payloads: List[Payload], config: MergeConfig) -> Payload:
    """Combine features from multiple branches."""
    feature_arrays = []
    for i, p in enumerate(payloads):
        X = primitives.get_features(p, p.context.selector)
        feature_arrays.append(X)

    merged = np.concatenate(feature_arrays, axis=1)

    # Use first payload as base, update with merged features
    result = payloads[0].copy()
    result.dataset.add_merged_features(merged, "merged", source=0)
    result.branch_path = []  # Exit branch mode

    return result
```

**Join operation** (predictions with OOF):
```python
def join_predictions_oof(payloads: List[Payload], config: MergeConfig) -> Payload:
    """Combine predictions from branches using OOF reconstruction."""
    all_predictions = []

    for p in payloads:
        for model_name in config.get_models_for_branch(p.branch_path[-1]):
            # Reconstruct OOF for train partition
            oof_preds = primitives.reconstruct_oof(p, [model_name], "train")
            all_predictions.append(oof_preds)

    merged = np.concatenate(all_predictions, axis=1)

    result = payloads[0].copy()
    result.dataset.add_merged_features(merged, "merged_predictions", source=0)
    result.branch_path = []

    return result
```

---

## Fold-as-Branches

### Conceptual Model

**Key insight from feedback**: Treating folds as branches unifies the execution model.

```
Standard CV (current):
    Model Controller internally loops over folds

DAG with fold-as-branches:
    ForkNode(FOLD) → N parallel paths → JoinNode(FOLD_AVERAGE)
```

### Fold Fork/Join

```python
# Fork by fold
fold_payloads = primitives.fork_by_fold(payload)
# Returns: [(0, payload_fold0), (1, payload_fold1), ...]

# Each fold payload has:
#   - fold_id set in context
#   - train partition = fold's train indices
#   - val partition = fold's val indices (OOF)

# Join folds (averaging/weighted averaging)
def join_folds(payloads: List[Payload], strategy: str) -> Payload:
    """Combine fold predictions into final result."""

    if strategy == "average":
        # Simple average of predictions
        test_preds = [p.predictions.filter(partition="test") for p in payloads]
        averaged = np.mean([p.y_pred for p in test_preds], axis=0)

    elif strategy == "weighted":
        # Weight by validation score
        weights = []
        for p in payloads:
            val_pred = p.predictions.filter(partition="val")
            weights.append(1.0 / val_pred.score)  # Lower error = higher weight
        weights = np.array(weights) / sum(weights)

        test_preds = [p.predictions.filter(partition="test").y_pred for p in payloads]
        averaged = np.average(test_preds, weights=weights, axis=0)

    # Create "avg" prediction entry
    result = payloads[0].copy()
    result.predictions.add_prediction(
        fold_id="avg",
        y_pred=averaged,
        ...
    )

    return result
```

### Backward Compatibility

For backward compatibility with existing controller-internal fold loops:

```python
class BaseModelController(OperatorController):
    def execute(self, ...):
        if runtime_context.dag_mode:
            # DAG handles fold iteration
            return self._execute_single_fold(...)
        else:
            # Legacy: internal fold loop
            return self._execute_all_folds(...)
```

---

## Rationale

### Why This Design Fits nirs4all

1. **Payload model mirrors existing architecture**: Dataset, Predictions, Artifacts, Context are already the core components

2. **Controllers become node executors**: Minimal change—controllers already implement `execute()` with similar signature

3. **Linear syntax preserved**: Users don't need to learn DAG syntax; the builder handles conversion

4. **Generator → branches**: Moving generator expansion to runtime branches (instead of static N pipelines) reduces memory and enables smarter selection

5. **Fold-as-branches**: Unifies the execution model and enables parallel fold training (future optimization)

### Why It Preserves Existing Code

| Component | Change Required |
|-----------|-----------------|
| Operator serialization | None |
| Predictions store | None |
| Artifact system | Add node_id field |
| Dataset/views | None |
| Controllers | Minor: check `dag_mode` flag |
| Splitters | None |
| Charts | None |

### Why It Avoids Overengineering

1. **No external workflow engine**: Built specifically for nirs4all semantics
2. **No distributed execution** (initially): Single-process, sequential with optional parallelism
3. **No complex scheduling**: Simple topological order, priority only for tie-breaking
4. **No DAG compilation**: Interpretation at runtime for dynamic expansion

### How It Supports Runtime-Only Build

1. **Placeholder nodes**: Marked with `dynamic_build=True`, expanded when inputs available
2. **Shape propagation**: Input shapes flow through edges, available at node execution
3. **Lazy operator construction**: `build(input_shape)` called just before `fit()`

---

## Examples

### Example 1: Basic Preprocessing + Model

```yaml
# Pipeline definition
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.KFold
    params:
      n_splits: 5
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

```
DAG:
    [Entry] → [MinMaxScaler] → [KFold] → [Fork:FOLD]
                                              ├─→ [PLS fold0] ─→ [Join:FOLD_AVG] → [Exit]
                                              ├─→ [PLS fold1] ─┘
                                              ├─→ [PLS fold2] ─┘
                                              ├─→ [PLS fold3] ─┘
                                              └─→ [PLS fold4] ─┘
```

### Example 2: Stacking with Branches

```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.KFold
    params:
      n_splits: 3
  - branch:
      - - class: nirs4all.operators.transforms.nirs.SNV
        - model:
            class: sklearn.cross_decomposition.PLSRegression
            params:
              n_components: 10
      - - class: nirs4all.operators.transforms.nirs.MSC
        - model:
            class: sklearn.ensemble.RandomForestRegressor
  - merge: predictions
  - model:
      class: sklearn.linear_model.Ridge
```

```
DAG:
    [Entry] → [MinMaxScaler] → [KFold] → [Fork:BRANCH]
                                              ├─→ [SNV] → [Fork:FOLD]
                                              │              ├─→ [PLS fold0] ─→ [Join:FOLD_AVG]─┐
                                              │              └─→ [PLS fold1] ─┘                  │
                                              │              └─→ [PLS fold2] ─┘                  │
                                              │                                                   │
                                              └─→ [MSC] → [Fork:FOLD]                            │
                                                             ├─→ [RF fold0] ─→ [Join:FOLD_AVG]───┤
                                                             └─→ [RF fold1] ─┘                   │
                                                             └─→ [RF fold2] ─┘                   │
                                                                                                  │
                                              [Join:PREDICTIONS] ←────────────────────────────────┘
                                                    │
                                                    ▼
                                              [Fork:FOLD]
                                                    ├─→ [Ridge fold0] ─→ [Join:FOLD_AVG] → [Exit]
                                                    └─→ [Ridge fold1] ─┘
                                                    └─→ [Ridge fold2] ─┘
```

### Example 3: Multi-Source with Source Branches

```yaml
pipeline:
  - source_branch:
      NIR:
        - class: nirs4all.operators.transforms.nirs.SNV
        - class: nirs4all.operators.transforms.nirs.FirstDerivative
      markers:
        - class: sklearn.feature_selection.VarianceThreshold
        - class: sklearn.preprocessing.MinMaxScaler
  - merge_sources: concat
  - class: sklearn.model_selection.KFold
    params:
      n_splits: 5
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 20
```

```
DAG:
    [Entry] → [Fork:SOURCE]
                  ├─→ [SNV] → [D1] ─────────────────┐
                  │                                  │
                  └─→ [VarThresh] → [MinMax] ───────┤
                                                     │
              [Join:SOURCES (concat)] ←─────────────┘
                          │
                          ▼
                      [KFold] → [Fork:FOLD]
                                    ├─→ [PLS fold0] ─→ [Join:FOLD_AVG] → [Exit]
                                    └─→ [PLS fold1] ─┘
                                    └─→ ...
```

---

## Summary

This DAG specification provides:

1. **Clear conceptual model**: Payload flows through nodes connected by edges
2. **Minimal primitives**: 15-20 atomic operations that controllers compose
3. **Preserved user syntax**: Linear pipeline lists, automatic DAG conversion
4. **OOF safety by default**: Fold boundaries respected in all prediction operations
5. **Dynamic expansion**: Runtime node construction for shape-dependent models
6. **Generator → branches**: Unified handling of exploration as runtime branching
7. **Fold-as-branches option**: Unified execution model, optional parallel execution
8. **Backward compatibility**: Existing controllers work with minor mode check

The design is **nirs4all-shaped**: domain-specific, pragmatic, and avoids generic workflow engine complexity.
