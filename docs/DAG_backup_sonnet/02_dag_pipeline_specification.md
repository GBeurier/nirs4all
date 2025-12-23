# Document 2: DAG Pipeline Specification

**Date**: December 2025
**Author**: Architecture Design
**Status**: Target Specification

---

## Table of Contents

1. [Objectives Restated](#objectives-restated)
2. [Core Concepts](#core-concepts)
3. [Generator Unification](#generator-unification)
4. [OOF Safety Model](#oof-safety-model)
5. [Folds as Branches](#folds-as-branches)
6. [Node Types](#node-types)
7. [Execution Engine](#execution-engine)
8. [Data Access Model](#data-access-model)
9. [Serialization](#serialization)
10. [Caching & Reproducibility](#caching--reproducibility)
11. [Branch/Merge Semantics](#branchmerge-semantics)
12. [Primitive Operations](#primitive-operations)
13. [Axiomatic Execution Sequences](#axiomatic-execution-sequences)
14. [Rationale](#rationale)
15. [YAML Pipeline Examples](#yaml-pipeline-examples)

---

## Objectives Restated

### Goals

1. **Preserve existing code**: Reuse controllers, dataset, predictions, artifact storage, serialization
2. **DAG execution**: Nodes are controllers, edges carry payload (dataset view + context)
3. **Linear syntax**: Keep `[step1, step2, step3]` where each step implicitly depends on the previous
4. **Dynamic expansion**: Support runtime node creation (TF model build, generator branches)
5. **Primitives-first**: Define small set of atomic operations; controllers = compositions
6. **Runtime decisions**: Auto-operators, best-model selection, dynamic feature dims
7. **Deterministic replay**: Save/reload pipeline paths including branches
8. **Efficient migration**: Incremental changes, backward-compatible

### Non-Goals

- Heavy workflow engine (Airflow, Prefect, etc.)
- Distributed execution (single-process, optional parallelism)
- Complete API redesign (evolution, not revolution)

---

## Generator Unification

### Key Insight: Generators = Implicit Branches

All generator syntax (`_or_`, `_range_`, etc.) can be **expanded before runtime** into explicit DAG branches. This unifies the conceptual model:

```
Generator Expansion Rule:
  {"_or_": [A, B, C]} → implicit branch with 3 paths + implicit merge
```

### Expansion Rules by Context

| Context | Generator In | Expansion Strategy | Implicit Merge |
|---------|--------------|-------------------|----------------|
| Top-level step | `{"_or_": [SNV, MSC]}` | 2 parallel pipelines (current behavior) | None (separate runs) |
| Inside `branch` | `{"_or_": [SNV, MSC]}` | 2 sub-branches | Explicit merge required |
| `feature_augmentation` | `{"_or_": [SNV, D1], size: 2}` | Expand operators in list | Concat features |
| `sample_augmentation` | `{"_or_": [Noise, Shift]}` | Expand operators in list | Add samples |
| `model` params | `{n_components: {"_range_": [5, 20, 5]}}` | N sequential models | Best selection |
| `model` list | `[PLS, RF, XGB]` | 3 sequential models | Best selection |

### Pre-Runtime Expansion

**All generators are expanded to DAG structure before execution begins:**

```python
# Before expansion:
pipeline = [
    MinMaxScaler(),
    {"feature_augmentation": {"_or_": [SNV, MSC, D1], "size": 2}},
    {"model": PLSRegression(n_components={"_range_": [5, 15, 5]})}
]

# After DAG expansion (conceptual):
DAG:
  START
    → MinMaxScaler
    → FORK(feature_aug)
        → SNV+MSC, SNV+D1, MSC+D1  # C(3,2) = 3 combinations
    → JOIN(concat_features)
    → FORK(model_params)
        → PLS(n=5), PLS(n=10), PLS(n=15)  # 3 param variants
    → JOIN(select_best)
  END
```

### Controller-Specific Expansion

| Controller | Expansion Behavior |
|------------|-------------------|
| `feature_augmentation` | Operators added to list, all applied, features concatenated |
| `sample_augmentation` | Operators added to list, all applied, samples added |
| `concat_transform` | Operators added to list, applied in parallel, features concatenated |
| `model` (single) | Parameters expanded → sequential models → best selected |
| `model` (list) | Each model trained → best selected |
| `branch` | Each variant becomes a branch → explicit merge required |

### Equivalence Rules

```python
# These are semantically equivalent:

# 1. Generator at top level (creates separate pipeline runs)
{"_or_": [PLS(n=5), PLS(n=10), PLS(n=15)]}
# Equivalent to 3 separate pipeline executions

# 2. Generator in branch (creates DAG branches)
{"branch": {"_or_": [PLS(n=5), PLS(n=10), PLS(n=15)]}}
# Equivalent to:
{"branch": [[PLS(n=5)], [PLS(n=10)], [PLS(n=15)]]}
# Followed by implicit or explicit merge

# 3. Model with param generator (sequential models in one branch)
{"model": {"class": "PLS", "params": {"n_components": {"_range_": [5, 15, 5]}}}}
# Equivalent to:
[{"model": PLS(n=5)}, {"model": PLS(n=10)}, {"model": PLS(n=15)}]
# With implicit "select best" aggregation
```

---

## OOF Safety Model

### Core Principle: Fold-Aware Data Access

**Once folds are assigned, all downstream data operations are automatically OOF-safe.**

```
Fold Assignment = Partition Boundary
  - Training data: only X_train, y_train visible to fit()
  - Validation predictions: constructed from held-out fold predictions
  - Test predictions: average of fold models (no leakage possible)
```

### OOF Reconstruction

When predictions are used as features (stacking, merge), OOF reconstruction is **mandatory**:

```
For each sample i in training set:
  Find fold f where sample i was in validation
  Use prediction from model trained on folds != f

Result: X_meta[i] = y_pred from model that never saw sample i
```

### Safety Guarantees by Operation

| Operation | Pre-Fold | Post-Fold |
|-----------|----------|-----------|
| `transform_feature` | Applied to all samples | Applied to all samples (stateless) |
| `fit` transform | Uses all samples | Uses only X_train per fold |
| `train_model` | N/A | Uses only X_train per fold |
| `predict` | N/A | Produces val predictions per fold |
| `merge: predictions` | N/A | Uses OOF reconstruction |
| `feature_augmentation` | Creates new features | Fit on train, transform all |
| `sample_augmentation` | Adds samples globally | Adds to train partition only |

### OOF Diagram

```
Dataset: [S0, S1, S2, S3, S4, S5, S6, S7, S8, S9]
                     ↓
              Fold Assignment (3-fold)
                     ↓
┌─────────────────────────────────────────────────────┐
│ Fold 0: Train=[S3-S9], Val=[S0,S1,S2]               │
│ Fold 1: Train=[S0,S1,S2,S6,S7,S8,S9], Val=[S3,S4,S5]│
│ Fold 2: Train=[S0-S5], Val=[S6,S7,S8,S9]            │
└─────────────────────────────────────────────────────┘
                     ↓
              Model Training
                     ↓
┌─────────────────────────────────────────────────────┐
│ Model_0.fit(S3-S9) → Model_0.predict(S0,S1,S2)      │
│ Model_1.fit(...) → Model_1.predict(S3,S4,S5)        │
│ Model_2.fit(S0-S5) → Model_2.predict(S6,S7,S8,S9)   │
└─────────────────────────────────────────────────────┘
                     ↓
              OOF Reconstruction
                     ↓
X_meta_train = [pred_0(S0), pred_0(S1), ..., pred_2(S9)]
             = OOF predictions (no leakage)
```

---

## Folds as Branches

### Key Insight: Fold Training = Fork/Join

Cross-validation can be modeled as a special form of branching:

```
ModelController execution:
  1. FORK by fold_id → N parallel training paths
  2. Each fold: train model, predict val/test
  3. JOIN with aggregation strategy
```

### Fold Fork/Join Pattern

```
             ┌─── Fold 0 ───┐
             │  Train M₀    │
             │  Pred V₀,T₀  │
             └──────┬───────┘
                    │
Input ──→ FORK ─────┼─── Fold 1 ───┐
          (folds)   │  Train M₁    │──→ JOIN ──→ Output
                    │  Pred V₁,T₁  │   (merge)
                    └──────┬───────┘
                           │
                    ┌─── Fold 2 ───┐
                    │  Train M₂    │
                    │  Pred V₂,T₂  │
                    └──────────────┘
```

### Fold Merge Strategies (Virtual Model)

The "virtual model" is the aggregation of fold models:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mean` | Average predictions across folds | Default for regression |
| `weighted_mean` | Weight by validation score | Better models contribute more |
| `vote` | Majority vote (classification) | Default for classification |
| `proba_mean` | Average class probabilities | Soft voting |
| `all` | Keep all fold predictions | For downstream stacking |

```python
# Virtual Model = Fold Merge Result
class VirtualModel:
    """Represents aggregated fold models."""

    fold_models: List[Model]          # Individual fold models
    fold_weights: Dict[int, float]    # Per-fold weights
    merge_strategy: str               # "mean", "weighted_mean", etc.

    def predict(self, X):
        preds = [m.predict(X) for m in self.fold_models]
        if self.merge_strategy == "mean":
            return np.mean(preds, axis=0)
        elif self.merge_strategy == "weighted_mean":
            weights = [self.fold_weights[i] for i in range(len(preds))]
            return np.average(preds, axis=0, weights=weights)
        # ...
```

### Implications for Prediction Mode

In prediction mode, the virtual model is used:

```python
# Training: Creates fold models + virtual model reference
artifacts = [
    ("fold_0", model_0),
    ("fold_1", model_1),
    ("fold_2", model_2),
    ("virtual", VirtualModelRef(merge="weighted_mean", weights={0: 0.3, 1: 0.4, 2: 0.3}))
]

# Prediction: Uses virtual model aggregation
def predict(X, artifacts):
    fold_preds = [artifacts[f"fold_{i}"].predict(X) for i in range(3)]
    weights = artifacts["virtual"].weights
    return weighted_average(fold_preds, weights)
```

---

## Core Concepts

### Payload

The unit of data flowing on DAG edges:

```python
@dataclass
class Payload:
    """Immutable snapshot of pipeline state at a DAG edge."""

    # Data views (references, not copies)
    dataset_view: DatasetView       # Selector + snapshot reference
    y_version: str                  # Current target transformation

    # Accumulated state
    processing_chains: List[List[str]]  # Per-source processing history
    branch_path: List[int]              # Current branch hierarchy

    # Shared stores (mutable, but append-only during execution)
    prediction_store: Predictions       # Shared across all paths
    artifact_registry: ArtifactRegistry # Shared across all paths

    # Lineage
    parent_node_id: Optional[str]
    step_index: int
```

### DatasetView

Immutable reference to a dataset slice:

```python
@dataclass(frozen=True)
class DatasetView:
    """Immutable specification of which data to access."""
    partition: str
    processing: Tuple[Tuple[str, ...], ...]  # Immutable per-source
    layout: str
    source_indices: Tuple[int, ...]
    sample_mask: Optional[np.ndarray]        # For fold/filter
    include_augmented: bool
    include_excluded: bool
```

### Node

```python
@dataclass
class DAGNode:
    """A node in the execution DAG."""
    node_id: str                        # "step_3_branch_0_fold_1"
    step_index: int                     # Original pipeline step index
    controller: Type[OperatorController]
    operator: Any                       # Deserialized operator instance
    config: Dict[str, Any]              # Original step config

    # Graph structure
    parents: List[str]                  # Input node IDs
    children: List[str]                 # Output node IDs (populated during execution)

    # Execution state
    status: NodeStatus                  # PENDING, RUNNING, COMPLETED, SKIPPED
    payload_in: Optional[Payload]       # Set when ready to execute
    payload_out: Optional[Payload]      # Set after execution
    artifacts: List[ArtifactRecord]     # Produced artifacts

    # Dynamic metadata
    branch_path: List[int]
    fold_id: Optional[int]
    is_dynamic: bool                    # Created at runtime
```

### DAG Structure

```
     ┌──────────┐
     │  START   │ ← Payload(dataset_view=initial, y_version="numeric")
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 1   │  MinMaxScaler
     │ (node_1) │
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 2   │  ShuffleSplit (assigns folds, no data change)
     │ (node_2) │
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 3   │  {branch: [[SNV], [MSC]]}
     │ (branch) │
     └────┬─────┘
          │
    ┌─────┼─────┐
    │           │
┌───▼───┐ ┌───▼───┐
│node_3a│ │node_3b│   SNV path    MSC path
│ SNV   │ │ MSC   │
└───┬───┘ └───┬───┘
    │         │
┌───▼───┐ ┌───▼───┐
│node_4a│ │node_4b│   PLS on each branch
│ PLS   │ │ PLS   │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
    ┌────▼─────┐
    │ Step 5   │  {merge: "predictions"}
    │ (merge)  │
    └────┬─────┘
         │
    ┌────▼─────┐
    │ Step 6   │  Ridge (meta-model)
    │ (model)  │
    └────┬─────┘
         │
    ┌────▼─────┐
    │   END    │
    └──────────┘
```

---

## Node Types

### 1. Operator Node (Standard)

Executes a single operator (transformer, model, splitter).

```python
class OperatorNode(DAGNode):
    """Standard node that applies an operator."""

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        result = self.controller().execute(
            step_info=self.to_parsed_step(),
            dataset=payload_in.dataset_view.materialize(),
            context=payload_in.to_execution_context(),
            runtime_context=runtime,
            ...
        )
        return payload_in.with_updates(result)
```

### 2. Fork Node (Branch)

Creates multiple output edges from one input.

```python
class ForkNode(DAGNode):
    """Node that splits execution into multiple branches."""

    branch_count: int
    branch_definitions: List[List[Any]]  # Substeps per branch

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        outputs = []
        for branch_id, branch_steps in enumerate(self.branch_definitions):
            branch_payload = payload_in.with_branch(branch_id)
            # Create child nodes for branch substeps
            for substep in branch_steps:
                child_node = runtime.dag.create_dynamic_node(substep, branch_payload)
                branch_payload = child_node.execute(branch_payload, runtime)
            outputs.append(branch_payload)
        return outputs
```

### 3. Join Node (Merge)

Combines multiple input edges into one output.

```python
class JoinNode(DAGNode):
    """Node that merges multiple branch outputs."""

    merge_mode: MergeMode  # FEATURES, PREDICTIONS, BOTH

    def execute(self, payloads_in: List[Payload], runtime: RuntimeContext) -> Payload:
        merged = self._merge_payloads(payloads_in, self.merge_mode)
        return merged.with_branch_exit()
```

### 4. Fold Node (CV Expansion)

Creates per-fold execution paths.

```python
class FoldNode(DAGNode):
    """Node that expands to per-fold model training."""

    n_folds: int

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        fold_payloads = []
        for fold_id in range(self.n_folds):
            fold_payload = payload_in.with_fold(fold_id)
            fold_payloads.append(fold_payload)
        return fold_payloads
```

### 5. Generator Node (Dynamic Expansion)

Expands at runtime based on generator syntax inside branches.

```python
class GeneratorNode(DAGNode):
    """Node that expands generator syntax into branches at runtime."""

    generator_spec: Dict[str, Any]  # {"_or_": [...], "count": 5}

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        expanded = expand_spec(self.generator_spec)
        outputs = []
        for variant in expanded:
            child_node = runtime.dag.create_dynamic_node(variant, payload_in)
            outputs.append(child_node.execute(payload_in, runtime))
        return outputs
```

### 6. View Node (Reshape/Morph)

Changes how data is accessed without modifying it.

```python
class ViewNode(DAGNode):
    """Node that creates a different view of the dataset."""

    view_spec: ViewSpec  # layout, concat_sources, morph_type

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        new_view = payload_in.dataset_view.with_spec(self.view_spec)
        return payload_in.with_view(new_view)
```

### 7. Select Node (Filter/Pick)

Selects subset of data or predictions.

```python
class SelectNode(DAGNode):
    """Node that filters data or selects best predictions."""

    select_spec: SelectSpec  # filter samples, pick best model, etc.

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        if self.select_spec.type == "samples":
            mask = self.select_spec.evaluate(payload_in)
            return payload_in.with_sample_mask(mask)
        elif self.select_spec.type == "best_model":
            # Query prediction_store, return single best
            ...
```

---

## Execution Engine

### DAGExecutor

```python
class DAGExecutor:
    """Executes the pipeline DAG."""

    def __init__(self, dag: DAG, runtime: RuntimeContext):
        self.dag = dag
        self.runtime = runtime
        self.completed: Set[str] = set()

    def execute(self, initial_payload: Payload) -> Payload:
        """Execute DAG from START to END."""
        ready_queue = self._get_ready_nodes()

        while ready_queue:
            node = ready_queue.pop(0)

            # Collect inputs from parents
            inputs = self._collect_inputs(node)

            # Execute node
            if isinstance(node, ForkNode):
                outputs = node.execute(inputs[0], self.runtime)
                self._register_fork_outputs(node, outputs)
            elif isinstance(node, JoinNode):
                output = node.execute(inputs, self.runtime)
                self._register_output(node, output)
            else:
                output = node.execute(inputs[0], self.runtime)
                self._register_output(node, output)

            # Mark completed, update ready queue
            self.completed.add(node.node_id)
            ready_queue.extend(self._get_newly_ready())

        return self.dag.end_node.payload_out

    def _get_ready_nodes(self) -> List[DAGNode]:
        """Nodes whose parents are all completed."""
        return [n for n in self.dag.nodes.values()
                if all(p in self.completed for p in n.parents)
                and n.node_id not in self.completed]
```

### Dynamic Node Creation

```python
def create_dynamic_node(self, step_config: Any, parent_payload: Payload) -> DAGNode:
    """Create a node at runtime (for generators, TF build, etc.)."""
    parsed = self.step_parser.parse(step_config)
    controller = self.router.route(parsed, step_config)

    node = DAGNode(
        node_id=self._generate_id(parent_payload),
        step_index=parent_payload.step_index,
        controller=controller,
        operator=parsed.operator,
        config=step_config,
        parents=[parent_payload.parent_node_id],
        is_dynamic=True,
        branch_path=parent_payload.branch_path
    )

    self.dag.add_node(node)
    return node
```

### Handling Dynamic Feature Dimensions

```python
# In ModelController.execute():
def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
    # Feature dimension only known at this point
    X_sample = payload_in.dataset_view.sample_features()
    n_features = X_sample.shape[1]

    # Build model with actual dimensions
    if hasattr(self.operator, 'build'):
        self.operator.build(input_shape=(None, n_features))

    # Continue with training
    ...
```

---

## Data Access Model

### Controlled Views

Controllers access data through views, not raw dataset:

```python
class DataAccessor:
    """Provides controlled dataset access to controllers."""

    def __init__(self, dataset: SpectroDataset, view: DatasetView):
        self._dataset = dataset
        self._view = view

    def x(self, partition: Optional[str] = None) -> np.ndarray:
        """Get features for current view, optionally filtered by partition."""
        selector = self._view.to_selector()
        if partition:
            selector = selector.with_partition(partition)
        return self._dataset.x(selector)

    def y(self, version: Optional[str] = None) -> np.ndarray:
        """Get targets, optionally specific version."""
        return self._dataset.y(version or self._view.y_version)

    def indices(self, partition: str) -> np.ndarray:
        """Get sample indices for a partition."""
        return self._dataset._indexer.x_indices(self._view.to_selector())
```

### Immutable Updates

Dataset modifications create new views, not in-place changes:

```python
# Old pattern (mutable):
dataset.update_features(old_proc, X_new, new_proc, source)
context.selector.processing[source].append(new_proc)

# New pattern (immutable):
new_view = view.with_added_processing(source, new_proc)
dataset.register_processing(source, new_proc, X_new)  # Append-only
return payload.with_view(new_view)
```

---

## Serialization

### Pipeline Serialization (Existing)

Reuse `serialize_component()` and YAML normalization:

```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
  - branch:
      - - class: nirs4all.operators.transforms.SNV
      - - class: nirs4all.operators.transforms.MSC
  - merge: predictions
  - model:
      class: sklearn.linear_model.Ridge
```

### DAG Serialization (New)

For replay/export, serialize the executed DAG:

```yaml
dag:
  nodes:
    - id: node_1
      step_index: 1
      controller: TransformController
      operator:
        class: sklearn.preprocessing.MinMaxScaler
      parents: [START]
      artifacts: ["s1.MinMaxScaler$abc123:all"]

    - id: node_3_b0
      step_index: 3
      controller: TransformController
      operator:
        class: nirs4all.operators.transforms.SNV
      parents: [node_2]
      branch_path: [0]
      artifacts: ["s3.SNV$abc123:all:b0"]

  edges:
    - from: START
      to: node_1
      payload_snapshot: {processing: [["raw"]], y_version: "numeric"}
```

### ExecutionTrace (Existing - Reuse)

The existing `ExecutionTrace` already captures most DAG information:

```python
@dataclass
class ExecutionStep:
    step_index: int
    operator_type: str
    operator_class: str
    operator_config: Dict[str, Any]
    execution_mode: StepExecutionMode
    branch_path: List[int]
    artifacts: StepArtifacts
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
```

---

## Caching & Reproducibility

### Artifact Addressing

Artifacts are addressed by deterministic IDs:

```
Format: s{step}.{class}${hash}:{fold}:{branch}
Example: s3.PLSRegression$a1b2c3:0:b0

Components:
  - s{step}: Pipeline step index
  - {class}: Operator class name
  - ${hash}: Config hash (6 chars)
  - :{fold}: Fold ID or "all"
  - :{branch}: Branch path (b0, b0_1 for nested)
```

### Processing Chain Keys

Features are keyed by transformation history:

```
raw → SNV → SNV>SG_w5 → SNV>SG_w5>D1

Lookup: dataset.x(processing=[["SNV", "SG_w5", "D1"]])
```

### Reproducibility Guarantees

1. **Same config + same data = same artifacts**
   - Config hash ensures unique artifact IDs
   - Random seeds propagated through context

2. **Prediction replay**
   - Load `ExecutionTrace` from manifest
   - Extract `MinimalPipeline` via `TraceBasedExtractor`
   - Execute only required nodes with loaded artifacts

3. **Retrain from checkpoint**
   - Load artifacts up to step N
   - Continue training from step N+1
   - Useful for transfer learning, fine-tuning

---

## Branch/Merge Semantics

### Fork (Branch) Operation

```
Input: 1 payload
Output: N payloads (one per branch)

Semantics:
  - Snapshot current feature state
  - Clone payload N times
  - Set branch_path = parent.branch_path + [branch_id]
  - Execute branch substeps on each clone
  - Each branch produces independent artifacts
```

### Join (Merge) Operation

```
Input: N payloads (from branches)
Output: 1 payload

Modes:
  features:     Concatenate feature arrays horizontally
  predictions:  Collect OOF predictions, concatenate as features
  both:         Features + predictions

Semantics:
  - Collect outputs from all input payloads
  - Apply aggregation strategy (separate, mean, weighted)
  - Clear branch_path (exit branch mode)
  - Return unified payload
```

### Split Variants

| Operation | Split By | Use Case |
|-----------|----------|----------|
| `branch` | Execution path | Different preprocessing strategies |
| `source_branch` | Data source | Per-modality preprocessing |
| `fold_expand` | CV fold | Per-fold model training |
| `generator_expand` | Parameter variants | Hyperparameter search |

### Merge Variants

| Operation | Collect From | Output |
|-----------|--------------|--------|
| `merge: features` | Branch feature snapshots | Concatenated features |
| `merge: predictions` | Prediction store (OOF) | Predictions as features |
| `merge_sources` | Multiple data sources | Unified feature matrix |
| `merge_predictions` | Sequential models | Late fusion |

---

## Primitive Operations

The primitives are ordered by their typical execution sequence. Each primitive is **atomic** and can be composed to build any controller.

### Axiom 1: Materialize View

```
materialize(view) → (X, y, indices)
  - Convert view specification to actual arrays
  - Apply layout, concatenation, filtering
  - Returns: features, targets, sample indices
  - INVARIANT: Read-only, does not modify dataset
```

### Axiom 2: Transform Feature (Stateless)

```
apply_transform(X, operator) → X'
  - Apply fitted transformer to features
  - INVARIANT: Operator already fitted, pure function
  - Used in: prediction mode, test partition
```

### Axiom 3: Fit Transform

```
fit_transform(view, operator, partition="train") → (operator', X')
  - Materialize X from view (train partition only)
  - Fit operator on X_train
  - Transform X_all (including val/test)
  - Returns: fitted operator, transformed features
  - INVARIANT: Fit only sees train data
```

### Axiom 4: Register Processing

```
register_processing(dataset, source, name, X_new, parent?) → view'
  - Append new processing to dataset (append-only)
  - Update view with new processing chain
  - INVARIANT: Does not overwrite existing data
```

### Axiom 5: Transform Target

```
fit_transform_target(view, operator) → (operator', y', version_name)
  - Fit transformer on y_train
  - Transform all y
  - Register new version in targets
  - INVARIANT: Supports inverse_transform for predictions
```

### Axiom 6: Add Samples

```
add_samples(dataset, X_new, y_new, origin_ids, augmentation_name) → view'
  - Append samples to dataset
  - Register lineage (origin, augmentation method)
  - Mark as augmented (excluded from test partition)
  - INVARIANT: Augmented samples only in train partition
```

### Axiom 7: Filter Samples

```
filter_samples(view, mask_or_predicate) → view'
  - Create new view with sample_mask applied
  - Does not modify underlying dataset
  - INVARIANT: Original samples preserved
```

### Axiom 8: Assign Folds

```
assign_folds(dataset, splitter, groups?) → folds
  - Compute train/val splits using splitter
  - Respect groups if GroupKFold
  - Store fold assignments in dataset
  - INVARIANT: Folds are deterministic given seed
  - EFFECT: Enables OOF-safe data access downstream
```

### Axiom 9: Fork Execution

```
fork(view, N, by="branch") → [view_0, view_1, ..., view_n]
  - Snapshot current state
  - Create N independent views
  - Set identifiers: branch_path, fold_id, or source_id
  - INVARIANT: Views are independent, no shared mutable state
```

### Axiom 10: Train Model (Single Fold)

```
train_fold(view, model, fold_id) → (model', predictions_val, predictions_test)
  - Materialize X_train, y_train for this fold
  - Clone model, fit on train
  - Predict on val (held-out fold)
  - Predict on test (separate partition)
  - INVARIANT: Model never sees val during training
```

### Axiom 11: Store Predictions

```
store_predictions(pred_store, predictions, metadata) → pred_id
  - Add predictions to store with full provenance
  - Metadata: model_name, fold_id, branch_path, step_idx
  - INVARIANT: Append-only, predictions are immutable
```

### Axiom 12: Store Artifact

```
store_artifact(registry, object, metadata) → artifact_id
  - Serialize and store fitted operator/model
  - Generate deterministic artifact_id
  - INVARIANT: Artifacts are addressable and immutable
```

### Axiom 13: Join Execution

```
join(views[], mode, aggregation?) → view'
  - Collect outputs from all input views
  - Apply merge mode: features, predictions, both
  - Apply aggregation: concat, mean, weighted, best
  - Clear branch context
  - INVARIANT: OOF reconstruction for prediction merge
```

### Axiom 14: Select Best

```
select_best(pred_store, criteria) → prediction_ref
  - Query store with filter (metric, partition, branch)
  - Rank by validation score
  - Return reference to best prediction/model
  - INVARIANT: Selection does not modify predictions
```

### Axiom 15: Aggregate Folds

```
aggregate_folds(predictions[], strategy) → virtual_prediction
  - Combine fold predictions into single prediction
  - Strategies: mean, weighted_mean, vote, all
  - Create "virtual model" reference
  - INVARIANT: Preserves individual fold predictions
```

---

## Axiomatic Execution Sequences

This section shows the **exact sequence of axioms** for each controller type.

### TransformController (X preprocessing)

```python
# Sequence: fit_transform → register_processing
def execute_transform(view, operator, source):
    # 1. Fit and transform
    operator_fitted, X_new = fit_transform(view, operator, partition="train")

    # 2. Register new processing
    new_view = register_processing(dataset, source, operator.name, X_new, parent=view.processing)

    # 3. Store artifact
    artifact_id = store_artifact(registry, operator_fitted, {"step": step_idx, "source": source})

    return new_view, [artifact_id]
```

### YProcessingController (y scaling)

```python
# Sequence: fit_transform_target
def execute_y_processing(view, operator):
    # 1. Fit and transform targets
    operator_fitted, y_new, version = fit_transform_target(view, operator)

    # 2. Store artifact for inverse_transform
    artifact_id = store_artifact(registry, operator_fitted, {"step": step_idx, "type": "y_transform"})

    # 3. Update view with new y version
    return view.with_y_version(version), [artifact_id]
```

### FeatureAugmentationController

```python
# Sequence: for each operator in list: fit_transform → register
def execute_feature_augmentation(view, operators):
    artifacts = []
    processing_names = []

    for op in operators:
        # 1. Fit and transform
        op_fitted, X_new = fit_transform(view, op, partition="train")

        # 2. Register as separate processing
        proc_name = f"{view.current_processing}>{op.name}"
        register_processing(dataset, source, proc_name, X_new)
        processing_names.append(proc_name)

        # 3. Store artifact
        artifacts.append(store_artifact(registry, op_fitted, {...}))

    # 4. View now has multiple processings (concatenated on materialize)
    return view.with_processings(processing_names), artifacts
```

### SampleAugmentationController

```python
# Sequence: for each operator: generate samples → add_samples
def execute_sample_augmentation(view, operators, count):
    artifacts = []

    for op in operators:
        X_orig, y_orig, indices = materialize(view.with_partition("train"))

        for i in range(count):
            # 1. Generate augmented samples
            X_aug = op.transform(X_orig)

            # 2. Add to dataset
            add_samples(dataset, X_aug, y_orig, origin_ids=indices, augmentation_name=op.name)

    # View unchanged (augmented samples auto-included when partition="train")
    return view, []
```

### SplitterController (CV assignment)

```python
# Sequence: assign_folds
def execute_splitter(view, splitter, groups=None):
    # 1. Assign folds
    folds = assign_folds(dataset, splitter, groups)

    # No artifacts (folds stored in dataset)
    # EFFECT: All downstream operations are now OOF-safe
    return view.with_folds(folds), []
```

### ModelController

```python
# Sequence: fork(folds) → train_fold × N → store_predictions → aggregate_folds → select_best
def execute_model(view, model):
    n_folds = len(view.folds)
    artifacts = []
    all_predictions = []

    # 1. Fork by fold (conceptually)
    fold_views = fork(view, n_folds, by="fold")

    for fold_id, fold_view in enumerate(fold_views):
        # 2. Train on this fold
        model_fitted, preds_val, preds_test = train_fold(fold_view, model, fold_id)

        # 3. Store predictions
        store_predictions(pred_store, preds_val, {"partition": "val", "fold": fold_id, ...})
        store_predictions(pred_store, preds_test, {"partition": "test", "fold": fold_id, ...})

        # 4. Store model artifact
        artifacts.append(store_artifact(registry, model_fitted, {"fold": fold_id, ...}))

        all_predictions.append((preds_val, preds_test))

    # 5. Aggregate folds → virtual model
    virtual_pred = aggregate_folds(all_predictions, strategy="weighted_mean")
    store_predictions(pred_store, virtual_pred, {"partition": "val", "fold": "virtual", ...})

    return view, artifacts
```

### BranchController

```python
# Sequence: fork → execute each branch → return multiple views
def execute_branch(view, branch_definitions):
    # 1. Fork by branch count
    branch_views = fork(view, len(branch_definitions), by="branch")

    result_views = []
    all_artifacts = []

    for branch_id, (branch_view, branch_steps) in enumerate(zip(branch_views, branch_definitions)):
        # 2. Execute each step in this branch
        current_view = branch_view
        for step in branch_steps:
            current_view, step_artifacts = execute_step(step, current_view)
            all_artifacts.extend(step_artifacts)

        result_views.append(current_view)

    # 3. Return multiple views (merge handles joining)
    return result_views, all_artifacts
```

### MergeController

```python
# Sequence: collect from branches → join → return single view
def execute_merge(views, mode, aggregation="concat"):
    # 1. Collect based on mode
    if mode == "features":
        # Collect feature snapshots from each branch
        features_list = [materialize(v)[0] for v in views]
        merged = np.concatenate(features_list, axis=1)

    elif mode == "predictions":
        # Collect OOF predictions from each branch
        for view in views:
            branch_preds = query_predictions(pred_store, branch_path=view.branch_path)
            # OOF reconstruction: for train, use held-out fold predictions
            oof_preds = reconstruct_oof(branch_preds)
        merged = np.concatenate([oof for oof in all_oof_preds], axis=1)

    # 2. Register merged features
    new_view = register_processing(dataset, source=0, name="merged", merged)

    # 3. Join: clear branch context
    return join(views, mode), []
```

### Generator in Branch (Expansion)

```python
# Before execution (in DAGBuilder):
def expand_generator_in_branch(generator_spec):
    # {"_or_": [A, B, C], "count": 2}
    variants = expand_spec(generator_spec)  # → [A, B, C] or combinations

    # Create branch definitions
    branch_defs = [[variant] for variant in variants]

    return {"branch": branch_defs}

# Execution is then standard BranchController
```

---

## Rationale

### Why This Design Fits nirs4all

1. **Preserves linear syntax**: `[step1, step2]` implicitly chains nodes
2. **Preserves controllers**: Controllers become node executors, no rewrite
3. **Preserves dataset**: DatasetView adds immutable layer without replacing Features
4. **Preserves artifacts**: ArtifactRegistry already supports DAG-like addressing
5. **Preserves predictions**: Prediction store is already shared/append-only

### Why It Avoids Overengineering

1. **No external scheduler**: Single-threaded topological sort
2. **No complex DSL**: Same YAML/Python syntax
3. **Optional parallelism**: Can add later for folds/branches
4. **Minimal new classes**: Payload, DatasetView, DAGNode, DAGExecutor

### Why It Supports Runtime Build

1. **Dynamic node creation**: `create_dynamic_node()` handles generators
2. **Feature dim at execution**: Node executes after parents complete
3. **Conditional expansion**: Nodes can create children based on results

### Why Primitives Matter

1. **Testable**: Each primitive has clear semantics
2. **Composable**: Controllers = combinations of primitives
3. **Debuggable**: Trace shows which primitives ran
4. **Extensible**: New controllers compose existing primitives

---

## YAML Pipeline Examples

### Example 1: Simple Preprocessing + Model

```yaml
# Classification pipeline with standard preprocessing
pipeline:
  # Step 1: Scale features
  - class: sklearn.preprocessing.StandardScaler

  # Step 2: Scale targets (for regression)
  - y_processing:
      class: sklearn.preprocessing.MinMaxScaler

  # Step 3: Cross-validation
  - class: sklearn.model_selection.StratifiedKFold
    params:
      n_splits: 5
      shuffle: true
      random_state: 42

  # Step 4: Train model
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

**DAG Structure**:
```
START → StandardScaler → MinMaxScaler(y) → StratifiedKFold → PLS[fold×5] → END
```

### Example 2: Multi-Branch with Merge

```yaml
# Compare preprocessing strategies, stack predictions
pipeline:
  # Step 1: Common preprocessing
  - class: sklearn.preprocessing.MinMaxScaler

  # Step 2: Cross-validation setup
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3
      test_size: 0.2

  # Step 3: Branching - different preprocessing paths
  - branch:
      # Branch 0: SNV + first derivative
      - - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.FirstDerivative
        - model:
            class: sklearn.cross_decomposition.PLSRegression
            params:
              n_components: 8

      # Branch 1: MSC + PCA
      - - class: nirs4all.operators.transforms.MSC
        - class: sklearn.decomposition.PCA
          params:
            n_components: 20
        - model:
            class: sklearn.ensemble.RandomForestRegressor
            params:
              n_estimators: 100

      # Branch 2: Raw + AutoML
      - - model:
            class: xgboost.XGBRegressor
            params:
              n_estimators: 200
              max_depth: 6

  # Step 4: Merge predictions from all branches
  - merge: predictions

  # Step 5: Meta-model (stacking)
  - model:
      class: sklearn.linear_model.Ridge
      params:
        alpha: 1.0
```

**DAG Structure**:
```
START → MinMax → ShuffleSplit → FORK(3)
                                  ├─ SNV → D1 → PLS[×3] ──────────┐
                                  ├─ MSC → PCA → RF[×3] ──────────┤
                                  └─ XGB[×3] ─────────────────────┘
                                                                   ↓
                                                             JOIN(predictions)
                                                                   ↓
                                                              Ridge[×3] → END
```

### Example 3: Multi-Source with Source-Specific Preprocessing

```yaml
# Multi-modal dataset: NIR spectra + chemical markers
pipeline:
  # Step 1: Source-specific preprocessing
  - source_branch:
      NIR:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.SavitzkyGolay
          params:
            window_length: 11
            polyorder: 2
        - class: sklearn.preprocessing.StandardScaler

      markers:
        - class: sklearn.feature_selection.VarianceThreshold
          params:
            threshold: 0.01
        - class: sklearn.preprocessing.MinMaxScaler

  # Step 2: Merge sources
  - merge_sources: concat

  # Step 3: Feature selection on combined
  - class: sklearn.feature_selection.SelectKBest
    params:
      k: 50

  # Step 4: Cross-validation
  - class: sklearn.model_selection.KFold
    params:
      n_splits: 5

  # Step 5: Train ensemble
  - model:
      class: sklearn.ensemble.GradientBoostingRegressor
      params:
        n_estimators: 100
        learning_rate: 0.1
```

**DAG Structure**:
```
START → SOURCE_FORK(2)
          ├─ NIR: SNV → SG → StdScaler ────────┐
          └─ markers: VarThresh → MinMax ──────┘
                                               ↓
                                        SOURCE_JOIN(concat)
                                               ↓
                                         SelectKBest
                                               ↓
                                            KFold
                                               ↓
                                          GBR[×5] → END
```

### Example 4: Generator Syntax with Runtime Expansion

```yaml
# Hyperparameter search via generator
pipeline:
  # Step 1: Preprocessing
  - class: sklearn.preprocessing.StandardScaler

  # Step 2: CV setup
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3

  # Step 3: Branch with generator - creates N branches at runtime
  - branch:
      _or_:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.MSC
        - class: nirs4all.operators.transforms.Detrend
        - class: nirs4all.operators.transforms.Gaussian
          params:
            sigma: 2
      count: 4  # Generate 4 combinations

  # Step 4: Model with range generator
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components:
          _range_: [5, 20, 5]  # [5, 10, 15, 20] - expands to 4 variants

  # Step 5: Select best
  - merge: features
```

**DAG Structure (at runtime)**:
```
START → StdScaler → ShuffleSplit → FORK(4 from _or_)
                                     ├─ SNV ─────────────┬─ PLS(n=5)[×3]
                                     │                   ├─ PLS(n=10)[×3]
                                     │                   ├─ PLS(n=15)[×3]
                                     │                   └─ PLS(n=20)[×3]
                                     ├─ MSC ─────────────┬─ ...
                                     ├─ Detrend ─────────┬─ ...
                                     └─ Gaussian(σ=2) ───┴─ ...
                                                         ↓
                                                   JOIN(features) → END
```

### Example 5: Outlier Handling + Sample Augmentation

```yaml
# Robust pipeline with augmentation
pipeline:
  # Step 1: Base scaling
  - class: sklearn.preprocessing.RobustScaler

  # Step 2: Outlier detection and exclusion
  - outlier_excluder:
      class: sklearn.ensemble.IsolationForest
      params:
        contamination: 0.05
      exclude_mode: train_only  # Exclude outliers from training, keep for test

  # Step 3: Sample augmentation (create variations)
  - sample_augmentation:
      transformers:
        - class: nirs4all.operators.augmentation.NoiseInjection
          params:
            noise_level: 0.01
        - class: nirs4all.operators.augmentation.SpectrumShift
          params:
            max_shift: 2
      count: 3  # 3 augmented copies per sample

  # Step 4: Feature augmentation (multiple preprocessings)
  - feature_augmentation:
      _or_:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.FirstDerivative
      size: 2  # Use both, concatenate

  # Step 5: Cross-validation
  - class: sklearn.model_selection.GroupKFold
    params:
      n_splits: 5

  # Step 6: Model
  - model:
      class: sklearn.neural_network.MLPRegressor
      params:
        hidden_layer_sizes: [64, 32]
        max_iter: 500
```

**DAG Structure**:
```
START → RobustScaler → IsolationForest(exclude) → SampleAug(×3)
                                                       ↓
                                               FeatureAug(SNV+D1)
                                                       ↓
                                                  GroupKFold
                                                       ↓
                                                  MLP[×5] → END
```

---

## Summary

This specification defines a DAG-based execution model for nirs4all that:

1. **Preserves** the linear pipeline syntax users know
2. **Formalizes** branch/merge as graph fork/join operations
3. **Supports** runtime node creation for dynamic scenarios
4. **Defines** 10 primitive operations that controllers compose
5. **Maintains** compatibility with existing serialization and artifacts
6. **Enables** deterministic replay, caching, and partial re-execution

The design is intentionally **pragmatic**: it adds DAG semantics without requiring a wholesale rewrite, leveraging the existing controller pattern and artifact infrastructure.
# Document 2: DAG Pipeline Specification

**Date**: December 2025
**Author**: Architecture Design
**Status**: Target Specification

---

## Table of Contents

1. [Objectives Restated](#objectives-restated)
2. [Core Concepts](#core-concepts)
3. [Node Types](#node-types)
4. [Execution Engine](#execution-engine)
5. [Data Access Model](#data-access-model)
6. [Serialization](#serialization)
7. [Caching & Reproducibility](#caching--reproducibility)
8. [Branch/Merge Semantics](#branchmerge-semantics)
9. [Primitive Operations](#primitive-operations)
10. [Rationale](#rationale)
11. [YAML Pipeline Examples](#yaml-pipeline-examples)

---

## Objectives Restated

### Goals

1. **Preserve existing code**: Reuse controllers, dataset, predictions, artifact storage, serialization
2. **DAG execution**: Nodes are controllers, edges carry payload (dataset view + context)
3. **Linear syntax**: Keep `[step1, step2, step3]` where each step implicitly depends on the previous
4. **Dynamic expansion**: Support runtime node creation (TF model build, generator branches)
5. **Primitives-first**: Define small set of atomic operations; controllers = compositions
6. **Runtime decisions**: Auto-operators, best-model selection, dynamic feature dims
7. **Deterministic replay**: Save/reload pipeline paths including branches
8. **Efficient migration**: Incremental changes, backward-compatible

### Non-Goals

- Heavy workflow engine (Airflow, Prefect, etc.)
- Distributed execution (single-process, optional parallelism)
- Complete API redesign (evolution, not revolution)

---

## Core Concepts

### Payload

The unit of data flowing on DAG edges:

```python
@dataclass
class Payload:
    """Immutable snapshot of pipeline state at a DAG edge."""

    # Data views (references, not copies)
    dataset_view: DatasetView       # Selector + snapshot reference
    y_version: str                  # Current target transformation

    # Accumulated state
    processing_chains: List[List[str]]  # Per-source processing history
    branch_path: List[int]              # Current branch hierarchy

    # Shared stores (mutable, but append-only during execution)
    prediction_store: Predictions       # Shared across all paths
    artifact_registry: ArtifactRegistry # Shared across all paths

    # Lineage
    parent_node_id: Optional[str]
    step_index: int
```

### DatasetView

Immutable reference to a dataset slice:

```python
@dataclass(frozen=True)
class DatasetView:
    """Immutable specification of which data to access."""
    partition: str
    processing: Tuple[Tuple[str, ...], ...]  # Immutable per-source
    layout: str
    source_indices: Tuple[int, ...]
    sample_mask: Optional[np.ndarray]        # For fold/filter
    include_augmented: bool
    include_excluded: bool
```

### Node

```python
@dataclass
class DAGNode:
    """A node in the execution DAG."""
    node_id: str                        # "step_3_branch_0_fold_1"
    step_index: int                     # Original pipeline step index
    controller: Type[OperatorController]
    operator: Any                       # Deserialized operator instance
    config: Dict[str, Any]              # Original step config

    # Graph structure
    parents: List[str]                  # Input node IDs
    children: List[str]                 # Output node IDs (populated during execution)

    # Execution state
    status: NodeStatus                  # PENDING, RUNNING, COMPLETED, SKIPPED
    payload_in: Optional[Payload]       # Set when ready to execute
    payload_out: Optional[Payload]      # Set after execution
    artifacts: List[ArtifactRecord]     # Produced artifacts

    # Dynamic metadata
    branch_path: List[int]
    fold_id: Optional[int]
    is_dynamic: bool                    # Created at runtime
```

### DAG Structure

```
     ┌──────────┐
     │  START   │ ← Payload(dataset_view=initial, y_version="numeric")
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 1   │  MinMaxScaler
     │ (node_1) │
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 2   │  ShuffleSplit (assigns folds, no data change)
     │ (node_2) │
     └────┬─────┘
          │
     ┌────▼─────┐
     │ Step 3   │  {branch: [[SNV], [MSC]]}
     │ (branch) │
     └────┬─────┘
          │
    ┌─────┼─────┐
    │           │
┌───▼───┐ ┌───▼───┐
│node_3a│ │node_3b│   SNV path    MSC path
│ SNV   │ │ MSC   │
└───┬───┘ └───┬───┘
    │         │
┌───▼───┐ ┌───▼───┐
│node_4a│ │node_4b│   PLS on each branch
│ PLS   │ │ PLS   │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
    ┌────▼─────┐
    │ Step 5   │  {merge: "predictions"}
    │ (merge)  │
    └────┬─────┘
         │
    ┌────▼─────┐
    │ Step 6   │  Ridge (meta-model)
    │ (model)  │
    └────┬─────┘
         │
    ┌────▼─────┐
    │   END    │
    └──────────┘
```

---

## Node Types

### 1. Operator Node (Standard)

Executes a single operator (transformer, model, splitter).

```python
class OperatorNode(DAGNode):
    """Standard node that applies an operator."""

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        result = self.controller().execute(
            step_info=self.to_parsed_step(),
            dataset=payload_in.dataset_view.materialize(),
            context=payload_in.to_execution_context(),
            runtime_context=runtime,
            ...
        )
        return payload_in.with_updates(result)
```

### 2. Fork Node (Branch)

Creates multiple output edges from one input.

```python
class ForkNode(DAGNode):
    """Node that splits execution into multiple branches."""

    branch_count: int
    branch_definitions: List[List[Any]]  # Substeps per branch

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        outputs = []
        for branch_id, branch_steps in enumerate(self.branch_definitions):
            branch_payload = payload_in.with_branch(branch_id)
            # Create child nodes for branch substeps
            for substep in branch_steps:
                child_node = runtime.dag.create_dynamic_node(substep, branch_payload)
                branch_payload = child_node.execute(branch_payload, runtime)
            outputs.append(branch_payload)
        return outputs
```

### 3. Join Node (Merge)

Combines multiple input edges into one output.

```python
class JoinNode(DAGNode):
    """Node that merges multiple branch outputs."""

    merge_mode: MergeMode  # FEATURES, PREDICTIONS, BOTH

    def execute(self, payloads_in: List[Payload], runtime: RuntimeContext) -> Payload:
        merged = self._merge_payloads(payloads_in, self.merge_mode)
        return merged.with_branch_exit()
```

### 4. Fold Node (CV Expansion)

Creates per-fold execution paths.

```python
class FoldNode(DAGNode):
    """Node that expands to per-fold model training."""

    n_folds: int

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        fold_payloads = []
        for fold_id in range(self.n_folds):
            fold_payload = payload_in.with_fold(fold_id)
            fold_payloads.append(fold_payload)
        return fold_payloads
```

### 5. Generator Node (Dynamic Expansion)

Expands at runtime based on generator syntax inside branches.

```python
class GeneratorNode(DAGNode):
    """Node that expands generator syntax into branches at runtime."""

    generator_spec: Dict[str, Any]  # {"_or_": [...], "count": 5}

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> List[Payload]:
        expanded = expand_spec(self.generator_spec)
        outputs = []
        for variant in expanded:
            child_node = runtime.dag.create_dynamic_node(variant, payload_in)
            outputs.append(child_node.execute(payload_in, runtime))
        return outputs
```

### 6. View Node (Reshape/Morph)

Changes how data is accessed without modifying it.

```python
class ViewNode(DAGNode):
    """Node that creates a different view of the dataset."""

    view_spec: ViewSpec  # layout, concat_sources, morph_type

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        new_view = payload_in.dataset_view.with_spec(self.view_spec)
        return payload_in.with_view(new_view)
```

### 7. Select Node (Filter/Pick)

Selects subset of data or predictions.

```python
class SelectNode(DAGNode):
    """Node that filters data or selects best predictions."""

    select_spec: SelectSpec  # filter samples, pick best model, etc.

    def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
        if self.select_spec.type == "samples":
            mask = self.select_spec.evaluate(payload_in)
            return payload_in.with_sample_mask(mask)
        elif self.select_spec.type == "best_model":
            # Query prediction_store, return single best
            ...
```

---

## Execution Engine

### DAGExecutor

```python
class DAGExecutor:
    """Executes the pipeline DAG."""

    def __init__(self, dag: DAG, runtime: RuntimeContext):
        self.dag = dag
        self.runtime = runtime
        self.completed: Set[str] = set()

    def execute(self, initial_payload: Payload) -> Payload:
        """Execute DAG from START to END."""
        ready_queue = self._get_ready_nodes()

        while ready_queue:
            node = ready_queue.pop(0)

            # Collect inputs from parents
            inputs = self._collect_inputs(node)

            # Execute node
            if isinstance(node, ForkNode):
                outputs = node.execute(inputs[0], self.runtime)
                self._register_fork_outputs(node, outputs)
            elif isinstance(node, JoinNode):
                output = node.execute(inputs, self.runtime)
                self._register_output(node, output)
            else:
                output = node.execute(inputs[0], self.runtime)
                self._register_output(node, output)

            # Mark completed, update ready queue
            self.completed.add(node.node_id)
            ready_queue.extend(self._get_newly_ready())

        return self.dag.end_node.payload_out

    def _get_ready_nodes(self) -> List[DAGNode]:
        """Nodes whose parents are all completed."""
        return [n for n in self.dag.nodes.values()
                if all(p in self.completed for p in n.parents)
                and n.node_id not in self.completed]
```

### Dynamic Node Creation

```python
def create_dynamic_node(self, step_config: Any, parent_payload: Payload) -> DAGNode:
    """Create a node at runtime (for generators, TF build, etc.)."""
    parsed = self.step_parser.parse(step_config)
    controller = self.router.route(parsed, step_config)

    node = DAGNode(
        node_id=self._generate_id(parent_payload),
        step_index=parent_payload.step_index,
        controller=controller,
        operator=parsed.operator,
        config=step_config,
        parents=[parent_payload.parent_node_id],
        is_dynamic=True,
        branch_path=parent_payload.branch_path
    )

    self.dag.add_node(node)
    return node
```

### Handling Dynamic Feature Dimensions

```python
# In ModelController.execute():
def execute(self, payload_in: Payload, runtime: RuntimeContext) -> Payload:
    # Feature dimension only known at this point
    X_sample = payload_in.dataset_view.sample_features()
    n_features = X_sample.shape[1]

    # Build model with actual dimensions
    if hasattr(self.operator, 'build'):
        self.operator.build(input_shape=(None, n_features))

    # Continue with training
    ...
```

---

## Data Access Model

### Controlled Views

Controllers access data through views, not raw dataset:

```python
class DataAccessor:
    """Provides controlled dataset access to controllers."""

    def __init__(self, dataset: SpectroDataset, view: DatasetView):
        self._dataset = dataset
        self._view = view

    def x(self, partition: Optional[str] = None) -> np.ndarray:
        """Get features for current view, optionally filtered by partition."""
        selector = self._view.to_selector()
        if partition:
            selector = selector.with_partition(partition)
        return self._dataset.x(selector)

    def y(self, version: Optional[str] = None) -> np.ndarray:
        """Get targets, optionally specific version."""
        return self._dataset.y(version or self._view.y_version)

    def indices(self, partition: str) -> np.ndarray:
        """Get sample indices for a partition."""
        return self._dataset._indexer.x_indices(self._view.to_selector())
```

### Immutable Updates

Dataset modifications create new views, not in-place changes:

```python
# Old pattern (mutable):
dataset.update_features(old_proc, X_new, new_proc, source)
context.selector.processing[source].append(new_proc)

# New pattern (immutable):
new_view = view.with_added_processing(source, new_proc)
dataset.register_processing(source, new_proc, X_new)  # Append-only
return payload.with_view(new_view)
```

---

## Serialization

### Pipeline Serialization (Existing)

Reuse `serialize_component()` and YAML normalization:

```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
  - branch:
      - - class: nirs4all.operators.transforms.SNV
      - - class: nirs4all.operators.transforms.MSC
  - merge: predictions
  - model:
      class: sklearn.linear_model.Ridge
```

### DAG Serialization (New)

For replay/export, serialize the executed DAG:

```yaml
dag:
  nodes:
    - id: node_1
      step_index: 1
      controller: TransformController
      operator:
        class: sklearn.preprocessing.MinMaxScaler
      parents: [START]
      artifacts: ["s1.MinMaxScaler$abc123:all"]

    - id: node_3_b0
      step_index: 3
      controller: TransformController
      operator:
        class: nirs4all.operators.transforms.SNV
      parents: [node_2]
      branch_path: [0]
      artifacts: ["s3.SNV$abc123:all:b0"]

  edges:
    - from: START
      to: node_1
      payload_snapshot: {processing: [["raw"]], y_version: "numeric"}
```

### ExecutionTrace (Existing - Reuse)

The existing `ExecutionTrace` already captures most DAG information:

```python
@dataclass
class ExecutionStep:
    step_index: int
    operator_type: str
    operator_class: str
    operator_config: Dict[str, Any]
    execution_mode: StepExecutionMode
    branch_path: List[int]
    artifacts: StepArtifacts
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
```

---

## Caching & Reproducibility

### Artifact Addressing

Artifacts are addressed by deterministic IDs:

```
Format: s{step}.{class}${hash}:{fold}:{branch}
Example: s3.PLSRegression$a1b2c3:0:b0

Components:
  - s{step}: Pipeline step index
  - {class}: Operator class name
  - ${hash}: Config hash (6 chars)
  - :{fold}: Fold ID or "all"
  - :{branch}: Branch path (b0, b0_1 for nested)
```

### Processing Chain Keys

Features are keyed by transformation history:

```
raw → SNV → SNV>SG_w5 → SNV>SG_w5>D1

Lookup: dataset.x(processing=[["SNV", "SG_w5", "D1"]])
```

### Reproducibility Guarantees

1. **Same config + same data = same artifacts**
   - Config hash ensures unique artifact IDs
   - Random seeds propagated through context

2. **Prediction replay**
   - Load `ExecutionTrace` from manifest
   - Extract `MinimalPipeline` via `TraceBasedExtractor`
   - Execute only required nodes with loaded artifacts

3. **Retrain from checkpoint**
   - Load artifacts up to step N
   - Continue training from step N+1
   - Useful for transfer learning, fine-tuning

---

## Branch/Merge Semantics

### Fork (Branch) Operation

```
Input: 1 payload
Output: N payloads (one per branch)

Semantics:
  - Snapshot current feature state
  - Clone payload N times
  - Set branch_path = parent.branch_path + [branch_id]
  - Execute branch substeps on each clone
  - Each branch produces independent artifacts
```

### Join (Merge) Operation

```
Input: N payloads (from branches)
Output: 1 payload

Modes:
  features:     Concatenate feature arrays horizontally
  predictions:  Collect OOF predictions, concatenate as features
  both:         Features + predictions

Semantics:
  - Collect outputs from all input payloads
  - Apply aggregation strategy (separate, mean, weighted)
  - Clear branch_path (exit branch mode)
  - Return unified payload
```

### Split Variants

| Operation | Split By | Use Case |
|-----------|----------|----------|
| `branch` | Execution path | Different preprocessing strategies |
| `source_branch` | Data source | Per-modality preprocessing |
| `fold_expand` | CV fold | Per-fold model training |
| `generator_expand` | Parameter variants | Hyperparameter search |

### Merge Variants

| Operation | Collect From | Output |
|-----------|--------------|--------|
| `merge: features` | Branch feature snapshots | Concatenated features |
| `merge: predictions` | Prediction store (OOF) | Predictions as features |
| `merge_sources` | Multiple data sources | Unified feature matrix |
| `merge_predictions` | Sequential models | Late fusion |

---

## Primitive Operations

### Axiom 1: Transform Feature

```
transform_feature(view, operator, source?) → view'
  - Apply TransformerMixin to X at specified processing
  - Register new processing in dataset
  - Return view with updated processing chain
```

### Axiom 2: Transform Target

```
transform_target(view, operator) → view'
  - Apply transformer to y
  - Register new y version
  - Return view with updated y_version
```

### Axiom 3: Add Samples

```
add_samples(view, X_new, indices, origin) → view'
  - Append samples to dataset (augmentation)
  - Register sample lineage
  - Return view (sample mask unchanged, new samples appended)
```

### Axiom 4: Filter Samples

```
filter_samples(view, mask) → view'
  - Create new view with sample_mask
  - Does not modify dataset
```

### Axiom 5: Assign Folds

```
assign_folds(view, splitter) → view'
  - Compute train/val splits
  - Store in dataset._folds
  - Return view (folds accessible via fold_id)
```

### Axiom 6: Train Model

```
train_model(view, model, fold_id?) → (predictions, artifact)
  - Fit model on X_train, y_train
  - Predict on X_val, X_test
  - Register predictions in store
  - Return artifact reference
```

### Axiom 7: Fork Execution

```
fork(view, branches) → [view_0, view_1, ..., view_n]
  - Snapshot current state
  - Create N independent views with branch_path
```

### Axiom 8: Join Execution

```
join([view_0, ..., view_n], mode) → view'
  - Collect features/predictions from all branches
  - Clear branch_path
  - Return unified view
```

### Axiom 9: Select Best

```
select_best(prediction_store, criteria) → prediction_ref
  - Query store with filter (metric, partition, branch)
  - Return reference to best prediction
```

### Axiom 10: Materialize View

```
materialize(view) → (X, y)
  - Convert view specification to actual arrays
  - Apply layout, concatenation, filtering
```

### Controller Compositions

```python
# TransformController = transform_feature composition
class TransformController:
    def execute(self, view, operator, source):
        X = materialize(view)
        operator.fit(X)
        X_new = operator.transform(X)
        return transform_feature(view, operator, source)

# ModelController = train_model + select_best composition
class ModelController:
    def execute(self, view, model):
        for fold_id in folds:
            preds, artifact = train_model(view, model, fold_id)
        best = select_best(prediction_store, metric)
        return view

# BranchController = fork composition
class BranchController:
    def execute(self, view, branch_defs):
        views = fork(view, len(branch_defs))
        for i, (v, steps) in enumerate(zip(views, branch_defs)):
            for step in steps:
                v = execute_step(step, v)
            views[i] = v
        return views

# MergeController = join composition
class MergeController:
    def execute(self, views, mode):
        return join(views, mode)
```

---

## Rationale

### Why This Design Fits nirs4all

1. **Preserves linear syntax**: `[step1, step2]` implicitly chains nodes
2. **Preserves controllers**: Controllers become node executors, no rewrite
3. **Preserves dataset**: DatasetView adds immutable layer without replacing Features
4. **Preserves artifacts**: ArtifactRegistry already supports DAG-like addressing
5. **Preserves predictions**: Prediction store is already shared/append-only

### Why It Avoids Overengineering

1. **No external scheduler**: Single-threaded topological sort
2. **No complex DSL**: Same YAML/Python syntax
3. **Optional parallelism**: Can add later for folds/branches
4. **Minimal new classes**: Payload, DatasetView, DAGNode, DAGExecutor

### Why It Supports Runtime Build

1. **Dynamic node creation**: `create_dynamic_node()` handles generators
2. **Feature dim at execution**: Node executes after parents complete
3. **Conditional expansion**: Nodes can create children based on results

### Why Primitives Matter

1. **Testable**: Each primitive has clear semantics
2. **Composable**: Controllers = combinations of primitives
3. **Debuggable**: Trace shows which primitives ran
4. **Extensible**: New controllers compose existing primitives

---

## YAML Pipeline Examples

### Example 1: Simple Preprocessing + Model

```yaml
# Classification pipeline with standard preprocessing
pipeline:
  # Step 1: Scale features
  - class: sklearn.preprocessing.StandardScaler

  # Step 2: Scale targets (for regression)
  - y_processing:
      class: sklearn.preprocessing.MinMaxScaler

  # Step 3: Cross-validation
  - class: sklearn.model_selection.StratifiedKFold
    params:
      n_splits: 5
      shuffle: true
      random_state: 42

  # Step 4: Train model
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

**DAG Structure**:
```
START → StandardScaler → MinMaxScaler(y) → StratifiedKFold → PLS[fold×5] → END
```

### Example 2: Multi-Branch with Merge

```yaml
# Compare preprocessing strategies, stack predictions
pipeline:
  # Step 1: Common preprocessing
  - class: sklearn.preprocessing.MinMaxScaler

  # Step 2: Cross-validation setup
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3
      test_size: 0.2

  # Step 3: Branching - different preprocessing paths
  - branch:
      # Branch 0: SNV + first derivative
      - - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.FirstDerivative
        - model:
            class: sklearn.cross_decomposition.PLSRegression
            params:
              n_components: 8

      # Branch 1: MSC + PCA
      - - class: nirs4all.operators.transforms.MSC
        - class: sklearn.decomposition.PCA
          params:
            n_components: 20
        - model:
            class: sklearn.ensemble.RandomForestRegressor
            params:
              n_estimators: 100

      # Branch 2: Raw + AutoML
      - - model:
            class: xgboost.XGBRegressor
            params:
              n_estimators: 200
              max_depth: 6

  # Step 4: Merge predictions from all branches
  - merge: predictions

  # Step 5: Meta-model (stacking)
  - model:
      class: sklearn.linear_model.Ridge
      params:
        alpha: 1.0
```

**DAG Structure**:
```
START → MinMax → ShuffleSplit → FORK(3)
                                  ├─ SNV → D1 → PLS[×3] ──────────┐
                                  ├─ MSC → PCA → RF[×3] ──────────┤
                                  └─ XGB[×3] ─────────────────────┘
                                                                   ↓
                                                             JOIN(predictions)
                                                                   ↓
                                                              Ridge[×3] → END
```

### Example 3: Multi-Source with Source-Specific Preprocessing

```yaml
# Multi-modal dataset: NIR spectra + chemical markers
pipeline:
  # Step 1: Source-specific preprocessing
  - source_branch:
      NIR:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.SavitzkyGolay
          params:
            window_length: 11
            polyorder: 2
        - class: sklearn.preprocessing.StandardScaler

      markers:
        - class: sklearn.feature_selection.VarianceThreshold
          params:
            threshold: 0.01
        - class: sklearn.preprocessing.MinMaxScaler

  # Step 2: Merge sources
  - merge_sources: concat

  # Step 3: Feature selection on combined
  - class: sklearn.feature_selection.SelectKBest
    params:
      k: 50

  # Step 4: Cross-validation
  - class: sklearn.model_selection.KFold
    params:
      n_splits: 5

  # Step 5: Train ensemble
  - model:
      class: sklearn.ensemble.GradientBoostingRegressor
      params:
        n_estimators: 100
        learning_rate: 0.1
```

**DAG Structure**:
```
START → SOURCE_FORK(2)
          ├─ NIR: SNV → SG → StdScaler ────────┐
          └─ markers: VarThresh → MinMax ──────┘
                                               ↓
                                        SOURCE_JOIN(concat)
                                               ↓
                                         SelectKBest
                                               ↓
                                            KFold
                                               ↓
                                          GBR[×5] → END
```

### Example 4: Generator Syntax with Runtime Expansion

```yaml
# Hyperparameter search via generator
pipeline:
  # Step 1: Preprocessing
  - class: sklearn.preprocessing.StandardScaler

  # Step 2: CV setup
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3

  # Step 3: Branch with generator - creates N branches at runtime
  - branch:
      _or_:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.MSC
        - class: nirs4all.operators.transforms.Detrend
        - class: nirs4all.operators.transforms.Gaussian
          params:
            sigma: 2
      count: 4  # Generate 4 combinations

  # Step 4: Model with range generator
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components:
          _range_: [5, 20, 5]  # [5, 10, 15, 20] - expands to 4 variants

  # Step 5: Select best
  - merge: features
```

**DAG Structure (at runtime)**:
```
START → StdScaler → ShuffleSplit → FORK(4 from _or_)
                                     ├─ SNV ─────────────┬─ PLS(n=5)[×3]
                                     │                   ├─ PLS(n=10)[×3]
                                     │                   ├─ PLS(n=15)[×3]
                                     │                   └─ PLS(n=20)[×3]
                                     ├─ MSC ─────────────┬─ ...
                                     ├─ Detrend ─────────┬─ ...
                                     └─ Gaussian(σ=2) ───┴─ ...
                                                         ↓
                                                   JOIN(features) → END
```

### Example 5: Outlier Handling + Sample Augmentation

```yaml
# Robust pipeline with augmentation
pipeline:
  # Step 1: Base scaling
  - class: sklearn.preprocessing.RobustScaler

  # Step 2: Outlier detection and exclusion
  - outlier_excluder:
      class: sklearn.ensemble.IsolationForest
      params:
        contamination: 0.05
      exclude_mode: train_only  # Exclude outliers from training, keep for test

  # Step 3: Sample augmentation (create variations)
  - sample_augmentation:
      transformers:
        - class: nirs4all.operators.augmentation.NoiseInjection
          params:
            noise_level: 0.01
        - class: nirs4all.operators.augmentation.SpectrumShift
          params:
            max_shift: 2
      count: 3  # 3 augmented copies per sample

  # Step 4: Feature augmentation (multiple preprocessings)
  - feature_augmentation:
      _or_:
        - class: nirs4all.operators.transforms.SNV
        - class: nirs4all.operators.transforms.FirstDerivative
      size: 2  # Use both, concatenate

  # Step 5: Cross-validation
  - class: sklearn.model_selection.GroupKFold
    params:
      n_splits: 5

  # Step 6: Model
  - model:
      class: sklearn.neural_network.MLPRegressor
      params:
        hidden_layer_sizes: [64, 32]
        max_iter: 500
```

**DAG Structure**:
```
START → RobustScaler → IsolationForest(exclude) → SampleAug(×3)
                                                       ↓
                                               FeatureAug(SNV+D1)
                                                       ↓
                                                  GroupKFold
                                                       ↓
                                                  MLP[×5] → END
```

---

## Summary

This specification defines a DAG-based execution model for nirs4all that:

1. **Preserves** the linear pipeline syntax users know
2. **Formalizes** branch/merge as graph fork/join operations
3. **Supports** runtime node creation for dynamic scenarios
4. **Defines** 10 primitive operations that controllers compose
5. **Maintains** compatibility with existing serialization and artifacts
6. **Enables** deterministic replay, caching, and partial re-execution

The design is intentionally **pragmatic**: it adds DAG semantics without requiring a wholesale rewrite, leveraging the existing controller pattern and artifact infrastructure.
