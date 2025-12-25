# NIRS4ALL v2.0: Architecture Overview

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 25, 2025
**Status**: Design Proposal (Revised per Critical Review)
**Document**: 1 of 5

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Design Philosophy](#design-philosophy)
3. [Three-Layer Architecture](#three-layer-architecture)
4. [Layer Interactions](#layer-interactions)
5. [Key Design Decisions](#key-design-decisions)
6. [NIRS4ALL-Specific Features Preservation](#nirs4all-specific-features-preservation)
7. [Error Handling Philosophy](#error-handling-philosophy)
8. [Reproducibility Guarantees](#reproducibility-guarantees)
9. [Feature Preservation Matrix](#feature-preservation-matrix)
10. [Technology Stack](#technology-stack)

---

## Executive Summary

NIRS4ALL v2.0 is a ground-up redesign that separates concerns into three distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                 │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Static API  │  │ sklearn Compat  │  │   CLI / Config      │  │
│  │ run/predict │  │ NIRSEstimator   │  │   YAML pipelines    │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     DAG EXECUTION ENGINE                         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ DAG Builder  │  │ Node Executor │  │  Artifact Manager    │  │
│  │ from syntax  │  │  parallel/seq │  │  cache & lineage     │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Branch/Merge │  │ Generator     │  │  Predictions Store   │  │
│  │ fork/join    │  │ expansion     │  │  ranking & query     │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  FeatureBlockStore                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ FeatureBlock│  │ FeatureBlock│  │ FeatureBlock    │   │   │
│  │  │ (NIR specs) │  │ (markers)   │  │ (predictions)   │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ SampleRegistry│  │ ViewResolver  │  │  TargetStore        │  │
│  │ Polars-backed │  │ lazy slicing  │  │  transform chain    │  │
│  └───────────────┘  └───────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Copy-on-Write Data Flow**: Blocks use CoW semantics for memory efficiency; views are lightweight references
3. **DAG-Native Execution**: All pipelines compile to a directed acyclic graph
4. **Reproducibility by Design**: Hash-based lineage for every transformation with full random state tracking
5. **Backend Agnosticism**: Data layer works with NumPy, Polars, or xarray
6. **Preserve v1 UX**: All flexible syntax options (class/instance/dict/YAML) remain supported
7. **Domain-Aware**: First-class support for NIRS-specific concepts (sample repetitions, aggregation keys, spectral preprocessing)

---

## Design Philosophy

### From Current Pain Points to Solutions

| Current Issue | Root Cause | v2.0 Solution |
|---------------|------------|---------------|
| Dataset mutations are hard to track | In-place updates to `SpectroDataset` | Immutable blocks + append-only store |
| Branching/merging is complex | Context copying, snapshot management | DAG with explicit fork/join nodes |
| Cross-validation leaks are possible | Manual partition management | Views enforce partition boundaries |
| Caching is ad-hoc | No lineage tracking | Hash-based block identification |
| sklearn integration is awkward | Different execution model | `NIRSEstimator` wraps DAG execution |
| Multi-source handling is fragile | Coupled source management | Independent blocks + alignment views |

### Guiding Constraints

1. **No Backward Compatibility Required**: This is a clean rewrite
2. **Preserve All Features**: Every current capability must exist in v2.0
3. **Preserve UX Flexibility**: Multiple syntax options (class, instance, dict, YAML, JSON) continue to work
4. **Simplify User Experience**: Fewer concepts, clearer mental model
5. **Enable Parallelism**: Design must support future parallel execution
6. **Support Experimentation**: REPL-friendly, incremental builds
7. **Fail-Fast with Clear Errors**: Explicit error messages with actionable guidance

### Abstraction Justifications

Some components may appear over-engineered for simple use cases. This section justifies why each abstraction exists and when simpler alternatives could be used.

#### VirtualModel: Why Not Just Return One Model?

**The Problem**: After cross-validation, should we return:
- The best single fold model?
- A retrained model on all data?
- All fold models combined?

**Why VirtualModel is Necessary**:

| Use Case | Simple Model | VirtualModel |
|----------|-------------|--------------|
| Single prediction | ✅ Works | ✅ Works |
| Uncertainty estimation | ❌ No variance info | ✅ Fold disagreement = uncertainty |
| Stacking/Ensembles | ❌ Single model only | ✅ Aggregates multiple models |
| Model interpretation | ❌ Which fold's model? | ✅ Aggregate or per-fold SHAP |
| Production deployment | ⚠️ Which model? | ✅ Clear serialization with weights |

**Simplification for Basic Users**:
```python
# Users who don't need VirtualModel complexity:
result = run(pipeline, data)
prediction = result.predict(X_new)  # VirtualModel is hidden

# Users who need it:
virtual_model = result.load_model()
predictions_with_uncertainty = virtual_model.predict_with_std(X_new)
```

#### TargetStore: Why Bidirectional Transform Chains?

**The Problem**: When we scale targets (e.g., MinMaxScaler on y), predictions come out in scaled space. We need to inverse-transform to original units.

**Why TargetStore is Necessary**:

| Scenario | Without TargetStore | With TargetStore |
|----------|---------------------|------------------|
| Scaled y, unscaled predictions | Manual inverse transform | ✅ Automatic |
| Multiple y transforms | Track all manually | ✅ Chain inversion |
| Classification with encoding | Decode labels manually | ✅ Automatic decoding |
| Multi-target with different scales | Very complex | ✅ Per-target chains |

**Real Example**:
```python
# Pipeline with y scaling
pipeline = [
    {"y_processing": MinMaxScaler()},  # Scale y to [0, 1]
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

result = run(pipeline, data)

# Without TargetStore: User must track scaler and manually inverse
# With TargetStore: Predictions automatically returned in original scale
print(result.y_pred)  # Already in original units!
```

**Simplification**: For 80% of users who don't transform y, TargetStore is transparent - it simply passes through unchanged.

#### When to Skip These Abstractions

For truly simple cases, direct sklearn usage remains valid:

```python
# If you don't need branching, stacking, or multi-source:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

simple_pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pls', PLSRegression(n_components=10))
])
simple_pipe.fit(X, y)  # Just use sklearn directly
```

The v2.0 architecture is designed for the 20% of use cases that require:
- Parallel preprocessing comparisons
- Stacking/ensemble methods
- Multi-source data fusion
- Complex branching with merging
- Reproducibility tracking
- Explainability across ensemble components

For simple linear pipelines, the overhead is minimal (VirtualModel wraps a single model, TargetStore has an empty chain).

---

## Three-Layer Architecture

### Layer 1: Data Layer (Document 2)

**Purpose**: Unified data storage and retrieval with lazy evaluation

```
┌─────────────────────────────────────────────────────────────────┐
│                     FeatureBlockStore                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Block Registry                                            │   │
│  │   block_id → (array, metadata, lineage_hash)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐   │
│  │ ViewResolver                                              │   │
│  │   view_spec → lazy slice (block_ids, row_ids, col_slice)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐   │
│  │ DatasetContext                                            │   │
│  │   sample_registry + targets + metadata + fold_indices     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components**:
- `FeatureBlock`: Copy-on-Write 3D array (samples × processings × features) + metadata
- `FeatureBlockStore`: Registry of all blocks with lineage tracking and garbage collection
- `SampleRegistry`: Polars DataFrame tracking sample identity, partition, groups, aggregation keys, repetitions
- `ViewSpec`: Declarative specification of data subset (partition, fold, branch, aggregation)
- `DatasetContext`: Complete training/prediction context bundle

### Layer 2: DAG Execution Engine (Document 3)

**Purpose**: Compile pipeline syntax to DAG, execute with parallelism

```
┌─────────────────────────────────────────────────────────────────┐
│                      DAG Engine                                  │
│                                                                  │
│  [Pipeline Syntax] ──► [DAG Builder] ──► [Executable DAG]       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Node Types                                                │   │
│  │   • TransformNode     (fit/transform operators)           │   │
│  │   • ModelNode         (fit/predict with folds)            │   │
│  │   • ForkNode          (branch/source_branch)              │   │
│  │   • JoinNode          (merge/merge_sources)               │   │
│  │   • GeneratorNode     (_or_/_range_ expansion)            │   │
│  │   • SplitterNode      (CV fold assignment)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Execution                                                 │   │
│  │   • Topological sort → execution order                    │   │
│  │   • Parallel execution within same level                  │   │
│  │   • Artifact caching via block lineage                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Concepts**:
- Pipelines are **always** compiled to DAG before execution
- Generators (`_or_`, `_range_`) expand to parallel branches at compile time with **configurable limits**
- Folds are modeled as implicit fork/join (N parallel training paths) with explicit synchronization
- All nodes produce output blocks using CoW semantics; edges carry `ViewSpec`
- Generator expansion is **limited** (default max 1000 variants) with warnings at >100

### Layer 3: API Layer (Document 4)

**Purpose**: User-friendly interfaces for training, prediction, explanation

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│                                                                  │
│  ┌────────────────────┐                                         │
│  │ Static Functions   │  nirs4all.run(pipeline, data)           │
│  │                    │  nirs4all.predict(model, data)          │
│  │                    │  nirs4all.explain(model, data)          │
│  └────────────────────┘                                         │
│                                                                  │
│  ┌────────────────────┐                                         │
│  │ sklearn Estimators │  NIRSRegressor(pipeline).fit(X, y)      │
│  │                    │  NIRSClassifier(pipeline).predict(X)    │
│  │                    │  NIRSSearchCV(pipelines).fit(X, y)      │
│  └────────────────────┘                                         │
│                                                                  │
│  ┌────────────────────┐                                         │
│  │ Result Objects     │  RunResult.best, .top(n), .export()     │
│  │                    │  PredictionResult with partition data   │
│  │                    │  ExplanationResult with SHAP values     │
│  └────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

**Key Interfaces**:
- `run()`: Train pipeline, return ranked results
- `predict()`: Apply trained model to new data
- `explain()`: Generate SHAP explanations
- `NIRSEstimator`: sklearn-compatible wrapper for nirs4all pipelines
- `NIRSSearchCV`: Grid search over pipeline configurations

---

## Layer Interactions

### Training Flow

```
User Code                 API Layer              DAG Engine            Data Layer
    │                         │                      │                     │
    │ run(pipeline, data)     │                      │                     │
    ├────────────────────────►│                      │                     │
    │                         │ create DatasetContext│                     │
    │                         ├─────────────────────────────────────────────►
    │                         │                      │ register source blocks
    │                         │                      │◄────────────────────┤
    │                         │ build_dag(pipeline)  │                     │
    │                         ├─────────────────────►│                     │
    │                         │ expanded DAG         │                     │
    │                         │◄─────────────────────┤                     │
    │                         │ execute(dag, context)│                     │
    │                         ├─────────────────────►│                     │
    │                         │                      │ for each node:      │
    │                         │                      │   resolve_view()    │
    │                         │                      ├────────────────────►│
    │                         │                      │   X, y data         │
    │                         │                      │◄────────────────────┤
    │                         │                      │   execute node      │
    │                         │                      │   register output   │
    │                         │                      ├────────────────────►│
    │                         │ RunResult            │                     │
    │◄────────────────────────┤                      │                     │
```

### Prediction Flow

```
User Code                 API Layer              DAG Engine            Data Layer
    │                         │                      │                     │
    │ predict(model, data)    │                      │                     │
    ├────────────────────────►│                      │                     │
    │                         │ load_artifacts(model)│                     │
    │                         ├─────────────────────────────────────────────►
    │                         │ create DatasetContext(new_data, artifacts) │
    │                         │                      │                     │
    │                         │ build_minimal_dag()  │                     │
    │                         ├─────────────────────►│                     │
    │                         │ execute(dag, mode="predict")              │
    │                         ├─────────────────────►│                     │
    │                         │                      │ apply transforms    │
    │                         │                      │ aggregate fold preds│
    │                         │ PredictionResult     │                     │
    │◄────────────────────────┤                      │                     │
```

---

## Key Design Decisions

### Decision 1: Immutable Blocks with Append-Only Store

**Rationale**: Current `SpectroDataset` mutates in-place, making it hard to:
- Track what transformations have been applied
- Roll back to previous states
- Cache intermediate results
- Enable parallel execution

**v2.0 Approach**:
```python
# Each transformation creates a NEW block
raw_block = store.register(X_raw, metadata={"source": "NIR"})
scaled_block = store.register(X_scaled, parent=raw_block,
                              transform="MinMaxScaler")

# Blocks are never modified; lineage is explicit
assert scaled_block.lineage_hash != raw_block.lineage_hash
```

### Decision 2: Views as Lazy References

**Rationale**: Copying data for each partition/fold is wasteful

**v2.0 Approach**:
```python
# ViewSpec is a lightweight object describing a slice
view = ViewSpec(
    block_ids=["block_001"],
    partition="train",
    fold_id=0,
    exclude_samples=[45, 67]  # outliers
)

# Data is only materialized when needed
X, y = context.materialize(view)
```

### Decision 3: DAG Compilation Before Execution

**Rationale**: Sequential execution misses parallelization opportunities and makes branching complex

**v2.0 Approach**:
```python
# Pipeline syntax compiles to DAG
dag = DAGBuilder().build([
    MinMaxScaler(),
    {"_or_": [SNV(), MSC()]},  # Creates fork node
    PLSRegression()
])

# DAG structure:
#   scale → fork → [snv_path, msc_path] → join → pls
```

### Decision 4: Folds as Implicit Fork/Join

**Rationale**: Cross-validation is just a special case of branching

**v2.0 Approach**:
```python
# SplitterNode creates N implicit branches (one per fold)
# ModelNode executes on each fold independently
# Final output is a "virtual model" (aggregated predictions)

class VirtualModel:
    fold_models: List[FittedModel]
    aggregation: str = "weighted_mean"

    def predict(self, X):
        preds = [m.predict(X) for m in self.fold_models]
        return aggregate(preds, self.aggregation)
```

### Decision 5: sklearn Estimator as DAG Wrapper

**Rationale**: Users want sklearn compatibility for `GridSearchCV`, SHAP, etc.

**v2.0 Approach**:
```python
class NIRSEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline, cv=5):
        self.pipeline = pipeline
        self.cv = cv

    def fit(self, X, y):
        # Build DAG and execute in training mode
        self.dag_ = DAGBuilder().build(self.pipeline)
        self.result_ = execute_dag(self.dag_, X, y, cv=self.cv)
        self.best_model_ = self.result_.best
        return self

    def predict(self, X):
        return self.best_model_.predict(X)
```

---

## NIRS4ALL-Specific Features Preservation

### Sample Repetitions and Aggregation Keys

NIRS data often contains multiple measurements per biological sample (repetitions). The v2.0 design preserves this domain-specific handling:

```python
# SampleRegistry tracks aggregation relationships
SAMPLE_REGISTRY_SCHEMA = {
    "_sample_id": pl.Int64,       # Unique measurement ID (obs_id)
    "_bio_id": pl.Int64,          # Biological sample ID (for aggregation)
    "repetition_idx": pl.Int32,   # Repetition number within bio_id
    "partition": pl.Utf8,
    "group": pl.Utf8,
    # ... other fields
}

# Aggregation support at prediction time
class AggregationSpec:
    """Specification for sample-level aggregation."""
    by: str                  # Column to group by (e.g., "_bio_id")
    method: str = "mean"     # "mean", "median", "vote" (classification)
    exclude_outliers: bool = False
    outlier_threshold: float = 0.95
```

### Repetition Transformation

Convert repetitions to sources or processings:

```python
# Current v1 syntax preserved
pipeline = [
    {"rep_to_sources": "Sample_ID"},      # Reps → separate sources
    {"rep_to_processings": {"column": "Sample_ID", "expected_reps": 3}},
]

# Becomes RepetitionTransformNode in DAG
class RepetitionTransformNode(DAGNode):
    column: str
    mode: Literal["sources", "processings"]
    expected_reps: Optional[int]
    unequal_strategy: Literal["error", "drop", "pad", "mask"]
```

### Flexible Pipeline Syntax

All v1 syntax options remain supported and normalize to canonical form:

```python
# All of these are equivalent and valid:
MinMaxScaler                                    # Class reference
MinMaxScaler()                                  # Instance
{"preprocessing": MinMaxScaler()}               # Dict with keyword
{"class": "sklearn.preprocessing.MinMaxScaler"} # Explicit class path
"sklearn.preprocessing.MinMaxScaler"            # String path

# Step parsing normalizes all to ParsedStep
@dataclass
class ParsedStep:
    operator: Any                    # Instantiated operator
    keyword: str                     # Resolved keyword
    operator_class: str              # Fully qualified class name
    params: Dict[str, Any]           # Constructor parameters
    original_syntax: Any             # Preserved for serialization
```

### Operator Registry Pattern

Controllers use priority-based registry for seamless integration:

```python
# Any sklearn TransformerMixin works automatically
from sklearn.preprocessing import StandardScaler
pipeline = [StandardScaler(), PLSRegression()]

# Custom transformers just need TransformerMixin
class MyTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return X * 2

# Automatic controller dispatch based on operator type
CONTROLLER_REGISTRY = [
    (TransformController, priority=100),     # Catches TransformerMixin
    (ModelController, priority=100),         # Catches predictors
    (TFModelController, priority=50),        # Higher priority for TF
    # ... specialized controllers have lower priority (higher precedence)
]
```

---

## Error Handling Philosophy

### Error Strategy: Fail-Fast with Rich Context

```python
class NIRSError(Exception):
    """Base exception for all nirs4all errors."""
    pass

class PipelineValidationError(NIRSError):
    """Invalid pipeline configuration."""
    def __init__(self, message: str, step_idx: int, step: Any, suggestions: List[str] = None):
        self.step_idx = step_idx
        self.step = step
        self.suggestions = suggestions or []
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        msg = f"Step {self.step_idx}: {message}\n"
        msg += f"  Step definition: {self.step!r}\n"
        if self.suggestions:
            msg += "  Suggestions:\n"
            for s in self.suggestions:
                msg += f"    - {s}\n"
        return msg

class DAGExecutionError(NIRSError):
    """Error during DAG execution."""
    def __init__(self, message: str, node_id: str, branch_id: Optional[int] = None, cause: Optional[Exception] = None):
        self.node_id = node_id
        self.branch_id = branch_id
        self.cause = cause
        super().__init__(message)

class DataValidationError(NIRSError):
    """Invalid data provided."""
    def __init__(self, message: str, field: str, expected: str, got: str):
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(f"{field}: {message} (expected {expected}, got {got})")

class GeneratorExplosionWarning(UserWarning):
    """Warning when generator expansion exceeds threshold."""
    pass
```

### Error Handling in DAG Execution

```python
class ExecutionEngine:
    def execute(self, dag, context, mode="train", fail_fast=True):
        errors = []

        for node_id in dag.topological_order():
            try:
                result = self._execute_node(node_id, ...)
            except Exception as e:
                error = DAGExecutionError(
                    message=str(e),
                    node_id=node_id,
                    branch_id=context.metadata.get("branch_id"),
                    cause=e
                )
                if fail_fast:
                    raise error from e
                errors.append(error)

        if errors:
            raise DAGExecutionError(
                f"Pipeline failed with {len(errors)} errors",
                node_id=errors[0].node_id,
                cause=errors[0]
            )
```

### Parallel Branch Error Handling

```python
class BranchExecutionError(DAGExecutionError):
    """One or more branches failed."""
    def __init__(self, branch_errors: Dict[int, Exception]):
        self.branch_errors = branch_errors
        failed = list(branch_errors.keys())
        super().__init__(
            f"Branches {failed} failed",
            node_id="fork",
            branch_id=failed[0]
        )

# Fork/Join with partial failure handling
class ForkNode:
    def execute(self, ..., continue_on_error: bool = False):
        results = {}
        errors = {}

        for branch_id, branch_def in enumerate(self.branches):
            try:
                results[branch_id] = self._execute_branch(branch_id, ...)
            except Exception as e:
                if not continue_on_error:
                    raise
                errors[branch_id] = e

        if errors:
            raise BranchExecutionError(errors)
        return results
```

---

## Reproducibility Guarantees

### Random State Propagation

```python
@dataclass
class ReproducibilityContext:
    """Tracks all sources of randomness for reproducibility.

    This context ensures deterministic execution by:
    1. Deriving per-node seeds from global seed + node_id
    2. Setting RNG state for ALL relevant libraries before each node executes
    3. Recording library versions for reproducibility reports
    """
    global_seed: int
    node_seeds: Dict[str, int]  # Deterministic per-node seeds
    library_versions: Dict[str, str]  # numpy, sklearn, torch, tensorflow, etc.
    platform_info: Dict[str, str]  # OS, Python version

    @classmethod
    def create(cls, seed: int) -> "ReproducibilityContext":
        import numpy as np
        import sklearn
        import platform

        versions = {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "nirs4all": nirs4all.__version__,
        }

        # Optionally capture deep learning framework versions
        try:
            import torch
            versions["torch"] = torch.__version__
        except ImportError:
            pass

        try:
            import tensorflow as tf
            versions["tensorflow"] = tf.__version__
        except ImportError:
            pass

        try:
            import jax
            versions["jax"] = jax.__version__
        except ImportError:
            pass

        return cls(
            global_seed=seed,
            node_seeds={},  # Populated during DAG build
            library_versions=versions,
            platform_info={
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
        )

    def get_node_seed(self, node_id: str) -> int:
        """Deterministic seed for a specific node."""
        if node_id not in self.node_seeds:
            # Hash-based deterministic seed derivation
            import hashlib
            h = hashlib.sha256(f"{self.global_seed}:{node_id}".encode())
            self.node_seeds[node_id] = int(h.hexdigest()[:8], 16)
        return self.node_seeds[node_id]

    def apply_node_seed(self, node_id: str) -> None:
        """Apply deterministic seed to ALL relevant RNG sources before node execution.

        This method is called by the ExecutionEngine before each node executes.
        It sets the random state for:
        - Python's random module
        - NumPy's random generator
        - PyTorch (if available)
        - TensorFlow (if available)
        - JAX (note: JAX uses explicit key passing, so we provide the key)

        Args:
            node_id: The node about to execute
        """
        import random
        import numpy as np

        seed = self.get_node_seed(node_id)

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch (if available)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        # TensorFlow (if available)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        # JAX note: JAX doesn't use global state; nodes using JAX should
        # call context.get_jax_key(node_id) to get a deterministic PRNGKey

    def get_jax_key(self, node_id: str) -> "jax.random.PRNGKey":
        """Get a deterministic JAX PRNGKey for a node.

        JAX uses explicit random key passing rather than global state.
        Nodes that use JAX should call this method to get their key.

        Args:
            node_id: The node requesting a key

        Returns:
            JAX PRNGKey derived from node seed
        """
        import jax
        seed = self.get_node_seed(node_id)
        return jax.random.PRNGKey(seed)
```

### Execution Order Guarantees

```python
class ExecutableDAG:
    def topological_order(self, deterministic: bool = True) -> List[str]:
        """Return nodes in reproducible execution order.

        Args:
            deterministic: If True, nodes at same level are sorted by ID
                          to ensure consistent ordering across runs.
        """
        from collections import deque

        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1

        # Sort initial queue for determinism
        ready = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        queue = deque(ready)
        order = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)

            # Collect and sort successors for determinism
            successors = sorted(self.get_successors(node_id))
            for succ in successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    # Insert in sorted position
                    for i, q_id in enumerate(queue):
                        if succ < q_id:
                            queue.insert(i, succ)
                            break
                    else:
                        queue.append(succ)

        return order
```

### Result Reproducibility Report

```python
@dataclass
class RunResult:
    # ... existing fields ...
    reproducibility: ReproducibilityContext

    def reproducibility_report(self) -> Dict[str, Any]:
        """Generate report on reproducibility factors."""
        return {
            "seed": self.reproducibility.global_seed,
            "node_seeds": self.reproducibility.node_seeds,
            "versions": self.reproducibility.library_versions,
            "platform": self.reproducibility.platform_info,
            "dag_hash": self.dag.content_hash(),
            "execution_order": self.dag.topological_order(),
        }
```

---

## Feature Preservation Matrix

Every feature from v1.x must exist in v2.0:

| v1.x Feature | v2.0 Location | Notes |
|--------------|---------------|-------|
| Multi-source datasets | `FeatureBlockStore` with multiple blocks | Block alignment via `SampleRegistry` |
| Processing chain tracking | Block `lineage_hash` + `processing_ids` | Automatic, not manual |
| Cross-validation | `SplitterNode` → implicit fork/join | `VirtualModel` for aggregation with sync protocol |
| Branching (`branch:`) | `ForkNode` + `JoinNode` | Explicit DAG edges with barrier sync |
| Source branching | `SourceForkNode` | Per-source parallel paths |
| Merging (`merge:`) | `JoinNode` with strategy | `features` or `predictions` mode |
| Generators (`_or_`, `_range_`) | `GeneratorExpansion` at compile | Pre-execution with limits (max_variants=1000) |
| Feature augmentation | `AugmentNode` | Creates new blocks with CoW |
| Sample augmentation | `SampleAugmentNode` | Adds rows, tracks `origin` |
| Y processing | `TargetTransformNode` | Transform chain with inverse support |
| Predictions storage | `PredictionStore` (Polars-backed) | Same as v1.x, minor API changes |
| Prediction ranking | `PredictionRanker` | Unchanged API |
| Artifact persistence | `ArtifactManager` | Block-based, hash-indexed, GC-enabled |
| Bundle export | `BundleExporter` | Exports DAG + blocks + reproducibility |
| SHAP explanations | `explain()` + `ExplainerBridge` | Pluggable explainer backends |
| **Aggregation** | `AggregationView` + `SampleRegistry._bio_id` | Sample-level prediction aggregation |
| **Repetition handling** | `RepetitionTransformNode` | Reps → sources or processings with policy |
| Outlier exclusion | `OutlierExcluderNode` | Marks samples in registry |
| Metadata filtering | `ViewSpec.filter` | Declarative sample selection |
| **Flexible syntax** | `StepParser` with normalization | Class/instance/dict/YAML/JSON all supported |
| **Operator registry** | `ControllerRegistry` priority dispatch | Seamless sklearn/TF/Torch/JAX integration |

---

## Technology Stack

### Core Dependencies

| Component | Library | Rationale |
|-----------|---------|-----------|
| DataFrames | **Polars** | Fast, lazy evaluation, no pandas overhead |
| Arrays | **NumPy** | Standard, universal compatibility |
| Optional 3D | **xarray** | Named dimensions for multi-source |
| ML Base | **scikit-learn** | Transformer/Estimator patterns |
| Deep Learning | **PyTorch** (primary), TF, JAX | Flexible, good SHAP support |
| Optimization | **Optuna** | Hyperparameter search |
| Explanations | **SHAP** | Model-agnostic + deep explainers |
| Serialization | **joblib** + **Parquet** | Efficient binary + columnar |
| Hashing | **hashlib** (MD5/SHA256) | Deterministic lineage |
| Parallelism | **concurrent.futures** | Thread/process pools |

### Optional Dependencies

| Feature | Library | Install Extra |
|---------|---------|---------------|
| xarray backend | xarray | `nirs4all[xarray]` |
| TensorFlow models | tensorflow | `nirs4all[tf]` |
| JAX models | jax, flax | `nirs4all[jax]` |
| Ray parallelism | ray | `nirs4all[ray]` |
| MLflow tracking | mlflow | `nirs4all[mlflow]` |

---

## Next Documents

1. **Document 2: Data Layer** - `FeatureBlockStore`, views, aggregations
2. **Document 3: DAG Engine** - Node types, execution, branching
3. **Document 4: API Layer** - Static API, sklearn estimators
4. **Document 5: Implementation Plan** - Phases, testing, migration

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Block** | Copy-on-Write 3D array + metadata + lineage hash |
| **View** | Lazy reference to a subset of data |
| **ViewSpec** | Declarative specification of a view |
| **DAG** | Directed Acyclic Graph of pipeline operations |
| **Node** | Single operation in the DAG |
| **Edge** | Data flow between nodes (carries ViewSpec) |
| **Fork** | Node that creates multiple parallel paths with barrier sync |
| **Join** | Node that merges parallel paths |
| **Lineage** | Hash-based tracking of block transformations |
| **VirtualModel** | Aggregated prediction from fold models with weighted aggregation |
| **DatasetContext** | Complete bundle of data + artifacts for execution |
| **bio_id** | Biological sample ID (for aggregation across repetitions) |
| **obs_id** | Observation/measurement ID (unique per measurement) |
| **Aggregation Key** | Column used to group repetitions for prediction aggregation |
| **CoW** | Copy-on-Write semantics for memory-efficient block operations |
| **ParsedStep** | Normalized step representation from any syntax |
| **ReproducibilityContext** | Complete snapshot of random state and versions |
