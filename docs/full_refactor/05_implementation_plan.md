# NIRS4ALL v2.0: Implementation Plan

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 25, 2025
**Status**: Design Proposal (Revised per Critical Review)
**Document**: 5 of 5

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 0: Interface Contracts](#phase-0-interface-contracts)
3. [Development Phases](#development-phases)
4. [Package Structure](#package-structure)
5. [Dependencies](#dependencies)
6. [Testing Strategy](#testing-strategy)
7. [Edge Case Testing](#edge-case-testing)
8. [Migration from v1.x](#migration-from-v1x)
9. [Migration Testing](#migration-testing)
10. [Risks and Mitigations](#risks-and-mitigations)
11. [Timeline](#timeline)

---

## Overview

### Implementation Principles

1. **Interface-First**: Define Protocol classes before implementation
2. **Bottom-Up Build**: Data layer → DAG engine → API layer
3. **Test-First Development**: Core abstractions tested before integration
4. **Incremental Rollout**: Features added incrementally, not big-bang
5. **Parallel Development**: Phases 1 and 2 can progress in parallel using Protocol contracts

### Success Criteria

- [ ] All existing examples (Q1-Q36) work on new architecture
- [ ] Benchmark performance equal or better than v1.x
- [ ] sklearn integration passes SHAP, GridSearchCV compatibility tests
- [ ] Bundle export/import preserves full reproducibility
- [ ] **Numerical equivalence within tolerance (r² ± 0.001)**
- [ ] **All edge case tests passing**

---

## Phase 0: Interface Contracts

**Goal**: Define Protocol interfaces that enable parallel development of Phases 1 and 2.

This phase addresses the critical review finding that "Phases 1 and 2 are parallelizable but have implicit dependencies."

### Core Protocols

```python
# nirs4all_v2/protocols.py
"""Interface contracts defined before implementation.

These Protocol classes define the interfaces between components,
enabling parallel development of the data layer and DAG engine.
"""

from typing import Protocol, Optional, List, Dict, Any, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for accessing feature data.

    Implemented by: DatasetContext
    Used by: ExecutionEngine, NodeRunner
    """

    def x(
        self,
        *,
        layout: str = "2d",
        source: Optional[str] = None,
        branch_id: Optional[int] = None
    ) -> np.ndarray:
        """Get feature matrix.

        Args:
            layout: "2d" for (samples, features), "3d" for multi-source
            source: Specific source to retrieve
            branch_id: Branch context for disambiguation

        Returns:
            Feature array
        """
        ...

    def y(
        self,
        *,
        original: bool = False
    ) -> np.ndarray:
        """Get target values.

        Args:
            original: If True, return original (untransformed) targets

        Returns:
            Target array
        """
        ...

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        ...

    @property
    def n_features(self) -> int:
        """Number of features (for primary source)."""
        ...

    @property
    def sample_ids(self) -> List[str]:
        """Unique sample identifiers."""
        ...


@runtime_checkable
class ViewProvider(Protocol):
    """Protocol for creating data views.

    Implemented by: ViewResolver
    Used by: DAG nodes for subsetting data
    """

    def materialize(
        self,
        spec: "ViewSpec"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Materialize a view specification to arrays.

        Args:
            spec: View specification defining samples, features, sources

        Returns:
            Tuple of (X, y) arrays
        """
        ...


@runtime_checkable
class BlockStore(Protocol):
    """Protocol for feature block storage.

    Implemented by: FeatureBlockStore
    Used by: DatasetContext, ViewResolver
    """

    def add(
        self,
        block: "FeatureBlock",
        parents: List[str]
    ) -> str:
        """Add block to store.

        Args:
            block: The feature block
            parents: List of parent block hashes

        Returns:
            Block hash (content-addressable ID)
        """
        ...

    def get(self, hash: str) -> "FeatureBlock":
        """Retrieve block by hash."""
        ...

    def get_lineage(self, hash: str) -> List[str]:
        """Get ancestry chain of block."""
        ...


@runtime_checkable
class ExecutableNode(Protocol):
    """Protocol for DAG nodes.

    Implemented by: TransformNode, ModelNode, ForkNode, etc.
    Used by: ExecutionEngine
    """

    @property
    def node_id(self) -> str:
        """Unique node identifier."""
        ...

    @property
    def node_type(self) -> str:
        """Node type string."""
        ...

    def execute(
        self,
        context: DataProvider,
        input_view: "ViewSpec",
        runtime_context: Dict[str, Any]
    ) -> "NodeResult":
        """Execute node.

        Args:
            context: Data provider
            input_view: Input data specification
            runtime_context: Runtime state (artifacts, fold info, etc.)

        Returns:
            NodeResult with outputs and artifacts
        """
        ...

    def supports_prediction_mode(self) -> bool:
        """Whether this node runs during prediction."""
        ...


@runtime_checkable
class PredictionAggregator(Protocol):
    """Protocol for aggregating fold predictions.

    Implemented by: VirtualModel
    Used by: JoinNode, predict()
    """

    def aggregate(
        self,
        fold_predictions: List[np.ndarray],
        fold_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Aggregate predictions from multiple folds.

        Args:
            fold_predictions: List of prediction arrays
            fold_weights: Optional weights for each fold

        Returns:
            Aggregated predictions
        """
        ...


@runtime_checkable
class SampleIndex(Protocol):
    """Protocol for sample indexing and splitting.

    Implemented by: SampleRegistry
    Used by: SplitterNode, FoldAwareMixin
    """

    @property
    def indices(self) -> np.ndarray:
        """All sample indices."""
        ...

    def split(
        self,
        fractions: List[float],
        stratify: Optional[np.ndarray] = None
    ) -> List["SampleIndex"]:
        """Split into subsets."""
        ...

    def get_bio_id(self, sample_idx: int) -> Optional[str]:
        """Get biological sample ID for a measurement."""
        ...


# ===== Result Types =====

@runtime_checkable
class NodeResult(Protocol):
    """Result from node execution."""

    @property
    def output_view(self) -> "ViewSpec":
        """View specification for downstream nodes."""
        ...

    @property
    def artifacts(self) -> Dict[str, Any]:
        """Artifacts produced (models, fitted transformers, etc.)."""
        ...

    @property
    def predictions(self) -> Optional[np.ndarray]:
        """Predictions if this is a model node."""
        ...
```

### Protocol Usage Pattern

```python
# Phase 1: Data Layer implementation
class DatasetContext:
    """Implements DataProvider protocol."""

    def x(self, *, layout="2d", source=None, branch_id=None) -> np.ndarray:
        # Implementation
        ...

# Phase 2: DAG Engine implementation (can proceed in parallel)
class ExecutionEngine:
    """Uses DataProvider protocol - doesn't depend on concrete DatasetContext."""

    def __init__(self):
        pass

    def execute(
        self,
        dag: ExecutableDAG,
        context: DataProvider,  # ← Protocol, not concrete class
        mode: str = "train"
    ) -> ExecutionResult:
        # Implementation against protocol
        for node in dag.topological_order():
            result = node.execute(context, ...)
```

### Phase 0 Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| Core Protocols | All Protocol classes | `protocols.py` |
| Type Stubs | Type hints for Protocol usage | `_types.pyi` |
| Mock Implementations | For testing Phase 2 without Phase 1 | `mocks.py` |
| Protocol Tests | Verify implementations satisfy protocols | `test_protocols.py` |

### Protocol Test Pattern

```python
# tests/test_protocols.py
from typing import get_type_hints
from nirs4all.protocols import DataProvider, BlockStore

def test_dataset_context_satisfies_protocol():
    """DatasetContext must implement DataProvider."""
    from nirs4all.data import DatasetContext

    # Check that it's an instance
    ctx = DatasetContext(...)
    assert isinstance(ctx, DataProvider)

    # Check method signatures match
    for method_name in ['x', 'y', 'n_samples', 'n_features']:
        assert hasattr(ctx, method_name)

def test_feature_block_store_satisfies_protocol():
    """FeatureBlockStore must implement BlockStore."""
    from nirs4all.data import FeatureBlockStore

    store = FeatureBlockStore()
    assert isinstance(store, BlockStore)
```

---

## Development Phases

### Phase 0: Foundation (Weeks 1-2)

**Goal**: Core abstractions with full test coverage

#### 0.1 FeatureBlock Core

```
nirs4all_v2/
└── data/
    ├── __init__.py
    ├── block.py           # FeatureBlock, FeatureBlockDescriptor
    ├── registry.py        # SampleRegistry
    └── tests/
        └── test_block.py
```

**Deliverables**:
- `FeatureBlock` with immutability guarantees
- `FeatureBlockDescriptor` for metadata
- Content hash computation
- Serialization to Parquet/NPY

**Tests**:
```python
def test_block_immutability():
    block = FeatureBlock(data=np.array([[1, 2], [3, 4]]))
    with pytest.raises(ImmutableError):
        block.data[0, 0] = 999

def test_block_hash_deterministic():
    data = np.array([[1, 2], [3, 4]])
    block1 = FeatureBlock(data=data)
    block2 = FeatureBlock(data=data)
    assert block1.hash == block2.hash
```

#### 0.2 SampleRegistry

**Deliverables**:
- Unique sample ID generation
- Lineage tracking
- Merge/split operations

**Tests**:
```python
def test_registry_split():
    reg = SampleRegistry(n_samples=100)
    train, val = reg.split([0.8, 0.2])
    assert len(train) == 80
    assert len(val) == 20
    assert train.lineage.parent == reg
```

#### 0.3 TargetStore

**Deliverables**:
- Target storage (Polars)
- Multi-target support
- Transform tracking

---

### Phase 1: Data Layer (Weeks 3-5)

**Goal**: Complete data layer with views and context

#### 1.1 FeatureBlockStore

```
nirs4all_v2/
└── data/
    ├── store.py           # FeatureBlockStore
    ├── views.py           # ViewSpec, ViewResolver
    └── context.py         # DatasetContext
```

**Deliverables**:
- Append-only block store
- Block lookup by hash
- Lineage graph

**Tests**:
```python
def test_store_lineage():
    store = FeatureBlockStore()
    original = FeatureBlock(data=X)
    store.add(original, parents=[])

    derived = FeatureBlock(data=X_scaled)
    store.add(derived, parents=[original.hash])

    assert original.hash in store.get_lineage(derived)
```

#### 1.2 ViewSpec and ViewResolver

**Deliverables**:
- Lazy view specification
- View materialization
- Cache invalidation

**Tests**:
```python
def test_view_composition():
    spec = (
        ViewSpec()
        .select_samples([0, 1, 2])
        .select_features([10, 20, 30])
        .select_source("nir")
    )
    X = spec.materialize(store)
    assert X.shape == (3, 3)
```

#### 1.3 DatasetContext

**Deliverables**:
- Unified dataset access
- Multi-source support
- Metadata access

---

### Phase 2: DAG Engine (Weeks 4-7)

**Goal**: DAG compilation and execution (can start in parallel with Phase 1)

#### 2.1 DAG Model

```
nirs4all_v2/
└── dag/
    ├── __init__.py
    ├── model.py           # DAGNode, DAGEdge, ExecutableDAG
    ├── nodes/
    │   ├── transform.py   # TransformNode
    │   ├── model.py       # ModelNode
    │   ├── fork.py        # ForkNode
    │   ├── join.py        # JoinNode
    │   └── splitter.py    # SplitterNode
    └── tests/
```

**Deliverables**:
- DAG node types
- Edge definitions
- Topological sort

**Tests**:
```python
def test_dag_topological_order():
    dag = ExecutableDAG()
    n1 = dag.add_node(TransformNode())
    n2 = dag.add_node(TransformNode())
    n3 = dag.add_node(ModelNode())
    dag.add_edge(n1, n2)
    dag.add_edge(n2, n3)

    order = dag.topological_order()
    assert order.index(n1) < order.index(n2) < order.index(n3)
```

#### 2.2 DAG Builder

```
nirs4all_v2/
└── dag/
    └── builder.py         # DAGBuilder, generator expansion
```

**Deliverables**:
- Pipeline → DAG conversion
- Generator expansion (`_or_`, `_range_`)
- Branch/merge handling

**Tests**:
```python
def test_generator_expansion():
    pipeline = [
        {"_or_": [SNV(), MSC()]},
        PLSRegression()
    ]
    builder = DAGBuilder()
    dags = builder.build(pipeline)
    assert len(dags) == 2  # Two variants
```

#### 2.3 Execution Engine

```
nirs4all_v2/
└── dag/
    ├── engine.py          # ExecutionEngine
    └── runner.py          # NodeRunner
```

**Deliverables**:
- Sequential execution
- Parallel execution (thread pool)
- Progress tracking

---

### Phase 3: Model Management (Weeks 6-8)

**Goal**: VirtualModel and artifacts

#### 3.1 VirtualModel

```
nirs4all_v2/
└── models/
    ├── __init__.py
    ├── virtual.py         # VirtualModel
    └── aggregation.py     # Fold aggregation strategies
```

**Deliverables**:
- Multi-fold model wrapper
- Prediction aggregation
- sklearn interface

**Tests**:
```python
def test_virtual_model_predict():
    models = [PLSRegression().fit(X, y) for _ in range(5)]
    vm = VirtualModel(models, aggregation="mean")

    y_pred = vm.predict(X_test)
    expected = np.mean([m.predict(X_test) for m in models], axis=0)
    np.testing.assert_array_almost_equal(y_pred, expected)
```

#### 3.2 ArtifactManager

```
nirs4all_v2/
└── artifacts/
    ├── __init__.py
    ├── manager.py         # ArtifactManager
    └── bundle.py          # Bundle export/import
```

**Deliverables**:
- Artifact serialization
- Hash-based lookup
- Bundle format

---

### Phase 4: API Layer (Weeks 8-10)

**Goal**: User-facing API

#### 4.1 Static API

```
nirs4all_v2/
└── api/
    ├── __init__.py
    ├── run.py             # run() function
    ├── predict.py         # predict() function
    ├── explain.py         # explain() function
    └── results.py         # Result objects
```

**Deliverables**:
- `run()` function
- `predict()` function
- `explain()` function
- Result objects

#### 4.2 sklearn Estimators

```
nirs4all_v2/
└── sklearn/
    ├── __init__.py
    ├── regressor.py       # NIRSRegressor
    ├── classifier.py      # NIRSClassifier
    └── search.py          # NIRSSearchCV
```

**Deliverables**:
- `NIRSRegressor`
- `NIRSClassifier`
- `NIRSSearchCV`

**Tests**:
```python
def test_sklearn_gridsearch_compatible():
    from sklearn.model_selection import GridSearchCV

    reg = NIRSRegressor([MinMaxScaler(), PLSRegression()])
    grid = GridSearchCV(reg, {"pls__n_components": [5, 10]}, cv=3)
    grid.fit(X, y)

    assert hasattr(grid, "best_estimator_")
    assert grid.best_estimator_.predict(X).shape == y.shape
```

---

### Phase 5: Operators (Weeks 9-11)

**Goal**: Port existing operators

#### 5.1 NIRS Transforms

```
nirs4all_v2/
└── transforms/
    ├── __init__.py
    ├── nirs.py            # SNV, MSC, Detrend, etc.
    ├── derivatives.py     # SavitzkyGolay, FirstDerivative
    └── signal.py          # Absorbance, Transmittance
```

**Porting Strategy**:
- Reuse v1 implementations where possible
- Ensure TransformerMixin compliance
- Add proper type hints

#### 5.2 Splitters

```
nirs4all_v2/
└── splitters/
    ├── __init__.py
    ├── kennard_stone.py   # KennardStoneSplit
    ├── spxy.py            # SPXYSplit
    └── group.py           # GroupKFold extensions
```

#### 5.3 Feature Selection

```
nirs4all_v2/
└── feature_selection/
    ├── __init__.py
    ├── cars.py            # CARS
    ├── mcuve.py           # MC-UVE
    └── interval.py        # Interval selection
```

---

### Phase 6: Integration (Weeks 11-13)

**Goal**: Full system integration

#### 6.1 Example Migration

- Port Q1-Q36 examples to new API
- Verify identical or improved results
- Update documentation

#### 6.2 Performance Benchmarks

```python
# bench/v2_comparison.py
def benchmark_pipeline(version, pipeline, data, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if version == "v1":
            run_v1(pipeline, data)
        else:
            run_v2(pipeline, data)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)
```

**Targets**:
- Simple pipelines: Equal or faster
- Complex pipelines with branching: 20-50% faster (due to better caching)
- Memory usage: 30% reduction (due to lazy views)

---

### Phase 7: Polish (Weeks 13-15)

**Goal**: Production readiness

#### 7.1 Documentation

- API reference (Sphinx)
- User guide updates
- Migration guide

#### 7.2 CLI

- Implement CLI commands
- Add shell completion
- Add progress bars

#### 7.3 Error Messages

- Custom exception types
- Helpful error messages with suggestions
- Debug mode with full traces

---

## Package Structure

### Final Structure

```
nirs4all_v2/
├── __init__.py            # Main exports
├── api/                   # User-facing API
│   ├── __init__.py
│   ├── run.py
│   ├── predict.py
│   ├── explain.py
│   ├── session.py
│   └── results.py
├── data/                  # Data layer
│   ├── __init__.py
│   ├── block.py           # FeatureBlock
│   ├── store.py           # FeatureBlockStore
│   ├── registry.py        # SampleRegistry
│   ├── targets.py         # TargetStore
│   ├── views.py           # ViewSpec, ViewResolver
│   └── context.py         # DatasetContext
├── dag/                   # DAG engine
│   ├── __init__.py
│   ├── model.py           # DAG model
│   ├── builder.py         # DAG builder
│   ├── engine.py          # Execution engine
│   └── nodes/             # Node types
│       ├── __init__.py
│       ├── base.py
│       ├── transform.py
│       ├── model.py
│       ├── fork.py
│       ├── join.py
│       └── splitter.py
├── models/                # Model management
│   ├── __init__.py
│   ├── virtual.py         # VirtualModel
│   └── aggregation.py
├── artifacts/             # Artifact management
│   ├── __init__.py
│   ├── manager.py
│   └── bundle.py
├── sklearn/               # sklearn integration
│   ├── __init__.py
│   ├── regressor.py
│   ├── classifier.py
│   └── search.py
├── transforms/            # NIRS transforms
│   ├── __init__.py
│   ├── nirs.py
│   ├── derivatives.py
│   └── signal.py
├── splitters/             # Data splitters
│   ├── __init__.py
│   ├── kennard_stone.py
│   ├── spxy.py
│   └── group.py
├── feature_selection/     # Feature selection
│   ├── __init__.py
│   └── ...
├── cli/                   # Command-line interface
│   ├── __init__.py
│   └── main.py
└── utils/                 # Utilities
    ├── __init__.py
    ├── hash.py
    ├── io.py
    └── logging.py
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| polars | ≥0.20 | DataFrames, Parquet I/O |
| scikit-learn | ≥1.3 | ML algorithms, transformers |
| pydantic | ≥2.0 | Data validation, serialization |
| click | ≥8.0 | CLI framework |
| rich | ≥13.0 | Console output, progress bars |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| xarray | ≥2023.0 | Named dimensions (optional) |
| torch | ≥2.0 | PyTorch models |
| tensorflow | ≥2.15 | TensorFlow models |
| jax | ≥0.4 | JAX models |
| shap | ≥0.45 | Explanations |
| optuna | ≥3.0 | Hyperparameter tuning |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥7.0 | Testing |
| pytest-cov | ≥4.0 | Coverage |
| pytest-benchmark | ≥4.0 | Performance tests |
| mypy | ≥1.0 | Type checking |
| ruff | ≥0.1 | Linting |
| sphinx | ≥7.0 | Documentation |

### pyproject.toml

```toml
[project]
name = "nirs4all"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "polars>=0.20",
    "scikit-learn>=1.3",
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
tensorflow = ["tensorflow>=2.15"]
jax = ["jax>=0.4", "flax>=0.7"]
shap = ["shap>=0.45"]
optuna = ["optuna>=3.0"]
all = ["nirs4all[torch,tensorflow,jax,shap,optuna]"]

dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-benchmark>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "myst-parser>=2.0",
]
```

---

## Testing Strategy

### Test Categories

#### Unit Tests

```
tests/
├── unit/
│   ├── data/
│   │   ├── test_block.py
│   │   ├── test_store.py
│   │   ├── test_views.py
│   │   └── test_context.py
│   ├── dag/
│   │   ├── test_model.py
│   │   ├── test_builder.py
│   │   ├── test_engine.py
│   │   └── test_nodes.py
│   ├── models/
│   │   └── test_virtual.py
│   └── api/
│       ├── test_run.py
│       ├── test_predict.py
│       └── test_sklearn.py
```

#### Integration Tests

```
tests/
├── integration/
│   ├── test_full_pipeline.py
│   ├── test_branching.py
│   ├── test_generators.py
│   └── test_export_import.py
```

#### Compatibility Tests

```
tests/
├── compatibility/
│   ├── test_v1_examples.py      # Run all Q* examples
│   ├── test_sklearn_compat.py   # GridSearchCV, SHAP
│   └── test_bundle_compat.py    # v1 bundles load in v2
```

### Testing Patterns

#### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.arrays(np.float64, st.tuples(
    st.integers(10, 100), st.integers(5, 50)
)))
def test_block_roundtrip(data):
    """FeatureBlock survives serialization roundtrip."""
    block = FeatureBlock(data=data)
    serialized = block.to_bytes()
    restored = FeatureBlock.from_bytes(serialized)
    np.testing.assert_array_equal(block.data, restored.data)
```

#### Snapshot Testing

```python
def test_dag_compilation_snapshot(snapshot):
    """DAG structure matches expected snapshot."""
    pipeline = [SNV(), PLSRegression(n_components=10)]
    dag = DAGBuilder().build(pipeline)

    snapshot.assert_match(dag.to_json(), "simple_pipeline.json")
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest tests/ --cov=nirs4all --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  compatibility:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - name: Run v1 examples
        run: |
          cd examples
          ./run.sh
```

---

## Edge Case Testing

Per the critical review, we must test edge cases that are often overlooked. This section defines degenerate inputs and expected behavior.

### Edge Case Categories

| Category | Cases | Expected Behavior |
|----------|-------|-------------------|
| **Empty Data** | 0 samples, 0 features | Raise `EmptyDataError` with helpful message |
| **Single Sample** | n_samples=1 | Warning; CV disabled or error |
| **Single Feature** | n_features=1 | Works; some transforms may warn |
| **NaN Values** | NaN in X or y | Configurable: raise, warn, impute |
| **Inf Values** | ±Inf in X or y | Raise `InvalidDataError` |
| **Constant Features** | std=0 for a feature | Warning; VarianceThreshold suggests removal |
| **Memory Limits** | Very large datasets | Chunked processing or clear OOM error |
| **Concurrent Access** | Multi-threaded access | Thread-safe or documented limitations |

### Edge Case Test Suite

```python
# tests/edge_cases/test_empty_data.py
import pytest
import numpy as np
from nirs4all import run
from nirs4all.exceptions import EmptyDataError, InvalidDataError


class TestEmptyData:
    """Tests for empty/degenerate data handling."""

    def test_zero_samples_raises(self):
        """Empty dataset should raise EmptyDataError."""
        X = np.array([]).reshape(0, 10)
        y = np.array([])

        with pytest.raises(EmptyDataError, match="Dataset has 0 samples"):
            run([MinMaxScaler(), PLSRegression()], (X, y))

    def test_zero_features_raises(self):
        """Dataset with no features should raise EmptyDataError."""
        X = np.array([]).reshape(10, 0)
        y = np.arange(10)

        with pytest.raises(EmptyDataError, match="Dataset has 0 features"):
            run([MinMaxScaler(), PLSRegression()], (X, y))


class TestSingleSample:
    """Tests for single-sample datasets."""

    def test_single_sample_cv_disabled(self):
        """Single sample should disable CV and warn."""
        X = np.array([[1, 2, 3]])
        y = np.array([1.0])

        with pytest.warns(UserWarning, match="Single sample.*CV disabled"):
            result = run([MinMaxScaler()], (X, y), cv=None)

    def test_single_sample_with_cv_raises(self):
        """Single sample with CV should raise."""
        X = np.array([[1, 2, 3]])
        y = np.array([1.0])

        with pytest.raises(ValueError, match="Cannot perform CV with 1 sample"):
            run([MinMaxScaler()], (X, y), cv=5)


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_nan_in_x_default_raises(self):
        """NaN in features should raise by default."""
        X = np.array([[1, np.nan, 3], [4, 5, 6]])
        y = np.array([1.0, 2.0])

        with pytest.raises(InvalidDataError, match="NaN values found in X"):
            run([MinMaxScaler()], (X, y))

    def test_nan_in_x_with_impute(self):
        """NaN in features should be imputed when configured."""
        X = np.array([[1, np.nan, 3], [4, 5, 6]])
        y = np.array([1.0, 2.0])

        result = run(
            [MinMaxScaler()],
            (X, y),
            nan_policy="impute"  # Use SimpleImputer
        )
        assert not np.any(np.isnan(result.y_pred))

    def test_nan_in_y_raises(self):
        """NaN in targets should always raise."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1.0, np.nan])

        with pytest.raises(InvalidDataError, match="NaN values found in y"):
            run([MinMaxScaler()], (X, y))


class TestInfHandling:
    """Tests for Inf value handling."""

    def test_inf_in_x_raises(self):
        """Inf in features should raise."""
        X = np.array([[1, np.inf, 3], [4, 5, 6]])
        y = np.array([1.0, 2.0])

        with pytest.raises(InvalidDataError, match="Inf values found in X"):
            run([MinMaxScaler()], (X, y))

    def test_neg_inf_in_x_raises(self):
        """-Inf in features should raise."""
        X = np.array([[1, -np.inf, 3], [4, 5, 6]])
        y = np.array([1.0, 2.0])

        with pytest.raises(InvalidDataError, match="Inf values found in X"):
            run([MinMaxScaler()], (X, y))


class TestConstantFeatures:
    """Tests for constant feature handling."""

    def test_constant_feature_warns(self):
        """Constant feature should warn."""
        X = np.array([[1, 2, 3], [1, 5, 6], [1, 8, 9]])  # First column constant
        y = np.array([1.0, 2.0, 3.0])

        with pytest.warns(UserWarning, match="Feature 0 has zero variance"):
            run([MinMaxScaler()], (X, y))

    def test_all_constant_raises(self):
        """All constant features should raise."""
        X = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(InvalidDataError, match="All features have zero variance"):
            run([MinMaxScaler()], (X, y))


class TestMemoryLimits:
    """Tests for memory limit handling."""

    @pytest.mark.slow
    def test_large_dataset_chunked(self):
        """Large dataset should use chunked processing."""
        # 1M samples × 1000 features = ~8GB
        # Should not load all at once
        X = np.random.randn(10000, 1000)  # Smaller for test
        y = np.random.randn(10000)

        # Should complete without OOM
        result = run([MinMaxScaler()], (X, y), chunked=True)
        assert result.y_pred.shape == y.shape

    def test_oom_gives_helpful_error(self):
        """OOM should give helpful error message."""
        # This test is conceptual - hard to trigger OOM reliably
        pass
```

### Property-Based Edge Case Testing

```python
# tests/edge_cases/test_property_based.py
from hypothesis import given, strategies as st, assume
import numpy as np

@given(
    n_samples=st.integers(min_value=2, max_value=100),
    n_features=st.integers(min_value=1, max_value=50)
)
def test_random_shapes_work(n_samples, n_features):
    """Pipeline should handle various shapes."""
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Should complete without error
    result = run([MinMaxScaler()], (X, y), cv=min(3, n_samples))
    assert result.y_pred.shape == (n_samples,)


@given(
    scale=st.floats(min_value=1e-10, max_value=1e10)
)
def test_scale_invariance(scale):
    """Scaling input should not break pipeline."""
    assume(np.isfinite(scale))

    X = np.random.randn(20, 10) * scale
    y = np.random.randn(20) * scale

    # Should complete
    result = run([StandardScaler()], (X, y), cv=3)
    assert np.all(np.isfinite(result.y_pred))
```

### Advanced Edge Case Testing

Per the critical review, additional edge cases require explicit testing:

```python
# tests/edge_cases/test_advanced.py
import pytest
import numpy as np
from nirs4all import run
from nirs4all.transforms import SNV, MSC
from sklearn.cross_decomposition import PLSRegression


class TestNestedBranching:
    """Tests for nested fork/join patterns."""

    def test_nested_branches_depth_2(self):
        """Nested branching should work to depth 2."""
        X = np.random.randn(50, 30)
        y = np.random.randn(50)

        pipeline = [
            MinMaxScaler(),
            # Outer branch
            {"branch": [
                # Inner branch within outer branch 0
                [SNV(), {"branch": [
                    [{"model": PLSRegression(n_components=5)}],
                    [{"model": PLSRegression(n_components=10)}]
                ]}, {"merge": "predictions"}],
                # Inner branch within outer branch 1
                [MSC(), {"model": PLSRegression(n_components=8)}]
            ]},
            {"merge": "predictions"},
            {"model": Ridge()}
        ]

        result = run(pipeline, (X, y), cv=3)
        assert result.y_pred.shape == (50,)

    def test_nested_branching_explosion_prevented(self):
        """Deeply nested branches should hit limits."""
        from nirs4all.exceptions import GeneratorExplosionError

        # Create pathologically deep nesting
        def make_deep_branch(depth):
            if depth == 0:
                return [{"model": PLSRegression(n_components=5)}]
            return [{"branch": [make_deep_branch(depth - 1) for _ in range(3)]}]

        pipeline = make_deep_branch(5)  # 3^5 = 243 branches

        with pytest.raises(GeneratorExplosionError):
            run(pipeline, (X, y), cv=3, max_variants=100)

    def test_branch_isolation_under_parallel(self):
        """Parallel branches should not share state."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Custom transform that tracks instance ID
        class StatefulTransform:
            _instance_count = 0

            def __init__(self):
                StatefulTransform._instance_count += 1
                self.instance_id = StatefulTransform._instance_count
                self.fit_count = 0

            def fit(self, X, y=None):
                self.fit_count += 1
                return self

            def transform(self, X):
                return X

        pipeline = [
            {"branch": [
                [StatefulTransform(), {"model": PLSRegression(5)}],
                [StatefulTransform(), {"model": PLSRegression(10)}]
            ]},
            {"merge": "predictions"}
        ]

        # Run with parallelism
        result = run(pipeline, (X, y), cv=3, parallel=True, n_workers=2)

        # Each branch should have its own instance
        # (actual verification would require inspection of artifacts)
        assert result.y_pred.shape == (100,)


class TestConcurrentExecution:
    """Tests for concurrent/parallel execution scenarios."""

    @pytest.mark.parametrize("n_workers", [1, 2, 4])
    def test_parallel_cv_deterministic(self, n_workers):
        """Parallel CV should give same results as sequential."""
        X = np.random.randn(80, 40)
        y = np.random.randn(80)

        pipeline = [MinMaxScaler(), KFold(5), {"model": PLSRegression(10)}]

        result_seq = run(pipeline, (X, y), random_state=42, parallel=False)
        result_par = run(pipeline, (X, y), random_state=42,
                        parallel=True, n_workers=n_workers)

        np.testing.assert_allclose(
            result_seq.y_pred, result_par.y_pred, rtol=1e-10
        )

    def test_concurrent_branch_execution(self):
        """Multiple branches executing concurrently."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # 5 branches with different processing times
        pipeline = [
            {"branch": [
                [SNV(), {"model": PLSRegression(5)}],
                [MSC(), {"model": PLSRegression(8)}],
                [Detrend(), {"model": PLSRegression(10)}],
                [FirstDerivative(), {"model": PLSRegression(12)}],
                [SecondDerivative(), {"model": PLSRegression(15)}]
            ]},
            {"merge": "predictions"}
        ]

        result = run(pipeline, (X, y), cv=3, parallel=True, n_workers=4)
        assert result.y_pred.shape == (100,)

    def test_thread_safety_under_load(self):
        """Stress test for thread safety."""
        import threading

        X = np.random.randn(50, 30)
        y = np.random.randn(50)
        pipeline = [MinMaxScaler(), KFold(3), {"model": PLSRegression(5)}]

        results = []
        errors = []

        def worker(seed):
            try:
                result = run(pipeline, (X, y), random_state=seed)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        assert len(results) == 10


class TestMixedGPUCPU:
    """Tests for mixed GPU/CPU workflows."""

    @pytest.mark.gpu
    def test_gpu_cpu_branch_mix(self):
        """Branches with GPU and CPU models should work together."""
        from nirs4all.models import Nicon  # GPU model

        X = np.random.randn(100, 50)
        y = np.random.randint(0, 3, 100)

        pipeline = [
            MinMaxScaler(),
            {"branch": [
                [{"model": PLSRegression(10)}],  # CPU
                [{"model": Nicon(n_classes=3, device="cuda")}]  # GPU
            ]},
            {"merge": "predictions"},
            {"model": LogisticRegression()}
        ]

        result = run(pipeline, (X, y), cv=3)
        assert result.y_pred.shape == (100,)

    @pytest.mark.gpu
    def test_gpu_memory_cleanup_after_branch(self):
        """GPU memory should be freed after branch completion."""
        import torch

        X = np.random.randn(100, 50)
        y = np.random.randint(0, 3, 100)

        initial_memory = torch.cuda.memory_allocated()

        pipeline = [
            MinMaxScaler(),
            {"branch": [
                [{"model": Nicon(n_classes=3, device="cuda")}]
            ]},
            {"merge": "predictions"}
        ]

        result = run(pipeline, (X, y), cv=3)

        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should return to approximately initial level
        assert final_memory < initial_memory * 1.1  # Allow 10% overhead

    @pytest.mark.gpu
    def test_fallback_to_cpu_on_oom(self):
        """GPU OOM should gracefully fallback to CPU."""
        # Create model that might OOM on small GPUs
        from nirs4all.models import Nicon

        X = np.random.randn(1000, 500)
        y = np.random.randint(0, 10, 1000)

        pipeline = [
            MinMaxScaler(),
            {
                "model": Nicon(
                    n_classes=10,
                    device="cuda",
                    fallback_to_cpu=True  # Enable fallback
                )
            }
        ]

        # Should complete (either on GPU or CPU)
        result = run(pipeline, (X, y), cv=3)
        assert result.y_pred.shape == (1000,)


class TestBranchWithDifferentSampleCounts:
    """Tests for branches that modify sample counts."""

    def test_augmentation_in_one_branch(self):
        """One branch augments, other doesn't."""
        X = np.random.randn(50, 30)
        y = np.random.randn(50)

        pipeline = [
            MinMaxScaler(),
            {"branch": [
                # Branch 0: with augmentation
                [
                    {"sample_augmentation": NoiseAugmentation(n_augmented=2)},
                    {"model": PLSRegression(10)}
                ],
                # Branch 1: no augmentation
                [{"model": PLSRegression(10)}]
            ]},
            {"merge": "predictions"}  # OOF reconstruction handles different sample counts
        ]

        result = run(pipeline, (X, y), cv=3)
        # Final predictions should match original sample count
        assert result.y_pred.shape == (50,)

    def test_filtering_in_branch(self):
        """Branch that removes outliers."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)
        # Add some outliers
        X[0] = X[0] * 100

        pipeline = [
            MinMaxScaler(),
            {"branch": [
                # Branch with outlier removal
                [
                    {"outlier_detection": MahalanobisOutlier(threshold=2.0)},
                    {"model": PLSRegression(10)}
                ],
                # Branch without
                [{"model": PLSRegression(10)}]
            ]},
            {"merge": "predictions"}
        ]

        result = run(pipeline, (X, y), cv=3)
        assert result.y_pred.shape == (100,)
```

---

## Migration from v1.x

### Compatibility Layer

Provide a compatibility shim for gradual migration:

```python
# nirs4all/compat/__init__.py
"""Compatibility layer for v1.x code."""

import warnings

def _warn_deprecated(old, new):
    warnings.warn(
        f"{old} is deprecated, use {new} instead",
        DeprecationWarning,
        stacklevel=3
    )

# Old imports map to new
from nirs4all.data import DatasetContext as SpectroDataset
from nirs4all.api import run

class PipelineRunner:
    """Compatibility shim for v1.x PipelineRunner."""

    def __init__(self, verbose=1, save_artifacts=True, **kwargs):
        _warn_deprecated("PipelineRunner", "nirs4all.run()")
        self.verbose = verbose
        self.save_artifacts = save_artifacts
        self.kwargs = kwargs

    def run(self, pipeline_config, dataset_config):
        from nirs4all import run as nirs_run
        return nirs_run(
            pipeline=pipeline_config.steps,
            data=dataset_config.path,
            name=pipeline_config.name,
            verbose=self.verbose,
            save_artifacts=self.save_artifacts,
            **self.kwargs
        )
```

### Migration Guide

#### Before (v1.x)

```python
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1, save_artifacts=True)
predictions, per_dataset = runner.run(
    PipelineConfigs([SNV(), PLSRegression()], "my_pipe"),
    DatasetConfigs("data.csv")
)

best = predictions.get_best()
```

#### After (v2.0)

```python
import nirs4all

result = nirs4all.run(
    [SNV(), PLSRegression()],
    "data.csv",
    name="my_pipe"
)

best = result.best
```

### Breaking Changes

| v1.x | v2.0 | Notes |
|------|------|-------|
| `PipelineRunner` | `nirs4all.run()` | Simplified API |
| `PipelineConfigs` | Direct list | No wrapper needed |
| `DatasetConfigs` | Path or dict | Flexible input |
| `SpectroDataset` | `DatasetContext` | New name, new impl |
| `predictions.get_best()` | `result.best` | Property access |
| `.n4a` bundle v1 | `.n4a` bundle v2 | New format (converter provided) |

### Bundle Converter

```python
def convert_bundle_v1_to_v2(v1_path: str, v2_path: str) -> None:
    """Convert v1.x bundle to v2 format."""
    from nirs4all.compat import load_v1_bundle
    from nirs4all.artifacts import BundleExporter

    # Load v1 bundle
    v1_bundle = load_v1_bundle(v1_path)

    # Convert artifacts
    v2_artifacts = _convert_artifacts(v1_bundle.artifacts)

    # Export as v2
    exporter = BundleExporter(v2_artifacts, None)
    exporter.export(v2_path)
```

---

## Migration Testing

Per the critical review, "All examples passing" doesn't guarantee behavior equivalence. This section defines rigorous migration testing.

### Numerical Equivalence Testing

v2 must produce predictions within tolerance of v1 for identical inputs and random seeds.

```python
# tests/migration/test_numerical_equivalence.py
import numpy as np
import pytest
from pathlib import Path

# Tolerance for numerical equivalence
R2_TOLERANCE = 0.001
RMSE_RELATIVE_TOLERANCE = 0.01


class TestNumericalEquivalence:
    """Test that v2 produces equivalent predictions to v1."""

    @pytest.fixture(scope="class")
    def v1_snapshots(self):
        """Load v1 prediction snapshots."""
        snapshot_dir = Path(__file__).parent / "snapshots" / "v1"
        snapshots = {}
        for f in snapshot_dir.glob("*.npz"):
            data = np.load(f)
            snapshots[f.stem] = {
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
                "r2": data["r2"],
                "rmse": data["rmse"]
            }
        return snapshots

    @pytest.mark.parametrize("example", [
        "Q1_regression",
        "Q2_multimodel",
        "Q5_predict",
        "Q18_stacking",
        "Q25_complex_pipeline_pls"
    ])
    def test_example_equivalence(self, example, v1_snapshots):
        """Run example and compare with v1 snapshot."""
        # Run v2
        result = self._run_example_v2(example)

        # Compare with v1 snapshot
        v1 = v1_snapshots[example]

        # Check R² within tolerance
        r2_v2 = self._compute_r2(v1["y_true"], result.y_pred)
        assert abs(r2_v2 - v1["r2"]) < R2_TOLERANCE, \
            f"R² divergence: v1={v1['r2']:.4f}, v2={r2_v2:.4f}"

        # Check RMSE within relative tolerance
        rmse_v2 = self._compute_rmse(v1["y_true"], result.y_pred)
        relative_diff = abs(rmse_v2 - v1["rmse"]) / v1["rmse"]
        assert relative_diff < RMSE_RELATIVE_TOLERANCE, \
            f"RMSE divergence: v1={v1['rmse']:.4f}, v2={rmse_v2:.4f} ({relative_diff*100:.1f}%)"

    def _run_example_v2(self, example_name):
        """Run example with v2 and return result."""
        # Import and run example module
        import importlib
        module = importlib.import_module(f"examples.{example_name}")
        return module.run_example()

    def _compute_r2(self, y_true, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)

    def _compute_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class TestDeterminism:
    """Test that v2 is deterministic with same random seed."""

    def test_same_seed_same_result(self):
        """Same random_state should produce identical results."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        result1 = run([MinMaxScaler(), PLSRegression()], (X, y), random_state=42)
        result2 = run([MinMaxScaler(), PLSRegression()], (X, y), random_state=42)

        np.testing.assert_array_equal(result1.y_pred, result2.y_pred)

    def test_different_seed_different_result(self):
        """Different random_state should produce different results (usually)."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        result1 = run([MinMaxScaler(), PLSRegression()], (X, y), random_state=42)
        result2 = run([MinMaxScaler(), PLSRegression()], (X, y), random_state=123)

        # May be same by chance, but very unlikely
        assert not np.allclose(result1.y_pred, result2.y_pred)
```

### Snapshot Generation

```python
# scripts/generate_v1_snapshots.py
"""Generate v1 prediction snapshots for migration testing.

Run this with v1.x installed before upgrading to v2.
"""

import numpy as np
from pathlib import Path
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

EXAMPLES = [
    ("Q1_regression", "examples/Q1_regression.py"),
    ("Q2_multimodel", "examples/Q2_multimodel.py"),
    # ... more examples
]

SNAPSHOT_DIR = Path("tests/migration/snapshots/v1")


def generate_snapshot(name: str, example_path: str):
    """Generate and save prediction snapshot."""
    # Import and run example
    import importlib.util
    spec = importlib.util.spec_from_file_location("example", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get predictions
    predictions = module.get_predictions()
    best = predictions.get_best()

    # Compute metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(best.y_true, best.y_pred)
    rmse = np.sqrt(mean_squared_error(best.y_true, best.y_pred))

    # Save
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        SNAPSHOT_DIR / f"{name}.npz",
        y_true=best.y_true,
        y_pred=best.y_pred,
        r2=r2,
        rmse=rmse
    )
    print(f"Saved: {name} (R²={r2:.4f}, RMSE={rmse:.4f})")


if __name__ == "__main__":
    for name, path in EXAMPLES:
        generate_snapshot(name, path)
```

### Divergence Flagging

When predictions diverge beyond tolerance, automatic review is triggered:

```python
# tests/migration/conftest.py
import pytest

DIVERGENCE_LOG = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture divergence details for review."""
    outcome = yield
    report = outcome.get_result()

    if report.failed and "divergence" in str(report.longrepr).lower():
        DIVERGENCE_LOG.append({
            "test": item.name,
            "details": str(report.longrepr)
        })


def pytest_sessionfinish(session, exitstatus):
    """Generate divergence report at end of session."""
    if DIVERGENCE_LOG:
        print("\n" + "=" * 60)
        print("NUMERICAL DIVERGENCE REPORT")
        print("=" * 60)
        for entry in DIVERGENCE_LOG:
            print(f"\n{entry['test']}:")
            print(entry['details'][:500])  # Truncate
        print("\nThese divergences require manual review before release.")
```

---

## Risks and Mitigations

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Polars performance regression | High | Low | Benchmark early, profile hotspots |
| DAG complexity explosion | Medium | Medium | Limit generator expansion, add warnings |
| sklearn compatibility breaks | High | Low | Extensive test suite, check_estimator() |
| Memory issues with large data | High | Medium | Streaming views, chunked processing |
| Hash collision | Low | Very Low | Use SHA-256, validate on collision |

### Schedule Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Scope creep | High | High | Strict phase gates, no new features |
| Integration issues | Medium | Medium | Continuous integration, early testing |
| Documentation lag | Medium | High | Doc-as-you-go policy |

### Migration Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| User adoption resistance | High | Medium | Comprehensive migration guide, compat layer |
| Subtle behavior changes | High | Medium | **Numerical equivalence testing** |
| v1 bundle incompatibility | Medium | Low | Provide converter tool |

---

## Timeline

Per the critical review, the original 15-week timeline was optimistic. This revised timeline adds **50% buffer** to each phase and defines **MVP scope** per phase.

### Revised Gantt Chart (with Buffer)

```
Week:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
       ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
Phase 0 ███        (3 weeks: 2 base + 1 buffer)
Phase 1    █████   (5 weeks: 3 base + 2 buffer, parallel with Phase 2)
Phase 2    ██████  (6 weeks: 4 base + 2 buffer, parallel with Phase 1)
Phase 3          █████  (5 weeks: 3 base + 2 buffer)
Phase 4               █████  (5 weeks: 3 base + 2 buffer)
Phase 5                    ████  (4 weeks: 3 base + 1 buffer)
Phase 6                         █████  (5 weeks: 3 base + 2 buffer)
Phase 7                              ████  (4 weeks: 3 base + 1 buffer)

Buffer zones: ░░░░░ (overlap for contingency)
```

### MVP Scope Per Phase

Each phase defines what MUST be done (MVP) vs what CAN be deferred.

| Phase | MVP (Must Have) | Deferrable |
|-------|-----------------|------------|
| **Phase 0** | Core protocols, FeatureBlock, SampleRegistry | Xarray integration |
| **Phase 1** | FeatureBlockStore, ViewSpec, DatasetContext | Multi-source optimization |
| **Phase 2** | DAG model, builder (linear), sequential execution | Parallel execution |
| **Phase 3** | VirtualModel (mean aggregation), basic bundle | Advanced aggregation strategies |
| **Phase 4** | `run()`, `predict()`, basic `explain()` | Full explainer abstraction |
| **Phase 5** | SNV, MSC, Detrend, SavGol; KFold, ShuffleSplit | CARS, MC-UVE, SPXY |
| **Phase 6** | Q1-Q5 examples passing | Q6-Q36 (add iteratively) |
| **Phase 7** | API docs, migration guide | Full user guide rewrite |

### Revised Milestones

| Week | Milestone | Deliverable | Exit Criteria |
|------|-----------|-------------|---------------|
| 3 | M0: Foundation | Protocols, FeatureBlock | All protocol tests pass |
| 8 | M1: Data Layer | DatasetContext MVP | ViewSpec materializes correctly |
| 8 | M2: DAG Engine | Sequential execution | Linear pipeline executes |
| 12 | M3: Models | VirtualModel, bundle export | Bundle round-trips |
| 15 | M4: API | run/predict working | Q1 example passes |
| 17 | M5: Operators | Core transforms ported | Transform tests pass |
| 19 | M6: Integration | Q1-Q10 examples passing | Numerical equivalence < tolerance |
| 22 | M7: Release | v2.0.0 RC | All examples, docs, benchmarks |

### Resource Allocation (Revised)

| Phase | Base Effort | Buffer (+50%) | Total | Parallelizable |
|-------|-------------|---------------|-------|----------------|
| Phase 0 | 3 weeks | 2 weeks | **5 weeks** | No |
| Phase 1 | 3 weeks | 2 weeks | **5 weeks** | Yes (with Phase 2) |

**Note on Phase 0**: The foundation phase (FeatureBlockStore, CoW semantics, Thread-safety, GC, Mmap support) is *critical*. Per external review, 3 weeks base + 2 weeks buffer is more realistic than 2+1. If this layer is poorly designed, the entire architecture collapses.
| Phase 2 | 4 weeks | 2 weeks | **6 weeks** | Yes (with Phase 1) |
| Phase 3 | 3 weeks | 2 weeks | **5 weeks** | No |
| Phase 4 | 3 weeks | 2 weeks | **5 weeks** | No |
| Phase 5 | 3 weeks | 1 week | **4 weeks** | Yes |
| Phase 6 | 3 weeks | 2 weeks | **5 weeks** | No |
| Phase 7 | 3 weeks | 1 week | **4 weeks** | Partially |
| **Total** | **24 weeks** | **13 weeks** | **37 weeks** | |

**Note**: With parallel execution of Phases 1+2, the critical path is ~22 weeks.

### Phase Gate Criteria

Before advancing to the next phase, these criteria must be met:

| Phase | Gate Criteria |
|-------|---------------|
| 0 → 1,2 | All Protocol classes defined and documented |
| 1 → 3 | DatasetContext satisfies DataProvider protocol |
| 2 → 3 | ExecutionEngine runs linear DAGs correctly |
| 3 → 4 | VirtualModel predict works; bundle exports |
| 4 → 5 | `nirs4all.run()` produces valid RunResult |
| 5 → 6 | Core transforms (SNV, MSC, etc.) pass sklearn tests |
| 6 → 7 | ≥80% examples passing; numerical equivalence holds |
| 7 → Release | All tests pass; docs complete; benchmarks acceptable |

---

## Appendix: Checklist

### Pre-Development

- [ ] Review and approve design documents
- [ ] Set up v2 branch
- [ ] Configure CI/CD for v2
- [ ] Create issue tracking board
- [ ] **Generate v1 prediction snapshots for migration testing**

### Per-Phase

- [ ] Write tests before implementation
- [ ] Document public APIs
- [ ] Review code before merge
- [ ] Update CHANGELOG
- [ ] **Check protocol compliance**

### Pre-Release

- [ ] All tests passing
- [ ] All examples ported
- [ ] **Numerical equivalence tests passing**
- [ ] **Edge case tests passing**
- [ ] Documentation complete
- [ ] Migration guide reviewed
- [ ] Performance benchmarks acceptable
- [ ] Security review (dependency audit)

---

## Next Steps

With all five design documents complete:

1. **Critical Review**: Analyze for weaknesses and gaps ✓ (Completed)
2. **Stakeholder Feedback**: Review with users
3. **Revised Design**: Incorporate feedback ✓ (This revision)
4. **Implementation**: Begin Phase 0
2. **Stakeholder Feedback**: Review with users
3. **Revised Design**: Incorporate feedback
4. **Implementation**: Begin Phase 0
