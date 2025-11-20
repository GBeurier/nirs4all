# Context Refactoring Proposal

**Status**: Proposal
**Priority**: CRITICAL BLOCKER for Runner Refactoring
**Complexity**: High - 176+ usage locations
**Impact**: Foundation for all pipeline improvements
**Breaking Change**: YES - Clean refactoring with no backward compatibility

---

## Executive Summary

The `context` object (Dict[str, Any]) is the most critical data structure in nirs4all's pipeline system. It serves three distinct roles simultaneously, creating tight coupling and maintainability issues. This proposal separates context into three typed components with a **clean breaking change**.

**Core Problem**: Context mixes data selection, execution state, and controller coordination in a single untyped dictionary.

**Solution**: Separate concerns into DataSelector, PipelineState, StepMetadata, and extensible custom data with atomic migration.

**Strategy**: No backward compatibility, no deprecation, no gradual migration. Clean break in one release.

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Current Context Usage](#current-context-usage)
3. [Proposed Architecture](#proposed-architecture)
4. [Migration Strategy](#migration-strategy)
5. [Integration with Runner Refactoring](#integration-with-runner-refactoring)
6. [Testing Strategy](#testing-strategy)
7. [Performance Considerations](#performance-considerations)

---

## Problem Analysis

### Current State

Context is a `Dict[str, Any]` that serves three distinct roles:

```python
context = {
    # 1. DATA SELECTOR (consumed by dataset.x() and .y())
    "partition": "train",              # Which data subset
    "processing": [["raw"], ["snv"]],  # Which transformations (per source)
    "layout": "2d",                     # Data shape preference

    # 2. PIPELINE EXECUTION STATE (modified as pipeline progresses)
    "y": "numeric",                     # Y transformation state
    # "processing" also tracks evolving chains

    # 3. CONTROLLER COMMUNICATION (multi-step coordination)
    "augment_sample": False,            # Transformer mode flag
    "add_feature": True,                # Feature operation flag
    "target_samples": [1, 2, 3],       # Sample augmentation targets
    "keyword": "sample_augmentation",   # Operator matching
    "step_id": "step_001",             # Step tracking
}
```

### Critical Issues

1. **Type Safety Violation**: No type hints, runtime errors common
2. **Mixed Concerns**: Data selection mixed with execution state mixed with coordination
3. **Implicit Coupling**: Controllers communicate via flags without explicit interface
4. **Backward Incompatibility Risk**: Any context change breaks dataset operations
5. **Testing Complexity**: Must mock entire dict for each test
6. **Documentation Burden**: No single source of truth for context keys

### Usage Statistics (from codebase analysis)

- **176+ context usage locations** across codebase
- **34 processing/partition modifications** in controllers
- **18+ controller types** all depend on context
- **dataset.x() and .y()** use context as Selector (type alias for Dict[str, Any])
- **All tests** initialize context manually

---

## Current Context Usage

### Role 1: Data Selector (Dataset Operations)

**Purpose**: Filter and select data from SpectroDataset

**Used By**:
- `SpectroDataset.x(selector, layout, concat_source, include_augmented)`
- `SpectroDataset.y(selector, include_augmented)`
- `dataset._indexer.x_indices(selector, include_augmented)`
- `dataset._indexer.y_indices(selector, include_augmented)`

**Key Fields**:
```python
{
    "partition": str,              # "train", "test", "all", "val"
    "processing": List[List[str]], # Processing chains per source
    "layout": str,                 # "2d", "3d" (optional, defaults vary)
    "fold_id": int,                # Optional fold filtering (not always used)
}
```

**Example Usage**:
```python
train_context = context.copy()
train_context["partition"] = "train"
X_train = dataset.x(train_context, layout="2d")
y_train = dataset.y(train_context, include_augmented=True)
```

**Critical Finding**: `Selector = Dict[str, Any]` is a type alias in dataset.py. This IS the context object passed to data operations.

### Role 2: Pipeline Execution State

**Purpose**: Track pipeline state as it evolves through transformations

**Modified By**:
- TransformerMixinController (updates processing chains)
- YTransformerController (updates y processing)
- ResamplerController (updates processing chains)
- FeatureAugmentationController (resets processing to all features)

**Key Fields**:
```python
{
    "processing": List[List[str]], # Evolves as transformers add/replace features
    "y": str,                       # Tracks y transformation state
}
```

**Example Modification Pattern**:
```python
# TransformerMixinController after applying PCA
context["processing"][source_idx] = ["raw_PCA_001"]

# YTransformerController after encoding
updated_context["y"] = "encoded_LabelEncoder_001"
```

**Critical Pattern**: Controllers use `copy.deepcopy(context)` to create local contexts for different partitions:
```python
train_context = copy.deepcopy(context)
train_context["partition"] = "train"
X_train = dataset.x(train_context, layout="2d")
```

### Role 3: Controller Communication Flags

**Purpose**: Coordinate multi-step operations between controllers

**Used By**:
- SampleAugmentationController (sets augment_sample flag)
- FeatureAugmentationController (sets add_feature flag)
- TransformerMixinController (reads flags to change behavior)

**Key Fields**:
```python
{
    "augment_sample": bool,         # Tells transformer to operate in sample mode
    "add_feature": bool,            # Tells transformer to add vs replace features
    "target_samples": List[int],    # Which samples to augment
    "keyword": str,                 # Operator keyword for controller matching
    "step_id": str,                 # Step identifier for tracking
}
```

**Example Coordination**:
```python
# SampleAugmentationController prepares context
local_context = copy.deepcopy(context)
local_context["augment_sample"] = True
local_context["target_samples"] = [sample_id]

# TransformerMixinController detects flag and changes behavior
if context.get("augment_sample", False):
    return self._execute_for_sample_augmentation(...)
```

---

## Proposed Architecture

### Component Separation

```python
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from copy import deepcopy


@dataclass
class DataSelector:
    """
    Pure data filtering configuration consumed by dataset.x() and .y().

    This is the typed replacement for the Selector type alias.

    Processing chains are kept here (not in PipelineState) because:
    - Future: Flow controllers with multiple processing paths
    - Future: Feature caching based on processing chains
    - Selector needs processing to choose from cached features
    - Processing identifies which data variant to retrieve

    Mutable by design since processing chains evolve through pipeline.
    """
    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [[]])
    layout: str = "2d"
    fold_id: Optional[int] = None  # Used by selector if exists, ignored otherwise
    include_augmented: bool = True

    def with_partition(self, partition: str) -> 'DataSelector':
        """Immutable update pattern for partition."""
        return DataSelector(
            partition=partition,
            processing=deepcopy(self.processing),
            layout=self.layout,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented,
        )

    def with_processing(self, processing: List[List[str]]) -> 'DataSelector':
        """Immutable update pattern for processing chains."""
        return DataSelector(
            partition=self.partition,
            processing=deepcopy(processing),
            layout=self.layout,
            fold_id=self.fold_id,
            include_augmented=self.include_augmented,
        )


@dataclass
class PipelineState:
    """
    Pipeline execution state modified as pipeline progresses.

    Tracks pipeline-level state (NOT data selection state).
    Processing chains are in DataSelector since they select data.

    Mutable by design since pipeline state evolves through execution.
    """
    y_processing: str = "numeric"
    step_number: int = 0
    mode: str = "train"  # "train", "predict", "explain"

    def update_y(self, new_y_processing: str) -> None:
        """Update y transformation state."""
        self.y_processing = new_y_processing


@dataclass
class StepMetadata:
    """
    Controller communication and coordination metadata.

    Used for multi-step coordination between controllers.
    Cleared/reset between major pipeline steps.
    """
    keyword: str = ""
    step_id: str = ""
    augment_sample: bool = False
    add_feature: bool = False
    target_samples: Optional[List[int]] = None

    def reset_flags(self) -> None:
        """Clear coordination flags after use."""
        self.augment_sample = False
        self.add_feature = False
        self.target_samples = None


class ExecutionContext:
    """
    Unified context with typed access and extensibility for custom controllers.

    Design Principles:
    - Typed property access for core context
    - Extensible custom dict for controller-specific data
    - Deep copy semantics (controllers expect isolation)
    - No backward compatibility with dict interface

    Custom Data Usage:
    Custom controllers can store/retrieve data via the `custom` dict:

    ```python
    # Custom controller storing state
    context.custom["my_controller_state"] = {"threshold": 0.5, "iterations": 10}

    # Another controller reading custom data
    config = context.custom.get("my_controller_state", {})
    threshold = config.get("threshold", 0.3)
    ```
    """

    def __init__(
        self,
        selector: Optional[DataSelector] = None,
        state: Optional[PipelineState] = None,
        metadata: Optional[StepMetadata] = None,
        custom: Optional[Dict[str, Any]] = None,
    ):
        self.selector = selector or DataSelector()
        self.state = state or PipelineState()
        self.metadata = metadata or StepMetadata()
        self.custom = custom or {}  # Extensibility for custom controllers

    def copy(self) -> 'ExecutionContext':
        """Deep copy for controller isolation."""
        return ExecutionContext(
            selector=DataSelector(
                partition=self.selector.partition,
                processing=deepcopy(self.selector.processing),
                layout=self.selector.layout,
                fold_id=self.selector.fold_id,
                include_augmented=self.selector.include_augmented,
            ),
            state=PipelineState(
                y_processing=self.state.y_processing,
                step_number=self.state.step_number,
                mode=self.state.mode,
            ),
            metadata=StepMetadata(
                keyword=self.metadata.keyword,
                step_id=self.metadata.step_id,
                augment_sample=self.metadata.augment_sample,
                add_feature=self.metadata.add_feature,
                target_samples=self.metadata.target_samples[:] if self.metadata.target_samples else None,
            ),
            custom=deepcopy(self.custom),
        )

    def __deepcopy__(self, memo: Dict) -> 'ExecutionContext':
        """Support for copy.deepcopy()."""
        return self.copy()

    def with_partition(self, partition: str) -> 'ExecutionContext':
        """Create new context with updated partition."""
        new_ctx = self.copy()
        new_ctx.selector.partition = partition
        return new_ctx

    def with_processing(self, processing: List[List[str]]) -> 'ExecutionContext':
        """Create new context with updated processing chains."""
        new_ctx = self.copy()
        new_ctx.selector.processing = deepcopy(processing)
        return new_ctx
```

### Design Rationale

1. **Processing stays in DataSelector**: Future flow controllers need processing for cache selection
2. **PipelineState is mutable**: State evolves through pipeline
3. **StepMetadata is mutable**: Flags set/cleared between steps
4. **Custom dict for extensibility**: Controllers can store custom data via `context.custom`
5. **Deep copy semantics preserved**: Controllers expect isolation via copy.deepcopy()
6. **No backward compatibility**: Clean breaking change, no dict interface emulation

---

## Migration Strategy

### Overview

3-phase clean break migration with atomic changeover.

### Phase 1: Implement New Context (Week 1)

**Goal**: Create new context module with complete implementation

**Steps**:
1. Create `nirs4all/pipeline/context.py` with all classes
2. Write 50+ unit tests for context classes
3. **DO NOT** modify existing code yet

**Testing**:
```python
def test_execution_context_creation():
    """Verify ExecutionContext creation and copying."""
    ctx = ExecutionContext(
        selector=DataSelector(partition="train", processing=[["raw"]]),
        state=PipelineState(y_processing="numeric"),
        metadata=StepMetadata(keyword="transform")
    )

    assert ctx.selector.partition == "train"
    assert ctx.state.y_processing == "numeric"

    # Test deep copy isolation
    ctx2 = ctx.copy()
    ctx2.selector.partition = "test"
    assert ctx.selector.partition == "train"  # Original unchanged

def test_custom_controller_data():
    """Verify custom data storage."""
    ctx = ExecutionContext()
    ctx.custom["my_controller"] = {"threshold": 0.5}

    ctx2 = ctx.copy()
    assert ctx2.custom["my_controller"]["threshold"] == 0.5

    ctx2.custom["my_controller"]["threshold"] = 0.8
    assert ctx.custom["my_controller"]["threshold"] == 0.5  # Isolated
```

**Deliverables**:
- [ ] `context.py` with DataSelector, PipelineState, StepMetadata, ExecutionContext
- [ ] 50+ unit tests with 100% coverage
- [ ] Documentation for new context architecture
- [ ] No changes to existing pipeline code

### Phase 2: Atomic Migration (Week 2)

**Goal**: Update ALL code to use new context in one atomic change

**Critical**: This phase updates 176+ locations simultaneously. Use type checker to guide migration.

**Changes Required**:

**1. Dataset Operations**:
```python
# dataset.py - Update Selector type
from nirs4all.pipeline.context import DataSelector

# Change type alias
Selector = DataSelector  # No more dict support

def x(
    self,
    selector: DataSelector,
    layout: str = "2d",
    concat_source: bool = True,
    include_augmented: bool = True
) -> np.ndarray:
    """Get X data with DataSelector."""
    # Use selector directly (it's already typed)
    partition = selector.partition
    processing = selector.processing
    # ... existing logic
```

**2. PipelineRunner**:
```python
# runner.py
from nirs4all.pipeline.context import ExecutionContext, DataSelector, PipelineState

class PipelineRunner:
    def _initialize_context(self, dataset: SpectroDataset) -> ExecutionContext:
        """Initialize typed context."""
        return ExecutionContext(
            selector=DataSelector(
                partition="all",
                processing=[["raw"]] * dataset.features_sources(),
            ),
            state=PipelineState(
                y_processing="numeric",
                mode="train",
            ),
        )

    def run(self, dataset: SpectroDataset) -> ExecutionContext:
        """Run pipeline with typed context."""
        context = self._initialize_context(dataset)

        for step in self.steps:
            context, artifacts = self._execute_step(step, dataset, context)

        return context  # Return ExecutionContext directly
```

**3. All Controllers** (18+ files):
```python
# Before (controllers/transforms/transformer.py)
def execute(self, step, operator, dataset, context, runner, ...):
    train_context = copy.deepcopy(context)
    train_context["partition"] = "train"
    X = dataset.x(train_context, layout="2d")

# After
def execute(self, step, operator, dataset, context: ExecutionContext, runner, ...):
    train_ctx = context.with_partition("train")
    X = dataset.x(train_ctx.selector, layout="2d")

    # Modify context directly (mutable)
    context.selector.processing[source_idx] = new_processing_names
    return context, artifacts
```

**4. Custom Controller Data**:
```python
# Custom controller storing state
def execute(self, step, operator, dataset, context: ExecutionContext, runner, ...):
    # Store custom configuration
    context.custom["my_controller_state"] = {
        "threshold": step.get("threshold", 0.5),
        "iterations": step.get("iterations", 10),
    }

    # Process...
    return context, artifacts

# Another controller reading custom data
def execute(self, step, operator, dataset, context: ExecutionContext, runner, ...):
    config = context.custom.get("my_controller_state", {})
    threshold = config.get("threshold", 0.3)
    # Use threshold...
```

**5. All Tests**:
```python
# Before
def test_transformer():
    context = {"partition": "train", "processing": [["raw"]]}
    result, artifacts = controller.execute(step, op, dataset, context, runner)

# After
def test_transformer():
    context = ExecutionContext(
        selector=DataSelector(partition="train", processing=[["raw"]]),
        state=PipelineState(),
    )
    result, artifacts = controller.execute(step, op, dataset, context, runner)
    assert result.selector.partition == "train"
```

**Migration Process**:
1. Run type checker to find all Dict[str, Any] context usage
2. Update signatures first (break compilation intentionally)
3. Fix all compile errors (176+ locations)
4. Run tests, fix failures
5. Commit atomically

**Deliverables**:
- [ ] Updated dataset.x() and .y() to accept DataSelector only
- [ ] Updated PipelineRunner to create/use ExecutionContext
- [ ] All 18+ controllers updated
- [ ] All tests updated
- [ ] Type checker passes
- [ ] All tests pass

### Phase 3: Validation (Week 3)

**Goal**: Ensure nothing broken, performance acceptable

**Testing Strategy**:
1. **Unit Tests**: All existing tests must pass
2. **Integration Tests**: Run all examples
   ```powershell
   cd C:\Workspace\ML\nirs4all\examples
   .\run.ps1 -l  # Run all examples with logging
   ```
3. **Performance**: Benchmark context creation/copy overhead
4. **Type Safety**: Verify mypy/pyright pass

**Performance Benchmarks**:
```python
def benchmark_context_operations():
    """Measure context overhead."""
    import timeit

    # Context creation
    def create_context():
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]] * 10),
            state=PipelineState(),
        )

    creation_time = timeit.timeit(create_context, number=10000)
    print(f"Context creation: {creation_time:.4f}s per 10k operations")

    # Context copy
    ctx = create_context()
    copy_time = timeit.timeit(lambda: ctx.copy(), number=10000)
    print(f"Context copy: {copy_time:.4f}s per 10k operations")

    # Acceptable: <10% overhead vs dict operations
```

**Deliverables**:
- [ ] All unit tests pass
- [ ] All integration tests (examples) pass
- [ ] Performance within 10% of old implementation
- [ ] Type checker fully satisfied
- [ ] Documentation updated

---

## Integration with Runner Refactoring

### Dependencies

**Context refactoring is a BLOCKER for runner refactoring because:**

1. PipelineOrchestrator needs typed context creation
2. StepRunner needs clear interface for context passing
3. Controllers need explicit communication protocol
4. Testing requires typed mocks

### Updated Runner Components

From `RUNNER_REFACTORING_PROPOSAL.md`, update ExecutionContext:

**Before** (insufficient):
```python
@dataclass(frozen=True)
class ExecutionContext:
    partition: str
    processing_chains: List[List[str]]
    y_processing: str
    step_number: int
```

**After** (with this proposal):
```python
# Use the comprehensive ExecutionContext from this proposal
from nirs4all.pipeline.context import ExecutionContext, DataSelector, PipelineState
```

### Integration Points

1. **PipelineOrchestrator**:
```python
class PipelineOrchestrator:
    def _create_context(self, dataset: SpectroDataset) -> ExecutionContext:
        """Create typed context with proper initialization."""
        return ExecutionContext(
            selector=DataSelector(
                partition="all",
                processing=[["raw"]] * dataset.n_sources,
            ),
            state=PipelineState(
                y_processing="numeric",
                mode="train",
            ),
        )
```

2. **StepRunner**:
```python
class StepRunner:
    def execute(
        self,
        step: Dict[str, Any],
        dataset: SpectroDataset,
        context: ExecutionContext,  # Typed parameter
    ) -> Tuple[ExecutionContext, List[Any]]:
        """Execute step with typed context."""
        controller = self.router.get_controller(step, context.metadata.keyword)

        # Pass typed context directly
        return controller.execute(step, operator, dataset, context, ...)
```

3. **ControllerRouter**:
```python
class ControllerRouter:
    def get_controller(
        self,
        step: Dict[str, Any],
        keyword: str
    ) -> OperatorController:
        """Match controller using metadata keyword."""
        # Use keyword from context.metadata
```

4. **Custom Controllers**:
```python
class MyCustomController(OperatorController):
    def execute(self, step, operator, dataset, context: ExecutionContext, runner, ...):
        """Execute with custom data propagation."""
        # Read custom config from previous controller
        prev_config = context.custom.get("upstream_controller", {})

        # Store own custom data for downstream controllers
        context.custom["my_controller"] = {
            "processed_features": ["f1", "f2", "f3"],
            "config": self.config,
        }

        # Standard execution
        context.selector.processing[0] = new_processing_names
        return context, artifacts
```

---

## Testing Strategy

### Unit Tests

**Context Classes** (50+ tests):
```python
def test_data_selector_immutability():
    """Verify DataSelector immutable updates."""
    selector = DataSelector(partition="train")
    new_selector = selector.with_partition("test")
    assert selector.partition == "train"
    assert new_selector.partition == "test"

def test_pipeline_state_mutability():
    """Verify PipelineState allows updates."""
    state = PipelineState(processing_chains=[["raw"]])
    state.update_processing(0, ["snv"])
    assert state.processing_chains[0] == ["snv"]

def test_execution_context_dict_interface():
    """Verify dict-like interface for backward compatibility."""
    ctx = ExecutionContext.from_dict({"partition": "train"})
    assert ctx["partition"] == "train"
    ctx["partition"] = "test"
    assert ctx.get_selector().partition == "test"

def test_execution_context_deep_copy():
    """Verify copy isolation."""
    ctx1 = ExecutionContext.from_dict({"partition": "train"})
    ctx2 = ctx1.copy()
    ctx2["partition"] = "test"
    assert ctx1["partition"] == "train"
    assert ctx2["partition"] == "test"
```

**Dataset Integration** (20+ tests):
```python
def test_dataset_x_with_data_selector():
    """Verify dataset.x() accepts DataSelector."""
    selector = DataSelector(partition="train", processing=[["raw"]])
    X = dataset.x(selector, layout="2d")

    # Compare with dict
    X_dict = dataset.x(selector.to_dict(), layout="2d")
    np.testing.assert_array_equal(X, X_dict)

def test_dataset_y_with_data_selector():
    """Verify dataset.y() accepts DataSelector."""
    selector = DataSelector(partition="train")
    y = dataset.y(selector)

    y_dict = dataset.y(selector.to_dict())
    np.testing.assert_array_equal(y, y_dict)
```

**Controller Migration** (18+ controller suites):
```python
def test_transformer_controller_with_typed_context():
    """Verify TransformerMixinController works with ExecutionContext."""
    ctx = ExecutionContext.from_dict({
        "partition": "all",
        "processing": [["raw"]],
    })

    controller = TransformerMixinController()
    result_ctx, artifacts = controller.execute(
        step, operator, dataset, ctx, runner
    )

    assert isinstance(result_ctx, ExecutionContext)
    assert len(result_ctx.state.processing_chains[0]) > 0
```

### Integration Tests

**Pipeline Execution** (existing tests continue to pass):
```python
def test_full_pipeline_with_typed_context():
    """Verify entire pipeline works with ExecutionContext."""
    runner = PipelineRunner(steps=[
        {"StandardScaler": StandardScaler()},
        {"PCA": PCA(n_components=10)},
        {"RandomForestClassifier": RandomForestClassifier()},
    ])

    result = runner.run(dataset)
    assert isinstance(result, dict)  # Backward compatibility
```

### Regression Tests

**Ensure No Breaking Changes**:
```python
def test_backward_compatibility_with_dict_context():
    """Verify existing dict context still works during migration."""
    # Existing code should work unchanged
    context = {
        "partition": "train",
        "processing": [["raw"]],
        "y": "numeric",
    }

    # Controllers accept dict during migration
    result_ctx, artifacts = controller.execute(
        step, operator, dataset, context, runner
    )

    assert isinstance(result_ctx, (dict, ExecutionContext))
```

---

## Performance Considerations

### Memory Impact

**Current**: Single dict, shallow copies common
**Proposed**: Three objects + custom dict, deep copies

**Mitigation**:
- Deep copy only when needed (controller isolation)
- Processing chains shared between selector and operations
- No adapter layer overhead

**Benchmarks** (to be measured in Phase 3):
```python
def benchmark_context_operations():
    """Measure context overhead."""
    # Context creation
    ctx = ExecutionContext(
        selector=DataSelector(partition="train", processing=[["raw"]] * 10),
        state=PipelineState(),
    )

    # Context copy
    ctx_copy = ctx.copy()

    # Acceptable: <10% overhead vs dict
```

### Optimization Opportunities

1. **Processing chain reuse**: Selector and operations share same list
2. **Custom dict lazy copy**: Only deep copy custom dict if modified
3. **Profile-guided optimization**: Measure real pipeline overhead

---

## Resolved Questions

**1. Custom controller data propagation?**
- ✅ **RESOLVED**: Use `context.custom` dict for controller-specific data
- Controllers store/retrieve via: `context.custom["controller_name"] = data`
- Enables arbitrary controller communication without polluting core context

**2. How to handle fold_id?**
- ✅ **RESOLVED**: Optional field in DataSelector
- Used by selector if exists (CV operations)
- Ignored if None (normal train/test operations)

**3. Backward compatibility strategy?**
- ✅ **RESOLVED**: NO backward compatibility
- Clean breaking change in one atomic migration
- Only public API signatures require attention
- Internal refactoring can break freely

**4. Processing chains in selector or state?**
- ✅ **RESOLVED**: Stay in DataSelector
- **Reason**: Future flow controllers with multiple processing paths
- **Reason**: Feature caching requires processing chains for data selection
- **Reason**: Processing identifies which cached variant to retrieve

---

## Summary

This context refactoring:

✅ **Separates three distinct concerns** (data selection, execution state, coordination)
✅ **Enables custom controller data** via extensible `context.custom` dict
✅ **Provides type safety** throughout pipeline
✅ **Unblocks runner refactoring** by providing clear interfaces
✅ **Improves testability** with explicit dependencies
✅ **Clean breaking change** - no backward compatibility burden
✅ **Processing in DataSelector** for future flow controllers and caching

**Critical Path**:
1. Implement new context module (Week 1)
2. Atomic migration of all code (Week 2)
3. Validation and testing (Week 3)
4. **THEN** proceed with runner refactoring

**Estimated Effort**: 3 weeks with one developer, 2 weeks with two developers

**Risk Assessment**: Medium-High - atomic migration of 176+ locations, but type checker guides us

**Key Advantages of Clean Break**:
- Faster implementation (3 weeks vs 8 weeks with BC)
- No technical debt from compatibility layers
- Clear before/after boundary
- Type checker catches all missed locations
- No gradual migration complexity

---

## Next Steps

1. [ ] Review and approve this proposal
2. [ ] Create GitHub issues for 3 phases
3. [ ] Implement context.py with all classes (Phase 1)
4. [ ] Write 50+ unit tests
5. [ ] Begin atomic migration (Phase 2)
6. [ ] Run full validation (Phase 3)

**Once approved, this proposal becomes the foundation for all pipeline modernization efforts.**
