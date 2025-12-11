# Reset Preprocessing Feature - Implementation Report

## Executive Summary

This report describes the design and implementation plan for a **preprocessing reset** mechanism in nirs4all. The feature allows users to define pipelines where preprocessing steps can be evaluated independently without cumulative effects, significantly improving simulation speed for preprocessing exploration studies.

---

## 1. Current State

### 1.1 Pipeline Execution Model

Currently, nirs4all pipelines execute steps **sequentially and cumulatively**:

```python
pipeline = [
    pp1,    # Preprocessing 1
    model,  # Model trained on pp1(data)
    pp2,    # Preprocessing 2 applied ON TOP of pp1
    model,  # Model trained on pp2(pp1(data))
]
```

**Data flow:**
```
raw_data → pp1 → X1 → model → predictions_1
                  ↓
                 pp2 → X2 → model → predictions_2
```

Where `X2 = pp2(pp1(raw_data))`, meaning **pp2 is applied to already-preprocessed data**.

### 1.2 Feature Augmentation Workaround

The current way to test multiple preprocessings independently is via `feature_augmentation`:

```python
pipeline = [
    {"feature_augmentation": [pp1, pp2, pp3]},  # Creates parallel processings
    model,  # Trained on all processings
]
```

This creates **parallel processing channels** but:
- All preprocessings run on the same raw data ✅
- All results are evaluated in a single model training ❌
- Cannot test different model configurations per preprocessing ❌
- Increases memory usage (all processings stored simultaneously) ❌

### 1.3 Current `_or_` Generator Behavior

The `_or_` syntax expands to **multiple separate pipelines**:

```python
pipeline = [
    {"_or_": [pp1, pp2]},  # 2 pipelines
    model,
]
# Expands to:
# Pipeline 1: [pp1, model]
# Pipeline 2: [pp2, model]
```

This works but requires **full dataset reload per pipeline**, causing:
- Repeated I/O for each pipeline
- Redundant cross-validation setup
- No benefit from shared computation

### 1.4 SpectroDataset Architecture

The `SpectroDataset` stores features in a 3D array structure:
- **Dimension 1**: Samples
- **Dimension 2**: Processings (raw, pp1, pp2, etc.)
- **Dimension 3**: Features

Key insight: **The raw data is always preserved** in the `Features` block. The `context.selector.processing` determines which processing is used for model training.

Relevant classes:
- `Features` ([features.py](../../nirs4all/data/features.py)): Multi-source feature management
- `FeatureSource` ([feature_source.py](../../nirs4all/data/_features/feature_source.py)): Single source 3D array storage
- `ExecutionContext` ([context.py](../../nirs4all/pipeline/config/context.py)): Pipeline state with `DataSelector.processing`

---

## 2. Objective

Enable a pipeline syntax that allows **independent preprocessing evaluation** with **shared data and cross-validation**:

```python
pipeline = [
    splitter,       # Cross-validation split (done once)
    {
        "_reset_": [
            [pp1, model],    # Branch 1: raw → pp1 → model
            [pp2, model],    # Branch 2: raw → pp2 → model
            [pp3, model],    # Branch 3: raw → pp3 → model
        ]
    }
]
```

**Expected behavior:**
1. `splitter` creates folds **once** (shared across all branches)
2. Each branch starts from **original raw data** (processing = ["raw"])
3. Each branch executes independently, producing separate predictions
4. Dataset is loaded **once**, not N times

### 2.1 Target Use Cases

**Use Case 1: Preprocessing Comparison**
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"y_processing": StandardScaler()},
    {
        "_reset_": [
            [SNV(), PLSRegression(n_components=10)],
            [FirstDerivative(), PLSRegression(n_components=10)],
            [MSC(), PLSRegression(n_components=10)],
        ]
    }
]
```

**Use Case 2: Preprocessing + Model Exploration**
```python
pipeline = [
    GroupKFold(n_splits=5),
    {
        "_reset_": [
            [SNV(), PLSRegression(n_components=5)],
            [SNV(), PLSRegression(n_components=10)],
            [FirstDerivative(), RandomForest()],
        ]
    }
]
```

**Use Case 3: Combined with Generators**
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {
        "_reset_": {
            "_or_": [pp1, pp2, pp3],  # Generate 3 branches
        }
    },
    {
        "_range_": [5, 15, 5],
        "param": "n_components",
        "model": PLSRegression
    }
]
```

### 2.2 Performance Goals

| Scenario | Current (N pipelines) | With `_reset_` |
|----------|----------------------|----------------|
| 10 preprocessings × 1 model | 10× dataset load, 10× split | 1× load, 1× split |
| 5 preprocessings × 3 CV folds | 5× full execution | 1× split, 5× branch |
| Memory for 1000-feature data | 10× peak memory | 1× + lightweight branches |

---

## 3. Proposed Solution

### 3.1 New Keyword: `_reset_`

Introduce a `_reset_` keyword that defines **parallel branches** starting from a **snapshot** of the current dataset/context state.

**Syntax Options:**

```python
# Option A: Explicit list of branches
{"_reset_": [
    [step1, step2],  # Branch 1
    [step3, step4],  # Branch 2
]}

# Option B: Generator-compatible
{"_reset_": {"_or_": [pp1, pp2, pp3]}}  # Each becomes a branch

# Option C: Named branches (for tracking)
{"_reset_": {
    "snv_branch": [SNV(), model],
    "d1_branch": [FirstDerivative(), model],
}}
```

### 3.2 Architecture Design

#### 3.2.1 ResetController

A new controller that:
1. **Captures** the current processing state (snapshot)
2. **Iterates** over branches
3. **Restores** processing state before each branch
4. **Executes** branch steps via `step_runner`
5. **Collects** artifacts from all branches

```python
@register_controller
class ResetController(OperatorController):
    priority = 5  # Execute before transformers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "_reset_" or (
            isinstance(step, dict) and "_reset_" in step
        )

    def execute(
        self,
        step_info: ParsedStep,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        ...
    ) -> Tuple[ExecutionContext, List]:
        # 1. Snapshot current state
        snapshot = self._create_snapshot(dataset, context)

        # 2. Get branches
        branches = self._expand_branches(step_info.operator)

        all_artifacts = []

        for branch_idx, branch_steps in enumerate(branches):
            # 3. Restore snapshot
            self._restore_snapshot(snapshot, dataset, context)
            branch_context = context.copy()

            # 4. Execute branch steps
            for step in branch_steps:
                result = runtime_context.step_runner.execute(
                    step, dataset, branch_context, runtime_context,
                    loaded_binaries=None, prediction_store=prediction_store
                )
                branch_context = result.updated_context
                all_artifacts.extend(result.artifacts)

        return context, all_artifacts
```

#### 3.2.2 Snapshot Strategy

Two approaches for state restoration:

**Approach A: Processing Selector Reset (Lightweight)**
```python
def _create_snapshot(self, dataset, context):
    return {
        'processing': deepcopy(context.selector.processing),
        'y_processing': context.state.y_processing,
    }

def _restore_snapshot(self, snapshot, dataset, context):
    context.selector.processing = deepcopy(snapshot['processing'])
    context.state.y_processing = snapshot['y_processing']
```

*Pros:* Fast, no data copying
*Cons:* Requires preprocessings to support "add" mode (which they do via `add_feature=True`)

**Approach B: Feature Block Clone (Full Isolation)**
```python
def _create_snapshot(self, dataset, context):
    return {
        'features': dataset._features.deep_copy(),  # New method needed
        'context': context.copy(),
    }

def _restore_snapshot(self, snapshot, dataset, context):
    dataset._features = snapshot['features'].deep_copy()
    return snapshot['context'].copy()
```

*Pros:* Complete isolation between branches
*Cons:* Memory overhead for large datasets

**Recommendation:** Start with **Approach A** (selector reset) as it leverages existing architecture. The current `FeatureSource` already preserves raw data and supports multiple processings. Branches would add new processings without affecting previous branches.

#### 3.2.3 Integration with Generator Syntax

The `_reset_` keyword should integrate with existing generators:

```python
# Input
{"_reset_": {"_or_": [pp1, pp2, pp3]}}

# Internal expansion (by PipelineConfigs or ResetController)
{"_reset_": [[pp1], [pp2], [pp3]]}
```

This requires either:
1. **Pre-expansion** in `PipelineConfigs` (like `_or_` at top level)
2. **Runtime expansion** in `ResetController` (call `expand_spec` internally)

**Recommendation:** Runtime expansion in the controller to keep generator logic centralized.

### 3.3 Context Flow Design

```
                    ┌─────────────────────────────┐
                    │       Initial Context       │
                    │  processing: [["raw"]]      │
                    │  y_processing: "numeric"    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │         Splitter            │
                    │  (creates folds, shared)    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │       _reset_ Entry         │
                    │   snapshot = current state  │
                    └──┬──────────┬───────────┬───┘
                       │          │           │
            ┌──────────▼──┐  ┌────▼────┐  ┌───▼──────────┐
            │  Branch 1   │  │ Branch 2│  │   Branch 3   │
            │ restore     │  │ restore │  │   restore    │
            │ pp1→model   │  │ pp2→mod │  │   pp3→model  │
            └──────────────┘  └─────────┘  └──────────────┘
                       │          │           │
                       ▼          ▼           ▼
                 predictions  predictions  predictions
                    (1)          (2)          (3)
```

### 3.4 Prediction Tracking

Each branch should produce distinct predictions with proper identification:

```python
prediction = {
    "pipeline_uid": "abc123",
    "branch_id": "branch_0",  # New field
    "branch_name": "SNV",     # Optional: from named branches
    "preprocessings": ["raw_SNV_001"],
    "metrics": {...},
    ...
}
```

---

## 4. Implementation Roadmap

### Phase 1: Core Reset Controller (Week 1-2)

**Tasks:**
1. Create `ResetController` in `nirs4all/controllers/control/reset.py`
2. Implement selector-based snapshot/restore
3. Add `_reset_` keyword detection in parser
4. Basic branch iteration with step execution
5. Unit tests for single-branch and multi-branch execution

**Deliverables:**
- Working `_reset_` with explicit branch lists
- Tests covering basic reset behavior

### Phase 2: Generator Integration (Week 2-3)

**Tasks:**
1. Add `_or_` expansion support within `_reset_`
2. Add `_range_` expansion for parameterized branches
3. Handle nested generators (e.g., `_reset_` containing `_or_` containing `_range_`)
4. Update serialization for reset steps

**Deliverables:**
- Full generator syntax support within reset blocks
- Serialization/deserialization working

### Phase 3: Prediction Tracking (Week 3-4)

**Tasks:**
1. Add `branch_id` and `branch_name` to prediction metadata
2. Update prediction aggregation for reset branches
3. Visualization support for branch comparison
4. Documentation and examples

**Deliverables:**
- Complete branch tracking in predictions
- Example notebooks demonstrating usage

### Phase 4: Optimization & Edge Cases (Week 4-5)

**Tasks:**
1. Memory optimization for large datasets
2. Parallel branch execution (optional)
3. Nested reset handling (reset within reset)
4. Error handling and recovery within branches

**Deliverables:**
- Performance benchmarks
- Edge case handling
- Full documentation

---

## 5. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `nirs4all/controllers/control/reset.py` | **New** | ResetController implementation |
| `nirs4all/controllers/registry.py` | Modify | Register ResetController |
| `nirs4all/pipeline/steps/parser.py` | Modify | Detect `_reset_` keyword |
| `nirs4all/pipeline/config/generator.py` | Modify | Add `_reset_` expansion support |
| `nirs4all/pipeline/config/keywords.py` | Modify | Add `RESET_KEYWORD` constant |
| `nirs4all/data/predictions.py` | Modify | Add branch tracking fields |
| `docs/reference/writing_pipelines.md` | Modify | Document `_reset_` syntax |
| `examples/QXX_reset_preprocessing.py` | **New** | Usage examples |
| `tests/unit/controllers/test_reset.py` | **New** | Unit tests |
| `tests/integration/test_reset_pipeline.py` | **New** | Integration tests |

---

## 6. Alternatives Considered

### 6.1 Extend `feature_augmentation`

Modify feature augmentation to support per-branch model training:
```python
{"feature_augmentation": [pp1, pp2], "per_branch_model": True}
```

**Rejected because:**
- Conflates feature augmentation (parallel channels) with branch execution
- Would require significant changes to model controllers
- Less intuitive than explicit `_reset_`

### 6.2 Pipeline Cloning at Orchestrator Level

Clone dataset before each `_or_` branch at the orchestrator level.

**Rejected because:**
- Loses shared fold information
- Higher memory usage
- No explicit user control over reset points

### 6.3 Lazy Preprocessing Execution

Defer preprocessing until model training, re-applying from raw each time.

**Rejected because:**
- Breaks existing architecture assumptions
- Would require major refactoring
- Performance penalty for repeated preprocessing

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Memory growth with many branches | Medium | High | Implement lazy cleanup between branches |
| Artifact naming conflicts | Low | Medium | Include branch_id in artifact names |
| Context leakage between branches | Medium | High | Deep copy context at each branch start |
| Generator edge cases | Medium | Medium | Comprehensive test coverage |
| Breaking existing pipelines | Low | High | New keyword, no changes to existing syntax |

---

## 8. Success Criteria

1. ✅ `_reset_` syntax works with explicit branch lists
2. ✅ `_reset_` + `_or_` generates correct number of branches
3. ✅ Folds are shared across all branches (single split)
4. ✅ Each branch produces independent predictions
5. ✅ Memory usage is O(1) per branch (not O(N))
6. ✅ Performance improvement > 50% vs separate pipeline runs
7. ✅ Full backward compatibility with existing pipelines

---

## 9. References

- [PipelineExecutor](../../nirs4all/pipeline/execution/executor.py): Step execution flow
- [ExecutionContext](../../nirs4all/pipeline/config/context.py): Context management
- [FeatureAugmentationController](../../nirs4all/controllers/data/feature_augmentation.py): Similar pattern for branch execution
- [TransformerController](../../nirs4all/controllers/transforms/transformer.py): Processing update logic
- [Generator Documentation](../../nirs4all/pipeline/config/generator.py): `_or_` and `_range_` expansion

---

**Author:** Senior Python/ML Developer
**Date:** December 2025
**Status:** Proposal - Pending Review
