# Pipeline Branching Feature - Specification Document

## Executive Summary

This document specifies a **pipeline branching** mechanism for nirs4all. The branching feature enables users to split a pipeline into multiple parallel sub-pipelines ("branches"), each with its own preprocessing context (X transformations, Y processing), while sharing common upstream state (splits, initial preprocessing). Steps declared after the branching block execute on each branch independently, enabling efficient multi-configuration experimentation.

---

## 1. Objectives

### 1.1 Core Requirements

The branching feature addresses the following primary requirements:

#### R1: Parallel Preprocessing Contexts
Allow multiple preprocessing chains to be evaluated independently within a single pipeline execution. Each branch maintains its own:
- **X processing chain**: Transformations applied to features (e.g., SNV, PCA, MSC)
- **Y processing state**: Target scaling/encoding per branch (e.g., StandardScaler, MinMaxScaler)
- **Feature state**: Each branch can have different feature dimensions (e.g., after PCA)

#### R2: Shared Upstream State
Branches should inherit and share:
- **Split configuration**: CV folds are computed once and shared across all branches
- **Upstream preprocessing**: Any transformations applied before branching
- **Dataset reference**: Single data load, no redundant I/O

#### R3: Post-Branch Step Execution
Steps declared after a `branch` block should execute on **each branch independently**:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    MinMaxScaler(),  # Applied once, shared by all branches
    {"branch": [
        [SNV(), PCA(n_components=10)],  # Branch A
        [FirstDerivative(), MSC()],      # Branch B
    ]},
    PLSRegression(n_components=5),  # Executed on Branch A and Branch B
]
```

The `PLSRegression` model trains twice: once on `MinMax→SNV→PCA` features, once on `MinMax→D1→MSC` features.

#### R4: In-Branch Model Training
Models declared **inside** a branch execute only for that branch:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": [
        [SNV(), {"y_processing": StandardScaler()}, TabPFN()],  # TabPFN on SNV data with scaled y
        [MSC(), Detrend(), PLSRegression(n_components=10)],     # PLS on MSC+Detrend data
    ]},
    {"finetune": PLSRegression()},  # Finetuned on BOTH branches
]
```

### 1.2 Target Use Cases

**Use Case 1: Preprocessing Comparison**
Compare multiple preprocessing strategies with shared model configuration:
```python
pipeline = [
    GroupKFold(n_splits=5),
    {"y_processing": StandardScaler()},
    {"branch": [
        [SNV()],
        [MSC()],
        [FirstDerivative()],
        [SecondDerivative()],
    ]},
    PLSRegression(n_components=10),  # Same model on each preprocessing
]
# Result: 4 prediction sets (one per preprocessing × 5 folds each)
```

**Use Case 2: Preprocessing + Model Exploration**
Different preprocessing-model combinations:
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [
        [SNV(), {"y_processing": StandardScaler()}, PLSRegression(n_components=5)],
        [SNV(), PLSRegression(n_components=10)],
        [FirstDerivative(), RandomForestRegressor()],
    ]},
]
# Result: 3 prediction sets with different preprocessing+model combos
```

**Use Case 3: Branches with Post-Branch Finetuning**
Train a base model per branch, then finetune a meta-learner:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": [
        [SavitzkyGolay(), PCA(n_components=15), {"y_processing": StandardScaler()}, TabPFN()],
        [EMSC(), Detrend(), SNV()],
    ]},
    {"finetune": PLSRegression()},  # Finetuned on each branch's output
]
```

**Use Case 4: Combining with Generators**
Use `_or_` and `_range_` generators inside branches:
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}},  # 3 branches from generator
    {"_range_": [5, 15, 5], "param": "n_components", "model": PLSRegression},  # 3 PLS variants per branch
]
# Result: 9 configurations (3 preprocessings × 3 n_components)
```

### 1.3 Extension Points (Future Phases)

The following extensions are out of scope for Phase 1 but inform the architecture:

#### E1: Feature-Based Branching (Spectral Regions)
Split features (spectral range) into branches:
```python
pipeline = [
    {"branch": {
        "by": "features",
        "ranges": [(500, 950), (950, 1700), (1700, 2500)],  # nm ranges
    }},
    PLSRegression(),  # One model per spectral region
]
```

#### E2: Sample-Based Branching
Split samples into branches (e.g., by group or condition):
```python
pipeline = [
    {"branch": {
        "by": "samples",
        "filter": {"group": [1, 2, 3]},  # One branch per group
    }},
    PLSRegression(),  # One model per sample group
]
```

#### E2b: Sample-Based Branching with Outlier Excluders
Use outlier detection to create branches with different sample exclusion strategies:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,  # Branch 0: No exclusion (baseline)
            {"method": "isolation_forest", "contamination": 0.05},  # Branch 1: IF 5%
            {"method": "isolation_forest", "contamination": 0.10},  # Branch 2: IF 10%
            {"method": "mahalanobis", "threshold": 3.0},  # Branch 3: Mahalanobis
            {"method": "leverage", "threshold": 2.0},  # Branch 4: High leverage points
        ],
    }},
    PLSRegression(n_components=10),  # Train on filtered samples per branch
]
# Result: 5 branches × 5 folds = 25 prediction sets
```

**Outlier excluder strategies**:

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `None` | No exclusion (baseline) | - |
| `isolation_forest` | Isolation Forest anomaly detection | `contamination`: float (0.01-0.5) |
| `mahalanobis` | Mahalanobis distance from centroid | `threshold`: float (z-score) |
| `leverage` | High leverage points (hat matrix diagonal) | `threshold`: float (multiplier of mean leverage) |
| `lof` | Local Outlier Factor | `contamination`: float, `n_neighbors`: int |
| `residual` | Q-residuals or Hotelling's T² | `method`: "q" or "t2", `threshold`: float |
| `custom` | User-defined excluder function | `func`: callable(X, y) → mask |

**Behavior notes**:
- Outlier detection is performed **after** shared preprocessing (before branch-specific transforms)
- Each branch maintains an **exclusion mask** that filters samples during training
- The exclusion is applied to training data only; validation/test sets remain complete for fair comparison
- Predictions from excluded samples are marked with `excluded=True` metadata
- Branch 0 with `None` strategy serves as baseline for comparison

**Combined with preprocessing branches**:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    MinMaxScaler(),  # Shared preprocessing
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [None, {"method": "isolation_forest", "contamination": 0.05}],
    }},
    # Nested preprocessing branches (see E5: Nested Branches)
    {"branch": [[SNV()], [MSC()], [FirstDerivative()]]},
    PLSRegression(n_components=10),
]
# Result: 2 outlier strategies × 3 preprocessings × 5 folds = 30 prediction sets
```

#### E3: Independent Splits per Branch
Allow each branch to have its own CV configuration:
```python
pipeline = [
    {"branch": [
        [GroupKFold(n_splits=5), SNV(), PLSRegression()],
        [ShuffleSplit(n_splits=3), MSC(), RandomForestRegressor()],
    ]},
]
```

#### E4: Branch Merging / Concatenation
Merge branch outputs for ensemble or stacking:
```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression()],
        [MSC(), PLSRegression()],
    ]},
    {"merge_branches": "concat"},  # Concatenate predictions
    StackingRegressor(),
]
```

#### E5: Nested Branches
Branches can be nested to create hierarchical experiment designs:
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {  # Level 1: Outlier handling
        "by": "outlier_excluder",
        "strategies": [None, {"method": "isolation_forest", "contamination": 0.05}],
    }},
    {"branch": [  # Level 2: Preprocessing
        [SNV(), PCA(n_components=10)],
        [MSC(), FirstDerivative()],
    ]},
    PLSRegression(n_components=5),  # Executed on all 4 branch combinations
]
# Result: 2 outlier × 2 preprocessing × 3 folds = 12 prediction sets
```

See [Section 3.5: Nested Branches](#35-nested-branches-design-and-implications) for detailed design considerations.

---

## 2. Current State

### 2.1 Existing Branching-Related Capabilities

#### 2.1.1 Indexer Branch Field
The `Indexer` class already has a **`branch` field** (Int8) in its schema:

```python
# From index_store.py
self._df = pl.DataFrame({
    "row": pl.Series([], dtype=pl.Int32),
    "sample": pl.Series([], dtype=pl.Int32),
    "branch": pl.Series([], dtype=pl.Int8),  # Already exists!
    ...
})
```

The `branch` field is:
- Stored as Int8 (supports up to 127 branches)
- Filterable via selectors: `{"branch": 0}` or `{"branch": [0, 1]}`
- Used in `add_samples()` and `add_rows()` methods
- Referenced in `ParameterNormalizer.convert_indexdict_to_params()`

**Current status**: The field exists but is **not actively used** in pipeline execution. All samples default to `branch=None` or `branch=0`.

#### 2.1.2 Feature Augmentation Controller
The `FeatureAugmentationController` implements a **parallel processing pattern**:

```python
{"feature_augmentation": [SNV(), MSC(), FirstDerivative()]}
```

Key behaviors:
- Iterates over each operation sequentially
- Each operation starts from a **fresh copy** of the original processing state
- Sets `add_feature=True` metadata to add new processing channels
- All channels are stored in the same `FeatureSource` (3D array)

**Differences from branching**:
- Feature augmentation creates **parallel feature channels** for a single model
- Branching creates **independent execution paths** for different models
- Feature augmentation doesn't support per-channel Y processing
- Feature augmentation doesn't support per-channel model training

#### 2.1.3 Generator System (`_or_`, `_range_`)
The generator system expands specs into multiple pipeline configurations:

```python
{"_or_": [SNV(), MSC()]}  # Expands to 2 separate pipelines
```

Key behaviors:
- Expansion happens at **configuration time** (before execution)
- Each variant becomes a **completely separate pipeline**
- Full dataset reload per pipeline
- No shared state between variants

**Differences from branching**:
- Generators create separate pipelines (full reload per variant)
- Branching creates shared-state sub-pipelines (single load)
- Generators are expanded before execution
- Branches are executed at runtime

#### 2.1.4 Execution Context Architecture
The `ExecutionContext` provides the infrastructure for branching:

```python
@dataclass
class DataSelector:
    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [["raw"]])
    branch: Optional[int] = None  # NOT YET PRESENT - needs to be added

@dataclass
class PipelineState:
    y_processing: str = "numeric"
    step_number: int = 0
    mode: str = "train"
```

**Key insight**: The context is already designed for copying and isolation:
- `context.copy()` creates deep copies
- `context.with_processing()` creates new context with updated processing
- Controllers can modify context without affecting other branches

### 2.2 Gap Analysis

| Capability | Current State | Required for Branching |
|------------|--------------|----------------------|
| Branch field in Indexer | ✅ Exists (unused) | ✅ Ready to use |
| Branch field in DataSelector | ❌ Not present | ⚠️ Needs addition |
| Parallel execution within pipeline | ⚠️ Only feature_augmentation | ⚠️ Needs BranchController |
| Independent Y processing per branch | ❌ Not supported | ⚠️ Needs context isolation |
| Post-branch step iteration | ❌ Not supported | ⚠️ Needs BranchController |
| Branch-aware predictions | ❌ Not present | ⚠️ Needs prediction metadata |
| Shared splits across branches | ⚠️ Implicit via dataset | ✅ Already works |

---

## 3. Implementation Proposals

### 3.1 Keyword and Syntax Design

#### 3.1.1 Primary Syntax: `branch` Keyword

```python
{"branch": [
    [step1, step2, ...],  # Branch 0
    [step3, step4, ...],  # Branch 1
    ...
]}
```

Each branch is a list of steps that execute sequentially within that branch.

#### 3.1.2 Named Branches (Optional Enhancement)

```python
{"branch": {
    "snv_pca": [SNV(), PCA(n_components=10)],
    "msc_d1": [MSC(), FirstDerivative()],
}}
```

Named branches provide better tracking in predictions and visualizations.

#### 3.1.3 Generator-Compatible Syntax

```python
# Implicit branches from _or_
{"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}}
# Expands to 3 branches: [[SNV()]], [[MSC()]], [[FirstDerivative()]]

# With nested steps
{"branch": {"_or_": [
    [SNV(), PCA()],
    [MSC(), {"y_processing": StandardScaler()}],
]}}
```

### 3.2 Architecture Design

#### 3.2.1 BranchController

A new controller that manages branch execution:

```python
@register_controller
class BranchController(OperatorController):
    """Controller for pipeline branching.

    Manages parallel sub-pipelines with independent preprocessing contexts.
    Subsequent steps in the parent pipeline execute on each branch.
    """
    priority = 5  # Before transformers, after splitters

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "branch" or (
            isinstance(step, dict) and "branch" in step
        )

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(
        self,
        step_info: ParsedStep,
        dataset: SpectroDataset,
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[ExecutionContext, List]:
        """Execute branch step.

        1. Snapshot current context state
        2. Expand branch specifications (handle _or_, named branches)
        3. For each branch:
           a. Restore context to snapshot
           b. Assign branch ID to context
           c. Execute branch steps via step_runner
           d. Store branch context for post-branch steps
        4. Return list of branch contexts for continuation
        """
        ...
```

#### 3.2.2 Context Extension

Add branch tracking to `DataSelector`:

```python
@dataclass
class DataSelector(MutableMapping):
    partition: str = "all"
    processing: List[List[str]] = field(default_factory=lambda: [["raw"]])
    layout: str = "2d"
    concat_source: bool = True
    fold_id: Optional[int] = None
    include_augmented: bool = False
    y: Optional[str] = None
    branch_id: Optional[int] = None  # NEW: Current branch ID
    branch_name: Optional[str] = None  # NEW: Optional branch name
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)
```

Add branch state tracking to `ExecutionContext.custom`:

```python
context.custom["branch_contexts"] = [
    {"branch_id": 0, "name": "snv_pca", "context": context_0},
    {"branch_id": 1, "name": "msc_d1", "context": context_1},
]
```

#### 3.2.3 Execution Flow

**Scenario: Post-branch step execution**

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [[SNV()], [MSC()]]},
    PLSRegression(n_components=10),  # Should run on both branches
]
```

**Execution flow:**

```
                    ┌─────────────────────────────┐
                    │  ShuffleSplit (creates folds)│
                    │  context.selector unchanged  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      BranchController        │
                    │  snapshot = context.copy()   │
                    └──────┬──────────────┬───────┘
                           │              │
              ┌────────────▼────┐   ┌─────▼────────────┐
              │   Branch 0      │   │    Branch 1      │
              │ restore snapshot│   │ restore snapshot │
              │ branch_id = 0   │   │ branch_id = 1    │
              │ exec: SNV()     │   │ exec: MSC()      │
              │ → ctx_0         │   │ → ctx_1          │
              └────────────────┘   └──────────────────┘
                           │              │
                    ┌──────▼──────────────▼───────┐
                    │   Store branch_contexts      │
                    │   in context.custom          │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  PipelineExecutor._execute_steps│
                    │  Detects branch_contexts     │
                    │  Iterates PLSRegression on   │
                    │  ctx_0 and ctx_1             │
                    └─────────────────────────────┘
```

**Key design decisions:**

1. **BranchController stores branch contexts** in `context.custom["branch_contexts"]`
2. **Post-branch steps iterate over branch contexts** automatically
3. **Each branch context has independent**:
   - `selector.processing` (the X preprocessing chain)
   - `state.y_processing` (the Y transformation)
   - `selector.branch_id` (for indexer filtering)
4. **Predictions include branch metadata** for tracking

#### 3.2.4 Step Execution Modification

The `PipelineExecutor._execute_steps()` method needs modification to handle branch iteration:

```python
def _execute_steps(self, steps, dataset, context, runtime_context, ...):
    for step in steps:
        # Check if we have active branches
        branch_contexts = context.custom.get("branch_contexts")

        if branch_contexts and not self._is_branch_step(step):
            # Execute step on each branch context
            for branch_ctx_info in branch_contexts:
                branch_context = branch_ctx_info["context"]
                result = self.step_runner.execute(
                    step, dataset, branch_context, runtime_context, ...
                )
                # Update the branch context in-place
                branch_ctx_info["context"] = result.updated_context
                all_artifacts.extend(result.artifacts)

            # Don't update main context processing, branches are independent
            continue

        # Normal step execution (non-branched or branch step itself)
        result = self.step_runner.execute(step, dataset, context, runtime_context, ...)
        context = result.updated_context
        all_artifacts.extend(result.artifacts)

    return context
```

#### 3.2.5 Prediction Tracking

Extend prediction metadata to include branch information:

```python
prediction = {
    "id": "abc123",
    "model_name": "PLSRegression",
    "dataset_name": "sample_data",
    "fold_id": 0,
    "branch_id": 0,  # NEW
    "branch_name": "snv_pca",  # NEW (optional)
    "preprocessings": ["raw_SNV_001"],
    "y_processing": "scaled_StandardScaler_001",
    "val_rmse": 0.123,
    "test_rmse": 0.145,
    ...
}
```

### 3.3 Serialization, Saving, and Loading

Branching introduces complexity for artifact persistence and prediction mode. This section details how to handle serialization across the full pipeline lifecycle.

#### 3.3.1 Current Serialization Architecture

The existing serialization system consists of:

| Component | Location | Role |
|-----------|----------|------|
| `ManifestManager` | `pipeline/storage/manifest_manager.py` | Saves YAML manifests with pipeline config, artifacts list, predictions |
| `ArtifactManager` | `pipeline/storage/artifacts/manager.py` | Persists binary artifacts (models, transformers) |
| `BinaryLoader` | `pipeline/storage/artifacts/binary_loader.py` | Loads artifacts by step number for predict/explain mode |
| `SimulationSaver` | `pipeline/storage/io.py` | Manages run directory structure, file I/O |
| `serialize_component()` | `pipeline/config/component_serialization.py` | Converts Python objects to JSON-serializable dicts |

**Manifest structure** (per pipeline):
```yaml
uid: "abc123-def456"
pipeline_id: "0001_pls_abc123"
name: "PLSRegression"
dataset: "sample_data"
created_at: "2025-12-11T10:30:00Z"
pipeline:
  steps:
    - class: "sklearn.preprocessing.MinMaxScaler"
    - class: "sklearn.cross_decomposition.PLSRegression"
      params: {n_components: 10}
generator_choices: []
artifacts:
  - name: "MinMaxScaler_1"
    step: 1
    format: "sklearn"
    path: "ab/abc123def.pkl"
    content_hash: "abc123def..."
  - name: "PLSRegression_2"
    step: 2
    format: "sklearn"
    path: "cd/cdef456abc.pkl"
    content_hash: "cdef456abc..."
predictions:
  - id: "pred_001"
    model_name: "PLSRegression"
    fold_id: 0
    val_rmse: 0.123
```

**Artifact loading flow** (predict mode):
```
PipelineRunner.predict()
  → Predictor._prepare_replay()
    → ManifestManager.load_manifest(pipeline_uid)
    → BinaryLoader.from_manifest(manifest, results_dir)
  → PipelineExecutor.execute()
    → For each step:
      → binary_loader.get_step_binaries(step_number)
      → Controller uses loaded_binaries instead of fitting
```

#### 3.3.2 Branch-Aware Artifact Naming

**Problem**: Without branch identification, artifacts from different branches would have name collisions.

**Current naming**: `{OperatorName}_{operation_count}` (e.g., `PLSRegression_1`)

**Branch-aware naming**: `{OperatorName}_{branch_id}_{operation_count}` (e.g., `PLSRegression_0_1`, `PLSRegression_1_1`)

**Implementation in controllers**:

```python
# In TransformerMixinController.execute()
def execute(self, step_info, dataset, context, runtime_context, ...):
    # Get branch_id from context (None if not in branch)
    branch_id = context.selector.branch_id

    # Build artifact name with optional branch prefix
    if branch_id is not None:
        artifact_name = f"{operator_name}_b{branch_id}_{runtime_context.next_op()}"
    else:
        artifact_name = f"{operator_name}_{runtime_context.next_op()}"

    # Persist with branch-aware name
    artifact = runtime_context.saver.persist_artifact(
        step_number=runtime_context.step_number,
        name=artifact_name,
        obj=transformer,
        format_hint='sklearn'
    )
```

**Manifest artifact entries with branching**:
```yaml
artifacts:
  # Pre-branch step (shared)
  - name: "MinMaxScaler_1"
    step: 1
    branch_id: null  # NEW: null means pre-branch
    format: "sklearn"
    path: "ab/abc123.pkl"

  # Branch 0 artifacts
  - name: "SNV_b0_1"
    step: 2
    branch_id: 0  # NEW
    branch_name: "snv_pca"  # NEW (optional)
    format: "sklearn"
    path: "cd/cdef456.pkl"
  - name: "PCA_b0_2"
    step: 2
    branch_id: 0
    format: "sklearn"
    path: "ef/efgh789.pkl"

  # Branch 1 artifacts
  - name: "MSC_b1_1"
    step: 2
    branch_id: 1
    branch_name: "msc_d1"
    format: "sklearn"
    path: "gh/ghij012.pkl"
```

#### 3.3.3 Pipeline JSON Serialization

The `pipeline.json` file must preserve branch structure for replay:

```json
{
  "steps": [
    {"class": "sklearn.model_selection.ShuffleSplit", "params": {"n_splits": 3}},
    {"class": "sklearn.preprocessing.MinMaxScaler"},
    {
      "branch": [
        [
          {"class": "nirs4all.operators.transforms.SNV"},
          {"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}
        ],
        [
          {"class": "nirs4all.operators.transforms.MSC"},
          {"class": "nirs4all.operators.transforms.FirstDerivative"}
        ]
      ]
    },
    {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 5}}
  ]
}
```

**Named branches serialization**:
```json
{
  "branch": {
    "snv_pca": [
      {"class": "nirs4all.operators.transforms.SNV"},
      {"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}
    ],
    "msc_d1": [
      {"class": "nirs4all.operators.transforms.MSC"},
      {"class": "nirs4all.operators.transforms.FirstDerivative"}
    ]
  }
}
```

#### 3.3.4 BinaryLoader Extension for Branches

The `BinaryLoader` must support loading artifacts by both step AND branch:

```python
class BinaryLoader:
    def __init__(self, artifacts: List[Dict[str, Any]], results_dir: Path):
        self.results_dir = Path(results_dir)
        self.artifacts_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self.artifacts_by_step_branch: Dict[Tuple[int, Optional[int]], List[Dict[str, Any]]] = {}  # NEW
        self._cache: Dict[str, Any] = {}

        for artifact in artifacts:
            step = artifact.get("step", -1)
            branch_id = artifact.get("branch_id")  # NEW: may be None

            # Group by step (backward compatible)
            if step not in self.artifacts_by_step:
                self.artifacts_by_step[step] = []
            self.artifacts_by_step[step].append(artifact)

            # Group by (step, branch_id) for branch-aware loading
            key = (step, branch_id)
            if key not in self.artifacts_by_step_branch:
                self.artifacts_by_step_branch[key] = []
            self.artifacts_by_step_branch[key].append(artifact)

    def get_step_binaries(
        self,
        step_id: int,
        branch_id: Optional[int] = None  # NEW parameter
    ) -> List[Tuple[str, Any]]:
        """Load binary artifacts for a specific step and optional branch.

        Args:
            step_id: Step number
            branch_id: Branch ID (None for pre-branch steps or all branches)

        Returns:
            List of (name, loaded_object) tuples
        """
        # Build cache key
        cache_key = f"step_{step_id}_branch_{branch_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try branch-specific first, then fall back to step-only
        key = (step_id, branch_id)
        if key in self.artifacts_by_step_branch:
            artifacts = self.artifacts_by_step_branch[key]
        elif branch_id is None and step_id in self.artifacts_by_step:
            # Pre-branch step: get all artifacts for this step
            artifacts = self.artifacts_by_step[step_id]
        else:
            # Fallback: try getting branch_id=None artifacts (pre-branch)
            key = (step_id, None)
            artifacts = self.artifacts_by_step_branch.get(key, [])

        loaded_binaries = []
        for artifact in artifacts:
            obj = self._load_artifact(artifact)
            loaded_binaries.append((artifact.get("name"), obj))

        self._cache[cache_key] = loaded_binaries
        return loaded_binaries
```

#### 3.3.5 Predict Mode Execution with Branches

The prediction flow must reconstruct branch contexts:

```python
# In PipelineExecutor._execute_steps() during predict mode
def _execute_steps(self, steps, dataset, context, runtime_context, ...):
    for step in steps:
        branch_contexts = context.custom.get("branch_contexts")

        if branch_contexts and not self._is_branch_step(step):
            # Execute on each branch with branch-specific binaries
            for branch_ctx_info in branch_contexts:
                branch_id = branch_ctx_info["branch_id"]
                branch_context = branch_ctx_info["context"]

                # Load binaries for this specific branch
                if self.mode in ("predict", "explain") and self.binary_loader:
                    loaded_binaries = self.binary_loader.get_step_binaries(
                        self.step_number,
                        branch_id=branch_id  # Branch-aware loading
                    )
                else:
                    loaded_binaries = None

                result = self.step_runner.execute(
                    step, dataset, branch_context, runtime_context,
                    loaded_binaries=loaded_binaries, ...
                )
                branch_ctx_info["context"] = result.updated_context
            continue

        # ... normal step execution
```

#### 3.3.6 Prediction Metadata with Branches

Each prediction must record its branch origin:

```python
prediction = {
    "id": "pred_abc123",
    "pipeline_uid": "0001_pls_def456",
    "model_name": "PLSRegression",
    "dataset_name": "sample_data",
    "fold_id": 0,
    "step_idx": 3,

    # Branch tracking
    "branch_id": 0,  # NEW
    "branch_name": "snv_pca",  # NEW (optional)

    # Preprocessing chain (includes branch-specific transforms)
    "preprocessings": "MinMax>SNV>PCA",
    "y_processing": "scaled_StandardScaler_001",

    # Metrics
    "val_rmse": 0.123,
    "test_rmse": 0.145,
    "y_pred": [...],
    "y_true": [...],
}
```

**Prediction filtering with branches**:
```python
# Get all predictions from branch 0
branch_0_preds = predictions.filter_predictions(branch_id=0)

# Get best prediction per branch
for branch_id in [0, 1]:
    branch_best = predictions.top(1, rank_metric='rmse', branch_id=branch_id)
```

#### 3.3.7 Generator Choices with Branches

When generators are used inside branches, the `generator_choices` must track which branch they belong to:

```yaml
generator_choices:
  # Pre-branch choices
  - {"_range_": 3}  # e.g., n_splits

  # Branch choices (from _or_ inside branch)
  - branch_id: 0
    choice: {"_or_": "nirs4all.operators.transforms.SNV"}
  - branch_id: 1
    choice: {"_or_": "nirs4all.operators.transforms.MSC"}
```

#### 3.3.8 Full Roundtrip Test Scenario

**Training phase**:
```python
pipeline = [
    ShuffleSplit(n_splits=2),
    MinMaxScaler(),  # Step 1: shared artifact
    {"branch": [
        [SNV(), PCA(n_components=10)],  # Step 2, Branch 0
        [MSC(), FirstDerivative()],      # Step 2, Branch 1
    ]},
    PLSRegression(n_components=5),  # Step 3: 2 artifacts (one per branch)
]

runner = PipelineRunner(save_files=True)
predictions, _ = runner.run(PipelineConfigs(pipeline), DatasetConfigs("data/"))
```

**Saved structure**:
```
workspace/runs/2025-12-11_sample_data/
├── _binaries/
│   ├── ab/abc123.pkl          # MinMaxScaler (shared)
│   ├── cd/cdef456.pkl         # SNV (branch 0)
│   ├── de/defg789.pkl         # PCA (branch 0)
│   ├── ef/efgh012.pkl         # MSC (branch 1)
│   ├── fg/fghi345.pkl         # FirstDerivative (branch 1)
│   ├── gh/ghij678.pkl         # PLSRegression (branch 0)
│   └── hi/hijk901.pkl         # PLSRegression (branch 1)
├── 0001_pls_abc123/
│   ├── manifest.yaml
│   └── pipeline.json
└── predictions.json
```

**Prediction phase**:
```python
# Load and predict with branch 0's model
runner = PipelineRunner()
pred_obj = predictions.top(1, branch_id=0)[0]  # Get best from branch 0
y_pred, _ = runner.predict(pred_obj, new_data)
```

**Prediction flow**:
1. Load `manifest.yaml` for pipeline `0001_pls_abc123`
2. Create `BinaryLoader` with branch-aware artifact mapping
3. Execute pipeline with `mode="predict"`
4. At step 2 (branch step):
   - Reconstruct branch contexts from manifest
   - Execute only the target branch (based on `pred_obj.branch_id`)
5. At step 3 (post-branch):
   - Load `PLSRegression_b0_1` for branch 0
   - Transform with loaded artifact
6. Return predictions

#### 3.3.9 Edge Cases and Error Handling

| Scenario | Handling |
|----------|----------|
| Missing branch artifact | Raise `FileNotFoundError` with clear message including branch_id |
| Branch count mismatch (train vs predict) | Raise `ValueError` during manifest load |
| Partial branch execution (predict only branch 0) | Filter artifacts by target branch_id |
| Legacy manifests (no branch_id) | Treat all artifacts as `branch_id=None` (backward compatible) |
| Generator inside branch | Store expanded choice with branch_id in generator_choices |

### 3.4 Rationale for Design Choices

#### Why a Controller vs. Generator Expansion?

| Approach | Pros | Cons |
|----------|------|------|
| Generator expansion (`_or_`) | Simple, reuses existing infrastructure | Full reload per variant, no shared state |
| Controller (proposed) | Shared state, single load, efficient | New controller, execution flow changes |

**Decision**: Controller approach because:
1. Branches share splits (critical for fair comparison)
2. Single dataset load (performance)
3. Runtime flexibility (can have in-branch models)
4. Aligns with `feature_augmentation` pattern

#### Why Store Branch Contexts in `context.custom`?

Alternatives considered:
1. **Return multiple contexts from controller** - Breaks controller interface
2. **Create new "BranchState" class** - Over-engineering for Phase 1
3. **Modify RuntimeContext** - Violates separation of concerns

**Decision**: `context.custom` because:
- Minimal interface changes
- Controllers already use custom dict for coordination
- Easy to extend later
- Non-breaking for existing controllers

#### Why Branch ID in DataSelector?

The `branch_id` enables:
1. Indexer filtering: `dataset.x({"branch": 0})`
2. Prediction tracking: Know which branch produced a result
3. Future sample-branching: Split samples into branches

### 3.5 Nested Branches: Design and Implications

Nested branches allow hierarchical experimental designs where multiple branching dimensions can be combined. This section details the architecture and implications.

#### 3.5.1 Nested Branch Semantics

When multiple `branch` steps appear in sequence, they create a **Cartesian product** of configurations:

```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [[A()], [B()]]},        # Level 1: 2 branches
    {"branch": [[X()], [Y()], [Z()]]}, # Level 2: 3 branches
    Model(),
]
# Result: 2 × 3 × 3 = 18 prediction sets (6 branch combos × 3 folds)
```

**Branch path identification**:

| Branch Path | Level 1 | Level 2 | Full Name |
|-------------|---------|---------|-----------|
| `(0, 0)` | A | X | `A_X` |
| `(0, 1)` | A | Y | `A_Y` |
| `(0, 2)` | A | Z | `A_Z` |
| `(1, 0)` | B | X | `B_X` |
| `(1, 1)` | B | Y | `B_Y` |
| `(1, 2)` | B | Z | `B_Z` |

#### 3.5.2 Branch Path Encoding

To uniquely identify nested branches, we use a **hierarchical branch path**:

```python
@dataclass
class DataSelector(MutableMapping):
    # ... existing fields ...
    branch_id: Optional[int] = None           # Leaf branch ID (flattened)
    branch_path: Optional[Tuple[int, ...]] = None  # Hierarchical path: (level1_id, level2_id, ...)
    branch_name: Optional[str] = None         # Concatenated name: "snv_pca_if05"
```

**Flattened vs hierarchical IDs**:
- `branch_id`: Single integer for fast filtering (0, 1, 2, 3, 4, 5 for 6 combos)
- `branch_path`: Tuple showing ancestry ((0, 0), (0, 1), ..., (1, 2))
- `branch_name`: Human-readable concatenated name

#### 3.5.3 Context Multiplication in BranchController

When a second `branch` step encounters existing branch contexts, it **multiplies** them:

```python
def execute(self, step_info, dataset, context, runtime_context, ...):
    existing_branches = context.custom.get("branch_contexts", [])
    new_branch_specs = step_info.content  # [[X()], [Y()], [Z()]]

    if not existing_branches:
        # First branch level: create initial contexts
        return self._create_initial_branches(new_branch_specs, context)

    # Nested branch: multiply existing branches by new specs
    multiplied_contexts = []
    for parent_ctx_info in existing_branches:
        parent_ctx = parent_ctx_info["context"]
        parent_path = parent_ctx.selector.branch_path or ()
        parent_name = parent_ctx_info.get("name", "")

        for new_branch_id, new_branch_steps in enumerate(new_branch_specs):
            child_ctx = parent_ctx.copy()
            child_ctx.selector.branch_path = parent_path + (new_branch_id,)
            child_ctx.selector.branch_name = f"{parent_name}_{new_branch_name}"

            # Execute new branch steps
            for step in new_branch_steps:
                child_ctx, _ = self.step_runner.execute(step, dataset, child_ctx, ...)

            multiplied_contexts.append({
                "branch_id": len(multiplied_contexts),  # Flattened ID
                "branch_path": child_ctx.selector.branch_path,
                "name": child_ctx.selector.branch_name,
                "context": child_ctx,
            })

    context.custom["branch_contexts"] = multiplied_contexts
    return context, []
```

#### 3.5.4 Nested Branch Serialization

**Manifest artifact entries with nested branches**:
```yaml
artifacts:
  # Level 1, Branch 0 (SNV)
  - name: "SNV_b(0,)_1"
    step: 1
    branch_path: [0]
    format: "sklearn"
    path: "ab/abc123.pkl"

  # Level 2, nested under Level 1 Branch 0
  - name: "PCA_b(0,0)_1"
    step: 2
    branch_path: [0, 0]
    branch_name: "snv_pca"
    format: "sklearn"
    path: "cd/cdef456.pkl"

  - name: "IsolationForest_b(0,1)_1"
    step: 2
    branch_path: [0, 1]
    branch_name: "snv_if"
    format: "sklearn"
    path: "ef/efgh789.pkl"
```

**BinaryLoader extension for nested branches**:
```python
def get_step_binaries(
    self,
    step_id: int,
    branch_path: Optional[Tuple[int, ...]] = None  # Support hierarchical lookup
) -> List[Tuple[str, Any]]:
    # Try exact branch_path match first
    # Fall back to partial path matching for shared artifacts
    ...
```

#### 3.5.5 Implications and Constraints

**Memory implications**:
- Each nested level multiplies context count: $N_{total} = \prod_{i=1}^{L} B_i$ where $B_i$ is branches at level $i$
- For 3 levels of 3 branches each: 27 contexts × memory per context
- **Recommendation**: Limit nesting depth to 2-3 levels; use generators for larger explorations

**Execution order**:
- Branches at same level execute sequentially (not parallel)
- Nested branches execute after parent branch completes
- Post-branch steps iterate over all leaf contexts

**Prediction tracking**:
```python
prediction = {
    "branch_id": 5,          # Flattened ID for fast filtering
    "branch_path": [1, 2],   # Path: Level1=1, Level2=2
    "branch_name": "msc_derivative_if10",
    # ...
}
```

**Filtering predictions by level**:
```python
# All predictions from level 1, branch 0 (regardless of level 2)
level1_b0 = predictions.filter_predictions(branch_path_startswith=(0,))

# Specific leaf branch
leaf = predictions.filter_predictions(branch_path=(1, 2))
```

#### 3.5.6 Nested Branches with Outlier Excluders

A common pattern combines outlier exclusion at level 1 with preprocessing at level 2:

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {  # Level 1: Outlier strategies
        "by": "outlier_excluder",
        "strategies": [
            None,
            {"method": "isolation_forest", "contamination": 0.05},
            {"method": "mahalanobis", "threshold": 3.0},
        ],
    }},
    {"branch": [  # Level 2: Preprocessing
        [SNV(), PCA(n_components=10)],
        [MSC(), FirstDerivative()],
    ]},
    PLSRegression(n_components=5),
]
# Result: 3 outlier × 2 preprocessing × 5 folds = 30 predictions
```

**Order matters**: Outlier exclusion should typically come **before** preprocessing branches to ensure:
1. Outlier detection happens on consistent feature space
2. Preprocessing doesn't mask outliers
3. Sample exclusion mask is shared within preprocessing branches

#### 3.5.7 Limitations for Phase 1

The following nested branch scenarios are **out of scope for Phase 1**:

| Scenario | Reason | Future Phase |
|----------|--------|--------------|
| In-branch nested branches | Complexity; requires recursive context management | Phase 3+ |
| Dynamic branch count | Branch count must be known at pipeline parse time | Phase 4+ |
| Cross-branch dependencies | Each branch must be fully independent | Not planned |
| Branch-specific splitters | All branches share upstream splits | E3 (future) |

### 3.6 Visualization and Reporting

Branch-aware visualization is essential for comparing experimental configurations. This section details the visualization requirements and implementations.

#### 3.6.1 Branch Comparison Charts

**Heatmap by branch and fold**:
```python
# Visualize RMSE across branches and folds
analyzer.plot_heatmap(
    x_var="branch_name",
    y_var="fold_id",
    display_metric="rmse",
    title="RMSE by Branch and Fold"
)
```

```
                  snv_pca  msc_d1  derivative
        fold_0    0.123    0.145    0.167
        fold_1    0.118    0.139    0.155
        fold_2    0.131    0.152    0.171
        fold_3    0.125    0.147    0.163
        fold_4    0.119    0.141    0.158
```

**Box plot by branch**:
```python
analyzer.plot_boxplot(
    group_by="branch_name",
    metric="rmse",
    title="RMSE Distribution by Branch"
)
```

**Bar chart with confidence intervals**:
```python
analyzer.plot_branch_comparison(
    metric="rmse",
    show_ci=True,
    ci_level=0.95,
    title="Branch Performance Comparison"
)
```

#### 3.6.2 Nested Branch Visualization

For nested branches, visualization must handle hierarchical structure:

**Grouped bar chart by nested levels**:
```python
analyzer.plot_nested_branches(
    level1_var="outlier_strategy",
    level2_var="preprocessing",
    metric="rmse",
    plot_type="grouped_bar"
)
```

```
        ┌─────────────────────────────────────────────────────┐
        │                    RMSE by Configuration             │
        ├─────────────────────────────────────────────────────┤
        │  ████ SNV_PCA   ████ MSC_D1   ████ Derivative       │
        │                                                      │
        │  0.15 ┼                                              │
        │       │  ███                                         │
        │  0.12 ┼  ███  ███                                    │
        │       │  ███  ███  ███     ███                       │
        │  0.09 ┼  ███  ███  ███     ███  ███                  │
        │       │  ███  ███  ███     ███  ███  ███             │
        │  0.06 ┼──┴────┴────┴───────┴────┴────┴──────         │
        │       No Outlier      IF 5%       Mahalanobis        │
        └─────────────────────────────────────────────────────┘
```

**Faceted plots by branch level**:
```python
analyzer.plot_facet_grid(
    row_var="outlier_strategy",
    col_var="preprocessing",
    metric="rmse",
    plot_type="violin"
)
```

#### 3.6.3 Branch Pipeline Diagram

Visualize the branching structure as a DAG:

```python
analyzer.plot_branch_diagram(predictions)
```

```
                    ┌─────────────┐
                    │ ShuffleSplit│
                    │  (n_splits=5)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ MinMaxScaler│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐┌─────▼─────┐┌─────▼─────┐
        │    SNV    ││    MSC    ││    D1     │
        │   (B0)    ││   (B1)    ││   (B2)    │
        └─────┬─────┘└─────┬─────┘└─────┬─────┘
              │            │            │
        ┌─────▼─────┐┌─────▼─────┐┌─────▼─────┐
        │    PCA    ││  Detrend  ││           │
        │ (n=10)    ││           ││           │
        └─────┬─────┘└─────┬─────┘└─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │PLSRegression│
                    │  (n=5)      │
                    └─────────────┘

        Legend: [B0] snv_pca  [B1] msc_detrend  [B2] derivative
        Predictions: 15 total (3 branches × 5 folds)
```

#### 3.6.4 Summary Statistics Table

Generate a summary table comparing branch performance:

```python
summary = analyzer.branch_summary(
    metrics=["rmse", "r2", "mae"],
    aggregate=["mean", "std", "min", "max"]
)
print(summary.to_markdown())
```

| Branch Name | RMSE Mean | RMSE Std | R² Mean | R² Std | MAE Mean | Samples |
|-------------|-----------|----------|---------|--------|----------|---------|
| snv_pca | 0.123 | 0.008 | 0.945 | 0.012 | 0.089 | 150 |
| msc_detrend | 0.145 | 0.011 | 0.932 | 0.015 | 0.102 | 150 |
| derivative | 0.167 | 0.015 | 0.918 | 0.019 | 0.118 | 150 |

**With outlier exclusion branches**:

| Outlier Strategy | Preprocessing | RMSE Mean | Excluded % | Samples |
|-----------------|---------------|-----------|------------|---------|
| None | snv_pca | 0.145 | 0% | 150 |
| IF 5% | snv_pca | 0.123 | 4.7% | 143 |
| Mahalanobis | snv_pca | 0.128 | 3.3% | 145 |
| None | msc_d1 | 0.167 | 0% | 150 |
| IF 5% | msc_d1 | 0.139 | 4.7% | 143 |
| Mahalanobis | msc_d1 | 0.144 | 3.3% | 145 |

#### 3.6.5 Export and Reporting

**HTML report with branch analysis**:
```python
analyzer.generate_report(
    output_path="reports/branch_comparison.html",
    include_sections=["summary", "heatmap", "boxplots", "diagram", "predictions_table"],
    branch_comparison=True
)
```

**LaTeX table for publications**:
```python
summary.to_latex("tables/branch_results.tex", caption="Branch comparison results")
```

#### 3.6.6 Implementation Notes

**File changes for visualization**:

| File | Change Type | Description |
|------|-------------|-------------|
| `nirs4all/visualization/predictions.py` | Modify | Add branch-aware plotting methods |
| `nirs4all/visualization/branch_diagram.py` | **New** | Branch DAG visualization |
| `nirs4all/analysis/branch_analysis.py` | **New** | Branch summary statistics |
| `nirs4all/reports/html_generator.py` | Modify | Include branch sections in reports |

**Dependencies**:
- `matplotlib`: Core plotting
- `seaborn`: Statistical visualizations (heatmaps, box plots)
- `graphviz` (optional): Branch diagram rendering

---

## 4. Implementation Roadmap

### Phase 1: Core Branch Controller (Week 1-2)

**Scope**: Basic branching with explicit branch lists

**Tasks**:
1. Add `branch_id` and `branch_name` fields to `DataSelector`
2. Create `BranchController` in `nirs4all/controllers/flow/branch.py`
3. Implement context snapshot/restore mechanism
4. Implement branch step iteration via `step_runner`
5. Modify `PipelineExecutor` to iterate post-branch steps over branch contexts
6. Add `branch_id` to prediction metadata
7. Unit tests for branch controller

**Deliverables**:
- Working `{"branch": [[steps], [steps]]}` syntax
- Post-branch steps execute on all branches
- Predictions include branch tracking

**Example that should work**:
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [
        [SNV()],
        [MSC()],
    ]},
    PLSRegression(n_components=10),
]
# Produces 6 predictions (2 branches × 3 folds)
```

### Phase 2: In-Branch Models and Y Processing (Week 2-3)

**Scope**: Full in-branch model training and Y processing isolation

**Tasks**:
1. Ensure `y_processing` steps inside branches affect only that branch
2. Verify model controllers work correctly within branches
3. Test mixed scenarios (in-branch model + post-branch model)
4. Handle artifact naming with branch prefixes (`{name}_b{branch_id}_{op_count}`)
5. Update `ArtifactManager.persist()` to accept and store `branch_id`
6. Modify controllers to pass `branch_id` when persisting artifacts
7. Add `branch_id` and `branch_name` to manifest artifact entries

**Deliverables**:
- In-branch Y processing works independently
- In-branch models train correctly
- Artifacts are uniquely named per branch
- Manifest includes branch metadata for all artifacts

**Example that should work**:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": [
        [SNV(), {"y_processing": StandardScaler()}, TabPFN()],
        [MSC(), PLSRegression(n_components=10)],
    ]},
    {"finetune": PLSRegression()},
]
```

### Phase 3: Generator Integration (Week 3-4)

**Scope**: Support `_or_` and `_range_` inside branch specifications

**Tasks**:
1. Expand generators within branch specifications before execution
2. Handle nested generator syntax
3. Support named branches from generator output
4. Update serialization for generator-expanded branches

**Deliverables**:
- `{"branch": {"_or_": [...]}}` works correctly
- Generator-produced branches have proper naming
- Integration tests with complex generator combinations

**Example that should work**:
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}},
    {"_range_": [5, 15, 5], "param": "n_components", "model": PLSRegression},
]
# Produces 27 predictions (3 branches × 3 n_components × 3 folds)
```

### Phase 4: Named Branches and Visualization (Week 4-5)

**Scope**: Named branch syntax and analysis support

**Tasks**:
1. Support dictionary syntax for named branches
2. Add branch_name to all prediction outputs
3. Update `PredictionAnalyzer` to support branch-based grouping
4. Add heatmaps and charts comparing branches
5. Documentation and examples

**Deliverables**:
- `{"branch": {"name": [steps]}}` syntax works
- Visualization tools support branch comparison
- Complete documentation

**Example that should work**:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "snv_pca": [SNV(), PCA(n_components=10)],
        "msc_detrend": [MSC(), Detrend()],
        "derivative": [FirstDerivative()],
    }},
    PLSRegression(n_components=5),
]

# Visualization
analyzer.plot_heatmap(x_var="branch_name", y_var="fold_id", display_metric="rmse")
```

### Phase 5: Prediction Mode and Persistence (Week 5-6)

**Scope**: Full predict/explain mode support with robust serialization

**Tasks**:
1. Extend `BinaryLoader` with `get_step_binaries(step, branch_id)` method
2. Add `artifacts_by_step_branch` dict for (step, branch_id) lookup
3. Ensure branch artifacts are saved with unique content-addressed paths
4. Modify `Predictor._prepare_replay()` to handle branch structure
5. Update `PipelineExecutor` to pass `branch_id` to binary loader in predict mode
6. Implement branch-specific prediction filtering in `Predictions.filter_predictions(branch_id=)`
7. Add roundtrip tests: train with branches → save → load → predict
8. Test partial branch prediction (predict only specific branch)
9. Backward compatibility: handle legacy manifests without `branch_id`
10. Clear error messages when branch artifacts are missing

**Deliverables**:
- Branched pipelines can be saved and reloaded
- Prediction mode works with branched pipelines
- Can predict using a specific branch's model
- Complete roundtrip tests
- Backward compatible with non-branched pipelines

**Example that should work**:
```python
# Training
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [[SNV()], [MSC()]]},
    PLSRegression(n_components=5),
]
runner = PipelineRunner(save_files=True)
predictions, _ = runner.run(PipelineConfigs(pipeline), DatasetConfigs("data/"))

# Prediction with branch 0's best model
best_branch_0 = predictions.top(1, rank_metric='rmse', branch_id=0)[0]
y_pred, _ = runner.predict(best_branch_0, new_data)

# Prediction with branch 1's best model
best_branch_1 = predictions.top(1, rank_metric='rmse', branch_id=1)[0]
y_pred_2, _ = runner.predict(best_branch_1, new_data)
```

### Phase 6: Visualization and Reporting (Week 6-7)

**Scope**: Branch-aware visualization and analysis tools

**Tasks**:
1. Add `plot_heatmap()` with branch grouping support
2. Add `plot_boxplot()` with branch-based comparison
3. Implement `plot_branch_diagram()` for DAG visualization
4. Create `branch_summary()` for tabular statistics
5. Add nested branch visualization support (grouped bars, facet grids)
6. Integrate branch sections into HTML report generator
7. Add LaTeX export for publication tables

**Deliverables**:
- Heatmaps, box plots, and bar charts comparing branches
- Branch DAG diagram visualization
- Summary statistics tables
- HTML report with branch analysis section

**Example that should work**:
```python
analyzer = PredictionAnalyzer(predictions)

# Heatmap
analyzer.plot_heatmap(x_var="branch_name", y_var="fold_id", display_metric="rmse")

# Summary table
summary = analyzer.branch_summary(metrics=["rmse", "r2"])
print(summary.to_markdown())

# Branch diagram
analyzer.plot_branch_diagram()

# HTML report
analyzer.generate_report("reports/branch_comparison.html", branch_comparison=True)
```

### Phase 7: Outlier Excluder Branches (Week 7-8)

**Scope**: Sample-based branching with outlier detection strategies

**Tasks**:
1. Implement `OutlierExcluderController` or extend `BranchController`
2. Add outlier detection methods: IsolationForest, Mahalanobis, Leverage, LOF
3. Implement sample exclusion mask per branch
4. Ensure exclusion applies only to training data
5. Add `excluded` field to prediction metadata
6. Visualize excluded samples in reports
7. Add tests for combined outlier + preprocessing branches

**Deliverables**:
- `{"branch": {"by": "outlier_excluder", ...}}` syntax works
- Multiple outlier strategies can be compared
- Exclusion metadata in predictions
- Visualization of exclusion impact

**Example that should work**:
```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,
            {"method": "isolation_forest", "contamination": 0.05},
            {"method": "mahalanobis", "threshold": 3.0},
        ],
    }},
    PLSRegression(n_components=10),
]
# Result: 3 strategies × 5 folds = 15 predictions with different sample sets
```

---

## 5. Integration Testing Requirements

### 5.1 Reproducibility Test Suite

A critical requirement is ensuring that **train → save → reload → predict** produces **identical results**. This section specifies the integration tests.

#### 5.1.1 Deterministic Roundtrip Test

```python
def test_branch_roundtrip_reproducibility():
    """Train with branches, save, reload, predict - results must match."""

    # Setup: fixed random seed for reproducibility
    np.random.seed(42)

    # 1. Define pipeline with branches
    pipeline = [
        ShuffleSplit(n_splits=3, random_state=42),
        MinMaxScaler(),
        {"branch": [
            [SNV(), PCA(n_components=5)],
            [MSC(), FirstDerivative()],
        ]},
        PLSRegression(n_components=3),
    ]

    # 2. Training run with save
    runner = PipelineRunner(save_files=True, workspace="./test_workspace")
    predictions_train, _ = runner.run(
        PipelineConfigs(pipeline),
        DatasetConfigs("data/sample_data")
    )

    # Store predictions for comparison
    train_results = {}
    for pred in predictions_train:
        key = (pred.branch_id, pred.fold_id)
        train_results[key] = {
            "y_pred": np.array(pred.y_pred),
            "y_true": np.array(pred.y_true),
            "val_rmse": pred.val_rmse,
        }

    # 3. Reload and predict for each branch
    reloaded_runner = PipelineRunner(workspace="./test_workspace")

    for branch_id in [0, 1]:
        best_pred = predictions_train.top(1, branch_id=branch_id)[0]

        # Predict on same data (validation fold)
        y_pred_reload, metrics = reloaded_runner.predict(
            best_pred,
            DatasetConfigs("data/sample_data"),
            fold_id=best_pred.fold_id  # Same fold for fair comparison
        )

        # 4. Assert exact match
        key = (branch_id, best_pred.fold_id)
        np.testing.assert_array_almost_equal(
            y_pred_reload,
            train_results[key]["y_pred"],
            decimal=10,
            err_msg=f"Predictions mismatch for branch {branch_id}"
        )

def test_nested_branch_roundtrip():
    """Verify nested branches produce identical results after reload."""

    pipeline = [
        ShuffleSplit(n_splits=2, random_state=42),
        {"branch": {
            "by": "outlier_excluder",
            "strategies": [None, {"method": "isolation_forest", "contamination": 0.05}],
        }},
        {"branch": [[SNV()], [MSC()]]},
        PLSRegression(n_components=3),
    ]

    # Train, save, reload, predict for each branch path
    # Assert identical results for all 4 branch combinations
    ...
```

#### 5.1.2 Artifact Integrity Tests

```python
def test_branch_artifacts_complete():
    """Verify all branch artifacts are saved correctly."""

    pipeline = [
        ShuffleSplit(n_splits=2),
        {"branch": [[SNV(), PCA(n_components=5)], [MSC()]]},
        PLSRegression(n_components=3),
    ]

    runner = PipelineRunner(save_files=True)
    predictions, _ = runner.run(...)

    # Load manifest
    manifest = ManifestManager.load_manifest(predictions[0].pipeline_uid)

    # Verify artifacts for both branches
    branch_0_artifacts = [a for a in manifest["artifacts"] if a.get("branch_id") == 0]
    branch_1_artifacts = [a for a in manifest["artifacts"] if a.get("branch_id") == 1]

    assert len(branch_0_artifacts) >= 3  # SNV, PCA, PLSRegression
    assert len(branch_1_artifacts) >= 2  # MSC, PLSRegression

    # Verify all artifact files exist
    for artifact in manifest["artifacts"]:
        path = workspace / "_binaries" / artifact["path"]
        assert path.exists(), f"Missing artifact: {artifact['name']}"

def test_branch_artifact_isolation():
    """Verify branch artifacts are isolated (no cross-contamination)."""

    # Train with branches
    # Reload and predict with branch 0 only
    # Verify only branch 0 artifacts are loaded
    # Verify branch 1 artifacts are not loaded (check logs or counters)
    ...
```

#### 5.1.3 Numerical Precision Tests

```python
def test_branch_prediction_precision():
    """Verify predictions maintain numerical precision across save/load."""

    pipeline = [...]

    runner = PipelineRunner(save_files=True)
    predictions, _ = runner.run(...)

    # Store original predictions with full precision
    original_y_pred = predictions[0].y_pred.copy()

    # Reload and predict
    reloaded_y_pred = runner.predict(predictions[0], ...)

    # Assert floating point precision (not just almost_equal)
    max_diff = np.max(np.abs(original_y_pred - reloaded_y_pred))
    assert max_diff < 1e-12, f"Precision loss: max diff = {max_diff}"

def test_branch_transformer_state():
    """Verify transformer states are correctly restored."""

    # Train with branches (MinMaxScaler, SNV, PCA per branch)
    # Save artifacts
    # Load artifacts manually
    # Verify transformer parameters match exactly:
    #   - MinMaxScaler.data_min_, data_max_
    #   - PCA.components_, mean_, explained_variance_
    ...
```

#### 5.1.4 Edge Case Tests

```python
def test_partial_branch_prediction():
    """Predict using only one branch when multiple exist."""

    # Train with 3 branches
    # Predict using branch 1 only
    # Verify only branch 1 artifacts loaded
    # Verify predictions are correct
    ...

def test_branch_mismatch_error():
    """Verify clear error when branch structure doesn't match manifest."""

    # Train with 2 branches, save
    # Modify manifest to have 3 branches
    # Attempt to predict
    # Assert ValueError with clear message
    ...

def test_legacy_manifest_compatibility():
    """Verify backward compatibility with manifests without branch_id."""

    # Create manifest without branch_id fields (simulate legacy)
    # Attempt to load and predict
    # Verify it works with branch_id=None fallback
    ...

def test_corrupted_artifact_handling():
    """Verify graceful handling of missing/corrupted artifacts."""

    # Train with branches, save
    # Delete one artifact file
    # Attempt to predict
    # Assert FileNotFoundError with helpful message including branch info
    ...
```

### 5.2 Performance Benchmarks

```python
def test_branch_memory_efficiency():
    """Verify dataset is loaded once, not per branch."""

    # Use memory profiler
    # Train with 5 branches
    # Assert memory usage is ~1x dataset size, not ~5x
    ...

def test_branch_execution_time():
    """Benchmark branch execution overhead."""

    # Train without branches: 5 separate runs
    # Train with branches: 1 run with 5 branches
    # Assert branched version is faster (shared splits, single load)
    ...
```

### 5.3 Test File Structure

```
tests/
├── integration/
│   ├── test_branch_roundtrip.py        # Reproducibility tests
│   ├── test_branch_artifacts.py        # Artifact integrity tests
│   ├── test_branch_predict.py          # Prediction mode tests
│   ├── test_branch_nested.py           # Nested branch tests
│   ├── test_branch_outlier.py          # Outlier excluder tests
│   └── test_branch_visualization.py    # Visualization tests
└── unit/
    ├── test_branch_controller.py       # BranchController unit tests
    ├── test_branch_serialization.py    # Serialization tests
    └── test_branch_binary_loader.py    # BinaryLoader tests
```

---

## 6. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| **Core Controller** | | |
| `nirs4all/controllers/flow/branch.py` | **New** | BranchController implementation |
| `nirs4all/controllers/flow/outlier_excluder.py` | **New** | OutlierExcluderBranch implementation |
| `nirs4all/controllers/registry.py` | Modify | Import and register BranchController |
| **Context & Parser** | | |
| `nirs4all/pipeline/config/context.py` | Modify | Add `branch_id`, `branch_name`, `branch_path` to DataSelector |
| `nirs4all/pipeline/execution/executor.py` | Modify | Branch context iteration in `_execute_steps` |
| `nirs4all/pipeline/steps/parser.py` | Modify | Add "branch" to WORKFLOW_KEYWORDS |
| `nirs4all/pipeline/config/_generator/keywords.py` | Modify | Add BRANCH_KEYWORD constant |
| **Serialization & Storage** | | |
| `nirs4all/pipeline/storage/manifest_manager.py` | Modify | Add branch_id/branch_name/branch_path to artifact entries |
| `nirs4all/pipeline/storage/artifacts/binary_loader.py` | Modify | Add branch-aware `get_step_binaries(step, branch_id)` |
| `nirs4all/pipeline/storage/artifacts/manager.py` | Modify | Pass branch_id when persisting artifacts |
| `nirs4all/pipeline/config/component_serialization.py` | Modify | Handle branch dict serialization |
| `nirs4all/pipeline/predictor.py` | Modify | Branch-aware artifact loading in predict mode |
| **Predictions** | | |
| `nirs4all/data/predictions.py` | Modify | Add branch_id, branch_name, branch_path, excluded fields; filter_predictions(branch_id=, branch_path=) |
| **Visualization & Reporting** | | |
| `nirs4all/visualization/predictions.py` | Modify | Support branch-based grouping in plots |
| `nirs4all/visualization/branch_diagram.py` | **New** | Branch DAG visualization |
| `nirs4all/analysis/branch_analysis.py` | **New** | Branch summary statistics and comparison |
| `nirs4all/reports/html_generator.py` | Modify | Include branch sections in reports |
| **Outlier Detection** | | |
| `nirs4all/operators/outlier_detection.py` | **New** | Outlier detection methods (IF, Mahalanobis, LOF, etc.) |
| **Documentation & Examples** | | |
| `docs/reference/writing_pipelines.md` | Modify | Document `branch` syntax |
| `docs/reference/branching.md` | **New** | Comprehensive branching documentation |
| `examples/QXX_branching.py` | **New** | Basic branching examples |
| `examples/QXX_nested_branches.py` | **New** | Nested branch examples |
| `examples/QXX_outlier_branches.py` | **New** | Outlier excluder branch examples |
| **Tests** | | |
| `tests/unit/controllers/test_branch.py` | **New** | Unit tests for BranchController |
| `tests/unit/controllers/test_outlier_excluder.py` | **New** | Unit tests for outlier excluder |
| `tests/integration/test_branch_pipeline.py` | **New** | Integration tests including serialization roundtrip |
| `tests/integration/test_branch_predict.py` | **New** | Prediction mode tests with branched pipelines |
| `tests/integration/test_branch_roundtrip.py` | **New** | Reproducibility/determinism tests |
| `tests/integration/test_branch_nested.py` | **New** | Nested branch tests |
| `tests/integration/test_branch_outlier.py` | **New** | Outlier excluder tests |
| `tests/integration/test_branch_visualization.py` | **New** | Visualization tests |

---

## 9. References

- [Pipeline Example](./branching_spec_full_example.md)
- [Branching Risk Mitigation](./branching_spec_risks.md)
- [ExecutionContext](../../nirs4all/pipeline/config/context.py): Context management
- [FeatureAugmentationController](../../nirs4all/controllers/data/feature_augmentation.py): Parallel execution pattern
- [PipelineExecutor](../../nirs4all/pipeline/execution/executor.py): Step execution flow
- [Indexer](../../nirs4all/data/indexer.py): Branch field in index schema
- [Generator System](../../nirs4all/pipeline/config/generator.py): `_or_` and `_range_` expansion
- [TransformerController](../../nirs4all/controllers/transforms/transformer.py): Processing update logic

---
---

**Author:** Senior Python/ML Developer
**Date:** December 2025
**Status:** Specification - Pending Review
