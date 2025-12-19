# Branching Generation Proposal

## In-Pipeline Variant Generation vs Multi-Pipeline Generation

**Author**: Copilot Analysis
**Date**: December 2025
**Status**: Draft Proposal

---

## Executive Summary

This document analyzes the current generator system and proposes modifications to optionally generate pipeline variants **within a single pipeline** using branching, instead of generating **multiple separate pipelines**. This approach could reduce execution overhead, enable better comparison workflows, and leverage the existing branching infrastructure.

---

## 1. Current Architecture Analysis

### 1.1 Generator System (expand_spec)

The generator system in `nirs4all/pipeline/config/_generator/` is responsible for expanding configuration specifications into concrete variants.

**Current Flow:**
```
User Spec (with _or_, _range_, etc.)
        ↓
   PipelineConfigs.__init__()
        ↓
   expand_spec_with_choices()
        ↓
   N separate pipeline configurations (self.steps = [config1, config2, ...])
        ↓
   PipelineOrchestrator loops over each config
        ↓
   N separate pipeline executions
```

**Key Files:**
- `nirs4all/pipeline/config/generator.py` - Public API
- `nirs4all/pipeline/config/_generator/core.py` - Main expansion logic
- `nirs4all/pipeline/config/_generator/strategies/` - Strategy pattern for each keyword
- `nirs4all/pipeline/config/pipeline_config.py` - Uses generator to create multiple configs

**Relevant Code in `PipelineConfigs`:**
```python
if self._has_gen_keys(self.steps):
    expanded_with_choices = expand_spec_with_choices(self.steps)
    self.steps = [config for config, choices in expanded_with_choices]  # N configs
```

### 1.2 Branching System

The branching system allows splitting a pipeline into parallel sub-pipelines with independent contexts.

**Current Flow:**
```
Single Pipeline with {"branch": [...]}
        ↓
   BranchController.execute()
        ↓
   Creates N branch_contexts with isolated ExecutionContext
        ↓
   Post-branch steps execute on each branch context
        ↓
   All results within single pipeline execution
```

**Key Files:**
- `nirs4all/controllers/data/branch.py` - BranchController
- `nirs4all/pipeline/execution/executor.py` - Handles branch context propagation

**Key Insight:** The branching system already supports generator syntax **inside** branch definitions:
```python
{"branch": {"_or_": [SNV(), MSC(), FirstDerivative()]}}  # Creates 3 branches
```

### 1.3 What Branching Currently Isolates

After detailed analysis of the branching implementation (`BranchController` + `PipelineExecutor`), here's what is and isn't isolated:

#### ✅ Properly Isolated (Branching Works Correctly)

| Component | Isolation Mechanism | Details |
|-----------|---------------------|---------|
| **X Data (Features)** | `_snapshot_features()` / `_restore_features()` | Deep copy of `dataset._features.sources` before each branch. Each branch starts from the same initial X state. |
| **ExecutionContext** | `initial_context.copy()` | Each branch gets a deep copy of the context. |
| **X Processing Chain** | `copy.deepcopy(context.selector.processing)` | Each branch tracks its own preprocessing chain names (e.g., `["raw", "SNV_1"]`). |
| **Y Processing State** | `context.state.y_processing` (via context copy) | Each branch has its own `y_processing` identifier (e.g., `"numeric_StandardScaler1"`). Y transforms create new named layers, so branches can have different y transformations. |
| **Post-Branch Feature State** | `features_snapshot` in `branch_contexts` | After branch steps complete, features are snapshotted. Post-branch steps (like models) restore this snapshot per-branch. |

#### ⚠️ Shared (But Not a Problem in Practice)

| Component | Why Shared | Why It's OK |
|-----------|------------|-------------|
| **Dataset Indexer** (`_indexer`) | Not snapshotted | Sample filtering typically runs BEFORE branching or in training mode only. Each filter creates unique exclusion reasons. |
| **Dataset Targets** (`_targets`) | Not snapshotted | Y transforms ADD new processing layers with unique names. Branches use different `y_processing` identifiers via context. |

#### ❌ True Limitation: Splitters

| Component | Why It's a Problem |
|-----------|-------------------|
| **Cross-Validation Splits** | Splitters define the fold structure and must run once before any branching. If you want to compare different CV strategies (e.g., 5-fold vs 10-fold), you need separate pipelines.  |

### 1.4 Step Categories Summary

| Step Type | Data Modification | Branching Support |
|-----------|-------------------|-------------------|
| **Models** | No (data unchanged) | ✅ Can run sequentially on same data |
| **Preprocessing/Transforms** | Yes (X modified) | ✅ Fully isolated via features snapshot |
| **Y Processing** | Yes (adds y layer) | ✅ Isolated via context y_processing |
| **Splitters** | Defines fold structure | ❌ Must be shared (run before branching) |
| **Feature Selection** | Yes (X columns reduced) | ✅ Fully isolated via features snapshot |
| **Sample Filtering** | Marks samples excluded | ✅ Works in practice (runs before branches) |
| **Charts/Visualization** | No | ✅ Can run on any branch context |

---

## 2. Proposed Behavior: In-Pipeline Branching Generation

### 2.1 Concept

Instead of generating N separate pipelines, generate variants **within** a single pipeline using the branching mechanism:

**Current (Multi-Pipeline):**
```yaml
pipeline_1: [Scaler, SNV, PLS(n=5)]
pipeline_2: [Scaler, SNV, PLS(n=10)]
pipeline_3: [Scaler, MSC, PLS(n=5)]
pipeline_4: [Scaler, MSC, PLS(n=10)]
```

**Proposed (In-Pipeline Branching):**
```yaml
single_pipeline:
  - Scaler
  - branch:  # Preprocessing variant branch
      - SNV
      - MSC
  # Post-branch, for each preprocessing branch:
  - branch:  # Model variant branch (nested or flat)
      - PLS(n=5)
      - PLS(n=10)
```

Or using automatic conversion:
```yaml
single_pipeline:
  - Scaler
  - transform: {_or_: [SNV, MSC], _branch_: true}  # New keyword
  - model: {_or_: [PLS(n=5), PLS(n=10)], _branch_: true}
```

### 2.2 Behavior Rules by Step Type

Since branching already provides proper isolation for most step types (see Section 1.3), the rules are simpler than originally thought:

#### Rule 1: Model Variants → Sequential Models OR Branches
Models don't modify data, so multiple model variants can run:
- **Sequentially** on the same context (simplest approach)
- **As branches** (if you want branch-level tracking/comparison)

```python
# Input: {"model": {"_or_": [PLS(n=5), PLS(n=10), PLS(n=15)]}}
#
# Option A: Sequential (simple, same exact data for all)
# [
#     {"model": PLS(n=5)},   # Runs on current context
#     {"model": PLS(n=10)},  # Runs on same context (data unchanged)
#     {"model": PLS(n=15)},  # Runs on same context
# ]
#
# Option B: Branch (if tracking is desired)
# {"branch": [[PLS(n=5)], [PLS(n=10)], [PLS(n=15)]]}
```

#### Rule 2: Any Data-Modifying Step → Create Branches
Preprocessing, Y processing, feature selection - all already work correctly in branches:

```python
# Input: {"transform": {"_or_": [SNV, MSC, D1]}}
#
# Expanded in-pipeline:
# {"branch": [
#     [SNV()],
#     [MSC()],
#     [D1()],
# ]}
```

**Implementation:** Wrap variants in a branch step. Branching handles isolation automatically.

#### Rule 3: Splitter Variants → Multi-Pipeline (Required)
Splitter variants define different CV structures and CANNOT be branched:

```python
# Input: ShuffleSplit(n_splits={"_or_": [3, 5, 10]})
#
# Must remain as multi-pipeline - branching not applicable
```

**This is the only case where multi-pipeline generation is truly required.**

#### Rule 4: Nested Variants → Leverages Existing Branch Multiplication
When multiple steps have variants, the existing `_multiply_branch_contexts` already handles Cartesian products:

```python
# Input:
# [
#     {"transform": {"_or_": [SNV, MSC]}},
#     {"model": {"_or_": [PLS(n=5), PLS(n=10)]}}
# ]
#
# Option A: Nested branches (4 combinations)
# {"branch": [
#     [SNV, {"branch": [[PLS(n=5)], [PLS(n=10)]]}],
#     [MSC, {"branch": [[PLS(n=5)], [PLS(n=10)]]}],
# ]}
#
# Option B: Flat branches with models inside (simpler)
# {"branch": [
#     [SNV, PLS(n=5)],
#     [SNV, PLS(n=10)],
#     [MSC, PLS(n=5)],
#     [MSC, PLS(n=10)],
# ]}
```

---

## 3. Implementation Approaches

### 3.1 Approach A: New Expansion Mode in PipelineConfigs

Add a mode flag to `PipelineConfigs` that changes expansion behavior:

```python
class PipelineConfigs:
    def __init__(
        self,
        definition,
        name="",
        max_generation_count=10000,
        generation_mode="multi_pipeline"  # NEW: or "in_pipeline_branch"
    ):
        if generation_mode == "in_pipeline_branch":
            self.steps = [self._expand_to_branches(self.steps)]  # Single config with branches
        else:
            # Current behavior
            self.steps = [config for config, _ in expand_spec_with_choices(self.steps)]
```

**Pros:**
- Non-breaking change
- User explicitly chooses behavior
- Clear separation of concerns

**Cons:**
- New code path to maintain
- Need to handle edge cases

### 3.2 Approach B: New Generator Keywords (_branch_, _sequential_)

Add keywords that control in-pipeline expansion:

```python
# _branch_: true wraps variants in a branch step
{"transform": {"_or_": [SNV, MSC], "_branch_": true}}

# _sequential_: true expands as consecutive steps (for models)
{"model": {"_or_": [PLS(n=5), PLS(n=10)], "_sequential_": true}}
```

**Implementation in strategies:**

```python
# In or_strategy.py
def expand(self, node, ...):
    if node.get("_branch_"):
        # Return a single branch step instead of multiple variants
        variants = self._expand_basic(node["_or_"], expand_nested)
        return [{"branch": [[v] for v in variants]}]
    elif node.get("_sequential_"):
        # Return a list of steps to inject
        variants = self._expand_basic(node["_or_"], expand_nested)
        return [variants]  # Single result containing list of steps
    else:
        # Current behavior
        return self._expand_basic(...)
```

**Pros:**
- Granular control at step level
- Composable with existing keywords
- Intuitive semantics

**Cons:**
- More complex generator system
- Need careful handling of nesting

### 3.3 Approach C: Post-Processing Transformation

Add a transformer that converts expanded pipelines to branched form:

```python
def pipelines_to_branched(pipelines: List[List]) -> List:
    """
    Takes N pipeline configurations and converts to single pipeline with branches.

    Identifies common prefix, groups by varying steps, creates branch structure.
    """
    # Find common prefix
    common_prefix = find_common_prefix(pipelines)

    # Find varying sections
    varying_sections = extract_varying_sections(pipelines, len(common_prefix))

    # Build branched pipeline
    return common_prefix + [{"branch": varying_sections}]
```

**Pros:**
- Works with existing expansion
- Transformation is optional and reversible
- Good for analysis/debugging

**Cons:**
- Additional processing step
- Complex grouping logic
- May not handle all patterns

### 3.4 Approach D: Smart Step Insertion (Controller-Level)

Let controllers handle variant expansion internally:

```python
# In BaseModelController
def execute(self, step_info, dataset, context, ...):
    model_config = step_info.operator

    # Check for generator syntax
    if has_generator_keys(model_config):
        expanded = expand_spec(model_config)

        # Execute each model variant (data unchanged between runs)
        for variant in expanded:
            self._execute_single_model(variant, ...)
```

**Pros:**
- Controller-specific handling
- No changes to generator
- Leverages existing controller architecture

**Cons:**
- Duplicate logic across controllers
- Harder to track/visualize expansions
- Less predictable behavior

---

## 4. Recommended Approach: Hybrid (B + A)

Combine **Approach B (new keywords)** with **Approach A (mode flag)** for maximum flexibility:

### 4.1 New Keywords

1. **`_branch_: true`** - Wrap variants in a branch step
2. **`_sequential_: true`** - Expand as consecutive steps (for non-modifying operations)
3. **`_inline_: true`** - Alias for `_sequential_` (alternative naming)

### 4.2 New Mode in PipelineConfigs

```python
PipelineConfigs(
    definition,
    generation_mode="auto"  # "multi_pipeline" | "in_pipeline" | "auto"
)
```

- `"multi_pipeline"`: Current behavior (N separate pipelines)
- `"in_pipeline"`: All variants become branches/sequential steps
- `"auto"`: Analyze step types and choose optimal strategy

### 4.3 Auto Mode Logic

```python
def _determine_expansion_mode(self, steps):
    """Analyze steps to determine optimal expansion mode."""

    variant_steps = self._find_variant_steps(steps)

    # If only model variants → sequential is optimal
    if all(self._is_model_step(s) for s in variant_steps):
        return "sequential"

    # If preprocessing variants → branching required
    if any(self._is_data_modifying_step(s) for s in variant_steps):
        return "branching"

    # If splitter variants → multi-pipeline recommended
    if any(self._is_splitter_step(s) for s in variant_steps):
        return "multi_pipeline"

    return "auto_branch"  # Default to branching for safety
```

---

## 5. Implementation Plan

### Phase 1: Keyword Support (Low Effort)

1. **Add `_branch_` keyword to generator**
   - Modify `or_strategy.py` to recognize `_branch_: true`
   - When set, wrap variants in a branch step structure
   - Add tests

2. **Add `_sequential_` keyword**
   - Modify `or_strategy.py` to recognize `_sequential_: true`
   - When set, return list of steps to inject (for model variants)
   - Add tests

3. **Update documentation**
   - Add examples to Q23_generator_syntax.py
   - Document behavior differences

**Files to Modify:**
- `nirs4all/pipeline/config/_generator/keywords.py` (add constants)
- `nirs4all/pipeline/config/_generator/strategies/or_strategy.py` (handle new keywords)
- `nirs4all/pipeline/config/generator.py` (export new keywords)

### Phase 2: Mode Flag (Medium Effort)

1. **Add `generation_mode` parameter to PipelineConfigs**
2. **Implement `_expand_to_branches()` method**
3. **Add auto-detection logic**
4. **Update orchestrator to handle single-pipeline with many branches**

**Files to Modify:**
- `nirs4all/pipeline/config/pipeline_config.py`
- `nirs4all/pipeline/execution/orchestrator.py` (optional, for stats)

### Phase 3: Enhanced Branch Support (Higher Effort)

1. **Nested branch optimization**
   - Flatten unnecessary nesting
   - Merge compatible branches

2. **Branch-level statistics**
   - Track which variants came from same generator
   - Enable grouped analysis

3. **Visualization**
   - Show expansion source in branch diagrams
   - Link predictions back to generator choices

**Files to Modify:**
- `nirs4all/controllers/data/branch.py`
- `nirs4all/visualization/analysis/branch.py`

---

## 6. Detailed Technical Changes

### 6.1 New Keywords in keywords.py

```python
# Add to nirs4all/pipeline/config/_generator/keywords.py

# In-pipeline expansion modifiers
BRANCH_MODIFIER = "_branch_"      # Wrap variants in branch step
SEQUENTIAL_MODIFIER = "_sequential_"  # Expand as consecutive steps
INLINE_MODIFIER = "_inline_"      # Alias for sequential

EXPANSION_MODIFIERS = frozenset({
    BRANCH_MODIFIER, SEQUENTIAL_MODIFIER, INLINE_MODIFIER
})

# Update PURE_OR_KEYS to include new modifiers
PURE_OR_KEYS = frozenset({
    OR_KEYWORD, SIZE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, WEIGHTS_KEYWORD,
    PICK_KEYWORD, ARRANGE_KEYWORD, THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD,
    MUTEX_KEYWORD, REQUIRES_KEYWORD, EXCLUDE_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
    BRANCH_MODIFIER, SEQUENTIAL_MODIFIER, INLINE_MODIFIER,  # NEW
})
```

### 6.2 OrStrategy Modifications

```python
# Add to nirs4all/pipeline/config/_generator/strategies/or_strategy.py

def expand(self, node, seed=None, expand_nested=None):
    choices = node[OR_KEYWORD]

    # Check for in-pipeline expansion modifiers
    use_branch = node.get("_branch_", False)
    use_sequential = node.get("_sequential_", False) or node.get("_inline_", False)

    # ... existing size/pick/arrange/constraint handling ...

    # Apply in-pipeline expansion if requested
    if use_branch:
        # Wrap each variant in a branch definition
        return [self._create_branch_step(result)]
    elif use_sequential:
        # Return as list of steps to inject (single-element outer list)
        return [result]  # result is already a list of variants

    # ... existing return ...

def _create_branch_step(self, variants: List[Any]) -> Dict[str, Any]:
    """Create a branch step containing all variants.

    Args:
        variants: List of expanded variant values

    Returns:
        Branch step dict: {"branch": [[v1], [v2], ...]}
    """
    # Wrap each variant in a list (branch expects list of step lists)
    branch_definitions = []
    for variant in variants:
        if isinstance(variant, list):
            branch_definitions.append(variant)
        else:
            branch_definitions.append([variant])

    return {"branch": branch_definitions}
```

### 6.3 PipelineConfigs Mode Support

```python
# Modify nirs4all/pipeline/config/pipeline_config.py

class PipelineConfigs:
    def __init__(
        self,
        definition: Union[Dict, List[Any], str],
        name: str = "",
        description: str = "No description provided",
        max_generation_count: int = 10000,
        generation_mode: str = "multi_pipeline"  # NEW PARAMETER
    ):
        # ... existing parsing ...

        self.generation_mode = generation_mode

        if self._has_gen_keys(self.steps):
            if generation_mode == "in_pipeline":
                # Convert generator syntax to branch syntax
                self.steps = [self._convert_to_branched(self.steps)]
                self.generator_choices = [[]]
            elif generation_mode == "auto":
                mode = self._analyze_optimal_mode(self.steps)
                if mode == "in_pipeline":
                    self.steps = [self._convert_to_branched(self.steps)]
                    self.generator_choices = [[]]
                else:
                    # Use multi-pipeline (existing behavior)
                    self._expand_multi_pipeline()
            else:
                # multi_pipeline - existing behavior
                self._expand_multi_pipeline()
        else:
            self.steps = [self.steps]
            self.generator_choices = [[]]

    def _convert_to_branched(self, steps: List[Any]) -> List[Any]:
        """Convert generator syntax to in-pipeline branches.

        Analyzes each step and determines whether to:
        - Wrap in branch (data-modifying steps)
        - Expand sequentially (models)
        - Keep as-is (no generators)
        """
        result = []

        for step in steps:
            if not self._has_gen_keys(step, skip_branch=True):
                result.append(step)
                continue

            # Determine step type
            step_type = self._classify_step(step)

            if step_type == "model":
                # Models: expand and inject sequentially
                expanded = expand_spec(step)
                result.extend(expanded)
            elif step_type in ("transform", "preprocessing", "feature_selection"):
                # Data-modifying: wrap in branch
                expanded = expand_spec(step)
                result.append({"branch": [[e] for e in expanded]})
            elif step_type == "splitter":
                # Splitters: cannot easily branch, raise warning
                logger.warning(
                    "Splitter variants in in_pipeline mode are experimental. "
                    "Consider using multi_pipeline mode for splitter variants."
                )
                # For now, just use first variant
                expanded = expand_spec(step)
                result.append(expanded[0])
            else:
                # Unknown: default to branch for safety
                expanded = expand_spec(step)
                result.append({"branch": [[e] for e in expanded]})

        return result

    def _classify_step(self, step: Any) -> str:
        """Classify a step by its type for expansion strategy."""
        if isinstance(step, dict):
            if "model" in step:
                return "model"
            if "transform" in step or "preprocessing" in step:
                return "transform"
            if "feature_selection" in step:
                return "feature_selection"
            if "y_processing" in step:
                return "y_processing"
            if "splitter" in step:
                return "splitter"
            if "branch" in step:
                return "branch"
            # Check for sklearn-style step (class in dict)
            if "class" in step:
                class_name = step["class"]
                # Heuristic: common model class patterns
                model_patterns = ["Regress", "Classif", "PLS", "SVM", "Random", "Gradient", "Neural"]
                if any(p in str(class_name) for p in model_patterns):
                    return "model"
                return "transform"
        # Check for class instances
        if hasattr(step, "fit") and hasattr(step, "predict"):
            return "model"
        if hasattr(step, "fit_transform") or hasattr(step, "transform"):
            return "transform"
        if hasattr(step, "split"):
            return "splitter"
        return "unknown"

    def _analyze_optimal_mode(self, steps: List[Any]) -> str:
        """Analyze steps to determine optimal generation mode."""
        variant_steps = []

        for step in steps:
            if self._has_gen_keys(step, skip_branch=True):
                step_type = self._classify_step(step)
                variant_steps.append((step, step_type))

        if not variant_steps:
            return "multi_pipeline"  # No variants, doesn't matter

        types = set(t for _, t in variant_steps)

        # Pure model variants → in_pipeline is more efficient
        if types == {"model"}:
            return "in_pipeline"

        # Has splitter variants → multi_pipeline safer
        if "splitter" in types:
            return "multi_pipeline"

        # Mixed → in_pipeline with branches
        return "in_pipeline"
```

---

## 7. Usage Examples

### 7.1 Current Behavior (Multi-Pipeline)

```python
# Generates 6 separate pipelines (2 preprocessors × 3 n_components)
pipeline = [
    ShuffleSplit(n_splits=5),
    {"transform": {"_or_": [SNV(), MSC()]}},
    PLSRegression(n_components={"_or_": [5, 10, 15]})
]

config = PipelineConfigs(pipeline)  # generation_mode="multi_pipeline" (default)
# config.steps has 6 entries
```

### 7.2 New In-Pipeline Mode

```python
# Single pipeline with branches
pipeline = [
    ShuffleSplit(n_splits=5),
    {"transform": {"_or_": [SNV(), MSC()]}},
    PLSRegression(n_components={"_or_": [5, 10, 15]})
]

config = PipelineConfigs(pipeline, generation_mode="in_pipeline")
# config.steps has 1 entry:
# [
#     ShuffleSplit(n_splits=5),
#     {"branch": [[SNV()], [MSC()]]},
#     PLSRegression(n_components=5),
#     PLSRegression(n_components=10),
#     PLSRegression(n_components=15),
# ]
```

### 7.3 Explicit Keyword Control

```python
# Mix behaviors in same pipeline
pipeline = [
    ShuffleSplit(n_splits=5),
    {"transform": {"_or_": [SNV(), MSC()], "_branch_": True}},  # Explicit branch
    {"model": {"_or_": [
        PLSRegression(n_components=5),
        PLSRegression(n_components=10),
    ], "_sequential_": True}}  # Explicit sequential
]

config = PipelineConfigs(pipeline)  # Keywords override default behavior
```

---

## 8. Considerations and Edge Cases

### 8.1 Branch Cartesian Product

When multiple branch points exist, the system already handles Cartesian products via `_multiply_branch_contexts`. In-pipeline generation should leverage this:

```python
# Two branch points → 2×3=6 branch combinations
pipeline = [
    {"branch": [[SNV()], [MSC()]]},  # 2 branches
    {"branch": [[PCA(10)], [PCA(20)], [PCA(30)]]}  # 3 branches
]
# Results in 6 branch paths: SNV+PCA10, SNV+PCA20, SNV+PCA30, MSC+PCA10, ...
```

### 8.2 Generator Choices Tracking

Currently, `generator_choices` tracks which values were selected for each pipeline. For in-pipeline mode, this should track per-branch:

```python
# In branch metadata
{
    "branch_id": 0,
    "name": "snv_pls5",
    "generator_choices": [
        {"_or_": SNV()},
        {"_or_": PLSRegression(n_components=5)}
    ]
}
```

### 8.3 Memory and Performance

**Multi-Pipeline:**
- Each pipeline loads dataset independently (if not cached)
- Parallel execution possible
- More artifacts stored

**In-Pipeline Branching:**
- Single dataset load
- Single splitter execution
- Branches share train/test split
- More memory efficient for many variants
- Better comparison (same exact splits)

### 8.4 Compatibility with Existing Features

| Feature | Multi-Pipeline | In-Pipeline Branch |
|---------|----------------|-------------------|
| Prediction tracking | ✅ Works | ✅ Works (branch_id in predictions) |
| Manifest system | ✅ Works | ✅ Works (single pipeline, many artifacts) |
| Export bundle | ✅ Works | ⚠️ Need branch-aware export |
| Retraining | ✅ Works | ⚠️ Need branch-aware retrain |
| Visualization | ✅ Works | ✅ Works (BranchAnalyzer) |

---

## 9. Testing Strategy

### 9.1 Unit Tests

1. **Keyword parsing tests** (`test_generator_keywords.py`)
   - `_branch_: true` creates branch step
   - `_sequential_: true` returns list of steps

2. **Mode conversion tests** (`test_pipeline_configs.py`)
   - `generation_mode="in_pipeline"` produces single config
   - Step classification is correct
   - Branching structure is valid

3. **Round-trip tests**
   - Multi-pipeline expansion → same results as in-pipeline branches

### 9.2 Integration Tests

1. **Example modification** (`Q23b_generator_branching.py`)
   - Demonstrate new keywords
   - Compare execution times
   - Verify predictions match

2. **Branching + generation** (`Q30_branching.py` extension)
   - Existing generator syntax inside branches
   - New `_branch_` keyword outside branches

---

## 10. Migration Path

### Phase 1: Opt-In (Non-Breaking)
- Add `generation_mode` parameter with default `"multi_pipeline"`
- Add new keywords `_branch_`, `_sequential_`
- Existing code unchanged

### Phase 2: Promotion
- Update examples to show new mode
- Add performance comparisons to docs
- Encourage new mode for non-splitter variants

### Phase 3: Default Change (Optional, Major Version)
- Consider changing default to `"auto"`
- Provide migration guide

---

## 11. Summary

### Key Finding: Branching Already Handles Almost Everything

The existing branching system properly isolates:
- ✅ X features (via snapshot/restore)
- ✅ Preprocessing chains (via context copy)
- ✅ Y processing (via context state)
- ✅ Feature selection (X columns are part of features snapshot)
- ✅ Sample filtering (works in practice, runs before branches)

**Only splitters require multi-pipeline generation** because they define the CV structure that must be shared across all branches.

### Effort Estimates

| Aspect | Effort Estimate | Priority |
|--------|-----------------|----------|
| Add `_branch_` keyword | Low (1-2 days) | High |
| Add `_sequential_` keyword | Low (1-2 days) | High |
| Add `generation_mode` param | Medium (2-3 days) | Medium |
| Auto mode logic | Low (1 day) | Medium |
| Enhanced tracking | Medium (3-5 days) | Low |
| Visualization updates | Low (1-2 days) | Low |

**Total Estimated Effort: 1.5-2 weeks**

### Recommended First Steps

1. Implement `_branch_` and `_sequential_` keywords in `or_strategy.py`
2. Add tests for new keywords
3. Update Q23_generator_syntax.py example
4. Gather feedback before implementing mode flag

---

## Appendix A: File Change Summary

| File | Changes |
|------|---------|
| `nirs4all/pipeline/config/_generator/keywords.py` | Add BRANCH_MODIFIER, SEQUENTIAL_MODIFIER constants |
| `nirs4all/pipeline/config/_generator/strategies/or_strategy.py` | Handle new keywords in expand() |
| `nirs4all/pipeline/config/generator.py` | Export new keyword constants |
| `nirs4all/pipeline/config/pipeline_config.py` | Add generation_mode param, conversion logic |
| `tests/unit/config/test_generator_keywords.py` | Tests for new keywords |
| `tests/unit/config/test_pipeline_configs_mode.py` | Tests for mode parameter |
| `examples/Q23b_generator_branching.py` | New example demonstrating feature |

---

## Appendix B: Alternative Naming Considered

| Current Proposal | Alternative 1 | Alternative 2 |
|------------------|---------------|---------------|
| `_branch_: true` | `_expand_mode_: "branch"` | `_variant_type_: "branch"` |
| `_sequential_: true` | `_expand_mode_: "inline"` | `_variant_type_: "sequential"` |
| `generation_mode` | `expansion_strategy` | `variant_handling` |

The current proposal uses clear, action-oriented names that align with existing keyword conventions (`_or_`, `_range_`, etc.).
