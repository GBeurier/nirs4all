# Multisource + Branching Reload: Deep Analysis and Fix Plan

**Date**: December 14, 2025
**Author**: Technical Analysis
**Status**: Investigation Complete - Ready for Implementation

---

## Executive Summary

The multisource + branching reload feature is failing because of a fundamental design issue in how execution traces record branch artifacts. The core problem is that **branch substeps are not recorded individually in the execution trace**, causing the minimal pipeline extractor to receive incorrect artifact mappings during prediction.

| Issue | Severity | Status |
|-------|----------|--------|
| Branch artifacts not traceable by branch_path | **CRITICAL** | 🔴 Root cause |
| Execution trace lumps all branch artifacts together | **CRITICAL** | 🔴 Root cause |
| X transformers not applied correctly during reload | **HIGH** | 🔴 Symptom |
| Y predictions are unscaled (work correctly) | **OK** | ✅ Working |
| MetaModel `_persist_model()` signature mismatch | **MEDIUM** | 🟡 Separate fix |

---

## Current Test Status

```
test_basic_branching_multisource          ❌ FAIL (NaN RMSE)
test_named_branching_multisource          ✅ PASS
test_branching_multisource_reload         ❌ FAIL (predictions mismatch)
test_sklearn_stacking_multisource         ✅ PASS
test_sklearn_stacking_multisource_reload  ✅ PASS (now passing!)
test_branching_with_stacking_multisource  ✅ PASS
test_in_branch_models_multisource         ✅ PASS
test_metamodel_multisource                ❌ FAIL (signature error)
test_metamodel_with_branches_multisource  ❌ FAIL (signature error)
```

---

## Deep Technical Analysis

### 1. The Problem: Execution Trace Does Not Track Branch Substeps

During training, the pipeline execution flow for branching is:

```
Step 1: MinMaxScaler (X transformer)  → recorded with branch_path=[]
Step 2: MinMaxScaler (Y transformer)  → recorded with branch_path=[]
Step 3: ShuffleSplit                  → recorded with branch_path=[]
Step 4: Branch Controller             → recorded with branch_path=[]
  ├─ Substep: SNV (branch_0)         → NOT recorded (artifacts lumped into Step 4)
  └─ Substep: SavGol (branch_1)      → NOT recorded (artifacts lumped into Step 4)
Step 5: PLSRegression (branch_0)      → recorded with branch_path=[0]
Step 5: PLSRegression (branch_1)      → recorded with branch_path=[1]
```

**The issue:** Step 4 (branch) is recorded as a single step with **ALL branch artifacts** listed together. The execution trace has NO way to know which artifacts belong to which branch.

Looking at the manifest, we see:

```yaml
- step_index: 4
  operator_type: branch
  operator_class: list
  branch_path: []    # ❌ Empty - no branch info!
  artifacts:
    artifact_ids:
    - 0162_...:0:4.1:all  # SNV for branch 0
    - 0162_...:0:4.2:all  # SNV for branch 0
    - 0162_...:0:4.3:all  # SNV for branch 0
    - 0162_...:1:4.4:all  # SavGol for branch 1
    - 0162_...:1:4.5:all  # SavGol for branch 1
    - 0162_...:1:4.6:all  # SavGol for branch 1
```

The artifact IDs **do contain branch info** (the `:0:` and `:1:` parts), but the execution trace step doesn't parse or expose this.

### 2. How Prediction Fails

During prediction, `TraceBasedExtractor.extract_for_branch()` is called:

```python
def extract_for_branch(self, trace, branch_path, full_pipeline):
    for exec_step in trace_steps:
        # Include step if it has matching branch_name or no branch
        if not exec_step.branch_path:
            include_step = True  # ❌ All shared steps included
        elif exec_step.branch_path == branch_path:
            include_step = True
        ...
```

For Step 4 (branch), `exec_step.branch_path = []`, so it's included as a "shared" step. But this means **ALL branch artifacts** (for both branches) are loaded and passed to the transformer controller.

### 3. What Happens in TransformerMixinController

During prediction:

```python
# transformer.py:206-221
if runtime_context.artifact_provider is not None:
    step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
        step_index,
        branch_path=context.selector.branch_path  # e.g., [0] for branch_0
    )
    if step_artifacts:
        transformer = _find_transformer_by_class(
            operator_name,
            artifacts_dict,
            global_transformer_index,
            artifacts_list=step_artifacts
        )
```

The `MinimalArtifactProvider.get_artifacts_for_step()` returns artifacts, but it doesn't filter by branch_path because the execution trace step doesn't have branch metadata!

```python
# minimal_predictor.py:121-135
def get_artifacts_for_step(self, step_index, branch_path=None, branch_id=None):
    step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
    if not step_artifacts:
        return []

    step = self.minimal_pipeline.get_step(step_index)
    for artifact_id in step_artifacts.artifact_ids:
        # Filter by branch path if specified
        if branch_path is not None and step:
            if step.branch_path and step.branch_path != branch_path:
                continue  # ❌ step.branch_path is [] so this check passes!
```

### 4. The Wrong Transformers Get Applied

Since filtering doesn't work, `_find_transformer_by_class()` is called with **all** artifacts:

```python
def _find_transformer_by_class(class_name, binaries_dict, search_index, artifacts_list):
    # Finds the N-th transformer matching class_name
    matches.sort(key=lambda x: x[0])  # Sorted by operation number
    if search_index < len(matches):
        return matches[search_index][2]
```

With 3 sources × 2 branches = 6 SNV/SavGol transformers in the list:
- Branch 0 prediction should get SNV artifacts 4.1, 4.2, 4.3
- Branch 1 prediction should get SavGol artifacts 4.4, 4.5, 4.6

But since all are in the list and sorted by operation number, **branch 0 gets the first 3 (correct)** but the order isn't guaranteed for more complex scenarios. More importantly, the issue is that **the branch step artifacts themselves contain both branch's artifacts**.

### 5. Why Sklearn Stacking Reload Works

The stacking test (`test_sklearn_stacking_multisource_reload`) works because:
1. There's no branching, so no branch artifact mixing
2. The StackingRegressor is a single model step with single artifact
3. X preprocessing (MinMaxScaler) is pre-branch and correctly loaded

---

## Root Cause Summary

**Primary Root Cause:**
The `BranchController.execute()` method does not record individual substeps in the execution trace. All artifacts created inside branches are lumped under the parent "branch" step without branch_path context.

**Secondary Issue:**
The `MinimalArtifactProvider` cannot filter artifacts by branch_path because:
1. The execution trace step has `branch_path: []`
2. Artifact filtering relies on step metadata, not artifact ID parsing

---

## Proposed Fix Architecture

### Option A: Record Branch Substeps in Execution Trace (Recommended)

**Approach:** Modify `BranchController.execute()` to record each branch's internal steps as separate trace entries with correct `branch_path`.

**Changes:**

1. **[branch.py]** - When executing substeps in branches, record each substep:

```python
# In BranchController.execute(), for each substep:
for substep in branch_steps:
    # Record branch substep start in trace
    if runtime_context.trace_recorder:
        runtime_context.trace_recorder.start_step(
            step_index=runtime_context.step_number,
            substep_index=runtime_context.substep_number,
            operator_type=...,
            operator_class=...,
            branch_path=new_branch_path,  # [branch_id]
            branch_name=branch_name
        )

    result = runtime_context.step_runner.execute(...)

    # Record substep end
    if runtime_context.trace_recorder:
        runtime_context.trace_recorder.end_step()
```

2. **[execution_trace.py]** - Add `substep_index` to `ExecutionStep`:

```python
@dataclass
class ExecutionStep:
    step_index: int
    substep_index: Optional[int] = None  # New field
    ...
```

3. **[extractor.py]** - Filter by branch_path when extracting:

```python
def extract_for_branch(self, trace, branch_path, full_pipeline):
    for exec_step in trace_steps:
        # Include if shared (no branch) OR matches our branch
        if not exec_step.branch_path:
            include_step = True
        elif self._branch_path_matches(branch_path, exec_step.branch_path):
            include_step = True
```

4. **[minimal_predictor.py]** - Properly filter artifacts:

```python
def get_artifacts_for_step(self, step_index, branch_path=None, branch_id=None):
    step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)

    # Parse artifact IDs to filter by branch
    results = []
    for artifact_id in step_artifacts.artifact_ids:
        artifact_branch = self._parse_branch_from_artifact_id(artifact_id)
        if branch_path is None or artifact_branch == branch_path[0]:
            obj = self._load_artifact(artifact_id)
            results.append((artifact_id, obj))
    return results
```

### Option B: Parse Branch from Artifact ID (Simpler but Less Clean)

**Approach:** Don't change trace recording, but filter artifacts by parsing the artifact_id format.

Artifact IDs already contain branch info:
- `0162_...:0:4.1:all` → branch 0
- `0162_...:1:4.4:all` → branch 1

**Changes:**

1. **[minimal_predictor.py]** - Parse artifact_id to extract branch:

```python
def _parse_branch_from_artifact_id(self, artifact_id: str) -> Optional[int]:
    """Parse branch index from artifact ID.

    Format: pipeline_id:branch:step.substep:fold
    Example: "0162_abc123:0:4.1:all" → branch=0
    """
    parts = artifact_id.split(":")
    if len(parts) >= 3:
        # Check if second part is numeric (branch index)
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def get_artifacts_for_step(self, step_index, branch_path=None, branch_id=None):
    step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
    if not step_artifacts:
        return []

    results = []
    target_branch = None
    if branch_path:
        target_branch = branch_path[0]
    elif branch_id is not None:
        target_branch = branch_id

    for artifact_id in step_artifacts.artifact_ids:
        # Filter by branch if specified
        if target_branch is not None:
            artifact_branch = self._parse_branch_from_artifact_id(artifact_id)
            if artifact_branch is not None and artifact_branch != target_branch:
                continue  # Skip artifacts from other branches

        obj = self._load_artifact(artifact_id)
        if obj is not None:
            results.append((artifact_id, obj))

    return results
```

2. **[artifact_loader.py]** - Update `load_for_step()` to filter by branch:

```python
def load_for_step(self, step_index, branch_path=None, fold_id=None, pipeline_id=None):
    results = []
    target_branch = branch_path[0] if branch_path else None

    for artifact_id, record in self._artifacts.items():
        if record.step_index != step_index:
            continue

        # Check branch_path in record
        if target_branch is not None:
            if record.branch_path:
                if record.branch_path[0] != target_branch:
                    continue
        ...
```

---

## Recommended Implementation Plan

### Phase 1: Quick Fix via Artifact ID Parsing (1-2 hours)

**Goal:** Get `test_branching_multisource_reload` passing with minimal code changes.

1. Update `MinimalArtifactProvider.get_artifacts_for_step()` to parse branch from artifact_id
2. Update `ArtifactLoader.load_for_step()` to respect branch_path filtering
3. Add helper function `_parse_branch_from_artifact_id()`

**Files to modify:**
- `nirs4all/pipeline/minimal_predictor.py`
- `nirs4all/pipeline/storage/artifacts/artifact_loader.py`

### Phase 2: MetaModel Signature Fix (15 min)

Add `custom_name` parameter to `MetaModelController._persist_model()`.

**Files to modify:**
- `nirs4all/controllers/models/meta_model.py`

### Phase 3: NaN RMSE Fix (1 hour)

Investigate why `test_basic_branching_multisource` reports NaN RMSE for some branches.

**Files to investigate:**
- `nirs4all/controllers/data/branch.py` (branch context y_processing)
- `nirs4all/core/metrics.py` (metric computation)

### Phase 4: Clean Architecture (Long-term)

Implement Option A to properly record branch substeps in the execution trace for a cleaner, more maintainable solution.

---

## Test Validation Checklist

After implementing fixes, run:

```bash
# Core branching + multisource tests
pytest tests/integration/pipeline/test_multisource_branching_stacking.py -v

# Expected results after Phase 1+2:
test_basic_branching_multisource          ✅ (if Phase 3 done)
test_named_branching_multisource          ✅
test_branching_multisource_reload         ✅
test_sklearn_stacking_multisource         ✅
test_sklearn_stacking_multisource_reload  ✅
test_branching_with_stacking_multisource  ✅
test_in_branch_models_multisource         ✅
test_metamodel_multisource                ✅ (if Phase 2 done)
test_metamodel_with_branches_multisource  ✅ (if Phase 2 done)
```

---

## Code References

### Key Files and Functions

| File | Function | Role |
|------|----------|------|
| `pipeline/predictor.py` | `_predict_with_minimal_pipeline()` | Entry point for minimal prediction |
| `pipeline/minimal_predictor.py` | `MinimalArtifactProvider.get_artifacts_for_step()` | **Critical fix location** |
| `pipeline/trace/extractor.py` | `extract_for_branch()` | Filters steps by branch |
| `controllers/data/branch.py` | `BranchController.execute()` | Creates branch contexts |
| `controllers/transforms/transformer.py` | `execute()` predict mode | Loads and applies transformers |
| `pipeline/storage/artifacts/artifact_loader.py` | `load_for_step()` | Loads artifacts by context |

### Artifact ID Format

```
{pipeline_id}:{branch}:{step}.{substep}:{fold}

Examples:
- "0162_abc:1.1:all"      → Step 1, substep 1, no branch, all folds
- "0162_abc:0:4.1:all"    → Branch 0, step 4, substep 1, all folds
- "0162_abc:0:5:0"        → Branch 0, step 5, fold 0
```

---

## Conclusion

The multisource + branching reload failure is caused by insufficient branch context in the execution trace. The recommended fix is to parse branch information from artifact IDs during prediction, which provides a clean solution without major architectural changes.

The fix should be implemented in `MinimalArtifactProvider.get_artifacts_for_step()` to filter artifacts by branch when `branch_path` is specified, using the branch index encoded in the artifact ID.
