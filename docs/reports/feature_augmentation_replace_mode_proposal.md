# Feature Augmentation Action Modes Proposal

**Date:** December 2024
**Author:** nirs4all Development Team
**Status:** Proposal v1.1

---

## Table of Contents

1. [Objective](#objective)
2. [Current State](#current-state)
3. [Proposal](#proposal)
4. [Rationale](#rationale)
5. [Implementation Roadmap](#implementation-roadmap)

---

## Objective

Extend the `feature_augmentation` controller to support three distinct **action modes** that control how preprocessing operations interact with existing processings:

| Action | Description | Growth Pattern |
|--------|-------------|----------------|
| **extend** | Add new processings to the set (default) | Linear |
| **add** | Chain operations on existing processings, keep originals | Multiplicative |
| **replace** | Chain operations on existing processings, discard originals | Multiplicative (no originals) |

### Motivation Examples

**Extend mode (default) - Linear growth:**
```python
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "extend"}
]
# Produces: raw_A, raw_B
# (A already exists, so only B is added to the set)
```

**Add mode - Multiplicative with originals:**
```python
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "add"}
]
# Produces: raw_A, raw_A_A, raw_A_B
# (original kept + chained versions added)
```

**Replace mode - Multiplicative without originals:**
```python
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "replace"}
]
# Produces: raw_A_A, raw_A_B
# (original replaced by chained versions)
```

### Use Cases

1. **Extend (default)**: Build a flat set of preprocessing variants to compare. Ideal for exploring independent preprocessing options.

2. **Add**: Explore combinations while keeping base processings as reference. Useful for ablation studies.

3. **Replace**: Build sequential preprocessing chains without exponential growth. Ideal for multi-stage pipelines.

---

## Current State

### Architecture Overview

The `FeatureAugmentationController` ([feature_augmentation.py](../../nirs4all/controllers/data/feature_augmentation.py)) manages feature preprocessing augmentation. Currently, it implements what we now call the **"add"** behavior:

#### Current Flow (Add Mode)

```
Pipeline: [StandardScaler(), {"feature_augmentation": [SNV, Gaussian]}]

1. StandardScaler applied → replaces "raw" with "scaled"
2. feature_augmentation iterates:
   a. SNV() → ADDS "scaled_SNV" processing (preserves "scaled")
   b. Gaussian() → ADDS "scaled_Gaussian" processing (preserves "scaled")

Result: 3 processings: ["scaled", "scaled_SNV", "scaled_Gaussian"]
```

#### Key Code Behavior

From `FeatureAugmentationController.execute()`:

```python
for i, operation in enumerate(step_info.original_step["feature_augmentation"]):
    # Reset to original processings for each operation
    source_processings = copy.deepcopy(original_source_processings)
    local_context = initial_context.copy()

    # Set add_feature=True → substeps ADD new processings
    local_context = local_context.with_metadata(add_feature=True)
    local_context = local_context.with_processing(copy.deepcopy(source_processings))

    # Execute substep (adds new processing)
    result = runtime_context.step_runner.execute(...)
```

**Key observations:**
- Each operation starts from the **same** original processing state
- `add_feature=True` causes substeps to **add** new processing dimensions
- Operations run in parallel (independent of each other), not sequentially
- Original processings are preserved alongside new ones

#### Current Sequential Behavior

When multiple `feature_augmentation` blocks are used sequentially (current "add" behavior):

```python
pipeline = [
    {"feature_augmentation": [PP1, PP2]},
    {"feature_augmentation": [PP3, PP4]},
]

# Current result:
# Step 1: raw → raw, raw_PP1, raw_PP2 (3 processings)
# Step 2: Each of the 3 processings gets PP3 and PP4 added:
#         raw, raw_PP3, raw_PP4,
#         raw_PP1, raw_PP1_PP3, raw_PP1_PP4,
#         raw_PP2, raw_PP2_PP3, raw_PP2_PP4
# Result: 9 processings (multiplicative growth with originals kept)
```

This multiplicative growth with originals kept is the current (and only) behavior.

---

## Proposal

### New API Syntax

Introduce an `action` parameter with three possible values:

```python
{"feature_augmentation": [PP1, PP2], "action": "extend"}   # Default
{"feature_augmentation": [PP1, PP2], "action": "add"}
{"feature_augmentation": [PP1, PP2], "action": "replace"}
```

### Action Semantics

#### 1. `extend` (Default)

**Behavior:** Add new processings to the existing set. If a processing already exists, it is not duplicated.

**Growth pattern:** Linear (set union)

```python
# Example 1: Simple extend
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "extend"}
]
# Initial: raw
# After A: raw_A
# After f_a: raw_A already exists, add raw_B
# Result: ["raw_A", "raw_B"]
```

```python
# Example 2: Sequential extend
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "extend"},
    {"feature_augmentation": [A, C], "action": "extend"}
]
# After A: raw_A
# After 1st f_a: raw_A, raw_B
# After 2nd f_a: raw_A already exists, raw_C is new
# Result: ["raw_A", "raw_B", "raw_C"]
```

**Use case:** Exploring independent preprocessing options without duplication.

---

#### 2. `add`

**Behavior:** Chain each augmentation operation on top of ALL existing processings. Keep original processings alongside new chained versions.

**Growth pattern:** Multiplicative with originals (n + n×m)

```python
# Example 1: Simple add
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "add"}
]
# Initial: raw
# After A: raw_A
# After f_a:
#   - Keep: raw_A
#   - Chain A on raw_A → raw_A_A
#   - Chain B on raw_A → raw_A_B
# Result: ["raw_A", "raw_A_A", "raw_A_B"]
```

```python
# Example 2: Sequential add
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "add"},
    {"feature_augmentation": [A, C], "action": "add"}
]
# After A: raw_A
# After 1st f_a: raw_A, raw_A_A, raw_A_B (3 processings)
# After 2nd f_a: For each of 3 processings, chain A and C:
#   - Keep: raw_A, raw_A_A, raw_A_B
#   - Chain A: raw_A_A, raw_A_A_A, raw_A_B_A
#   - Chain C: raw_A_C, raw_A_A_C, raw_A_B_C
# Result: ["raw_A", "raw_A_A", "raw_A_B",
#          "raw_A_A", "raw_A_A_A", "raw_A_B_A",  # Note: raw_A_A appears twice
#          "raw_A_C", "raw_A_A_C", "raw_A_B_C"]
# After deduplication: ["raw_A", "raw_A_A", "raw_A_B", "raw_A_A_A",
#                       "raw_A_A_C", "raw_A_B_A", "raw_A_B_C", "raw_A_C"]
```

**Use case:** Ablation studies where you need base processings as reference points.

---

#### 3. `replace`

**Behavior:** Chain each augmentation operation on top of ALL existing processings. Discard original processings (replace them with chained versions).

**Growth pattern:** Multiplicative without originals (n×m)

```python
# Example 1: Simple replace
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "replace"}
]
# Initial: raw
# After A: raw_A
# After f_a:
#   - Chain A on raw_A → raw_A_A
#   - Chain B on raw_A → raw_A_B
#   - DISCARD raw_A
# Result: ["raw_A_A", "raw_A_B"]
```

```python
# Example 2: Sequential replace
pipeline = [
    A,
    {"feature_augmentation": [A, B], "action": "replace"},
    {"feature_augmentation": [A, C], "action": "replace"}
]
# After A: raw_A
# After 1st f_a: raw_A_A, raw_A_B (originals replaced)
# After 2nd f_a: For each of 2 processings, chain A and C:
#   - Chain A on raw_A_A → raw_A_A_A
#   - Chain C on raw_A_A → raw_A_A_C
#   - Chain A on raw_A_B → raw_A_B_A
#   - Chain C on raw_A_B → raw_A_B_C
#   - DISCARD raw_A_A, raw_A_B
# Result: ["raw_A_A_A", "raw_A_A_C", "raw_A_B_A", "raw_A_B_C"]
```

**Use case:** Multi-stage preprocessing pipelines where you want clean chains without intermediate processings.

---

### Comparison Table

| Scenario | extend | add | replace |
|----------|--------|-----|---------|
| `A, {f_a:[A,B]}` | raw_A, raw_B | raw_A, raw_A_A, raw_A_B | raw_A_A, raw_A_B |
| Count | 2 | 3 | 2 |
| `A, {f_a:[A,B]}, {f_a:[A,C]}` | raw_A, raw_B, raw_C | raw_A, raw_A_A, raw_A_B, raw_A_C, raw_A_A_A, raw_A_A_C, raw_A_B_A, raw_A_B_C | raw_A_A_A, raw_A_A_C, raw_A_B_A, raw_A_B_C |
| Count | 3 | 8 | 4 |

### Processing Naming Convention

Chained preprocessings use the `_` separator:

| Original | Operation | Result |
|----------|-----------|--------|
| `raw` | A | `raw_A` |
| `raw_A` | B | `raw_A_B` |
| `raw_A_B` | C | `raw_A_B_C` |

This naming convention:
- Clearly shows the preprocessing chain order
- Allows parsing of the chain if needed
- Is consistent with existing codebase patterns

---

## Rationale

### Why Three Actions?

1. **extend** - The simplest case: "I want to try these preprocessing options independently."
2. **add** - Exploration with baselines: "I want to try chaining these on what I have, but keep the originals for comparison."
3. **replace** - Pure chaining: "I want to build preprocessing chains, discard intermediates."

### Design Decisions

#### Why `action` instead of `mode`?

The term "action" better conveys that this controls **what the controller does** with the processings:
- `extend` → extend the set
- `add` → add chained versions
- `replace` → replace with chained versions

#### Why `extend` as Default?

1. **Safest behavior**: No multiplicative growth, no loss of existing processings
2. **Most intuitive**: "Add these preprocessing options to my pipeline"
3. **Backward compatibility consideration**: Current behavior is closer to "add", but "extend" is simpler for most use cases

**Note:** If backward compatibility with existing pipelines is critical, we can make `add` the default instead.

#### Why Not Boolean Flags?

We considered:
```python
{"feature_augmentation": [...], "chain": True, "keep_originals": True}
```

But this creates ambiguity:
- `chain=False, keep_originals=True` → extend? unclear
- `chain=True, keep_originals=False` → replace
- `chain=True, keep_originals=True` → add

A single `action` parameter is clearer.

### Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| `action` parameter | Clear semantics, extensible | New concept to learn |
| Boolean flags | Familiar pattern | Ambiguous combinations |
| Separate controllers | Clear separation | Code duplication |
| Implicit detection | No API change | Unpredictable |

### Edge Cases

#### Edge Case 1: Extend with Existing Processing

```python
# Initial: ["raw_A"]
{"feature_augmentation": [A, B], "action": "extend"}
# A produces raw_A (already exists) → skip
# B produces raw_B (new) → add
# Result: ["raw_A", "raw_B"]
```

#### Edge Case 2: Empty Feature Augmentation

```python
{"feature_augmentation": [], "action": "replace"}
# Result: No change (no-op)
```

#### Edge Case 3: `None` Operation

```python
{"feature_augmentation": [None, B], "action": "add"}
# None should be skipped (no operation)
# Result: Original + B chained on original
```

#### Edge Case 4: Mixed Actions in Sequence

```python
pipeline = [
    {"feature_augmentation": [A, B], "action": "extend"},   # raw_A, raw_B
    {"feature_augmentation": [C], "action": "replace"},      # raw_A_C, raw_B_C
    {"feature_augmentation": [D, E], "action": "add"},       # raw_A_C, raw_B_C, raw_A_C_D, raw_A_C_E, raw_B_C_D, raw_B_C_E
]
# Result: 6 processings with clear chain history
```

---

## Implementation Roadmap

### Phase 1: Core Implementation (3-4 days)

| Task | Duration | Description |
|------|----------|-------------|
| 1.1 Parse `action` parameter | 0.5 day | Extract and validate `action` from step dict |
| 1.2 Implement `extend` mode | 1 day | Set-based addition with deduplication |
| 1.3 Implement `add` mode | 1 day | Chain on all processings, keep originals (current behavior, refactored) |
| 1.4 Implement `replace` mode | 1 day | Chain on all processings, discard originals |
| 1.5 Processing naming | 0.5 day | Ensure proper chain naming (`raw_A_B_C`) |

### Phase 2: Testing (2 days)

| Task | Duration | Description |
|------|----------|-------------|
| 2.1 Unit tests for each mode | 1 day | Isolated tests for extend/add/replace |
| 2.2 Sequential mode tests | 0.5 day | Test mode combinations in sequence |
| 2.3 Edge case tests | 0.5 day | Empty list, None, duplicates |

### Phase 3: Documentation & Examples (1 day)

| Task | Duration | Description |
|------|----------|-------------|
| 3.1 Update docstrings | 0.25 day | Controller documentation |
| 3.2 User guide section | 0.25 day | Document all three modes |
| 3.3 Example script | 0.5 day | `Q_feature_augmentation_modes.py` |

### Implementation Details

#### Modified `FeatureAugmentationController.execute()`

```python
def execute(self, step_info, dataset, context, runtime_context, ...):
    initial_context = context.copy()
    original_processings = copy.deepcopy(initial_context.selector.processing)
    all_artifacts = []

    # Parse action mode
    action = step_info.original_step.get("action", "extend")
    if action not in ("extend", "add", "replace"):
        raise ValueError(f"Invalid action: {action}. Must be 'extend', 'add', or 'replace'.")

    operations = step_info.original_step["feature_augmentation"]

    if action == "extend":
        # EXTEND MODE: Add new processings to set (no chaining)
        existing_processings = set(original_processings[0])
        for operation in operations:
            # Apply operation independently (not chained)
            local_context = initial_context.copy()
            local_context = local_context.with_metadata(add_feature=True)
            # Execute operation on raw/base
            result = runtime_context.step_runner.execute(operation, ...)
            # Add to set if not already present
            new_proc = result.updated_context.selector.processing[0][-1]
            if new_proc not in existing_processings:
                existing_processings.add(new_proc)
        # Final: unique set of processings

    elif action == "add":
        # ADD MODE: Chain on all existing, keep originals
        new_processings = list(original_processings[0])  # Start with originals
        for operation in operations:
            for proc_name in original_processings[0]:
                # Chain operation on this processing
                local_context = initial_context.copy()
                local_context = local_context.with_metadata(add_feature=True)
                local_context = local_context.with_processing([[proc_name]])
                result = runtime_context.step_runner.execute(operation, ...)
                # Collect new chained processing
                chained_proc = result.updated_context.selector.processing[0][-1]
                new_processings.append(chained_proc)
        # Final: originals + all chained versions

    elif action == "replace":
        # REPLACE MODE: Chain on all existing, discard originals
        new_processings = []
        for operation in operations:
            for proc_name in original_processings[0]:
                # Chain operation on this processing
                local_context = initial_context.copy()
                local_context = local_context.with_metadata(add_feature=True)
                local_context = local_context.with_processing([[proc_name]])
                result = runtime_context.step_runner.execute(operation, ...)
                # Collect new chained processing
                chained_proc = result.updated_context.selector.processing[0][-1]
                new_processings.append(chained_proc)
        # Final: only chained versions (originals discarded)
        # Need to remove original processings from dataset

    # Update context with final processing list
    context = context.with_processing([new_processings])
    return context, all_artifacts
```

### Timeline Summary

**Total: 6-7 days**

```
Day 1:   Parse action, implement extend mode
Day 2:   Implement add mode (refactor current behavior)
Day 3:   Implement replace mode
Day 4:   Processing naming, edge cases
Day 5:   Unit tests
Day 6:   Integration tests, sequential mode tests
Day 7:   Documentation, example script
```

### Dependencies

```
1.1 Parse action
    │
    ├──► 1.2 extend mode ──┐
    │                      │
    ├──► 1.3 add mode ─────┼──► 1.5 Processing naming
    │                      │         │
    └──► 1.4 replace mode ─┘         ▼
                                 2.1-2.3 Testing
                                     │
                                     ▼
                                 3.1-3.3 Documentation
```

### Files to Modify

1. **[nirs4all/controllers/data/feature_augmentation.py](../../nirs4all/controllers/data/feature_augmentation.py)**
   - Add `action` parameter parsing
   - Implement extend/add/replace modes in `execute()`

2. **[nirs4all/pipeline/config/context.py](../../nirs4all/pipeline/config/context.py)** (if needed)
   - May need to track action mode in metadata

3. **[tests/unit/controllers/data/test_feature_augmentation.py](../../tests/unit/controllers/data/)** (new)
   - Unit tests for all three modes

4. **[tests/integration/pipeline/test_basic_pipeline.py](../../tests/integration/pipeline/test_basic_pipeline.py)**
   - Integration tests for action modes

5. **[examples/Q_feature_augmentation_modes.py](../../examples/)** (new)
   - Example demonstrating all three modes

6. **[Roadmap.md](../../Roadmap.md)**
   - Update to reflect new feature

---

## Summary

This proposal introduces an `action` parameter to the `feature_augmentation` controller with three modes:

| Action | Behavior | Use Case |
|--------|----------|----------|
| **extend** (default) | Add new processings to set | Explore independent options |
| **add** | Chain on all + keep originals | Ablation with baselines |
| **replace** | Chain on all + discard originals | Multi-stage pipelines |

### Quick Reference

```python
# Extend: Build a flat set of preprocessing options
{"feature_augmentation": [SNV, Gaussian, Detrend], "action": "extend"}

# Add: Chain while keeping originals for comparison
{"feature_augmentation": [FirstDerivative], "action": "add"}

# Replace: Pure chaining, clean pipeline stages
{"feature_augmentation": [PCA(50)], "action": "replace"}
```

### Migration Notes

- Current behavior is closest to `add` mode
- New default is `extend` for simpler semantics
- Existing pipelines may need `"action": "add"` to preserve current behavior

---

*Document generated for nirs4all project - December 2024 - v1.1*
