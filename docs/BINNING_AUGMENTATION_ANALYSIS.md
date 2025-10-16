# Binning-Based Balanced Augmentation - Analysis & Design Summary

**Date**: October 16, 2025
**Status**: Design Phase Complete ✅
**Implementation Status**: Ready to Begin ⏳

---

## Executive Summary

This document summarizes the comprehensive analysis and design for enhancing nirs4all's balanced augmentation with **binning support for regression contexts**.

### What We're Building

A feature that lets users apply balanced augmentation to **continuous regression targets** by automatically dividing them into virtual "bins" (classes), ensuring augmentation is distributed across the full range of y values.

### Current State (Before)

```yaml
# Balanced augmentation (works great for classification)
sample_augmentation:
  balance: "y"
  max_factor: 1.0

# For regression: Treats each unique y value as separate class ❌
# 100 unique values = 100 "classes" → meaningless balancing
```

### Desired State (After)

```yaml
# Balanced augmentation for regression (NEW!)
sample_augmentation:
  balance: "y"
  bins: 10                      # NEW: Divide y into 10 virtual classes
  binning_strategy: "quantile"  # NEW: Equal probability per bin
```

---

## Key Insights from Code Analysis

### 1. Current Architecture is Highly Modular

**Finding**: The existing sample augmentation is well-designed with clear separation of concerns:

```
SampleAugmentationController (orchestration)
    ↓ delegates to
BalancingCalculator (calculation)
    ↓ works with
TransformerMixinController (execution)
    ↓ stored by
Dataset API (persistence)
```

**Implication**: Binning can be added cleanly at the controller level with minimal disruption.

### 2. BalancingCalculator is Label-Agnostic

**Finding**: `calculate_balanced_counts()` works with ANY discrete labels:
- Classification: `[0, 1, 2]`
- Strings: `['cat', 'dog', 'bird']`
- Binned regression: `[0, 1, 2, 3, ...]` ← We can add this!

**Implication**: No changes needed to BalancingCalculator; just feed it binned labels.

### 3. Leak Prevention is Robust

**Finding**: CV splits use `include_augmented=False` to select only base samples for splitting.

**Implication**: Binning happens before splitting, so leak prevention is unaffected.

### 4. Task Detection is Needed

**Finding**: Current system doesn't distinguish regression from classification.

**Implication**: Need automatic task type detection (look at y dtype, unique value ratio).

---

## Design Overview

### Core Components

#### 1. BinningCalculator (NEW)
**Purpose**: Divide continuous y into discrete bins

**Methods**:
- `bin_continuous_targets(y, bins, strategy, random_state)` - Main entry point
- `quantile_binning(y, bins)` - Equal probability bins
- `equal_width_binning(y, bins)` - Uniform width bins
- `validate_bins_param(bins)` - Input validation

**Location**: `nirs4all/utils/binning.py`

#### 2. Enhanced SampleAugmentationController
**Changes**: Add binning logic to `_execute_balanced()` method

**Key Logic**:
```python
if "bins" in config or "binning_strategy" in config:
    # 1. Bin continuous y values
    # 2. Replace y labels with bin indices
    # 3. Proceed with existing balanced logic
else:
    # 1. Use existing discrete balancing
```

**Location**: `nirs4all/controllers/dataset/op_sample_augmentation.py`

#### 3. Unchanged BalancingCalculator
**Why**: Works with any discrete labels! No changes needed.

---

## Binning Strategies

### Strategy 1: Quantile Binning (Default)

**How it works**:
```python
np.quantile(y, [0, 1/n, 2/n, ..., 1])
```

**Characteristics**:
- Equal probability per bin (~n_samples/bins per bin)
- Adapts to data distribution
- Better for skewed data

**Example**:
```
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bins = 3

Quantiles: [0%, 33%, 67%, 100%] → edges: [1, 3.67, 7.33, 10]

Result:
- Bin 0: [1, 2, 3] (3 samples)
- Bin 1: [4, 5, 6, 7] (4 samples)
- Bin 2: [8, 9, 10] (3 samples)
```

### Strategy 2: Equal Width Binning

**How it works**:
```python
np.linspace(y.min(), y.max(), bins+1)
```

**Characteristics**:
- Uniform width across range
- Predictable bin boundaries
- Better for uniform data

**Example**:
```
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bins = 3

Width = (10-1)/3 = 3 per bin
Edges: [1, 4, 7, 10]

Result:
- Bin 0: [1, 2, 3] (3 samples)
- Bin 1: [4, 5, 6] (3 samples)
- Bin 2: [7, 8, 9, 10] (4 samples)
```

---

## API Design

### Configuration Format

#### YAML
```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10                    # NEW
  binning_strategy: "quantile"  # NEW
  max_factor: 2.0             # Works with binning
  random_state: 42
```

#### JSON
```json
{
  "sample_augmentation": {
    "transformers": [{"StandardScaler": {}}],
    "balance": "y",
    "bins": 10,
    "binning_strategy": "quantile",
    "max_factor": 2.0
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `balance` | str | Required | Must be `"y"` to enable binning |
| `bins` | int | None | Number of virtual classes (1-1000) |
| `binning_strategy` | str | `"quantile"` | Strategy: quantile or equal_width |
| `max_factor` | float | 1.0 | Max augmentation multiplier (works with bins) |
| `selection` | str | `"random"` | Transformer selection method |
| `random_state` | int | None | Random seed |

---

## Data Flow

```
User provides continuous regression data (y ∈ [0, 100])
    │
    ▼
SampleAugmentationController.execute()
    │
    ├─ Detect: balance="y" AND bins parameter provided
    │
    ▼
BinningCalculator.bin_continuous_targets()
    │
    ├─ Apply strategy (quantile or equal_width)
    │
    ▼
Returns bin indices: [0, 0, 1, 1, 2, 2, 3, 3, 3]
    │
    ▼
Replace y with bin indices
    │
    ▼
BalancingCalculator.calculate_balanced_counts()
    │
    ├─ Treats bins like discrete classes
    │
    ▼
Augmentation counts: {sample_0→2, sample_1→1, ...}
    │
    ▼
Apply transformers (existing logic)
    │
    ▼
Dataset stores augmented samples with origin tracking
    │
    ▼
CV splitting (uses include_augmented=False) → LEAK-FREE!
```

---

## Implementation Roadmap

### Phase 1: BinningCalculator (1-2 days)
- Create `nirs4all/utils/binning.py`
- Implement binning strategies
- Add validation and edge case handling
- Write ~20 unit tests

**Deliverable**: Pure utility module, independently testable

### Phase 2: Controller Integration (2-3 days)
- Modify `SampleAugmentationController`
- Add binning detection logic
- Integrate with balanced mode
- Add controller-level tests

**Deliverable**: Binning support in sample augmentation

### Phase 3: BalancingCalculator (1 day)
- Verify compatibility (no code changes needed)
- Update docstrings
- Add clarification tests

**Deliverable**: Documentation clarifying generic label support

### Phase 4: End-to-End Testing (2-3 days)
- Create comprehensive E2E tests
- Test full pipeline integration
- Verify leak prevention
- Test regression/classification detection

**Deliverable**: Production-ready test suite

### Phase 5: Documentation (1-2 days)
- Update SAMPLE_AUGMENTATION.md
- Update quick reference guide
- Create example script (Q13)
- Add best practices

**Deliverable**: Complete documentation and examples

### Phase 6: Validation (1 day)
- Code review
- Performance testing
- Edge case validation
- Final polish

**Deliverable**: Ready for merge!

**Total Timeline**: ~8-12 days (sequential)

---

## Key Design Decisions

### ✅ Decision 1: Optional Binning
**Rationale**: Backward compatibility + gradual adoption
**Implication**: Existing configs work unchanged; users opt-in to binning

### ✅ Decision 2: Binning at Controller Level
**Rationale**: Keeps binning logic localized; doesn't touch dataset API
**Implication**: Minimal coupling; easy to extend later

### ✅ Decision 3: Quantile as Default
**Rationale**: Better for most real-world distributions
**Implication**: Sensible defaults; works for 90% of use cases

### ✅ Decision 4: Automatic Task Type Detection
**Rationale**: User convenience; YAML config self-contained
**Implication**: Heuristic needed; documented behavior for edge cases

### ✅ Decision 5: Reusable BinningCalculator
**Rationale**: Utility could be useful beyond augmentation
**Implication**: Independent module; testable in isolation

---

## Backward Compatibility

✅ **100% Backward Compatible**

**Existing code continues to work**:
```yaml
# Old (still works)
sample_augmentation:
  balance: "y"
  max_factor: 1.0

# New capability
sample_augmentation:
  balance: "y"
  bins: 10              # Only applies if present
  binning_strategy: "quantile"
```

**No breaking changes**:
- All new parameters optional
- Default behavior unchanged
- Classification unaffected
- Existing tests pass

---

## Testing Strategy

### Unit Tests (~60 tests total)

**test_binning.py** (~20 tests):
- Quantile binning correctness
- Equal width binning correctness
- Edge cases (NaN, constant y, single bin)
- Bin boundary validation
- Reproducibility

**test_sample_augmentation_controller.py** (+7 tests):
- Balanced mode with binning
- Both strategies
- Parameter validation
- Classification unaffected

**test_balancing.py** (+1 test):
- Compatibility with bin indices

### Integration Tests (~10 tests)

**test_binning_augmentation_e2e.py** (NEW):
- Full pipeline with binning
- Leak prevention maintained
- Regression/classification detection
- Strategy comparison
- Real-world scenarios

### Total Coverage
- >90% code coverage for new code
- All edge cases tested
- Performance validated
- Backward compatibility verified

---

## File Changes Summary

### New Files (4)
1. `nirs4all/utils/binning.py` - BinningCalculator class
2. `tests/unit/test_binning.py` - Binning tests
3. `tests/integration/test_binning_augmentation_e2e.py` - E2E tests
4. `examples/Q13_binned_balanced_augmentation.py` - Example script

### Modified Files (4)
1. `nirs4all/controllers/dataset/op_sample_augmentation.py` (+50 lines)
2. `nirs4all/utils/balancing.py` (+10 lines docstring)
3. `docs/SAMPLE_AUGMENTATION.md` (+150 lines)
4. `docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md` (+50 lines)

### Documentation Files (3)
1. `docs/BINNING_AUGMENTATION_DESIGN.md` - Comprehensive design
2. `docs/BINNING_AUGMENTATION_ROADMAP.md` - Implementation roadmap
3. `docs/BINNING_AUGMENTATION_ARCHITECTURE.md` - Visual architecture

**Total**: ~10 files, ~700 lines of code, ~300 lines of tests, ~400 lines of documentation

---

## Risk Assessment

### Low Risk ✅
- BinningCalculator is pure utility (no side effects)
- No changes to dataset API
- Fully optional (no impact if unused)
- Controller changes isolated

### Medium Risk ⚠️
- Task type detection heuristic (could misclassify edge cases)
- Integration with balanced logic (well-tested path though)

### Mitigation
- Comprehensive unit tests (edge cases)
- Clear logging of task type detection
- Extensive validation testing
- User documentation

### Overall Risk Level: **LOW** 🟢
- Mature architecture in place
- Well-tested patterns to build on
- Conservative, incremental changes
- Extensive testing planned

---

## Success Criteria

### Functional
- ✅ Quantile binning produces correct equal-probability bins
- ✅ Equal-width binning produces correct uniform-width bins
- ✅ Augmentation counted correctly per bin
- ✅ Classification tasks unaffected
- ✅ Leak prevention maintained
- ✅ Backward compatibility verified

### Quality
- ✅ >90% test coverage
- ✅ All edge cases handled
- ✅ <5% performance overhead
- ✅ Clear error messages

### Documentation
- ✅ API fully documented
- ✅ Binning strategies explained
- ✅ Working examples provided
- ✅ Best practices guide

---

## Quick Start for Implementation

### Step 1: Create BinningCalculator
```bash
# Create file: nirs4all/utils/binning.py
# Implement:
# - BinningCalculator.bin_continuous_targets()
# - BinningCalculator.quantile_binning()
# - BinningCalculator.equal_width_binning()
# - BinningCalculator.validate_bins_param()
```

### Step 2: Integrate with Controller
```python
# In SampleAugmentationController._execute_balanced()
if "bins" in config:
    binned_labels = BinningCalculator.bin_continuous_targets(...)
    # Use binned_labels instead of original y
```

### Step 3: Test Everything
```bash
# Write unit tests for BinningCalculator
# Write integration tests for full pipeline
# Run all existing tests (should still pass)
```

### Step 4: Document
```markdown
# Add to SAMPLE_AUGMENTATION.md:
## Binning-Based Balanced Mode (Regression)
[Examples and explanation]
```

---

## Supporting Documents

This design is supported by three detailed documents:

1. **BINNING_AUGMENTATION_DESIGN.md** (This File)
   - Comprehensive technical specification
   - Architecture details
   - API design
   - Risk assessment

2. **BINNING_AUGMENTATION_ROADMAP.md**
   - Implementation phases
   - Timeline estimates
   - File structure
   - Practical checklists

3. **BINNING_AUGMENTATION_ARCHITECTURE.md**
   - Visual diagrams
   - Data flow examples
   - Quick reference
   - Decision tree

---

## Next Steps

1. **Review & Approve Design**
   - Review this document
   - Review supporting documents
   - Approve architecture

2. **Start Implementation**
   - Phase 1: BinningCalculator
   - Phase 2-6: Follow roadmap

3. **Code Review Checkpoints**
   - After each phase
   - Focus on design consistency
   - Verify tests pass

4. **Merge to Main**
   - After all phases complete
   - All tests passing
   - Documentation complete
   - Code review approved

---

## Questions & Clarifications

### Q: Will this break existing augmentation pipelines?
**A**: No! Binning is optional. Existing configs work unchanged.

### Q: What if my y values are already discrete?
**A**: System detects this and skips binning automatically.

### Q: Can I use binning with metadata columns?
**A**: Not in v1, only with `balance: "y"`. Can be extended later.

### Q: What's the performance impact?
**A**: Minimal. Binning is ~O(n log n) (one-time, before augmentation).

### Q: How do I choose bins vs binning_strategy?
**A**: Use `bins: 10` (default quantile) for most cases. See best practices guide.

---

## Conclusion

This feature is **well-designed, low-risk, and ready for implementation**. The architecture leverages the existing robust foundation while adding meaningful new capability for regression tasks.

**Status**: ✅ Design Phase Complete
**Recommendation**: ✅ Approve for Implementation
**Timeline**: ~8-12 days

---

## Document References

- **Full Design**: `docs/BINNING_AUGMENTATION_DESIGN.md`
- **Roadmap**: `docs/BINNING_AUGMENTATION_ROADMAP.md`
- **Architecture**: `docs/BINNING_AUGMENTATION_ARCHITECTURE.md`
- **Current Code**: `nirs4all/controllers/dataset/op_sample_augmentation.py`
- **Current Tests**: `tests/unit/test_sample_augmentation_controller.py`
