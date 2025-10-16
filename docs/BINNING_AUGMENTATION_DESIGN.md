# Binning-Based Balanced Augmentation Design Document

## Executive Summary

This document outlines the design for enhancing the balanced augmentation feature with **binning support** in regression contexts. When `balance: "y"` is specified with regression data, users can now treat continuous target values as virtual "bins" (classes) for balancing purposes, ensuring augmentation is distributed across the full target range.

**Status**: Design Phase
**Target Version**: Post-Sample-Augmentation-v1
**Complexity**: Medium

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Feature Overview](#feature-overview)
3. [Architecture](#architecture)
4. [API Design](#api-design)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)
7. [Documentation Updates](#documentation-updates)
8. [Backward Compatibility](#backward-compatibility)

---

## Problem Statement

### Current Limitation

The current balanced augmentation feature works excellently for **classification tasks** where samples naturally belong to discrete classes. However, in **regression contexts** with continuous targets (e.g., y ∈ [0, 100]), the current implementation treats each unique y value as a separate "class":

```python
# Example: Regression dataset with continuous y
y = [10.5, 10.7, 15.2, 15.8, 50.1, 50.3, 50.5, 95.0, 95.2]

# Current behavior: 9 "classes" (one per unique value)
# Balance result: Tries to make each value equally represented
# Problem: Too granular, not meaningful for continuous data
```

### Use Cases for Binning

1. **Continuous Target Balancing**: Ensure augmentation covers the full range of y values
2. **Regression with Class-like Structure**: Targets naturally fall into ranges (e.g., low/medium/high grades)
3. **Non-linear Distribution**: Quantile binning adapts to actual data distribution
4. **Interpretability**: Bins are easier to understand than individual target values

---

## Feature Overview

### Proposed Enhancement

Add optional `bins` and `binning_strategy` parameters to balanced augmentation:

```yaml
# Balanced augmentation with binning (for regression)
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10                    # NEW: Number of bins for continuous y
  binning_strategy: "quantile"  # NEW: Strategy for binning
  max_factor: 2.0
  random_state: 42
```

### Key Behaviors

1. **Binning Only for Regression**: Automatically detect task type; apply binning only if y is continuous
2. **Virtual Classes from Bins**: Treat each bin as a virtual class for balancing
3. **Smart Sample Selection**: Samples are randomly selected from the pool of binned samples when augmenting
4. **Configurable Strategy**: Support multiple binning approaches (quantile, equal-width, custom)

---

## Architecture

### Component Interactions

```
SampleAugmentationController
    ├─ Check balance source is "y"
    ├─ If bins/binning_strategy provided:
    │  ├─ Load y values
    │  ├─ Detect task type (classification vs regression)
    │  ├─ Call BinningCalculator.bin_continuous_targets()
    │  ├─ Get binned labels (virtual classes)
    │  └─ Proceed with balanced counting using binned labels
    └─ Else:
       └─ Use existing balanced logic (discrete classes)

BinningCalculator (NEW)
    ├─ bin_continuous_targets()
    │  ├─ Validate bins parameter
    │  ├─ Apply binning strategy
    │  └─ Return bin assignments
    ├─ quantile_binning()
    │  ├─ np.quantile for equal probability bins
    │  └─ Return bin edges
    ├─ equal_width_binning()
    │  ├─ Uniform width across range
    │  └─ Return bin edges
    └─ validate_bins_param()
       └─ Type checking & range validation

BalancingCalculator (MODIFIED)
    └─ calculate_balanced_counts()
       └─ Works with binned labels same as classification
```

### Data Flow with Binning

```
y values (continuous)
    ↓ (1. Detect regression)
Regression detected
    ↓ (2. Apply binning strategy)
y_binned = [1, 1, 2, 2, 3, 3, 3, 4, 4]
    ↓ (3. Use binned labels for balancing)
BalancingCalculator.calculate_balanced_counts(y_binned, ...)
    ↓ (4. Get augmentation counts)
augmentation_counts = {sample_id → count, ...}
    ↓ (5. Proceed normally)
Transform & create augmented samples
```

### New Class: BinningCalculator

```python
class BinningCalculator:
    """Utilities for binning continuous regression targets."""

    @staticmethod
    def bin_continuous_targets(
        y: np.ndarray,
        bins: int = 10,
        strategy: str = "quantile",
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin continuous y values into virtual classes.

        Args:
            y: Continuous target values
            bins: Number of bins (1-based bin indices returned)
            strategy: Binning approach
                - "quantile": Equal probability bins (default)
                - "equal_width": Equal width bins
                - "custom": Future extension point
            random_state: Random seed for reproducibility

        Returns:
            (bin_indices, bin_edges)
            - bin_indices: [0, 0, 1, 1, 2, 2, 2, 3, 3] (bin assignment per sample)
            - bin_edges: Boundaries for each bin
        """
        # Validation
        # Apply strategy
        # Return bins and edges
        pass

    @staticmethod
    def quantile_binning(y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """np.quantile based binning - equal probability per bin."""
        pass

    @staticmethod
    def equal_width_binning(y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Linear space binning - equal width per bin."""
        pass

    @staticmethod
    def validate_bins_param(bins: Any) -> int:
        """Validate and coerce bins parameter."""
        pass
```

---

## API Design

### Pipeline Configuration

#### YAML Format

```yaml
# Basic binning (defaults to quantile, 10 bins)
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10
  binning_strategy: "quantile"

# With max_factor and seed
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  balance: "y"
  bins: 5
  binning_strategy: "equal_width"
  max_factor: 2.0
  selection: "random"
  random_state: 42

# Metadata binning (future extension - not in v1)
# sample_augmentation:
#   transformers: [...]
#   balance: "continuous_metadata_column"
#   bins: 10
#   binning_strategy: "quantile"
```

#### JSON Format

```json
{
  "sample_augmentation": {
    "transformers": [
      {"StandardScaler": {}},
      {"MinMaxScaler": {}}
    ],
    "balance": "y",
    "bins": 5,
    "binning_strategy": "quantile",
    "max_factor": 2.0,
    "random_state": 42
  }
}
```

### Python API

```python
# Direct dataset augmentation with binning
from nirs4all.utils.binning import BinningCalculator

# Step 1: Bin continuous targets
y = dataset.y({"partition": "train"}, include_augmented=False)
bin_indices, bin_edges = BinningCalculator.bin_continuous_targets(
    y,
    bins=10,
    strategy="quantile",
    random_state=42
)

# Step 2: Use binned labels for balanced counting
augmentation_counts = BalancingCalculator.calculate_balanced_counts(
    base_labels=bin_indices[:len(base_samples)],
    base_sample_indices=base_samples,
    all_labels=bin_indices,
    all_sample_indices=all_samples,
    max_factor=2.0
)
```

### Parameters

| Parameter | Type | Default | Scope | Description |
|-----------|------|---------|-------|-------------|
| `bins` | int | N/A | Balanced mode with `balance="y"` | Number of bins for continuous targets |
| `binning_strategy` | str | `"quantile"` | Balanced mode with `bins` | Strategy: `"quantile"` or `"equal_width"` |
| `balance` | str | - | - | Required; must be `"y"` or metadata column |
| `max_factor` | float | 1.0 | All balanced modes | Max augmentation multiplier |
| `selection` | str | `"random"` | All modes | Transformer assignment: `"random"` or `"all"` |
| `random_state` | int | None | All modes | Random seed for reproducibility |

### Behavior Specifications

#### Task Type Detection

```python
def _is_regression(y: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Detect if task is regression or classification.

    Heuristics:
    1. If y is integer and unique_values / len(y) < threshold → classification
    2. If y is float → regression (continuous by default)
    3. Count of unique values:
       - < 20: Could be either (check dtype)
       - >= 20: Assume regression

    Args:
        y: Target values
        threshold: Ratio of unique values to total samples

    Returns:
        True if regression (apply binning), False otherwise
    """
```

#### Binning Strategy Details

**Quantile Binning** (Default)
- Ensures equal probability per bin
- Recommended for non-uniform distributions
- Uses `np.quantile(y, [0/n, 1/n, 2/n, ..., 1])`
- Example: y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=3 → [0, 3, 6, 10]
- Result: Bins of approximately equal size [1-4), [4-7), [7-10]

**Equal Width Binning**
- Uniform width across full range
- Recommended for uniform distributions
- Uses `np.linspace(y.min(), y.max(), bins+1)`
- Example: y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=3 → [1, 4, 7, 10]
- Result: Equal width bins [1-4), [4-7), [7-10]

#### Bin Assignment

Bins are right-inclusive: `[edge_i, edge_{i+1}]`

```python
# Using np.digitize with right=True parameter
bin_indices = np.digitize(y, bin_edges, right=True)
# Returns: 0-based bin index for each sample
# Samples < bin_edges[0] → bin 0
# Samples >= bin_edges[-1] → bin (n_bins - 1)
```

#### Edge Cases

1. **y contains NaN**: Reject with error message
2. **Single bin (bins=1)**: All samples treated as same class (no augmentation)
3. **bins > unique_y values**: Warning but proceed (may create empty bins)
4. **Constant y**: All samples in one bin (degenerate case, no augmentation)
5. **Metadata column continuous**: Future extension (not v1)

---

## Implementation Plan

### Phase 1: Core Binning Calculator (1-2 days)

**Files to Create**:
- `nirs4all/utils/binning.py` - BinningCalculator class with binning strategies

**Implementation**:
1. Create `BinningCalculator` class
2. Implement `bin_continuous_targets()` main method
3. Implement `quantile_binning()` strategy
4. Implement `equal_width_binning()` strategy
5. Add validation methods
6. Add comprehensive docstrings and type hints

**Tests**:
- `tests/unit/test_binning.py`
  - Test quantile binning with various distributions
  - Test equal width binning
  - Test edge cases (NaN, constant y, single bin)
  - Test bin edge correctness
  - Test reproducibility with random_state

---

### Phase 2: SampleAugmentationController Integration (2-3 days)

**Files to Modify**:
- `nirs4all/controllers/dataset/op_sample_augmentation.py`

**Changes**:
1. Update `execute()` docstring to document bins/binning_strategy
2. Modify `_execute_balanced()` to:
   - Check for `bins` and `binning_strategy` parameters
   - Call `BinningCalculator.bin_continuous_targets()` if provided
   - Detect task type (regression vs classification)
   - Replace y labels with binned labels before calling `calculate_balanced_counts()`
   - Add logging for debugging

3. Add new method:
   ```python
   def _should_apply_binning(self, config: Dict) -> bool:
       """Check if binning should be applied based on config."""
       return "bins" in config or "binning_strategy" in config

   def _apply_binning_to_balanced(
       self,
       config: Dict,
       labels_all_train: np.ndarray,
       labels_base_train: np.ndarray,
       origin_indices: np.ndarray
   ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
       """Apply binning to continuous labels and return updated labels."""
       # Returns: (updated_labels_base, updated_labels_all, bin_edges)
   ```

**Tests**:
- `tests/unit/test_sample_augmentation_controller.py` (new tests)
  - Test balanced mode with bins parameter
  - Test both binning strategies
  - Test that binning produces valid virtual classes
  - Test interaction with max_factor
  - Test that binning respects partition boundaries

---

### Phase 3: BalancingCalculator Enhancements (1 day)

**Files to Modify**:
- `nirs4all/utils/balancing.py`

**Changes**:
1. No logic changes needed! (It already accepts any discrete labels)
2. Add docstring clarification that it works with binned labels
3. Consider adding optional utility method to report bin statistics
4. Add type hints to accept integer bin indices

**Tests**:
- Add tests in `test_balancing.py` to confirm it works with binned integer labels

---

### Phase 4: Integration & E2E Testing (2-3 days)

**Files to Create/Modify**:
- `tests/integration/test_binning_augmentation_e2e.py` (new)

**Test Scenarios**:
1. **End-to-end with regression dataset**:
   - Load continuous target regression dataset
   - Apply binned balanced augmentation
   - Verify bins are created correctly
   - Verify augmentation counts match binned classes

2. **Classification vs Regression detection**:
   - Discrete y values → no binning applied
   - Continuous y values → binning applied
   - Mixed int/float values → correct detection

3. **Leak prevention with binning**:
   - CV splits still use `include_augmented=False`
   - Validation folds don't see augmented samples
   - Binning doesn't affect leak prevention

4. **Binning with augmentation and cross-validation**:
   - Full pipeline: augment → split → model
   - Verify folds are created from base samples only
   - Verify training can access augmented samples

5. **Different binning strategies**:
   - Compare quantile vs equal_width on same dataset
   - Verify expected bin distributions

---

### Phase 5: Documentation (1-2 days)

**Files to Create/Update**:
- `docs/SAMPLE_AUGMENTATION.md` - Add binning section
- `docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md` - Add binning examples
- `docs/BINNING_AUGMENTATION_DESIGN.md` - This file

**New Example**:
- `examples/Q13_binned_balanced_augmentation.py` (inspired by Q12)

**Documentation Sections**:
1. When to use binning (regression contexts)
2. Binning strategies explanation with visualizations
3. Examples for each strategy
4. Troubleshooting binning-specific issues

---

### Phase 6: Validation & Polish (1 day)

**Tasks**:
1. Code review and refactoring
2. Performance testing with large y distributions
3. Edge case validation
4. Backward compatibility verification (no breaking changes)
5. Documentation review

---

## Timeline Summary

| Phase | Duration | Files | Status |
|-------|----------|-------|--------|
| 1: BinningCalculator | 1-2 days | binning.py, test_binning.py | Design ✓, Implement ⏳ |
| 2: Controller Integration | 2-3 days | op_sample_augmentation.py, test_*_controller.py | Design ✓, Implement ⏳ |
| 3: BalancingCalculator | 1 day | balancing.py, test_balancing.py | Design ✓, Implement ⏳ |
| 4: E2E Testing | 2-3 days | test_binning_augmentation_e2e.py | Design ✓, Implement ⏳ |
| 5: Documentation | 1-2 days | docs/*, examples/ | Design ✓, Implement ⏳ |
| 6: Validation | 1 day | All files | Design ✓, Implement ⏳ |
| **Total** | **~8-12 days** | ~10 files | Design ✓ |

---

## Testing Strategy

### Unit Tests

#### `tests/unit/test_binning.py`

```python
class TestBinContinuousTargets:
    def test_quantile_binning_10bins(self)
    def test_quantile_binning_3bins(self)
    def test_equal_width_binning_10bins(self)
    def test_equal_width_binning_edge_case_single_bin(self)
    def test_binning_with_nan_raises_error(self)
    def test_binning_with_constant_y(self)
    def test_binning_reproducible_with_random_state(self)
    def test_bin_edges_in_correct_order(self)
    def test_bin_indices_in_valid_range(self)
    def test_binning_with_negative_values(self)

class TestQuantileBinning:
    def test_quantile_bins_have_equal_size(self)
    def test_quantile_respects_data_distribution(self)

class TestEqualWidthBinning:
    def test_equal_width_bins_have_equal_spacing(self)
    def test_equal_width_respects_min_max(self)

class TestValidateBinsParam:
    def test_valid_int_bins(self)
    def test_invalid_bins_negative(self)
    def test_invalid_bins_zero(self)
    def test_invalid_bins_float(self)
    def test_invalid_bins_string(self)
```

#### Enhanced `tests/unit/test_sample_augmentation_controller.py`

```python
class TestBalancedModeWithBinning:
    def test_balanced_with_bins_quantile(self, regression_dataset, mock_runner)
    def test_balanced_with_bins_equal_width(self, regression_dataset, mock_runner)
    def test_balanced_with_bins_and_max_factor(self, regression_dataset, mock_runner)
    def test_binning_skipped_for_classification(self, imbalanced_dataset, mock_runner)
    def test_binning_with_custom_random_state(self, regression_dataset, mock_runner)
    def test_invalid_bins_parameter_raises_error(self, regression_dataset, mock_runner)
    def test_binning_strategy_invalid_raises_error(self, regression_dataset, mock_runner)
```

### Integration Tests

#### `tests/integration/test_binning_augmentation_e2e.py`

```python
class TestBinningAugmentationE2E:
    def test_full_pipeline_with_binning_and_cv(self)
    def test_leak_prevention_with_binning(self)
    def test_regression_vs_classification_detection(self)
    def test_binned_augmentation_produces_correct_distribution(self)
    def test_comparison_quantile_vs_equal_width(self)
```

### Test Data

**Fixtures needed**:
- `regression_dataset`: Continuous y values, no class structure
- `uniform_continuous_y`: y ~ Uniform[0, 100]
- `normal_continuous_y`: y ~ Normal(50, 20)
- `skewed_continuous_y`: y with skewed distribution
- `imbalanced_classification_y`: Discrete classes (to verify no binning applied)

---

## Documentation Updates

### Primary Documentation

#### `docs/SAMPLE_AUGMENTATION.md` - New Section

```markdown
### Binning-Based Balanced Mode (Regression)

**When to use**: Regression tasks with continuous targets where you want
balanced augmentation across the full range of y values.

#### Overview

In regression, binning divides the continuous y-axis into discrete bins (virtual classes),
then applies balanced augmentation to ensure each bin has sufficient augmented samples.

**Example scenario**:
- Raw y: [5.2, 5.8, 25.1, 25.5, 75.2, 75.8, 95.1, 95.3]
- With 4 quantile bins: [0-5), [5-25), [25-75), [75-100]
- Virtual classes: [bin_0:2, bin_1:2, bin_2:2, bin_3:2] (already balanced!)

#### Binning Strategies

**Quantile Binning** (Recommended)
- Creates bins with equal probability
- Each bin has ~n_samples/bins items
- Better for skewed distributions
- Example: y ∈ [0, 100] with 10 quantile bins → each bin has ~10% of samples

**Equal Width Binning**
- Creates bins with equal width
- Width = (max - min) / bins
- Better for uniform distributions
- Example: y ∈ [0, 100] with 10 equal bins → each bin spans 10 units

#### Usage Examples

[Examples with YAML and results]

#### Best Practices

- Start with 10 bins (good default)
- Use quantile for real-world data (usually skewed)
- Monitor augmentation distribution across bins
- Verify bins capture meaningful subranges of y
```

#### `docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md` - Add Binning Section

```markdown
## Binning Mode (Regression)

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10              # NEW
  binning_strategy: "quantile"  # NEW
  max_factor: 2.0
  random_state: 42
```

| Parameter | Default | Options |
|-----------|---------|---------|
| `bins` | N/A | 1-100 (recommended: 5-20) |
| `binning_strategy` | `"quantile"` | `"quantile"`, `"equal_width"` |
```

### Example Script

#### `examples/Q13_binned_balanced_augmentation.py`

```python
"""
Q13 Binned Balanced Augmentation Example
========================================
Demonstrates sample augmentation with binning for regression tasks.
Shows how to balance augmentation across continuous y ranges.
"""

# Regression data with continuous targets
# Load dataset
# Apply binned balanced augmentation with different strategies
# Visualize results (histogram of augmentations per bin)
# Train model and compare performance
```

---

## Backward Compatibility

### No Breaking Changes

✅ **Fully backward compatible**:
1. New parameters are **optional**
   - `bins`: When omitted, existing balanced logic applies
   - `binning_strategy`: Only used if `bins` is provided

2. Existing configurations continue to work:
   ```yaml
   # Still works (no binning)
   sample_augmentation:
     transformers: [...]
     balance: "y"
     max_factor: 1.0
   ```

3. Classification tasks unaffected:
   - Binning only applies when continuous y detected
   - Integer y values treated as discrete classes (existing behavior)

4. Metadata binning not in v1:
   - Only `balance: "y"` with continuous values triggers binning
   - Metadata column balancing unchanged

### Migration Path

**For users with existing regression augmentation**:
1. No action required - existing configs work as-is
2. To use binning, add `bins: 10` parameter
3. Test on small subset before production use

---

## Implementation Notes

### Key Design Decisions

1. **Binning only for y, not metadata (v1)**
   - Rationale: Metadata columns may be discrete by design
   - Future: Can extend for continuous metadata columns

2. **No binning parameters in dataset API**
   - Rationale: Binning is augmentation strategy, not data API concern
   - All binning logic isolated in controllers and calculator

3. **Automatic regression detection**
   - Rationale: User convenience (YAML config should be self-contained)
   - Falls back gracefully if detection uncertain

4. **Quantile as default**
   - Rationale: Better for most real-world distributions
   - Equal width still available for specific use cases

### Potential Extensions (Post-v1)

1. **Custom binning function**: Allow user-defined bin edges
2. **Continuous metadata**: Support binning on metadata columns
3. **Adaptive binning**: Learn optimal bin count from data
4. **Visualization tools**: Plot bin distributions before/after augmentation
5. **Bin-aware sampling**: Select augmented samples randomly from within bin

---

## Risk Assessment

### Low Risk Areas ✅
- BinningCalculator is pure utility (no side effects)
- No changes to core dataset API
- Fully optional feature (no impact if unused)

### Medium Risk Areas ⚠️
- Task type detection heuristic (could misclassify edge cases)
- Integration with existing balanced logic (but well-tested path)

### Mitigation Strategies
- Comprehensive unit tests for edge cases
- Clear logging of task type detection
- Validation tests on real datasets
- User warnings for ambiguous cases

---

## Success Criteria

### Functional ✅
- [ ] Quantile binning produces equal-size bins
- [ ] Equal-width binning produces uniform-width bins
- [ ] Augmentation counts calculated correctly per bin
- [ ] No data leakage in CV splits with binning
- [ ] Classification tasks unaffected

### Quality ✅
- [ ] >90% test coverage for binning code
- [ ] All edge cases handled gracefully
- [ ] Backward compatibility verified
- [ ] Performance acceptable (<5% overhead vs non-binning)

### Documentation ✅
- [ ] Clear API documentation
- [ ] Working examples for both strategies
- [ ] Troubleshooting guide
- [ ] Best practices documented

---

## Appendix A: Example Binning Outputs

### Quantile Binning Example

```
y (raw): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
bins: 3

Quantiles: [0.0, 0.33, 0.67, 1.0]
Bin edges: [1.0, 5.67, 10.33, 15.0]

Bin assignments:
- [1.0, 5.67): samples [1, 2, 3, 4, 5] → bin 0 (5 samples)
- [5.67, 10.33): samples [6, 7, 8, 9, 10] → bin 1 (5 samples)
- [10.33, 15.0]: samples [11, 12, 13, 14, 15] → bin 2 (5 samples)

Result: Each bin has 5 samples (balanced by design)
```

### Equal Width Binning Example

```
y (raw): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
bins: 3

Range: [1, 15], Width: (15-1)/3 = 4.67
Bin edges: [1.0, 5.67, 10.33, 15.0]

Bin assignments:
- [1.0, 5.67): samples [1, 2, 3, 4, 5] → bin 0 (5 samples)
- [5.67, 10.33): samples [6, 7, 8, 9, 10] → bin 1 (5 samples)
- [10.33, 15.0]: samples [11, 12, 13, 14, 15] → bin 2 (5 samples)

Result: Uniform width, approximately equal samples per bin
```

### Non-uniform Distribution Example

```
y (raw): [1, 1, 1, 2, 2, 3, 10, 10, 10, 10, 20, 20, 20, 20, 20]
bins: 3

QUANTILE BINNING:
Quantiles: [0%, 33%, 67%, 100%]
Bin edges: [1.0, 2.0, 10.0, 20.0]
Result: ~5 samples per bin (equal probability)

EQUAL WIDTH BINNING:
Range: [1, 20], Width: 6.33
Bin edges: [1.0, 7.33, 13.67, 20.0]
Result: Unequal samples
- Bin 0: [1, 1, 1, 2, 2, 3] (6 samples)
- Bin 1: [10, 10, 10, 10] (4 samples)
- Bin 2: [20, 20, 20, 20, 20] (5 samples)
```

---

## References

- **NumPy Quantile**: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
- **NumPy Digitize**: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
- **Existing Design**: Sample Augmentation Implementation Summary
- **Related Docs**: SAMPLE_AUGMENTATION.md, SAMPLE_AUGMENTATION_QUICK_REFERENCE.md
