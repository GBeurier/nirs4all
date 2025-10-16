# Binning-Based Balanced Augmentation - Implementation Roadmap

## Quick Summary

This roadmap outlines the implementation of **binning support for balanced augmentation** in regression contexts. When `balance: "y"` is specified for regression data, users can now configure binning to treat continuous target values as virtual "bins" (classes) for augmentation balancing.

---

## Phase Breakdown

### Phase 1: Core Binning Calculator ⏰ 1-2 days

**Goal**: Implement reusable binning utilities

**File**: `nirs4all/utils/binning.py` (New)

**What to implement**:
```python
class BinningCalculator:
    @staticmethod
    def bin_continuous_targets(
        y: np.ndarray,
        bins: int = 10,
        strategy: str = "quantile",
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (bin_indices, bin_edges)
        - bin_indices: 0-based bin assignment per sample
        - bin_edges: Boundaries of each bin
        """

    @staticmethod
    def quantile_binning(y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Equal probability bins - uses np.quantile"""

    @staticmethod
    def equal_width_binning(y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Equal width bins - uses np.linspace"""

    @staticmethod
    def validate_bins_param(bins: Any) -> int:
        """Type checking and validation"""
```

**Key Methods**:
- Quantile binning: Equal probability per bin
- Equal width binning: Uniform width per bin
- Validation: Check bins parameter (int, 1-1000)
- Edge cases: NaN handling, constant y, single bin

**Test File**: `tests/unit/test_binning.py`
- ~15-20 tests covering all strategies and edge cases

---

### Phase 2: SampleAugmentationController Integration ⏰ 2-3 days

**Goal**: Add binning logic to balanced augmentation workflow

**File**: `nirs4all/controllers/dataset/op_sample_augmentation.py` (Modify)

**What to modify**:
1. Update `_execute_balanced()` to detect and apply binning:
   ```python
   def _execute_balanced(self, config: Dict, ...):
       # ... existing code ...

       # NEW: Check for binning parameters
       if "bins" in config or "binning_strategy" in config:
           labels_all_train = self._apply_binning(
               labels_all_train,
               config.get("bins", 10),
               config.get("binning_strategy", "quantile"),
               config.get("random_state")
           )
           labels_base_train = labels_all_train[:len(base_train_samples)]

       # ... rest of existing logic (works with binned labels) ...
   ```

2. Add helper method:
   ```python
   def _apply_binning(self, y, bins, strategy, random_state):
       """Apply binning and return bin indices"""
       bin_indices, _ = BinningCalculator.bin_continuous_targets(
           y, bins, strategy, random_state
       )
       return bin_indices
   ```

3. Update docstring to document new parameters

**Test File**: Enhanced `tests/unit/test_sample_augmentation_controller.py`
- Test balanced mode with `bins` parameter
- Test both strategies (quantile, equal_width)
- Test binning + max_factor interaction
- Verify classification tasks unaffected

---

### Phase 3: BalancingCalculator Clarification ⏰ 1 day

**Goal**: Ensure compatibility with binned labels

**File**: `nirs4all/utils/balancing.py` (Minor updates only)

**What to do**:
1. Add clarifying docstring:
   ```python
   def calculate_balanced_counts(...):
       """
       Works with any discrete labels (classes or bins).
       Can accept:
       - Classification labels: [0, 1, 2, ...]
       - String classes: ['cat', 'dog', ...]
       - Binned regression labels: [0, 1, 2, ...] from BinningCalculator
       """
   ```

2. No logic changes needed (already generic!)

**Test File**: `tests/unit/test_balancing.py`
- Add test confirming it works with bin indices

---

### Phase 4: End-to-End Integration Testing ⏰ 2-3 days

**Goal**: Verify complete workflow from pipeline to augmented dataset

**File**: `tests/integration/test_binning_augmentation_e2e.py` (New)

**Test scenarios**:
1. **Basic binning workflow**:
   - Load regression dataset
   - Apply binned balanced augmentation
   - Verify correct number of augmented samples per bin

2. **Classification vs Regression detection**:
   - Discrete y → no binning applied
   - Continuous y → binning applied
   - Correct detection in both cases

3. **Leak prevention maintained**:
   - CV splits use only base samples
   - Validation never sees augmented samples
   - Binning doesn't break leak prevention

4. **Full pipeline integration**:
   - Pipeline: augment → split → model
   - Works without errors
   - Produces reasonable results

5. **Strategy comparison**:
   - Quantile vs equal_width on same data
   - Different distributions produce different bin assignments
   - Both valid and useful in different scenarios

---

### Phase 5: Documentation & Examples ⏰ 1-2 days

**Files to create/update**:

1. **`docs/SAMPLE_AUGMENTATION.md`** (Update)
   - Add new section: "Binning-Based Balanced Mode (Regression)"
   - Explain use cases and strategies
   - Show examples with YAML configs
   - Compare quantile vs equal width
   - Add best practices

2. **`docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md`** (Update)
   - Add binning parameters to table
   - Add quick syntax example
   - Add to "common patterns" section

3. **`examples/Q13_binned_balanced_augmentation.py`** (New)
   - Inspired by Q12_sample_augmentation.py
   - Demonstrate binning with real regression data
   - Show visualization of bin distributions
   - Compare different strategies
   - Show augmentation impact

4. **`docs/BINNING_AUGMENTATION_DESIGN.md`** (Already created)
   - Comprehensive design document
   - Reference for developers
   - Detailed architecture and rationale

---

### Phase 6: Validation & Polish ⏰ 1 day

**Tasks**:
- Code review and optimization
- Performance testing (large y ranges)
- Edge case verification
- Backward compatibility testing
- Documentation review and polish

---

## API Summary

### Pipeline Configuration

```yaml
# Balanced augmentation with binning (regression)
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10                    # Number of bins for continuous targets
  binning_strategy: "quantile"  # or "equal_width"
  max_factor: 2.0             # Optional: max augmentation factor
  random_state: 42            # Optional: reproducibility
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `balance` | str | - | Required: `"y"` for target-based balancing |
| `bins` | int | N/A | NEW: Number of bins (1-1000, recommended 5-20) |
| `binning_strategy` | str | `"quantile"` | NEW: `"quantile"` or `"equal_width"` |
| `max_factor` | float | 1.0 | Max augmentation multiplier |
| `selection` | str | `"random"` | Transformer assignment method |
| `random_state` | int | None | Reproducibility seed |

### When Binning is Applied

✅ **Binning enabled if**:
- `balance: "y"` is specified AND
- `bins` parameter provided OR `binning_strategy` provided

❌ **Binning disabled if**:
- `balance: "y"` without `bins` parameter (uses existing discrete logic)
- `balance: "metadata_column"` (only y supports binning in v1)
- Classification task detected (even with bins parameter)

---

## File Changes Summary

### New Files (3)
- `nirs4all/utils/binning.py` - BinningCalculator class
- `tests/unit/test_binning.py` - Binning tests
- `tests/integration/test_binning_augmentation_e2e.py` - E2E tests
- `examples/Q13_binned_balanced_augmentation.py` - Example script

### Modified Files (4)
- `nirs4all/controllers/dataset/op_sample_augmentation.py` - Integration (+40-50 lines)
- `nirs4all/utils/balancing.py` - Docstring clarification (+5-10 lines)
- `docs/SAMPLE_AUGMENTATION.md` - Add binning section (+100-150 lines)
- `docs/SAMPLE_AUGMENTATION_QUICK_REFERENCE.md` - Add binning reference (+30-50 lines)

### Total New Lines
- ~400-500 lines of implementation code
- ~300-400 lines of test code
- ~150-200 lines of documentation

---

## Implementation Strategy

### Key Design Principles

1. **Backward Compatible**: Binning is optional; existing configs work unchanged
2. **Automatic Detection**: System detects regression vs classification
3. **Reusable Utilities**: BinningCalculator usable outside augmentation context
4. **Minimal Changes**: Controller only adds ~50 lines for binning support
5. **Well Tested**: Comprehensive unit and integration tests

### Data Flow with Binning

```
y values (continuous)
    ↓
BinningCalculator.bin_continuous_targets()
    ↓
bin_indices = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    ↓
Use bin_indices instead of y for:
BalancingCalculator.calculate_balanced_counts()
    ↓
augmentation_counts = {sample_id → count, ...}
    ↓
Proceed with normal augmentation workflow
```

### Binning Strategies Explained

**Quantile (Default)**
- Creates bins with equal probability
- `np.quantile(y, [0/n, 1/n, 2/n, ..., n/n])`
- Best for: Skewed or unknown distributions
- Example: y ∈ [0, 100] with 10 bins → ~10% per bin

**Equal Width**
- Creates bins with uniform width
- `np.linspace(y.min(), y.max(), bins+1)`
- Best for: Uniform distributions
- Example: y ∈ [0, 100] with 10 bins → each 10 units wide

---

## Testing Approach

### Unit Test Structure

```
test_binning.py (20 tests)
├── TestBinContinuousTargets (8 tests)
│   ├── test_quantile_binning_basic
│   ├── test_equal_width_binning_basic
│   ├── test_edge_case_single_bin
│   ├── test_edge_case_nan_values
│   ├── test_edge_case_constant_y
│   ├── test_reproducibility
│   ├── test_bin_indices_valid_range
│   └── test_bin_edges_correct_order
├── TestQuantileBinning (3 tests)
├── TestEqualWidthBinning (3 tests)
└── TestValidateBinsParam (6 tests)

test_sample_augmentation_controller.py (Enhanced)
├── TestBalancedModeWithBinning (7 tests)
│   ├── test_balanced_with_bins_quantile
│   ├── test_balanced_with_bins_equal_width
│   ├── test_binning_with_max_factor
│   ├── test_classification_unaffected
│   ├── test_custom_random_state
│   ├── test_invalid_bins_parameter
│   └── test_invalid_strategy_parameter

test_binning_augmentation_e2e.py (10+ integration tests)
├── test_full_pipeline_with_binning_and_cv
├── test_leak_prevention_with_binning
├── test_regression_vs_classification_detection
├── test_bin_distributions_correct
├── test_comparison_quantile_vs_equal_width
└── ... (scenario-specific tests)
```

---

## Success Criteria

### Functionality ✅
- [ ] Quantile binning produces correct equal-probability bins
- [ ] Equal-width binning produces correct uniform-width bins
- [ ] Augmentation counts calculated correctly per virtual bin
- [ ] Classification tasks work without binning
- [ ] Regression tasks properly detected and binned
- [ ] Leak prevention maintained

### Quality ✅
- [ ] >90% test coverage for new code
- [ ] All edge cases handled gracefully
- [ ] Performance impact <5% vs non-binning
- [ ] Clear error messages for invalid parameters

### Documentation ✅
- [ ] API fully documented with examples
- [ ] Binning strategies explained clearly
- [ ] Working example script included
- [ ] Best practices guide provided

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| 1. BinningCalculator | 1-2 days | ⏳ Ready to implement |
| 2. Controller Integration | 2-3 days | ⏳ Ready to implement |
| 3. BalancingCalculator | 1 day | ⏳ Ready to implement |
| 4. E2E Testing | 2-3 days | ⏳ Ready to implement |
| 5. Documentation | 1-2 days | ⏳ Ready to implement |
| 6. Validation | 1 day | ⏳ Ready to implement |
| **TOTAL** | **~8-12 days** | ⏳ **Estimated** |

---

## Next Steps

1. **Review this roadmap** and design document
2. **Approve architecture and API** before implementation
3. **Start Phase 1** with BinningCalculator implementation
4. **Proceed sequentially** through phases (some parallelization possible)
5. **Code review** after each phase completion
6. **Merge to main** after all phases complete and tests pass

---

## Appendix: Usage Examples

### Example 1: Basic Regression Augmentation with Binning

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      balance: "y"
      bins: 10              # 10 virtual classes from continuous y
      binning_strategy: "quantile"
      random_state: 42
  - split:
      - StratifiedKFold: {n_splits: 5}
  - model:
      - LinearRegression: {}
```

### Example 2: Comparing Strategies

```python
# Quantile binning (data-driven, recommended)
config_quantile = {
    "sample_augmentation": {
        "transformers": [...],
        "balance": "y",
        "bins": 5,
        "binning_strategy": "quantile"  # Equal probability bins
    }
}

# Equal width binning (range-driven)
config_equal = {
    "sample_augmentation": {
        "transformers": [...],
        "balance": "y",
        "bins": 5,
        "binning_strategy": "equal_width"  # Uniform spacing
    }
}
```

### Example 3: With Additional Parameters

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  balance: "y"
  bins: 15                    # More bins for finer control
  binning_strategy: "equal_width"  # For uniform y distribution
  max_factor: 3.0             # Allow up to 3x augmentation
  selection: "random"         # Random transformer assignment
  random_state: 42            # Reproducibility
```

---

## Document Location

This roadmap document can be found at:
- Main design: `docs/BINNING_AUGMENTATION_DESIGN.md`
- This summary: `docs/BINNING_AUGMENTATION_ROADMAP.md`
