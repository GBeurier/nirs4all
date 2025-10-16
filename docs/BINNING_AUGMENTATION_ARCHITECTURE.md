# Binning-Based Balanced Augmentation - Visual Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Pipeline Configuration                      │
│                                                                          │
│  sample_augmentation:                                                   │
│    balance: "y"                                                          │
│    bins: 10                    ← NEW PARAMETER                           │
│    binning_strategy: "quantile"  ← NEW PARAMETER                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  SampleAugmentationController        │
        │  _execute_balanced()                 │
        └────────────┬─────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    [bins parameter?]    [existing balanced logic]
         YES                    NO (backward compat)
         │                      │
         ▼                      ▼
    ┌──────────────────────┐   ┌──────────────────┐
    │  BinningCalculator   │   │ BalancingCalculator
    │  .bin_continuous_    │   │ (unchanged)
    │   targets()          │   └──────────────────┘
    │                      │
    │ Strategy:            │
    │ • quantile_binning   │
    │ • equal_width_       │
    │   binning            │
    └────────┬─────────────┘
             │
             ▼
    Bin Indices: [0, 0, 1, 1, 2, 2, 3, 3, 3]
             │
             ▼
    ┌──────────────────────────────────────┐
    │ BalancingCalculator                  │
    │ .calculate_balanced_counts()         │
    │ (uses binned labels like classes)    │
    └────────┬─────────────────────────────┘
             │
             ▼
    Augmentation Counts: {0→2, 1→1, 2→0, ...}
             │
             ▼
    ┌──────────────────────────────────────┐
    │ Emit transformer steps                │
    │ (delegation to TransformerMixin)      │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ TransformerMixinController           │
    │ (applies transformations)            │
    └────────┬─────────────────────────────┘
             │
             ▼
    Dataset augmented with origin tracking
             │
             ▼
    ┌──────────────────────────────────────┐
    │ CrossValidatorController             │
    │ Uses include_augmented=False         │
    │ (leak prevention maintained!)        │
    └──────────────────────────────────────┘
```

## Data Transformation Flow

```
STEP 1: Load continuous regression data
┌─────────────────────────────────────┐
│ y (continuous values)               │
│ [5.2, 5.8, 25.1, 25.5, 75.2, 75.8] │
│ (6 samples, real range [0, 100])    │
└─────────────────────────────────────┘

STEP 2: Apply Binning Strategy
┌────────────────────────────────────────────┐
│ QUANTILE BINNING (bins=3)                  │
│                                            │
│ Quantiles: [0%, 33%, 67%, 100%]           │
│ Bin edges: [5.2, 25.7, 75.5]             │
│                                            │
│ Assignments:                               │
│ • y=5.2  → bin 0 ✓                       │
│ • y=5.8  → bin 0 ✓                       │
│ • y=25.1 → bin 1 ✓                       │
│ • y=25.5 → bin 1 ✓                       │
│ • y=75.2 → bin 2 ✓                       │
│ • y=75.8 → bin 2 ✓                       │
│                                            │
│ Result: [0, 0, 1, 1, 2, 2]               │
│         ^bin0  ^bin1  ^bin2               │
└────────────────────────────────────────────┘

STEP 3: Calculate Balanced Augmentation
┌──────────────────────────────────────────┐
│ Bin Distribution (Current):               │
│ • bin 0: 2 samples (33%)                 │
│ • bin 1: 2 samples (33%)                 │
│ • bin 2: 2 samples (33%)                 │
│                                          │
│ Ideal Distribution (max_factor=2.0):     │
│ • All bins: target = 2 samples            │
│                                          │
│ Augmentation needed:                      │
│ • bin 0: 0 (already 2)                   │
│ • bin 1: 0 (already 2)                   │
│ • bin 2: 0 (already 2)                   │
│                                          │
│ Total augmented: 0 (already balanced!)    │
└──────────────────────────────────────────┘

STEP 4: Execute Augmentation
┌──────────────────────────────────────────┐
│ For each sample in augmentation pool:     │
│                                          │
│ • Apply transformer(s)                    │
│ • Store augmented sample with origin id   │
│ • Track sample bin membership             │
│                                          │
│ Result: Base + Augmented in dataset       │
└──────────────────────────────────────────┘

STEP 5: CV Splitting with Leak Prevention
┌──────────────────────────────────────────┐
│ Split uses: include_augmented=False       │
│ (only 6 base samples for splitting)       │
│                                          │
│ Fold 1:                                  │
│ • Train: 5 base + N augmented ✓          │
│ • Test: 1 base (no augmented) ✓          │
│                                          │
│ Fold 2:                                  │
│ • Train: 5 base + N augmented ✓          │
│ • Test: 1 base (no augmented) ✓          │
│                                          │
│ (No data leakage!)                       │
└──────────────────────────────────────────┘
```

## Binning Strategies Comparison

```
Dataset: y = [10, 15, 20, 25, 30, 35, 40, 45, 50, 95]
         (10 samples, range [10, 95])

SCENARIO: bins=3

┌────────────────────────────────────────────────────┐
│ QUANTILE BINNING (Equal Probability)              │
├────────────────────────────────────────────────────┤
│ Percentiles: 0%, 33%, 67%, 100%                   │
│ Computation: np.quantile(y, [0, 0.33, 0.67, 1.0])│
│ Edges: [10, 24.67, 42.33, 95]                    │
│                                                   │
│ Bin 0: [10, 24.67) → [10, 15, 20] (3 samples)   │
│ Bin 1: [24.67, 42.33) → [25, 30, 35, 40] (4)   │
│ Bin 2: [42.33, 95] → [45, 50, 95] (3 samples)   │
│                                                   │
│ ✓ Each bin ~3.3 samples (equal probability)      │
│ ✓ Better for skewed distributions                │
│ ✓ Adapts to actual data distribution             │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ EQUAL WIDTH BINNING (Uniform Spacing)             │
├────────────────────────────────────────────────────┤
│ Range: 95 - 10 = 85                              │
│ Width: 85 / 3 = 28.33 per bin                    │
│ Computation: np.linspace(10, 95, 4)              │
│ Edges: [10, 38.33, 66.67, 95]                   │
│                                                   │
│ Bin 0: [10, 38.33) → [10, 15, 20, 25, 30, 35] (6) │
│ Bin 1: [38.33, 66.67) → [40, 45, 50] (3 samples) │
│ Bin 2: [66.67, 95] → [95] (1 sample)            │
│                                                   │
│ ✓ Uniform spacing across y range                 │
│ ✓ Better for uniform distributions               │
│ ✓ Bins may have different sample counts          │
└────────────────────────────────────────────────────┘

WHICH TO USE?
├─ Quantile: Default, use for unknown distributions ✓
└─ Equal Width: Use for known uniform distributions
```

## Implementation Phases Timeline

```
DAY 1-2: Phase 1 - BinningCalculator
┌──────────────────────────────────────┐
│ ✓ Create BinningCalculator class     │
│ ✓ Implement quantile_binning()       │
│ ✓ Implement equal_width_binning()    │
│ ✓ Add validation logic               │
│ ✓ Write ~20 unit tests               │
└──────────────────────────────────────┘

DAY 3-5: Phase 2 - Controller Integration
┌──────────────────────────────────────┐
│ ✓ Update SampleAugmentationController│
│ ✓ Add binning detection logic        │
│ ✓ Integrate with balanced mode       │
│ ✓ Add controller-level tests         │
└──────────────────────────────────────┘

DAY 6: Phase 3 - BalancingCalculator
┌──────────────────────────────────────┐
│ ✓ Verify compatibility (no changes)  │
│ ✓ Update docstrings                  │
│ ✓ Add clarification tests            │
└──────────────────────────────────────┘

DAY 7-9: Phase 4 - Integration Testing
┌──────────────────────────────────────┐
│ ✓ Create E2E test file               │
│ ✓ Test full pipeline with binning    │
│ ✓ Verify leak prevention             │
│ ✓ Test regression/classification     │
└──────────────────────────────────────┘

DAY 10-11: Phase 5 - Documentation
┌──────────────────────────────────────┐
│ ✓ Update SAMPLE_AUGMENTATION.md      │
│ ✓ Update Quick Reference             │
│ ✓ Create example script (Q13)         │
│ ✓ Add API documentation              │
└──────────────────────────────────────┘

DAY 12: Phase 6 - Validation & Polish
┌──────────────────────────────────────┐
│ ✓ Code review                        │
│ ✓ Performance testing                │
│ ✓ Edge case validation               │
│ ✓ Final documentation review         │
│ ✓ Ready for merge!                   │
└──────────────────────────────────────┘

TOTAL: ~8-12 days (sequential, some phases parallelizable)
```

## API Parameters Reference

```
SampleAugmentationController Configuration
═════════════════════════════════════════════

EXISTING PARAMETERS (unchanged):
├─ transformers: [list]     Required, transformer instances
├─ balance: str             "y" or metadata column name
├─ max_factor: float        (default 1.0)
├─ selection: str           "random" or "all"
└─ random_state: int        Random seed

NEW PARAMETERS (optional, for regression):
├─ bins: int                Number of virtual classes
│                          Default: None (no binning)
│                          Range: 1-1000 (recommended 5-20)
│
└─ binning_strategy: str    How to create bins
                           Default: "quantile" (if bins provided)
                           Options:
                           • "quantile" - equal probability
                           • "equal_width" - uniform spacing

ACTIVATION:
- Binning applied if: balance="y" AND (bins specified OR binning_strategy specified)
- No binning if: balance specifies metadata column
- Auto-detection: Continuous y → apply binning if requested
                 Discrete y → use existing logic (bins ignored)
```

## Decision Tree: When to Use Binning

```
                         balance: "y"?
                            │
                    ┌───────┴───────┐
                   NO              YES
                    │               │
                    ▼               ▼
              (Don't balance)   bins parameter?
                               │
                       ┌───────┴───────┐
                      NO              YES
                       │               │
                       ▼               ▼
            (Existing discrete    Apply Binning
             class logic)         │
                                 ▼
                            Detect task type
                            │
                    ┌───────┴────────┐
              Classification     Regression
                    │                 │
                    ▼                 ▼
            (Ignore bins,        Apply binning strategy
             use discrete        │
             logic)          ┌───┴────┐
                          Quantile  Equal-Width
                             │           │
                             ▼           ▼
                        Create equal  Create uniform
                        probability   width bins
                        bins
```

## File Structure

```
nirs4all/
├── utils/
│   ├── balancing.py (existing, unchanged logic)
│   └── binning.py ← NEW FILE
│       └── BinningCalculator
│
├── controllers/
│   └── dataset/
│       └── op_sample_augmentation.py (modified)
│           └── _apply_binning() ← NEW METHOD
│
└── docs/
    ├── SAMPLE_AUGMENTATION.md (updated)
    ├── SAMPLE_AUGMENTATION_QUICK_REFERENCE.md (updated)
    ├── BINNING_AUGMENTATION_DESIGN.md ← NEW FILE
    └── BINNING_AUGMENTATION_ROADMAP.md ← NEW FILE

examples/
└── Q13_binned_balanced_augmentation.py ← NEW FILE

tests/
├── unit/
│   ├── test_binning.py ← NEW FILE
│   └── test_sample_augmentation_controller.py (enhanced)
│
└── integration/
    └── test_binning_augmentation_e2e.py ← NEW FILE
```

## Key Benefits of Design

✅ **Minimal Changes**
- Only ~50 lines added to controller
- BinningCalculator is independent utility
- Full backward compatibility

✅ **Automatic Handling**
- Task type detected automatically
- No manual class definition needed
- Sensible defaults (quantile, 10 bins)

✅ **Reusable Components**
- BinningCalculator can be used standalone
- Works with any regression binning task
- Easy to extend with new strategies

✅ **Leak Prevention Maintained**
- No changes to split logic
- Binning happens before splitting
- CV remains leak-free

✅ **Well-Tested**
- Unit tests for binning logic
- Controller integration tests
- E2E pipeline tests
- Edge case coverage

✅ **User-Friendly**
- Simple YAML configuration
- Sensible defaults
- Clear documentation
- Working examples
```

Perfect! Now let me create one final summary document:
