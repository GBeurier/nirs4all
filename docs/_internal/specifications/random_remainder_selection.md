# Random Remainder Selection Feature

## Overview
Implemented random remainder distribution in the `BalancingCalculator.calculate_balanced_counts()` method. When augmentations cannot be evenly distributed across samples, the remainder is now randomly assigned instead of always going to the first samples.

## Changes Made

### Core Change: `nirs4all/utils/balancing.py`
- Added `random_state` parameter to `calculate_balanced_counts()` method
- Changed remainder distribution from deterministic to probabilistic:
  - **Before**: First N samples got remainder augmentations
  - **After**: Random selection of which samples get remainder augmentations

### Algorithm
```python
# For each class needing augmentation:
total_needed = target - current_count
aug_per_base = total_needed // base_count      # Base augmentations for all
remainder = total_needed % base_count           # Remainder to distribute

# Initialize all samples with base amount
for sample_id in base_samples:
    augmentation_map[sample_id] = aug_per_base

# Randomly select which samples get the remainder
if remainder > 0:
    remainder_indices = rng.choice(base_count, size=remainder, replace=False)
    for idx in remainder_indices:
        augmentation_map[base_samples[idx]] += 1
```

### Example
With 5 samples needing 12 total augmentations (17 target - 5 current):
- **Base**: 12 ÷ 5 = 2 augmentations per sample
- **Remainder**: 12 % 5 = 2 samples need one more
- **Result**: 2 samples get 3, 3 samples get 2 (random selection)

```
Seed 42:  [3, 2, 2, 3, 2]  ← samples 0 and 3 selected
Seed 99:  [2, 2, 3, 3, 2]  ← samples 2 and 3 selected
```

## Benefits
1. **Fair Selection**: All samples have equal chance of getting remainder augmentations
2. **Prevents Bias**: No longer biased toward first samples
3. **Reproducible**: Use `random_state` for deterministic results when needed
4. **Statistical**: Better distribution when running experiments

## Tests Added
- `test_random_remainder_distribution()`: Verifies remainder is random
- `test_random_remainder_different_seeds()`: Confirms different seeds produce different results
- All 58 tests passing (25 balancing + 11 binning + 22 controller tests)

## Usage
```python
from nirs4all.controllers.data.balancing import BalancingCalculator
import numpy as np

counts = BalancingCalculator.calculate_balanced_counts(
    labels, indices, labels, indices,
    target_size=100,
    random_state=42  # Optional: for reproducibility
)
```

## Files Modified
- `nirs4all/controllers/data/balancing.py` - Added `random_state` parameter and random selection
- `tests/unit/test_balancing.py` - Updated tests, added 2 new tests
- `examples/Q14_random_remainder_selection.py` - New example demonstrating the feature
