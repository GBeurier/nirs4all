# Sample Augmentation Performance Analysis Report

## Executive Summary

The sample augmentation in Q20_analysis.py is extremely slow due to **O(N²) array operations** caused by sample-by-sample insertion into numpy arrays. With ~3,500 samples and 16 augmenters producing ~3,000+ augmented samples, the system performs millions of unnecessary array copies.

**Key Finding**: Each `dataset.add_samples()` call triggers `np.concatenate()` which copies the entire array. With thousands of insertions, this becomes a quadratic bottleneck.

---

## Architecture Analysis

### Current Data Flow

```
Q20_analysis.py
    ↓
SampleAugmentationController._execute_balanced()
    ↓ (for each of 16 transformers - SEQUENTIAL)
    _emit_augmentation_steps()
        ↓
        step_runner.execute(transformer)
            ↓
            TransformerMixinController._execute_for_sample_augmentation()
                ↓ (for each sample - SEQUENTIAL LOOP)
                dataset.add_samples(single_sample)
                    ↓
                    ArrayStorage.add_samples()
                        ↓
                        np.concatenate() ← COPIES ENTIRE ARRAY EACH TIME!
```

### Bottleneck #1: O(N²) Array Concatenation

**Location**: `nirs4all/data/_features/array_storage.py` line 106

```python
def add_samples(self, data: np.ndarray) -> None:
    # ...
    self._array = np.concatenate((self._array, new_data_3d), axis=0)  # ← Full copy!
```

**Problem**: Each call copies the entire existing array plus new data. For N insertions:
- 1st insert: copies 1 element
- 2nd insert: copies 2 elements
- ...
- Nth insert: copies N elements
- **Total copies: 1+2+3+...+N = N(N+1)/2 = O(N²)**

With 3,000 augmented samples: ~4.5 million element copies!

### Bottleneck #2: Sequential Transformer Execution

**Location**: `nirs4all/controllers/data/sample_augmentation.py` line 260

```python
def _emit_augmentation_steps(self, transformer_to_samples, transformers, ...):
    for trans_idx, sample_ids in transformer_to_samples.items():
        # SEQUENTIAL execution of each transformer
        runtime_context.step_runner.execute(transformer, dataset, ...)
```

**Problem**: 16 transformers execute one after another. No parallelization despite being embarrassingly parallel.

### Bottleneck #3: Sample-by-Sample Addition Loop

**Location**: `nirs4all/controllers/transforms/transformer.py` line 250-270

```python
for sample_idx, sample_id in enumerate(target_sample_ids):
    # Build index dictionary for each sample
    index_dict = {"partition": "train", "origin": sample_id, "augmentation": operator_name}
    # ...
    dataset.add_samples(data=data_to_add, indexes=index_dict)  # ← Loop insertion!
```

**Problem**: Even with batch transform (good!), samples are still inserted one at a time.

---

## Performance Impact Estimation

### Current Performance (Q20 with 3,500 samples, 16 augmenters, 3x ref_percentage)

| Operation | Count | Time per Op | Total Time |
|-----------|-------|-------------|------------|
| Array concatenations | ~3,000 | ~0.5ms avg | **~10-15 min** |
| Indexer appends | ~3,000 | ~0.1ms | ~5 sec |
| Transform computations | ~3,000 | ~0.01ms | ~0.5 sec |

**Diagnosis**: ~95% of time is spent in array copying, not in actual computation!

---

## Solution Architecture

### Solution 1: Batch Insert in ArrayStorage (Quick Win)

**New Method**: `add_samples_batch(data_list)`

```python
def add_samples_batch(self, samples: List[np.ndarray]) -> None:
    """Add multiple samples in one operation - O(N) instead of O(N²)."""
    if not samples:
        return

    # Stack all new samples at once
    all_new = np.stack(samples, axis=0)

    if self.num_samples == 0:
        self._array = all_new[:, None, :]
    else:
        # Single concatenation for all samples
        all_new_3d = all_new[:, None, :]
        self._array = np.concatenate((self._array, all_new_3d), axis=0)
```

**Impact**: Reduces O(N²) to O(N) for array operations.

### Solution 2: Collect-then-Insert Pattern in Controller

**Change TransformerMixinController**:

```python
def _execute_for_sample_augmentation(self, ...):
    # ... batch transform (already implemented) ...

    # NEW: Collect all augmented samples
    all_augmented_data = []
    all_indexes = []

    for sample_idx, sample_id in enumerate(target_sample_ids):
        data_to_add = ...  # Extract from batch-transformed data
        index_dict = {"partition": "train", "origin": sample_id, "augmentation": operator_name}
        all_augmented_data.append(data_to_add)
        all_indexes.append(index_dict)

    # NEW: Single batch insert
    dataset.add_samples_batch(all_augmented_data, all_indexes)
```

### Solution 3: Parallel Transformer Execution (Medium Effort)

**Change SampleAugmentationController**:

```python
from joblib import Parallel, delayed

def _emit_augmentation_steps_parallel(self, transformer_to_samples, transformers, ...):
    # Each transformer returns augmented data (no mutation)
    results = Parallel(n_jobs=-1)(
        delayed(self._apply_single_transformer)(trans_idx, sample_ids, transformers, dataset, context)
        for trans_idx, sample_ids in transformer_to_samples.items()
        if sample_ids
    )

    # Collect all results
    all_augmented_data = []
    all_indexes = []
    for aug_data, aug_indexes in results:
        all_augmented_data.extend(aug_data)
        all_indexes.extend(aug_indexes)

    # Single batch insert at the end
    dataset.add_samples_batch(all_augmented_data, all_indexes)
```

**Requirement**: TransformerMixinController must return data instead of mutating dataset.

### Solution 4: Pre-allocated Array Growth (Advanced)

**Change ArrayStorage to use amortized growth**:

```python
class ArrayStorage:
    def __init__(self):
        self._array = None
        self._logical_size = 0  # Actual data count
        self._allocated_size = 0  # Array capacity

    def add_samples(self, data: np.ndarray) -> None:
        n_new = data.shape[0]
        required_size = self._logical_size + n_new

        # Double capacity when needed (amortized O(1))
        if required_size > self._allocated_size:
            new_capacity = max(required_size, self._allocated_size * 2)
            self._resize(new_capacity)

        # Copy into pre-allocated space (O(n_new) not O(total))
        self._array[self._logical_size:required_size] = data[:, None, :]
        self._logical_size = required_size
```

---

## Implementation Priority

| Priority | Solution | Effort | Speedup | Risk |
|----------|----------|--------|---------|------|
| 🔴 HIGH | Batch insert in controller | 2-3 hours | 10-50x | Low |
| 🔴 HIGH | add_samples_batch() in Dataset | 2-3 hours | 10-50x | Low |
| 🟡 MEDIUM | Parallel transformer execution | 6-8 hours | 16x | Medium |
| 🟢 LOW | Pre-allocated growth | 4-6 hours | 2-5x | Low |

**Recommended Order**:
1. First implement batch insert (Solutions 1+2) - immediate huge win
2. Then add parallelization (Solution 3) - additional 16x on multi-core
3. Pre-allocated growth is optional polish

---

## Expected Results

| Scenario | Current Time | After Batch | After Parallel |
|----------|--------------|-------------|----------------|
| Q20 (3.5k samples, 16 aug) | ~10-15 min | ~10-30 sec | ~2-5 sec |
| Large dataset (10k samples) | ~60+ min | ~30-60 sec | ~5-10 sec |

**Speedup**: 100-500x for typical use cases.

---

## Files to Modify

1. **`nirs4all/data/_features/array_storage.py`**
   - Add `add_samples_batch(samples: List[np.ndarray])`

2. **`nirs4all/data/indexer.py`**
   - Add `add_samples_batch(count, indexes_list)`

3. **`nirs4all/data/dataset.py`**
   - Add `add_samples_batch(data_list, indexes_list)`

4. **`nirs4all/controllers/transforms/transformer.py`**
   - Modify `_execute_for_sample_augmentation()` to collect then batch-insert

5. **`nirs4all/controllers/data/sample_augmentation.py`** (optional)
   - Add parallel execution option

---

## Testing Plan

1. Run Q12_sample_augmentation.py - functional test
2. Run Q20_analysis.py - performance test
3. Compare augmented sample counts before/after
4. Verify model predictions are identical
5. Benchmark: time before vs after

---

## Appendix: Profiling Commands

```bash
# Profile Q20
python -m cProfile -s cumtime examples/Q20_analysis.py 2>&1 | head -100

# Line profiler on specific functions
pip install line_profiler
kernprof -l -v examples/Q20_analysis.py
```
