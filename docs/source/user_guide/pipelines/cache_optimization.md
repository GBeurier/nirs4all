# Cache Optimization

NIRS4ALL includes a sophisticated caching system designed to dramatically reduce execution time and memory usage when running pipelines with multiple variants. This guide covers when, why, and how to use caching effectively.

## What is Caching in nirs4all?

The caching system optimizes two common scenarios:

1. **Generator-heavy pipelines**: When using `_or_`, `_range_`, or `_cartesian_` keywords to create multiple pipeline variants, preprocessing steps before the generator can be reused across all variants.

2. **Branch-heavy pipelines**: When using branching and merging for stacking or ensemble methods, copy-on-write (CoW) snapshots reduce memory overhead.

### Without Caching

When you run a generator pipeline like this:

```python
pipeline = [
    StandardNormalVariate(),
    {"_or_": [SavitzkyGolay(window_length=11), SavitzkyGolay(window_length=15)]},
    {"_range_": [5, 20, 5], "param": "n_components", "model": PLSRegression},
]
```

This creates 2 × 4 = 8 variants. Without caching, SNV runs 8 times (once per variant).

### With Caching

With step caching enabled, SNV runs once and its output is reused for all 8 variants via copy-on-write (CoW) `SharedBlocks` references. Cache hits are near-free — they only acquire an additional reference to the cached data without copying it. The actual data copy is deferred to when the next pipeline step mutates the data.

## Quick Start

Enable caching with `CacheConfig`:

```python
from nirs4all.config.cache_config import CacheConfig
import nirs4all

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    cache=CacheConfig(
        step_cache_enabled=True,        # Enable step-level caching
        use_cow_snapshots=True,         # Enable CoW branch snapshots
        step_cache_max_mb=2048,         # 2 GB cache budget
    ),
    verbose=1
)
```

At the end of execution (when `verbose >= 1`), you'll see cache statistics:

```
=== Cache Statistics ===
Step Cache Entries  : 12
Step Cache Hits     : 45
Step Cache Hit Rate : 78.9%
Step Cache Size     : 512.3 MB
Snapshot Time       : 245.3 ms
Restore Time        : 1.2 ms
Peak Memory (RSS)   : 1847 MB
```

Note the asymmetry: snapshot time (creating entries) is much higher than restore time (cache hits) because restore uses CoW references — no data is copied.

## Configuration Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `step_cache_enabled` | `False` | Enable step-level caching for reusing preprocessing across variants |
| `step_cache_max_mb` | `2048` | Maximum memory budget for step cache in megabytes |
| `step_cache_max_entries` | `200` | Maximum number of cached entries (acts as a secondary limit) |
| `use_cow_snapshots` | `True` | Use copy-on-write snapshots for branch features (reduces memory) |
| `log_cache_stats` | `True` | Log cache statistics at end of run (requires `verbose >= 1`) |
| `log_step_memory` | `True` | Log per-step RSS memory usage (requires `verbose >= 2`) |
| `memory_warning_threshold_mb` | `3072` | RSS threshold in MB that triggers memory warnings |

### Step Cache vs CoW Snapshots

These are **independent** features that complement each other:

- **Step cache** (`step_cache_enabled=True`): Reuses preprocessing outputs across generator variants
- **CoW snapshots** (`use_cow_snapshots=True`): Reduces memory overhead when branching

You can enable one, both, or neither depending on your use case.

## When to Use Cache

### ✅ Ideal Use Cases

1. **Generator-heavy pipelines** with `_or_`, `_range_`, or `_cartesian_`
2. **Expensive preprocessing** (derivatives, wavelet transforms, EMSC)
3. **Large datasets** where preprocessing dominates runtime
4. **Branching pipelines** for stacking or ensemble methods
5. **Hyperparameter tuning** with Optuna (many variants tested)

### ❌ When NOT to Use Cache

1. **Simple pipelines** with no generators (no variants to reuse)
2. **Memory-constrained environments** (cache uses RAM)
3. **Tiny datasets** (overhead exceeds benefit)
4. **Non-deterministic preprocessing** (rare edge case)

## Complete Example

Here's a realistic pipeline that benefits from caching:

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    SavitzkyGolay,
    Detrend,
    EMSC,
)
from nirs4all.config.cache_config import CacheConfig

# Define pipeline with multiple generators
pipeline = [
    ShuffleSplit(n_splits=3, test_size=0.25),

    # Stage 1: Scatter correction (4 variants)
    {"_or_": [None, StandardNormalVariate(), EMSC(), Detrend()]},

    # Stage 2: Smoothing (3 variants)
    {"_or_": [None, SavitzkyGolay(window_length=11), SavitzkyGolay(window_length=15)]},

    # Stage 3: Model with parameter sweep (4 variants)
    {"_range_": [5, 20, 5], "param": "n_components", "model": PLSRegression},
]
# Total variants: 4 × 3 × 4 = 48

# Run with caching enabled
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    cache=CacheConfig(
        step_cache_enabled=True,
        use_cow_snapshots=True,
        step_cache_max_mb=2048,
    ),
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Total variants tested: {result.num_predictions}")
```

With caching, this pipeline runs in a fraction of the time compared to naive execution.

## How Step Caching Works (CoW)

Each cached entry stores lightweight CoW `SharedBlocks` references to the preprocessed feature arrays, not deep copies. This means:

1. **Snapshot (cache put)**: Calls `ensure_shared().acquire()` per feature source — O(n_sources), creates a shared reference to the existing data.
2. **Restore (cache hit)**: Calls `restore_from_shared(shared.acquire())` — O(n_sources), near-free. No data is copied.
3. **On mutation**: When the next pipeline step modifies the data, `SharedBlocks.detach()` creates an independent copy — the copy is deferred to when it's actually needed.

This design ensures cache hits never slow down the pipeline, even for cheap transforms.

## Memory Management

### Cache Budget (`step_cache_max_mb`)

The cache uses LRU eviction:

1. Tracks total memory used by cached entries
2. When adding a new entry would exceed the budget, evicts least-recently-used entries
3. Evicted entries automatically release their `SharedBlocks` references

**Choosing a budget:**

- Small datasets (< 1000 samples): 512 MB is sufficient
- Medium datasets (1000-10000 samples): 1024-2048 MB
- Large datasets (> 10000 samples): 2048-4096 MB

### Memory Warnings

If process RSS exceeds `memory_warning_threshold_mb` (default 3072 MB), you'll see a warning:

```
WARNING: High memory usage detected (3521 MB RSS)
```

This helps you spot memory issues before they cause crashes.

## Performance Benchmarks

See the working benchmark example for detailed performance comparisons:

```bash
cd examples/developer/06_internals
python D03_cache_performance.py
```

This script compares three configurations:

1. **No cache** (baseline)
2. **CoW snapshots only** (branch memory reduction)
3. **Full cache** (CoW step cache + CoW branches)

It also includes a **strict reproducibility test** that verifies cached and uncached runs produce bitwise-identical prediction scores (tolerance: 1e-10). This guarantees that caching does not alter results.

**Key takeaway**: Step caching provides speedup for generator-heavy pipelines (shared preprocessing prefixes computed once), CoW reduces branch memory overhead.

## Cache Behavior by Pipeline Stage

Not all pipeline steps benefit equally from caching:

| Stage | Cached? | Notes |
|-------|---------|-------|
| Splitter (e.g., `ShuffleSplit`) | ✅ Yes | Fold assignments are cached |
| Preprocessing transforms | ✅ Yes | Main benefit of step cache |
| Generators (`_or_`, `_range_`) | ❌ No | Generators create variants, don't transform data |
| Models | ❌ No | Models are trained per-variant (not reusable) |
| Branches | ✅ Yes (CoW) | CoW snapshots reduce memory |

## Advanced: Observability

For deep performance analysis, enable per-step memory logging:

```python
cache = CacheConfig(
    step_cache_enabled=True,
    use_cow_snapshots=True,
    log_step_memory=True,  # Per-step RSS logging (verbose >= 2)
)

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=cache,
    verbose=2  # Required for log_step_memory
)
```

Output includes RSS delta after each step:

```
Step [Splitter] ShuffleSplit (RSS: +12 MB)
Step [Transform] StandardNormalVariate (RSS: +45 MB)
Step [Transform] SavitzkyGolay (RSS: +23 MB)
...
```

This helps identify memory-intensive steps.

## Troubleshooting

### Cache Not Being Used

**Symptom**: Cache hit rate is 0% despite using generators.

**Causes:**

1. Generators are before preprocessing steps (no prefix to reuse)
2. Dataset changes between runs (cache keys are dataset-specific)
3. Cache budget is too small (entries evicted before reuse)

**Solution**: Move generators after preprocessing, increase `step_cache_max_mb`.

### High Memory Usage

**Symptom**: Process RSS grows beyond expectations.

**Causes:**

1. Cache budget is too large for available RAM
2. Large datasets with many cached entries
3. Memory leak in custom operators

**Solution**: Reduce `step_cache_max_mb`, use CoW snapshots, profile custom operators.

### Slower Performance with Cache

**Symptom**: Caching makes pipelines slower, not faster.

**Causes:**

1. No generators (nothing to reuse — cache has no effect)
2. Cache thrashing (budget too small, constant eviction and re-computation)

**Solution**: Disable caching for pipelines without generators, or increase cache budget to reduce thrashing.

Note: With the CoW implementation, cache hit overhead is negligible (~0.01ms per restore). The main cost is the initial snapshot when storing entries.

## Best Practices

1. **Start with defaults**: `step_cache_enabled=True`, `use_cow_snapshots=True`
2. **Monitor stats**: Check cache hit rate and memory usage
3. **Adjust budget**: Increase if hit rate is low, decrease if memory is tight
4. **Profile first**: Run without cache to establish baseline
5. **Test incrementally**: Enable CoW first, then step cache

## See Also

- [D03 Cache Performance Example](../../../examples/developer/06_internals/D03_cache_performance.py) - Runnable benchmark with reproducibility test
- [Pipeline Syntax Reference](../../reference/pipeline_syntax.md) - Generator keywords

```{seealso}
**Related Examples:**
- [D03: Cache Performance](../../../examples/developer/06_internals/D03_cache_performance.py) - Benchmark step caching, CoW snapshots, and reproducibility verification
- [D01: Generator Syntax](../../../examples/developer/02_generators/D01_generator_syntax.py) - Generator syntax that benefits from caching
```
