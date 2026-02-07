# CacheConfig

**Module**: `nirs4all.config.cache_config`

Configuration for step-level caching and memory optimization in pipeline execution.

## Class Definition

```{eval-rst}
.. autoclass:: nirs4all.config.cache_config.CacheConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

## Quick Reference

### Constructor

```python
from nirs4all.config.cache_config import CacheConfig

cache = CacheConfig(
    step_cache_enabled=True,
    step_cache_max_mb=2048,
    step_cache_max_entries=200,
    use_cow_snapshots=True,
    log_cache_stats=True,
    log_step_memory=True,
    memory_warning_threshold_mb=3072,
)
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_cache_enabled` | `bool` | `False` | Enable step-level caching for cross-variant reuse |
| `step_cache_max_mb` | `int` | `2048` | Maximum memory budget for step cache (MB) |
| `step_cache_max_entries` | `int` | `200` | Maximum number of cached step entries |
| `use_cow_snapshots` | `bool` | `True` | Use copy-on-write snapshots for branch features |
| `log_cache_stats` | `bool` | `True` | Log cache statistics at end of run (requires `verbose >= 1`) |
| `log_step_memory` | `bool` | `True` | Log per-step memory stats (requires `verbose >= 2`) |
| `memory_warning_threshold_mb` | `int` | `3072` | RSS threshold for memory warnings (MB) |

## Usage

### Basic Usage

Enable caching with defaults:

```python
import nirs4all
from nirs4all.config.cache_config import CacheConfig

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    cache=CacheConfig(step_cache_enabled=True),
    verbose=1
)
```

### Custom Configuration

Fine-tune cache behavior:

```python
cache = CacheConfig(
    step_cache_enabled=True,
    step_cache_max_mb=4096,          # 4 GB cache budget
    step_cache_max_entries=500,      # More entries
    use_cow_snapshots=True,
    log_cache_stats=True,
    log_step_memory=False,           # Disable per-step logging
    memory_warning_threshold_mb=8192 # 8 GB warning threshold
)

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=cache,
    verbose=1
)
```

### Disable Caching

To explicitly disable caching:

```python
cache = CacheConfig(
    step_cache_enabled=False,
    use_cow_snapshots=False
)
```

Or simply omit the `cache` parameter (caching is off by default).

## Feature Details

### Step-Level Caching

When `step_cache_enabled=True`, preprocessing steps are cached and reused across generator variants.

**Example**: A pipeline with `_or_` and `_range_` generators:

```python
pipeline = [
    StandardNormalVariate(),  # Cached
    {"_or_": [None, SavitzkyGolay(11), SavitzkyGolay(15)]},  # 3 variants
    {"_range_": [5, 20, 5], "param": "n_components", "model": PLSRegression},  # 4 variants
]
# Total: 3 × 4 = 12 variants
# SNV runs once, output reused 12 times
```

**Cache Key**: Based on step type, parameters, and input dataset hash.

**Eviction Policy**: Least-recently-used (LRU) when `step_cache_max_mb` is exceeded.

### Copy-on-Write Snapshots

When `use_cow_snapshots=True`, branch features use CoW to reduce memory overhead.

**Example**: Branch pipeline:

```python
pipeline = [
    StandardNormalVariate(),
    {"branch": [
        [SavitzkyGolay(11), PLSRegression(10)],
        [SavitzkyGolay(15), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},
    {"model": Ridge()},
]
# Each branch gets a CoW snapshot (shares memory until modified)
```

**Memory Savings**: Typically 30-50% reduction in branch memory usage.

### Observability

#### Cache Statistics (`log_cache_stats=True`)

At end of run (when `verbose >= 1`):

```
=== Cache Statistics ===
Step Cache Entries  : 12
Step Cache Hits     : 45
Step Cache Hit Rate : 78.9%
Step Cache Size     : 512.3 MB
Peak Memory (RSS)   : 1847 MB
```

#### Per-Step Memory (`log_step_memory=True`)

During execution (when `verbose >= 2`):

```
Step [Transform] StandardNormalVariate (RSS: +45 MB)
Step [Transform] SavitzkyGolay (RSS: +23 MB)
Step [Model] PLSRegression (RSS: +12 MB)
```

#### Memory Warnings

When process RSS exceeds `memory_warning_threshold_mb`:

```
WARNING: High memory usage detected (3521 MB RSS)
```

## Performance Considerations

### When to Enable Caching

✅ **Recommended for:**

- Generator-heavy pipelines (`_or_`, `_range_`, `_cartesian_`)
- Expensive preprocessing (derivatives, wavelet transforms, EMSC)
- Large datasets (> 1000 samples)
- Branching/stacking pipelines
- Hyperparameter tuning with Optuna

❌ **Not recommended for:**

- Simple pipelines without generators
- Memory-constrained environments
- Small datasets (< 100 samples)
- Pipelines with only model steps

### Memory Budget Guidelines

| Dataset Size | Recommended `step_cache_max_mb` |
|--------------|----------------------------------|
| < 1000 samples | 512 MB |
| 1000-10000 samples | 1024-2048 MB |
| > 10000 samples | 2048-4096 MB |

**Note**: Actual memory usage depends on feature count and preprocessing complexity.

### Typical Performance

From `D03_cache_performance.py` benchmark (500 samples, generator-heavy pipeline):

| Configuration | Time | Speedup | Memory |
|---------------|------|---------|--------|
| No cache | 45.2s | 1.0× | +840 MB |
| CoW only | 46.1s | 0.98× | +620 MB |
| Full cache | 12.7s | **3.6×** | +480 MB |

## Troubleshooting

### Cache Hit Rate is Low

**Symptoms**: Cache hit rate < 20% despite using generators.

**Causes:**

1. Generators are before preprocessing steps
2. Cache budget too small (entries evicted before reuse)
3. Dataset hash changes between runs

**Solutions:**

- Move generators after preprocessing
- Increase `step_cache_max_mb`
- Use consistent dataset paths/configs

### High Memory Usage

**Symptoms**: Process RSS grows beyond expectations.

**Causes:**

1. Cache budget exceeds available RAM
2. Large datasets with many cached entries
3. Memory leak in custom operators

**Solutions:**

- Reduce `step_cache_max_mb`
- Enable `use_cow_snapshots`
- Profile custom operators

### Slower with Cache

**Symptoms**: Caching makes pipeline slower.

**Causes:**

1. Small dataset (overhead exceeds benefit)
2. No generators (nothing to reuse)
3. Cache thrashing (constant eviction)

**Solutions:**

- Disable caching for small datasets
- Increase cache budget
- Profile with `verbose=2`

## Implementation Details

### Cache Architecture

```
CacheConfig → PipelineRunner → PipelineOrchestrator → RuntimeContext
                                                           ↓
                                                      StepCache
                                                           ↓
                                                   (memory, LRU)
```

**Key Components:**

- **RuntimeContext**: Holds cache instance during execution
- **StepCache**: In-memory cache with LRU eviction
- **Cache Key**: Hash of (step_type, params, input_dataset_hash)

### Thread Safety

The cache is **not thread-safe**. Do not share `CacheConfig` instances across concurrent `nirs4all.run()` calls.

For parallel execution, use separate `CacheConfig` instances per process.

## See Also

- {doc}`/user_guide/pipelines/cache_optimization` - User guide with examples
- [D03 Cache Performance Example](../../../examples/developer/06_internals/D03_cache_performance.py) - Runnable benchmark
- [Architecture Documentation](../../developer/caching.md) - Deep dive into implementation
- {doc}`/reference/pipeline_syntax` - Generator keywords reference
