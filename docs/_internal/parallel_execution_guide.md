# Parallel Execution Guide

## Overview

The nirs4all library now supports parallel execution at two levels:
1. **Branch-level parallelization**: Execute multiple branches concurrently
2. **Optuna parallelization**: Run multiple optimization trials concurrently within each branch

This guide explains how to use these features effectively and avoid common pitfalls.

## Quick Start

### Basic Parallel Branches

```python
import nirs4all
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from nirs4all.operators.transforms import SNV, MSC

pipeline = [
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), Ridge(alpha=1.0)],
    ], "parallel": True},  # ← Enable parallel execution
]

result = nirs4all.run(pipeline, dataset="sample_data/regression")
```

### Explicit Worker Control

```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), Ridge(alpha=1.0)],
    ], "n_jobs": 2},  # ← Use exactly 2 parallel workers
]
```

### Parallel Optuna Optimization

```python
pipeline = [
    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 50,
            "n_jobs": 4,  # ← Parallel Optuna trials
            "model_params": {
                "n_components": ("int", 1, 25),
            },
        },
    },
]
```

## Configuration Options

### Branch-Level Parallelization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel` | bool | `False` | Enable auto-detected parallelization |
| `n_jobs` | int | `1` | Number of parallel workers. `-1` = auto (min of branches and CPU count) |

### Optuna Parallelization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | int | `1` | Number of parallel Optuna trials (in `finetune_params`) |

## Smart Parallelization Detection

The system automatically **disables** parallel execution when it detects:

### 1. Models with Internal Parallelization

```python
# ❌ Will auto-disable branch parallelization
{"branch": [
    [SNV(), PLSRegression()],
    [MSC(), RandomForestRegressor(n_jobs=-1)],  # ← Has internal parallelization
], "parallel": True}
```

**Why?** Running parallel branches with models that already use multiple threads causes
over-subscription and **slower** performance.

**Solution:** Use sequential branches or set model `n_jobs=1`:
```python
# ✅ Corrected version
{"branch": [
    [SNV(), PLSRegression()],
    [MSC(), RandomForestRegressor(n_jobs=1)],  # ← No internal parallelization
], "parallel": True}
```

### 2. Neural Network Models

```python
# ❌ Will auto-disable branch parallelization
{"branch": [
    [SNV(), PLSRegression()],
    [StandardScaler(), NICON()],  # ← Neural network model
], "parallel": True}
```

**Why?** Neural networks often use GPU or have complex thread management that conflicts
with process-level parallelization.

**Solution:** Group neural nets in a separate sequential branch:
```python
# ✅ Better strategy
# Group 1: Light models (parallel)
{"branch": [
    [SNV(), PLSRegression()],
    [MSC(), Ridge()],
], "n_jobs": 2},

# Group 2: Neural net (sequential)
{"branch": [[StandardScaler(), NICON()]], "parallel": False},
```

### 3. GPU Models

```python
# ❌ Will auto-disable branch parallelization
{"branch": [
    [SNV(), PLSRegression()],
    [StandardScaler(), TabPFNRegressor(device="cuda")],  # ← GPU model
], "parallel": True}
```

**Why?** Multiple processes competing for GPU resources causes memory issues and conflicts.

## Best Practices

### 1. Avoid Nested Parallelization

**❌ Don't do this:**
```python
pipeline = [
    {"branch": [
        [{"model": PLSRegression(), "finetune_params": {"n_trials": 50, "n_jobs": 4}}],
        [{"model": Ridge(), "finetune_params": {"n_trials": 50, "n_jobs": 4}}],
    ], "n_jobs": 2},  # ← Branch parallel + Optuna parallel = nested!
]
```

**Why?** This creates `2 * 4 = 8` concurrent processes, causing CPU over-subscription and
slower performance.

**✅ Choose one level of parallelization:**

**Option A: Parallel branches, sequential Optuna**
```python
pipeline = [
    {"branch": [
        [{"model": PLSRegression(), "finetune_params": {"n_trials": 50, "n_jobs": 1}}],
        [{"model": Ridge(), "finetune_params": {"n_trials": 50, "n_jobs": 1}}],
    ], "n_jobs": 2},  # ← Parallel branches only
]
```

**Option B: Sequential branches, parallel Optuna**
```python
pipeline = [
    {"branch": [
        [{"model": PLSRegression(), "finetune_params": {"n_trials": 50, "n_jobs": 4}}],
        [{"model": Ridge(), "finetune_params": {"n_trials": 50, "n_jobs": 4}}],
    ], "parallel": False},  # ← Parallel Optuna only
]
```

### 2. Group by Computational Intensity

**✅ Good strategy:**
```python
# Light models: Can run in parallel
{"branch": [
    [SNV(), PLSRegression()],
    [MSC(), Ridge()],
    [StandardScaler(), ElasticNet()],
], "n_jobs": 3},

# Heavy model: Runs alone
{"branch": [[Detrend(), RandomForestRegressor(n_estimators=500, n_jobs=-1)]], "parallel": False},
```

### 3. Memory Management

Parallel execution creates **deep copies** of the dataset for each worker.

**For large datasets:**
- Use `n_jobs` to limit workers (e.g., `n_jobs=2` instead of auto)
- Monitor memory usage during development
- Consider sequential execution if memory is constrained

```python
# Limit workers for large datasets
{"branch": [...], "n_jobs": 2}  # Instead of n_jobs=-1
```

### 4. Predict/Explain Mode

Parallel execution is automatically **disabled** in predict/explain mode due to artifact
loading requirements. This is expected behavior.

## Common Patterns

### Pattern 1: Pure Parallel (Recommended)

Best for: Multiple lightweight models with similar computational cost

```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), Ridge(alpha=1.0)],
        [StandardScaler(), ElasticNet(alpha=0.5)],
        [Detrend(), Lasso(alpha=0.1)],
    ], "n_jobs": 4},  # All 4 branches in parallel
]
```

### Pattern 2: Mixed Strategy

Best for: Combination of light and heavy models

```python
pipeline = [
    # Light models (parallel)
    {"branch": [
        [SNV(), PLSRegression()],
        [MSC(), Ridge()],
    ], "n_jobs": 2},

    # Heavy model (sequential)
    {"branch": [[StandardScaler(), RandomForestRegressor(n_jobs=-1)]], "parallel": False},
]
```

### Pattern 3: Optuna-Focused

Best for: Expensive hyperparameter search

```python
pipeline = [
    {"branch": [
        [{
            "model": PLSRegression(),
            "finetune_params": {
                "n_trials": 100,
                "n_jobs": 8,  # Parallel Optuna trials
                "model_params": {"n_components": ("int", 1, 30)},
            },
        }],
        [{
            "model": Ridge(),
            "finetune_params": {
                "n_trials": 100,
                "n_jobs": 8,  # Parallel Optuna trials
                "model_params": {"alpha": ("float_log", 1e-5, 1e3)},
            },
        }],
    ], "parallel": False},  # Sequential branches (Optuna is parallel)
]
```

### Pattern 4: Generator-Based Branches

Works with parallel execution:

```python
pipeline = [
    {"branch": {
        "_or_": [SNV, MSC, Detrend, StandardScaler],  # Generates 4 branches
    }, "n_jobs": 4},  # All branches in parallel
    PLSRegression(n_components=10),
]
```

## Performance Tips

### 1. Profile First

Run with `verbose=2` to see execution times:
```python
result = nirs4all.run(pipeline, dataset, verbose=2)
```

### 2. Start Conservative

Begin with `n_jobs=2` and increase gradually:
```python
{"branch": [...], "n_jobs": 2}  # Start here
{"branch": [...], "n_jobs": 4}  # Then try this
{"branch": [...], "n_jobs": -1}  # Finally auto-detect
```

### 3. Monitor Resource Usage

Use system monitoring tools:
```bash
# Linux/macOS
htop

# Monitor memory
watch -n 1 free -h
```

### 4. Benchmark

Compare sequential vs. parallel:
```bash
# Sequential
time python my_pipeline.py

# Parallel (2 workers)
time python my_pipeline.py --parallel-branches 2

# Parallel (auto)
time python my_pipeline.py --parallel-branches -1
```

## Troubleshooting

### Issue: "Parallel execution disabled: Branch X uses n_jobs=Y"

**Cause:** Model has internal parallelization (`n_jobs > 1`)

**Fix:** Set model `n_jobs=1` or use sequential branches
```python
# Before
RandomForestRegressor(n_jobs=-1)

# After
RandomForestRegressor(n_jobs=1)
```

### Issue: Out of Memory

**Cause:** Too many parallel workers for dataset size

**Fix:** Reduce `n_jobs`:
```python
{"branch": [...], "n_jobs": 2}  # Instead of -1
```

### Issue: Slower with Parallel Execution

**Possible causes:**
1. Nested parallelization (branch + Optuna + model)
2. CPU over-subscription
3. I/O bottlenecks

**Fix:** Profile and adjust:
```python
# Disable one level of parallelization
{"branch": [...], "parallel": False}  # Try sequential first
```

### Issue: DuckDB/SQLite Errors

**Cause:** Concurrent database writes (should not happen with current implementation)

**Note:** Workers use local prediction stores and merge results in main process.
If you see this, it's a bug - please report it.

## Implementation Details

### Process Isolation

Each parallel worker:
- Gets a **deep copy** of the dataset
- Has its own **prediction store** (merged later)
- **Disables** DuckDB/WorkspaceStore writes (to avoid conflicts)
- **Disables** trace recording (not thread-safe)

### Thread Safety

The following are **disabled** in parallel workers:
- WorkspaceStore operations (DuckDB writes)
- ArtifactRegistry writes
- Execution trace recording

Predictions are collected locally and **merged** in the main process after all workers complete.

### Backend

Uses `joblib.Parallel` with `backend='loky'` for:
- Process isolation (no shared state)
- Robust error handling
- Progress reporting

## Examples

See the following files for complete examples:
- `examples/developer/06_internals/D04_parallel_branches.py` - Comprehensive examples
- `bench/tabpfn_paper/full_run_parallel.py` - Production example with TabPFN paper

## FAQ

**Q: Can I use parallel branches with parallel Optuna?**

A: Not recommended. Choose one level of parallelization to avoid CPU over-subscription.

**Q: Does parallel execution work in predict mode?**

A: No, it's automatically disabled in predict/explain mode due to artifact loading requirements.

**Q: How many workers should I use?**

A: Start with `n_jobs=2`, then increase. Use `n_jobs=-1` for auto-detection, but monitor
resource usage first.

**Q: Does this work with neural networks?**

A: System auto-detects neural nets and disables parallel execution. Group them separately
for sequential execution.

**Q: Can I parallelize Optuna across datasets?**

A: Optuna parallelization is per-dataset. For cross-dataset parallelization, use
Optuna's distributed optimization with shared storage.

**Q: Why is my parallel run slower?**

A: Common causes: nested parallelization, dataset copying overhead, I/O bottlenecks.
Profile with `verbose=2` and reduce `n_jobs`.
