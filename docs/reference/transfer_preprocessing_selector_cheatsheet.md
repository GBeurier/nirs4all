# TransferPreprocessingSelector Cheat Sheet

## Purpose

Find preprocessings that:
1. **Minimize distribution gap** between train and test sets
2. **Preserve predictive information** (don't destroy useful signal)

---

## Quick Start

```python
from nirs4all.analysis import TransferPreprocessingSelector

# Simple usage with preset
selector = TransferPreprocessingSelector(preset="balanced")
results = selector.fit(X_train, X_test)

# Get top recommendations
top_pp = results.to_preprocessing_list(top_k=10)
```

---

## Stages Overview

| Stage | Name | What it does | Enabled by |
|-------|------|--------------|------------|
| 1 | Single Eval | Tests all base preprocessings | Always on |
| 1b | Generator Stacked | Tests stacked combos from `preprocessing_spec` | `preprocessing_spec` |
| 2 | Stacking | Combines top-K singles into depth-2/3 pipelines | `run_stage2=True` |
| 2b | Generator Augmented | Tests augmentation from `preprocessing_spec` | `preprocessing_spec` + `pick` |
| 3 | Augmentation | Concatenates features from multiple pipelines | `run_stage3=True` |
| 4 | Supervised Validation | Validates with proxy models (Ridge, PLS) | `run_stage4=True` + `y_source` |

---

## Presets

| Preset | Stage 2 | Stage 3 | Stage 4 | Speed |
|--------|---------|---------|---------|-------|
| `fast` | ❌ | ❌ | ❌ | ⚡ Fastest |
| `balanced` | ✅ (top-5, depth-2) | ❌ | ❌ | 🔄 Medium |
| `thorough` | ✅ (top-10, depth-3) | ✅ | ❌ | 🐢 Slow |
| `full` | ✅ (top-15, depth-3) | ✅ | ✅ | 🐌 Very slow |

---

## Generator Mode (preprocessing_spec)

### Cartesian Product with None → Generates All Depths

```python
preprocessing_spec = {
    "_cartesian_": [
        {"_or_": [None, SNV(), MSC()]},      # Stage 1: scatter correction
        {"_or_": [None, SavitzkyGolay()]},   # Stage 2: smoothing
        {"_or_": [None, FirstDerivative()]}, # Stage 3: derivative
    ]
}
# Generates: singletons, pairs, AND triples
# [None, None, D1()] → [D1()]  (singleton)
# [SNV(), None, D1()] → [SNV(), D1()]  (pair)
# [SNV(), SavGol(), D1()] → full pipeline
```

### Important: Exhaustive cartesian makes Stage 2 redundant!

```python
# If preprocessing_spec already generates all combinations:
selector = TransferPreprocessingSelector(
    preprocessing_spec=GLOBAL_PP,
    preset=None,         # Disable preset
    run_stage2=False,    # Already covered by cartesian
    run_stage4=True,     # Keep supervised validation
)
```

---

## Key Parameters

```python
TransferPreprocessingSelector(
    # Presets
    preset="balanced",           # "fast", "balanced", "thorough", "full", None

    # Generator mode
    preprocessing_spec=None,     # Dict with _cartesian_, _or_, etc.

    # Stage 2: Stacking
    run_stage2=False,
    stage2_top_k=5,              # How many singles to stack
    stage2_max_depth=2,          # Max pipeline depth (2 or 3)

    # Stage 3: Augmentation (feature concatenation)
    run_stage3=False,
    stage3_top_k=5,
    stage3_max_order=2,          # Concat 2 or 3 pipelines

    # Stage 4: Supervised validation
    run_stage4=False,
    stage4_top_k=10,

    # Metrics
    n_components=10,             # PCA components for metrics
    k_neighbors=10,              # For trustworthiness

    # Performance
    n_jobs=-1,                   # Parallel workers (-1 = all cores)
    verbose=1,
)
```

---

## Common Patterns

### Pattern 1: Quick Exploration
```python
selector = TransferPreprocessingSelector(preset="fast")
results = selector.fit(dataset_config)
```

### Pattern 2: Exhaustive with Generator
```python
GLOBAL_PP = {
    "_cartesian_": [
        {"_or_": [None, MSC(), SNV(), EMSC()]},
        {"_or_": [None, SavitzkyGolay(), Gaussian()]},
        {"_or_": [None, FirstDerivative(), SecondDerivative()]},
    ]
}

selector = TransferPreprocessingSelector(
    preprocessing_spec=GLOBAL_PP,
    preset=None,
    run_stage2=False,  # Cartesian already covers this
    run_stage4=True,   # Validate top candidates
    stage4_top_k=20,
)
results = selector.fit(X_train, X_test, y_train)
```

### Pattern 3: Discovery Mode (let Stage 2 explore)
```python
# Small base set, let Stage 2 find combinations
selector = TransferPreprocessingSelector(
    preset="thorough",  # Enables stage2 with depth-3
    # No preprocessing_spec → uses get_base_preprocessings()
)
```

---

## Results API

```python
results = selector.fit(X_train, X_test)

# Get rankings
results.best                    # Top TransferResult
results.top(k=10)               # Top 10 results
results.ranking                 # All results (sorted)

# Export
results.to_preprocessing_list(top_k=10)  # List of transforms
results.to_pipeline_spec()               # Pipeline-ready format
results.summary()                        # Text summary

# Metrics
results.raw_metrics             # Baseline (no preprocessing) metrics
results.timing                  # Stage timings
```

---

## Metrics Explained

| Metric | What it measures | Better when |
|--------|------------------|-------------|
| `mmd` | Maximum Mean Discrepancy | Lower |
| `wasserstein` | Distribution distance | Lower |
| `coral` | Covariance alignment | Lower |
| `trustworthiness` | Local structure preservation | Higher |
| `transfer_score` | Weighted combination | **Higher** |

---

## ⚠️ Gotchas

1. **Stage 2 only stacks singletons** - It doesn't stack your 4th-order pipelines together
2. **Generator mode bypasses `preprocessings` dict** - Objects are used directly
3. **None in cartesian = optional stage** - Generates lower-order pipelines
4. **Stage 4 requires y_source** - Pass labels for supervised validation
