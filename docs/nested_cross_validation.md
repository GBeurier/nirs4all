# Nested Cross-Validation for NIRS4ALL

This document explains the nested cross-validation functionality implemented in NIRS4ALL for more rigorous hyperparameter optimization and model evaluation.

## Overview

NIRS4ALL now supports three different cross-validation strategies for model finetuning:

1. **Simple CV** - Finetune on full training data, then train on folds
2. **Per-fold CV** - Finetune on each fold individually
3. **Nested CV** - Inner folds for finetuning, outer folds for training (academic-level rigor)

## Configuration

### Basic CV Mode Selection

Add the `cv_mode` parameter to your `finetune_params`:

```python
"finetune_params": {
    "cv_mode": "nested",  # Options: "simple", "per_fold", "nested"
    # ... other parameters
}
```

### Parameter Aggregation Strategies

Control how parameters are aggregated across folds with `param_strategy`:

```python
"finetune_params": {
    "param_strategy": "per_fold_best",  # Options: "global_best", "per_fold_best", "weighted_average"
    # ... other parameters
}
```

### Nested CV Configuration

For nested CV, specify inner cross-validation settings:

```python
"finetune_params": {
    "cv_mode": "nested",
    "inner_cv": 3,  # Number of inner folds or sklearn CV object
    "param_strategy": "per_fold_best",
    # ... other parameters
}
```

## Detailed Strategies

### 1. Simple CV (`cv_mode: "simple"`)

**How it works:**
- Combines all training data from all folds
- Runs hyperparameter optimization once on this combined dataset
- Uses the single best parameter set to train individual models on each fold

**Pros:**
- Fastest execution
- Uses maximum data for parameter optimization
- Simple to understand and implement

**Cons:**
- Less rigorous parameter validation
- May overfit to the specific train/test split used for optimization
- Not suitable for academic publications requiring rigorous validation

**Best for:**
- Quick prototyping
- When computational resources are limited
- Exploratory data analysis

### 2. Per-fold CV (`cv_mode: "per_fold"`)

**How it works:**
- Runs separate hyperparameter optimization on each fold
- Each fold gets its own best parameter set
- Can aggregate parameters using different strategies

**Parameter Strategies:**
- `"per_fold_best"`: Each fold uses its own optimized parameters
- `"global_best"`: Select the single best performing parameter set across all folds

**Pros:**
- More rigorous than simple CV
- Accounts for fold-to-fold variability in optimal parameters
- Moderate computational cost

**Cons:**
- Higher computational cost than simple CV
- May still have some optimization bias per fold
- Parameter sets may vary significantly between folds

**Best for:**
- Production models where fold variability is important
- When you have sufficient computational resources
- Balancing rigor with computational cost

### 3. Nested CV (`cv_mode: "nested"`)

**How it works:**
- For each outer fold:
  - Creates inner cross-validation folds from the training data
  - Runs hyperparameter optimization using inner folds
  - Trains final model on full outer training data with best parameters
  - Evaluates on outer test data

**Parameter Strategies:**
- `"per_fold_best"`: Each outer fold uses its own inner-CV optimized parameters
- `"weighted_average"`: Average parameters weighted by inner-CV performance

**Pros:**
- Most rigorous approach
- Unbiased parameter selection and performance estimation
- Suitable for academic publications
- Provides realistic performance estimates

**Cons:**
- Highest computational cost
- Complex to understand and debug
- May require significant time and resources

**Best for:**
- Research and academic work
- When unbiased performance estimates are critical
- Final model validation before deployment
- Publications requiring rigorous validation

## Computational Cost Analysis

Assuming 5 outer folds and 20 optimization trials:

| Strategy | Model Training Count | Relative Cost |
|----------|---------------------|---------------|
| Simple CV | ~25 models | 1x (baseline) |
| Per-fold CV | ~100 models | 4x |
| Nested CV (3 inner folds) | ~150 models | 6x |

## Example Configurations

### Simple CV with Random Forest
```python
{
    "model": RandomForestRegressor(),
    "finetune_params": {
        "cv_mode": "simple",
        "n_trials": 20,
        "model_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None]
        }
    }
}
```

### Nested CV with TensorFlow Model
```python
{
    "model": nicon,  # TensorFlow model function
    "finetune_params": {
        "cv_mode": "nested",
        "inner_cv": 3,
        "param_strategy": "per_fold_best",
        "n_trials": 10,
        "model_params": {
            "filters1": [4, 8, 16],
            "dropout_rate": ("float", 0.1, 0.5)
        }
    }
}
```

## Best Practices

1. **Choose appropriate CV mode:**
   - Simple CV: Prototyping, resource constraints
   - Per-fold CV: Production models, balanced approach
   - Nested CV: Research, publications, final validation

2. **Adjust trial counts:**
   - Simple CV: More trials (20-50)
   - Per-fold CV: Moderate trials (10-20)
   - Nested CV: Fewer trials (5-15) due to computational cost

3. **Use silent training during optimization:**
   ```python
   "train_params": {
       "verbose": 0  # Silent during trials
   }
   ```

4. **Monitor computational cost:**
   - Start with smaller trial counts
   - Estimate total runtime before full runs
   - Consider using fewer inner folds for nested CV

5. **Parameter strategy selection:**
   - Use `"per_fold_best"` for most cases
   - Consider `"weighted_average"` for nested CV with numerical parameters
   - Use `"global_best"` when parameter consistency across folds is important

## Error Handling

The implementation includes several fallback mechanisms:

- Falls back to standard training if Optuna is not available
- Handles parameter application failures gracefully
- Provides informative error messages for configuration issues
- Continues processing even if some folds fail

## Integration with Existing Code

The nested CV functionality is fully backward compatible:
- Existing configurations continue to work unchanged
- Default behavior remains the same (simple training/finetuning)
- New parameters are optional with sensible defaults