# Nested Cross-Validation for NIRS4ALL

This document explains the comprehensive cross-validation functionality implemented in NIRS4ALL for rigorous hyperparameter optimization and model evaluation.

## Overview

NIRS4ALL supports three different cross-validation strategies for model finetuning:

1. **Simple CV** - Finetune on full training data, then train on folds
2. **Per-fold CV** - Finetune on each fold individually
3. **Nested CV** - Inner folds for finetuning, outer folds for training (academic-level rigor)

## Parameter Strategies

NIRS4ALL now supports seven different parameter optimization strategies:

1. **GLOBAL_BEST** - Select the single best parameter set from all fold optimizations
2. **PER_FOLD_BEST** - Each fold uses its own individually optimized parameters
3. **WEIGHTED_AVERAGE** - Average parameters weighted by fold performance
4. **GLOBAL_AVERAGE** - Optimize parameters by averaging performance across ALL folds simultaneously ⭐ **NEW**
5. **ENSEMBLE_BEST** - Optimize for ensemble prediction performance ⭐ **NEW**
6. **ROBUST_BEST** - Optimize for minimum worst-case performance (min-max) ⭐ **NEW**
7. **STABILITY_BEST** - Optimize for parameter stability (minimize performance variance) ⭐ **NEW**

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

Control how parameters are optimized across folds with `param_strategy`:

```python
"finetune_params": {
    "param_strategy": "global_average",  # See full list above
    # ... other parameters
}
```

### Nested CV Configuration

For nested CV, specify inner cross-validation settings:

```python
"finetune_params": {
    "cv_mode": "nested",
    "inner_cv": 3,  # Number of inner folds or sklearn CV object
    "param_strategy": "global_average",
    # ... other parameters
}
```

### Full Training Data Option

**NEW**: You can now optimize parameters using cross-validation but train the final model on the full training dataset:

```python
"finetune_params": {
    "cv_mode": "per_fold",
    "param_strategy": "global_average",
    "use_full_train_for_final": True,  # ⭐ NEW OPTION
    "n_trials": 15,
    "model_params": {
        "n_components": ("int", 1, 20)
    }
}
```

**How it works:**
- Uses cross-validation folds for rigorous hyperparameter optimization
- After finding optimal parameters, combines ALL training data from all folds
- Trains a single final model on the combined training dataset
- Tests on the combined test data from all folds

**Benefits:**
- Maximizes training data for the final model
- Maintains rigorous hyperparameter optimization
- Often improves final model performance
- Simpler deployment (single model instead of ensemble)

## Detailed Parameter Strategies

### 1. Global Best (`param_strategy: "global_best"`)

**How it works:**
- Runs separate optimization on each fold
- Selects the single best performing parameter set across all folds
- Applies this parameter set to all folds

**Pros:**
- Simple and intuitive
- Ensures consistent parameters across folds
- Moderate computational cost

**Cons:**
- May not be optimal for individual folds
- Can overfit to the best performing fold

### 2. Per-Fold Best (`param_strategy: "per_fold_best"`)

**How it works:**
- Runs separate optimization on each fold
- Each fold uses its own best parameter set

**Pros:**
- Accounts for fold-to-fold variability
- Each fold gets its optimal parameters

**Cons:**
- Parameter sets may vary significantly between folds
- Higher computational cost

### 3. Global Average (`param_strategy: "global_average"`) ⭐ **NEW**

**How it works:**
- Each parameter set is evaluated across ALL folds simultaneously
- The objective function averages validation scores across all folds
- Returns the parameter set with the best average performance

**Pros:**
- Most generalizable parameter selection
- Single parameter set optimized for average performance
- Reduces overfitting to individual folds

**Cons:**
- Highest computational cost (each trial trains on all folds)
- May not be optimal for any single fold

**Best for:**
- When you want the most generalizable parameters
- Production models where consistent performance is important
- Research requiring rigorous parameter validation

### 4. Ensemble Best (`param_strategy: "ensemble_best"`) ⭐ **NEW**

**How it works:**
- Optimizes parameters for ensemble prediction performance
- Trains multiple models with different parameter sets
- Combines predictions and optimizes ensemble performance

**Pros:**
- Can improve overall prediction performance
- Leverages diversity in parameter space

**Cons:**
- Most complex implementation
- Highest computational requirements

### 5. Robust Best (`param_strategy: "robust_best"`) ⭐ **NEW**

**How it works:**
- Optimizes for minimum worst-case performance
- Uses min-max objective: minimize the maximum error across folds
- Ensures consistent performance across all folds

**Pros:**
- Guarantees consistent performance
- Good for risk-averse applications

**Cons:**
- May sacrifice average performance for consistency
- Can be conservative in parameter selection

### 6. Stability Best (`param_strategy: "stability_best"`) ⭐ **NEW**

**How it works:**
- Optimizes for parameter stability
- Minimizes variance in performance across folds
- Balances average performance with consistency

**Pros:**
- Provides stable, predictable performance
- Good balance between performance and consistency

**Cons:**
- May not achieve best possible performance
- Complex multi-objective optimization

## Computational Cost Analysis

Assuming 5 outer folds and 20 optimization trials:

| Strategy | Model Training Count | Relative Cost | Best Use Case |
|----------|---------------------|---------------|---------------|
| Simple CV | ~25 models | 1x (baseline) | Prototyping |
| Per-fold CV (per_fold_best) | ~100 models | 4x | Standard use |
| Per-fold CV (global_average) | ~500 models | 20x | Rigorous optimization |
| Nested CV (per_fold_best) | ~150 models | 6x | Academic research |
| Nested CV (global_average) | ~750 models | 30x | Ultimate rigor |

## Example Configurations

### Global Average with Random Forest
```python
{
    "model": RandomForestRegressor(),
    "finetune_params": {
        "cv_mode": "per_fold",
        "param_strategy": "global_average",
        "n_trials": 15,  # Fewer trials due to higher cost per trial
        "model_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None]
        }
    }
}
```

### Robust Best with TensorFlow Model
```python
{
    "model": nicon,  # TensorFlow model function
    "finetune_params": {
        "cv_mode": "nested",
        "param_strategy": "robust_best",
        "inner_cv": 3,
        "n_trials": 10,
        "model_params": {
            "filters1": [4, 8, 16],
            "dropout_rate": ("float", 0.1, 0.5)
        }
    }
}
```

### Simple CV with Global Average for Quick Testing
```python
{
    "model": PLSRegression(),
    "finetune_params": {
        "cv_mode": "simple",
        "param_strategy": "global_average",
        "n_trials": 20,
        "model_params": {
            "n_components": ("int", 1, 20)
        }
    }
}
```

## Best Practices

1. **Choose appropriate strategy:**
   - `global_average`: For most generalizable parameters
   - `per_fold_best`: For fold-specific optimization
   - `robust_best`: For consistent performance requirements
   - `stability_best`: For balanced performance and stability

2. **Adjust trial counts based on computational cost:**
   - `global_average`: Fewer trials (10-20) due to high cost per trial
   - `per_fold_best`: Standard trials (20-30)
   - `robust_best`/`stability_best`: Moderate trials (15-25)

3. **Use appropriate CV modes with strategies:**
   - `simple` + `global_average`: Quick but rigorous optimization
   - `per_fold` + `global_average`: Standard rigorous approach
   - `nested` + `global_average`: Ultimate rigor for research

4. **Monitor computational resources:**
   - Start with smaller trial counts to estimate runtime
   - Consider using fewer folds for initial experiments
   - Use `verbose=1` to monitor progress

5. **Strategy selection guide:**
   ```python
   # For prototyping and quick experiments
   "param_strategy": "per_fold_best"

   # For production models requiring generalizability
   "param_strategy": "global_average"

   # For applications requiring consistent performance
   "param_strategy": "robust_best"

   # For research requiring rigorous validation
   "param_strategy": "global_average" + "cv_mode": "nested"
   ```

## Integration with Existing Code

All new parameter strategies are fully backward compatible:
- Existing configurations continue to work unchanged
- Default behavior remains the same (per_fold_best)
- New parameters are optional with sensible defaults
- Progressive adoption: start with existing strategies, then explore new ones

## Error Handling and Fallbacks

The implementation includes several robust fallback mechanisms:

- Falls back to standard training if Optuna is not available
- Handles parameter application failures gracefully
- Provides informative error messages for configuration issues
- Continues processing even if some folds fail
- Automatic parameter validation and constraint handling