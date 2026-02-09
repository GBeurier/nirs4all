# Hyperparameter Tuning

Automatically optimize model hyperparameters using Optuna integration.

## Overview

nirs4all provides built-in hyperparameter optimization through **Optuna**, a state-of-the-art hyperparameter optimization framework. This allows you to automatically find the best model configuration without manual trial-and-error.

Key features:
- **Multiple search methods**: Grid search, random search, Bayesian optimization (TPE), CMA-ES
- **Flexible parameter types**: Integer ranges, float ranges, log-scale, categorical choices
- **Tuning approaches**: Global search, per-preprocessing group, per-fold optimization
- **Pruning**: Early termination of unpromising trials (median, successive halving, hyperband)
- **Multi-phase search**: Exploration followed by exploitation
- **Custom metrics**: Optimize for RMSE, R2, MAE, accuracy, etc.
- **Reproducibility**: Seed support for deterministic results
- **Seamless integration**: Works with any sklearn-compatible model

## Basic Usage

Enable hyperparameter tuning by adding `finetune_params` to your model step:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
import nirs4all

pipeline = [
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 20,
            "sampler": "tpe",
            "model_params": {
                "n_components": ('int', 1, 20),
            }
        }
    }
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset="path/to/data",
    name="HyperparameterTuning"
)

print(f"Best score: {result.best_score:.4f}")
```

## Configuration Options

### finetune_params Structure

```python
{
    "model": SomeModel(),
    "finetune_params": {
        "n_trials": 20,              # Number of optimization trials
        "sampler": "tpe",            # Search method
        "verbose": 1,                # Logging verbosity (0, 1, or 2)
        "approach": "single",        # Tuning approach
        "eval_mode": "best",         # For grouped: "best", "mean", "robust_best"
        "seed": 42,                  # Reproducible optimization
        "metric": "rmse",           # Metric to optimize (auto-infers direction)
        "pruner": "median",          # Prune unpromising trials
        "model_params": {            # Parameters to optimize
            "param1": ('int', 1, 10),
            "param2": ('float', 0.01, 1.0),
            "param3": [True, False],
        },
        "train_params": {            # Training params (also sampable)
            "epochs": ('int', 50, 300),
            "verbose": 0,            # Static value (not sampled)
        },
    }
}
```

### Parameter Type Specifications

#### Tuple Format (Most Common)

```python
# Integer ranges
"n_components": ('int', 1, 20)          # Uniform integer 1-20
"n_estimators": ('int_log', 10, 1000)   # Log-uniform integer

# Float ranges
"learning_rate": ('float', 0.001, 0.1)      # Uniform float
"alpha": ('float_log', 1e-4, 1e2)           # Log-uniform (for regularization)

# Categorical choices
"kernel": ['linear', 'rbf', 'poly']
"scale": [True, False]
```

#### Dict Format (Most Flexible)

```python
# Integer with step
"n_estimators": {'type': 'int', 'min': 10, 'max': 200, 'step': 10}

# Integer with log scale
"n_components": {'type': 'int', 'min': 1, 'max': 1000, 'log': True}

# Float with log scale
"learning_rate": {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True}

# Categorical
"max_depth": {'type': 'categorical', 'choices': [3, 5, 7, 10]}
```

### Search Methods (Samplers)

| Method | Description | Best For |
|--------|-------------|----------|
| `grid` | Exhaustive search over all combinations | Small categorical spaces |
| `random` | Random sampling | Quick baseline, large spaces |
| `tpe` | Tree-Parzen Estimator (Bayesian) | Medium to large spaces (default) |
| `cmaes` | CMA Evolution Strategy | Continuous parameters |
| `auto` | Automatic selection based on parameter types | When unsure |

### Pruning

Pruners terminate unpromising trials early, saving computation time:

| Pruner | Description |
|--------|-------------|
| `none` | No pruning (default) |
| `median` | Prune if worse than median of completed trials |
| `successive_halving` | Prune poorest fraction at each step |
| `hyperband` | Adaptive resource allocation with early stopping |

```python
"finetune_params": {
    "n_trials": 50,
    "sampler": "tpe",
    "pruner": "median",         # Enable pruning
    "approach": "grouped",       # Pruning works with grouped approach
    "model_params": {
        "n_components": ('int', 1, 30),
    }
}
```

### Custom Metrics

By default, finetuning minimizes MSE for regression or maximizes balanced accuracy for classification. Use `metric` to optimize for a different objective:

```python
# Optimize for R2 (auto-infers direction=maximize)
"finetune_params": {
    "metric": "r2",
    "model_params": {"alpha": ('float_log', 1e-4, 1e2)},
}

# Optimize for RMSE (auto-infers direction=minimize)
"finetune_params": {
    "metric": "rmse",
    "model_params": {"n_components": ('int', 1, 20)},
}
```

Supported metrics and auto-inferred directions:

| Metric | Direction | Task |
|--------|-----------|------|
| `mse`, `rmse`, `mae` | minimize | Regression |
| `r2` | maximize | Regression |
| `accuracy`, `balanced_accuracy`, `f1` | maximize | Classification |

### Tuning Approaches

#### Single (Global Search)
Runs one optimization across all data:
```python
"approach": "single"
```
- **Fastest**: One search for the entire pipeline
- **Use when**: You want one optimal configuration

#### Grouped (Per-Preprocessing)
Optimizes separately for each preprocessing variant:
```python
"approach": "grouped",
"eval_mode": "best"  # or "mean" or "robust_best"
```
- **Balanced**: Each preprocessing gets its own optimal hyperparameters
- **Use when**: Using `feature_augmentation` to compare preprocessings

#### Individual (Per-Fold)
Optimizes separately for each cross-validation fold:
```python
"approach": "individual"
```
- **Most thorough**: Different hyperparameters per fold
- **Use when**: Maximum customization, have computational budget

## Advanced Features

### Multi-Phase Optimization

Run different samplers sequentially on a shared study. Phase 1 explores broadly, Phase 2 exploits promising regions:

```python
"finetune_params": {
    "seed": 42,
    "metric": "rmse",
    "phases": [
        {"n_trials": 50, "sampler": "random"},   # Broad exploration
        {"n_trials": 100, "sampler": "tpe"},      # Focused exploitation
    ],
    "model_params": {
        "alpha": ('float_log', 1e-4, 1e2),
        "l1_ratio": ('float', 0.0, 1.0),
    },
}
```

### Force-Params (Seeding Known Configurations)

Enqueue a known good configuration as trial 0 so the optimizer always evaluates your baseline first:

```python
"finetune_params": {
    "n_trials": 50,
    "sampler": "tpe",
    "force_params": {"n_components": 5},
    "model_params": {
        "n_components": ('int', 1, 20),
    },
}
```

### Training Parameter Tuning

For neural networks and other models with training parameters, use `train_params` to tune training configuration alongside model hyperparameters:

```python
"finetune_params": {
    "n_trials": 30,
    "sampler": "tpe",
    "model_params": {
        "filters_1": [8, 16, 32],
        "dropout_rate": ('float', 0.1, 0.5),
    },
    "train_params": {
        "epochs": ('int', 10, 100),      # Sampled by Optuna
        "batch_size": [16, 32, 64],       # Sampled (categorical)
        "verbose": 0,                     # Static value (not sampled)
    },
}
```

### Reproducibility

Use `seed` for deterministic optimization results:

```python
"finetune_params": {
    "n_trials": 50,
    "sampler": "tpe",
    "seed": 42,              # Same seed + data = same results
    "model_params": {...},
}
```

### Stack Helper (Stacking Model Optimization)

For sklearn `StackingRegressor` and `StackingClassifier`, use the `stack_params()` helper to easily finetune the final estimator (metamodel):

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from nirs4all.optimization.optuna import stack_params

# Define base estimators
base_estimators = [
    ("pls", PLSRegression(n_components=5)),
    ("ridge", Ridge(alpha=1.0)),
]

pipeline = [
    ShuffleSplit(n_splits=3, random_state=42),
    {
        "model": StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(),
            cv=3,
        ),
        "finetune_params": {
            "n_trials": 20,
            "sampler": "tpe",
            "model_params": stack_params(
                final_estimator_params={
                    "alpha": ('float_log', 1e-3, 1e2),      # Finetune metamodel alpha
                    "fit_intercept": [True, False],          # Finetune metamodel fit_intercept
                },
                passthrough=True,  # Stack-level parameter (optional)
            ),
        }
    }
]

result = nirs4all.run(pipeline=pipeline, dataset="data/")
```

The `stack_params()` helper automatically namespaces final estimator parameters with the `final_estimator__` prefix required by sklearn. This works seamlessly with the existing nested parameter system.

## Combining with Preprocessing Search

Combine `feature_augmentation` with hyperparameter tuning to find the best preprocessing + hyperparameter combination:

```python
from nirs4all.operators.transforms import StandardNormalVariate, Detrend, FirstDerivative

pipeline = [
    # Generate preprocessing variants
    {"feature_augmentation": [StandardNormalVariate, Detrend, FirstDerivative], "action": "extend"},

    ShuffleSplit(n_splits=3, random_state=42),

    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 10,
            "sampler": "tpe",
            "approach": "grouped",  # Optimize per preprocessing
            "model_params": {
                "n_components": ('int', 1, 20),
            }
        }
    }
]

result = nirs4all.run(pipeline=pipeline, dataset="data/")
```

## Pipeline Generators vs Optuna Tuning

nirs4all offers two approaches to hyperparameter exploration:

### 1. Pipeline Generators (`_range_`, `_or_`)
Create multiple complete pipelines upfront:
```python
pipeline = [
    {"model": PLSRegression(), "_range_": [1, 10], "param": "n_components"}
]
```
- **Pros**: Full control, parallel execution, complete reproducibility
- **Cons**: Combinatorial explosion with many parameters

### 2. Optuna Tuning (`finetune_params`)
Intelligent search within a single pipeline:
```python
pipeline = [
    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 50,
            "sampler": "tpe",
            "model_params": {"n_components": ('int', 1, 20)}
        }
    }
]
```
- **Pros**: Efficient for large spaces, early stopping, Bayesian optimization
- **Cons**: Sequential trials, less transparent

### When to Use Each

| Scenario | Recommended |
|----------|-------------|
| Few parameters (< 10 combinations) | `_range_` / `_or_` |
| Many parameters (> 50 combinations) | `finetune_params` |
| Neural network architecture + training | `finetune_params` with `train_params` |
| Reproducible benchmark | `_range_` / `_or_` |
| Quick exploration | `finetune_params` with `random` |

## Viewing Optimization Results

Access optimization details from the predictions:

```python
result = nirs4all.run(pipeline, dataset)

# Best configuration
print(f"Best score: {result.best_score:.4f}")

# Top configurations
for pred in result.top(5, display_metrics=['rmse', 'r2']):
    print(f"  {pred.get('model_name')}: RMSE={pred.get('rmse'):.4f}")

# Visualize results
from nirs4all.visualization.predictions import PredictionAnalyzer

analyzer = PredictionAnalyzer(result.predictions)
analyzer.plot_top_k(k=10, rank_metric='rmse')
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings", rank_metric='rmse')
```

## Best Practices

1. **Start small**: Begin with `n_trials=10` and `sampler="random"` to establish a baseline
2. **Use TPE for refinement**: Switch to `sampler="tpe"` with more trials for fine-tuning
3. **Set reasonable bounds**: Keep parameter ranges realistic to avoid wasted trials
4. **Use log-scale for regularization**: Parameters like `alpha`, `learning_rate` benefit from `('float_log', ...)`
5. **Monitor progress**: Use `verbose=2` to see trial-by-trial progress
6. **Use cross-validation**: Ensure robust evaluation with multiple folds
7. **Seed for reproducibility**: Always set `seed` when comparing configurations
8. **Multi-phase for large spaces**: Use random exploration followed by TPE exploitation
9. **Custom metrics**: Use `metric` to align optimization with your evaluation criteria

## See Also

- {doc}`training` - Basic model training
- {doc}`/reference/generator_keywords` - Pipeline generators (`_range_`, `_or_`)
- {doc}`/user_guide/pipelines/writing_pipelines` - Pipeline syntax

```{seealso}
**Related Examples:**
- [U02: Hyperparameter Tuning](../../../examples/user/04_models/U02_hyperparameter_tuning.py) - Grid, random, Bayesian search
- [U05: Advanced Finetuning](../../../examples/user/04_models/U05_advanced_finetuning.py) - Multi-phase, custom metrics, force-params
- [D01: Generator Syntax](../../../examples/developer/02_generators/D01_generator_syntax.py) - Generator syntax for parameter sweeps
```
