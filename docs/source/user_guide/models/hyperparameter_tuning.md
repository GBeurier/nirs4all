# Hyperparameter Tuning

Automatically optimize model hyperparameters using Optuna integration.

## Overview

nirs4all provides built-in hyperparameter optimization through **Optuna**, a state-of-the-art hyperparameter optimization framework. This allows you to automatically find the best model configuration without manual trial-and-error.

Key features:
- **Multiple search methods**: Grid search, random search, Bayesian optimization (TPE), CMA-ES, Hyperband
- **Flexible parameter types**: Integer ranges, float ranges, categorical choices
- **Tuning approaches**: Global search, per-preprocessing group, per-fold optimization
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
            "sample": "tpe",
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
        "sample": "tpe",             # Search method
        "verbose": 1,                # Logging verbosity (0, 1, or 2)
        "approach": "single",        # Tuning approach
        "eval_mode": "best",         # For grouped: "best" or "avg"
        "model_params": {            # Parameters to optimize
            "param1": ('int', 1, 10),
            "param2": ('float', 0.01, 1.0),
            "param3": [True, False],
        }
    }
}
```

### Parameter Type Specifications

#### Integer Range
```python
"n_components": ('int', 1, 20)  # Integer from 1 to 20
```

#### Float Range
```python
"learning_rate": ('float', 0.001, 0.1)  # Float from 0.001 to 0.1
```

#### Categorical Choices
```python
"kernel": ['linear', 'rbf', 'poly']  # One of these values
"scale": [True, False]               # Boolean choice
```

### Search Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `grid` | Exhaustive search over all combinations | Small parameter spaces |
| `random` | Random sampling | Quick baseline, large spaces |
| `tpe` | Tree-Parzen Estimator (Bayesian) | Medium to large spaces |
| `cmaes` | CMA Evolution Strategy | Continuous parameters |
| `hyperband` | Successive halving with early stopping | Neural networks |

#### Grid Search
```python
"finetune_params": {
    "n_trials": 10,
    "sample": "grid",
    "model_params": {
        "n_components": [5, 10, 15, 20],  # Categorical for grid
    }
}
```

:::{note}
Grid search requires categorical parameters (lists). Range specifications are
automatically treated as continuous by TPE/random samplers.
:::

#### Bayesian Optimization (TPE)
```python
"finetune_params": {
    "n_trials": 50,
    "sample": "tpe",
    "model_params": {
        "n_components": ('int', 1, 30),
        "max_depth": ('int', 3, 15),
        "learning_rate": ('float', 0.01, 0.3),
    }
}
```

#### Hyperband (Early Stopping)
```python
"finetune_params": {
    "n_trials": 100,
    "sample": "hyperband",
    "model_params": {
        "n_components": ('int', 1, 50),
    }
}
```

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
"eval_mode": "best"  # or "avg"
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

## Combining with Preprocessing Search

Combine `feature_augmentation` with hyperparameter tuning to find the best preprocessing + hyperparameter combination:

```python
from nirs4all.operators.transforms import SNV, Detrend, FirstDerivative

pipeline = [
    # Generate preprocessing variants
    {"feature_augmentation": [SNV, Detrend, FirstDerivative], "action": "extend"},

    ShuffleSplit(n_splits=3, random_state=42),

    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 10,
            "sample": "tpe",
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
    # Generates 10 separate pipelines
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
            "sample": "tpe",
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
| Neural network epochs | `finetune_params` with `hyperband` |
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

## Examples

### Random Forest with Multiple Parameters

```python
from sklearn.ensemble import RandomForestRegressor

pipeline = [
    SNV(),
    ShuffleSplit(n_splits=3, random_state=42),
    {
        "model": RandomForestRegressor(n_jobs=-1, random_state=42),
        "finetune_params": {
            "n_trials": 30,
            "sample": "tpe",
            "model_params": {
                "n_estimators": [50, 100, 200],
                "max_depth": ('int', 3, 15),
                "min_samples_split": ('int', 2, 10),
            }
        }
    }
]
```

### Neural Network with Hyperband

```python
from nirs4all.operators.models.tensorflow.nicon import nicon

pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=2, random_state=42),
    {
        "model": nicon,
        "finetune_params": {
            "n_trials": 50,
            "sample": "hyperband",
            "model_params": {
                "filters1": [8, 16, 32],
                "dropout_rate": ('float', 0.1, 0.5),
            }
        },
        "train_params": {
            "epochs": 100,
            "verbose": 0
        }
    }
]
```

## Best Practices

1. **Start small**: Begin with `n_trials=10` and `sample="random"` to establish a baseline
2. **Use TPE for refinement**: Switch to `sample="tpe"` with more trials for fine-tuning
3. **Set reasonable bounds**: Keep parameter ranges realistic to avoid wasted trials
4. **Monitor progress**: Use `verbose=2` to see trial-by-trial progress
5. **Use cross-validation**: Ensure robust evaluation with multiple folds
6. **Consider computational cost**: `hyperband` is efficient for expensive models

## See Also

- {doc}`training` - Basic model training
- {doc}`/reference/generator_keywords` - Pipeline generators (`_range_`, `_or_`)
- {doc}`/user_guide/pipelines/writing_pipelines` - Pipeline syntax
- [U14_hyperparameter_tuning.py](https://github.com/GBeurier/nirs4all/blob/main/examples/user/04_models/U14_hyperparameter_tuning.py) - Complete example
