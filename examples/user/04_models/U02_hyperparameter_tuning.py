"""
U02 - Hyperparameter Tuning: Automated Optimization
====================================================

Automatically tune model hyperparameters using Optuna.

This tutorial covers:

* finetune_params for automated tuning
* Search methods: grid, random, tpe, auto
* Tuning approaches: single, grouped, individual
* Combining tuning with preprocessing search

Prerequisites
-------------
Complete :ref:`U01_multi_model` first.

Next Steps
----------
See :ref:`U03_stacking_ensembles` for model ensembles.

Duration: ~1 minute
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    SavitzkyGolay,
    StandardNormalVariate,
)
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U02 Hyperparameter Tuning Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Introduction to Hyperparameter Tuning
# =============================================================================
print("\n" + "=" * 60)
print("U02 - Hyperparameter Tuning")
print("=" * 60)

print("""
Hyperparameter tuning finds optimal model settings automatically.

nirs4all uses Optuna for efficient hyperparameter optimization:

  📊 SEARCH METHODS (sampler)
     grid      - Exhaustive grid search
     random    - Random sampling
     tpe       - Tree-Parzen Estimator (Bayesian)
     cmaes     - CMA-ES (good for continuous params)
     binary    - Binary search (for unimodal integers like PLS n_components)
     auto      - Automatic selection based on search space

  📈 TUNING APPROACHES
     single    - One global search across all variants
     grouped   - Search per preprocessing group
     individual - Search per fold (most thorough)

  📉 PARAMETER TYPES (Tuple Format)
     ('int', min, max)        - Integer uniform sampling
     ('int_log', min, max)    - Integer log-uniform sampling
     ('float', min, max)      - Float uniform sampling
     ('float_log', min, max)  - Float log-uniform (for learning rates, regularization)
     [val1, val2, ...]        - Categorical choices

  📋 PARAMETER TYPES (Dict Format - most flexible)
     {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
     {'type': 'float', 'min': 0.0, 'max': 1.0, 'log': True}
     {'type': 'categorical', 'choices': [v1, v2, v3]}
""")

# =============================================================================
# Section 2: Basic Grid Search
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Basic Grid Search")
print("-" * 60)

print("""
Grid search exhaustively tests all combinations.
Good for small parameter spaces.
""")

pipeline_grid = [
    StandardNormalVariate(),
    MinMaxScaler(),

    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-GridSearch",
        "finetune_params": {
            "n_trials": 2,              # Number of trials
            "sampler": "grid",            # Grid search
            "verbose": 1,                # 0=silent, 1=basic, 2=detailed
            "approach": "single",        # Global search
            "model_params": {
                "n_components": ('int', 1, 10),  # Search 1-10 components
            },
        }
    },
]

result_grid = nirs4all.run(
    pipeline=pipeline_grid,
    dataset="sample_data/regression",
    name="GridSearch",
    verbose=1
)

print(f"\nBest RMSE: {result_grid.best_score:.4f}")
print("Top configurations:")
for pred in result_grid.top(3, display_metrics=['rmse', 'r2']):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 3: Bayesian Optimization (TPE)
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Bayesian Optimization (TPE)")
print("-" * 60)

print("""
TPE (Tree-Parzen Estimator) learns from previous trials.
More efficient than grid search for larger spaces.
""")

pipeline_tpe = [
    StandardNormalVariate(),
    FirstDerivative(),

    ShuffleSplit(n_splits=3, random_state=42),

    {
        "model": RandomForestRegressor(n_jobs=-1, random_state=42),
        "name": "RF-TPE",
        "finetune_params": {
            "n_trials": 2,              # Number of trials
            "sampler": "tpe",             # Bayesian optimization
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_estimators": [2, 5],           # Categorical (reduced)
                "max_depth": ('int', 3, 6),          # Integer range
            },
        }
    },
]

result_tpe = nirs4all.run(
    pipeline=pipeline_tpe,
    dataset="sample_data/regression",
    name="TPE",
    verbose=1
)

print(f"\nBest RMSE: {result_tpe.best_score:.4f}")

# =============================================================================
# Section 4: Log-Scale Parameters
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Log-Scale Parameters")
print("-" * 60)

print("""
Use 'float_log' or 'int_log' for parameters that span multiple orders
of magnitude (learning rates, regularization, etc.).

Log-uniform sampling ensures each order of magnitude gets equal
exploration probability:
  ('float_log', 1e-5, 1e-1) → samples 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 equally
""")

from sklearn.linear_model import Ridge

pipeline_log = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=3, random_state=42),

    {
        "model": Ridge(),
        "name": "Ridge-LogScale",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "verbose": 1,
            "approach": "single",
            "model_params": {
                # Log-uniform: good for regularization spanning 1e-4 to 1e+2
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }
    },
]

result_log = nirs4all.run(
    pipeline=pipeline_log,
    dataset="sample_data/regression",
    name="LogScale",
    verbose=1
)

print(f"\nBest RMSE: {result_log.best_score:.4f}")

# =============================================================================
# Section 5: Pruning and Seed Support
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Pruning and Reproducibility")
print("-" * 60)

print("""
Pruners terminate unpromising trials early, saving time.
Seed ensures reproducible optimization results.

  PRUNERS
     none               - No pruning (default)
     median             - Prune if worse than median of completed trials
     successive_halving - Prune poorest fraction at each step
     hyperband          - Adaptive resource allocation with early stopping

  SEED
     seed: 42           - Same seed + data → same results
""")

pipeline_pruning = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-Pruned",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "pruner": "median",     # Prune unpromising trials
            "seed": 42,             # Reproducible results
            "verbose": 1,
            "approach": "grouped",
            "model_params": {
                "n_components": ('int', 1, 15),
            },
        }
    },
]

result_pruning = nirs4all.run(
    pipeline=pipeline_pruning,
    dataset="sample_data/regression",
    name="Pruning",
    verbose=1
)

print(f"\nBest RMSE: {result_pruning.best_score:.4f}")

# =============================================================================
# Section 5b: Custom Metric and Direction
# =============================================================================
print("\n" + "-" * 60)
print("Section 5b: Custom Metric and Direction")
print("-" * 60)

print("""
By default, optimization minimizes MSE (regression) or maximizes balanced
accuracy (classification). Use 'metric' and 'direction' to customize:

  SUPPORTED METRICS
     mse, rmse, mae          - minimize (regression)
     r2                      - maximize (regression)
     accuracy, balanced_accuracy, f1 - maximize (classification)

  Direction is auto-inferred from the metric name, but can be overridden.
""")

pipeline_metric = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": Ridge(),
        "name": "Ridge-R2Metric",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "metric": "r2",              # Optimize for R2 instead of MSE
            # direction auto-inferred as "maximize" for r2
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }
    },
]

result_metric = nirs4all.run(
    pipeline=pipeline_metric,
    dataset="sample_data/regression",
    name="CustomMetric",
    verbose=1
)

print(f"\nBest R2-optimized score: {result_metric.best_score:.4f}")

# =============================================================================
# Section 6: Tuning Approaches
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Tuning Approaches")
print("-" * 60)

print("""
Different approaches for multi-preprocessing scenarios:

  single     - One global search (fastest)
  grouped    - Search per preprocessing (balanced)
  individual - Search per fold (most thorough)
""")

# Generate preprocessing variants
pipeline_grouped = [
    {"feature_augmentation": [
        StandardNormalVariate,
        Detrend,
    ], "action": "extend"},

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-Grouped",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "grid",
            "verbose": 1,
            "approach": "grouped",       # Search per preprocessing
            "eval_mode": "best",         # Use best trial per group
            "model_params": {
                "n_components": ('int', 2, 8),
            },
        }
    },
]

result_grouped = nirs4all.run(
    pipeline=pipeline_grouped,
    dataset="sample_data/regression",
    name="Grouped",
    verbose=1
)

print(f"\nGrouped search - Best RMSE: {result_grouped.best_score:.4f}")
print("Each preprocessing gets its own optimal n_components!")

# =============================================================================
# Section 7: Combined Preprocessing + Model Tuning
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Combined Preprocessing + Model Tuning")
print("-" * 60)

print("""
Combine feature_augmentation with hyperparameter tuning
to find the best preprocessing + hyperparameters together.
""")

pipeline_combined = [
    MinMaxScaler(),

    # Explore preprocessing
    {"feature_augmentation": [
        StandardNormalVariate,
        Detrend,
    ], "action": "extend"},

    ShuffleSplit(n_splits=2, random_state=42),

    # Tune PLS only (faster for demonstration)
    {
        "model": PLSRegression(),
        "name": "PLS-Combined",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 10),
            },
        }
    },
]

result_combined = nirs4all.run(
    pipeline=pipeline_combined,
    dataset="sample_data/regression",
    name="Combined",
    verbose=1
)

print(f"\nTotal configurations: {result_combined.num_predictions}")
print(f"Best RMSE: {result_combined.best_score:.4f}")

# Show top results
print("\nTop 5 configurations:")
for i, pred in enumerate(result_combined.top(5, display_metrics=['rmse', 'r2']), 1):
    preproc = pred.get('preprocessings', 'N/A')
    model = pred.get('model_name', 'Unknown')
    print(f"   {i}. {preproc} + {model}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Section 8: Visualizing Tuning Results
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Visualizing Tuning Results")
print("-" * 60)

if args.plots:
    analyzer = PredictionAnalyzer(result_combined.predictions)

    # Top-k comparison
    fig1 = analyzer.plot_top_k(k=10, rank_metric='rmse')

    # Heatmap: model vs preprocessing
    fig2 = analyzer.plot_heatmap(
        x_var="model_name",
        y_var="preprocessings",
        rank_metric='rmse'
    )

    # Candlestick: performance distribution
    fig3 = analyzer.plot_candlestick(
        variable="model_name",
        display_partition="test"
    )

    print("Charts generated (use --show to display)")

    if args.show:
        plt.show()
else:
    print("Use --plots to generate visualization charts")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Hyperparameter Tuning Configuration:

  {
    "model": PLSRegression(),
    "finetune_params": {
      "n_trials": 2,                    # Number of trials
      "sampler": "tpe",                   # Search method
      "verbose": 1,                      # 0=silent, 1=basic, 2=detailed
      "approach": "single",              # Tuning approach
      "eval_mode": "best",               # For grouped: "best" or "mean"
      "seed": 42,                        # Reproducible optimization
      "pruner": "median",               # Prune unpromising trials
      "metric": "rmse",                  # Metric to optimize (auto-infers direction)
      "model_params": {
        "n_components": ('int', 1, 20),  # Integer range
        "scale": [True, False],          # Categorical
        "tol": ('float', 1e-6, 1e-4),    # Float range
      },
      "train_params": {                  # Training params (also sampable)
        "epochs": ('int', 50, 300),      # Sampled by Optuna
        "verbose": 0,                    # Static value (not sampled)
      },
    }
  }

Search Methods:
  grid      - Exhaustive (small spaces)
  random    - Random sampling (quick baseline)
  tpe       - Bayesian (efficient for medium spaces)
  cmaes     - CMA-ES (continuous parameter spaces)
  binary    - Binary search (unimodal integers like PLS n_components, 2-3x faster)
  auto      - Automatic selection based on search space

Tuning Approaches:
  single     - Global search (fastest)
  grouped    - Per preprocessing group (balanced)
  individual - Per fold (most thorough)

Parameter Types (Tuple Format):
  ('int', min, max)        - Integer uniform sampling
  ('int_log', min, max)    - Integer log-uniform sampling
  ('float', min, max)      - Float uniform sampling
  ('float_log', min, max)  - Float log-uniform (for learning rates, regularization)
  [val1, val2, val3]       - Categorical choices

Parameter Types (Dict Format - most flexible):
  {'type': 'int', 'min': 1, 'max': 10}              - Basic integer
  {'type': 'int', 'min': 1, 'max': 10, 'step': 2}   - Stepped integer
  {'type': 'int', 'min': 1, 'max': 1000, 'log': True}  - Log-scale integer
  {'type': 'float', 'min': 0.0, 'max': 1.0}         - Basic float
  {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True}  - Log-scale float
  {'type': 'categorical', 'choices': [v1, v2, v3]}  - Categorical

Examples:
  # Learning rate: log-uniform from 1e-5 to 1e-1
  'lr': ('float_log', 1e-5, 1e-1)

  # Regularization: log-uniform from 1e-6 to 1.0
  'alpha': ('float_log', 1e-6, 1.0)

  # Number of layers: integer from 1 to 5
  'n_layers': ('int', 1, 5)

  # Batch size: log-scale powers of 2 feel
  'batch_size': {'type': 'int', 'min': 16, 'max': 256, 'log': True}

Next: U03_stacking_ensembles.py - Model stacking and ensembles
""")

if __name__ == "__main__":
    pass
