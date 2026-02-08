"""
U02 - Hyperparameter Tuning: Automated Optimization
====================================================

Automatically tune model hyperparameters using Optuna.

This tutorial covers:

* finetune_params for automated tuning
* Search methods: grid, random, Bayesian, Hyperband
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
    StandardNormalVariate,
    FirstDerivative,
    Detrend,
    SavitzkyGolay,
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

  📊 SEARCH METHODS
     grid      - Exhaustive grid search
     random    - Random sampling
     tpe       - Tree-Parzen Estimator (Bayesian)
     cmaes     - CMA Evolution Strategy
     hyperband - Early stopping + successive halving

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
            "sample": "grid",            # Grid search
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
            "sample": "tpe",             # Bayesian optimization
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
            "sample": "tpe",
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
# Section 5: Hyperband for Deep Learning
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Hyperband Optimization")
print("-" * 60)

print("""
Hyperband uses early stopping to quickly discard bad configs.
Especially useful for neural networks with many epochs.
""")

pipeline_hyperband = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-Hyperband",
        "finetune_params": {
            "n_trials": 2,
            "sample": "hyperband",       # Hyperband with early stopping
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 15),
            },
        }
    },
]

result_hyperband = nirs4all.run(
    pipeline=pipeline_hyperband,
    dataset="sample_data/regression",
    name="Hyperband",
    verbose=1
)

print(f"\nBest RMSE: {result_hyperband.best_score:.4f}")


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
            "sample": "grid",
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
            "sample": "tpe",
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
        partition="test"
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
      "sample": "tpe",                   # Search method
      "verbose": 1,                      # 0=silent, 1=basic, 2=detailed
      "approach": "single",              # Tuning approach
      "eval_mode": "best",               # For grouped: "best" or "avg"
      "model_params": {
        "n_components": ('int', 1, 20),  # Integer range
        "scale": [True, False],          # Categorical
        "tol": ('float', 1e-6, 1e-4),    # Float range
      },
    }
  }

Search Methods:
  grid      - Exhaustive (small spaces)
  random    - Random sampling (quick baseline)
  tpe       - Bayesian (efficient for medium spaces)
  cmaes     - Evolution strategy (continuous params)
  hyperband - Early stopping (neural networks)

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
