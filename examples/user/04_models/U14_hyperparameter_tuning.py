"""
U14 - Hyperparameter Tuning: Automated Optimization
====================================================

Automatically tune model hyperparameters using Optuna.

This tutorial covers:

* finetune_params for automated tuning
* Search methods: grid, random, Bayesian, Hyperband
* Tuning approaches: single, grouped, individual
* Combining tuning with preprocessing search

Prerequisites
-------------
Complete :ref:`U13_multi_model` first.

Next Steps
----------
See :ref:`U15_stacking_ensembles` for model ensembles.

Duration: ~6 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
parser = argparse.ArgumentParser(description='U14 Hyperparameter Tuning Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Introduction to Hyperparameter Tuning
# =============================================================================
print("\n" + "=" * 60)
print("U14 - Hyperparameter Tuning")
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

  📉 PARAMETER TYPES
     ('int', min, max)    - Integer range
     ('float', min, max)  - Float range
     [val1, val2, ...]    - Categorical choices
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
            "n_trials": 10,              # Number of trials
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

print(f"\nBest RMSE: {result_grid.best_rmse:.4f}")
print("Top configurations:")
for pred in result_grid.top(3):
    print(f"   {pred.get('model_name', 'Unknown')}: RMSE={pred['rmse']:.4f}")


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
        "model": RandomForestRegressor(random_state=42),
        "name": "RF-TPE",
        "finetune_params": {
            "n_trials": 15,              # Number of trials
            "sample": "tpe",             # Bayesian optimization
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_estimators": [50, 100, 200],       # Categorical
                "max_depth": ('int', 3, 15),          # Integer range
                "min_samples_split": ('int', 2, 10),  # Integer range
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

print(f"\nBest RMSE: {result_tpe.best_rmse:.4f}")


# =============================================================================
# Section 4: Hyperband for Deep Learning
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Hyperband Optimization")
print("-" * 60)

print("""
Hyperband uses early stopping to quickly discard bad configs.
Especially useful for neural networks with many epochs.
""")

pipeline_hyperband = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": GradientBoostingRegressor(random_state=42),
        "name": "GB-Hyperband",
        "finetune_params": {
            "n_trials": 20,
            "sample": "hyperband",       # Hyperband with early stopping
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_estimators": ('int', 20, 200),
                "max_depth": ('int', 2, 10),
                "learning_rate": ('float', 0.01, 0.3),
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

print(f"\nBest RMSE: {result_hyperband.best_rmse:.4f}")


# =============================================================================
# Section 5: Tuning Approaches
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Tuning Approaches")
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
            "n_trials": 5,
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

print(f"\nGrouped search - Best RMSE: {result_grouped.best_rmse:.4f}")
print("Each preprocessing gets its own optimal n_components!")


# =============================================================================
# Section 6: Combined Preprocessing + Model Tuning
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Combined Preprocessing + Model Tuning")
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
        FirstDerivative,
        SavitzkyGolay,
    ], "action": "extend"},

    ShuffleSplit(n_splits=3, random_state=42),

    # Tune PLS
    {
        "model": PLSRegression(),
        "name": "PLS-Combined",
        "finetune_params": {
            "n_trials": 10,
            "sample": "tpe",
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 15),
            },
        }
    },

    # Tune RandomForest
    {
        "model": RandomForestRegressor(random_state=42),
        "name": "RF-Combined",
        "finetune_params": {
            "n_trials": 10,
            "sample": "tpe",
            "verbose": 1,
            "approach": "single",
            "model_params": {
                "n_estimators": [50, 100, 150],
                "max_depth": ('int', 3, 12),
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
print(f"Best RMSE: {result_combined.best_rmse:.4f}")

# Show top results
print("\nTop 5 configurations:")
for i, pred in enumerate(result_combined.top(5), 1):
    preproc = pred.get('preprocessings', 'N/A')
    model = pred.get('model_name', 'Unknown')
    print(f"   {i}. {preproc} + {model}: RMSE={pred['rmse']:.4f}")


# =============================================================================
# Section 7: Visualizing Tuning Results
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Visualizing Tuning Results")
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
      "n_trials": 20,                    # Number of trials
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

Parameter Types:
  ('int', min, max)     - Integer range
  ('float', min, max)   - Float range
  [val1, val2, val3]    - Categorical choices

Next: U15_stacking_ensembles.py - Model stacking and ensembles
""")
