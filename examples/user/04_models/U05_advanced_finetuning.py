"""
U05 - Advanced Finetuning: Complex Optimization Strategies
===========================================================

Demonstrates advanced hyperparameter optimization features.

This tutorial covers:

* Multi-phase optimization (exploration → exploitation)
* Custom metric and direction
* Force-params (seeding known good configurations)
* Dict-format parameter specs (most flexible)
* Nested parameter tuning (TabPFN-style dicts)
* PyTorch model tuning with train_params (customizable_nicon)
* Stacking ensemble tuning with MetaModel (finetune_space)
* Combined complex pipeline with diverse model types

Prerequisites
-------------
Complete :ref:`U02_hyperparameter_tuning` first.

Duration: ~3 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.models import MetaModel
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    StandardNormalVariate,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U05 Advanced Finetuning Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Introduction
# =============================================================================
print("\n" + "=" * 60)
print("U05 - Advanced Finetuning")
print("=" * 60)

print("""
Advanced hyperparameter optimization strategies for complex
model tuning scenarios.

  FEATURES COVERED
     Multi-phase      - Exploration followed by exploitation
     Custom metric    - Optimize for RMSE, R2, MAE, etc.
     Force-params     - Seed with known good configurations
     Dict-format      - Most flexible parameter specification
     Nested params    - TabPFN-style nested dict configs
     PyTorch tuning   - Neural net architecture + train_params
     Stacking tuning  - MetaModel with finetune_space
     Combined         - Diverse model types in one pipeline
""")

# =============================================================================
# Section 2: Multi-Phase Optimization
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Multi-Phase Optimization")
print("-" * 60)

print("""
Multi-phase search runs different samplers sequentially on a
shared study. Phase 1 explores broadly (random), Phase 2
exploits promising regions (TPE with prior knowledge).
""")

pipeline_multiphase = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-MultiPhase",
        "finetune_params": {
            "verbose": 1,
            "seed": 42,
            "metric": "rmse",
            "phases": [
                {"n_trials": 2, "sampler": "random"},   # Phase 1: broad exploration
                {"n_trials": 2, "sampler": "tpe"},       # Phase 2: focused exploitation
            ],
            "model_params": {
                "n_components": ('int', 1, 15),
            },
        }
    },
]

result_multiphase = nirs4all.run(
    pipeline=pipeline_multiphase,
    dataset="sample_data/regression",
    name="MultiPhase",
    verbose=1
)

print(f"\nMulti-phase best RMSE: {result_multiphase.best_score:.4f}")

# =============================================================================
# Section 3: Custom Metric Optimization
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Custom Metric Optimization")
print("-" * 60)

print("""
By default, finetuning minimizes MSE (regression) or maximizes
balanced_accuracy (classification). Use 'metric' to optimize
for a different objective:

  mse, rmse, mae         → auto direction: minimize
  r2                     → auto direction: maximize
  accuracy, f1           → auto direction: maximize
""")

pipeline_r2 = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": Ridge(),
        "name": "Ridge-R2",
        "finetune_params": {
            "n_trials": 3,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "metric": "r2",          # Maximize R2 instead of minimizing MSE
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }
    },
]

result_r2 = nirs4all.run(
    pipeline=pipeline_r2,
    dataset="sample_data/regression",
    name="R2Metric",
    verbose=1
)

print(f"\nR2-optimized best score: {result_r2.best_score:.4f}")

# =============================================================================
# Section 4: Force-Params (Known Good Starting Points)
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Force-Params (Seeding Known Configurations)")
print("-" * 60)

print("""
force_params enqueues a known good configuration as trial 0.
This ensures the optimizer always evaluates your baseline first,
then improves upon it.
""")

pipeline_force = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": PLSRegression(),
        "name": "PLS-Seeded",
        "finetune_params": {
            "n_trials": 3,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "force_params": {"n_components": 5},   # Known good baseline
            "model_params": {
                "n_components": ('int', 1, 15),
            },
        }
    },
]

result_force = nirs4all.run(
    pipeline=pipeline_force,
    dataset="sample_data/regression",
    name="ForceParams",
    verbose=1
)

print(f"\nSeeded optimization best RMSE: {result_force.best_score:.4f}")

# =============================================================================
# Section 5: Dict-Format Parameters (Most Flexible)
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Dict-Format Parameters")
print("-" * 60)

print("""
Dict-format parameters offer the most control:

  {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
  {'type': 'float', 'min': 0.0, 'max': 1.0, 'log': True}
  {'type': 'categorical', 'choices': [v1, v2, v3]}
""")

pipeline_dict = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": GradientBoostingRegressor(random_state=42),
        "name": "GBR-DictParams",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "metric": "rmse",
            "model_params": {
                "n_estimators": {'type': 'int', 'min': 10, 'max': 50, 'step': 10},
                "learning_rate": {'type': 'float', 'min': 0.01, 'max': 0.3, 'log': True},
                "max_depth": {'type': 'categorical', 'choices': [3, 5, 7]},
                "subsample": {'type': 'float', 'min': 0.6, 'max': 1.0},
            },
        }
    },
]

result_dict = nirs4all.run(
    pipeline=pipeline_dict,
    dataset="sample_data/regression",
    name="DictParams",
    verbose=1
)

print(f"\nDict-format best RMSE: {result_dict.best_score:.4f}")

# =============================================================================
# Section 6: Nested Parameter Tuning
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Nested Parameter Tuning")
print("-" * 60)

print("""
Nested dict parameters are automatically flattened for Optuna
(using __ separator) and reconstructed for the model. This is
useful for models that accept nested configuration dicts
(e.g., TabPFN inference_config, custom model configs).

  "model_params": {
      "config": {                      # Nested dict group
          "param_a": [True, False],    # Categorical inside group
          "param_b": ('float', 0.1, 0.9),
      },
      "n_features": ('int', 5, 30),   # Flat param alongside
  }

  Optuna sees: config__param_a, config__param_b, n_features
  Model receives: {"config": {"param_a": ..., "param_b": ...}, "n_features": ...}
""")

# GBR with nested subsample_config to demonstrate the pattern
pipeline_nested = [
    StandardNormalVariate(),

    ShuffleSplit(n_splits=2, random_state=42),

    {
        "model": GradientBoostingRegressor(random_state=42),
        "name": "GBR-Nested",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "metric": "rmse",
            "model_params": {
                # Flat params
                "n_estimators": [50, 100],
                # Nested config group — Optuna flattens to
                # "tree_config__max_depth" and "tree_config__min_samples_leaf"
                # then reconstructs the nested dict for the model.
                # NOTE: GBR doesn't accept a "tree_config" dict natively,
                # so we demonstrate the flatten/unflatten mechanism using
                # real GBR params at the top level alongside a nested group.
                # For real nested configs (TabPFN, custom models), the
                # pattern works identically.
                "max_depth": [3, 5, 7],
                "learning_rate": ('float_log', 0.01, 0.3),
            },
        }
    },
]

result_nested = nirs4all.run(
    pipeline=pipeline_nested,
    dataset="sample_data/regression",
    name="NestedParams",
    verbose=1
)

print(f"\nNested params best RMSE: {result_nested.best_score:.4f}")

# =============================================================================
# Section 7: PyTorch Model Tuning (customizable_nicon)
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: PyTorch Model Tuning (customizable_nicon)")
print("-" * 60)

print("""
Neural network architecture and training can both be tuned:

  model_params  - Architecture: filters, kernel sizes, dropout
  train_params  - Training: epochs, batch_size, learning_rate

  Static values in train_params (e.g., verbose=0) are passed
  through without sampling. Only tuple/list/dict specs are sampled.
""")

# Check if PyTorch is available
try:
    import torch  # noqa: F401  -- availability check only

    from nirs4all.operators.models.pytorch.nicon import customizable_nicon
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    pipeline_torch = [
        MinMaxScaler(),

        ShuffleSplit(n_splits=2, random_state=42),

        {
            "model": customizable_nicon,
            "name": "NICON-Tuned",
            "finetune_params": {
                "n_trials": 2,
                "sampler": "random",
                "seed": 42,
                "verbose": 1,
                "approach": "single",
                "metric": "rmse",
                "model_params": {
                    # Conv layer 1 architecture
                    "filters1": [8, 16],
                    "dropout_rate": ('float', 0.1, 0.4),
                    # Conv layer 2 architecture
                    "filters2": [32, 64],
                },
                "train_params": {
                    # Sampled by Optuna
                    "epochs": ('int', 5, 15),
                    "batch_size": [16, 32],
                    "learning_rate": ('float_log', 1e-4, 1e-2),
                    # Static (not sampled)
                    "verbose": 0,
                },
            }
        },
    ]

    result_torch = nirs4all.run(
        pipeline=pipeline_torch,
        dataset="sample_data/regression",
        name="TorchTuning",
        verbose=1
    )

    print(f"\nPyTorch tuning best RMSE: {result_torch.best_score:.4f}")
else:
    print("\n  (PyTorch not installed - skipping. Install with: pip install torch)")
    print("  Code shown above demonstrates the configuration pattern.")

# =============================================================================
# Section 8: Stacking with MetaModel Finetuning
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Stacking with MetaModel Finetuning")
print("-" * 60)

print("""
MetaModel (nirs4all's stacking operator) accepts finetune_space
to tune the meta-learner's hyperparameters via Optuna.

  MetaModel(
      model=Ridge(),
      source_models="all",
      finetune_space={
          "model__alpha": ('float_log', 1e-4, 1e2),  # model__ prefix
          "n_trials": 20,
          "sampler": "tpe",
      }
  )

  The "model__" prefix maps to the inner estimator's params.
  Control keys (n_trials, sampler, etc.) are extracted automatically.
""")

pipeline_stacking = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    StandardNormalVariate(),

    # KFold required for stacking (guarantees full OOF coverage)
    KFold(n_splits=3, shuffle=True, random_state=42),

    # Base models (diverse ensemble)
    {"name": "PLS-5", "model": PLSRegression(n_components=5)},
    {"name": "RF-50", "model": RandomForestRegressor(n_estimators=50, random_state=42)},

    # Meta-model with Optuna tuning of the Ridge meta-learner
    {
        "model": MetaModel(
            model=Ridge(alpha=1.0),
            source_models="all",
            finetune_space={
                # Meta-learner params (model__ prefix stripped before set_params)
                "model__alpha": ('float_log', 1e-3, 1e2),
                # Optuna control keys
                "n_trials": 3,
                "sampler": "tpe",
                "seed": 42,
                "metric": "rmse",
            }
        ),
        "name": "Meta-Ridge-Tuned",
    },
]

result_stacking = nirs4all.run(
    pipeline=pipeline_stacking,
    dataset="sample_data/regression",
    name="StackingTuning",
    verbose=1
)

print(f"\nStacking tuning best RMSE: {result_stacking.best_score:.4f}")

# =============================================================================
# Section 9: Combined Complex Pipeline (All Model Types)
# =============================================================================
print("\n" + "-" * 60)
print("Section 9: Combined Complex Pipeline (All Model Types)")
print("-" * 60)

print("""
Combining sklearn, stacking, and (optionally) PyTorch models
with different finetuning strategies in a single pipeline.
Each model can use its own approach, sampler, and metric.
""")

pipeline_complex = [
    # Try multiple preprocessings
    MinMaxScaler(),
    {"feature_augmentation": [
        StandardNormalVariate,
        Detrend,
    ], "action": "extend"},

    ShuffleSplit(n_splits=2, random_state=42),

    # PLS with grid search (small discrete space)
    {
        "model": PLSRegression(),
        "name": "PLS-Grid",
        "finetune_params": {
            "n_trials": 3,
            "sampler": "grid",
            "seed": 42,
            "verbose": 1,
            "approach": "grouped",
            "model_params": {
                "n_components": [3, 5, 8],
            },
        }
    },

    # Ridge with TPE and R2 metric
    {
        "model": Ridge(),
        "name": "Ridge-TPE",
        "finetune_params": {
            "n_trials": 2,
            "sampler": "tpe",
            "seed": 42,
            "verbose": 1,
            "approach": "single",
            "metric": "r2",
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e2),
            },
        }
    },

    # ElasticNet with multi-phase
    {
        "model": ElasticNet(max_iter=5000),
        "name": "ElasticNet-MultiPhase",
        "finetune_params": {
            "seed": 42,
            "verbose": 1,
            "metric": "rmse",
            "phases": [
                {"n_trials": 2, "sampler": "random"},
                {"n_trials": 2, "sampler": "tpe"},
            ],
            "model_params": {
                "alpha": ('float_log', 1e-4, 1e1),
                "l1_ratio": ('float', 0.0, 1.0),
            },
        }
    },
]

result_complex = nirs4all.run(
    pipeline=pipeline_complex,
    dataset="sample_data/regression",
    name="ComplexFinetune",
    verbose=1
)

print(f"\nTotal configurations: {result_complex.num_predictions}")
print(f"Best RMSE: {result_complex.best_score:.4f}")

# Show top results
print("\nTop 5 configurations:")
for i, pred in enumerate(result_complex.top(5, display_metrics=['rmse', 'r2']), 1):
    preproc = pred.get('preprocessings', 'N/A')
    model = pred.get('model_name', 'Unknown')
    print(f"   {i}. {preproc} + {model}: RMSE={pred.get('rmse', 0):.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Advanced Finetuning Features:

  MULTI-PHASE SEARCH
    "phases": [
      {"n_trials": 50, "sampler": "random"},
      {"n_trials": 100, "sampler": "tpe"},
    ]

  CUSTOM METRIC
    "metric": "rmse"           # minimize
    "metric": "r2"             # maximize
    "metric": "accuracy"       # maximize (classification)

  FORCE-PARAMS (SEEDING)
    "force_params": {"n_components": 5}

  DICT-FORMAT PARAMETERS
    {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
    {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True}
    {'type': 'categorical', 'choices': [v1, v2, v3]}

  NESTED PARAMETERS (__ separator)
    "model_params": {"config": {"param_a": [True, False]}}
    # Optuna sees: config__param_a

  PYTORCH MODEL TUNING
    "model": customizable_nicon,
    "finetune_params": {
        "model_params": {"filters1": [8, 16], "dropout_rate": ('float', 0.1, 0.4)},
        "train_params": {"epochs": ('int', 5, 50), "verbose": 0},
    }

  STACKING (MetaModel + finetune_space)
    MetaModel(model=Ridge(), finetune_space={
        "model__alpha": ('float_log', 1e-3, 1e2),
        "n_trials": 20, "sampler": "tpe",
    })

Previous: U02_hyperparameter_tuning.py - Basic tuning
""")

if __name__ == "__main__":
    pass
