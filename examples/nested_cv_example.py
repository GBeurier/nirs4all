"""
Nested Cross-Validation Example for NIRS4ALL

This example demonstrates the three different cross-validation strategies for finetuning:

1. **Simple CV**: Finetune on full training data, then train on folds
2. **Per-fold CV**: Finetune on each fold individually
3. **Nested CV**: Inner folds for finetuning, outer folds for training (academic-level)

Each strategy provides different levels of rigor and computational cost.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from nirs4all.operators.models.cirad_tf import nicon
from nirs4all.operators.transformations import StandardNormalVariate as SNV

# Dataset configuration (same as before)
dataset_config = {
    "folder": "./sample_data"
}

# ==============================================================================
# 1. SIMPLE CV FINETUNING
# ==============================================================================
# Finetune on all training data to find best parameters, then train each fold model with those parameters.
# Fastest method, but least rigorous for parameter selection.

simple_cv_pipeline = {
    "pipeline": [
        {"feature_augmentation": [None, SNV]},
        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
            "train_params": {
                "verbose": 1  # Show training progress
            },
            "finetune_params": {
                "cv_mode": "simple",  # Simple CV finetuning strategy
                "n_trials": 20,
                "approach": "random",
                "verbose": 1,  # Show finetuning progress
                "model_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                },
                "train_params": {
                    "verbose": 0  # Silent training during finetuning trials
                }
            },
        }
    ]
}

# ==============================================================================
# 2. PER-FOLD CV FINETUNING
# ==============================================================================
# Finetune separately on each fold. Can use global best params or per-fold best params.
# More rigorous than simple CV, moderate computational cost.

per_fold_cv_pipeline = {
    "pipeline": [
        {"feature_augmentation": [None, SNV]},
        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
            "train_params": {
                "verbose": 1
            },
            "finetune_params": {
                "cv_mode": "per_fold",  # Per-fold finetuning strategy
                "param_strategy": "per_fold_best",  # Options: "global_best", "per_fold_best"
                "n_trials": 15,
                "approach": "random",
                "verbose": 1,
                "model_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                },
                "train_params": {
                    "verbose": 0
                }
            },
        }
    ]
}

# ==============================================================================
# 3. NESTED CV (ACADEMIC LEVEL)
# ==============================================================================
# Inner folds for parameter optimization, outer folds for performance estimation.
# Most rigorous method, highest computational cost. Best for research/publication.

nested_cv_pipeline = {
    "pipeline": [
        {"feature_augmentation": [None, SNV]},
        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
            "train_params": {
                "verbose": 1
            },
            "finetune_params": {
                "cv_mode": "nested",  # Nested CV strategy
                "inner_cv": 3,  # Number of inner folds for parameter optimization (can also be a CV object)
                "param_strategy": "per_fold_best",  # Options: "per_fold_best", "weighted_average"
                "n_trials": 10,  # Reduced trials per inner CV due to computational cost
                "approach": "random",
                "verbose": 2,  # Higher verbosity for nested CV details
                "model_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "train_params": {
                    "verbose": 0  # Silent training during inner CV trials
                }
            },
        }
    ]
}

# ==============================================================================
# 4. NESTED CV WITH TENSORFLOW/KERAS MODEL
# ==============================================================================
# Example with deep learning model using nested CV

nested_cv_tensorflow_pipeline = {
    "pipeline": [
        {"feature_augmentation": [None, SNV]},
        {
            "model": nicon,  # TensorFlow model function
            "train_params": {
                "epochs": 50,
                "batch_size": 32,
                "verbose": 0  # Silent final training
            },
            "finetune_params": {
                "cv_mode": "nested",
                "inner_cv": KFold(n_splits=3, shuffle=True, random_state=42),  # Custom CV object
                "param_strategy": "weighted_average",  # Weight by performance
                "n_trials": 8,  # Lower trials for deep learning due to computational cost
                "approach": "random",
                "verbose": 2,
                "model_params": {
                    "filters1": [4, 8, 16],
                    "filters2": [32, 64, 128],
                    "dropout_rate": ("float", 0.1, 0.5),  # Continuous parameter
                    "spatial_dropout": ("float", 0.05, 0.2),
                },
                "train_params": {
                    "epochs": 10,  # Faster training during optimization
                    "batch_size": 16,
                    "verbose": 0
                }
            },
        }
    ]
}

# ==============================================================================
# CONFIGURATION PARAMETERS REFERENCE
# ==============================================================================

finetune_params_reference = {
    # === CV MODE SELECTION ===
    "cv_mode": "simple",  # Options: "simple", "per_fold", "nested"

    # === NESTED CV SPECIFIC ===
    "inner_cv": 3,  # int (n_splits) or sklearn CV object for inner folds
    "outer_cv": None,  # Usually handled by main pipeline folds

    # === PARAMETER AGGREGATION STRATEGY ===
    "param_strategy": "per_fold_best",  # Options: "global_best", "per_fold_best", "weighted_average"

    # === STANDARD OPTUNA PARAMETERS ===
    "n_trials": 20,
    "approach": "random",  # Options: "random", "grid", "auto"
    "verbose": 1,  # 0=silent, 1=basic, 2=detailed, 3=debug

    # === MODEL PARAMETERS TO OPTIMIZE ===
    "model_params": {
        "param1": [1, 2, 3],  # Categorical
        "param2": ("int", 1, 10),  # Integer range
        "param3": ("float", 0.1, 1.0),  # Float range
    },

    # === TRAINING PARAMETERS DURING OPTIMIZATION ===
    "train_params": {
        "verbose": 0,  # Usually silent during trials
        # Other framework-specific training parameters
    }
}

# ==============================================================================
# COMPUTATIONAL COST COMPARISON
# ==============================================================================
"""
Assuming 5 outer folds and typical parameter optimization:

1. Simple CV:
   - 1 finetuning run (all training data) + 5 fold training runs
   - Total models trained: ~20 (trials) + 5 = 25 models
   - Fastest option

2. Per-fold CV:
   - 5 finetuning runs (one per fold)
   - Total models trained: ~5 × 20 (trials) = 100 models
   - Moderate computational cost

3. Nested CV:
   - 5 outer folds × 3 inner folds × trials per inner CV
   - Total models trained: ~5 × 3 × 10 = 150 models
   - Highest computational cost but most rigorous

Choose based on your computational budget and required rigor level.
"""

if __name__ == "__main__":
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.pipeline.config import PipelineConfigs
    from nirs4all.dataset.loader import get_dataset

    # Load dataset
    data = get_dataset(dataset_config)

    # Example: Run simple CV finetuning
    print("="*80)
    print("SIMPLE CV EXAMPLE")
    print("="*80)

    config = PipelineConfigs(simple_cv_pipeline, "simple_cv_example")
    runner = PipelineRunner()

    try:
        res_dataset, history, pipeline = runner.run(config, data)
        print("✅ Simple CV completed successfully!")
    except Exception as e:
        print(f"❌ Simple CV failed: {e}")

    # Uncomment to try other examples:
    # config = PipelineConfigs(per_fold_cv_pipeline, "per_fold_cv_example")
    # config = PipelineConfigs(nested_cv_pipeline, "nested_cv_example")
    # config = PipelineConfigs(nested_cv_tensorflow_pipeline, "nested_cv_tf_example")