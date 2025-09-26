#!/usr/bin/env python3
"""
Configuration examples for all NIRS4ALL parameter strategies.

This file demonstrates how to configure each parameter strategy with different
cross-validation modes and provides guidance on when to use each approach.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

# =============================================================================
# PARAMETER STRATEGY CONFIGURATIONS
# =============================================================================

def get_per_fold_best_config():
    """Traditional per-fold parameter optimization (default behavior)."""
    return {
        "name": "per_fold_best_example",
        "steps": [
            {
                "name": "pls_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "per_fold_best",  # Each fold gets its own best params
                    "n_trials": 20,
                    "model_params": {
                        "n_components": ("int", 1, 20)
                    }
                }
            }
        ]
    }

def get_global_best_config():
    """Select single best parameter set from all fold optimizations."""
    return {
        "name": "global_best_example",
        "steps": [
            {
                "name": "rf_model",
                "controller": "sklearn",
                "model": RandomForestRegressor(random_state=42),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_best",  # Single best params for all folds
                    "n_trials": 15,
                    "model_params": {
                        "n_estimators": [50, 100, 200, 300],
                        "max_depth": [5, 10, 20, None],
                        "min_samples_split": ("int", 2, 20)
                    }
                }
            }
        ]
    }

def get_global_average_config():
    """NEW: Optimize parameters by averaging performance across all folds."""
    return {
        "name": "global_average_example",
        "steps": [
            {
                "name": "pls_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",  # ⭐ NEW STRATEGY
                    "n_trials": 12,  # Fewer trials due to higher computational cost
                    "verbose": 1,
                    "model_params": {
                        "n_components": ("int", 1, 15)
                    },
                    "train_params": {
                        "verbose": 0  # Silent during optimization
                    }
                }
            }
        ]
    }

def get_full_training_config():
    """NEW: Optimize with CV but train final model on full training data."""
    return {
        "name": "full_training_example",
        "steps": [
            {
                "name": "pls_full_train",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",
                    "use_full_train_for_final": True,  # ⭐ NEW OPTION
                    "n_trials": 15,
                    "verbose": 1,
                    "model_params": {
                        "n_components": ("int", 1, 20)
                    },
                    "train_params": {
                        "verbose": 0
                    }
                }
            }
        ]
    }

# =============================================================================
# CROSS-VALIDATION MODE CONFIGURATIONS
# =============================================================================

def get_simple_cv_config():
    """Simple CV: Optimize on combined training data, then train on folds."""
    return {
        "name": "simple_cv_example",
        "steps": [
            {
                "name": "svr_model",
                "controller": "sklearn",
                "model": SVR(),
                "finetune_params": {
                    "cv_mode": "simple",  # Fastest approach
                    "param_strategy": "global_average",  # Can use any strategy
                    "n_trials": 25,  # More trials since it's faster per trial
                    "model_params": {
                        "C": ("float", 0.1, 10.0),
                        "gamma": ("float", 0.001, 1.0),
                        "kernel": ["rbf", "linear"]
                    }
                }
            }
        ]
    }

def get_nested_cv_config():
    """Nested CV: Most rigorous approach with inner and outer folds."""
    return {
        "name": "nested_cv_example",
        "steps": [
            {
                "name": "rf_model",
                "controller": "sklearn",
                "model": RandomForestRegressor(random_state=42),
                "finetune_params": {
                    "cv_mode": "nested",  # Most rigorous
                    "inner_cv": 3,  # Inner folds for parameter optimization
                    "param_strategy": "global_average",  # Optimize across inner folds
                    "n_trials": 8,  # Fewer trials due to very high computational cost
                    "verbose": 2,
                    "model_params": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [10, 20, None],
                        "min_samples_split": ("int", 2, 10)
                    }
                }
            }
        ]
    }

# =============================================================================
# SPECIALIZED CONFIGURATIONS
# =============================================================================

def get_quick_prototyping_config():
    """Fast configuration for quick prototyping and testing."""
    return {
        "name": "quick_prototype",
        "steps": [
            {
                "name": "pls_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "simple",
                    "param_strategy": "per_fold_best",
                    "n_trials": 10,  # Quick optimization
                    "verbose": 1,
                    "model_params": {
                        "n_components": ("int", 1, 10)
                    }
                }
            }
        ]
    }

def get_production_ready_config():
    """Configuration for production deployment with rigorous optimization."""
    return {
        "name": "production_model",
        "steps": [
            {
                "name": "ensemble_model",
                "controller": "sklearn",
                "model": RandomForestRegressor(random_state=42),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",  # Most generalizable
                    "n_trials": 20,
                    "verbose": 1,
                    "model_params": {
                        "n_estimators": [200, 300, 500],
                        "max_depth": [15, 20, 25, None],
                        "min_samples_split": ("int", 2, 15),
                        "min_samples_leaf": ("int", 1, 8),
                        "max_features": ["sqrt", "log2", None]
                    },
                    "train_params": {
                        "verbose": 0
                    }
                }
            }
        ]
    }

def get_research_config():
    """Configuration for research with maximum rigor."""
    return {
        "name": "research_model",
        "steps": [
            {
                "name": "rigorous_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "nested",
                    "inner_cv": 5,  # 5 inner folds
                    "param_strategy": "global_average",  # Most unbiased
                    "n_trials": 15,
                    "verbose": 2,
                    "model_params": {
                        "n_components": ("int", 1, 25)
                    },
                    "train_params": {
                        "verbose": 0
                    }
                }
            }
        ]
    }

# =============================================================================
# USAGE EXAMPLES AND RECOMMENDATIONS
# =============================================================================

# Configuration selection guide
CONFIGURATION_GUIDE = {
    "prototyping": {
        "config": get_quick_prototyping_config,
        "description": "Fast testing and development",
        "computational_cost": "Low",
        "use_when": "Initial exploration, quick iterations"
    },

    "standard": {
        "config": get_global_average_config,
        "description": "Balanced performance and generalizability",
        "computational_cost": "Medium-High",
        "use_when": "Standard model development, good balance"
    },

    "full_training": {
        "config": get_full_training_config,
        "description": "CV optimization + single model on full data",
        "computational_cost": "Medium-High",
        "use_when": "Want single unified model with rigorous optimization"
    },

    "production": {
        "config": get_production_ready_config,
        "description": "Optimized for deployment reliability",
        "computational_cost": "High",
        "use_when": "Production deployment, need consistent performance"
    },

    "research": {
        "config": get_research_config,
        "description": "Maximum rigor for publications",
        "computational_cost": "Very High",
        "use_when": "Academic research, publications, unbiased evaluation"
    }
}

def print_configuration_guide():
    """Print the configuration selection guide."""
    print("🎯 NIRS4ALL Parameter Strategy Configuration Guide")
    print("=" * 60)

    for name, info in CONFIGURATION_GUIDE.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Computational Cost: {info['computational_cost']}")
        print(f"  Use When: {info['use_when']}")

def get_config_by_use_case(use_case: str):
    """Get configuration by use case name."""
    if use_case in CONFIGURATION_GUIDE:
        return CONFIGURATION_GUIDE[use_case]["config"]()
    else:
        available = list(CONFIGURATION_GUIDE.keys())
        raise ValueError(f"Unknown use case '{use_case}'. Available: {available}")

# Parameter strategy comparison
STRATEGY_COMPARISON = {
    "per_fold_best": {
        "cost": "Medium",
        "generalizability": "Medium",
        "consistency": "Low",
        "best_for": "Standard optimization with fold-specific needs"
    },

    "global_best": {
        "cost": "Medium",
        "generalizability": "Medium-High",
        "consistency": "High",
        "best_for": "Consistent parameters across folds"
    },

    "global_average": {
        "cost": "High",
        "generalizability": "Very High",
        "consistency": "Very High",
        "best_for": "Most generalizable parameters, production use"
    }
}

def print_strategy_comparison():
    """Print parameter strategy comparison."""
    print("\n📊 Parameter Strategy Comparison")
    print("=" * 45)

    for strategy, info in STRATEGY_COMPARISON.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Computational Cost: {info['cost']}")
        print(f"  Generalizability: {info['generalizability']}")
        print(f"  Parameter Consistency: {info['consistency']}")
        print(f"  Best For: {info['best_for']}")

if __name__ == "__main__":
    print_configuration_guide()
    print_strategy_comparison()

    print(f"\n💡 Quick Start Examples:")
    print("  from config_examples import get_global_average_config")
    print("  config = get_global_average_config()")
    print("  # Use config with PipelineRunner")

    print(f"\n  # Or by use case:")
    print("  config = get_config_by_use_case('standard')")