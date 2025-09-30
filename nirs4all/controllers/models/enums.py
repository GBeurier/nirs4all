"""
Model controller enums and constants.

This module contains all the enumeration types used by the model controllers.
"""

from enum import Enum


class ModelMode(Enum):
    """Enumeration of model execution modes."""
    TRAIN = "train"
    FINETUNE = "finetune"
    PREDICT = "predict"


class CVMode(Enum):
    """Enumeration of cross-validation finetuning strategies."""
    SIMPLE = "simple"  # Finetune on full train data, then train on folds
    PER_FOLD = "per_fold"  # Finetune on each fold individually
    NESTED = "nested"  # Inner folds for finetuning, outer folds for training


class ParamStrategy(Enum):
    """Parameter aggregation strategies for cross-validation."""
    GLOBAL_BEST = "global_best"  # Use single best params for all folds
    PER_FOLD_BEST = "per_fold_best"  # Use best params per fold
    WEIGHTED_AVERAGE = "weighted_average"  # Average params weighted by performance
    GLOBAL_AVERAGE = "global_average"  # Optimize params by averaging performance across all folds
    ENSEMBLE_BEST = "ensemble_best"  # Optimize for ensemble prediction performance
    ROBUST_BEST = "robust_best"  # Optimize for minimum worst-case performance (min-max)
    STABILITY_BEST = "stability_best"  # Optimize for parameter stability (minimize performance variance)
