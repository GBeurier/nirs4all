"""
Model Configuration - Type-safe configuration classes for model controllers

This module provides type-safe configuration classes to replace the complex
dictionary-based configuration system used in the legacy BaseModelController.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union
from enum import Enum


class CVMode(Enum):
    """Enumeration of cross-validation finetuning strategies."""
    SIMPLE = "simple"  # Finetune on full train data, then train on folds
    PER_FOLD = "per_fold"  # Finetune on each fold individually
    NESTED = "nested"  # Inner folds for finetuning, outer folds for training
    GLOBAL_AVERAGE = "global_average"  # Optimize params by averaging performance across all folds


class ParamStrategy(Enum):
    """Parameter aggregation strategies for cross-validation."""
    GLOBAL_BEST = "global_best"  # Use single best params for all folds
    PER_FOLD_BEST = "per_fold_best"  # Use best params per fold
    WEIGHTED_AVERAGE = "weighted_average"  # Average params weighted by performance
    GLOBAL_AVERAGE = "global_average"  # Optimize params by averaging performance across all folds
    ENSEMBLE_BEST = "ensemble_best"  # Optimize for ensemble prediction performance
    ROBUST_BEST = "robust_best"  # Optimize for minimum worst-case performance (min-max)
    STABILITY_BEST = "stability_best"  # Optimize for parameter stability (minimize performance variance)


@dataclass
class ModelConfig:
    """Type-safe model configuration."""

    model_params: Dict[str, Any] = field(default_factory=dict)
    train_params: Dict[str, Any] = field(default_factory=dict)
    finetune_params: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    model_instance: Optional[Any] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Factory method to create config from dictionary with validation."""
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")

        # Extract and validate required fields
        model_params = config_dict.get('model_params', config_dict.get('train_params', {}))
        train_params = config_dict.get('train_params', {})
        finetune_params = config_dict.get('finetune_params')
        name = config_dict.get('name')
        model_instance = config_dict.get('model_instance')

        return cls(
            model_params=model_params,
            train_params=train_params,
            finetune_params=finetune_params,
            name=name,
            model_instance=model_instance
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary format."""
        result: Dict[str, Any] = {
            'model_params': self.model_params,
            'train_params': self.train_params,
        }
        if self.finetune_params is not None:
            result['finetune_params'] = self.finetune_params
        if self.name is not None:
            result['name'] = self.name
        if self.model_instance is not None:
            result['model_instance'] = self.model_instance
        return result


@dataclass
class CVConfig:
    """Cross-validation configuration."""

    mode: CVMode = CVMode.SIMPLE
    param_strategy: ParamStrategy = ParamStrategy.PER_FOLD_BEST
    inner_cv: Optional[int] = None
    outer_cv: Optional[int] = None
    use_full_train_for_final: bool = False
    n_folds: int = 3

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CVConfig':
        """Factory method to create CV config from dictionary."""
        cv_mode_str = config_dict.get('cv_mode', 'simple')
        param_strategy_str = config_dict.get('param_strategy', 'per_fold_best')

        try:
            mode = CVMode(cv_mode_str)
        except ValueError as exc:
            raise ValueError(f"Invalid CV mode: {cv_mode_str}") from exc

        try:
            param_strategy = ParamStrategy(param_strategy_str)
        except ValueError as exc:
            raise ValueError(f"Invalid parameter strategy: {param_strategy_str}") from exc

        return cls(
            mode=mode,
            param_strategy=param_strategy,
            inner_cv=config_dict.get('inner_cv'),
            outer_cv=config_dict.get('outer_cv'),
            use_full_train_for_final=config_dict.get('use_full_train_for_final', False),
            n_folds=config_dict.get('n_folds', 3)
        )


@dataclass
class FinetuneConfig:
    """Finetuning configuration."""

    n_trials: int = 10
    approach: str = 'auto'
    model_params: Dict[str, Any] = field(default_factory=dict)
    train_params: Dict[str, Any] = field(default_factory=dict)
    verbose: int = 0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FinetuneConfig':
        """Factory method to create finetune config from dictionary."""
        return cls(
            n_trials=config_dict.get('n_trials', 10),
            approach=config_dict.get('approach', 'auto'),
            model_params=config_dict.get('model_params', {}),
            train_params=config_dict.get('train_params', {}),
            verbose=config_dict.get('verbose', 0)
        )


@dataclass
class DataSplit:
    """Type-safe data split representation."""

    X_train: Any
    y_train: Any
    X_val: Optional[Any] = None
    y_val: Optional[Any] = None
    X_test: Optional[Any] = None
    y_test: Optional[Any] = None
    fold_idx: Optional[int] = None

    @property
    def has_validation(self) -> bool:
        """Check if validation data is available."""
        return self.X_val is not None and self.y_val is not None

    @property
    def has_test(self) -> bool:
        """Check if test data is available."""
        return self.X_test is not None and self.y_test is not None
