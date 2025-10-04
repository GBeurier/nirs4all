"""
Optuna Manager - Handles hyperparameter optimization with Optuna

This module provides a clean interface for hyperparameter optimization
using Optuna, with support for different sampling strategies and
framework-specific parameter handling.
"""
import os
os.environ['DISABLE_EMOJIS'] = '1'  # Set to '1' to disable emojis in print statements

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class OptunaManager:
    """
    Manager for Optuna hyperparameter optimization.

    This class handles the complete Optuna optimization workflow,
    including parameter sampling, study creation, and result tracking.
    """

    def __init__(self):
        """Initialize the Optuna manager."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")

    def create_study(
        self,
        direction: str = "minimize",
        sampler: Optional[str] = None,
        search_space: Optional[Dict[str, List]] = None
    ) -> Any:
        """
        Create an Optuna study with appropriate sampler.

        Args:
            direction: Optimization direction ("minimize" or "maximize")
            sampler: Sampler type ("tpe", "grid", or None for auto)
            search_space: Search space for grid sampler

        Returns:
            Optuna study instance
        """
        if not OPTUNA_AVAILABLE or optuna is None:
            raise ImportError("Optuna is not available")

        if sampler == "grid" and search_space:
            # Use GridSampler for exhaustive search
            sampler_instance = GridSampler(search_space)
        else:
            # Use TPE sampler by default
            sampler_instance = TPESampler()

        study = optuna.create_study(direction=direction, sampler=sampler_instance)
        return study

    def optimize_hyperparameters(
        self,
        objective_function: Callable[[Any], float],
        finetune_config: Dict[str, Any],
        verbose: int = 0
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization.

        Args:
            objective_function: Function to optimize (takes trial, returns score)
            finetune_config: Finetuning configuration
            verbose: Verbosity level

        Returns:
            Tuple of (best_params, best_score)
        """
        if not OPTUNA_AVAILABLE or optuna is None:
            raise ImportError("Optuna is not available")

        # Extract optimization parameters
        n_trials = finetune_config.get('n_trials', 10)
        approach = finetune_config.get('approach', 'auto')

        # Determine if grid search is suitable
        is_grid_suitable = self._is_grid_search_suitable(finetune_config)

        if approach == 'auto':
            approach = 'grid' if is_grid_suitable else 'tpe'

        # Create search space for grid search
        search_space = None
        if approach == 'grid':
            search_space = self._create_grid_search_space(finetune_config)

        # Create study
        direction = "minimize"  # Most ML metrics are minimized
        study = self.create_study(direction, approach, search_space)

        # Configure logging
        if verbose < 2 and optuna is not None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        if verbose > 1:
            print(f"🎯 Starting hyperparameter optimization with {approach} sampling ({n_trials} trials)...")

        study.optimize(objective_function, n_trials=n_trials)

        # Get best results
        best_params = study.best_params
        best_score = study.best_value

        if verbose > 1:
            print(f"🏆 Optimization completed. Best score: {best_score:.4f}")
            print(f"📊 Best parameters: {best_params}")

        return best_params, best_score

    def sample_hyperparameters(
        self,
        trial: Any,
        finetune_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters for an Optuna trial.

        Args:
            trial: Optuna trial instance
            finetune_config: Finetuning configuration

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        # Get model parameters to optimize - support both nested and flat structure
        model_params = finetune_config.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_config (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_config.items()
                            if k not in ['n_trials', 'approach', 'train_params', 'verbose']}

        for param_name, param_config in model_params.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                # Tuple format: ('type', min, max)
                param_type, min_val, max_val = param_config
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                # Range parameter - determine type from values
                min_val, max_val = param_config
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, float(min_val), float(max_val))
            elif isinstance(param_config, dict):
                # Complex parameter configuration
                param_type = param_config.get('type', 'categorical')
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                else:
                    raise ValueError(f"Unknown parameter type in config: {param_type}")
            else:
                # Single value - pass through unchanged
                params[param_name] = param_config

        return params

    def _is_grid_search_suitable(self, finetune_config: Dict[str, Any]) -> bool:
        """
        Check if grid search is suitable (all parameters are categorical).

        Args:
            finetune_config: Finetuning configuration

        Returns:
            True if grid search is suitable
        """
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_config.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_config (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_config.items()
                            if k not in ['n_trials', 'approach', 'train_params', 'verbose']}

        for param_name, param_config in model_params.items():
            # Only categorical (list) parameters are suitable for grid search
            # Tuple parameters (ranges) need random/TPE sampling
            if not isinstance(param_config, list):
                return False
        return True

    def _create_grid_search_space(self, finetune_config: Dict[str, Any]) -> Dict[str, List]:
        """
        Create grid search space for categorical parameters only.

        Args:
            finetune_config: Finetuning configuration

        Returns:
            Search space dictionary
        """
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_config.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_config (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_config.items()
                            if k not in ['n_trials', 'approach', 'train_params', 'verbose']}

        search_space = {}
        for param_name, param_config in model_params.items():
            # Only include list parameters (categorical) in grid search
            # Explicitly ignore tuples which are range specifications
            if isinstance(param_config, list):
                search_space[param_name] = param_config
            # Ignore tuples like ('int', min, max) and ('float', min, max)
            elif isinstance(param_config, tuple):
                pass  # Skip range parameters
        return search_space

    def create_objective_function(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        controller: Any,
        train_params: Optional[Dict[str, Any]] = None
    ) -> Callable[[Any], float]:
        """
        Create an objective function for Optuna optimization.

        Args:
            model_config: Model configuration
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            controller: Model controller instance
            train_params: Training parameters

        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Sample hyperparameters
            trial_params = self.sample_hyperparameters(trial, model_config.get('finetune_params', {}))

            # Create model with trial parameters
            base_model = controller._get_model_instance(model_config)
            model = controller.model_manager.clone_model(base_model)

            # Apply trial parameters
            if hasattr(model, 'set_params') and trial_params:
                try:
                    model.set_params(**trial_params)
                except Exception as e:
                    # Invalid parameter combination, return worst score
                    return float('inf')

            # Prepare data
            X_train_prep, y_train_prep = controller._prepare_data(X_train, y_train, {})
            X_val_prep, y_val_prep = controller._prepare_data(X_val, y_val, {})

            # Train model
            trial_train_params = train_params.copy() if train_params else {}
            trial_train_params['verbose'] = 0  # Silent training during optimization

            trained_model = controller._train_model(
                model, X_train_prep, y_train_prep,
                X_val=X_val_prep, y_val=y_val_prep,
                train_params=trial_train_params
            )

            # Evaluate model
            score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)

            return score

        return objective
