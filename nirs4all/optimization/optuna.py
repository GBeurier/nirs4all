"""
Optuna Manager - External hyperparameter optimization logic

This module combines the best practices from the original optuna_manager for parameter handling
and sampling with fold-based optimization strategies. It provides a clean interface for
hyperparameter optimization across different strategies and frameworks.
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'  # Set to '1' to disable emojis in print statements

from typing import Any, Dict, List, Optional, Callable, Union, TYPE_CHECKING
import numpy as np
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext

try:
    import optuna
    from optuna.samplers import TPESampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from nirs4all.controllers.models.factory import ModelFactory


class OptunaManager:
    """
    External Optuna manager for hyperparameter optimization.

    Combines robust parameter handling with flexible fold-based optimization strategies:
    - Individual fold optimization
    - Grouped fold optimization
    - Single optimization (no folds)
    - Smart sampler selection (TPE, Grid)
    - Multiple evaluation modes (best, avg, robust_best)
    """

    def __init__(self):
        """Initialize the Optuna manager."""
        self.is_available = OPTUNA_AVAILABLE
        if not self.is_available:
            logger.warning("Optuna not available - finetuning will be skipped")

    def finetune(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List],
        finetune_params: Dict[str, Any],
        context: 'ExecutionContext',
        controller: Any  # The model controller instance
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main finetune entry point - delegates to appropriate optimization strategy.

        Args:
            model_config: Model configuration
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            folds: List of (train_indices, val_indices) tuples or None
            finetune_params: Finetuning configuration
            context: Pipeline context
            controller: Model controller instance

        Returns:
            Best parameters (dict) or list of best parameters per fold
        """
        if not self.is_available:
            logger.warning("Optuna not available, skipping finetuning")
            return {}

        # Extract configuration
        strategy = finetune_params.get('approach', 'grouped')
        eval_mode = finetune_params.get('eval_mode', 'best')
        n_trials = finetune_params.get('n_trials', 50)
        verbose = finetune_params.get('verbose', 0)

        if verbose > 1:
            logger.info("Starting hyperparameter optimization:")
            logger.info(f"   Strategy: {strategy}")
            logger.info(f"   Eval mode: {eval_mode}")
            logger.info(f"   Trials: {n_trials}")
            logger.info(f"   Folds: {len(folds) if folds else 0}")

        # Route to appropriate optimization strategy
        if folds and strategy == 'individual':
            # Individual fold optimization: best_params = [], foreach fold: best_params.append(optuna.loop(...))
            return self._optimize_individual_folds(
                dataset,
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, verbose
            )

        elif folds and strategy == 'grouped':
            # Grouped fold optimization: return best_param = optuna.loop(objective(folds, data, evalMode))
            return self._optimize_grouped_folds(
                dataset,
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, eval_mode, verbose
            )

        else:
            # Single optimization (no folds): return optuna.loop(objective(data))
            # X_val, y_val = X_test, y_test  # Use test as validation
            X_val, y_val = X_train, y_train  # Use train as validation
            return self._optimize_single(
                dataset,
                model_config, X_train, y_train, X_val, y_val,
                finetune_params, n_trials, context, controller, verbose
            )

    def _optimize_individual_folds(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int
    ) -> List[Dict[str, Any]]:
        """
        Optimize each fold individually.

        Returns list of best parameters for each fold.
        """
        best_params_list = []

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            if verbose > 1:
                logger.info(f"Optimizing fold {fold_idx + 1}/{len(folds)}")

            # Extract fold data
            X_train_fold = X_train[train_indices]
            y_train_fold = y_train[train_indices]
            X_val_fold = X_train[val_indices]
            y_val_fold = y_train[val_indices]

            # Run optimization for this fold
            fold_best_params = self._run_single_optimization(
                dataset,
                model_config, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                finetune_params, n_trials, context, controller, verbose=0
            )

            best_params_list.append(fold_best_params)

            if verbose > 1:
                logger.info(f"   Fold {fold_idx + 1} best: {fold_best_params}")

        return best_params_list

    def _optimize_grouped_folds(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        eval_mode: str,
        verbose: int
    ) -> Dict[str, Any]:
        """
        Optimize using grouped fold evaluation.

        Single optimization where objective function evaluates across all folds.
        """
        # Create objective function that evaluates across all folds
        def objective(trial):
            # Sample hyperparameters
            sampled_params = self.sample_hyperparameters(trial, finetune_params)

            # Process parameters if controller supports it
            if hasattr(controller, 'process_hyperparameters'):
                sampled_params = controller.process_hyperparameters(sampled_params)

            if verbose > 2:
                logger.debug(f"Trial params: {sampled_params}")

            # Train on all folds and collect scores
            scores = []
            for train_indices, val_indices in folds:
                X_train_fold = X_train[train_indices]
                y_train_fold = y_train[train_indices]
                X_val_fold = X_train[val_indices]
                y_val_fold = y_train[val_indices]
                try:
                    model = controller._get_model_instance(dataset, model_config, force_params=sampled_params)  # noqa: SLF001

                    # Prepare data
                    X_train_prep, y_train_prep = controller._prepare_data(X_train_fold, y_train_fold, context)  # noqa: SLF001
                    X_val_prep, y_val_prep = controller._prepare_data(X_val_fold, y_val_fold, context)  # noqa: SLF001

                    # Train and evaluate - pass train_params from finetune_params
                    train_params_for_trial = finetune_params.get('train_params', {}).copy()

                    # Merge sampled params into train_params
                    # This ensures 'compile' and 'fit' dicts from processed params are available to _train_model
                    train_params_for_trial.update(sampled_params)

                    # Ensure task_type is passed for models that need it (e.g., TensorFlow)
                    if 'task_type' not in train_params_for_trial:
                        train_params_for_trial['task_type'] = dataset.task_type
                    trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                    score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)  # noqa: SLF001
                    scores.append(score)

                except Exception as e:
                    if verbose >= 2:
                        logger.debug(f"   Fold failed: {e}")
                    scores.append(float('inf'))

            # Return evaluation based on eval_mode
            return self._aggregate_scores(scores, eval_mode)

        # Run optimization with the multi-fold objective
        study = self._create_study(finetune_params)
        self._configure_logging(verbose)

        if verbose > 1:
            logger.starting(f"Running grouped optimization ({n_trials} trials)...")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if verbose > 1:
            logger.success(f"Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

        return study.best_params

    def _optimize_single(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int
    ) -> Dict[str, Any]:
        """Optimize without folds - single train/val split."""
        return self._run_single_optimization(
            dataset,
            model_config, X_train, y_train, X_val, y_val,
            finetune_params, n_trials, context, controller, verbose
        )

    def _run_single_optimization(
        self,
        dataset: 'SpectroDataset',
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: 'ExecutionContext',
        controller: Any,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Run single optimization study for a train/val split.

        Core optimization logic used by both individual fold and single optimization.
        """
        def objective(trial):
            # Sample hyperparameters
            sampled_params = self.sample_hyperparameters(trial, finetune_params)

            # Process parameters if controller supports it
            if hasattr(controller, 'process_hyperparameters'):
                sampled_params = controller.process_hyperparameters(sampled_params)

            if verbose > 2:
                logger.debug(f"Trial params: {sampled_params}")

            try:
                model = controller._get_model_instance(dataset, model_config, force_params=sampled_params)  # noqa: SLF001

                # Prepare data
                X_train_prep, y_train_prep = controller._prepare_data(X_train, y_train, context)  # noqa: SLF001
                X_val_prep, y_val_prep = controller._prepare_data(X_val, y_val, context)  # noqa: SLF001

                # Train and evaluate - pass train_params from finetune_params
                train_params_for_trial = finetune_params.get('train_params', {}).copy()

                # Merge sampled params into train_params
                train_params_for_trial.update(sampled_params)

                # Ensure task_type is passed for models that need it (e.g., TensorFlow)
                if 'task_type' not in train_params_for_trial:
                    train_params_for_trial['task_type'] = dataset.task_type
                trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **train_params_for_trial)  # noqa: SLF001
                score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)  # noqa: SLF001

                return score

            except Exception as e:
                if verbose >= 2:
                    logger.warning(f"Trial failed: {e}")
                return float('inf')

        # Create and run optimization
        study = self._create_study(finetune_params)
        self._configure_logging(verbose)

        if verbose > 1:
            logger.starting(f"Running optimization ({n_trials} trials)...")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if verbose > 1:
            logger.success(f"Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

        return study.best_params

    def _create_study(self, finetune_params: Dict[str, Any]) -> Any:
        """
        Create an Optuna study with appropriate sampler.

        Uses grid sampler for categorical-only parameters, TPE otherwise.
        """
        if not OPTUNA_AVAILABLE or optuna is None:
            raise ImportError("Optuna is not available")

        # Determine optimal sampler strategy
        # Support both 'sampler' and 'sample' keys for backward compatibility
        sampler_type = finetune_params.get('sampler', finetune_params.get('sample', 'auto'))

        if sampler_type == 'auto':
            # Auto-detect best sampler based on parameter types
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            sampler_type = 'grid' if is_grid_suitable else 'tpe'
        elif sampler_type == 'grid':
            # Verify grid is suitable even if explicitly requested
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            if not is_grid_suitable:
                logger.warning("Grid sampler requested but parameters are not all categorical. Using TPE sampler instead.")
                sampler_type = 'tpe'

        # Create sampler instance
        if sampler_type == 'grid':
            search_space = self._create_grid_search_space(finetune_params)
            sampler = GridSampler(search_space)
        else:
            sampler = TPESampler()

        # Create study
        direction = "minimize"  # Most ML metrics are loss-based (minimize)
        study = optuna.create_study(direction=direction, sampler=sampler)

        return study

    def _configure_logging(self, verbose: int):
        """Configure Optuna logging based on verbosity level."""
        if verbose < 2 and optuna is not None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _aggregate_scores(self, scores: List[float], eval_mode: str) -> float:
        """
        Aggregate fold scores based on evaluation mode.

        Args:
            scores: List of scores from different folds
            eval_mode: How to aggregate ('best', 'avg', 'robust_best')

        Returns:
            Aggregated score
        """
        if eval_mode == 'best':
            return min(scores)
        elif eval_mode == 'avg':
            return np.sum(scores)
        elif eval_mode == 'robust_best':
            # Exclude infinite scores (failed trials) then take best
            valid_scores = [s for s in scores if s != float('inf')]
            return min(valid_scores) if valid_scores else float('inf')
        else:
            # Default to average
            return np.sum(scores)

    def sample_hyperparameters(
        self,
        trial: Any,
        finetune_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters for an Optuna trial.

        Robust parameter handling supporting multiple formats:
        - Categorical: [val1, val2, val3]
        - Range tuple: (min, max) or ('type', min, max)
        - Dict config: {'type': 'int', 'min': 1, 'max': 10}
        - Single values: passed through unchanged

        Args:
            trial: Optuna trial instance
            finetune_params: Finetuning configuration

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        # Get model parameters - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # Legacy support: look for parameters directly in finetune_params
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                          if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'sample', 'train_params', 'verbose']}

        for param_name, param_config in model_params.items():
            params[param_name] = self._sample_single_parameter(trial, param_name, param_config)

        return params

    # Supported parameter type strings for tuple format ('type', min, max)
    _INT_TYPES = ('int', int, 'builtins.int')
    _INT_LOG_TYPES = ('int_log', 'log_int')
    _FLOAT_TYPES = ('float', float, 'builtins.float')
    _FLOAT_LOG_TYPES = ('float_log', 'log_float')
    _ALL_RANGE_TYPES = _INT_TYPES + _INT_LOG_TYPES + _FLOAT_TYPES + _FLOAT_LOG_TYPES

    def _sample_single_parameter(self, trial: Any, param_name: str, param_config: Any) -> Any:
        """
        Sample a single parameter based on its configuration.

        Supported formats:

        **Tuple format** (most common):
            - ('int', min, max)        → Integer uniform sampling
            - ('int_log', min, max)    → Integer log-uniform sampling
            - ('float', min, max)      → Float uniform sampling
            - ('float_log', min, max)  → Float log-uniform sampling (recommended for learning rates, regularization)
            - (min, max)               → Inferred type based on values (int if both int, else float)

        **List format**:
            - [val1, val2, val3]       → Categorical sampling

        **Dict format** (most flexible):
            - {'type': 'categorical', 'choices': [v1, v2, v3]}
            - {'type': 'int', 'min': 1, 'max': 10}
            - {'type': 'int', 'min': 1, 'max': 10, 'step': 2}
            - {'type': 'int', 'min': 1, 'max': 1000, 'log': True}
            - {'type': 'float', 'min': 0.0, 'max': 1.0}
            - {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1}
            - {'type': 'float', 'min': 1e-5, 'max': 1e-1, 'log': True}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0, 'element_type': 'float'}

        **Single value**:
            - Any scalar value → Passed through unchanged

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Parameter configuration

        Returns:
            Sampled parameter value
        """
        if isinstance(param_config, list):
            # Check if this is actually a type specification that was converted from tuple to list
            # Pattern: ['int'/'float'/etc., min, max] from tuple ('type', min, max)
            if (len(param_config) == 3 and
                param_config[0] in self._ALL_RANGE_TYPES and
                isinstance(param_config[1], (int, float)) and
                isinstance(param_config[2], (int, float))):
                # This is a range specification, not a categorical list
                param_type, min_val, max_val = param_config
                return self._suggest_from_type(trial, param_name, param_type, min_val, max_val)

            # Regular categorical parameter: [val1, val2, val3]
            return trial.suggest_categorical(param_name, param_config)

        elif isinstance(param_config, tuple) and len(param_config) == 3:
            # Explicit type tuple: ('type', min, max)
            param_type, min_val, max_val = param_config
            return self._suggest_from_type(trial, param_name, param_type, min_val, max_val)

        elif isinstance(param_config, tuple) and len(param_config) == 2:
            # Range tuple: (min, max) - infer type from values
            min_val, max_val = param_config
            if isinstance(min_val, int) and isinstance(max_val, int):
                return trial.suggest_int(param_name, min_val, max_val)
            else:
                return trial.suggest_float(param_name, float(min_val), float(max_val))

        elif isinstance(param_config, dict):
            # Dictionary configuration with full options
            return self._suggest_from_dict(trial, param_name, param_config)

        else:
            # Single value - pass through unchanged
            return param_config

    def _suggest_from_type(
        self,
        trial: Any,
        param_name: str,
        param_type: Any,
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        log: bool = False
    ) -> Any:
        """
        Suggest a parameter value based on type string.

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_type: Type indicator ('int', 'float', 'int_log', 'float_log', etc.)
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (optional, for discrete sampling)
            log: Whether to use log-uniform sampling (overridden by type suffix)

        Returns:
            Sampled value
        """
        # Determine if log scale from type suffix
        use_log = log or param_type in self._INT_LOG_TYPES or param_type in self._FLOAT_LOG_TYPES

        if param_type in self._INT_TYPES or param_type in self._INT_LOG_TYPES:
            # Integer sampling
            int_step = int(step) if step is not None else 1
            return trial.suggest_int(param_name, int(min_val), int(max_val), step=int_step, log=use_log)

        elif param_type in self._FLOAT_TYPES or param_type in self._FLOAT_LOG_TYPES:
            # Float sampling
            return trial.suggest_float(param_name, float(min_val), float(max_val), step=step, log=use_log)

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for parameter '{param_name}'. "
                f"Supported types: 'int', 'int_log', 'float', 'float_log' (or Python int/float types)"
            )

    def _suggest_from_dict(self, trial: Any, param_name: str, param_config: Dict[str, Any]) -> Any:
        """
        Suggest a parameter value from a dictionary configuration.

        Supports:
            - {'type': 'categorical', 'choices': [v1, v2, v3]}
            - {'type': 'int', 'min': 1, 'max': 10, 'step': 2, 'log': False}
            - {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1, 'log': True}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0, 'element_type': 'float'}

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Dictionary with 'type' and range/choices

        Returns:
            Sampled value
        """
        param_type = param_config.get('type', 'categorical')

        if param_type == 'categorical':
            choices = param_config.get('choices', param_config.get('values', []))
            if not choices:
                raise ValueError(f"Categorical parameter '{param_name}' requires 'choices' or 'values' list")
            return trial.suggest_categorical(param_name, choices)

        elif param_type in ('int', 'int_log'):
            min_val = param_config.get('min', param_config.get('low'))
            max_val = param_config.get('max', param_config.get('high'))
            step = param_config.get('step', 1)
            log = param_config.get('log', param_type == 'int_log')

            if min_val is None or max_val is None:
                raise ValueError(f"Integer parameter '{param_name}' requires 'min'/'max' or 'low'/'high'")

            return trial.suggest_int(param_name, int(min_val), int(max_val), step=int(step), log=log)

        elif param_type in ('float', 'float_log'):
            min_val = param_config.get('min', param_config.get('low'))
            max_val = param_config.get('max', param_config.get('high'))
            step = param_config.get('step')  # None means continuous
            log = param_config.get('log', param_type == 'float_log')

            if min_val is None or max_val is None:
                raise ValueError(f"Float parameter '{param_name}' requires 'min'/'max' or 'low'/'high'")

            return trial.suggest_float(param_name, float(min_val), float(max_val), step=step, log=log)

        elif param_type == 'sorted_tuple':
            return self._suggest_sorted_tuple(trial, param_name, param_config)

        else:
            raise ValueError(
                f"Unknown parameter type '{param_type}' for parameter '{param_name}'. "
                f"Supported types: 'categorical', 'int', 'int_log', 'float', 'float_log', 'sorted_tuple'"
            )

    def _suggest_sorted_tuple(self, trial: Any, param_name: str, param_config: Dict[str, Any]) -> tuple:
        """
        Suggest a sorted tuple of values.

        Generates N values within a range and returns them as a sorted tuple.
        Useful for parameters like 'alphas' that need ordered sequences.

        Supports:
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': ('int', 3, 5), 'min': 0.0, 'max': 2.0}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0, 'element_type': 'int'}
            - {'type': 'sorted_tuple', 'length': 4, 'min': 0.0, 'max': 2.0, 'step': 0.5}

        Args:
            trial: Optuna trial instance
            param_name: Name of the parameter
            param_config: Dictionary with length, min, max, and optional element_type/step

        Returns:
            Sorted tuple of sampled values
        """
        # Get tuple length - can be fixed or a range
        length_config = param_config.get('length', 3)
        if isinstance(length_config, int):
            length = length_config
        elif isinstance(length_config, (tuple, list)) and len(length_config) == 3:
            # ('int', min, max) or ['int', min, max] format
            _, min_len, max_len = length_config
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        elif isinstance(length_config, (tuple, list)) and len(length_config) == 2:
            # (min, max) or [min, max] format
            min_len, max_len = length_config
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        elif isinstance(length_config, dict):
            min_len = length_config.get('min', length_config.get('low', 2))
            max_len = length_config.get('max', length_config.get('high', 5))
            length = trial.suggest_int(f"{param_name}_length", int(min_len), int(max_len))
        else:
            length = int(length_config)

        # Get value range
        min_val = param_config.get('min', param_config.get('low', 0.0))
        max_val = param_config.get('max', param_config.get('high', 1.0))
        step = param_config.get('step')
        element_type = param_config.get('element_type', 'float')

        # Sample individual elements
        values = []
        for i in range(length):
            elem_name = f"{param_name}_{i}"
            if element_type in ('int', 'int_log'):
                log = param_config.get('log', element_type == 'int_log')
                int_step = int(step) if step is not None else 1
                val = trial.suggest_int(elem_name, int(min_val), int(max_val), step=int_step, log=log)
            else:  # float or float_log
                log = param_config.get('log', element_type == 'float_log')
                val = trial.suggest_float(elem_name, float(min_val), float(max_val), step=step, log=log)
            values.append(val)

        # Sort and return as tuple
        return tuple(sorted(values))

    def _is_grid_search_suitable(self, finetune_params: Dict[str, Any]) -> bool:
        """
        Check if grid search is suitable (all parameters are categorical).

        Grid search only works well when all parameters are categorical (discrete choices).
        Continuous parameters need random/TPE sampling.
        """
        model_params = finetune_params.get('model_params', {})

        # Legacy support
        if not model_params:
            model_params = {
                k: v for k, v in finetune_params.items()
                if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'sample', 'train_params', 'verbose']
            }

        # Debug: uncomment these lines to debug grid suitability checks
        # print(f"[DEBUG] Checking grid suitability. Model params: {model_params}")

        for _, param_config in model_params.items():
            # Check if this is a range specification disguised as a list (from tuple-to-list conversion)
            is_list = isinstance(param_config, list)
            has_len_3 = len(param_config) == 3 if is_list else False
            is_type_spec = param_config[0] in self._ALL_RANGE_TYPES if is_list and has_len_3 else False
            is_min_num = isinstance(param_config[1], (int, float)) if is_list and has_len_3 else False
            is_max_num = isinstance(param_config[2], (int, float)) if is_list and has_len_3 else False

            is_range_spec = is_list and has_len_3 and is_type_spec and is_min_num and is_max_num

            if is_range_spec:
                # This is a range specification, not categorical
                # print(f"[DEBUG] Parameter is a range spec disguised as list, grid search not suitable")
                return False

            # Only categorical (list) parameters are suitable for grid search
            if not isinstance(param_config, list):
                # print(f"[DEBUG] Parameter is not a list (type: {type(param_config)}), grid search not suitable")
                return False

        result = True and len(model_params) > 0  # Need at least one parameter
        # print(f"[DEBUG] Grid search suitable: {result} (num params: {len(model_params)})")
        return result

    def _create_grid_search_space(self, finetune_params: Dict[str, Any]) -> Dict[str, List]:
        """
        Create grid search space for categorical parameters only.

        Returns search space suitable for GridSampler.
        """
        model_params = finetune_params.get('model_params', {})

        # Legacy support
        if not model_params:
            model_params = {
                k: v for k, v in finetune_params.items()
                if k not in ['n_trials', 'approach', 'eval_mode', 'sampler', 'sample', 'train_params', 'verbose']
            }

        search_space = {}
        for param_name, param_config in model_params.items():
            # Only include categorical (list) parameters in grid search
            if isinstance(param_config, list):
                search_space[param_name] = param_config
            # Skip non-categorical parameters

        return search_space
