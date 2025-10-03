"""
Optuna Controller - External hyperparameter optimization logic

This module handles all Optuna-related functionality that was previously
embedded in the model controller. It provides a clean interface for
hyperparameter optimization across different strategies and frameworks.
"""

from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptunaController:
    """
    External Optuna controller for hyperparameter optimization.

    Handles all optimization strategies as specified in the user's pseudo-code:
    - Individual fold optimization
    - Grouped fold optimization
    - Single optimization (no folds)
    """

    def __init__(self):
        self.is_available = OPTUNA_AVAILABLE

    def finetune(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        controller: Any  # The model controller instance
    ) -> Dict[str, Any]:
        """
        Main finetune method following user's pseudo-code:

        if folds and mode == individual:
            best_params = []
            foreach fold:
                get x_train_fold_i, x_val_fold_i
                best_params.append(optuna.loop(objective(x_train_fold_i, x_val_fold_i)))
            return best_params

        if folds and mode == grouped:
            return best_param = optuna.loop(objective(folds, data, evalMode))

        if not fold:
            return optuna.loop(objective(folds, data))
        """

        if not self.is_available:
            print("⚠️ Optuna not available, skipping finetuning")
            return {}

        strategy = finetune_params.get('approach', 'individual')
        eval_mode = finetune_params.get('eval_mode', 'best')
        n_trials = finetune_params.get('n_trials', 50)
        verbose = finetune_params.get('verbose', 0)

        if folds and strategy == 'individual':
            # best_params = [], foreach fold: best_params.append(optuna.loop(...))
            return self._optimize_individual_folds(
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, verbose
            )

        elif folds and strategy == 'grouped':
            # return best_param = optuna.loop(objective(folds, data, evalMode))
            return self._optimize_grouped_folds(
                model_config, X_train, y_train, folds, finetune_params,
                n_trials, context, controller, eval_mode, verbose
            )

        else:
            # if not fold: return optuna.loop(objective(folds, data))
            X_val, y_val = X_test, y_test  # Use test as validation
            return self._optimize_single(
                model_config, X_train, y_train, X_val, y_val,
                finetune_params, n_trials, context, controller, verbose
            )

    def _optimize_individual_folds(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: Dict[str, Any],
        controller: Any,
        verbose: int
    ) -> List[Dict[str, Any]]:
        """Optimize each fold individually."""

        best_params = []
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            if verbose > 1:
                print(f"🎯 Optimizing fold {fold_idx + 1}/{len(folds)}")

            # get x_train_fold_i, x_val_fold_i
            X_train_fold = X_train[train_indices]
            y_train_fold = y_train[train_indices]
            X_val_fold = X_train[val_indices]
            y_val_fold = y_train[val_indices]

            # best_params.append(optuna.loop(objective(...)))
            fold_best_params = self._run_optimization(
                model_config, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                finetune_params, n_trials, context, controller
            )
            best_params.append(fold_best_params)

        return best_params

    def _optimize_grouped_folds(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        folds: List,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: Dict[str, Any],
        controller: Any,
        eval_mode: str,
        verbose: int
    ) -> Dict[str, Any]:
        """Optimize using grouped fold evaluation."""

        def objective(trial):
            # Sample hyperparameters
            sampled_params = self._sample_hyperparameters(trial, finetune_params, controller)

            # Train on all folds and collect scores
            scores = []
            for train_indices, val_indices in folds:
                X_train_fold = X_train[train_indices]
                y_train_fold = y_train[train_indices]
                X_val_fold = X_train[val_indices]
                y_val_fold = y_train[val_indices]

                # Create and train model
                base_model = controller._get_model_instance(model_config)
                model = controller.model_utils.clone_model(base_model)
                if hasattr(model, 'set_params'):
                    model.set_params(**sampled_params)

                X_train_prep, y_train_prep = controller._prepare_data(X_train_fold, y_train_fold, context)
                X_val_prep, y_val_prep = controller._prepare_data(X_val_fold, y_val_fold, context)

                try:
                    trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep)
                    score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)
                    scores.append(score)
                except Exception:
                    scores.append(float('inf'))

            # Return evaluation based on eval_mode
            if eval_mode == 'best':
                return min(scores)
            elif eval_mode == 'avg':
                return np.mean(scores)
            elif eval_mode == 'robust_best':
                scores = [s for s in scores if s != float('inf')]
                return min(scores) if scores else float('inf')
            else:
                return np.mean(scores)

        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params

    def _optimize_single(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: Dict[str, Any],
        controller: Any,
        verbose: int
    ) -> Dict[str, Any]:
        """Optimize without folds."""

        return self._run_optimization(
            model_config, X_train, y_train, X_val, y_val,
            finetune_params, n_trials, context, controller
        )

    def _run_optimization(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        finetune_params: Dict[str, Any],
        n_trials: int,
        context: Dict[str, Any],
        controller: Any
    ) -> Dict[str, Any]:
        """Run single optimization study."""

        def objective(trial):
            # Sample hyperparameters
            sampled_params = self._sample_hyperparameters(trial, finetune_params, controller)

            # Create model with sampled params
            base_model = controller._get_model_instance(model_config)
            model = controller.model_utils.clone_model(base_model)
            if hasattr(model, 'set_params'):
                model.set_params(**sampled_params)

            # Prepare data
            X_train_prep, y_train_prep = controller._prepare_data(X_train, y_train, context)
            X_val_prep, y_val_prep = controller._prepare_data(X_val, y_val, context)

            # Train and evaluate
            try:
                trained_model = controller._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep)
                score = controller._evaluate_model(trained_model, X_val_prep, y_val_prep)
                return score
            except Exception as e:
                print(f"⚠️ Trial failed: {e}")
                return float('inf')

        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params

    def _sample_hyperparameters(
        self,
        trial,
        finetune_params: Dict[str, Any],
        controller: Any
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters - delegates to controller's implementation
        for framework-specific parameter handling.
        """

        # Delegate to controller's implementation
        if hasattr(controller, '_sample_hyperparameters'):
            return controller._sample_hyperparameters(trial, finetune_params)

        # Fallback implementation for basic parameter types
        params = {}
        model_params = finetune_params.get('model_params', {})

        for param_name, param_config in model_params.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')

                if param_type == 'int':
                    low = param_config.get('low', 1)
                    high = param_config.get('high', 100)
                    params[param_name] = trial.suggest_int(param_name, low, high)

                elif param_type == 'float':
                    low = param_config.get('low', 1e-5)
                    high = param_config.get('high', 1.0)
                    log = param_config.get('log', False)
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)

                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    if choices:
                        params[param_name] = trial.suggest_categorical(param_name, choices)

        return params

    def create_objective_function(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        controller: Any,
        train_params: Dict[str, Any]
    ) -> Callable:
        """
        Create objective function for optimization.
        This is used by the original abstract controller for compatibility.
        """

        def objective(trial):
            # Sample hyperparameters using controller's method
            sampled_params = self._sample_hyperparameters(trial, {}, controller)

            # Create and train model
            base_model = controller._get_model_instance(model_config)
            model = controller.model_utils.clone_model(base_model)

            if hasattr(model, 'set_params') and sampled_params:
                model.set_params(**sampled_params)

            try:
                trained_model = controller._train_model(
                    model, X_train, y_train, X_val, y_val, **train_params
                )
                score = controller._evaluate_model(trained_model, X_val, y_val)
                return score
            except Exception:
                return float('inf')

        return objective

    def optimize_hyperparameters(
        self,
        objective_function: Callable,
        finetune_params: Dict[str, Any],
        verbose: int = 0
    ) -> tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization with given objective function.
        Used for compatibility with original abstract controller.
        """

        if not self.is_available:
            return {}, float('inf')

        n_trials = finetune_params.get('n_trials', 50)

        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_function, n_trials=n_trials, show_progress_bar=False)

        return study.best_params, study.best_value