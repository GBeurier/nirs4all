"""
Base Model Controller - Abstract controller for machine learning models

This module provides the base abstract controller that handles the three main modes:
- Training: Fit model on training data
- Fine-tuning: Hyperparameter optimization using Optuna
- Prediction: Generate predictions and store results

All model controllers (sklearn, tensorflow, pytorch) inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import pickle
import numpy as np
from enum import Enum

from nirs4all.controllers.controller import OperatorController

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class ModelMode(Enum):
    """Enumeration of model execution modes."""
    TRAIN = "train"
    FINETUNE = "finetune"
    PREDICT = "predict"


class BaseModelController(OperatorController, ABC):
    """
    Abstract base controller for machine learning models.

    Handles the common pipeline for:
    1. Data preparation (train/test splits, cross-validation)
    2. Model training with configurable parameters
    3. Hyperparameter tuning with Optuna (optional)
    4. Prediction and result storage

    Each framework-specific controller must implement:
    - _get_model_instance: Create model from configuration
    - _train_model: Framework-specific training logic
    - _predict_model: Framework-specific prediction logic
    - _prepare_data: Framework-specific data formatting
    """

    priority = 15  # Higher priority than transformers, lower than data operations

    @classmethod
    def use_multi_source(cls) -> bool:
        """Models can handle multi-source datasets."""
        return True

    @abstractmethod
    def _get_model_instance(self, model_config: Dict[str, Any]) -> Any:
        """Create a model instance from configuration."""
        pass

    @abstractmethod
    def _train_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        train_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Train the model with framework-specific logic."""
        pass

    @abstractmethod
    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions with framework-specific logic."""
        pass

    @abstractmethod
    def _prepare_data(self, X: Any, y: Any, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data in framework-specific format (2D, 3D, tensors, etc.)."""
        pass

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Execute model controller in one of three modes: train, finetune, or predict.

        Args:
            step: Pipeline step configuration
            operator: Model operator (unused, model comes from step config)
            dataset: Dataset containing features and targets
            context: Pipeline context with processing state
            runner: Pipeline runner instance
            source: Data source index

        Returns:
            Tuple of (updated_context, binaries_list)
        """
        print(f"🤖 Executing model controller: {self.__class__.__name__}")

        # Extract model configuration from step
        model_config = self._extract_model_config(step)
        train_params = model_config.get('train_params', {})
        finetune_params = model_config.get('finetune_params', {})

        # Determine execution mode
        mode = self._determine_mode(model_config)
        print(f"🎯 Model mode: {mode.value}")

        # Prepare data
        X_train, y_train, X_test, y_test = self._prepare_train_test_data(dataset, context)

        # Execute based on mode
        if mode == ModelMode.FINETUNE and finetune_params:
            return self._execute_finetune(
                model_config, X_train, y_train, X_test, y_test,
                train_params, finetune_params, context, runner
            )
        elif mode == ModelMode.TRAIN:
            return self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner
            )
        else:
            # Default to train mode if no specific mode determined
            return self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner
            )

    def _extract_model_config(self, step: Any) -> Dict[str, Any]:
        """Extract model configuration from step."""
        if isinstance(step, dict):
            if 'model' in step:
                # Step format: {"model": model_obj, "train_params": {...}, "finetune_params": {...}}
                config = step.copy()
                model_obj = step['model']

                # Handle serialized model format
                if isinstance(model_obj, dict) and '_runtime_instance' in model_obj:
                    config['model_instance'] = model_obj['_runtime_instance']
                elif hasattr(model_obj, '_runtime_instance'):
                    config['model_instance'] = model_obj._runtime_instance
                else:
                    config['model_instance'] = model_obj
                return config
            else:
                # Direct model object
                return {'model_instance': step}
        else:
            # Direct model object
            return {'model_instance': step}

    def _determine_mode(self, model_config: Dict[str, Any]) -> ModelMode:
        """Determine execution mode based on configuration."""
        if 'finetune_params' in model_config and model_config['finetune_params']:
            return ModelMode.FINETUNE
        else:
            return ModelMode.TRAIN

    def _prepare_train_test_data(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any]
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare training and test data from dataset.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Get training data
        train_context = context.copy()
        train_context["partition"] = "train"
        X_train = dataset.x(train_context, "2d", concat_source=True)
        y_train = dataset.y(train_context)

        # Get test data (all data for now, will be filtered by folds if needed)
        test_context = context.copy()
        if "partition" in test_context:
            del test_context["partition"]  # Get all data
        X_test = dataset.x(test_context, "2d", concat_source=True)
        y_test = dataset.y(test_context)

        print(f"📊 Data shapes - Train: X{X_train.shape}, y{y_train.shape} | Test: X{X_test.shape}, y{y_test.shape}")

        return X_train, y_train, X_test, y_test

    def _execute_train(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training mode."""
        print("🏋️ Training model...")

        # Get model instance and clone it to avoid modifying original
        base_model = self._get_model_from_config(model_config)
        model = self._clone_model(base_model)

        # Prepare data in framework-specific format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_test_prep, _ = self._prepare_data(X_test, y_test, context)

        # Train model
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep,
            train_params=train_params
        )

        # Generate predictions
        y_pred = self._predict_model(trained_model, X_test_prep)

        # Store results and serialize model
        binaries = self._store_results(trained_model, y_pred, y_test, runner, "trained")

        print("✅ Training completed successfully")
        return context, binaries

    def _execute_finetune(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute fine-tuning mode with Optuna."""
        print("🎛️ Fine-tuning model with Optuna...")

        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available, falling back to training mode")
            return self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner
            )

        # Prepare data
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_test_prep, _ = self._prepare_data(X_test, y_test, context)

        best_model = None
        best_score = float('inf')
        best_params = {}

        def objective(trial):
            nonlocal best_model, best_score, best_params

            # Sample hyperparameters
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Create model with trial parameters - get base model and clone it
            base_model = self._get_model_from_config(model_config)
            model = self._clone_model(base_model)

            # Apply trial parameters to the model if possible
            if hasattr(model, 'set_params') and trial_params:
                try:
                    model.set_params(**trial_params)
                except Exception as e:
                    print(f"⚠️ Could not set parameters {trial_params}: {e}")

            # Train model - use train_params from finetune_params if available, otherwise use top-level train_params
            trial_train_params = finetune_params.get('train_params', train_params)
            trained_model = self._train_model(
                model, X_train_prep, y_train_prep,
                train_params=trial_train_params
            )

            # Evaluate model (use validation split or cross-validation)
            score = self._evaluate_model(trained_model, X_train_prep, y_train_prep)

            # Keep track of best model
            if score < best_score:
                best_score = score
                best_model = trained_model
                best_params = trial_params.copy()

            return score

        # Run optimization with configurable search method
        search_method = finetune_params.get('approach', 'auto')  # Default to auto like base_finetuner
        n_trials = finetune_params.get('n_trials', 10)

        # Auto-determine search method if not specified
        if search_method == 'auto':
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            search_method = 'grid' if is_grid_suitable else 'random'

        # Only use grid search if explicitly requested or auto-determined AND we have categorical params
        # Never use grid search if approach was explicitly set to 'random'
        has_categorical = any(isinstance(v, list) for k, v in finetune_params.items()
                              if k not in ['n_trials', 'approach'])
        use_grid = (search_method == 'grid' and has_categorical and
                    finetune_params.get('approach', 'auto') != 'random')

        if use_grid:
            # Use GridSampler for exhaustive search of categorical parameters only
            import optuna.samplers
            search_space = self._create_grid_search_space(finetune_params)
            if search_space:  # Only use grid search if we have categorical parameters
                sampler = optuna.samplers.GridSampler(search_space)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                # Calculate total combinations for grid search
                n_trials = 1
                for param_values in search_space.values():
                    n_trials *= len(param_values)
                print(f"🔍 Using grid search with {n_trials} combinations")
            else:
                # Fallback to random if no categorical parameters found
                study = optuna.create_study(direction="minimize")
                print(f"🎯 Using random search with {n_trials} trials (no categorical params for grid)")
        else:
            # Use random/TPE sampler for continuous parameters or when explicitly requested
            study = optuna.create_study(direction="minimize")
            print(f"🎯 Using random search with {n_trials} trials")

        study.optimize(objective, n_trials=n_trials)

        print(f"🏆 Best parameters: {best_params}")
        print(f"🎯 Best score: {best_score:.4f}")

        # Retrain best model with top-level train_params (final training parameters)
        if train_params != finetune_params.get('train_params', train_params):
            print("🔄 Retraining best model with final training parameters...")
            base_model = self._get_model_from_config(model_config)
            final_model = self._clone_model(base_model)

            # Apply best parameters to the final model
            if hasattr(final_model, 'set_params') and best_params:
                try:
                    final_model.set_params(**best_params)
                except Exception as e:
                    print(f"⚠️ Could not set best parameters {best_params}: {e}")

            # Train with final train_params
            final_trained_model = self._train_model(
                final_model, X_train_prep, y_train_prep,
                train_params=train_params
            )
            best_model = final_trained_model

        # Generate final predictions with best model
        y_pred = self._predict_model(best_model, X_test_prep)

        # Store results
        binaries = self._store_results(best_model, y_pred, y_test, runner, "finetuned")

        print("✅ Fine-tuning completed successfully")
        return context, binaries

    def _get_model_from_config(self, model_config: Dict[str, Any]) -> Any:
        """Get model instance from configuration."""
        if 'model_instance' in model_config:
            model = model_config['model_instance']
            # If it's still a string or serialized format, try to get the runtime instance
            if isinstance(model, dict) and '_runtime_instance' in model:
                return model['_runtime_instance']
            elif hasattr(model, '_runtime_instance'):
                return model._runtime_instance  # noqa: SLF001
            return model
        else:
            return self._get_model_instance(model_config)

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters for Optuna trial."""
        params = {}

        # Get model parameters to optimize - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

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
                    params[param_name] = trial.suggest_int(param_name, param_config['min'], param_config['max'])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['min'], param_config['max'])

        return params

    def _is_grid_search_suitable(self, finetune_params: Dict[str, Any]) -> bool:
        """Check if grid search is suitable (all parameters are categorical)."""
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

        for param_name, param_config in model_params.items():
            # Only categorical (list) parameters are suitable for grid search
            # Tuple parameters (ranges) need random/TPE sampling
            if not isinstance(param_config, list):
                return False
        return True

    def _create_grid_search_space(self, finetune_params: Dict[str, Any]) -> Dict[str, List]:
        """Create grid search space for categorical parameters only."""
        # Get model parameters to check - support both nested and flat structure
        model_params = finetune_params.get('model_params', {})

        # If no model_params found, look for parameters directly in finetune_params (legacy support)
        if not model_params:
            model_params = {k: v for k, v in finetune_params.items()
                            if k not in ['n_trials', 'approach', 'train_params']}

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

    @abstractmethod
    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate model performance (return score to minimize)."""
        pass

    def _clone_model(self, model: Any) -> Any:
        """Clone a model using framework-specific cloning methods."""
        framework = self._detect_framework(model)

        if framework == 'sklearn':
            from sklearn.base import clone
            return clone(model)
        elif framework == 'tensorflow':
            try:
                from tensorflow.keras.models import clone_model
                return clone_model(model)
            except ImportError:
                pass

        # Fallback to deepcopy
        from copy import deepcopy
        return deepcopy(model)

    def _detect_framework(self, model: Any) -> str:
        """Detect the framework from the model instance."""
        if hasattr(model, 'framework'):
            return model.framework

        import inspect
        if inspect.isclass(model):
            model_desc = f"{model.__module__}.{model.__name__}"
        else:
            model_desc = f"{model.__class__.__module__}.{model.__class__.__name__}"

        if 'tensorflow' in model_desc or 'keras' in model_desc:
            return 'tensorflow'
        elif 'torch' in model_desc:
            return 'pytorch'
        elif 'sklearn' in model_desc:
            return 'sklearn'
        else:
            return 'unknown'

    def _store_results(
        self,
        model: Any,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        runner: 'PipelineRunner',
        model_type: str = "model"
    ) -> List[Tuple[str, bytes]]:
        """Store model and prediction results as binaries."""
        binaries = []

        # Serialize trained model
        try:
            model_binary = pickle.dumps(model)
            model_filename = f"{model_type}_{model.__class__.__name__}_{runner.next_op()}.pkl"
            binaries.append((model_filename, model_binary))
        except Exception as e:
            print(f"⚠️ Could not serialize model: {e}")

        # Store predictions as CSV
        try:
            predictions_csv = "y_true,y_pred\n"
            for true_val, pred_val in zip(y_true.flatten(), y_pred.flatten()):
                predictions_csv += f"{true_val},{pred_val}\n"

            pred_filename = f"predictions_{model_type}_{runner.next_op()}.csv"
            binaries.append((pred_filename, predictions_csv.encode('utf-8')))
        except Exception as e:
            print(f"⚠️ Could not store predictions: {e}")

        return binaries
