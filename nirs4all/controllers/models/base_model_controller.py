"""
Base Model Controller - Abstract controller for machine learning models

This module provides the base abstract controller that handles the three main modes:
- Training: Fit model on training data
- Fine-tuning: Hyperparameter optimization using Optuna
- Prediction: Generate predictions and store results

All model controllers (sklearn, tensorflow, pytorch) inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import pickle
import numpy as np
from enum import Enum
from pathlib import Path

from nirs4all.dataset.helpers import Layout
from nirs4all.controllers.controller import OperatorController
from nirs4all.utils.model_utils import ModelUtils, TaskType

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


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

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Model controllers support prediction mode."""
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

    @abstractmethod
    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for this model type ('2d' or '3d')."""
        pass

    def _detect_task_type(self, y: np.ndarray) -> TaskType:
        """
        Detect task type from target values.

        Args:
            y: Target values array

        Returns:
            TaskType: Detected task type
        """
        return ModelUtils.detect_task_type(y)

    def _configure_loss_and_metrics(
        self,
        model_config: Dict[str, Any],
        task_type: TaskType,
        framework: str = "sklearn"
    ) -> Dict[str, Any]:
        """
        Configure loss function and metrics based on task type.

        Args:
            model_config: Model configuration dictionary
            task_type: Detected task type
            framework: ML framework name

        Returns:
            Dict: Updated model configuration with loss and metrics
        """
        config = model_config.copy()

        # Configure loss function
        if 'loss' not in config.get('train_params', {}):
            if 'train_params' not in config:
                config['train_params'] = {}
            default_loss = ModelUtils.get_default_loss(task_type, framework)
            config['train_params']['loss'] = default_loss
            print(f"📊 Auto-detected {task_type.value} task, using default loss: {default_loss}")
        else:
            # Validate provided loss
            provided_loss = config['train_params']['loss']
            if not ModelUtils.validate_loss_compatibility(provided_loss, task_type, framework):
                print(f"⚠️ Warning: Loss '{provided_loss}' may not be compatible with {task_type.value} task")

        # Configure metrics
        if 'metrics' not in config.get('train_params', {}):
            default_metrics = ModelUtils.get_default_metrics(task_type, framework)
            config['train_params']['metrics'] = default_metrics
            print(f"📈 Using default metrics for {task_type.value}: {default_metrics}")

        return config

    def _calculate_and_print_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        partition: str = "test",
        model_name: str = "model",
        metrics: Optional[List[str]] = None,
        show_detailed_scores: bool = True
    ) -> Dict[str, float]:
        """
        Calculate scores and print them in a formatted way.

        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: Task type for appropriate metrics
            partition: Data partition name (train, test, val, etc.)
            model_name: Model name for display
            metrics: Specific metrics to calculate (None for defaults)
            show_detailed_scores: If True, show all scores; if False, show only best score

        Returns:
            Dict[str, float]: Calculated scores
        """
        scores = ModelUtils.calculate_scores(y_true, y_pred, task_type, metrics)

        if scores:
            best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
            best_score = scores.get(best_metric)

            # Show detailed scores only if requested
            if show_detailed_scores:
                # Detailed score display removed to prevent duplicate output
                pass

            # Always show best score summary
            if best_score is not None:
                direction = "↑" if higher_is_better else "↓"
                if show_detailed_scores:
                    print(f"🏆 Best score ({best_metric}): {best_score:.4f} {direction}")
        else:
            if show_detailed_scores:
                print(f"⚠️ No scores calculated for {model_name} on {partition} set")

        return scores

    def _get_framework_name(self) -> str:
        """Get framework name for this controller."""
        class_name = self.__class__.__name__.lower()
        if 'sklearn' in class_name:
            return "sklearn"
        elif 'tensorflow' in class_name or 'keras' in class_name:
            return "tensorflow"
        elif 'pytorch' in class_name or 'torch' in class_name:
            return "pytorch"
        else:
            return "sklearn"  # Default fallback

    def _get_unique_model_name(self, step: Any, trained_model: Any, runner: 'PipelineRunner') -> str:
        """
        Generate a unique model name that preserves the original function name
        and adds indexing for multiple instances.

        Args:
            step: The pipeline step configuration
            trained_model: The trained model instance
            runner: The pipeline runner (for getting step index)

        Returns:
            str: Unique model name like "nicon_1", "nicon_2", "RandomForestRegressor_1", etc.
        """
        base_name = None

        # Try to get the original function name from the step configuration
        if isinstance(step, dict):
            # Handle direct model configuration
            if 'model' in step:
                model_obj = step['model']
            elif 'model_instance' in step:
                model_obj = step['model_instance']
            else:
                # The step dict itself might be the model configuration
                model_obj = step

            # Handle function objects directly (like nicon, decon)
            if hasattr(model_obj, '__name__'):
                base_name = model_obj.__name__
            # Handle serialized function objects
            elif isinstance(model_obj, dict) and 'function' in model_obj:
                func_path = model_obj['function']
                if '.' in func_path:
                    base_name = func_path.split('.')[-1]  # Get the function name
            # Handle serialized class objects
            elif isinstance(model_obj, dict) and 'class' in model_obj:
                class_path = model_obj['class']
                if '.' in class_path:
                    base_name = class_path.split('.')[-1]  # Get the class name
            # Handle class instances (like RandomForestRegressor, SVR)
            elif hasattr(model_obj, '__class__'):
                base_name = model_obj.__class__.__name__
        elif hasattr(step, '__name__'):
            # Direct function object
            base_name = step.__name__
        elif hasattr(step, '__class__'):
            # Direct class instance
            base_name = step.__class__.__name__

        # Fallback to trained model class name if we couldn't determine the original name
        if not base_name:
            base_name = trained_model.__class__.__name__

        # Get a unique index from the runner's operation counter
        unique_index = runner.next_op()

        # Combine base name with unique index
        unique_name = f"{base_name}_{unique_index}"

        return unique_name

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Execute model controller in one of three modes: train, finetune, or predict.
        Handles both single training and cross-validation with multiple folds.

        Args:
            step: Pipeline step configuration
            operator: Model operator (unused, model comes from step config)
            dataset: Dataset containing features and targets
            context: Pipeline context with processing state
            runner: Pipeline runner instance
            source: Data source index
            mode: Execution mode ("train" or "predict")
            loaded_binaries: Pre-loaded binary objects for prediction mode

        Returns:
            Tuple of (updated_context, binaries_list)
        """
        # print(f"🤖 Executing model controller: {self.__class__.__name__}")

        # In prediction mode, use loaded model for prediction
        if mode == "predict":
            return self._execute_prediction_mode(
                step, operator, dataset, context, runner, loaded_binaries
            )

        # Training/finetuning mode - original logic
        return self._execute_training_mode(
            step, operator, dataset, context, runner
        )

    def _execute_training_mode(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training/finetuning mode."""
        # Extract model configuration and parameters
        # Use the deserialized operator if available, otherwise fall back to step
        model_config = self._extract_model_config(step, operator)
        train_params = model_config.get('train_params', {})
        finetune_params = model_config.get('finetune_params', {})

        # Determine execution mode (train vs finetune)
        mode = self._determine_mode(model_config)

        # Prepare data (can return single tuple or list of tuples for folds)
        data_splits = self._prepare_train_test_data(dataset, context)

        # Check if we have folds (list of tuples) or single split (single tuple)
        if isinstance(data_splits, list):
            # Cross-validation mode: determine finetuning strategy
            if mode == ModelMode.FINETUNE and finetune_params:
                cv_mode = self._determine_cv_mode(finetune_params)

                if cv_mode == CVMode.NESTED:
                    return self._execute_nested_cv(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
                elif cv_mode == CVMode.PER_FOLD:
                    return self._execute_per_fold_cv(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
                else:  # CVMode.SIMPLE or default
                    # First finetune on all training data to get best params
                    return self._execute_simple_cv_finetune(
                        model_config, data_splits, train_params, finetune_params,
                        context, runner, dataset
                    )
            else:
                # Standard cross-validation without finetuning
                return self._execute_cross_validation(
                    model_config, data_splits, train_params, finetune_params,
                    mode, context, runner, dataset
                )
        else:
            # Single training mode: no folds
            X_train, y_train, X_test, y_test = data_splits
            if mode == ModelMode.FINETUNE and finetune_params:
                return self._execute_finetune(
                    model_config, X_train, y_train, X_test, y_test,
                    train_params, finetune_params, context, runner, dataset
                )
            else:
                return self._execute_train(
                    model_config, X_train, y_train, X_test, y_test,
                    train_params, context, runner, dataset
                )

    def _execute_prediction_mode(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[List[Tuple[str, bytes]]]
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute prediction mode using loaded model."""
        if not loaded_binaries:
            raise ValueError("No loaded binaries provided for prediction mode")

        # Find the model binary (look for .pkl files containing trained models)
        model_binary = None
        for name, binary in loaded_binaries:
            if name.endswith('.pkl') and ('model' in name.lower() or 'trained' in name.lower() or 'finetuned' in name.lower()):
                model_binary = binary
                break

        if model_binary is None:
            raise ValueError("No model binary found in loaded binaries")

        # Load the trained model (handle both object and bytes formats)
        import pickle
        if isinstance(model_binary, bytes):
            # Binary data that needs to be unpickled
            trained_model = pickle.loads(model_binary)
        else:
            # Already loaded as an object
            trained_model = model_binary

        # Get the preferred layout for this model type
        layout_str = self.get_preferred_layout()
        layout: Layout = layout_str  # type: ignore

        # Prepare prediction data - use test partition or all data if no test partition
        prediction_context = context.copy()

        # Try to use test partition, fallback to train if test doesn't exist, then to all data
        available_partitions = list(dataset.indexes.keys()) if hasattr(dataset, 'indexes') else []

        if "test" in available_partitions:
            prediction_context["partition"] = "test"
            print(f"🎯 Using 'test' partition for prediction ({sum(dataset.indexes['test'])} samples)")
        elif "train" in available_partitions:
            prediction_context["partition"] = "train"
            print(f"🎯 Using 'train' partition for prediction ({sum(dataset.indexes['train'])} samples)")
        else:
            # Remove partition constraint to use all data
            prediction_context.pop("partition", None)
            print("🎯 Using all available data for prediction")

        X_pred = dataset.x(prediction_context, layout, concat_source=True)

        # Prepare data in framework-specific format (create dummy y for interface compatibility)
        try:
            if hasattr(X_pred, 'shape'):
                dummy_y = np.zeros(X_pred.shape[0])
            elif isinstance(X_pred, list) and X_pred:
                dummy_y = np.zeros(len(X_pred))
            elif isinstance(X_pred, np.ndarray):
                dummy_y = np.zeros(len(X_pred))
            else:
                dummy_y = np.zeros(1)  # fallback
        except Exception:
            dummy_y = np.zeros(1)  # fallback
        X_pred_prep, _ = self._prepare_data(X_pred, dummy_y, context)

        # Generate predictions
        y_pred = self._predict_model(trained_model, X_pred_prep)

        # Store predictions in dataset if y data is available for comparison
        try:
            y_true = dataset.y(prediction_context)
            if y_true is not None and len(y_true) > 0:
                # Store predictions for comparison
                unique_model_name = self._get_unique_model_name(step, trained_model, runner)
                self._store_predictions_in_dataset(
                    dataset=getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown',
                    pipeline=getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown',
                    model=unique_model_name,
                    partition="prediction",
                    y_true=y_true,
                    y_pred=y_pred,
                    fold_idx=None,
                    context=context,
                    dataset_obj=dataset
                )
        except Exception:
            # No y data available for prediction, that's okay
            pass

        # Store predictions as binary output
        binaries = []
        try:
            predictions_csv = "y_pred\n"
            for pred_val in y_pred.flatten():
                predictions_csv += f"{pred_val}\n"
            pred_filename = f"predictions_predict_{runner.next_op()}.csv"
            binaries.append((pred_filename, predictions_csv.encode('utf-8')))
        except Exception as e:
            print(f"⚠️ Could not store predictions: {e}")

        return context, binaries

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from step."""
        # If we have a deserialized operator, use it directly
        if operator is not None:
            if isinstance(step, dict):
                # Step format: {"model": model_obj, "train_params": {...}, "finetune_params": {...}}
                config = step.copy()
                config['model_instance'] = operator
                return config
            else:
                # Direct model object
                return {'model_instance': operator}

        # Fall back to original logic for backward compatibility
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

    def _determine_cv_mode(self, finetune_params: Dict[str, Any]) -> CVMode:
        """Determine the cross-validation finetuning mode from parameters."""
        cv_mode_str = finetune_params.get('cv_mode', 'simple')

        # Check for nested CV indicators
        if (finetune_params.get('inner_cv') is not None or
            finetune_params.get('outer_cv') is not None or
            cv_mode_str == 'nested'):
            return CVMode.NESTED
        elif cv_mode_str == 'per_fold':
            return CVMode.PER_FOLD
        else:
            return CVMode.SIMPLE

    def _create_inner_folds(self, X_train: Any, y_train: Any, inner_cv: Any) -> List[Tuple[Any, Any, Any, Any]]:
        """Create inner cross-validation folds for nested CV."""
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError:
            raise ImportError("scikit-learn is required for nested cross-validation")

        # Default to 3-fold if not specified
        if inner_cv is None:
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        elif isinstance(inner_cv, int):
            inner_cv = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

        inner_folds = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            # Use numpy indexing if possible
            if hasattr(X_train, '__getitem__') and hasattr(X_train, 'shape'):
                X_inner_train = X_train[train_idx]
                X_inner_val = X_train[val_idx]
            else:
                X_inner_train = X_train
                X_inner_val = X_train

            if hasattr(y_train, '__getitem__') and hasattr(y_train, 'shape'):
                y_inner_train = y_train[train_idx]
                y_inner_val = y_train[val_idx]
            else:
                y_inner_train = y_train
                y_inner_val = y_train

            inner_folds.append((X_inner_train, y_inner_train, X_inner_val, y_inner_val))

        return inner_folds

    def _execute_simple_cv_finetune(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute simple CV: finetune on full training data, then train on folds."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))

        if verbose > 0:
            print("🔍 Simple CV: Finetuning on full training data...")

        # Combine all training data for finetuning
        all_X_train = []
        all_y_train = []
        for X_train, y_train, _, _ in data_splits:
            all_X_train.append(X_train)
            all_y_train.append(y_train)

        # Concatenate all training data
        import numpy as np
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)

        # Use first fold's test data for finetuning evaluation (could be improved)
        _, _, X_test_sample, y_test_sample = data_splits[0]

        # Execute finetuning to get best parameters
        _, finetune_binaries = self._execute_finetune(
            model_config, combined_X_train, combined_y_train,
            X_test_sample, y_test_sample, train_params, finetune_params,
            context, runner, dataset
        )

        # Extract best parameters from the finetuning process
        best_params = getattr(self, '_last_best_params', {})

        if verbose > 0:
            print(f"🏆 Best parameters found: {best_params}")
            print(f"🔄 Training {len(data_splits)} fold models with best parameters...")

        # Now train models on each fold with the best parameters
        all_binaries = []
        all_binaries.extend(finetune_binaries)

        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
            # Create model with best parameters
            base_model = self._get_model_from_config(model_config)
            model = self._clone_model(base_model)

            # Apply best parameters
            if hasattr(model, 'set_params') and best_params:
                try:
                    model.set_params(**best_params)
                except Exception as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply best parameters to fold {fold_idx+1}: {e}")

            # Train on this fold
            fold_context, fold_binaries = self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner, dataset, fold_idx
            )

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_simple_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_simple_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        if verbose > 0:
            print("✅ Simple CV completed successfully")

        return context, all_binaries

    def _execute_per_fold_cv(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute per-fold CV: finetune on each fold individually."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))
        param_strategy = ParamStrategy(finetune_params.get('param_strategy', 'per_fold_best'))

        # Handle global average strategy - optimize across all folds simultaneously
        if param_strategy == ParamStrategy.GLOBAL_AVERAGE:
            return self._execute_global_average_optimization(
                model_config, data_splits, train_params, finetune_params,
                context, runner, dataset
            )

        if verbose > 0:
            print(f"🔍 Per-fold CV: Finetuning on each fold with {param_strategy.value} strategy...")

        all_binaries = []
        all_best_params = []

        # Finetune on each fold
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
            if verbose > 0:
                print(f"🎛️ Finetuning fold {fold_idx+1}/{len(data_splits)}...")

            # Execute finetuning for this fold
            fold_context, fold_binaries = self._execute_finetune(
                model_config, X_train, y_train, X_test, y_test,
                train_params, finetune_params, context, runner, dataset, fold_idx
            )

            # Store best parameters for this fold
            fold_best_params = getattr(self, '_last_best_params', {})
            all_best_params.append(fold_best_params)

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_per_fold_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_per_fold_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        # Handle parameter aggregation strategy
        if param_strategy == ParamStrategy.GLOBAL_BEST:
            # Use the best performing parameters across all folds
            global_best_params = self._select_global_best_params(all_best_params, data_splits)
            if verbose > 0:
                print(f"🏆 Global best parameters: {global_best_params}")

            # Check if we should train a single model on full training data
            use_full_train = finetune_params.get('use_full_train_for_final', False)
            if use_full_train:
                return self._train_single_model_on_full_data(
                    model_config, data_splits, global_best_params, train_params,
                    context, runner, dataset, "global_best", verbose
                )

        elif param_strategy == ParamStrategy.GLOBAL_AVERAGE:
            # This case is handled before the fold loop above
            pass
        elif param_strategy in [ParamStrategy.ENSEMBLE_BEST, ParamStrategy.ROBUST_BEST, ParamStrategy.STABILITY_BEST]:
            # These strategies are planned for future implementation
            if verbose > 0:
                print(f"⚠️ Parameter strategy {param_strategy.value} is not yet implemented. Using per_fold_best instead.")

        # For PER_FOLD_BEST, check if we should train on full data (though this is less common)
        use_full_train = finetune_params.get('use_full_train_for_final', False)
        if use_full_train and param_strategy == ParamStrategy.PER_FOLD_BEST:
            # Use the first fold's parameters as representative (or could average them)
            representative_params = all_best_params[0] if all_best_params else {}
            if verbose > 0:
                print(f"🔄 Training single model on full data with representative parameters from fold 1")
            return self._train_single_model_on_full_data(
                model_config, data_splits, representative_params, train_params,
                context, runner, dataset, "per_fold_repr", verbose
            )

        if verbose > 0:
            print("✅ Per-fold CV completed successfully")

        # Compute average and weighted average predictions
        self._compute_aggregate_predictions(runner, dataset, verbose)

        return context, all_binaries

    def _compute_aggregate_predictions(
        self,
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        verbose: int = 0
    ) -> None:
        """
        Compute average and weighted average predictions from fold results.

        Args:
            runner: Pipeline runner instance
            dataset: Dataset object containing predictions
            verbose: Verbosity level
        """
        if not hasattr(dataset, '_predictions') or dataset._predictions is None:
            return

        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'

        # Find all unique models in current predictions
        models = set()
        for pred_data in dataset._predictions._predictions.values():
            if (pred_data.get('dataset') == dataset_name
                and pred_data.get('pipeline') == pipeline_name):
                models.add(pred_data['model'])

        if verbose > 0:
            print(f"🧮 Computing aggregate predictions for {len(models)} models...")

        for model in models:
            # Compute average predictions for test folds
            avg_result = dataset._predictions.calculate_average_predictions(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=model,
                partition_pattern="test_fold",
                store_result=True
            )

            if avg_result and verbose > 0:
                print(f"  ✅ Computed average test predictions for {model}")

            # Compute weighted average predictions for test folds
            weighted_result = dataset._predictions.calculate_weighted_average_predictions(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=model,
                test_partition_pattern="test_fold",
                val_partition_pattern="val_fold",
                metric='rmse',
                store_result=True
            )

            if weighted_result and verbose > 0:
                print(f"  ✅ Computed weighted average test predictions for {model}")
                weights = weighted_result['metadata'].get('weights', [])
                if weights:
                    print(f"    Fold weights: {[f'{w:.3f}' for w in weights]}")

    def _train_single_model_on_full_data(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        best_params: Dict[str, Any],
        train_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        model_suffix: str = "full_train",
        verbose: int = 0
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Train a single model on the full training dataset using optimized parameters.

        Instead of training separate models on each fold, this combines all training data
        and trains one model, which can be more effective when you have limited data
        but still want the benefits of rigorous hyperparameter optimization.

        Args:
            model_config: Model configuration
            data_splits: List of (X_train, y_train, X_test, y_test) tuples from folds
            best_params: Optimized parameters to apply to the model
            train_params: Training parameters
            context: Pipeline context
            runner: Pipeline runner instance
            dataset: Dataset object
            model_suffix: Suffix for model naming
            verbose: Verbosity level

        Returns:
            Tuple of (context, binaries_list)
        """
        if verbose > 0:
            print(f"🎯 Training single model on full training data ({model_suffix})...")

        # Combine all training data from folds
        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []

        for X_train, y_train, X_test, y_test in data_splits:
            all_X_train.append(X_train)
            all_y_train.append(y_train)
            all_X_test.append(X_test)
            all_y_test.append(y_test)

        # Concatenate all data
        import numpy as np
        combined_X_train = np.concatenate(all_X_train, axis=0)
        combined_y_train = np.concatenate(all_y_train, axis=0)
        combined_X_test = np.concatenate(all_X_test, axis=0)
        combined_y_test = np.concatenate(all_y_test, axis=0)

        if verbose > 0:
            print(f"📊 Combined training data: {combined_X_train.shape[0]} samples")
            print(f"📊 Combined test data: {combined_X_test.shape[0]} samples")

        # Create and configure model with best parameters
        base_model = self._get_model_from_config(model_config)
        model = self._clone_model(base_model)

        if hasattr(model, 'set_params') and best_params:
            try:
                model.set_params(**best_params)
                if verbose > 0:
                    print(f"✅ Applied optimized parameters: {best_params}")
            except Exception as e:
                if verbose > 0:
                    print(f"⚠️ Could not apply parameters: {e}")

        # Prepare data in framework-specific format
        X_train_prep, y_train_prep = self._prepare_data(combined_X_train, combined_y_train, context)
        X_test_prep, _ = self._prepare_data(combined_X_test, combined_y_test, context)

        if verbose > 0:
            print(f"🏋️ Training model with {X_train_prep.shape[0]} samples...")

        # Train the model
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep, train_params=train_params
        )

        # Generate predictions on combined test set
        y_pred_test = self._predict_model(trained_model, X_test_prep)

        # Generate predictions on combined training set (global train predictions)
        y_pred_train = self._predict_model(trained_model, X_train_prep)

        # Store predictions in dataset
        unique_model_name = self._get_unique_model_name(model_config, trained_model, runner)
        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'

        # Store test predictions
        self._store_predictions_in_dataset(
            dataset=dataset_name,
            pipeline=pipeline_name,
            model=unique_model_name,
            partition=f"test_{model_suffix}",
            y_true=combined_y_test,
            y_pred=y_pred_test,
            fold_idx=None,  # No fold since trained on full data
            context=context,
            dataset_obj=dataset
        )

        # Store global train predictions
        self._store_predictions_in_dataset(
            dataset=dataset_name,
            pipeline=pipeline_name,
            model=unique_model_name,
            partition=f"global_train_{model_suffix}",
            y_true=combined_y_train,
            y_pred=y_pred_train,
            fold_idx=None,
            context=context,
            dataset_obj=dataset
        )

        # Store results and serialize model
        binaries = self._store_results(trained_model, y_pred_test, combined_y_test, runner, f"{model_suffix}_model")

        if verbose > 0:
            print("✅ Single model training on full data completed successfully")

        return context, binaries

    def _execute_global_average_optimization(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute global average optimization: optimize parameters across all folds simultaneously."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))

        if verbose > 0:
            print(f"🌍 Global Average CV: Optimizing parameters across all {len(data_splits)} folds simultaneously...")

        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for global average parameter optimization")

        # Configure Optuna logging
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_params = {}
        best_score = float('inf')

        def objective(trial):
            nonlocal best_params, best_score

            # Sample hyperparameters for this trial
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Evaluate these parameters on all folds and average the scores
            fold_scores = []
            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
                # Create and configure model for this fold
                base_model = self._get_model_from_config(model_config)
                model = self._clone_model(base_model)

                if hasattr(model, 'set_params') and trial_params:
                    try:
                        model.set_params(**trial_params)
                    except Exception:
                        return float('inf')  # Invalid parameters

                # Prepare data and train
                X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
                X_test_prep, y_test_prep = self._prepare_data(X_test, y_test, context)

                # Use silent training for optimization
                fold_train_params = finetune_params.get('train_params', train_params.copy())
                fold_train_params['verbose'] = 0

                trained_model = self._train_model(
                    model, X_train_prep, y_train_prep, train_params=fold_train_params
                )

                # Evaluate on test set
                score = self._evaluate_model(trained_model, X_test_prep, y_test_prep)
                fold_scores.append(score)

            # Calculate average score across all folds
            avg_score = np.mean(fold_scores)

            # Track best parameters
            if avg_score < best_score:
                best_score = avg_score
                best_params = trial_params.copy()

            return avg_score

        # Run optimization
        study = optuna.create_study(direction="minimize")
        n_trials = finetune_params.get('n_trials', 10)

        if verbose > 0:
            print(f"🎯 Optimizing with {n_trials} trials, evaluating each on all {len(data_splits)} folds...")

        study.optimize(objective, n_trials=n_trials)

        if verbose > 0:
            print(f"🏆 Global best parameters: {best_params}")
            print(f"📊 Best average score: {best_score:.4f}")

        # Store best parameters for potential reuse
        self._last_best_params = best_params

        # Check if we should train on full training data or individual folds
        use_full_train = finetune_params.get('use_full_train_for_final', False)

        if use_full_train:
            return self._train_single_model_on_full_data(
                model_config, data_splits, best_params, train_params,
                context, runner, dataset, "global_avg", verbose
            )

        # Default behavior: train final models on each fold using the globally optimal parameters
        if verbose > 0:
            print(f"🔄 Training {len(data_splits)} final models with global best parameters...")

        all_binaries = []
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(data_splits):
            # Create model with global best parameters
            base_model = self._get_model_from_config(model_config)
            model = self._clone_model(base_model)

            # Apply global best parameters
            if hasattr(model, 'set_params') and best_params:
                try:
                    model.set_params(**best_params)
                except Exception as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply global parameters to fold {fold_idx+1}: {e}")

            # Train final model for this fold
            fold_context, fold_binaries = self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner, dataset, fold_idx
            )

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_global_avg_cv_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_global_avg_cv_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        if verbose > 0:
            print("✅ Global Average CV completed successfully")

        return context, all_binaries

    def _execute_nested_cv(
        self,
        model_config: Dict[str, Any],
        outer_folds: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute nested CV: inner folds for finetuning, outer folds for training."""
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))
        param_strategy = ParamStrategy(finetune_params.get('param_strategy', 'per_fold_best'))
        inner_cv = finetune_params.get('inner_cv', 3)

        if verbose > 0:
            print(f"🔍 Nested CV: {len(outer_folds)} outer folds with inner CV finetuning...")
            print(f"📊 Parameter strategy: {param_strategy.value}")

        all_binaries = []
        all_fold_results = []

        for outer_idx, (X_outer_train, y_outer_train, X_outer_test, y_outer_test) in enumerate(outer_folds):
            if verbose > 0:
                print(f"🏋️ Outer fold {outer_idx+1}/{len(outer_folds)}...")

            # Create inner folds for finetuning
            inner_folds = self._create_inner_folds(X_outer_train, y_outer_train, inner_cv)

            if verbose > 1:
                print(f"  📋 Created {len(inner_folds)} inner folds for finetuning")

            # Choose optimization strategy
            if param_strategy == ParamStrategy.GLOBAL_AVERAGE:
                # Optimize using global average across inner folds
                fold_best_params = self._optimize_global_average_on_inner_folds(
                    model_config, inner_folds, train_params, finetune_params, context, verbose
                )
            else:
                # Standard nested CV: finetune using inner folds
                fold_best_params = self._finetune_on_inner_folds(
                    model_config, inner_folds, train_params, finetune_params, context, verbose
                )

            if verbose > 1:
                print(f"  🏆 Best params for outer fold {outer_idx+1}: {fold_best_params}")

            # Train final model on full outer training data with best parameters
            base_model = self._get_model_from_config(model_config)
            final_model = self._clone_model(base_model)

            # Apply best parameters
            if hasattr(final_model, 'set_params') and fold_best_params:
                try:
                    final_model.set_params(**fold_best_params)
                except Exception as e:
                    if verbose > 0:
                        print(f"⚠️ Could not apply parameters to outer fold {outer_idx+1}: {e}")

            # Prepare data and train
            X_train_prep, y_train_prep = self._prepare_data(X_outer_train, y_outer_train, context)
            X_test_prep, _ = self._prepare_data(X_outer_test, y_outer_test, context)

            trained_model = self._train_model(
                final_model, X_train_prep, y_train_prep, train_params=train_params
            )

            # Generate predictions
            y_pred = self._predict_model(trained_model, X_test_prep)

            # Store results for this fold
            fold_binaries = self._store_results(
                trained_model, y_pred, y_outer_test, runner, f"nested_cv_outer_fold{outer_idx+1}"
            )

            # Store fold results for potential parameter aggregation
            fold_score = self._evaluate_model(trained_model, X_test_prep, y_outer_test)
            all_fold_results.append({
                'fold_idx': outer_idx,
                'best_params': fold_best_params,
                'score': fold_score,
                'binaries': fold_binaries
            })

            all_binaries.extend(fold_binaries)

        # Handle parameter aggregation across outer folds
        if param_strategy == ParamStrategy.WEIGHTED_AVERAGE:
            self._compute_weighted_average_params(all_fold_results, verbose)

        # Check if we should train a single model on full training data
        use_full_train = finetune_params.get('use_full_train_for_final', False)
        if use_full_train:
            if verbose > 0:
                print("🎯 Training single model on full training data with nested CV optimized parameters...")

            # Use the best parameters from the first outer fold as representative
            # (In practice, you might want to average parameters across outer folds)
            representative_params = all_fold_results[0]['best_params'] if all_fold_results else {}

            return self._train_single_model_on_full_data(
                model_config, outer_folds, representative_params, train_params,
                context, runner, dataset, "nested_cv_full", verbose
            )

        if verbose > 0:
            print("✅ Nested CV completed successfully")

        return context, all_binaries

    def _finetune_on_inner_folds(
        self,
        model_config: Dict[str, Any],
        inner_folds: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Finetune using inner folds and return best parameters."""
        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available for nested CV")
            return {}

        best_params = {}
        best_score = float('inf')

        def objective(trial):
            nonlocal best_params, best_score

            # Sample hyperparameters
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Cross-validate on inner folds
            fold_scores = []
            for X_train, y_train, X_val, y_val in inner_folds:
                # Create and configure model
                base_model = self._get_model_from_config(model_config)
                model = self._clone_model(base_model)

                if hasattr(model, 'set_params') and trial_params:
                    try:
                        model.set_params(**trial_params)
                    except Exception:
                        pass

                # Prepare data and train
                X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
                X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context)

                # Use silent training for inner CV
                inner_train_params = finetune_params.get('train_params', train_params.copy())
                inner_train_params['verbose'] = 0

                trained_model = self._train_model(
                    model, X_train_prep, y_train_prep, train_params=inner_train_params
                )

                # Evaluate
                score = self._evaluate_model(trained_model, X_val_prep, y_val_prep)
                fold_scores.append(score)

            # Average score across inner folds
            avg_score = np.mean(fold_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = trial_params.copy()

            return avg_score

        # Configure and run optimization
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        n_trials = finetune_params.get('n_trials', 10)

        if verbose > 2:
            print(f"    🎯 Running {n_trials} inner CV trials...")

        study.optimize(objective, n_trials=n_trials)

        return best_params

    def _optimize_global_average_on_inner_folds(
        self,
        model_config: Dict[str, Any],
        inner_folds: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        context: Dict[str, Any],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Optimize using global average across inner folds (for nested CV)."""
        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available for nested CV")
            return {}

        if verbose > 2:
            print(f"    🌍 Global average optimization across {len(inner_folds)} inner folds")

        best_params = {}
        best_score = float('inf')

        def objective(trial):
            nonlocal best_params, best_score

            # Sample hyperparameters
            trial_params = self._sample_hyperparameters(trial, finetune_params)

            # Evaluate on all inner folds and average
            fold_scores = []
            for X_train, y_train, X_val, y_val in inner_folds:
                # Create and configure model
                base_model = self._get_model_from_config(model_config)
                model = self._clone_model(base_model)

                if hasattr(model, 'set_params') and trial_params:
                    try:
                        model.set_params(**trial_params)
                    except Exception:
                        return float('inf')

                # Prepare data and train
                X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
                X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context)

                # Use silent training for inner CV
                inner_train_params = finetune_params.get('train_params', train_params.copy())
                inner_train_params['verbose'] = 0

                trained_model = self._train_model(
                    model, X_train_prep, y_train_prep, train_params=inner_train_params
                )

                # Evaluate
                score = self._evaluate_model(trained_model, X_val_prep, y_val_prep)
                fold_scores.append(score)

            # Average score across inner folds
            avg_score = np.mean(fold_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = trial_params.copy()

            return avg_score

        # Configure and run optimization
        if verbose < 3:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        n_trials = finetune_params.get('n_trials', 10)

        if verbose > 2:
            print(f"    🎯 Running {n_trials} inner CV trials with global averaging...")

        study.optimize(objective, n_trials=n_trials)

        return best_params

    def _select_global_best_params(
        self,
        all_best_params: List[Dict[str, Any]],
        data_splits: List[Tuple[Any, Any, Any, Any]]
    ) -> Dict[str, Any]:
        """Select the globally best parameters from all folds."""
        # For now, return the first set of parameters
        # This could be improved by tracking scores and selecting the best performing set
        if all_best_params:
            return all_best_params[0]
        return {}

    def _compute_weighted_average_params(
        self,
        fold_results: List[Dict[str, Any]],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Compute weighted average parameters based on fold performance."""
        if verbose > 0:
            print("📊 Computing weighted average parameters...")

        # For numerical parameters, compute weighted average
        # For categorical parameters, select most frequent
        # This is a placeholder implementation
        return {}

    def _prepare_train_test_data(
        self,
        dataset: 'SpectroDataset',
        context: Dict[str, Any]
    ) -> Union[Tuple[Any, Any, Any, Any], List[Tuple[Any, Any, Any, Any]]]:
        """
        Prepare training and test data from dataset, handling cross-validation folds.

        If dataset has folds, returns a list of (X_train, y_train, X_val, y_val) tuples, one per fold.
        If no folds, returns a single tuple (X_train, y_train, X_test, y_test).

        Returns:
            Union[Tuple[Any, Any, Any, Any], List[Tuple[Any, Any, Any, Any]]]
        """
        # Get the preferred layout for this model type
        layout_str = self.get_preferred_layout()
        layout: Layout = layout_str  # type: ignore

        # Check if dataset has folds
        if hasattr(dataset, 'num_folds') and dataset.num_folds > 0:
            # Prepare fold-based train/validation splits
            folds_data = []

            # Get all training data first
            train_context = context.copy()
            train_context["partition"] = "train"
            X_all_train = dataset.x(train_context, layout, concat_source=True)
            y_all_train = dataset.y(train_context)

            # For each fold, create train/validation splits
            for fold_idx, (train_indices, val_indices) in enumerate(dataset.folds):
                X_train_fold = X_all_train[train_indices]
                y_train_fold = y_all_train[train_indices]
                X_val_fold = X_all_train[val_indices]
                y_val_fold = y_all_train[val_indices]

                folds_data.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
                # print(f"📊 Fold {fold_idx+1}: Train {X_train_fold.shape}, Val {X_val_fold.shape}")

            return folds_data
        else:
            # No folds: use standard train/test split
            train_context = context.copy()
            train_context["partition"] = "train"
            X_train = dataset.x(train_context, layout, concat_source=True)
            y_train = dataset.y(train_context)

            test_context = context.copy()
            test_context["partition"] = "test"
            X_test = dataset.x(test_context, layout, concat_source=True)
            y_test = dataset.y(test_context)

            # print(f"📊 No folds - Train: X{X_train.shape}, y{y_train.shape} | Test: X{X_test.shape}, y{y_test.shape}")
            return X_train, y_train, X_test, y_test

    def _execute_cross_validation(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        train_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        mode: ModelMode,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute cross-validation training, creating one model per fold."""
        verbose = train_params.get('verbose', 0)
        all_binaries = []

        # if verbose > 0:
        #     print(f"🔄 Cross-validation: training {len(data_splits)} fold models...")

        # Get test data for final evaluation (after training all folds)
        test_context = context.copy()
        test_context["partition"] = "test"
        layout_str = self.get_preferred_layout()
        layout: Layout = layout_str  # type: ignore
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)

        for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(data_splits):
            # if verbose > 0:
            #     print(f"🏋️ Training fold {fold_idx+1}/{len(data_splits)}...")

            if mode == ModelMode.FINETUNE and finetune_params:
                # Fine-tune for this fold
                fold_context, fold_binaries = self._execute_finetune(
                    model_config, X_train, y_train, X_val, y_val,
                    train_params, finetune_params, context, runner, dataset, fold_idx
                )
            else:
                # Train for this fold
                fold_context, fold_binaries = self._execute_train(
                    model_config, X_train, y_train, X_val, y_val,
                    train_params, context, runner, dataset, fold_idx
                )

            # Store test predictions for each fold model
            # This is now handled within _execute_train when fold_idx is provided

            # Add fold suffix to binary names
            fold_binaries_renamed = []
            for name, binary in fold_binaries:
                name_parts = name.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_fold{fold_idx+1}.{name_parts[1]}"
                else:
                    new_name = f"{name}_fold{fold_idx+1}"
                fold_binaries_renamed.append((new_name, binary))

            all_binaries.extend(fold_binaries_renamed)

        # After all folds are complete, automatically generate aggregated predictions
        self._generate_aggregated_predictions(
            model_config, data_splits, context, runner, dataset
        )

        # if verbose > 0:
        #     print("✅ Cross-validation completed successfully")

        return context, all_binaries

    def _generate_aggregated_predictions(
        self,
        model_config: Dict[str, Any],
        data_splits: List[Tuple[Any, Any, Any, Any]],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> None:
        """Generate aggregated predictions (mean and weighted mean) for all fold predictions."""
        # Get base model name without suffix
        base_model = self._get_model_from_config(model_config)
        base_model_name = base_model.__class__.__name__

        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'

        # Generate average predictions for train, val, and test
        for partition_pattern in ['train_fold', 'val_fold', 'test_fold']:
            # Calculate simple average
            avg_result = dataset._predictions.calculate_average_predictions(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=base_model_name,  # Use base model name
                partition_pattern=partition_pattern,
                store_result=True
            )

            if avg_result:
                print(f"✨ Generated averaged {partition_pattern} predictions")

        # Generate weighted average test predictions based on validation performance
        weighted_avg_result = dataset._predictions.calculate_weighted_average_predictions(
            dataset=dataset_name,
            pipeline=pipeline_name,
            model=base_model_name,  # Use base model name
            test_partition_pattern="test_fold",
            val_partition_pattern="val_fold",
            metric='rmse',  # Use RMSE for weighting
            store_result=True
        )

        if weighted_avg_result:
            print("✨ Generated weighted averaged test predictions")

    def _execute_train(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        train_params: Dict[str, Any],
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute training mode."""
        verbose = train_params.get('verbose', 0)

        # if verbose > 0:
            # print("🏋️ Training model...")

        # Get model instance and clone it to avoid modifying original
        base_model = self._get_model_from_config(model_config)
        model = self._clone_model(base_model)

        # Prepare data in framework-specific format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context)
        X_test_prep, _ = self._prepare_data(X_test, y_test, context)

        # print(X_train_prep.shape, y_train_prep.shape, X_test_prep.shape, y_test.shape)

        # Train model
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep,
            train_params=train_params
        )

        # Generate predictions for test/validation set
        y_pred_test = self._predict_model(trained_model, X_test_prep)

        # Generate predictions for training set
        y_pred_train = self._predict_model(trained_model, X_train_prep)

        # Store predictions in dataset
        unique_model_name = self._get_unique_model_name(model_config, model, runner)
        dataset_name = getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown'
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown'

        # Store test/validation predictions (X_test is actually validation data in fold context)
        if fold_idx is not None:
            # This is a fold, so X_test is validation data
            self._store_predictions_in_dataset(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=unique_model_name,
                partition=f"val_fold_{fold_idx}",
                y_true=y_test,
                y_pred=y_pred_test,
                fold_idx=fold_idx,
                context=context,
                dataset_obj=dataset
            )

            # Store training predictions
            self._store_predictions_in_dataset(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=unique_model_name,
                partition=f"train_fold_{fold_idx}",
                y_true=y_train,
                y_pred=y_pred_train,
                fold_idx=fold_idx,
                context=context,
                dataset_obj=dataset
            )

            # Also generate test predictions for this fold model on global test data
            test_context = context.copy()
            test_context["partition"] = "test"
            layout_str = self.get_preferred_layout()
            layout: Layout = layout_str  # type: ignore
            X_global_test = dataset.x(test_context, layout, concat_source=True)
            y_global_test = dataset.y(test_context)

            # Prepare global test data and predict
            X_global_test_prep, _ = self._prepare_data(X_global_test, y_global_test, context)
            y_pred_global_test = self._predict_model(trained_model, X_global_test_prep)

            # Store test predictions for this fold
            self._store_predictions_in_dataset(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=unique_model_name,
                partition=f"test_fold_{fold_idx}",
                y_true=y_global_test,
                y_pred=y_pred_global_test,
                fold_idx=fold_idx,
                context=context,
                dataset_obj=dataset
            )
        else:
            # This is global test data
            self._store_predictions_in_dataset(
                dataset=dataset_name,
                pipeline=pipeline_name,
                model=unique_model_name,
                partition="test",
                y_true=y_test,
                y_pred=y_pred_test,
                fold_idx=fold_idx,
                context=context,
                dataset_obj=dataset
            )

        # Store results and serialize model
        binaries = self._store_results(trained_model, y_pred_test, y_test, runner, "trained")

        # Always calculate and display final test scores (even at verbose=0)
        task_type = self._detect_task_type(y_train)
        test_scores = self._calculate_and_print_scores(
            y_test, y_pred_test, task_type, "test", unique_model_name,
            show_detailed_scores=False  # Don't show detailed scores, just calculate them
        )

        # Display concise final summary with best score (1 line)
        best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
        best_score = test_scores.get(best_metric)
        if best_score is not None:
            direction = "↑" if higher_is_better else "↓"
            # Calculate scaled score if possible
            scaled_score = self._calculate_scaled_score(y_test, y_pred_test, dataset)

            # Format: metric=value (scaled_value)↓ (other scores)
            if scaled_score is not None and scaled_score != best_score:
                score_display = f"{best_metric}={best_score:.4f} ({scaled_score:.4f}){direction}"
            else:
                score_display = f"{best_metric}={best_score:.4f}{direction}"

            # Add other scores if available
            other_scores = {k: v for k, v in test_scores.items() if k != best_metric}
            if other_scores:
                other_scores_str = ModelUtils.format_scores(other_scores)
                print(f"✅ {unique_model_name} - test: {score_display} ({other_scores_str})")
            else:
                print(f"✅ {unique_model_name} - test: {score_display}")
        else:
            print(f"✅ Model {unique_model_name} completed successfully")

        # Save predictions to dataset results folder
        self._save_predictions_to_results_folder(dataset, runner)

        # if verbose > 0:
        #     print("✅ Training completed successfully")
        return context, binaries

    def _calculate_scaled_score(self, y_true, y_pred, dataset) -> Optional[float]:
        """Calculate score - just return None to disable the (scaled) display."""
        # The user is right - the scaling was meaningless and confusing
        # Just return None so the display shows: mse=0.0149↓ without (scaled_value)
        return None

    def _save_predictions_to_results_folder(self, dataset: 'SpectroDataset', runner: 'PipelineRunner') -> None:
        """Save predictions to results folder above pipeline config folder."""
        try:
            if hasattr(runner, 'saver') and hasattr(runner.saver, 'current_path'):
                # Get the base results folder
                base_path = Path(runner.saver.current_path)
                dataset_name = getattr(runner.saver, 'dataset_name', dataset.name)
                predictions_file = base_path / f"{dataset_name}_predictions.json"
                dataset._predictions.save_to_file(str(predictions_file))
        except Exception as e:
            print(f"⚠️ Could not save predictions to file: {e}")

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
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset',
        fold_idx: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute fine-tuning mode with Optuna."""
        # Check verbose setting from finetune_params or train_params (0=silent, 1=basic, 2=detailed)
        verbose = finetune_params.get('verbose', train_params.get('verbose', 0))

        # if verbose > 0:
            # print("🎛️ Fine-tuning model with Optuna...")

        try:
            import optuna
        except ImportError:
            print("⚠️ Optuna not available, falling back to training mode")
            return self._execute_train(
                model_config, X_train, y_train, X_test, y_test,
                train_params, context, runner, dataset
            )        # Prepare data
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
            trial_train_params = finetune_params.get('train_params', train_params.copy())
            # Ensure training is silent during trials unless explicitly verbose
            if 'verbose' not in trial_train_params:
                trial_train_params['verbose'] = 0  # Silent during trials

            trained_model = self._train_model(
                model, X_train_prep, y_train_prep,
                train_params=trial_train_params
            )            # Evaluate model (use validation split or cross-validation)
            score = self._evaluate_model(trained_model, X_train_prep, y_train_prep)

            # Keep track of best model
            if score < best_score:
                best_score = score
                best_model = trained_model
                best_params = trial_params.copy()

            return score

        # Configure Optuna logging
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization with configurable search method
        search_method = finetune_params.get('approach', 'auto')
        n_trials = finetune_params.get('n_trials', 10)

        # Auto-determine search method if not specified
        if search_method == 'auto':
            is_grid_suitable = self._is_grid_search_suitable(finetune_params)
            search_method = 'grid' if is_grid_suitable else 'random'

        # Only use grid search if explicitly requested or auto-determined AND we have categorical params
        has_categorical = any(isinstance(v, list) for k, v in finetune_params.items()
                              if k not in ['n_trials', 'approach'])
        use_grid = (search_method == 'grid' and has_categorical and
                    finetune_params.get('approach', 'auto') != 'random')

        if use_grid:
            # Use GridSampler for exhaustive search
            import optuna.samplers
            search_space = self._create_grid_search_space(finetune_params)
            if search_space:
                sampler = optuna.samplers.GridSampler(search_space)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                # Calculate total combinations for grid search
                n_trials = 1
                for param_values in search_space.values():
                    n_trials *= len(param_values)
                # if verbose:
                    # print(f"🔍 Starting grid search with {n_trials} combinations")
            else:
                # Fallback to random if no categorical parameters found
                study = optuna.create_study(direction="minimize")
                # if verbose:
                    # print(f"🎯 Starting random search with {n_trials} trials (no categorical params for grid)")
        else:
            # Use random/TPE sampler
            study = optuna.create_study(direction="minimize")
            # if verbose:
                # print(f"🎯 Starting {search_method} search with {n_trials} trials")

        # Show concise message about optimization
        print(f"🔍 Optimizing {len(finetune_params.get('model_params', {}))} parameters with {search_method} search ({n_trials} trials)...")
        # print(X_train_prep.shape, y_train_prep.shape, X_test_prep.shape, y_test.shape)
        study.optimize(objective, n_trials=n_trials)

        # Store best parameters for access by nested CV methods
        self._last_best_params = best_params

        # if verbose:
            # print(f"🏆 Best parameters: {best_params}, Best score: {best_score:.4f}")
        # Retrain best model with top-level train_params (final training parameters)
        if train_params != finetune_params.get('train_params', train_params):
            # print(f"🏆 Training with best parameters: {best_params}")
            # if verbose > 1:
                # print("🔄 Using final training parameters for best model...")

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
        else:
            # Just show the best parameters found
            print(f"🏆 Training with best parameters: {best_params}")


        # Generate final predictions with best model
        y_pred = self._predict_model(best_model, X_test_prep)

        # Store predictions in dataset
        unique_model_name = self._get_unique_model_name(model_config, best_model, runner)
        self._store_predictions_in_dataset(
            dataset=getattr(runner.saver, 'dataset_name', 'unknown') or 'unknown',
            pipeline=getattr(runner.saver, 'pipeline_name', 'unknown') or 'unknown',
            model=unique_model_name,
            partition=f"test_fold_{fold_idx}" if fold_idx is not None else "test",
            y_true=y_test,
            y_pred=y_pred,
            fold_idx=fold_idx,
            context=context,
            dataset_obj=dataset
        )

        # Store results
        binaries = self._store_results(best_model, y_pred, y_test, runner, "finetuned")

        # Always calculate and display final test scores for finetuned model (even at verbose=0)
        task_type = self._detect_task_type(y_train)
        unique_model_name = self._get_unique_model_name(model_config, best_model, runner)
        test_scores = self._calculate_and_print_scores(
            y_test, y_pred, task_type, "test", unique_model_name,
            show_detailed_scores=False  # Don't show detailed scores, just calculate them
        )

        # Always print finetuning summary with best score
        best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
        best_score = test_scores.get(best_metric)
        if best_score is not None:
            direction = "↑" if higher_is_better else "↓"
            # Calculate scaled score if possible
            scaled_score = self._calculate_scaled_score(y_test, y_pred, dataset)

            # Format: metric=value (scaled_value)↓ (other scores)
            if scaled_score is not None and scaled_score != best_score:
                score_display = f"{best_metric}={best_score:.4f} ({scaled_score:.4f}){direction}"
            else:
                score_display = f"{best_metric}={best_score:.4f}{direction}"

            # Add other scores if available
            other_scores = {k: v for k, v in test_scores.items() if k != best_metric}
            if other_scores:
                other_scores_str = ModelUtils.format_scores(other_scores)
                print(f"🏆 {unique_model_name} - test: {score_display} ({other_scores_str})")
            else:
                print(f"🏆 {unique_model_name} - test: {score_display}")

            if verbose > 0:  # Only show parameters at verbose > 0
                print(f"🔧 Optimized parameters: {best_params}")
        else:
            print(f"✅ Finetuned model {unique_model_name} completed successfully")

        # Save predictions to dataset results folder
        self._save_predictions_to_results_folder(dataset, runner)

        # if verbose > 0:
            # print("✅ Fine-tuning completed successfully")
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

    def _store_predictions_in_dataset(
        self,
        dataset: str,
        pipeline: str,
        model: str,
        partition: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_idx: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        dataset_obj: Optional['SpectroDataset'] = None
    ) -> None:
        """
        Store predictions in the dataset's prediction storage.

        Args:
            dataset: Dataset name
            pipeline: Pipeline name
            model: Model name
            partition: Partition name
            y_true: True values
            y_pred: Predicted values
            fold_idx: Fold index if applicable
            context: Pipeline context with processing information
            dataset_obj: Optional dataset object to store predictions in
        """
        if dataset_obj is None:
            # Can't store predictions without dataset object
            return

        # Get current y processing from context
        current_y_processing = context.get('y', 'numeric') if context else 'numeric'

        # Inverse transform predictions to original space if possible
        try:
            if hasattr(dataset_obj, '_targets') and current_y_processing != 'numeric':
                # Transform both y_true and y_pred back to numeric space
                y_true_transformed = dataset_obj._targets.transform_predictions(
                    y_true, from_processing=current_y_processing, to_processing='numeric'
                )
                y_pred_transformed = dataset_obj._targets.transform_predictions(
                    y_pred, from_processing=current_y_processing, to_processing='numeric'
                )
            else:
                y_true_transformed = y_true
                y_pred_transformed = y_pred
        except Exception as e:
            print(f"⚠️ Could not inverse transform predictions: {e}")
            y_true_transformed = y_true
            y_pred_transformed = y_pred

        # Store predictions
        dataset_obj._predictions.add_prediction(
            dataset=dataset,
            pipeline=pipeline,
            model=model,
            partition=partition,
            y_true=y_true_transformed,
            y_pred=y_pred_transformed,
            fold_idx=fold_idx,
            metadata={
                'y_processing': current_y_processing,
                'model_type': model,
                'partition': partition
            }
        )

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

