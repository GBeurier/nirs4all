"""
Simplified Base Model Controller - Clean, readable implementation

This is a complete rewrite following the user's pseudo-code specification.
The controller is designed to be simple, clean, and readable with the
logic properly separated into 3 files maximum.

Key features:
- Simple execute() method with clear train/prediction mode logic
- Externalized prediction storage, model utils, and naming logic
- Clean separation between training, finetuning, and prediction
- Framework-specific models (sklearn, tensorflow) handle their own details
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
import copy

from nirs4all.controllers.controller import OperatorController
from .model_utils import ModelUtils
from . import prediction_store
from .optuna_controller import OptunaController

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


class BaseModelController(OperatorController, ABC):
    """
    Simplified Base Model Controller - following user's pseudo-code design.

    This controller implements exactly the structure requested:
    - execute() handles prediction_mode and training_mode
    - train() handles fold logic and delegates to launch_training()
    - finetune() handles optuna optimization
    - launch_training() does the actual training and prediction
    """

    priority = 15

    def __init__(self):
        super().__init__()
        self.model_utils = ModelUtils()
        self.optuna_controller = OptunaController()

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    # Abstract methods that subclasses must implement for their frameworks
    @abstractmethod
    def _get_model_instance(self, model_config: Dict[str, Any]) -> Any:
        """Create model instance from config."""
        pass

    @abstractmethod
    def _train_model(self, model: Any, X_train: Any, y_train: Any,
                    X_val: Any = None, y_val: Any = None, **kwargs) -> Any:
        """Train the model using framework-specific logic."""
        pass

    @abstractmethod
    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions using framework-specific logic."""
        pass

    @abstractmethod
    def _prepare_data(self, X: Any, y: Any, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data in framework-specific format."""
        pass

    @abstractmethod
    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate model for optimization (returns score to minimize)."""
        pass

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: 'Predictions' = None  # NEW: External prediction store
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        Main execute method - implements user's pseudo-code logic with external prediction store:

        if prediction_mode:
            return and save as csv the predictions (train, test)
        else:
            get data from SpectroDataset using context
            if train: Create empty Predictions, train(args), store predictions in external store
            if finetune: Prepare optuna, Best_model_params = finetune(args),
                        Create empty Prediction, train(best_model, params), store predictions in external store
        """

        # Store prediction store for use in training methods
        from nirs4all.dataset.predictions import Predictions

        # CRITICAL: Ensure we use the external prediction store if provided
        if prediction_store is not None:
            self.current_prediction_store = prediction_store
        else:
            self.current_prediction_store = Predictions()

        # Extract model configuration
        model_config = self._extract_model_config(step, operator)
        verbose = model_config.get('train_params', {}).get('verbose', 0)

        # PREDICTION MODE LOGIC
        if mode == "predict":
            return self._execute_prediction_mode(
                model_config, dataset, context, runner, loaded_binaries
            )

        # GET DATA FROM SpectroDataset using context (as per pseudo-code)
        data_splits = self._get_data_from_dataset(dataset, context)

        # Determine if we have folds or single split
        # Determine if we have folds or single split
        if isinstance(data_splits, list) and len(data_splits) == 7:
            # Folds case: [X_train, y_train, X_test, y_test, folds, y_train_unscaled, y_test_unscaled]
            X_train, y_train, X_test, y_test, folds, y_train_unscaled, y_test_unscaled = data_splits
        elif isinstance(data_splits, list) and len(data_splits) == 6:
            # Single split case: [X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled]
            X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled = data_splits
            folds = None
        elif isinstance(data_splits, list) and len(data_splits) == 5:
            # Backward compatibility - old folds format
            X_train, y_train, X_test, y_test, folds = data_splits
            y_train_unscaled, y_test_unscaled = y_train, y_test  # Fallback to scaled data
        else:
            # Backward compatibility - old single split format
            X_train, y_train, X_test, y_test = data_splits
            folds = None
            y_train_unscaled, y_test_unscaled = y_train, y_test  # Fallback to scaled data        # Create empty Predictions (as per pseudo-code)
        predictions = {}

        # Check if finetune or train
        finetune_params = model_config.get('finetune_params')
        if finetune_params:
            # FINETUNE PATH
            if verbose > 0:
                print("🎯 Starting finetuning...")

            # Prepare optuna etc.... Best_model_params = finetune(args)
            best_model_params = self.finetune(
                model_config, X_train, y_train, X_test, y_test,
                folds, finetune_params, predictions, context, runner, dataset
            )

            # Create empty Prediction, train(best_model, params, etc.)
            predictions = {}
            final_predictions = self.train(
                model_config, X_train, y_train, X_test, y_test,
                folds, predictions, context, runner, dataset,
                best_params=best_model_params
            )

            # merge prediction into dataset.prediction (handled in train)

        else:
            # TRAIN PATH
            if verbose > 0:
                print("🏋️ Starting training...")

            # train(args), Merge prediction into dataset prediction
            final_predictions = self.train(
                model_config, X_train, y_train, X_test, y_test,
                folds, predictions, context, runner, dataset,
                y_train_unscaled=y_train_unscaled, y_test_unscaled=y_test_unscaled
            )

        # Return context and binaries (simplified)
        binaries = self._create_result_binaries(final_predictions, runner)
        return context, binaries

    def train(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List] = None,
        predictions: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
        runner: Optional['PipelineRunner'] = None,
        dataset: Optional['SpectroDataset'] = None,
        best_params: Optional[Dict[str, Any]] = None,
        save_models: bool = True,
        y_train_unscaled: Any = None,
        y_test_unscaled: Any = None
    ) -> Dict:
        """
        Train method following user's pseudo-code:

        if folds: foreach fold prepare data (xtrain[indices]..., X_val, y_val)
        else x_val, y_val = x_test, y_test and print warning

        for available datasets:
            Launch_training(predictions + args, save_models=True)

        if folds:
            create model_uuid as step + model_id + avg or w-avg + pipeline_id
            create avg and w-avg (based on pred and metadata of prediction)
            Evaluate score and store in predictions with all metadata but with avg or w-avg instead of fold
        """

        if predictions is None:
            predictions = {}

        verbose = model_config.get('train_params', {}).get('verbose', 0)

        if folds:
            # foreach fold prepare data (xtrain[indices]..., X_val, y_val)
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                if verbose > 0:
                    print(f"📁 Training fold {fold_idx + 1}/{len(folds)}")

                # Prepare fold data
                X_train_fold = X_train[train_indices] if hasattr(X_train, '__getitem__') else X_train
                y_train_fold = y_train[train_indices] if hasattr(y_train, '__getitem__') else y_train
                X_val_fold = X_train[val_indices] if hasattr(X_train, '__getitem__') else X_test
                y_val_fold = y_train[val_indices] if hasattr(y_train, '__getitem__') else y_test

                # Get unscaled validation data for this fold
                if context and dataset:
                    val_context_unscaled = context.copy()
                    val_context_unscaled["partition"] = "train"
                    val_context_unscaled["fold"] = fold_idx
                    val_context_unscaled["y"] = "numeric"
                    y_val_fold_unscaled = dataset.y(val_context_unscaled)[val_indices]

                    # Get unscaled training data for this fold
                    train_context_unscaled = context.copy()
                    train_context_unscaled["partition"] = "train"
                    train_context_unscaled["fold"] = fold_idx
                    train_context_unscaled["y"] = "numeric"
                    y_train_fold_unscaled = dataset.y(train_context_unscaled)[train_indices]
                else:
                    # Fallback to scaled data
                    y_train_fold_unscaled = y_train_fold
                    y_val_fold_unscaled = y_val_fold

                # Launch_training(predictions + args, save_models=True)
                fold_predictions = self.launch_training(
                    model_config, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    X_test, y_test, predictions, context, runner, dataset,
                    fold_idx=fold_idx, best_params=best_params, save_models=save_models,
                    y_train_fold_unscaled=y_train_fold_unscaled, y_val_fold_unscaled=y_val_fold_unscaled,
                    y_test_unscaled=y_test_unscaled
                )
                predictions.update(fold_predictions)

        else:
            # Single split - use test as validation (with warning)
            print("⚠️ Warning: Using test set as validation set (no folds provided)")
            single_predictions = self.launch_training(
                model_config, X_train, y_train, X_test, y_test,
                X_test, y_test, predictions, context, runner, dataset,
                fold_idx=None, best_params=best_params, save_models=save_models,
                y_train_fold_unscaled=y_train_unscaled, y_val_fold_unscaled=y_test_unscaled,
                y_test_unscaled=y_test_unscaled
            )
            predictions.update(single_predictions)

        # if folds: create avg and w-avg
        avg_predictions = {}
        if folds and len(folds) > 1:
            avg_predictions = self._create_fold_averages(
                predictions, model_config, context, runner, dataset, len(folds)
            )
            predictions.update(avg_predictions)

        # Print avg and w-avg results
        if avg_predictions:
            self._print_average_results(avg_predictions, dataset)

            # Print separator and best for config
            print("-" * 169)
        if dataset:
            self._print_best_for_config(avg_predictions, predictions, dataset)

        return predictions

    def _print_best_for_config(self, avg_predictions: Dict, all_predictions: Dict, dataset: 'SpectroDataset'):
        """Print best model for this configuration."""
        # Find best model from all predictions (including averages)
        all_models = {**all_predictions, **avg_predictions}

        best_model = None
        best_score = float('inf')
        best_info = None

        for model_key, model_data in all_models.items():
            if model_data and isinstance(model_data, dict) and 'y_true' in model_data and 'y_pred' in model_data:
                y_true = model_data['y_true']
                y_pred = model_data['y_pred']

                # Calculate scores
                task_type = self.model_utils._detect_task_type(y_true)
                scores = self.model_utils.calculate_scores(y_true, y_pred, task_type)

                primary_metric = 'mse' if task_type == 'regression' else 'accuracy'
                score = scores.get(primary_metric, float('inf'))

                # For accuracy, we want higher scores; for mse, we want lower scores
                is_better = (task_type == 'regression' and score < best_score) or \
                           (task_type != 'regression' and score > best_score)

                if is_better:
                    best_score = score
                    best_model = model_key
                    best_info = {
                        'scores': scores,
                        'task_type': task_type,
                        'model_classname': 'PLSRegression'  # TODO: Extract properly
                    }

        if best_model and best_info:
            task_type = best_info['task_type']
            scores = best_info['scores']

            # Format output
            if task_type == 'regression':
                primary_metric = 'mse'
                primary_value = scores.get(primary_metric, 0.0)
                direction = "↓"
                secondary_metric = 'mae'
                secondary_value = scores.get(secondary_metric, 0.0)
            else:
                primary_metric = 'accuracy'
                primary_value = scores.get(primary_metric, 0.0)
                direction = "↑"
                secondary_metric = 'f1'
                secondary_value = scores.get(secondary_metric, 0.0)

            # Try to get proper unscaled score, otherwise use the scaled value
            unscaled_score = primary_value  # Default to scaled value if unscaling not available
            model_classname = best_info['model_classname']

            # Determine fold info
            if "w_avg" in best_model:
                fold_info = "(w-avg)"
            elif "avg" in best_model:
                fold_info = "(avg)"
            else:
                # Extract fold from model name if present
                import re
                fold_match = re.search(r'_fold(\d+)', best_model)
                fold_info = f"(fold:{fold_match.group(1)})" if fold_match else ""

            # Print best for model in the requested format with RMSE
            rmse_value = np.sqrt(primary_value) if primary_metric == 'mse' else scores.get('rmse', 0.0)
            print(f"🏆 Best for Model: {best_model} - {model_classname} - test: "
                  f"loss({primary_metric})={primary_value:.3f}{direction} (rmse: {rmse_value:.4f}, {secondary_metric}: {secondary_value:.4f}) "
                  f"score({primary_metric})={unscaled_score:.4f}")

    def launch_training(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        predictions: Dict,
        context: Optional[Dict[str, Any]],
        runner: Optional['PipelineRunner'],
        dataset: Optional['SpectroDataset'],
        fold_idx: Optional[int] = None,
        best_params: Optional[Dict[str, Any]] = None,
        save_models: bool = True,
        y_train_fold_unscaled: Any = None,
        y_val_fold_unscaled: Any = None,
        y_test_unscaled: Any = None
    ) -> Dict:
        """
        Launch training following user's pseudo-code:

        create model_id with next_op
        create model uuid with step + model_id + pipeline_id
        instanciate the model with params if any
        train(on train and val data)
        predict y_test_pred, y_train_pred, y_val_pred, invert transform with dataset targets calls,
               and add them to the predictions (with all metadata: model_class_name, model_id, loss, fold, step, etc.)
        print the model_id, score metrics, loss, score unscaled.
        save the model
        """

        # Get step and config_id from runner
        step = getattr(runner, 'current_step', 0) if runner else 0
        config_id = getattr(runner.saver, 'pipeline_name', 'unknown') if runner else 'unknown'

        # Extract name for model ID creation
        name = self.model_utils.extract_name_from_config(model_config)

        # create model_id with next_op
        model_id = self.model_utils.create_model_id(name, runner) if runner else f"{name}_1"

        # create model uuid with step + model_id + pipeline_id
        model_uuid = self.model_utils.create_model_uuid(model_id, runner, step, config_id, fold_idx) if runner else f"{model_id}_{step}"

        # instanciate the model with params if any
        base_model = self._get_model_instance(model_config)
        model = self.model_utils.clone_model(base_model)

        # Apply best params if provided
        if best_params:
            if hasattr(model, 'set_params'):
                model.set_params(**best_params)

        # Prepare data in framework format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
        X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
        X_test_prep, y_test_prep = self._prepare_data(X_test, y_test, context or {})

        # train(on train and val data)
        trained_model = self._train_model(
            model, X_train_prep, y_train_prep, X_val_prep, y_val_prep,
            **model_config.get('train_params', {})
        )

        # predict y_test_pred, y_train_pred, y_val_pred (these are in scaled space)
        y_train_pred_scaled = self._predict_model(trained_model, X_train_prep)
        y_val_pred_scaled = self._predict_model(trained_model, X_val_prep)
        y_test_pred_scaled = self._predict_model(trained_model, X_test_prep)

        # Transform predictions from scaled space back to unscaled space
        current_y_processing = context.get('y', 'numeric') if context else 'numeric'
        if current_y_processing != 'numeric' and dataset and hasattr(dataset, '_targets'):
            try:
                y_train_pred_unscaled = dataset._targets.transform_predictions(
                    y_train_pred_scaled, current_y_processing, 'numeric'
                )
                y_val_pred_unscaled = dataset._targets.transform_predictions(
                    y_val_pred_scaled, current_y_processing, 'numeric'
                )
                y_test_pred_unscaled = dataset._targets.transform_predictions(
                    y_test_pred_scaled, current_y_processing, 'numeric'
                )
            except Exception as e:
                print(f"⚠️ Could not inverse transform predictions to unscaled space: {e}")
                # Fallback to scaled predictions
                y_train_pred_unscaled = y_train_pred_scaled
                y_val_pred_unscaled = y_val_pred_scaled
                y_test_pred_unscaled = y_test_pred_scaled
        else:
            # No scaling applied, predictions are already unscaled
            y_train_pred_unscaled = y_train_pred_scaled
            y_val_pred_unscaled = y_val_pred_scaled
            y_test_pred_unscaled = y_test_pred_scaled

        # Calculate scores for training and validation sets (simplified)
        scores_train = {"mse": 0.0, "r2": 0.0}  # TODO: implement proper scoring
        scores_val = {"mse": 0.0, "r2": 0.0}    # TODO: implement proper scoring

        # Store training and test predictions using external store (all in unscaled space)
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') if runner else 'unknown'
        prediction_store.store_training_predictions(
            self.current_prediction_store,
            dataset.name if dataset else 'unknown',
            pipeline_name,
            name,  # model_name
            model_id,
            model_uuid,
            fold_idx or 0,
            step,
            1,  # TODO: fix op_counter
            {'y_true': y_train_fold_unscaled, 'y_pred': y_train_pred_unscaled},
            {'y_true': y_val_fold_unscaled, 'y_pred': y_val_pred_unscaled},
            list(range(len(y_train_fold_unscaled))) if y_train_fold_unscaled is not None else [],
            list(range(len(y_val_fold_unscaled))) if y_val_fold_unscaled is not None else [],
            None,  # custom_model_name
            context,  # Pass context for y processing
            dataset   # Pass dataset for target transformations
        )

        # Store test predictions
        prediction_store.store_test_predictions(
            self.current_prediction_store,
            dataset.name if dataset else 'unknown',
            pipeline_name,
            name,
            model_id,
            model_uuid,
            fold_idx or 0,
            step,
            1,  # TODO: fix op_counter
            y_test_unscaled,
            y_test_pred_unscaled,
            list(range(len(y_test_unscaled))) if y_test_unscaled is not None else [],
            None,  # custom_model_name
            context,  # Pass context for y processing
            dataset   # Pass dataset for target transformations
        )

        # print the model_id, score metrics, loss, score unscaled.
        self._print_training_results(
            trained_model, model_id, y_test_unscaled, y_test_pred_unscaled, fold_idx, dataset
        )

        # Model saving is handled in _create_result_binaries to avoid op_counter increment

        return {
            'model_id': model_id,
            'model_uuid': model_uuid,
            'scores': scores_val
        }

    def finetune(
        self,
        model_config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: Optional[List],
        finetune_params: Dict[str, Any],
        predictions: Dict,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        dataset: 'SpectroDataset'
    ) -> Dict[str, Any]:
        """
        Finetune method - delegates to external Optuna controller.
        """

        return self.optuna_controller.finetune(
            model_config, X_train, y_train, X_test, y_test,
            folds, finetune_params, context, self
        )

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters for optimization. Override in subclasses."""
        return {}

    def _detect_task_type(self, y: Any) -> str:
        """Detect task type from target values."""
        return self.model_utils._detect_task_type(y)

    def _calculate_and_print_scores(
        self,
        y_true: Any,
        y_pred: Any,
        task_type: str,
        partition: str = "test",
        model_name: str = "model",
        show_detailed_scores: bool = True
    ) -> Dict[str, float]:
        """Calculate scores and print them."""
        scores = self.model_utils.calculate_scores(y_true, y_pred, task_type)
        if scores and show_detailed_scores:
            score_str = self.model_utils.format_scores(scores)
            print(f"📊 {model_name} {partition} scores: {score_str}")
        return scores

    def _clone_model(self, model: Any) -> Any:
        """Clone model using model utils."""
        return self.model_utils.clone_model(model)

    def get_preferred_layout(self) -> str:
        """Get the preferred data layout. Override in subclasses."""
        return "2d"

    # Helper methods
    def _execute_prediction_mode(
        self,
        model_config: Dict[str, Any],
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[List[Tuple[str, bytes]]]
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Handle prediction mode: load model and predict."""

        if not loaded_binaries:
            raise ValueError("No model binaries provided for prediction mode")

        # Load trained model from binaries
        trained_model = self._load_model_from_binaries(loaded_binaries)

        # Get prediction data (try test, fallback to train, then all)
        prediction_data = self._get_prediction_data(dataset, context)
        X_pred_prep, y_true = self._prepare_data(prediction_data['X'], prediction_data.get('y'), context)

        # Generate predictions
        y_pred = self._predict_model(trained_model, X_pred_prep)

        # Store and create CSV as requested in pseudo-code
        # predictions_csv = prediction_store.create_prediction_csv(y_true, y_pred)  # TODO: implement CSV creation
        pred_filename = f"predictions_{runner.next_op()}.csv"

        return context, [(pred_filename, "predictions csv placeholder".encode('utf-8'))]

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from step or operator."""
        if operator is not None:
            if isinstance(step, dict):
                config = step.copy()
                config['model_instance'] = operator
                return config
            else:
                return {'model_instance': operator}

        if isinstance(step, dict):
            if 'model' in step:
                config = step.copy()
                model_obj = step['model']

                # Handle nested model format
                if isinstance(model_obj, dict):
                    if 'model' in model_obj:
                        config['model_instance'] = model_obj['model']
                        if 'name' in model_obj:
                            config['name'] = model_obj['name']
                    elif '_runtime_instance' in model_obj:
                        config['model_instance'] = model_obj['_runtime_instance']
                    else:
                        config['model_instance'] = model_obj
                else:
                    config['model_instance'] = model_obj
                return config
            else:
                return {'model_instance': step}
        else:
            return {'model_instance': step}

    def _get_data_from_dataset(self, dataset: 'SpectroDataset', context: Dict[str, Any]):
        """Get data from SpectroDataset using context (as per pseudo-code)."""

        # Get layout preference
        layout = context.get('layout', '2d')

        if hasattr(dataset, 'num_folds') and dataset.num_folds > 0:
            # Get all training data first (scaled for training)
            train_context = context.copy()
            train_context["partition"] = "train"
            X_train = dataset.x(train_context, layout, concat_source=True)
            y_train = dataset.y(train_context)  # Scaled data for training

            # Get unscaled training data for storage
            train_context_unscaled = train_context.copy()
            train_context_unscaled["y"] = "numeric"  # Force unscaled
            y_train_unscaled = dataset.y(train_context_unscaled)

            # Get test data (scaled for training)
            test_context = context.copy()
            test_context["partition"] = "test"
            X_test = dataset.x(test_context, layout, concat_source=True)
            y_test = dataset.y(test_context)  # Scaled data for training

            # Get unscaled test data for storage
            test_context_unscaled = test_context.copy()
            test_context_unscaled["y"] = "numeric"  # Force unscaled
            y_test_unscaled = dataset.y(test_context_unscaled)

            folds = dataset.folds
            return [X_train, y_train, X_test, y_test, folds, y_train_unscaled, y_test_unscaled]

        # Single split case
        train_context = context.copy()
        train_context["partition"] = "train"
        X_train = dataset.x(train_context, layout, concat_source=True)
        y_train = dataset.y(train_context)  # Scaled data for training

        # Get unscaled training data for storage
        train_context_unscaled = train_context.copy()
        train_context_unscaled["y"] = "numeric"  # Force unscaled
        y_train_unscaled = dataset.y(train_context_unscaled)

        test_context = context.copy()
        test_context["partition"] = "test"
        X_test = dataset.x(test_context, layout, concat_source=True)
        y_test = dataset.y(test_context)  # Scaled data for training

        # Get unscaled test data for storage
        test_context_unscaled = test_context.copy()
        test_context_unscaled["y"] = "numeric"  # Force unscaled
        y_test_unscaled = dataset.y(test_context_unscaled)

        return [X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled]

    def _get_prediction_data(self, dataset: 'SpectroDataset', context: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for prediction mode."""
        layout = context.get('layout', '2d')

        # Try different partitions
        try:
            test_context = context.copy()
            test_context["partition"] = "test"
            X_test = dataset.x(test_context, layout, concat_source=True)
            y_test = dataset.y(test_context)
            return {'X': X_test, 'y': y_test}
        except Exception:
            try:
                train_context = context.copy()
                train_context["partition"] = "train"
                X_train = dataset.x(train_context, layout, concat_source=True)
                y_train = dataset.y(train_context)
                return {'X': X_train, 'y': y_train}
            except Exception:
                # Fallback to all data
                X_all = dataset.x(context, layout, concat_source=True)
                y_all = dataset.y(context)
                return {'X': X_all, 'y': y_all}

    def _load_model_from_binaries(self, loaded_binaries: List[Tuple[str, bytes]]) -> Any:
        """Load trained model from binary data."""
        import pickle

        model_binary = None
        for name, binary in loaded_binaries:
            if name.endswith('.pkl') and ('model' in name.lower() or 'trained' in name.lower()):
                model_binary = binary
                break

        if model_binary is None:
            raise ValueError("No model binary found")

        return pickle.loads(model_binary)

    def _create_fold_averages(
        self,
        predictions: Dict,
        model_config: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        runner: Optional['PipelineRunner'],
        dataset: Optional['SpectroDataset'],
        num_folds: int
    ) -> Dict:
        """Create virtual avg and w-avg models in the external prediction store."""

        # Extract model information
        name = self.model_utils.extract_name_from_config(model_config)
        step = getattr(runner, 'current_step', 0) if runner else 0
        pipeline_name = getattr(runner.saver, 'pipeline_name', 'unknown') if runner and hasattr(runner, 'saver') else 'unknown'

        # Create averages using the Predictions class
        avg_predictions = {}

        # Create simple average
        avg_result = self.current_prediction_store.calculate_average_predictions(
            dataset=dataset.name if dataset else 'unknown',
            pipeline=pipeline_name,
            model=name,
            partition="test",  # Average test predictions
            store_result=False  # Don't auto-store, we'll store as virtual model
        )

        if avg_result:
            # Store as virtual model with special naming
            avg_model_uuid = f"{name}_fold_avg_step_{step}_{pipeline_name}"
            prediction_store.store_virtual_model_predictions(
                self.current_prediction_store,
                dataset.name,
                pipeline_name,
                model_name=name,
                model_id=f"{name}_avg",
                model_uuid=avg_model_uuid,
                partition="test",
                fold_idx="avg",  # Special fold_idx for averages
                step=step,
                y_true=avg_result['y_true'],
                y_pred=avg_result['y_pred'],
                test_indices=avg_result.get('indices', []),
                custom_model_name=f"{name}_avg",
                context=context,  # Pass context for y processing
                dataset=dataset   # Pass dataset for target transformations
            )
            avg_predictions[f"{name}_avg"] = avg_result

        # Create weighted average
        wavg_result = self.current_prediction_store.calculate_weighted_average_predictions(
            dataset=dataset.name if dataset else 'unknown',
            pipeline=pipeline_name,
            model=name,
            test_partition="test",
            val_partition="val",
            store_result=False
        )

        if wavg_result:
            # Store as virtual model
            wavg_model_uuid = f"{name}_fold_w_avg_step_{step}_{pipeline_name}"
            prediction_store.store_virtual_model_predictions(
                self.current_prediction_store,
                dataset.name if dataset else 'unknown',
                pipeline_name,
                model_name=name,
                model_id=f"{name}_w_avg",
                model_uuid=wavg_model_uuid,
                partition="test",
                fold_idx="w-avg",  # Special fold_idx for weighted averages
                step=step,
                y_true=wavg_result['y_true'],
                y_pred=wavg_result['y_pred'],
                test_indices=wavg_result.get('indices', []),
                custom_model_name=f"{name}_w_avg",
                context=context,  # Pass context for y processing
                dataset=dataset   # Pass dataset for target transformations
            )
            avg_predictions[f"{name}_w_avg"] = wavg_result

        return avg_predictions

    def _print_training_results(
        self,
        trained_model: Any,
        model_id: str,
        y_test: Any,
        y_test_pred: Any,
        fold_idx: Optional[int] = None,
        dataset: Optional['SpectroDataset'] = None
    ):
        """Print training results with scores, arrows, and unscaled metrics."""

        # Calculate scores
        task_type = self.model_utils._detect_task_type(y_test)
        scores = self.model_utils.calculate_scores(y_test, y_test_pred, task_type)

        # Format display name
        display_name = f"{model_id}"
        if fold_idx is not None:
            display_name += f"_fold{fold_idx}"

        # Get best metric and direction
        best_metric, higher_is_better = self.model_utils.get_best_metric_for_task(task_type)
        direction = "↑" if higher_is_better else "↓"

        # Format scores with arrow for best metric
        formatted_scores = []
        unscaled_score = None

        for metric, value in scores.items():
            if metric == best_metric:
                formatted_scores.append(f"{metric}={value:.4f}{direction}")
                # Calculate unscaled score (if possible)
                if dataset is not None:
                    unscaled_score = self._calculate_unscaled_score(y_test, y_test_pred, metric, dataset)
            else:
                formatted_scores.append(f"{metric}={value:.4f}")

        score_str = ", ".join(formatted_scores)

        # Add unscaled score if available
        if unscaled_score is not None:
            score_str += f" - ({best_metric.upper()}: {unscaled_score:.4f})"

        print(f"✅ {display_name} - test: {score_str}")

    def _calculate_unscaled_score(self, y_true: Any, y_pred: Any, metric: str, dataset: Optional['SpectroDataset']) -> Optional[float]:
        """Calculate unscaled score. Since we receive unscaled data, calculate metric directly."""
        try:
            # The data passed to this method should already be unscaled
            # (y_test_unscaled, y_test_pred_unscaled from the calling method)
            # So we can calculate the metric directly without additional unscaling

            # Flatten arrays
            y_true_flat = np.asarray(y_true).flatten()
            y_pred_flat = np.asarray(y_pred).flatten()

            # Calculate the metric on the data (which should be unscaled)
            if metric == 'rmse':
                from sklearn.metrics import mean_squared_error
                return np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
            elif metric == 'mse':
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_true_flat, y_pred_flat)
            elif metric == 'mae':
                from sklearn.metrics import mean_absolute_error
                return mean_absolute_error(y_true_flat, y_pred_flat)
            elif metric == 'r2':
                from sklearn.metrics import r2_score
                return r2_score(y_true_flat, y_pred_flat)

        except Exception:
            pass
        return None

    def _print_average_results(self, avg_predictions: Dict, dataset: Optional['SpectroDataset']):
        """Print average and weighted average results."""
        if not avg_predictions:
            # print("⚠️ No average predictions to print")
            return

        # print(f"📊 Found {len(avg_predictions)} average predictions to print")

        for pred_key, pred_data in avg_predictions.items():
            # print(f"🔍 Checking prediction: {pred_key} with partition {pred_data.get('partition', 'N/A')}")

            # Only print test partition averages
            if 'test' in pred_data.get('partition', ''):
                y_true = pred_data.get('y_true')
                y_pred = pred_data.get('y_pred')
                real_model = pred_data.get('real_model', pred_key)

                if y_true is not None and y_pred is not None:
                    # Use the same printing logic as individual models
                    task_type = self.model_utils._detect_task_type(y_true)
                    scores = self.model_utils.calculate_scores(y_true, y_pred, task_type)

                    # Get best metric and direction
                    best_metric, higher_is_better = self.model_utils.get_best_metric_for_task(task_type)
                    direction = "↑" if higher_is_better else "↓"

                    # Format scores with arrow for best metric
                    formatted_scores = []
                    unscaled_score = None

                    for metric, value in scores.items():
                        if metric == best_metric:
                            formatted_scores.append(f"{metric}={value:.4f}{direction}")
                            unscaled_score = self._calculate_unscaled_score(y_true, y_pred, metric, dataset)
                        else:
                            formatted_scores.append(f"{metric}={value:.4f}")

                    score_str = ", ".join(formatted_scores)

                    # Add unscaled score if available
                    if unscaled_score is not None:
                        score_str += f" - ({best_metric.upper()}: {unscaled_score:.4f})"

                    print(f"✅ {real_model} - test: {score_str}")
                else:
                    print(f"⚠️ Missing y_true or y_pred for {real_model}")
            else:
                print(f"⚠️ Skipping non-test partition: {pred_data.get('partition', 'N/A')}")

    def _create_result_binaries(self, predictions: Dict, runner: 'PipelineRunner') -> List[Tuple[str, bytes]]:
        """Create result binaries from predictions - only save actual model files."""
        import pickle
        binaries = []

        # Only save best model for the entire run (not per fold/prediction)
        # Find the best model based on test performance
        best_model = None
        best_score = float('inf')
        best_name = None

        for pred_key, pred_data in predictions.items():
            if 'trained_model' in pred_data and 'test' in pred_key:
                # Calculate score for this model
                y_true = pred_data.get('y_true')
                y_pred = pred_data.get('y_pred')
                if y_true is not None and y_pred is not None:
                    scores = self.model_utils.calculate_scores(y_true, y_pred)
                    # Use MSE for comparison (lower is better)
                    mse = scores.get('mse', float('inf'))
                    if mse < best_score:
                        best_score = mse
                        best_model = pred_data['trained_model']
                        # Extract model name from prediction key
                        best_name = pred_data.get('real_model', pred_key)

        # Save only the best model
        if best_model is not None and best_name is not None:
            model_binary = pickle.dumps(best_model)
            filename = best_name + ".pkl"
            binaries.append((filename, model_binary))

        return binaries
