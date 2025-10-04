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
from tabnanny import verbose
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
import copy

from nirs4all.controllers.controller import OperatorController
from .model_controller_helper import ModelControllerHelper
from .optuna_controller import OptunaController
from nirs4all.dataset.predictions import Predictions
from nirs4all.utils.model_utils import ModelUtils
import nirs4all.dataset.evaluator as Evaluator

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
        self.model_helper = ModelControllerHelper()
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


    def get_xy(self, dataset: 'SpectroDataset', context: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
        layout = self.get_preferred_layout()
        train_context = copy.deepcopy(context)
        train_context['partition'] = 'train'
        test_context = copy.deepcopy(context)
        test_context['partition'] = 'test'

        X_train = dataset.x(train_context, layout=layout)
        y_train = dataset.y(train_context)
        X_test = dataset.x(test_context, layout=layout)
        y_test = dataset.y(test_context)
        train_context['y'] = 'numeric'
        test_context['y'] = 'numeric'
        y_train_unscaled = dataset.y(train_context)
        y_test_unscaled = dataset.y(test_context)
        return X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled


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

        self.prediction_store = prediction_store
        model_config = self._extract_model_config(step, operator)
        self.verbose = model_config.get('train_params', {}).get('verbose', 0)

        # if mode == "predict":
            # return self._execute_prediction_mode( model_config, dataset, context, runner, loaded_binaries)

        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled = self.get_xy(dataset, context)
        folds = dataset.folds

        binaries = []

        finetune_params = model_config.get('finetune_params')
        if finetune_params:
            # FINETUNE PATH
            if verbose > 0:
                print("🎯 Starting finetuning...")

            # # Prepare optuna etc.... Best_model_params = finetune(args)
            # best_model_params = self.finetune(
            #     model_config, X_train, y_train, X_test, y_test,
            #     folds, finetune_params, predictions, context, runner, dataset
            # )

            # # Create empty Prediction, train(best_model, params, etc.)
            # predictions = {}
            # final_predictions = self.train(
            #     model_config, X_train, y_train, X_test, y_test,
            #     folds, predictions, context, runner, dataset,
            #     best_params=best_model_params
            # )

            # merge prediction into dataset.prediction (handled in train)

        else:
            # TRAIN PATH
            if self.verbose > 0:
                print("🏋️ Starting training...")

            binaries = self.train(
                dataset, model_config, context, runner, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds
            )

        return context, binaries

    def train(
        self,
        dataset, model_config, context, runner, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        best_params=None
    ) -> Dict:

        verbose = model_config.get('train_params', {}).get('verbose', 0)

        binaries = []
        if len(folds) > 0:
            folds_models = []
            fold_val_indices = []
            scores = []
            base_model_name = ""
            model_classname = ""
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                if verbose > 0:
                    print(f"📁 Training fold {fold_idx + 1}/{len(folds)}")
                fold_val_indices.append(val_indices)
                X_train_fold = X_train[train_indices]
                y_train_fold = y_train[train_indices]
                y_train_fold_unscaled = y_train_unscaled[train_indices]
                X_val_fold = X_train[val_indices]
                y_val_fold = y_train[val_indices]
                y_val_fold_unscaled = y_train_unscaled[val_indices]

                model, model_id, score, model_name = self.launch_training(
                    dataset, model_config, context, runner, prediction_store,
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test,
                    y_train_fold_unscaled, y_val_fold_unscaled, y_test_unscaled,
                    train_indices, val_indices,
                    fold_idx=fold_idx, best_params=best_params
                )
                folds_models.append((model_id, model, score))
                base_model_name = model_name
                binaries.append((f"{model_id}.pkl", self._binarize_model(model)))
                scores.append(score)
                model_classname = model.__class__.__name__

            self._create_fold_averages(
                base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
                folds_models, fold_val_indices, scores,
                X_train, X_test, y_train_unscaled, y_test_unscaled
            )

        else:
            print("\033[91m⚠️⚠️⚠️⚠️⚠️  WARNING: Using test set as validation set (no folds provided) ⚠️⚠️⚠️⚠️⚠️⚠️\033[0m")

            model, model_id, score, model_name = self.launch_training(
                dataset, model_config, context, runner, prediction_store,
                X_train, y_train, X_test, y_test, X_test,
                y_train_unscaled, y_test_unscaled, y_test_unscaled
            )
            binaries.append((f"{model_id}.pkl", self._binarize_model(model)))

        return binaries


    def launch_training(
        self,
        dataset, model_config, context, runner, prediction_store,
        X_train, y_train, X_val, y_val, X_test,
        y_train_unscaled, y_val_unscaled, y_test_unscaled,
        train_indices=None, val_indices=None, fold_idx=None, best_params=None) :

        # instanciate the model with params if any
        base_model = self._get_model_instance(model_config)
        model = self.model_helper.clone_model(base_model)

        # Generate identifiers
        step_id = context['step_id']
        pipeline_name = runner.saver.pipeline_name
        dataset_name = dataset.name
        model_classname = self.model_helper.extract_core_name(model)
        model_name = model_config.get('name', model_classname)
        operation_counter = runner.next_op()

        # Apply best params if provided ####TODO RETRIEVE THE HOLD METHOD (construct with params !!!!)
        if best_params:
            if hasattr(model, 'set_params'):
                model.set_params(**best_params)

        # Prepare data in framework format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
        X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
        X_test_prep, _ = self._prepare_data(X_test, None, context or {})

        # if self.verbose > 0:
        # print("🚀 Training model...")
        # print("Dataset:", dataset_name, "Shape:", X_train.shape)

        trained_model = self._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **model_config.get('train_params', {}))

        # predict y_test_pred, y_train_pred, y_val_pred (these are in scaled space)
        y_train_pred_scaled = self._predict_model(trained_model, X_train_prep)
        y_val_pred_scaled = self._predict_model(trained_model, X_val_prep)
        y_test_pred_scaled = self._predict_model(trained_model, X_test_prep)

        # Transform predictions from scaled space back to unscaled space
        current_y_processing = context.get('y', 'numeric') if context else 'numeric'
        if current_y_processing != 'numeric':
            y_train_pred_unscaled = dataset._targets.transform_predictions(
                y_train_pred_scaled, current_y_processing, 'numeric'
            )
            y_val_pred_unscaled = dataset._targets.transform_predictions(
                y_val_pred_scaled, current_y_processing, 'numeric'
            )
            y_test_pred_unscaled = dataset._targets.transform_predictions(
                y_test_pred_scaled, current_y_processing, 'numeric'
            )
        else:
            y_train_pred_unscaled = y_train_pred_scaled
            y_val_pred_unscaled = y_val_pred_scaled
            y_test_pred_unscaled = y_test_pred_scaled


        metric, higher_is_better = ModelUtils.deprec_get_best_metric_for_task(dataset.task_type)
        direction = "↑" if higher_is_better else "↓"
        score_train = Evaluator.eval(y_train_unscaled, y_train_pred_unscaled, metric)
        score_val = Evaluator.eval(y_val_unscaled, y_val_pred_unscaled, metric)
        score_test = Evaluator.eval(y_test_unscaled, y_test_pred_unscaled, metric)

        if train_indices is None:
            train_indices = list(range(len(y_train_unscaled)))
        else:
            train_indices = [int(idx) for idx in train_indices]  # Convert numpy int types to Python int

        if val_indices is None:
            val_indices = list(range(len(y_val_unscaled)))
        else:
            val_indices = [int(idx) for idx in val_indices]  # Convert numpy int types to Python int

        test_indices = list(range(len(y_test_unscaled)))

        # Store individual predictions for each partition
        for partition_name, indices, y_true_part, y_pred_part in [
            ("train", train_indices, y_train_unscaled, y_train_pred_unscaled),
            ("val", val_indices, y_val_unscaled, y_val_pred_unscaled),
            ("test", test_indices, y_test_unscaled, y_test_pred_unscaled)
        ]:
            pred_id = prediction_store.add_prediction(
                dataset_name=dataset_name,
                dataset_path=dataset_name,
                config_name=pipeline_name,
                config_path=f"{dataset_name}/{pipeline_name}",
                step_idx=step_id,
                op_counter=operation_counter,
                model_name=model_name,
                model_classname=model_classname,
                model_path=f"{dataset_name}/{pipeline_name}/{step_id}_{model_name}_{operation_counter}.pkl",
                fold_id=fold_idx,
                sample_indices=indices,
                weights=None,
                metadata={},
                partition=partition_name,
                y_true=y_true_part,
                y_pred=y_pred_part,
                val_score=score_val,
                test_score=score_test,
                metric=metric,
                task_type=dataset.task_type,
                n_samples=len(y_true_part),
                n_features=X_train.shape[1],
                preprocessings=dataset.short_preprocessings_str()
            )

        short_desc = f"✅ {model_name} - {metric}{direction} [test: {score_test:.4f}], [val: {score_val:.4f}]"
        if fold_idx not in [None, 'None', 'avg', 'w-avg']:
            short_desc += f", (fold: {fold_idx}, id: {operation_counter}) - [{pred_id}]"
        print(short_desc)

        return trained_model, f"{model_name}_{operation_counter}", score_val, model_name

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
        return ModelUtils.deprec_detect_task_type(y)

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
        scores = ModelUtils.deprec_calculate_scores(y_true, y_pred, task_type)
        if scores and show_detailed_scores:
            score_str = ModelUtils.deprec_format_scores(scores)
            print(f"📊 {model_name} {partition} scores: {score_str}")
        return scores

    def _clone_model(self, model: Any) -> Any:
        """Clone model using model utils."""
        return self.model_helper.clone_model(model)

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
        base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
        folds_models, fold_val_indices, scores,
        X_train, X_test, y_train_unscaled, y_test_unscaled
    ):

        # X_val is the concatenation of all fold val sets
        X_val = np.vstack([X_train[val_idx] for val_idx in fold_val_indices])
        y_val_unscaled = np.hstack([y_train_unscaled[val_idx] for val_idx in fold_val_indices])
        all_val_indices = np.hstack(fold_val_indices)

        # Generate all predictions for train, val, test for each fold model
        all_train_preds = []
        all_val_preds = []
        all_test_preds = []

        for fold_model_tuple in folds_models:
            # Extract the actual model from the tuple (model_id, model, score)
            _, fold_model, _ = fold_model_tuple
            fold_train_preds = self._predict_model(fold_model, X_train)
            fold_val_preds = self._predict_model(fold_model, X_val)
            fold_test_preds = self._predict_model(fold_model, X_test)

            current_y_processing = context.get('y', 'numeric') if context else 'numeric'
            if current_y_processing != 'numeric':
                fold_train_preds_unscaled = dataset._targets.transform_predictions(
                    fold_train_preds, current_y_processing, 'numeric'
                )
                fold_val_preds_unscaled = dataset._targets.transform_predictions(
                    fold_val_preds, current_y_processing, 'numeric'
                )
                fold_test_preds_unscaled = dataset._targets.transform_predictions(
                    fold_test_preds, current_y_processing, 'numeric'
                )
            else:
                fold_train_preds_unscaled = fold_train_preds
                fold_val_preds_unscaled = fold_val_preds
                fold_test_preds_unscaled = fold_test_preds

            all_train_preds.append(fold_train_preds_unscaled)
            all_val_preds.append(fold_val_preds_unscaled)
            all_test_preds.append(fold_test_preds_unscaled)

        all_train_avg_preds = np.mean(all_train_preds, axis=0)
        all_val_avg_preds = np.mean(all_val_preds, axis=0)
        all_test_avg_preds = np.mean(all_test_preds, axis=0)

        metric, higher_is_better = ModelUtils.deprec_get_best_metric_for_task(dataset.task_type)
        direction = "↑" if higher_is_better else "↓"

        # Evaluate avgerage predictions
        score_train = Evaluator.eval(y_train_unscaled, all_train_avg_preds, metric)
        score_val = Evaluator.eval(y_val_unscaled, all_val_avg_preds, metric)
        score_test = Evaluator.eval(y_test_unscaled, all_test_avg_preds, metric)
        avg_counter = runner.next_op()

        folds_id = list(range(len(folds_models)))  # Fold IDs for averaging

        # Store average predictions for each partition
        for partition_name, indices, y_true_part, y_pred_part in [
            ("train", list(range(len(y_train_unscaled))), y_train_unscaled, all_train_avg_preds),
            ("val", all_val_indices.tolist(), y_val_unscaled, all_val_avg_preds),
            ("test", list(range(len(y_test_unscaled))), y_test_unscaled, all_test_avg_preds)
        ]:
            pred_id = prediction_store.add_prediction(
                dataset_name=dataset.name,
                dataset_path=dataset.name,
                config_name=runner.saver.pipeline_name,
                config_path=f"{dataset.name}/{runner.saver.pipeline_name}",
                step_idx=context['step_id'],
                op_counter=avg_counter,
                model_name=f"{base_model_name}",
                model_classname=model_classname,
                model_path="",
                fold_id="avg",
                sample_indices=indices,
                weights=None,
                metadata={},
                partition=partition_name,
                y_true=y_true_part,
                y_pred=y_pred_part,
                val_score=score_val,
                test_score=score_test,
                metric=metric,
                task_type=dataset.task_type,
                n_samples=len(y_true_part),
                n_features=X_train.shape[1],
                preprocessings=dataset.short_preprocessings_str()
            )

        short_desc = f"✅ {base_model_name} - {metric}{direction} [test: {score_test:.4f}], [val: {score_val:.4f}], (avg, id: {avg_counter}) - [{pred_id}]"
        print(short_desc)

        # Weighted average predictions based on fold scores
        scores = np.asarray(scores, dtype=float)
        weights = ModelUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)
        all_train_w_avg_preds = np.zeros_like(all_train_preds[0], dtype=float)
        for arr, weight in zip(all_train_preds, weights):
            all_train_w_avg_preds += weight * arr

        all_val_w_avg_preds = np.zeros_like(all_val_preds[0], dtype=float)
        for arr, weight in zip(all_val_preds, weights):
            all_val_w_avg_preds += weight * arr

        all_test_w_avg_preds = np.zeros_like(all_test_preds[0], dtype=float)
        for arr, weight in zip(all_test_preds, weights):
            all_test_w_avg_preds += weight * arr

        # Evaluate weighted average predictions
        score_train_w = Evaluator.eval(y_train_unscaled, all_train_w_avg_preds, metric)
        score_val_w = Evaluator.eval(y_val_unscaled, all_val_w_avg_preds, metric)
        score_test_w = Evaluator.eval(y_test_unscaled, all_test_w_avg_preds, metric)
        w_avg_counter = runner.next_op()

        # Store weighted average predictions for each partition
        for partition_name, indices, y_true_part, y_pred_part, weight_list in [
            ("train", list(range(len(y_train_unscaled))), y_train_unscaled, all_train_w_avg_preds, weights.tolist()),
            ("val", all_val_indices.tolist(), y_val_unscaled, all_val_w_avg_preds, weights.tolist()),
            ("test", list(range(len(y_test_unscaled))), y_test_unscaled, all_test_w_avg_preds, weights.tolist())
        ]:
            pred_id = prediction_store.add_prediction(
                dataset_name=dataset.name,
                dataset_path=dataset.name,
                config_name=runner.saver.pipeline_name,
                config_path=f"{dataset.name}/{runner.saver.pipeline_name}",
                step_idx=context['step_id'],
                op_counter=w_avg_counter,
                model_name=f"{base_model_name}",
                model_classname=model_classname,
                model_path="",
                fold_id='w_avg',  # No specific fold for weighted average
                sample_indices=indices,
                weights=weight_list,
                metadata={},
                partition=partition_name,
                y_true=y_true_part,
                y_pred=y_pred_part,
                val_score=score_val_w,
                test_score=score_test_w,
                metric=metric,
                task_type=dataset.task_type,
                n_samples=len(y_true_part),
                n_features=X_train.shape[1],
                preprocessings=dataset.short_preprocessings_str()
            )

        short_desc = f"✅ {base_model_name} - {metric}{direction} [test: {score_test_w:.4f}], [val: {score_val_w:.4f}], (w_avg, id: {w_avg_counter}) - [{pred_id}]"
        print(short_desc)

    def _binarize_model(self, model: Any) -> bytes:
        """Serialize model to binary using pickle."""
        import pickle
        try:
            return pickle.dumps(model)
        except Exception as e:
            print(f"⚠️ Could not serialize model: {e}")
            return b""
