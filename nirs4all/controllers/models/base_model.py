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
from .helper import ModelControllerHelper
from ...optimization.optuna import OptunaManager
from nirs4all.data.predictions import Predictions
from nirs4all.utils.model_utils import ModelUtils, TaskType
from nirs4all.utils.emoji import CHECK, ARROW_UP, ARROW_DOWN, SEARCH, FOLDER, CHART, WEIGHT_LIFT, WARNING
import nirs4all.utils.evaluator as Evaluator

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.io import ArtifactMeta


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
        self.optuna_manager = OptunaManager()

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    # Abstract methods that subclasses must implement for their frameworks
    @abstractmethod
    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create model instance from config using ModelBuilderFactory."""
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

        # For classification tasks, use the transformed targets for evaluation
        # For regression tasks, use the original "numeric" targets
        if dataset.task_type and dataset.task_type.is_classification:
            # Use the same y context as the model training (transformed targets)
            y_train_unscaled = dataset.y(train_context)
            y_test_unscaled = dataset.y(test_context)
        else:
            # Use numeric targets for regression
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
    ) -> Tuple[Dict[str, Any], List['ArtifactMeta']]:
        # # DEBUG
        # step_repr = step if not callable(step) else f'<callable>'
        # operator_repr = operator if not callable(operator) else f'<callable>'
        # print(f"DEBUG execute() step type: {type(step)}, step: {step_repr}")
        # print(f"DEBUG execute() operator type: {type(operator)}, operator: {operator_repr}")

        self.prediction_store = prediction_store
        model_config = self._extract_model_config(step, operator)
        self.verbose = model_config.get('train_params', {}).get('verbose', 0)

        # if mode == "predict":
            # return self._execute_prediction_mode( model_config, dataset, context, runner, loaded_binaries)

        # In predict/explain mode, restore task_type from target_model if not set
        if mode in ("predict", "explain") and dataset.task_type is None:
            if hasattr(runner, 'target_model') and runner.target_model:
                task_type_str = runner.target_model.get('task_type', 'regression')
                dataset.set_task_type(task_type_str)

        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled = self.get_xy(dataset, context)
        folds = dataset.folds

        binaries = []
        finetune_params = model_config.get('finetune_params')
        if runner.verbose > 0:
            print(f"{SEARCH} Model config: {model_config}")

        if finetune_params:
            self.mode = "finetune"
            if verbose > 0:
                print("{TARGET} Starting finetuning...")

            best_model_params = self.finetune(
                dataset,
                model_config, X_train, y_train, X_test, y_test,
                folds, finetune_params, self.prediction_store, context, runner
            )
            # print("Best model params found:", best_model_params)
            print(f"{CHART} Best parameters: {best_model_params}")

            binaries = self.train(
                dataset, model_config, context, runner, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
                loaded_binaries=loaded_binaries, mode="finetune", best_params=best_model_params
            )
        else:
            # TRAIN PATH
            if self.verbose > 0:
                print(f"{WEIGHT_LIFT}Starting training...")

            binaries = self.train(
                dataset, model_config, context, runner, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
                loaded_binaries=loaded_binaries, mode=mode
            )

        return context, binaries

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
        predictions: Dict,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Finetune method - delegates to external Optuna manager.

        Clean delegation that passes all necessary data to the optuna manager
        and returns the optimized parameters for use in training.
        """
        # Store dataset reference for model building

        self.dataset = dataset

        return self.optuna_manager.finetune(
            dataset,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            folds=folds,
            finetune_params=finetune_params,
            context=context,
            controller=self
        )



    def train(
        self,
        dataset, model_config, context, runner, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        best_params=None, loaded_binaries=None, mode="train"
    ) -> List['ArtifactMeta']:

        verbose = model_config.get('train_params', {}).get('verbose', 0)

        binaries = []
        if len(folds) > 0:
            folds_models = []
            fold_val_indices = []
            scores = []
            all_fold_predictions = []
            base_model_name = ""
            model_classname = ""
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                # if mode == "predict":
                    # print(f"indices length: train {len(train_indices)}, val {len(val_indices)}")
                    # print("data sizes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

                if verbose > 0:
                    print(f"{FOLDER} Training fold {fold_idx + 1}/{len(folds)}")
                fold_val_indices.append(val_indices)
                X_train_fold = X_train[train_indices] if X_train.shape[0] > 0 else np.array([])
                y_train_fold = y_train[train_indices] if y_train.shape[0] > 0 else np.array([])
                y_train_fold_unscaled = y_train_unscaled[train_indices] if y_train_unscaled.shape[0] > 0 else np.array([])
                X_val_fold = X_train[val_indices] if X_train.shape[0] > 0 else np.array([])
                y_val_fold = y_train[val_indices] if y_train.shape[0] > 0 else np.array([])
                y_val_fold_unscaled = y_train_unscaled[val_indices] if y_train_unscaled.shape[0] > 0 else np.array([])


                if isinstance(best_params, list):
                    best_params_fold = best_params[fold_idx] if fold_idx < len(best_params) else None
                else:
                    best_params_fold = best_params
                model, model_id, score, model_name, prediction_data = self.launch_training(
                    dataset, model_config, context, runner, prediction_store,
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test,
                    y_train_fold_unscaled, y_val_fold_unscaled, y_test_unscaled,
                    train_indices, val_indices,
                    fold_idx=fold_idx, best_params= best_params_fold,
                    loaded_binaries=loaded_binaries, mode=mode
                )
                folds_models.append((model_id, model, score))
                all_fold_predictions.append(prediction_data)
                base_model_name = model_name

                # Only persist in train mode, not in predict/explain modes
                if mode == "train":
                    artifact = self._persist_model(runner, model, model_id)
                    binaries.append(artifact)

                scores.append(score)
                model_classname = model.__class__.__name__

            # Compute weights based on scores
            metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
            weights = ModelUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)

            # Create fold averages and get average predictions data
            if dataset.task_type and dataset.task_type.is_regression:
                avg_predictions, w_avg_predictions = self._create_fold_averages(
                    base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
                    folds_models, fold_val_indices, scores,
                    X_train, X_test, y_train_unscaled, y_test_unscaled, mode=mode, best_params=best_params
                )
                # Collect ALL predictions (folds + averages) and add them in one shot with same weights
                all_fold_predictions = all_fold_predictions + [avg_predictions, w_avg_predictions]
            # for p in all_predictions:
            #     fold_id = p['fold_id']
            #     for part in p['partitions']:
            #         if len(part[1]) > 0:
            #             print(f"Fold {fold_id} - Partition {part[0]}: {part[2].shape}")

            self._add_all_predictions(prediction_store, all_fold_predictions, weights, mode=mode)

        else:
            print(f"\033[91m{WARNING}{WARNING}{WARNING}{WARNING} WARNING: Using test set as validation set (no folds provided) {WARNING}{WARNING}{WARNING}{WARNING}{WARNING}{WARNING}\033[0m")

            model, model_id, score, model_name, prediction_data = self.launch_training(
                dataset, model_config, context, runner, prediction_store,
                X_train, y_train, X_test, y_test, X_test,
                y_train_unscaled, y_test_unscaled, y_test_unscaled,
                loaded_binaries=loaded_binaries, mode=mode
            )
            artifact = self._persist_model(runner, model, model_id)
            binaries.append(artifact)

            # Add predictions for single model case (no weights)
            self._add_all_predictions(prediction_store, [prediction_data], None, mode=mode)

        return binaries


    def launch_training(
        self,
        dataset, model_config, context, runner, prediction_store,
        X_train, y_train, X_val, y_val, X_test,
        y_train_unscaled, y_val_unscaled, y_test_unscaled,
        train_indices=None, val_indices=None, fold_idx=None, best_params=None,
        loaded_binaries=None, mode="train"):

        # Extract model classname from config consistently (works for both train and predict modes)
        # This ensures the same name is used for saving and loading models
        model_classname = self.model_helper.extract_core_name(model_config)

        # In prediction/explain mode, we don't need to instantiate the model yet
        # We just need to extract metadata to find the model in binaries
        if mode in ("predict", "explain"):
            base_model = None  # Will be loaded from binaries later
        else:
            base_model = self._get_model_instance(dataset, model_config)

        # Generate identifiers
        step_id = context['step_id']
        pipeline_name = runner.saver.pipeline_id
        dataset_name = dataset.name
        model_name = model_config.get('name', model_classname)
        operation_counter = runner.next_op()
        new_operator_name = f"{model_name}_{operation_counter}"

        if mode != "predict" and mode != "explain":
            if mode == "finetune":
                if best_params is not None:
                    print(f"Training model {model_name} with: {best_params}...")

                model = self._get_model_instance(dataset, model_config, force_params=best_params)
            else:
                model = self.model_helper.clone_model(base_model)

        else:
            # Load model from binaries
            if loaded_binaries is None:
                raise ValueError("loaded_binaries must be provided in prediction mode")

            # Try to find model with various naming patterns for backward compatibility:
            # 1. Exact match: "Q4_PLS_10_1"
            # 2. With .pkl extension: "Q4_PLS_10_1.pkl"
            # 3. With .joblib extension: "Q4_PLS_10_1.joblib"
            binaries_dict = dict(loaded_binaries)
            model = binaries_dict.get(new_operator_name)
            if model is None:
                model = binaries_dict.get(f"{new_operator_name}.pkl")
            if model is None:
                model = binaries_dict.get(f"{new_operator_name}.joblib")

            if model is None:
                available_keys = list(binaries_dict.keys())
                raise ValueError(
                    f"Model binary for {new_operator_name} not found in loaded_binaries. "
                    f"Available keys: {available_keys}"
                )

            if mode == "explain":
                # print(f"Using model {model_name} for explanation...")
                # print(f"Model ID: {runner.target_model['id']}, Name: {runner.target_model['model_name']}, Step: {runner.target_model['step_idx']}, Fold: {runner.target_model['fold_id']}, Op Counter: {runner.target_model['op_counter']}")
                # print(f"vs Current: Name: {model_name}, Step: {step_id}, Fold: {fold_idx}, Op Counter: {operation_counter}")
                 # Set the model to be explained in the runner for SHAP
                if runner.target_model["model_name"] == model_name and \
                        runner.target_model["step_idx"] == step_id:
                        # runner.target_model["op_counter"] == operation_counter:
                        # runner.target_model["fold_id"] == fold_idx and \
                    # print("✅ Model matches the target model for explanation.")
                    runner._captured_model = (model, self)

        # Apply best params if provided ####TODO RETRIEVE THE HOLD METHOD (construct with params !!!!)
        if best_params is not None:
            if hasattr(model, 'set_params'):
                model.set_params(**best_params)

        # Prepare data in framework format
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
        X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
        X_test_prep, _ = self._prepare_data(X_test, None, context or {})

        # if self.verbose > 0:
        # print("🚀 Training model...")
        # print("Dataset:", dataset_name, "Shape:", X_train.shape)
        if mode != "predict" and mode != "explain":
            trained_model = self._train_model(model, X_train_prep, y_train_prep, X_val_prep, y_val_prep, **model_config.get('train_params', {}))
        else:
            trained_model = model

        # predict y_test_pred, y_train_pred, y_val_pred (these are in scaled space)
        # print(X_train_prep.shape, y_train_prep.shape, X_val_prep.shape, y_val_prep.shape, X_test_prep.shape)
        y_train_pred_scaled = self._predict_model(trained_model, X_train_prep) if y_train_prep.shape[0] > 0 else np.array([])
        y_val_pred_scaled = self._predict_model(trained_model, X_val_prep) if y_val_prep.shape[0] > 0 else np.array([])
        y_test_pred_scaled = self._predict_model(trained_model, X_test_prep) if X_test_prep.shape[0] > 0 else np.array([])
        # print(y_train_pred_scaled.shape, y_val_pred_scaled.shape, y_test_pred_scaled.shape)

        # Transform predictions from scaled space back to unscaled space
        current_y_processing = context.get('y', 'numeric') if context else 'numeric'

        # For classification tasks, keep predictions in the same space as the targets
        # For regression tasks, transform back to numeric space
        if dataset.task_type and 'classification' in dataset.task_type:
            # Keep predictions in the transformed space for classification
            y_train_pred_unscaled = y_train_pred_scaled
            y_val_pred_unscaled = y_val_pred_scaled
            y_test_pred_unscaled = y_test_pred_scaled
        elif current_y_processing != 'numeric':
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

        # print("Predicted fold:", fold_idx, "Shapes:", y_test_pred_unscaled.shape, "Tests", y_test_pred_unscaled[:5])
        # print("UNSCALED PRED:", y_train_pred_unscaled.shape, y_val_pred_unscaled.shape, y_test_pred_unscaled.shape)

        metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
        _direction = ARROW_UP if higher_is_better else ARROW_DOWN
        score_train = Evaluator.eval(y_train_unscaled, y_train_pred_unscaled, metric)
        score_val = Evaluator.eval(y_val_unscaled, y_val_pred_unscaled, metric)
        score_test = Evaluator.eval(y_test_unscaled, y_test_pred_unscaled, metric)

        # print(f"{CHART} {model_name} scores: Train {metric} {direction} {score_train:.4f}, Val {metric} {direction} {score_val:.4f}, Test {metric} {direction} {score_test:.4f}")

        if train_indices is None:
            train_indices = list(range(len(y_train_unscaled)))
        else:
            train_indices = [int(idx) for idx in train_indices]  # Convert numpy int types to Python int

        if val_indices is None:
            val_indices = list(range(len(y_val_unscaled)))
        else:
            val_indices = [int(idx) for idx in val_indices]  # Convert numpy int types to Python int

        test_indices = list(range(len(y_test_unscaled)))

        # Prepare prediction data for return (don't store yet)
        pipeline_uid = getattr(runner, 'pipeline_uid', None)
        prediction_data = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_name,
            'config_name': pipeline_name,
            'config_path': f"{dataset_name}/{pipeline_name}",
            'pipeline_uid': pipeline_uid,
            'step_idx': step_id,
            'op_counter': operation_counter,
            'model_name': model_name,
            'model_classname': model_classname,
            'model_path': f"{dataset_name}/{pipeline_name}/{step_id}_{model_name}_{operation_counter}.pkl",
            'fold_id': fold_idx,
            'val_score': score_val,
            'test_score': score_test,
            'train_score': score_train,
            'metric': metric,
            'task_type': dataset.task_type,
            'n_features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'preprocessings': dataset.short_preprocessings_str(),
            'best_params': {} if best_params is None else str(best_params),
            'partitions': [
                ("train", train_indices, y_train_unscaled, y_train_pred_unscaled),
                ("val", val_indices, y_val_unscaled, y_val_pred_unscaled),
                ("test", test_indices, y_test_unscaled, y_test_pred_unscaled)
            ]
        }
        # if y_test_pred_unscaled.shape[0] > 0:
        #     # print("{CHART} y_test_pred_unscaled:")
        #     print(f"mean: {np.mean(y_test_pred_unscaled):.4f}, std: {np.std(y_test_pred_unscaled):.4f}, min: {np.min(y_test_pred_unscaled):.4f}, max: {np.max(y_test_pred_unscaled):.4f}")

        return trained_model, f"{model_name}_{operation_counter}", score_val, model_name, prediction_data



    def _detect_task_type(self, y: Any) -> TaskType:
        """Detect task type from target values."""
        return ModelUtils.detect_task_type(y)

    def _calculate_and_print_scores(
        self,
        y_true: Any,
        y_pred: Any,
        task_type: TaskType,
        partition: str = "test",
        model_name: str = "model",
        show_detailed_scores: bool = True
    ) -> Dict[str, float]:
        """Calculate scores and print them."""
        scores = ModelUtils.calculate_scores(y_true, y_pred, task_type)
        if scores and show_detailed_scores:
            score_str = ModelUtils.format_scores(scores)
            print(f"{CHART} {model_name} {partition} scores: {score_str}")
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
        loaded_binaries: Optional[List[Tuple[str, Any]]]
    ) -> Tuple[Dict[str, Any], List['ArtifactMeta']]:
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

        # TODO: This should use persist_artifact for predictions CSV
        # For now, return empty list since predictions are handled separately
        return context, []

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from step or operator."""
        # # DEBUG
        # print(f"DEBUG _extract_model_config called")
        # print(f"  step: {step if not callable(step) else '<callable>'}")
        # print(f"  operator: {operator if not callable(operator) else '<callable>'}")
        # print(f"  operator is not None: {operator is not None}")

        if operator is not None:
            # print(f"DEBUG operator branch taken")
            if isinstance(step, dict):
                config = step.copy()
                config['model_instance'] = operator

                # Preserve function/class keys from nested model structure for name extraction
                if 'model' in step and isinstance(step['model'], dict):
                    if 'function' in step['model']:
                        config['function'] = step['model']['function']
                    elif 'class' in step['model']:
                        config['class'] = step['model']['class']

                # print(f"DEBUG returning config (step is dict): {list(config.keys())}")
                return config
            else:
                # print(f"DEBUG returning model_instance wrapper")
                return {'model_instance': operator}

        if isinstance(step, dict):
            # If step is already a serialized format with 'function', 'class', or 'import',
            # pass it through as-is for ModelBuilderFactory
            if any(key in step for key in ('function', 'class', 'import')):
                # print(f"DEBUG returning step as-is: {step}")
                return step

            if 'model' in step:
                config = step.copy()
                model_obj = step['model']

                # Handle nested model format
                if isinstance(model_obj, dict):
                    if 'model' in model_obj:
                        config['model_instance'] = model_obj['model']
                        if 'name' in model_obj:
                            config['name'] = model_obj['name']
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

    def _load_model_from_binaries(self, loaded_binaries: List[Tuple[str, Any]]) -> Any:
        """Load trained model from loaded artifacts.

        Note: The BinaryLoader already deserializes the objects using the new
        serializer infrastructure, so we just need to find the right artifact.

        Args:
            loaded_binaries: List of (name, loaded_object) tuples

        Returns:
            The loaded model object
        """
        model_obj = None
        for name, obj in loaded_binaries:
            if name.endswith('.pkl') and ('model' in name.lower() or 'trained' in name.lower()):
                model_obj = obj
                break

        if model_obj is None:
            raise ValueError("No model object found in loaded binaries")

        return model_obj

    def _create_fold_averages(
        self,
        base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
        folds_models, fold_val_indices, scores,
        X_train, X_test, y_train_unscaled, y_test_unscaled,
        mode="train", best_params=None
    ) -> Tuple[Dict, Dict]:

        # X_val is the concatenation of all fold val sets
        X_val = np.vstack([X_train[val_idx] for val_idx in fold_val_indices])
        # print("Val shape for averages:", X_val.shape)
        y_val_unscaled = np.vstack([y_train_unscaled[val_idx] for val_idx in fold_val_indices])
        # print("Val shape for averages:", y_val_unscaled.shape)
        all_val_indices = np.hstack(fold_val_indices)
        # print("Val indices shape for averages:", all_val_indices.shape)

        # Generate all predictions for train, val, test for each fold model
        all_train_preds = []
        all_val_preds = []
        all_test_preds = []

        for fold_model_tuple in folds_models:
            # Extract the actual model from the tuple (model_id, model, score)
            _, fold_model, _ = fold_model_tuple
            fold_train_preds = self._predict_model(fold_model, X_train) if X_train.shape[0] > 0 else np.array([])
            fold_val_preds = self._predict_model(fold_model, X_val) if X_val.shape[0] > 0 else np.array([])
            fold_test_preds = self._predict_model(fold_model, X_test) if X_test.shape[0] > 0 else np.array([])

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

        # print("Predicted fold:", 'avg', "Shapes:", all_test_avg_preds.shape, "Tests", all_test_avg_preds[:5])

        score_val = 0.0
        score_test = 0.0
        score_train = 0.0
        metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
        if mode != "predict" and mode != "explain":
            _direction = ARROW_UP if higher_is_better else ARROW_DOWN
            # Evaluate average predictions
            score_train = Evaluator.eval(y_train_unscaled, all_train_avg_preds, metric)
            score_val = Evaluator.eval(y_val_unscaled, all_val_avg_preds, metric)
            score_test = Evaluator.eval(y_test_unscaled, all_test_avg_preds, metric)

        avg_counter = runner.next_op()

        # Prepare average predictions data for return
        prediction_array = []
        prediction_array.append(("train", list(range(len(y_train_unscaled))), y_train_unscaled, all_train_avg_preds))
        if mode != "predict" and mode != "explain":
            prediction_array.append(("val", all_val_indices.tolist(), y_val_unscaled, all_val_avg_preds))
        prediction_array.append(("test", list(range(len(y_test_unscaled))), y_test_unscaled, all_test_avg_preds))

        avg_predictions = {
            'dataset_name': dataset.name,
            'dataset_path': dataset.name,
            'config_name': runner.saver.pipeline_id,
            'config_path': f"{dataset.name}/{runner.saver.pipeline_id}",
            'pipeline_uid': getattr(runner, 'pipeline_uid', None),
            'step_idx': context['step_id'],
            'op_counter': avg_counter,
            'model_name': f"{base_model_name}",
            'model_classname': str(model_classname),
            'model_path': "",
            'fold_id': "avg",
            'val_score': score_val,
            'test_score': score_test,
            'train_score': score_train,
            'metric': metric,
            'task_type': dataset.task_type,
            'n_features': X_train.shape[1],
            'preprocessings': dataset.short_preprocessings_str(),
            'partitions': prediction_array,
            'best_params': {} if best_params is None else str(best_params),
        }

        # Weighted average predictions based on fold scores
        scores = np.asarray(scores, dtype=float)

        if mode == "predict" or mode == "explain":
            # Check if weights exist in target_model (they only exist for avg/w_avg predictions)
            if "weights" in runner.target_model and runner.target_model["weights"] is not None:
                weights_from_model = runner.target_model["weights"]
                # Handle different formats: list, string (JSON serialized), or numpy array
                if isinstance(weights_from_model, str):
                    # Parse JSON string
                    import json
                    weights = np.array(json.loads(weights_from_model), dtype=float)
                elif isinstance(weights_from_model, list):
                    weights = np.array(weights_from_model, dtype=float)
                else:
                    weights = np.asarray(weights_from_model, dtype=float)
            else:
                # Fallback: use equal weights when scores are NaN (prediction mode)
                if np.all(np.isnan(scores)):
                    weights = np.ones(len(scores), dtype=float) / len(scores)
                else:
                    weights = ModelUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)
        else:
            weights = ModelUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)

        all_train_w_avg_preds = np.zeros_like(all_train_preds[0], dtype=float)
        for arr, weight in zip(all_train_preds, weights):
            all_train_w_avg_preds += weight * arr

        if mode != "predict" and mode != "explain":
            all_val_w_avg_preds = np.zeros_like(all_val_preds[0], dtype=float)
            for arr, weight in zip(all_val_preds, weights):
                all_val_w_avg_preds += weight * arr

        all_test_w_avg_preds = np.zeros_like(all_test_preds[0], dtype=float)
        for arr, weight in zip(all_test_preds, weights):
            all_test_w_avg_preds += weight * arr

        # print("Predicted fold:", 'w_avg', "Shapes:", all_test_w_avg_preds.shape, "Tests", all_test_w_avg_preds[:5])

        # Evaluate weighted average predictions
        score_val_w = 0.0
        score_test_w = 0.0
        score_train_w = 0.0
        if mode != "predict" and mode != "explain":
            score_train_w = Evaluator.eval(y_train_unscaled, all_train_w_avg_preds, metric)
            score_val_w = Evaluator.eval(y_val_unscaled, all_val_w_avg_preds, metric)
            score_test_w = Evaluator.eval(y_test_unscaled, all_test_w_avg_preds, metric)
        w_avg_counter = runner.next_op()

        # Prepare weighted average predictions data for return
        prediction_array_w = []
        prediction_array_w.append(("train", list(range(len(y_train_unscaled))), y_train_unscaled, all_train_w_avg_preds))
        if mode != "predict" and mode != "explain":
            prediction_array_w.append(("val", all_val_indices.tolist(), y_val_unscaled, all_val_w_avg_preds))
        prediction_array_w.append(("test", list(range(len(y_test_unscaled))), y_test_unscaled, all_test_w_avg_preds))

        w_avg_predictions = {
            'dataset_name': dataset.name,
            'dataset_path': dataset.name,
            'config_name': runner.saver.pipeline_id,
            'config_path': f"{dataset.name}/{runner.saver.pipeline_id}",
            'pipeline_uid': getattr(runner, 'pipeline_uid', None),
            'step_idx': context['step_id'],
            'op_counter': w_avg_counter,
            'model_name': f"{base_model_name}",
            'model_classname': str(model_classname),
            'model_path': "",
            'fold_id': 'w_avg',
            'val_score': score_val_w,
            'test_score': score_test_w,
            'train_score': score_train_w,
            'metric': metric,
            'task_type': dataset.task_type,
            'n_features': X_train.shape[1],
            'preprocessings': dataset.short_preprocessings_str(),
            'weights': weights.tolist(),
            'partitions': prediction_array_w,
            'best_params': {} if best_params is None else str(best_params),
        }

        return avg_predictions, w_avg_predictions

    def _add_all_predictions(self, prediction_store, all_predictions, weights, mode="train"):
        """Add all predictions with the same weights array."""
        for prediction_data in all_predictions:
            if prediction_data is None:
                continue

            # Print the model description once per prediction_data
            model_name = prediction_data['model_name']
            fold_id = prediction_data['fold_id']
            op_counter = prediction_data['op_counter']
            val_score = prediction_data['val_score']
            test_score = prediction_data['test_score']
            metric = prediction_data['metric']

            # Determine direction symbol based on metric (assume lower is better for most metrics)
            direction = ARROW_UP if metric in ['r2', 'accuracy'] else ARROW_DOWN

            first_partition = True

            for partition_name, indices, y_true_part, y_pred_part in prediction_data['partitions']:
                if len(indices) == 0:
                    continue
                # print(f"Adding predictions for fold {fold_id}, partition {partition_name} with {len(indices)} samples.")
                pred_id = prediction_store.add_prediction(
                    dataset_name=prediction_data['dataset_name'],
                    dataset_path=prediction_data['dataset_path'],
                    config_name=prediction_data['config_name'],
                    config_path=prediction_data['config_path'],
                    pipeline_uid=prediction_data.get('pipeline_uid'),
                    step_idx=prediction_data['step_idx'],
                    op_counter=prediction_data['op_counter'],
                    model_name=prediction_data['model_name'],
                    model_classname=prediction_data['model_classname'],
                    model_path=prediction_data['model_path'],
                    fold_id=prediction_data['fold_id'],
                    sample_indices=indices,
                    weights=weights,  # ALL predictions get the SAME weights array
                    metadata={},
                    partition=partition_name,
                    y_true=y_true_part,
                    y_pred=y_pred_part,
                    val_score=prediction_data['val_score'],
                    test_score=prediction_data['test_score'],
                    train_score=prediction_data['train_score'],
                    metric=prediction_data['metric'],
                    task_type=prediction_data['task_type'],
                    n_samples=len(y_true_part),
                    n_features=prediction_data['n_features'],
                    preprocessings=prediction_data['preprocessings'],
                    best_params=prediction_data['best_params']
                )

                # Print only once per prediction_data (for the first partition)
                if first_partition:
                    short_desc = f"{CHECK}{model_name}"
                    if mode != "predict" and mode != "explain":
                        short_desc += f" {metric} {direction}"
                        short_desc += f" [test: {test_score:.4f}], [val: {val_score:.4f}]"
                    else:
                        short_desc += f" from (step: {prediction_data['step_idx']}, "

                    if fold_id not in [None, 'None', 'avg', 'w_avg']:
                        short_desc += f", (fold: {fold_id}, id: {op_counter})"
                    elif fold_id in ['avg', 'w_avg']:
                        short_desc += f", ({fold_id}, id: {op_counter})"

                    short_desc += f" - [{pred_id}]"
                    if mode != "predict" and mode != "explain":
                        print(short_desc)
                    first_partition = False

    def _persist_model(self, runner: 'PipelineRunner', model: Any, model_id: str) -> 'ArtifactMeta':
        """Persist model using the new serializer infrastructure.

        Args:
            runner: Pipeline runner with saver instance
            model: The trained model to persist
            model_id: Identifier for the model

        Returns:
            ArtifactMeta: Metadata about the persisted artifact
        """
        # Detect framework hint from model type
        model_type = type(model).__module__
        if 'sklearn' in model_type:
            format_hint = 'sklearn'
        elif 'tensorflow' in model_type or 'keras' in model_type:
            format_hint = 'tensorflow'
        elif 'torch' in model_type:
            format_hint = 'pytorch'
        elif 'xgboost' in model_type:
            format_hint = 'xgboost'
        elif 'catboost' in model_type:
            format_hint = 'catboost'
        elif 'lightgbm' in model_type:
            format_hint = 'lightgbm'
        else:
            format_hint = None  # Let serializer auto-detect

        return runner.saver.persist_artifact(
            step_number=runner.step_number,
            name=f"{model_id}.pkl",
            obj=model,
            format_hint=format_hint
        )


