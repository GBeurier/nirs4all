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
from ...optimization.optuna import OptunaManager
from nirs4all.data.predictions import Predictions
from nirs4all.data.ensemble_utils import EnsembleUtils
from nirs4all.core.task_type import TaskType
from .utilities import ModelControllerUtils as ModelUtils
from nirs4all.core import metrics as evaluator
from nirs4all.utils.emoji import CHECK, ARROW_UP, ARROW_DOWN, SEARCH, FOLDER, CHART, WEIGHT_LIFT, WARNING
from nirs4all.pipeline.storage.artifacts.artifact_persistence import ArtifactMeta
from .components import (
    ModelIdentifierGenerator,
    PredictionTransformer,
    PredictionDataAssembler,
    ModelLoader,
    ScoreCalculator,
    IndexNormalizer
)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep


class BaseModelController(OperatorController, ABC):
    """Abstract base controller for machine learning model training and prediction.

    This controller provides a unified interface for training, finetuning, and predicting
    with machine learning models across different frameworks (scikit-learn, TensorFlow, PyTorch).
    It implements cross-validation, fold averaging, hyperparameter optimization, and
    comprehensive prediction tracking.

    The controller delegates framework-specific operations to subclasses while handling:
        - Cross-validation fold management
        - Model identification and naming
        - Prediction storage and tracking
        - Score calculation and aggregation
        - Fold-averaged predictions (simple and weighted)

    Attributes:
        optuna_manager (OptunaManager): Manager for hyperparameter optimization.
        identifier_generator (ModelIdentifierGenerator): Component for model naming.
        prediction_transformer (PredictionTransformer): Component for prediction scaling.
        prediction_assembler (PredictionDataAssembler): Component for assembling prediction records.
        model_loader (ModelLoader): Component for loading persisted models.
        score_calculator (ScoreCalculator): Component for calculating evaluation scores.
        index_normalizer (IndexNormalizer): Component for normalizing sample indices.
        prediction_store (Predictions): External storage for predictions.
        verbose (int): Verbosity level for logging.
    """

    priority = 15

    def __init__(self):
        super().__init__()
        self.optuna_manager = OptunaManager()

        # Initialize components for modular operations
        self.identifier_generator = ModelIdentifierGenerator()
        self.prediction_transformer = PredictionTransformer()
        self.prediction_assembler = PredictionDataAssembler()
        self.model_loader = ModelLoader()
        self.score_calculator = ScoreCalculator()
        self.index_normalizer = IndexNormalizer()

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    # Abstract methods that subclasses must implement for their frameworks
    @abstractmethod
    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create model instance from configuration using framework-specific builder.

        Args:
            dataset: SpectroDataset containing training data and metadata.
            model_config: Model configuration dictionary with architecture and parameters.
            force_params: Optional parameters to override config values (used in finetuning).

        Returns:
            Framework-specific model instance ready for training.
        """
        pass

    @abstractmethod
    def _train_model(self, model: Any, X_train: Any, y_train: Any,
                     X_val: Any = None, y_val: Any = None, **kwargs) -> Any:
        """Train the model using framework-specific training logic.

        Args:
            model: Model instance to train.
            X_train: Training features.
            y_train: Training targets.
            X_val: Optional validation features.
            y_val: Optional validation targets.
            **kwargs: Additional framework-specific training parameters.

        Returns:
            Trained model instance.
        """
        pass

    @abstractmethod
    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions using framework-specific prediction logic.

        Args:
            model: Trained model instance.
            X: Input features for prediction.

        Returns:
            NumPy array of predictions.
        """
        pass

    @abstractmethod
    def _prepare_data(self, X: Any, y: Any, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """Prepare data in framework-specific format (e.g., tensors, DataFrames).

        Args:
            X: Input features to prepare.
            y: Target values to prepare.
            context: Execution context with preprocessing and partition information.

        Returns:
            Tuple of (prepared_X, prepared_y) in framework-specific format.
        """
        pass

    @abstractmethod
    def _clone_model(self, model: Any) -> Any:
        """Clone model using framework-specific cloning method.

        Each framework has its own best practice for cloning models:
        - sklearn: use sklearn.base.clone()
        - tensorflow/keras: use keras.models.clone_model()
        - pytorch: use copy.deepcopy() or custom cloning

        Args:
            model: Model instance to clone.

        Returns:
            Cloned model instance with same architecture but fresh weights.
        """
        pass

    @abstractmethod
    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate model performance for hyperparameter optimization.

        Args:
            model: Trained model instance to evaluate.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Validation score to minimize (e.g., RMSE, negative accuracy).
        """
        pass

    def save_model(self, model: Any, filepath: str) -> None:
        """Optional: Save model in framework-specific format.

        Default implementation delegates to artifact_serialization.persist().
        Subclasses can override to use framework-specific formats:
        - TensorFlow: .h5 or .keras format
        - PyTorch: .ckpt or .pt format
        - sklearn: .joblib format

        Args:
            model: Trained model to save.
            filepath: Path to save (without extension, will be added by implementation).
        """
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist
        persist(model, filepath)

    def load_model(self, filepath: str) -> Any:
        """Optional: Load model from framework-specific format.

        Default implementation delegates to artifact_serialization.load().
        Subclasses can override to use framework-specific loading.

        Args:
            filepath: Path to load from.

        Returns:
            Loaded model instance.
        """
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import load
        return load(filepath)

    def get_xy(self, dataset: 'SpectroDataset', context: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Extract train/test splits with scaled and unscaled targets.

        For classification tasks, both scaled and unscaled targets are transformed.
        For regression tasks, scaled targets are used for training while unscaled
        (numeric) targets are used for evaluation.

        In prediction mode, uses all available data (partition=None) instead of splitting.

        Args:
            dataset: SpectroDataset with partitioned data.
            context: Execution context with partition and preprocessing info.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled).
        """
        layout = self.get_preferred_layout()

        # Check if we're in prediction/explain mode
        mode = context.get('mode', 'train') if isinstance(context, dict) else (context.state.mode if hasattr(context, 'state') else 'train')

        if mode in ("predict", "explain"):
            # In prediction mode, use all available data (no partition split)
            if isinstance(context, dict):
                pred_context = copy.deepcopy(context)
                pred_context['partition'] = None
            else:
                # ExecutionContext - use with_partition method
                pred_context = context.with_partition(None)

            X_all = dataset.x(pred_context, layout=layout)
            y_all = dataset.y(pred_context)

            # Return empty training data and all data as "test" for prediction
            empty_X = np.array([]).reshape(0, X_all.shape[1] if len(X_all.shape) > 1 else 0)
            empty_y = np.array([])

            # For unscaled targets
            if dataset.task_type and dataset.task_type.is_classification:
                y_all_unscaled = y_all
            else:
                # For regression, get numeric (unscaled) targets
                # Build selector dict for y() call
                if isinstance(pred_context, dict):
                    pred_context['y'] = 'numeric'
                    y_all_unscaled = dataset.y(pred_context)
                else:
                    # ExecutionContext - convert to dict for dataset.y()
                    y_selector = {'partition': None, 'y': 'numeric'}
                    y_all_unscaled = dataset.y(y_selector)

            return empty_X, empty_y, X_all, y_all, empty_y, y_all_unscaled

        # Normal training mode: split into train/test
        if isinstance(context, dict):
            train_context = copy.deepcopy(context)
            train_context['partition'] = 'train'
            test_context = copy.deepcopy(context)
            test_context['partition'] = 'test'
        else:
            train_context = context.with_partition('train')
            test_context = context.with_partition('test')

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
            if isinstance(train_context, dict):
                train_context['y'] = 'numeric'
                test_context['y'] = 'numeric'
            else:
                train_context = train_context.with_y('numeric')
                test_context = test_context.with_y('numeric')

            y_train_unscaled = dataset.y(train_context)
            y_test_unscaled = dataset.y(test_context)
        return X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled


    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: 'Predictions' = None
    ) -> Tuple[Dict[str, Any], List['ArtifactMeta']]:
        """Execute model training, finetuning, or prediction.

        This is the main entry point for model execution. It handles:
            - Extracting model configuration
            - Restoring task type in predict/explain modes
            - Delegating to finetune() or train() based on configuration
            - Managing prediction storage

        Args:
            step_info: Parsed step containing model configuration and operator.
            dataset: SpectroDataset with features and targets.
            context: Execution context with step_id, partition info, etc.
            runner: PipelineRunner managing pipeline execution.
            source: Data source index (default: -1).
            mode: Execution mode ('train', 'finetune', 'predict', 'explain').
            loaded_binaries: Optional list of (name, bytes) tuples for prediction mode.
            prediction_store: External Predictions storage instance.

        Returns:
            Tuple of (updated_context, list_of_artifact_metadata).
        """
        # Extract for compatibility with existing code
        step = step_info.original_step
        operator = step_info.operator

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
        """Optimize hyperparameters using Optuna.

        Delegates to OptunaManager for Bayesian hyperparameter optimization.
        Returns optimized parameters that will be used in subsequent training.

        Args:
            dataset: SpectroDataset for optimization.
            model_config: Base model configuration.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            folds: List of (train_idx, val_idx) tuples for cross-validation.
            finetune_params: Optuna configuration with search space and trials.
            predictions: Prediction storage dictionary.
            context: Execution context.
            runner: PipelineRunner instance.

        Returns:
            Dictionary of optimized parameters (single model) or list of dicts (per-fold).
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
        """Orchestrate model training across folds with prediction tracking.

        Manages the complete training workflow:
            - Iterates through cross-validation folds
            - Delegates to launch_training() for each fold
            - Creates fold-averaged predictions for regression tasks
            - Persists trained models as artifacts
            - Stores all predictions with weights

        Args:
            dataset: SpectroDataset with features and targets.
            model_config: Model configuration dictionary.
            context: Execution context with step_id and preprocessing info.
            runner: PipelineRunner managing execution.
            prediction_store: External Predictions storage.
            X_train: Training features (all folds).
            y_train: Training targets (scaled).
            X_test: Test features.
            y_test: Test targets (scaled).
            y_train_unscaled: Training targets (unscaled for evaluation).
            y_test_unscaled: Test targets (unscaled for evaluation).
            folds: List of (train_idx, val_idx) tuples or empty list.
            best_params: Optional hyperparameters from finetuning.
            loaded_binaries: Optional model binaries for prediction mode.
            mode: Execution mode ('train', 'finetune', 'predict', 'explain').

        Returns:
            List of ArtifactMeta objects for persisted models.
        """

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
            weights = EnsembleUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)

            # Create fold averages and get average predictions data
            if dataset.task_type and dataset.task_type.is_regression:
                avg_predictions, w_avg_predictions = self._create_fold_averages(
                    base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
                    folds_models, fold_val_indices, scores,
                    X_train, X_test, y_train_unscaled, y_test_unscaled, mode=mode, best_params=best_params
                )
                # Collect ALL predictions (folds + averages) and add them in one shot with same weights
                all_fold_predictions = all_fold_predictions + [avg_predictions, w_avg_predictions]

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
        """Execute single model training or prediction.

        This refactored method uses modular components to handle:
        - Model identification and naming
        - Model loading for predict/explain modes
        - Training execution
        - Prediction transformation
        - Score calculation
        - Prediction data assembly

        Args:
            dataset: SpectroDataset instance
            model_config: Model configuration dictionary
            context: Execution context with step_id, y processing, etc.
            runner: PipelineRunner instance
            prediction_store: Predictions storage instance
            X_train, y_train: Training data (scaled)
            X_val, y_val: Validation data (scaled)
            X_test: Test data (scaled)
            y_train_unscaled, y_val_unscaled, y_test_unscaled: True values (unscaled)
            train_indices, val_indices: Sample indices for each partition
            fold_idx: Optional fold index for CV
            best_params: Optional hyperparameters from optimization
            loaded_binaries: Optional binaries for predict/explain mode
            mode: Execution mode ('train', 'finetune', 'predict', 'explain')

        Returns:
            Tuple of (trained_model, model_id, val_score, model_name, prediction_data)
        """
        # === 1. GENERATE IDENTIFIERS ===
        identifiers = self.identifier_generator.generate(model_config, runner, context, fold_idx)

        # Debug: check identifiers
        if identifiers.step_id == '' or identifiers.step_id == 0:
            print(f"\n⚠️  WARNING in launch_training: step_id={identifiers.step_id}")
            print(f"   context.get('step_number')={context.get('step_number', 'NOT_FOUND')}")
            print(f"   context.state.step_number={context.state.step_number if hasattr(context, 'state') else 'NO_STATE'}")

        # === 2. GET OR LOAD MODEL ===
        if mode in ("predict", "explain"):
            # Load from binaries
            if loaded_binaries is None:
                raise ValueError("loaded_binaries must be provided in prediction mode")
            model = self.model_loader.load(identifiers.model_id, loaded_binaries, fold_idx)

            # Capture model for SHAP explanation
            if mode == "explain" and self._should_capture_for_explanation(runner, identifiers):
                if hasattr(runner, 'explainer') and hasattr(runner.explainer, 'capture_model'):
                    runner.explainer.capture_model(model, self)

            trained_model = model
        else:
            # Create new model for training
            if mode == "finetune" and best_params is not None:
                if self.verbose > 0:
                    print(f"Training model {identifiers.name} with: {best_params}...")
                model = self._get_model_instance(dataset, model_config, force_params=best_params)
            else:
                base_model = self._get_model_instance(dataset, model_config)
                model = self._clone_model(base_model)

            # === 3. TRAIN MODEL ===
            X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
            X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
            X_test_prep, _ = self._prepare_data(X_test, None, context or {})

            # Pass task_type to train_model
            train_params = model_config.get('train_params', {}).copy()
            train_params['task_type'] = dataset.task_type

            trained_model = self._train_model(
                model, X_train_prep, y_train_prep, X_val_prep, y_val_prep,
                **train_params
            )

        # === 4. GENERATE PREDICTIONS (scaled) ===
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
        X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
        X_test_prep, _ = self._prepare_data(X_test, None, context or {})

        # Generate predictions for all partitions with data (based on X, not y)
        predictions_scaled = {
            'train': self._predict_model(trained_model, X_train_prep) if X_train_prep.shape[0] > 0 else np.array([]),
            'val': self._predict_model(trained_model, X_val_prep) if X_val_prep.shape[0] > 0 else np.array([]),
            'test': self._predict_model(trained_model, X_test_prep) if X_test_prep.shape[0] > 0 else np.array([])
        }

        # === 5. TRANSFORM PREDICTIONS TO UNSCALED ===
        predictions_unscaled = {
            'train': self.prediction_transformer.transform_to_unscaled(predictions_scaled['train'], dataset, context),
            'val': self.prediction_transformer.transform_to_unscaled(predictions_scaled['val'], dataset, context),
            'test': self.prediction_transformer.transform_to_unscaled(predictions_scaled['test'], dataset, context)
        }

        # === 6. CALCULATE SCORES ===
        true_values = {
            'train': y_train_unscaled,
            'val': y_val_unscaled,
            'test': y_test_unscaled
        }

        partition_scores = self.score_calculator.calculate(
            true_values,
            predictions_unscaled,
            dataset.task_type
        )

        # === 7. NORMALIZE INDICES ===
        # In predict mode with no y, use X shape to determine sample counts
        n_samples = {
            'train': len(y_train_unscaled) if y_train_unscaled is not None and len(y_train_unscaled) > 0 else (len(y_train_prep) if y_train_prep is not None and len(y_train_prep) > 0 else X_train_prep.shape[0]),
            'val': len(y_val_unscaled) if y_val_unscaled is not None and len(y_val_unscaled) > 0 else (len(y_val_prep) if y_val_prep is not None and len(y_val_prep) > 0 else X_val_prep.shape[0]),
            'test': len(y_test_unscaled) if y_test_unscaled is not None and len(y_test_unscaled) > 0 else X_test_prep.shape[0]
        }

        indices = {
            'train': self.index_normalizer.normalize(train_indices, n_samples['train']),
            'val': self.index_normalizer.normalize(val_indices, n_samples['val']),
            'test': self.index_normalizer.normalize(None, n_samples['test'])
        }

        # === 8. ASSEMBLE PREDICTION DATA ===
        scores_dict = {
            'train': partition_scores.train,
            'val': partition_scores.val,
            'test': partition_scores.test,
            'metric': partition_scores.metric
        }

        prediction_data = self.prediction_assembler.assemble(
            dataset=dataset,
            identifiers=identifiers,
            scores=scores_dict,
            predictions=predictions_unscaled,
            true_values=true_values,
            indices=indices,
            runner=runner,
            X_shape=X_train.shape,
            best_params=best_params
        )

        return trained_model, identifiers.model_id, partition_scores.val, identifiers.name, prediction_data

    def _should_capture_for_explanation(self, runner, identifiers) -> bool:
        """Check if current model should be captured for SHAP explanation.

        Compares model name and step index with runner's target_model to determine
        if this is the model requiring explanation.

        Args:
            runner: PipelineRunner with target_model info.
            identifiers: ModelIdentifiers with name and step_id.

        Returns:
            True if model should be captured for explanation, False otherwise.
        """
        target = runner.target_model
        # Convert both to string for comparison to handle int/string mismatch
        target_step = str(target["step_idx"])
        ident_step = str(identifiers.step_id)
        return (target["model_name"] == identifiers.name and
                target_step == ident_step)



    def _print_prediction_summary(self, prediction_data, pred_id, mode):
        """Print formatted summary for a single prediction.

        Displays model name, metric, test/val scores, and fold information
        with appropriate directional indicators (↑ for metrics to maximize, ↓ to minimize).

        Args:
            prediction_data: Prediction dictionary with scores and metadata.
            pred_id: Unique prediction identifier.
            mode: Execution mode.
        """
        model_name = prediction_data['model_name']
        fold_id = prediction_data['fold_id']
        op_counter = prediction_data['op_counter']
        val_score = prediction_data['val_score']
        test_score = prediction_data['test_score']
        metric = prediction_data['metric']
        direction = ARROW_UP if metric in ['r2', 'accuracy'] else ARROW_DOWN

        summary = f"{CHECK}{model_name} {metric} {direction} [test: {test_score:.4f}], [val: {val_score:.4f}]"
        if fold_id not in [None, 'None', 'avg', 'w_avg']:
            summary += f", (fold: {fold_id}, id: {op_counter})"
        elif fold_id in ['avg', 'w_avg']:
            summary += f", ({fold_id}, id: {op_counter})"
        summary += f" - [{pred_id}]"
        print(summary)

    def get_preferred_layout(self) -> str:
        """Get preferred data layout for the framework.

        Returns:
            Data layout string ('2d' for NumPy arrays, '3d' for TensorFlow, etc.).

        Note:
            Override in subclasses for framework-specific layouts.
        """
        return "2d"

    def _calculate_and_print_scores(
        self,
        y_true: Any,
        y_pred: Any,
        task_type: TaskType,
        partition: str = "test",
        model_name: str = "model",
        show_detailed_scores: bool = True
    ) -> Dict[str, float]:
        """Calculate evaluation scores and print formatted output.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            task_type: TaskType enum indicating regression or classification.
            partition: Partition name for display ('train', 'val', 'test').
            model_name: Model name for display.
            show_detailed_scores: Whether to print detailed score breakdown.

        Returns:
            Dictionary of metric names and scores.
        """
        scores = evaluator.eval_multi(y_true, y_pred, task_type.value)
        if scores and show_detailed_scores:
            score_str = ModelUtils.format_scores(scores)
            print(f"{CHART} {model_name} {partition} scores: {score_str}")
        return scores

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract and normalize model configuration from step or operator.

        Handles various configuration formats:
            - Dictionary with 'model' key
            - Dictionary with 'function'/'class'/'import' keys (serialized)
            - Direct model instance
            - Nested model dictionaries

        Args:
            step: Pipeline step configuration (dict or model instance).
            operator: Optional model operator instance.

        Returns:
            Normalized configuration dictionary with 'model_instance' or builder keys.
        """

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
            # pass it through as-is for ModelFactory
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



    def _create_fold_averages(
        self,
        base_model_name, dataset, model_config, context, runner, prediction_store, model_classname,
        folds_models, fold_val_indices, scores,
        X_train, X_test, y_train_unscaled, y_test_unscaled,
        mode="train", best_params=None
    ) -> Tuple[Dict, Dict]:
        """Create simple and weighted fold-averaged predictions.

        Generates two averaged predictions:
            1. Simple average: Equal weight to all folds
            2. Weighted average: Weights based on validation scores

        Uses modular components for prediction transformation and score calculation.

        Args:
            base_model_name: Base name for averaged models.
            dataset: SpectroDataset with task type and preprocessing info.
            model_config: Model configuration dictionary.
            context: Execution context.
            runner: PipelineRunner instance.
            prediction_store: Predictions storage.
            model_classname: Model class name string.
            folds_models: List of (model_id, model, score) tuples from folds.
            fold_val_indices: List of validation indices for each fold.
            scores: List of validation scores for each fold.
            X_train: Training features (all folds).
            X_test: Test features.
            y_train_unscaled: Training targets (unscaled).
            y_test_unscaled: Test targets (unscaled).
            mode: Execution mode.
            best_params: Optional hyperparameters.

        Returns:
            Tuple of (avg_prediction_dict, weighted_avg_prediction_dict).
        """

        # Prepare validation data
        X_val = np.vstack([X_train[val_idx] for val_idx in fold_val_indices])
        y_val_unscaled = np.vstack([y_train_unscaled[val_idx] for val_idx in fold_val_indices])
        all_val_indices = np.hstack(fold_val_indices)

        # Collect predictions from all folds (using prediction_transformer component)
        all_train_preds = []
        all_val_preds = []
        all_test_preds = []

        for _, fold_model, _ in folds_models:
            preds_scaled = {
                'train': self._predict_model(fold_model, X_train) if X_train.shape[0] > 0 else np.array([]),
                'val': self._predict_model(fold_model, X_val) if X_val.shape[0] > 0 else np.array([]),
                'test': self._predict_model(fold_model, X_test) if X_test.shape[0] > 0 else np.array([])
            }

            # Use prediction_transformer component for unscaling
            preds_unscaled = {
                'train': self.prediction_transformer.transform_to_unscaled(preds_scaled['train'], dataset, context),
                'val': self.prediction_transformer.transform_to_unscaled(preds_scaled['val'], dataset, context),
                'test': self.prediction_transformer.transform_to_unscaled(preds_scaled['test'], dataset, context)
            }

            all_train_preds.append(preds_unscaled['train'])
            all_val_preds.append(preds_unscaled['val'])
            all_test_preds.append(preds_unscaled['test'])

        # Simple average
        avg_preds = {
            'train': np.mean(all_train_preds, axis=0),
            'val': np.mean(all_val_preds, axis=0) if mode not in ("predict", "explain") else np.array([]),
            'test': np.mean(all_test_preds, axis=0)
        }

        # Use score_calculator component
        true_values = {'train': y_train_unscaled, 'val': y_val_unscaled, 'test': y_test_unscaled}
        avg_scores = self.score_calculator.calculate(true_values, avg_preds, dataset.task_type) if mode not in ("predict", "explain") else None

        # Weighted average
        metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
        weights = self._get_fold_weights(scores, higher_is_better, mode, runner)

        w_avg_preds = {
            'train': np.sum([w * p for w, p in zip(weights, all_train_preds)], axis=0),
            'val': np.sum([w * p for w, p in zip(weights, all_val_preds)], axis=0) if mode not in ("predict", "explain") else np.array([]),
            'test': np.sum([w * p for w, p in zip(weights, all_test_preds)], axis=0)
        }

        w_avg_scores = self.score_calculator.calculate(true_values, w_avg_preds, dataset.task_type) if mode not in ("predict", "explain") else None

        # Use prediction_assembler component to create prediction dicts
        avg_predictions = self._assemble_avg_prediction(dataset, runner, context, base_model_name, model_classname,
                                                         avg_preds, avg_scores, true_values, all_val_indices,
                                                         "avg", best_params, mode)

        w_avg_predictions = self._assemble_avg_prediction(dataset, runner, context, base_model_name, model_classname,
                                                           w_avg_preds, w_avg_scores, true_values, all_val_indices,
                                                           "w_avg", best_params, mode, weights)

        return avg_predictions, w_avg_predictions

    def _get_fold_weights(self, scores, higher_is_better, mode, runner):
        """Calculate weights for fold averaging based on validation scores.

        In prediction/explain modes, restores weights from target model if available.
        Otherwise, computes weights using EnsembleUtils._scores_to_weights().

        Args:
            scores: Array of validation scores for each fold.
            higher_is_better: Whether higher scores are better (True for R², False for RMSE).
            mode: Execution mode.
            runner: PipelineRunner with target_model info.

        Returns:
            NumPy array of normalized weights summing to 1.0.
        """
        scores = np.asarray(scores, dtype=float)

        if mode in ("predict", "explain") and "weights" in runner.target_model:
            weights_from_model = runner.target_model["weights"]
            # Check if weights exist and are not None/empty
            if weights_from_model is not None:
                if isinstance(weights_from_model, str):
                    import json
                    return np.array(json.loads(weights_from_model), dtype=float)
                elif isinstance(weights_from_model, (list, np.ndarray)):
                    weights_array = np.asarray(weights_from_model, dtype=float)
                    if len(weights_array) > 0:
                        return weights_array

        if np.all(np.isnan(scores)):
            return np.ones(len(scores), dtype=float) / len(scores)

        return EnsembleUtils._scores_to_weights(scores, higher_is_better=higher_is_better)

    def _assemble_avg_prediction(self, dataset, runner, context, model_name, model_classname,
                                  predictions, scores, true_values, val_indices, fold_id, best_params, mode, weights=None):
        """Assemble prediction dictionary for averaged model.

        Creates a complete prediction record with all metadata, scores, and partition data
        for simple or weighted averaged predictions.

        Args:
            dataset: SpectroDataset with name and preprocessing info.
            runner: PipelineRunner with pipeline metadata.
            context: Execution context with step_id.
            model_name: Base model name.
            model_classname: Model class name string.
            predictions: Dict of {partition: predictions_array}.
            scores: PartitionScores object with train/val/test scores.
            true_values: Dict of {partition: true_values_array}.
            val_indices: Validation sample indices.
            fold_id: Fold identifier ('avg' or 'w_avg').
            best_params: Optional hyperparameters dictionary.
            mode: Execution mode.
            weights: Optional array of fold weights.

        Returns:
            Dictionary ready for prediction storage with all required fields.
        """
        op_counter = runner.next_op()

        partitions = [
            ("train", list(range(len(true_values['train']))), true_values['train'], predictions['train'])
        ]
        if mode not in ("predict", "explain"):
            partitions.append(("val", val_indices.tolist(), true_values['val'], predictions['val']))
        partitions.append(("test", list(range(len(true_values['test']))), true_values['test'], predictions['test']))

        result = {
            'dataset_name': dataset.name,
            'dataset_path': dataset.name,
            'config_name': runner.saver.pipeline_id,
            'config_path': f"{dataset.name}/{runner.saver.pipeline_id}",
            'pipeline_uid': getattr(runner, 'pipeline_uid', None),
            'step_idx': context.state.step_number,  # Use step_number (int) not step_id (str)
            'op_counter': op_counter,
            'model_name': model_name,
            'model_classname': str(model_classname),
            'model_path': "",
            'fold_id': fold_id,
            'val_score': scores.val if scores else 0.0,
            'test_score': scores.test if scores else 0.0,
            'train_score': scores.train if scores else 0.0,
            'metric': scores.metric if scores else ModelUtils.get_best_score_metric(dataset.task_type)[0],
            'task_type': dataset.task_type,
            'target_processing': context.state.y_processing,  # Track which target processing was used
            'n_features': true_values['train'].shape[1] if len(true_values['train'].shape) > 1 else 1,
            'preprocessings': dataset.short_preprocessings_str(),
            'partitions': partitions,
            'best_params': best_params if best_params else {}
        }

        if weights is not None:
            result['weights'] = weights.tolist()

        return result

    def _add_all_predictions(self, prediction_store, all_predictions, weights, mode="train"):
        """Add all predictions to storage and print summaries.

        Iterates through prediction records, adds each partition to the store,
        and prints formatted summaries in train mode.

        Args:
            prediction_store: Predictions storage instance.
            all_predictions: List of prediction dictionaries.
            weights: Optional array of fold weights (applied to all predictions).
            mode: Execution mode ('train', 'finetune', 'predict', 'explain').
        """
        for idx, prediction_data in enumerate(all_predictions):
            if not prediction_data:
                continue

            partitions = prediction_data.get('partitions', [])

            # Add each partition's predictions
            pred_id = None
            for partition_name, indices, y_true_part, y_pred_part in partitions:
                if len(indices) == 0:
                    continue

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
                    weights=weights,
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

            # Print summary (only once per model)
            if pred_id and mode not in ("predict", "explain"):
                self._print_prediction_summary(prediction_data, pred_id, mode)

    def _persist_model(self, runner: 'PipelineRunner', model: Any, model_id: str) -> 'ArtifactMeta':
        """Persist trained model to disk using serializer infrastructure.

        Auto-detects model framework (sklearn, tensorflow, pytorch, xgboost, catboost, lightgbm)
        and delegates to appropriate serializer for optimal storage format.

        Args:
            runner: PipelineRunner with saver instance.
            model: Trained model to persist.
            model_id: Unique identifier for the model.

        Returns:
            ArtifactMeta with persistence metadata (path, format, size, etc.).
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


