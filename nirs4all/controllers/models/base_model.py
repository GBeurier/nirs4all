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
from joblib import Parallel, delayed
import multiprocessing

from nirs4all.controllers.controller import OperatorController
from ...optimization.optuna import OptunaManager
from nirs4all.data.predictions import Predictions
from nirs4all.data.ensemble_utils import EnsembleUtils
from nirs4all.core.task_type import TaskType
from .utilities import ModelControllerUtils as ModelUtils
from nirs4all.core import metrics as evaluator
from nirs4all.core.logging import get_logger
from nirs4all.pipeline.storage.artifacts.artifact_persistence import ArtifactMeta

logger = get_logger(__name__)
from nirs4all.pipeline.storage.artifacts.types import ArtifactType
from .components import (
    ModelIdentifierGenerator,
    PredictionTransformer,
    PredictionDataAssembler,
    ScoreCalculator,
    IndexNormalizer,
    PartitionScores
)

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.config.context import ExecutionContext


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
    def _prepare_data(self, X: Any, y: Any, context: 'ExecutionContext') -> Tuple[Any, Any]:
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

    def _predict_proba_model(self, model: Any, X: Any) -> Optional[np.ndarray]:
        """Get class probabilities for classification models.

        Returns probability distributions for each sample. Used for soft voting
        in fold averaging for classification tasks.

        Args:
            model: Trained model instance.
            X: Input features for prediction.

        Returns:
            NumPy array of shape (n_samples, n_classes) with class probabilities,
            or None if the model doesn't support probability predictions.

        Note:
            Default implementation returns None. Subclasses should override
            this method for classification support.
        """
        return None

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

    def get_xy(self, dataset: 'SpectroDataset', context: 'ExecutionContext') -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Extract train/test splits with scaled and unscaled targets.

        For classification tasks, both scaled and unscaled targets are transformed.
        For regression tasks, scaled targets are used for training while unscaled
        (numeric) targets are used for evaluation.

        In prediction mode, uses all available data (partition=None) instead of splitting.

        Also handles sample_partitioner branches, which restrict data to a subset of samples.

        Args:
            dataset: SpectroDataset with partitioned data.
            context: Execution context with partition and preprocessing info.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled).
        """
        # Use layout from context (set by controller, may include force_layout)
        layout = context.selector.layout if hasattr(context, 'selector') and hasattr(context.selector, 'layout') else self.get_preferred_layout()

        # Check if we're in prediction/explain mode
        mode = context.state.mode

        # Check for sample partition (from sample_partitioner branches)
        sample_partition = context.custom.get("sample_partition")
        partition_sample_indices = None
        if sample_partition:
            partition_sample_indices = set(sample_partition.get("sample_indices", []))

        if mode in ("predict", "explain"):
            # In prediction mode, use all available data (no partition split)
            pred_context = context.with_partition(None)

            X_all = dataset.x(pred_context.selector, layout=layout)

            # Build selector for y with processing from state
            pred_context.selector['y'] = pred_context.state.y_processing
            y_all = dataset.y(pred_context.selector)

            # If sample partition is active, filter to partition samples
            if partition_sample_indices:
                all_sample_ids = dataset._indexer.x_indices(
                    pred_context.selector, include_augmented=True, include_excluded=False
                )
                mask = np.array([int(sid) in partition_sample_indices for sid in all_sample_ids])
                X_all = X_all[mask]
                y_all = y_all[mask]

            # Return empty training data and all data as "test" for prediction
            empty_X = np.array([]).reshape(0, X_all.shape[1] if len(X_all.shape) > 1 else 0)
            empty_y = np.array([])

            # For unscaled targets
            if dataset.task_type and dataset.task_type.is_classification:
                y_all_unscaled = y_all
            else:
                # For regression, get numeric (unscaled) targets
                pred_context.selector['y'] = 'numeric'
                y_all_unscaled = dataset.y(pred_context.selector)
                if partition_sample_indices:
                    y_all_unscaled = y_all_unscaled[mask]

            return empty_X, empty_y, X_all, y_all, empty_y, y_all_unscaled

        # Normal training mode: split into train/test
        train_context = context.with_partition('train')
        test_context = context.with_partition('test')

        X_train = dataset.x(train_context.selector, layout=layout)
        X_test = dataset.x(test_context.selector, layout=layout)

        # Build selectors for y with processing from state
        train_context.selector['y'] = train_context.state.y_processing
        y_train = dataset.y(train_context.selector)

        test_context.selector['y'] = test_context.state.y_processing
        y_test = dataset.y(test_context.selector)

        # If sample partition is active, filter train and test data
        if partition_sample_indices:
            # Get sample IDs for train partition
            train_sample_ids = dataset._indexer.x_indices(
                train_context.selector, include_augmented=True, include_excluded=False
            )
            train_mask = np.array([int(sid) in partition_sample_indices for sid in train_sample_ids])
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]

            # Get sample IDs for test partition
            test_sample_ids = dataset._indexer.x_indices(
                test_context.selector, include_augmented=True, include_excluded=False
            )
            test_mask = np.array([int(sid) in partition_sample_indices for sid in test_sample_ids])
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

        # For classification tasks, use the transformed targets for evaluation
        # For regression tasks, use the original "numeric" targets
        if dataset.task_type and dataset.task_type.is_classification:
            # Use the same y context as the model training (transformed targets)
            y_train_unscaled = dataset.y(train_context.selector)
            y_test_unscaled = dataset.y(test_context.selector)
            # Apply partition mask if active
            if partition_sample_indices:
                y_train_unscaled = y_train_unscaled[train_mask]
                y_test_unscaled = y_test_unscaled[test_mask]
        else:
            # Use numeric targets for regression
            train_context.selector['y'] = 'numeric'
            test_context.selector['y'] = 'numeric'

            y_train_unscaled = dataset.y(train_context.selector)
            y_test_unscaled = dataset.y(test_context.selector)
            # Apply partition mask if active
            if partition_sample_indices:
                y_train_unscaled = y_train_unscaled[train_mask]
                y_test_unscaled = y_test_unscaled[test_mask]
        return X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled

    def _remap_folds_to_positions(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        mode: str
    ) -> List[Tuple[List[int], List[int]]]:
        """Convert fold sample IDs to positional indices for the current active samples.

        Folds are stored with absolute sample IDs (from the indexer), which remain
        valid even after sample filtering excludes samples. This method converts
        those sample IDs to positional indices into the current X_train array.

        Also handles:
        - Branch-specific outlier exclusion masks from outlier_excluder branches
        - Sample partitioning from sample_partitioner branches

        Args:
            dataset: SpectroDataset with the fold information.
            context: Execution context with partition info.
            mode: Execution mode ('train', 'predict', 'explain').

        Returns:
            List of (train_indices, val_indices) tuples with positional indices.
        """
        raw_folds = dataset.folds

        if not raw_folds:
            return []

        # In predict/explain mode, folds are not used for training, just return as-is
        if mode in ("predict", "explain"):
            return raw_folds

        # Get current active sample IDs for train partition (respecting indexer exclusions)
        train_context = context.with_partition("train")
        active_sample_ids = dataset._indexer.x_indices(  # noqa: SLF001
            train_context.selector, include_augmented=True, include_excluded=False
        )

        # Check for sample partition (from sample_partitioner branches)
        # When active, we need to filter active_sample_ids to only those in the partition
        sample_partition = context.custom.get("sample_partition")
        partition_sample_ids_set = None
        if sample_partition:
            partition_sample_ids_set = set(sample_partition.get("sample_indices", []))
            # Filter active_sample_ids to only those in the partition
            active_sample_ids = np.array([
                sid for sid in active_sample_ids
                if int(sid) in partition_sample_ids_set
            ])

        # Build a mapping from sample ID to positional index
        id_to_pos = {int(sid): pos for pos, sid in enumerate(active_sample_ids)}

        # Check for branch-specific outlier exclusion (from outlier_excluder branches)
        outlier_exclusion = context.custom.get("outlier_exclusion")
        excluded_sample_ids_set = set()
        if outlier_exclusion:
            # The outlier_exclusion stores:
            # - mask: boolean array where True = keep, False = exclude
            # - sample_indices: the sample IDs corresponding to the mask
            mask = outlier_exclusion.get("mask")
            sample_indices = outlier_exclusion.get("sample_indices", [])
            if mask is not None and len(sample_indices) > 0:
                # Get sample IDs that are excluded by this branch's outlier detection
                for i, sid in enumerate(sample_indices):
                    if i < len(mask) and not mask[i]:
                        excluded_sample_ids_set.add(int(sid))

        # Remap each fold's sample IDs to positional indices
        # Only include samples that are still active (in partition, not excluded by indexer or outlier_excluder)
        remapped_folds = []
        for train_ids, val_ids in raw_folds:
            # For train indices: filter to partition and exclude outlier_excluder samples
            train_indices = [
                id_to_pos[int(sid)] for sid in train_ids
                if int(sid) in id_to_pos and int(sid) not in excluded_sample_ids_set
            ]
            # For val indices: filter to partition (keep all samples in partition for evaluation)
            val_indices = [id_to_pos[int(sid)] for sid in val_ids if int(sid) in id_to_pos]
            remapped_folds.append((train_indices, val_indices))

        return remapped_folds

    def _get_partition_sample_indices(
        self,
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        mode: str
    ) -> Tuple[List[int], List[int]]:
        """Get sample IDs for train and test partitions.

        For OOF reconstruction and stacking, we need the actual sample IDs
        (not positional indices) to match predictions stored in the prediction store.

        Args:
            dataset: SpectroDataset with partition info.
            context: Execution context with partition info.
            mode: Execution mode ('train', 'predict', 'explain').

        Returns:
            Tuple of (train_sample_ids, test_sample_ids) as lists.
        """
        if mode in ("predict", "explain"):
            # In predict mode, there's no train/test split - all data is "test"
            pred_context = context.with_partition(None)
            all_ids = dataset._indexer.x_indices(
                pred_context.selector, include_augmented=False, include_excluded=False
            )
            return [], list(all_ids)

        # Get sample IDs for train and test partitions
        train_context = context.with_partition("train")
        test_context = context.with_partition("test")

        train_sample_ids = dataset._indexer.x_indices(
            train_context.selector, include_augmented=False, include_excluded=False
        )
        test_sample_ids = dataset._indexer.x_indices(
            test_context.selector, include_augmented=False, include_excluded=False
        )

        # Check for sample partition filtering
        sample_partition = context.custom.get("sample_partition")
        if sample_partition:
            partition_sample_ids_set = set(sample_partition.get("sample_indices", []))
            train_sample_ids = [sid for sid in train_sample_ids if int(sid) in partition_sample_ids_set]
            test_sample_ids = [sid for sid in test_sample_ids if int(sid) in partition_sample_ids_set]

        return list(train_sample_ids), list(test_sample_ids)

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: 'Predictions' = None
    ) -> Tuple['ExecutionContext', List['ArtifactMeta']]:
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
            runtime_context: Runtime context managing execution state.
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

        # In predict/explain mode, restore task_type from target_model if not set
        if mode in ("predict", "explain") and dataset.task_type is None:
            if hasattr(runtime_context, 'target_model') and runtime_context.target_model:
                task_type_str = runtime_context.target_model.get('task_type', 'regression')
                dataset.set_task_type(task_type_str)

        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled = self.get_xy(dataset, context)

        # Get actual sample IDs for train and test partitions (for stacking/OOF reconstruction)
        train_sample_ids, test_sample_ids = self._get_partition_sample_indices(dataset, context, mode)

        # Convert fold sample IDs to positional indices
        # Folds now store absolute sample IDs, which remain valid even after sample filtering
        # We need to convert them to positional indices into the current X_train array
        folds = self._remap_folds_to_positions(dataset, context, mode)

        if self.verbose > 0:
            logger.debug(f"Model config: {model_config}")

        finetune_params = model_config.get('finetune_params')

        if mode == "finetune" or (mode == "train" and finetune_params):
             return self._execute_finetune(
                dataset, model_config, context, runtime_context, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
                finetune_params, loaded_binaries,
                train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
            )
        elif mode == "train":
            return self._execute_train(
                dataset, model_config, context, runtime_context, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
                loaded_binaries,
                train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
            )
        elif mode in ("predict", "explain"):
             return self._execute_predict(
                dataset, model_config, context, runtime_context, prediction_store,
                X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
                loaded_binaries, mode,
                train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
            )
        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    def _execute_finetune(
        self, dataset, model_config, context, runtime_context, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        finetune_params, loaded_binaries,
        train_sample_ids=None, test_sample_ids=None
    ):
        self.mode = "finetune"
        if self.verbose > 0:
            logger.info("Starting finetuning...")

        best_model_params = self.finetune(
            dataset,
            model_config, X_train, y_train, X_test, y_test,
            folds, finetune_params, self.prediction_store, context, runtime_context
        )
        logger.info(f"Best parameters: {best_model_params}")

        binaries = self.train(
            dataset, model_config, context, runtime_context, prediction_store,
            X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
            loaded_binaries=loaded_binaries, mode="train", best_params=best_model_params,
            train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
        )
        return context, binaries

    def _execute_train(
        self, dataset, model_config, context, runtime_context, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        loaded_binaries,
        train_sample_ids=None, test_sample_ids=None
    ):
        if self.verbose > 0:
            logger.starting("Starting training...")

        binaries = self.train(
            dataset, model_config, context, runtime_context, prediction_store,
            X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
            loaded_binaries=loaded_binaries, mode="train",
            train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
        )
        return context, binaries

    def _execute_predict(
        self, dataset, model_config, context, runtime_context, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        loaded_binaries, mode,
        train_sample_ids=None, test_sample_ids=None
    ):
        binaries = self.train(
            dataset, model_config, context, runtime_context, prediction_store,
            X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
            loaded_binaries=loaded_binaries, mode=mode,
            train_sample_ids=train_sample_ids, test_sample_ids=test_sample_ids
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
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
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
            runtime_context: Runtime context.

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
        dataset, model_config, context, runtime_context, prediction_store,
        X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
        best_params=None, loaded_binaries=None, mode="train",
        train_sample_ids=None, test_sample_ids=None
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
            runtime_context: Runtime context.
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
            train_sample_ids: List of actual sample IDs for train partition.
            test_sample_ids: List of actual sample IDs for test partition.

        Returns:
            List of ArtifactMeta objects for persisted models.
        """

        verbose = model_config.get('train_params', {}).get('verbose', 0)
        n_jobs = model_config.get('train_params', {}).get('n_jobs', 1)

        # Auto-detect n_jobs if -1
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        binaries = []

        # In predict/explain mode, skip fold iteration entirely
        # Just load the model and predict on X_test (which contains all prediction samples)
        if mode in ("predict", "explain"):
            # For prediction, we run once per fold to match training fold structure
            # but we predict on X_test (all samples) not X_train
            n_folds = len(folds) if folds else 1
            all_fold_predictions = []

            for fold_iter in range(n_folds):
                # Use X_test as both val and test data (for prediction on all samples)
                # X_train and y_train are empty in predict mode
                # When no folds exist (folds is empty), pass fold_idx=None to load shared artifact
                # When folds exist, pass the actual fold index
                actual_fold_idx = fold_iter if folds else None
                model, model_id, score, model_name, prediction_data = self.launch_training(
                    dataset, model_config, context, runtime_context, prediction_store,
                    X_train, y_train, X_test, y_test, X_test,
                    y_train_unscaled, y_test_unscaled, y_test_unscaled,
                    train_indices=None, val_indices=None,
                    fold_idx=actual_fold_idx, best_params=best_params,
                    loaded_binaries=loaded_binaries, mode=mode,
                    test_sample_ids=test_sample_ids
                )
                all_fold_predictions.append(prediction_data)

            self._add_all_predictions(prediction_store, all_fold_predictions, None, mode=mode)
            return binaries

        if len(folds) > 0:
            folds_models = []
            fold_val_indices = []
            scores = []
            all_fold_predictions = []
            base_model_name = ""
            model_classname = ""

            # Prepare arguments for parallel execution
            fold_args = []
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
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

                fold_args.append((
                    dataset, model_config, context, runtime_context, prediction_store,
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test,
                    y_train_fold_unscaled, y_val_fold_unscaled, y_test_unscaled,
                    train_indices, val_indices,
                    fold_idx, best_params_fold,
                    loaded_binaries, mode,
                    test_sample_ids  # Added for proper test sample indexing
                ))

            if verbose > 0:
                logger.info(f"Training {len(folds)} folds (n_jobs={n_jobs})...")

            # Execute folds (parallel or sequential)
            if n_jobs > 1 and mode == "train": # Only parallelize training, not prediction/finetuning for now to avoid complexity
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.launch_training)(*args) for args in fold_args
                )
            else:
                results = [self.launch_training(*args) for args in fold_args]

            # Process results
            for i, (model, model_id, score, model_name, prediction_data) in enumerate(results):
                folds_models.append((model_id, model, score))
                all_fold_predictions.append(prediction_data)
                base_model_name = model_name
                scores.append(score)
                model_classname = model.__class__.__name__

                # Only persist in train mode, not in predict/explain modes
                if mode == "train":
                    artifact = self._persist_model(
                        runtime_context, model, model_id,
                        branch_id=context.selector.branch_id,
                        branch_name=context.selector.branch_name,
                        branch_path=context.selector.branch_path,
                        fold_id=i,
                        custom_name=model_name
                    )
                    binaries.append(artifact)
                    # Attach artifact_id to prediction_data for deterministic loading
                    artifact_id = getattr(artifact, 'artifact_id', None)
                    if artifact_id:
                        prediction_data['model_artifact_id'] = artifact_id

            # Compute weights based on scores
            metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
            weights = EnsembleUtils._scores_to_weights(np.array(scores), higher_is_better=higher_is_better)

            # Create fold averages and get average predictions data
            if len(folds) > 1:
                avg_predictions, w_avg_predictions = self._create_fold_averages(
                    base_model_name, dataset, model_config, context, runtime_context, prediction_store, model_classname,
                    folds_models, fold_val_indices, scores,
                    X_train, X_test, y_train_unscaled, y_test_unscaled, mode=mode, best_params=best_params,
                    test_sample_ids=test_sample_ids
                )
                # Collect ALL predictions (folds + averages) and add them in one shot with same weights
                all_fold_predictions = all_fold_predictions + [avg_predictions, w_avg_predictions]

            self._add_all_predictions(prediction_store, all_fold_predictions, weights, mode=mode)

        else:
            logger.warning("WARNING: Using test set as validation set (no folds provided)")

            model, model_id, score, model_name, prediction_data = self.launch_training(
                dataset, model_config, context, runtime_context, prediction_store,
                X_train, y_train, X_test, y_test, X_test,
                y_train_unscaled, y_test_unscaled, y_test_unscaled,
                loaded_binaries=loaded_binaries, mode=mode,
                test_sample_ids=test_sample_ids
            )
            artifact = self._persist_model(
                runtime_context, model, model_id,
                branch_id=context.selector.branch_id,
                branch_name=context.selector.branch_name,
                branch_path=context.selector.branch_path,
                fold_id=None,  # Single model, no folds
                custom_name=model_name
            )
            binaries.append(artifact)
            # Attach artifact_id to prediction_data for deterministic loading
            artifact_id = getattr(artifact, 'artifact_id', None)
            if artifact_id:
                prediction_data['model_artifact_id'] = artifact_id

            # Add predictions for single model case (no weights)
            self._add_all_predictions(prediction_store, [prediction_data], None, mode=mode)

        return binaries

    def process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process hyperparameters before use.

        Can be overridden by subclasses to structure parameters (e.g. nesting for TensorFlow).

        Args:
            params: Flat dictionary of sampled parameters.

        Returns:
            Processed dictionary of parameters.
        """
        return params

    def launch_training(
        self,
        dataset, model_config, context, runtime_context, prediction_store,
        X_train, y_train, X_val, y_val, X_test,
        y_train_unscaled, y_val_unscaled, y_test_unscaled,
        train_indices=None, val_indices=None, fold_idx=None, best_params=None,
        loaded_binaries=None, mode="train", test_sample_ids=None):
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
            runtime_context: Runtime context.
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
            test_sample_ids: List of actual sample IDs for test partition (for stacking).

        Returns:
            Tuple of (trained_model, model_id, val_score, model_name, prediction_data)
        """
        # === 1. GENERATE IDENTIFIERS ===
        identifiers = self.identifier_generator.generate(model_config, runtime_context, context, fold_idx)

        # Debug: check identifiers
        if identifiers.step_id == '' or identifiers.step_id == 0:
            logger.warning(f"WARNING in launch_training: step_id={identifiers.step_id}")
            logger.warning(f"context.state.step_number={context.state.step_number}")

        # === 2. GET OR LOAD MODEL ===
        if mode in ("predict", "explain"):
            model = None

            # V3: Use artifact_provider for chain-based loading
            if runtime_context.artifact_provider is not None:
                step_index = runtime_context.step_number
                import logging
                logging.debug(f"MODEL: loading from artifact_provider step={step_index}, fold={fold_idx}")

                # Get artifact with branch awareness
                branch_path = context.selector.branch_path if hasattr(context.selector, 'branch_path') else None

                if fold_idx is not None:
                    # Get fold-specific artifact using get_fold_artifacts (supports branch_path)
                    fold_artifacts = runtime_context.artifact_provider.get_fold_artifacts(
                        step_index, branch_path=branch_path
                    )
                    # Find artifact for this specific fold
                    for fid, artifact_obj in fold_artifacts:
                        if fid == fold_idx:
                            model = artifact_obj
                            break
                else:
                    # Get primary artifact (for non-CV or single model case)
                    # Use get_artifacts_for_step which supports branch_path
                    step_artifacts = runtime_context.artifact_provider.get_artifacts_for_step(
                        step_index, branch_path=branch_path
                    )
                    if step_artifacts:
                        model = step_artifacts[0][1]

                if model is not None:
                    logging.debug(f"MODEL: loaded model type={type(model).__name__}")
                    if self.verbose > 0:
                        logger.debug(f"Loaded model via artifact_provider for step {step_index}, fold={fold_idx}")
            if model is None:
                raise ValueError(
                    f"Model not found at step {runtime_context.step_number}, fold={fold_idx}. "
                    f"Ensure artifact_provider is properly configured."
                )

            # Capture model for SHAP explanation
            if mode == "explain" and self._should_capture_for_explanation(runtime_context, identifiers):
                if hasattr(runtime_context, 'explainer') and hasattr(runtime_context.explainer, 'capture_model'):
                    runtime_context.explainer.capture_model(model, self)

            trained_model = model
        else:
            # Create new model for training
            if mode == "finetune" and best_params is not None:
                if self.verbose > 0:
                    print(f"Training model {identifiers.name} with: {best_params}...")
                model = self._get_model_instance(dataset, model_config, force_params=best_params)
            else:
                # Support model_params for customizing NN architecture at training time
                model_params = model_config.get('model_params', {})
                if model_params:
                    base_model = self._get_model_instance(dataset, model_config, force_params=model_params)
                else:
                    base_model = self._get_model_instance(dataset, model_config)
                model = self._clone_model(base_model)

            # === 3. TRAIN MODEL ===
            X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, context or {})
            X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, context or {})
            X_test_prep, _ = self._prepare_data(X_test, None, context or {})

            # Log data shapes before training
            if self.verbose > 0:
                logger.debug(f"Training data shapes - X_train: {X_train_prep.shape}, y_train: {y_train_prep.shape if y_train_prep is not None else 'None'}, "
                      f"X_val: {X_val_prep.shape}, y_val: {y_val_prep.shape if y_val_prep is not None else 'None'}, "
                      f"X_test: {X_test_prep.shape}")

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

        # === 4b. GENERATE PROBABILITIES FOR CLASSIFICATION ===
        is_classification = dataset.task_type and dataset.task_type.is_classification
        probabilities = {'train': None, 'val': None, 'test': None}
        if is_classification:
            probabilities = {
                'train': self._predict_proba_model(trained_model, X_train_prep) if X_train_prep.shape[0] > 0 else None,
                'val': self._predict_proba_model(trained_model, X_val_prep) if X_val_prep.shape[0] > 0 else None,
                'test': self._predict_proba_model(trained_model, X_test_prep) if X_test_prep.shape[0] > 0 else None
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

        # Calculate full metrics for all partitions
        full_scores = {}
        for partition in ['train', 'val', 'test']:
            if len(true_values[partition]) > 0 and len(predictions_unscaled[partition]) > 0:
                full_scores[partition] = self._calculate_and_print_scores(
                    true_values[partition],
                    predictions_unscaled[partition],
                    dataset.task_type,
                    partition=partition,
                    model_name=identifiers.name,
                    show_detailed_scores=False  # (partition == 'test') # Only show detailed scores for test
                )
            else:
                full_scores[partition] = {}

        # === 7. NORMALIZE INDICES ===
        # In predict mode with no y, use X shape to determine sample counts
        n_samples = {
            'train': len(y_train_unscaled) if y_train_unscaled is not None and len(y_train_unscaled) > 0 else (len(y_train_prep) if y_train_prep is not None and len(y_train_prep) > 0 else X_train_prep.shape[0]),
            'val': len(y_val_unscaled) if y_val_unscaled is not None and len(y_val_unscaled) > 0 else (len(y_val_prep) if y_val_prep is not None and len(y_val_prep) > 0 else X_val_prep.shape[0]),
            'test': len(y_test_unscaled) if y_test_unscaled is not None and len(y_test_unscaled) > 0 else X_test_prep.shape[0]
        }

        # For test indices: use actual sample IDs if provided, otherwise use range
        test_indices_normalized = test_sample_ids if test_sample_ids is not None else self.index_normalizer.normalize(None, n_samples['test'])

        indices = {
            'train': self.index_normalizer.normalize(train_indices, n_samples['train']),
            'val': self.index_normalizer.normalize(val_indices, n_samples['val']),
            'test': test_indices_normalized
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
            runner=runtime_context,
            X_shape=X_train.shape,
            best_params=best_params,
            context=context
        )

        # Add full scores to prediction data
        prediction_data['scores'] = full_scores

        # Add probabilities for classification tasks
        prediction_data['probabilities'] = probabilities

        return trained_model, identifiers.model_id, partition_scores.val, identifiers.name, prediction_data

    def _should_capture_for_explanation(self, runtime_context, identifiers) -> bool:
        """Check if current model should be captured for SHAP explanation.

        Compares model name and step index with runner's target_model to determine
        if this is the model requiring explanation.

        Args:
            runtime_context: Runtime context with target_model info.
            identifiers: ModelIdentifiers with name and step_id.

        Returns:
            True if model should be captured for explanation, False otherwise.
        """
        target = runtime_context.target_model
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
        direction = "↑" if metric in ['r2', 'accuracy', 'balanced_accuracy'] else "↓"

        summary = f"{model_name} {metric} {direction} [test: {test_score:.4f}], [val: {val_score:.4f}]"
        if fold_id not in [None, 'None', 'avg', 'w_avg']:
            summary += f", (fold: {fold_id}, id: {op_counter})"
        elif fold_id in ['avg', 'w_avg']:
            summary += f", ({fold_id}, id: {op_counter})"
        summary += f" - [{pred_id}]"
        logger.success(summary)

    def get_preferred_layout(self) -> str:
        """Get preferred data layout for the framework.

        Returns:
            Data layout string ('2d' for NumPy arrays, '3d' for TensorFlow, etc.).

        Note:
            Override in subclasses for framework-specific layouts.
        """
        return "2d"

    def get_effective_layout(self, step_info: Optional['ParsedStep'] = None) -> str:
        """Get effective data layout, respecting force_layout if specified.

        This method checks if the step configuration has a force_layout override.
        If not, it falls back to the controller's preferred layout.

        Args:
            step_info: ParsedStep containing potential force_layout override.

        Returns:
            Data layout string to use for this step.
        """
        if step_info is not None and hasattr(step_info, 'force_layout') and step_info.force_layout is not None:
            return step_info.force_layout
        return self.get_preferred_layout()

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
        # Get default metrics for the task type
        default_metrics = evaluator.get_default_metrics(task_type.value)

        # Calculate all default metrics
        scores_list = evaluator.eval_list(y_true, y_pred, default_metrics)
        scores = dict(zip(default_metrics, scores_list))

        if scores and show_detailed_scores:
            score_str = ModelUtils.format_scores(scores)
            logger.info(f"{model_name} {partition} scores: {score_str}")
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
        base_model_name, dataset, model_config, context, runtime_context, prediction_store, model_classname,
        folds_models, fold_val_indices, scores,
        X_train, X_test, y_train_unscaled, y_test_unscaled,
        mode="train", best_params=None, test_sample_ids=None
    ) -> Tuple[Dict, Dict]:
        """Create simple and weighted fold-averaged predictions.

        Generates two averaged predictions:
            1. Simple average: Equal weight to all folds
            2. Weighted average: Weights based on validation scores

        For regression: Uses arithmetic mean of predictions.
        For classification: Uses soft voting (average probabilities, then argmax).
                           Falls back to hard voting if probabilities unavailable.

        Args:
            base_model_name: Base name for averaged models.
            dataset: SpectroDataset with task type and preprocessing info.
            model_config: Model configuration dictionary.
            context: Execution context.
            runtime_context: Runtime context.
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
            test_sample_ids: List of actual sample IDs for test partition.

        Returns:
            Tuple of (avg_prediction_dict, weighted_avg_prediction_dict).
        """
        is_classification = dataset.task_type and dataset.task_type.is_classification

        # Prepare validation data
        X_val = np.vstack([X_train[val_idx] for val_idx in fold_val_indices])
        y_val_unscaled = np.vstack([y_train_unscaled[val_idx] for val_idx in fold_val_indices])
        all_val_indices = np.hstack(fold_val_indices)

        # Calculate weights based on scores
        metric, higher_is_better = ModelUtils.get_best_score_metric(dataset.task_type)
        weights = self._get_fold_weights(scores, higher_is_better, mode, runtime_context)

        # Initialize probabilities (will be set for classification with soft voting)
        avg_probs = {'train': None, 'val': None, 'test': None}
        w_avg_probs = {'train': None, 'val': None, 'test': None}

        if is_classification:
            # Use classification-specific averaging (soft or hard voting)
            avg_preds, w_avg_preds, avg_probs, w_avg_probs = self._create_classification_fold_averages(
                folds_models, X_train, X_val, X_test, weights, mode
            )
        else:
            # Use regression averaging (arithmetic mean)
            avg_preds, w_avg_preds = self._create_regression_fold_averages(
                folds_models, X_train, X_val, X_test, weights, dataset, context, mode
            )

        # Calculate scores for averaged predictions
        true_values = {'train': y_train_unscaled, 'val': y_val_unscaled, 'test': y_test_unscaled}

        avg_scores = self.score_calculator.calculate(
            true_values, avg_preds, dataset.task_type
        ) if mode not in ("predict", "explain") else None

        w_avg_scores = self.score_calculator.calculate(
            true_values, w_avg_preds, dataset.task_type
        ) if mode not in ("predict", "explain") else None

        # IMPORTANT: Override val_score for avg/w_avg to avoid data leakage.
        # The validation predictions above are computed by having each fold model
        # predict on ALL validation samples, but each fold's model has seen its own
        # validation samples during training. This causes biased (overly optimistic) val scores.
        # Instead, use the average of individual fold validation scores (which are unbiased OOF scores).
        if mode not in ("predict", "explain") and len(scores) > 0:
            # Simple average of fold val_scores for 'avg' fold
            avg_val_score = float(np.mean(scores))
            if avg_scores is not None:
                avg_scores = PartitionScores(
                    train=avg_scores.train,
                    val=avg_val_score,  # Override with unbiased average
                    test=avg_scores.test,
                    metric=avg_scores.metric,
                    higher_is_better=avg_scores.higher_is_better
                )

            # Weighted average of fold val_scores for 'w_avg' fold
            w_avg_val_score = float(np.sum([w * s for w, s in zip(weights, scores)]))
            if w_avg_scores is not None:
                w_avg_scores = PartitionScores(
                    train=w_avg_scores.train,
                    val=w_avg_val_score,  # Override with unbiased weighted average
                    test=w_avg_scores.test,
                    metric=w_avg_scores.metric,
                    higher_is_better=w_avg_scores.higher_is_better
                )

        # Calculate full metrics for average predictions
        avg_full_scores = {}
        w_avg_full_scores = {}
        if mode not in ("predict", "explain"):
            for partition in ['train', 'val', 'test']:
                if len(true_values[partition]) > 0 and len(avg_preds[partition]) > 0:
                    avg_full_scores[partition] = self._calculate_and_print_scores(
                        true_values[partition],
                        avg_preds[partition],
                        dataset.task_type,
                        partition=partition,
                        model_name=f"{base_model_name}_avg",
                        show_detailed_scores=False
                    )
                else:
                    avg_full_scores[partition] = {}

                if len(true_values[partition]) > 0 and len(w_avg_preds[partition]) > 0:
                    w_avg_full_scores[partition] = self._calculate_and_print_scores(
                        true_values[partition],
                        w_avg_preds[partition],
                        dataset.task_type,
                        partition=partition,
                        model_name=f"{base_model_name}_w_avg",
                        show_detailed_scores=False
                    )
                else:
                    w_avg_full_scores[partition] = {}

        # Use prediction_assembler component to create prediction dicts
        avg_predictions = self._assemble_avg_prediction(
            dataset, runtime_context, context, base_model_name, model_classname,
            avg_preds, avg_scores, true_values, all_val_indices,
            "avg", best_params, mode, X_train.shape,
            test_sample_ids=test_sample_ids
        )
        avg_predictions['scores'] = avg_full_scores
        avg_predictions['probabilities'] = avg_probs

        w_avg_predictions = self._assemble_avg_prediction(
            dataset, runtime_context, context, base_model_name, model_classname,
            w_avg_preds, w_avg_scores, true_values, all_val_indices,
            "w_avg", best_params, mode, X_train.shape, weights,
            test_sample_ids=test_sample_ids
        )
        w_avg_predictions['scores'] = w_avg_full_scores
        w_avg_predictions['probabilities'] = w_avg_probs

        return avg_predictions, w_avg_predictions

    def _create_regression_fold_averages(
        self,
        folds_models, X_train, X_val, X_test, weights, dataset, context, mode
    ) -> Tuple[Dict, Dict]:
        """Create fold-averaged predictions for regression using arithmetic mean.

        Args:
            folds_models: List of (model_id, model, score) tuples.
            X_train, X_val, X_test: Feature arrays for each partition.
            weights: Fold weights for weighted averaging.
            dataset: SpectroDataset for prediction transformation.
            context: Execution context.
            mode: Execution mode.

        Returns:
            Tuple of (simple_avg_preds, weighted_avg_preds) dictionaries.
        """
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

        # Weighted average
        w_avg_preds = {
            'train': np.sum([w * p for w, p in zip(weights, all_train_preds)], axis=0),
            'val': np.sum([w * p for w, p in zip(weights, all_val_preds)], axis=0) if mode not in ("predict", "explain") else np.array([]),
            'test': np.sum([w * p for w, p in zip(weights, all_test_preds)], axis=0)
        }

        return avg_preds, w_avg_preds

    def _create_classification_fold_averages(
        self,
        folds_models, X_train, X_val, X_test, weights, mode
    ) -> Tuple[Dict, Dict]:
        """Create fold-averaged predictions for classification using soft voting.

        Uses probability averaging (soft voting) when probabilities are available,
        otherwise falls back to hard voting (majority vote).

        Args:
            folds_models: List of (model_id, model, score) tuples.
            X_train, X_val, X_test: Feature arrays for each partition.
            weights: Fold weights for weighted voting.
            mode: Execution mode.

        Returns:
            Tuple of (simple_avg_preds, weighted_avg_preds) dictionaries.
        """
        # Collect probabilities or class predictions from all folds
        all_train_probs = []
        all_val_probs = []
        all_test_probs = []
        all_train_preds = []
        all_val_preds = []
        all_test_preds = []
        use_soft_voting = True

        for _, fold_model, _ in folds_models:
            # Try to get probabilities first
            train_probs = self._predict_proba_model(fold_model, X_train) if X_train.shape[0] > 0 else None
            val_probs = self._predict_proba_model(fold_model, X_val) if X_val.shape[0] > 0 else None
            test_probs = self._predict_proba_model(fold_model, X_test) if X_test.shape[0] > 0 else None

            if train_probs is None or test_probs is None:
                # Model doesn't support probabilities, use hard voting
                use_soft_voting = False
                all_train_preds.append(
                    self._predict_model(fold_model, X_train) if X_train.shape[0] > 0 else np.array([])
                )
                all_val_preds.append(
                    self._predict_model(fold_model, X_val) if X_val.shape[0] > 0 else np.array([])
                )
                all_test_preds.append(
                    self._predict_model(fold_model, X_test) if X_test.shape[0] > 0 else np.array([])
                )
            else:
                all_train_probs.append(train_probs)
                all_val_probs.append(val_probs if val_probs is not None else np.array([]))
                all_test_probs.append(test_probs)
                # Also collect class predictions for potential fallback
                all_train_preds.append(
                    self._predict_model(fold_model, X_train) if X_train.shape[0] > 0 else np.array([])
                )
                all_val_preds.append(
                    self._predict_model(fold_model, X_val) if X_val.shape[0] > 0 else np.array([])
                )
                all_test_preds.append(
                    self._predict_model(fold_model, X_test) if X_test.shape[0] > 0 else np.array([])
                )

        if use_soft_voting and all_train_probs:
            # Soft voting: average probabilities, then argmax
            # avg: simple average of probabilities
            # w_avg: uses fold weights AND confidence weighting for meaningful differences
            avg_preds, avg_probs = self._soft_vote_partitions(
                all_train_probs, all_val_probs, all_test_probs,
                weights=None, mode=mode, use_confidence_weighting=False
            )
            w_avg_preds, w_avg_probs = self._soft_vote_partitions(
                all_train_probs, all_val_probs, all_test_probs,
                weights=weights, mode=mode, use_confidence_weighting=True
            )
        else:
            # Hard voting: majority vote on class predictions
            avg_preds = self._hard_vote_partitions(
                all_train_preds, all_val_preds, all_test_preds, weights=None, mode=mode
            )
            w_avg_preds = self._hard_vote_partitions(
                all_train_preds, all_val_preds, all_test_preds, weights=weights, mode=mode
            )
            # No probabilities for hard voting
            avg_probs = {'train': None, 'val': None, 'test': None}
            w_avg_probs = {'train': None, 'val': None, 'test': None}

        return avg_preds, w_avg_preds, avg_probs, w_avg_probs

    def _soft_vote_partitions(
        self,
        all_train_probs, all_val_probs, all_test_probs, weights, mode,
        use_confidence_weighting: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply soft voting to each partition.

        Args:
            all_*_probs: Lists of probability arrays from each fold.
            weights: Optional fold weights.
            mode: Execution mode.
            use_confidence_weighting: If True, weight by prediction confidence.

        Returns:
            Tuple of:
                - predictions: Dictionary with class predictions for each partition.
                - probabilities: Dictionary with averaged probabilities for each partition.
        """
        predictions = {}
        probabilities = {}

        # Train partition
        if all_train_probs and all_train_probs[0].size > 0:
            predictions['train'], probabilities['train'] = EnsembleUtils.compute_soft_voting_average(
                all_train_probs, weights, use_confidence_weighting
            )
        else:
            predictions['train'] = np.array([])
            probabilities['train'] = None

        # Validation partition
        if mode not in ("predict", "explain") and all_val_probs and all_val_probs[0].size > 0:
            predictions['val'], probabilities['val'] = EnsembleUtils.compute_soft_voting_average(
                all_val_probs, weights, use_confidence_weighting
            )
        else:
            predictions['val'] = np.array([])
            probabilities['val'] = None

        # Test partition
        if all_test_probs and all_test_probs[0].size > 0:
            predictions['test'], probabilities['test'] = EnsembleUtils.compute_soft_voting_average(
                all_test_probs, weights, use_confidence_weighting
            )
        else:
            predictions['test'] = np.array([])
            probabilities['test'] = None

        return predictions, probabilities

    def _hard_vote_partitions(
        self,
        all_train_preds, all_val_preds, all_test_preds, weights, mode
    ) -> Dict[str, np.ndarray]:
        """Apply hard voting (majority vote) to each partition.

        Args:
            all_*_preds: Lists of class prediction arrays from each fold.
            weights: Optional fold weights.
            mode: Execution mode.

        Returns:
            Dictionary with voted predictions for each partition.
        """
        result = {}

        # Train partition
        if all_train_preds and all_train_preds[0].size > 0:
            result['train'] = EnsembleUtils.compute_hard_voting(all_train_preds, weights)
        else:
            result['train'] = np.array([])

        # Validation partition
        if mode not in ("predict", "explain") and all_val_preds and all_val_preds[0].size > 0:
            result['val'] = EnsembleUtils.compute_hard_voting(all_val_preds, weights)
        else:
            result['val'] = np.array([])

        # Test partition
        if all_test_preds and all_test_preds[0].size > 0:
            result['test'] = EnsembleUtils.compute_hard_voting(all_test_preds, weights)
        else:
            result['test'] = np.array([])

        return result

    def _get_fold_weights(self, scores, higher_is_better, mode, runtime_context):
        """Calculate weights for fold averaging based on validation scores.

        In prediction/explain modes, restores weights from target model if available.
        Otherwise, computes weights using EnsembleUtils._scores_to_weights().

        Args:
            scores: Array of validation scores for each fold.
            higher_is_better: Whether higher scores are better (True for R², False for RMSE).
            mode: Execution mode.
            runtime_context: Runtime context with target_model info.

        Returns:
            NumPy array of normalized weights summing to 1.0.
        """
        scores = np.asarray(scores, dtype=float)

        if mode in ("predict", "explain") and runtime_context.target_model and "weights" in runtime_context.target_model:
            weights_from_model = runtime_context.target_model["weights"]
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
                                  predictions, scores, true_values, val_indices, fold_id, best_params, mode, X_shape, weights=None,
                                  test_sample_ids=None):
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
            X_shape: Shape of input features (for n_features).
            weights: Optional array of fold weights.
            test_sample_ids: List of actual sample IDs for test partition.

        Returns:
            Dictionary ready for prediction storage with all required fields.
        """
        op_counter = runner.next_op()

        # Build partitions with actual sample IDs (not range indices)
        # For test: use test_sample_ids if provided, otherwise fallback to range
        test_indices = test_sample_ids if test_sample_ids is not None else list(range(len(true_values['test'])))

        partitions = [
            ("train", list(range(len(true_values['train']))), true_values['train'], predictions['train'])
        ]
        if mode not in ("predict", "explain"):
            partitions.append(("val", val_indices.tolist(), true_values['val'], predictions['val']))
        partitions.append(("test", test_indices, true_values['test'], predictions['test']))

        # Extract metadata for each partition
        # Note: For 'val' partition in CV, samples come from train partition
        # so we use train metadata with val_indices to select the right rows
        partition_metadata = {}

        # Cache train metadata since val samples come from train
        train_meta_df = None
        try:
            train_meta_df = dataset.metadata({"partition": "train"})
        except (KeyError, AttributeError, ValueError, TypeError):
            pass

        for partition_name in ['train', 'val', 'test']:
            if partition_name in predictions:
                try:
                    pred_len = len(predictions[partition_name])

                    if partition_name == 'val':
                        # Validation samples are from train partition - use val_indices
                        meta_df = train_meta_df
                        indices_to_use = val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
                    elif partition_name == 'train':
                        meta_df = train_meta_df
                        indices_to_use = None  # Use all rows
                    else:  # test
                        meta_df = dataset.metadata({"partition": partition_name})
                        indices_to_use = None  # Use all rows

                    if meta_df is not None and len(meta_df) > 0:
                        metadata_dict = {}
                        for col in meta_df.columns:
                            col_data = meta_df[col].to_numpy()

                            if indices_to_use is not None:
                                # Select specific rows for validation
                                if len(indices_to_use) > 0 and max(indices_to_use) < len(col_data):
                                    selected_data = col_data[indices_to_use]
                                    if len(selected_data) == pred_len:
                                        metadata_dict[col] = selected_data.tolist()
                            else:
                                # Direct match for train/test
                                if len(col_data) == pred_len:
                                    metadata_dict[col] = col_data.tolist()

                        if metadata_dict:
                            partition_metadata[partition_name] = metadata_dict
                except (KeyError, AttributeError, ValueError, TypeError, IndexError):
                    pass

        # Get trace_id from runtime context (Phase 2)
        trace_id = runner.get_trace_id() if hasattr(runner, 'get_trace_id') else None

        result = {
            'dataset_name': dataset.name,
            'dataset_path': dataset.name,
            'config_name': runner.saver.pipeline_id,
            'config_path': f"{dataset.name}/{runner.saver.pipeline_id}",
            'pipeline_uid': getattr(runner, 'pipeline_uid', None),
            'trace_id': trace_id,  # Phase 2: Link to execution trace
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
            'n_features': X_shape[1] if len(X_shape) > 1 else 1,
            'preprocessings': dataset.short_preprocessings_str(),
            'partitions': partitions,
            'partition_metadata': partition_metadata,
            'best_params': best_params if best_params else {},
            'branch_id': getattr(context.selector, 'branch_id', None),
            'branch_name': getattr(context.selector, 'branch_name', None) or "",
        }

        # Add outlier exclusion info if available (from outlier_excluder branches)
        exclusion_info = context.custom.get('exclusion_info', {})
        if exclusion_info:
            result['exclusion_count'] = exclusion_info.get('n_excluded', 0)
            result['exclusion_rate'] = exclusion_info.get('exclusion_rate', 0.0)

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
            probabilities = prediction_data.get('probabilities', {})
            partition_metadata = prediction_data.get('partition_metadata', {})

            # Add each partition's predictions
            pred_id = None
            for partition_name, indices, y_true_part, y_pred_part in partitions:
                if len(indices) == 0:
                    continue

                # Get probabilities for this partition if available
                y_proba_part = probabilities.get(partition_name) if probabilities else None

                # Get metadata for this partition (for aggregation support)
                metadata = partition_metadata.get(partition_name, {})

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
                    metadata=metadata,
                    partition=partition_name,
                    y_true=y_true_part,
                    y_pred=y_pred_part,
                    y_proba=y_proba_part,
                    val_score=prediction_data['val_score'],
                    test_score=prediction_data['test_score'],
                    train_score=prediction_data['train_score'],
                    metric=prediction_data['metric'],
                    task_type=prediction_data['task_type'],
                    n_samples=len(y_true_part),
                    n_features=prediction_data['n_features'],
                    preprocessings=prediction_data['preprocessings'],
                    best_params=prediction_data['best_params'],
                    scores=prediction_data.get('scores', {}),
                    branch_id=prediction_data.get('branch_id'),
                    branch_name=prediction_data.get('branch_name'),
                    exclusion_count=prediction_data.get('exclusion_count'),
                    exclusion_rate=prediction_data.get('exclusion_rate'),
                    model_artifact_id=prediction_data.get('model_artifact_id'),
                    trace_id=prediction_data.get('trace_id')
                )

            # Print summary (only once per model)
            if pred_id and mode not in ("predict", "explain"):
                self._print_prediction_summary(prediction_data, pred_id, mode)

    def _persist_model(
        self,
        runtime_context: 'RuntimeContext',
        model: Any,
        model_id: str,
        branch_id: Optional[int] = None,
        branch_name: Optional[str] = None,
        branch_path: Optional[List[int]] = None,
        fold_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        custom_name: Optional[str] = None
    ) -> 'ArtifactMeta':
        """Persist trained model to disk using registry or legacy saver.

        Uses artifact_registry.register_with_chain() for V3 chain-based identification,
        enabling complete execution path tracking for deterministic reload.

        Also records the artifact in the execution trace for deterministic replay.

        Args:
            runtime_context: Runtime context with saver/registry instances.
            model: Trained model to persist.
            model_id: Unique identifier for the model.
            branch_id: Optional branch identifier for branched pipelines.
            branch_name: Optional branch name for branched pipelines.
            branch_path: Optional list of branch indices for nested branching.
            fold_id: Optional fold identifier for CV artifacts.
            params: Optional model parameters for inspection.
            custom_name: User-defined name for the model (e.g., "Q5_PLS_10").

        Returns:
            ArtifactMeta or ArtifactRecord with persistence metadata.
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

        # Use artifact registry if available (V3 system)
        if runtime_context.artifact_registry is not None:
            registry = runtime_context.artifact_registry
            pipeline_id = runtime_context.saver.pipeline_id if runtime_context.saver else "unknown"
            step_index = runtime_context.step_number

            # Use branch_path or convert branch_id to branch_path
            bp = branch_path or ([branch_id] if branch_id is not None else [])

            # Use substep_number as substep_index when in a subpipeline
            substep_index = runtime_context.substep_number if runtime_context.substep_number >= 0 else None

            # V3: Build operator chain for this artifact
            from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorNode, OperatorChain

            # Get the current chain from trace recorder or build new one
            if runtime_context.trace_recorder is not None:
                current_chain = runtime_context.trace_recorder.current_chain()
            else:
                current_chain = OperatorChain(pipeline_id=pipeline_id)

            # Create node for this model
            model_node = OperatorNode(
                step_index=step_index,
                operator_class=model.__class__.__name__,
                branch_path=bp,
                fold_id=fold_id,
                substep_index=substep_index,
                operator_name=custom_name,
            )

            # Build chain path for this artifact
            artifact_chain = current_chain.append(model_node)
            chain_path = artifact_chain.to_path()

            # Generate V3 artifact ID using chain
            artifact_id = registry.generate_id(chain_path, fold_id, pipeline_id)

            # Register artifact with V3 chain tracking
            record = registry.register(
                obj=model,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.MODEL,
                params=params or {},
                format_hint=format_hint,
                custom_name=custom_name,
                chain_path=chain_path,
                source_index=None,  # Models don't have source_index
            )

            # Record artifact in execution trace with chain_path
            runtime_context.record_step_artifact(
                artifact_id=artifact_id,
                is_primary=(fold_id is None),  # Primary if not fold-specific
                fold_id=fold_id,
                chain_path=chain_path,
                branch_path=bp,
                metadata={"class_name": model.__class__.__name__, "custom_name": custom_name}
            )

            return record

        # No registry available - skip persistence (for unit tests)
        # In production, artifact_registry should always be set by the runner
        return None


