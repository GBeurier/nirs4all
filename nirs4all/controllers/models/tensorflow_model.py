"""
TensorFlow Model Controller - Controller for TensorFlow/Keras models

This controller handles TensorFlow/Keras models with support for:
- Training on 2D/3D data with proper tensor formatting
- Model compilation with loss functions and metrics
- Early stopping and callbacks support
- Integration with Optuna for hyperparameter tuning
- Model persistence and prediction storage

Matches TensorFlow/Keras model objects and model configurations.

Lazy loading pattern: TensorFlow is only imported when actually needed
for training or prediction, not at module import time.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    try:
        from tensorflow import keras
    except ImportError:
        pass

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.core.task_type import TaskType
from nirs4all.utils.backend import is_available, require_backend, is_gpu_available

logger = get_logger(__name__)

# Fast availability check at module level - no imports
TENSORFLOW_AVAILABLE = is_available('tensorflow')

# Lazy-loaded module cache
_tf_modules: Dict[str, Any] = {}


def _get_tf():
    """Lazy load TensorFlow with caching."""
    if 'tf' not in _tf_modules:
        require_backend('tensorflow', feature='TensorFlow neural networks')
        import tensorflow as tf
        _tf_modules['tf'] = tf
        _tf_modules['keras'] = tf.keras
    return _tf_modules['tf']


def _get_keras():
    """Lazy load Keras with caching."""
    if 'keras' not in _tf_modules:
        _get_tf()  # This will populate the cache
    return _tf_modules['keras']

@register_controller
class TensorFlowModelController(BaseModelController):
    """Controller for TensorFlow/Keras models.

    This controller manages the complete lifecycle of TensorFlow/Keras models including:
    - Model instantiation from various configuration formats
    - Data preparation with proper tensor formatting (2D/3D)
    - Model compilation with task-appropriate loss functions and metrics
    - Training with callbacks (early stopping, model checkpointing)
    - Hyperparameter tuning via Optuna integration
    - Model evaluation and prediction
    - Binary serialization for model persistence

    The controller automatically detects TensorFlow models and functions decorated
    with @framework('tensorflow'). It uses lazy loading to avoid importing
    TensorFlow until actually needed.

    Attributes:
        priority (int): Controller priority for matching (4).
    """

    priority = 4  # Higher priority than Sklearn (6)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Determine if this controller should handle the given step.

        Matches TensorFlow/Keras models, functions decorated with @framework('tensorflow'),
        and serialized model configurations containing TensorFlow components.

        Args:
            step: Pipeline step configuration (dict, model instance, or function).
            operator: Optional operator instance extracted from step.
            keyword: Optional keyword identifier for the step.

        Returns:
            True if this controller should handle the step, False otherwise.
            Returns False immediately if TensorFlow is not installed.
        """
        if not TENSORFLOW_AVAILABLE:
            return False

        # Check if step contains a TensorFlow model or function
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            if cls._is_tensorflow_model_or_function(model):
                return True
            # Handle dictionary config for model
            if isinstance(model, dict) and 'class' in model:
                class_name = model['class']
                if isinstance(class_name, str) and ('tensorflow' in class_name or 'keras' in class_name):
                    return True

        # Check direct TensorFlow objects or functions
        if cls._is_tensorflow_model_or_function(step):
            return True

        # Check operator if provided
        if operator is not None and cls._is_tensorflow_model_or_function(operator):
            return True

        return False

    @classmethod
    def _is_tensorflow_model(cls, obj: Any) -> bool:
        """Check if object is a TensorFlow/Keras model instance.

        Uses module introspection first to avoid importing TensorFlow
        for non-TensorFlow objects.

        Args:
            obj: Object to check.

        Returns:
            True if object is a keras.Model, keras.Sequential, or has fit/predict/compile
            methods characteristic of Keras models. False otherwise or if TensorFlow
            is not available.
        """
        if not TENSORFLOW_AVAILABLE:
            return False

        if obj is None:
            return False

        # Quick check via module name first (no import needed)
        module = getattr(type(obj), '__module__', '')
        if 'tensorflow' not in module and 'keras' not in module:
            # Also check for fit/predict/compile which are keras signatures
            if not (hasattr(obj, 'fit') and hasattr(obj, 'predict') and hasattr(obj, 'compile')):
                return False

        try:
            keras = _get_keras()
            return (isinstance(obj, keras.Model) or
                   isinstance(obj, keras.Sequential))
        except Exception:
            return False

    @classmethod
    def _is_tensorflow_model_or_function(cls, obj: Any) -> bool:
        """Check if object is a TensorFlow model, function, or serialized configuration.

        Recognizes:
        - TensorFlow/Keras model instances
        - Functions decorated with @framework('tensorflow')
        - Serialized function dictionaries with 'function' key pointing to TensorFlow code

        Args:
            obj: Object to check (model, function, dict, or other).

        Returns:
            True if object is TensorFlow-related, False otherwise.
        """
        if not TENSORFLOW_AVAILABLE:
            return False

        # Check if it's a TensorFlow model instance
        if cls._is_tensorflow_model(obj):
            return True

        # Check if it's a function decorated with @framework('tensorflow')
        if callable(obj) and hasattr(obj, 'framework'):
            return obj.framework == 'tensorflow'

        # Check if it's a serialized function dictionary
        if isinstance(obj, dict):
            # Check for type='function' format
            if obj.get('type') == 'function' and obj.get('framework') == 'tensorflow':
                return True

            if 'function' in obj:
                function_val = obj['function']

                # Case 1: function is a callable object
                if callable(function_val):
                    if hasattr(function_val, 'framework'):
                        return function_val.framework == 'tensorflow'
                    # Fallback: check module name
                    module_name = getattr(function_val, '__module__', '')
                    return 'tensorflow' in module_name or 'keras' in module_name

                # Case 2: function is a string path
                if isinstance(function_val, str):
                    function_path = function_val
                    # Try to import the function and check its framework
                    try:
                        mod_name, _, func_name = function_path.rpartition(".")
                        mod = __import__(mod_name, fromlist=[func_name])
                        func = getattr(mod, func_name)
                        return hasattr(func, 'framework') and func.framework == 'tensorflow'
                    except (ImportError, AttributeError, ValueError):
                        # If we can't import, check the path for tensorflow indicators
                        return 'tensorflow' in function_path.lower() or 'tf' in function_path.lower()

        return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create TensorFlow model instance from configuration.

        Delegates to ModelFactory which handles various input formats:
        - Model instances (returns as-is)
        - Callables/functions (calls with input_dim, num_classes parameters)
        - Serialized configs ({'function': path, 'params': {...}})
        - File paths to saved models

        The factory automatically extracts input dimensions from the dataset and
        injects them as parameters when calling model factory functions.

        Args:
            dataset: SpectroDataset providing input dimensions and task type.
            model_config: Model configuration (dict, instance, callable, or string path).
            force_params: Optional parameters to override in the model configuration.

        Returns:
            Instantiated and compiled TensorFlow/Keras model ready for training.

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        require_backend('tensorflow', feature='TensorFlow models')

        # Import factory here to avoid circular imports at module level
        from .factory import ModelFactory

        # Delegate entirely to ModelFactory
        model = ModelFactory.build_single_model(
            model_config,
            dataset,
            force_params or {}
        )

        return model

    def _train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """Train TensorFlow/Keras model with comprehensive configuration.

        Training pipeline:
        1. Detects task type (regression/classification) from targets
        2. Prepares compilation config (loss, optimizer, metrics)
        3. Compiles the model
        4. Prepares fit config (epochs, batch_size, callbacks, validation_split)
        5. Trains the model with model.fit()
        6. Logs training results if verbose > 1

        Uses modular configuration components:
        - TensorFlowCompilationConfig: Handles optimizer, loss, metrics
        - TensorFlowFitConfig: Handles callbacks, validation, early stopping

        Args:
            model: Uncompiled or compiled TensorFlow/Keras model.
            X_train: Training features as numpy array.
            y_train: Training targets as numpy array.
            X_val: Optional validation features.
            y_val: Optional validation targets.
            **kwargs: Training parameters including:
                - epochs (int): Number of training epochs (default: 100)
                - batch_size (int): Batch size for training (default: 32)
                - patience (int): Early stopping patience (default: 10)
                - verbose (int): Verbosity level (0-3)
                - learning_rate (float): Optimizer learning rate
                - loss (str): Loss function name
                - optimizer (str): Optimizer name
                - metrics (List[str]): Evaluation metrics

        Returns:
            Trained model with history attribute attached.

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        require_backend('tensorflow', feature='TensorFlow training')

        # Import TensorFlow-specific utilities here (lazy)
        from .tensorflow import (
            TensorFlowCompilationConfig,
            TensorFlowFitConfig,
        )

        train_params = kwargs
        verbose = train_params.get('verbose', 0)

        # Get task type from train_params (passed by base controller)
        task_type = train_params.get('task_type')
        if task_type is None:
            raise ValueError("task_type must be provided in train_params")

        if not is_gpu_available('tensorflow') and verbose > 0:
            logger.warning("No GPU detected. Training TensorFlow model on CPU may be slow.")

        # 1. Prepare compilation configuration
        compile_config = TensorFlowCompilationConfig.prepare(train_params, task_type)
        model.compile(**compile_config)
        if verbose > 2:
            print(f"   Compilation config: {compile_config}")

        # 2. Prepare fit configuration (includes callbacks)
        fit_config = TensorFlowFitConfig.prepare(train_params, X_val, y_val, verbose)

        # 3. Train the model
        history = model.fit(
            X_train, y_train,
            **fit_config
        )

        # Store training history
        model.history = history

        # 4. Log training results
        if verbose > 1:
            self._log_training_results(model, X_train, y_train, X_val, y_val, task_type)

        return model

    def _log_training_results(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        task_type: TaskType
    ) -> None:
        """Log training and validation performance scores.

        Args:
            model: Trained TensorFlow model.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            task_type: Task type enum for score calculation.
        """
        print("\n   Training completed - Evaluating performance:")

        # Training scores
        y_train_pred = self._predict_model(model, X_train)
        train_scores = self._calculate_and_print_scores(
            y_train, y_train_pred, task_type, partition="train",
            model_name=model.__class__.__name__, show_detailed_scores=False
        )

        # Validation scores if available
        if X_val is not None and y_val is not None:
            y_val_pred = self._predict_model(model, X_val)
            val_scores = self._calculate_and_print_scores(
                y_val, y_val_pred, task_type, partition="validation",
                model_name=model.__class__.__name__, show_detailed_scores=False
            )

            # Show comparison
            primary_metric = 'accuracy' if task_type.is_classification else 'r2'
            train_score = train_scores.get(primary_metric, 0)
            val_score = val_scores.get(primary_metric, 0)

            if task_type.is_classification:
                print(f"\n   Accuracy: Train={train_score:.4f} | Val={val_score:.4f}")
            else:
                print(f"\n   R² Score: Train={train_score:.4f} | Val={val_score:.4f}")

            # Warn about overfitting
            if train_score - val_score > 0.15:  # 15% difference
                print(f"   ⚠️  Warning: Possible overfitting detected (Train-Val diff: {train_score - val_score:.4f})")

    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate predictions with TensorFlow model.

        Handles output format normalization:
        - Multiclass classification (2D with >1 columns): Converts softmax probabilities
          to class indices via argmax
        - Regression/binary (1D or 2D with 1 column): Ensures column vector shape

        Args:
            model: Trained TensorFlow/Keras model.
            X: Input features as numpy array (will be prepared for TensorFlow format).

        Returns:
            Predictions as 2D numpy array (n_samples, 1) for regression/binary,
            or class indices (0 to n_classes-1) for multiclass classification.
        """
        # Prepare data to ensure correct shape for model
        X_prepared, _ = self._prepare_data(X, None, {})

        predictions = model.predict(X_prepared, verbose=0)

        # For multiclass classification, convert probabilities to class indices
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            # Multi-output: likely multiclass classification with softmax
            # Convert probabilities to class predictions (encoded labels 0-N)
            predictions = np.argmax(predictions, axis=1).reshape(-1, 1).astype(np.float32)
        elif predictions.ndim == 1:
            # Single output: reshape to column vector
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _predict_proba_model(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Get class probabilities from TensorFlow classification model.

        Returns raw softmax/sigmoid outputs before argmax conversion.
        For binary classification with single sigmoid output, converts to
        2-column format [1-p, p].

        Args:
            model: Trained TensorFlow/Keras model.
            X: Input features as numpy array.

        Returns:
            Class probabilities as (n_samples, n_classes) array,
            or None for regression models.
        """
        # Prepare data to ensure correct shape for model
        X_prepared, _ = self._prepare_data(X, None, {})

        predictions = model.predict(X_prepared, verbose=0)

        # For regression (single output), return None
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if predictions.shape[1] == 1:
            # Binary classification with sigmoid: convert to 2-column format
            # Or could be regression - check if values are in [0, 1] range
            if np.all((predictions >= 0) & (predictions <= 1)):
                return np.column_stack([1 - predictions, predictions])
            else:
                # Regression output, not probabilities
                return None

        # Multiclass: already has probability distribution
        return predictions

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for TensorFlow with proper tensor formatting.

        Delegates to TensorFlowDataPreparation which handles:
        - Conversion to float32
        - 2D data: (samples, features) → (samples, features, 1) for Conv1D
        - 3D data: Conditional transpose if shape[1] < shape[2]
          (e.g., (samples, 3, 200) → (samples, 200, 3) for Conv1D)
        - Target flattening for 2D targets with single column

        Args:
            X: Input features as numpy array (2D or 3D).
            y: Optional target values as numpy array.
            context: Execution context.

        Returns:
            Tuple of (prepared_X, prepared_y) in TensorFlow-compatible formats.
        """
        # Import here to avoid loading TensorFlow at module import time
        from .tensorflow import TensorFlowDataPreparation
        return TensorFlowDataPreparation.prepare_data(X, y, context)

    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate TensorFlow model on validation data.

        Uses model.evaluate() to compute loss. Falls back to MSE calculation
        from predictions if evaluation fails.

        Args:
            model: Trained TensorFlow/Keras model.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Loss value as float. Returns float('inf') if evaluation fails completely.
        """
        try:
            # Use model's evaluate method
            loss = model.evaluate(X_val, y_val, verbose=0)

            # If evaluate returns list (loss + metrics), take the loss
            if isinstance(loss, list):
                return loss[0]
            else:
                return loss

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error in TensorFlow model evaluation: {e}")
            try:
                # Fallback: use predictions and calculate MSE
                y_pred = model.predict(X_val, verbose=0)
                mse = np.mean((y_val - y_pred) ** 2)
                return float(mse)
            except (ValueError, TypeError, AttributeError):
                return float('inf')

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for TensorFlow models.

        TensorFlow Conv1D expects input shape (features, channels) where:
        - features = number of wavelengths/spectral points (timesteps for convolution)
        - channels = number of preprocessing methods

        The '3d_transpose' layout returns (samples, features, processings) which is correct for Conv1D.
        """
        return "3d_transpose"

    def _clone_model(self, model: Any) -> Any:
        """Clone TensorFlow model using framework-specific cloning method.

        Implements the abstract _clone_model from BaseModelController.

        Cloning strategy:
        - Callable functions: Return as-is (will be called with proper input_shape later)
        - Keras model instances: Use keras.models.clone_model() to create fresh copy
        - Other objects: Return as-is for ModelFactory to handle

        Args:
            model: Model instance, function, or configuration to clone.

        Returns:
            Cloned model (for instances) or original object (for functions/configs).
        """
        if callable(model) and hasattr(model, 'framework') and model.framework == 'tensorflow':
            # Don't clone functions - they will be called later with proper input shape
            return model

        if TENSORFLOW_AVAILABLE:
            keras = _get_keras()
            if isinstance(model, (keras.Model, keras.Sequential)):
                # TensorFlow model instance: use clone_model
                return keras.models.clone_model(model)

        # Return as is (will be handled by ModelFactory)
        return model

    # Remove the _extract_model_config override - use base class implementation
    # The base class correctly returns {'model_instance': operator, 'train_params': {...}}
    # and ModelFactory now handles 'model_instance' key properly

    def process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process hyperparameters for TensorFlow model tuning.

        Supports TensorFlow-specific parameter organization:
        - Parameters prefixed with 'compile_' are grouped under 'compile' key
          (e.g., 'compile_learning_rate' → compile['learning_rate'])
        - Parameters prefixed with 'fit_' are grouped under 'fit' key
          (e.g., 'fit_batch_size' → fit['batch_size'])
        - Other parameters are treated as model architecture parameters

        Args:
            params: Dictionary of sampled parameters.

        Returns:
            Dictionary of processed hyperparameters with proper nesting for
            TensorFlow compilation and fitting.
        """
        tf_params = {}

        for key, value in params.items():
            if key.startswith('compile_'):
                # Parameters for model compilation
                compile_key = key.replace('compile_', '')
                if 'compile' not in tf_params:
                    tf_params['compile'] = {}
                tf_params['compile'][compile_key] = value
            elif key.startswith('fit_'):
                # Parameters for model fitting
                fit_key = key.replace('fit_', '')
                if 'fit' not in tf_params:
                    tf_params['fit'] = {}
                tf_params['fit'][fit_key] = value
            else:
                # Model architecture parameters
                tf_params[key] = value

        return tf_params if tf_params else params

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: ExecutionContext,
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, bytes]]] = None,
        prediction_store: 'Predictions' = None
    ) -> Tuple[ExecutionContext, List[Tuple[str, bytes]]]:
        """Execute TensorFlow model training, finetuning, or prediction.

        Sets the preferred data layout to '3d_transpose' for TensorFlow Conv1D models,
        then delegates to the base class execute method.

        Args:
            step_info: Parsed step containing model configuration and operator.
            dataset: SpectroDataset with features, targets, and fold information.
            context: Execution context with step_id, processing history, partition info.
            runtime_context: Runtime context managing execution state.
            source: Data source index (default: -1 for primary source).
            mode: Execution mode - 'train', 'finetune', 'predict', or 'explain'.
            loaded_binaries: Optional list of (name, bytes) tuples for prediction mode,
                containing serialized model and preprocessing artifacts.
            prediction_store: External Predictions storage instance for managing
                prediction results across pipeline steps.

        Returns:
            Tuple of (updated_context, list_of_artifact_metadata) where:
            - updated_context: Context dict with added model information
            - artifact_metadata: List of serialized binary artifacts for persistence

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is not available. Please install with: "
                "pip install nirs4all[tensorflow] or pip install nirs4all[gpu]"
            )

        # Set layout preference for TensorFlow models (force_layout overrides preferred)
        context = context.with_layout(self.get_effective_layout(step_info))

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)






