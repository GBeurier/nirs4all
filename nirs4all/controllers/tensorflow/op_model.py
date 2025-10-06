"""
TensorFlow Model Controller - Controller for TensorFlow/Keras models

This controller handles TensorFlow/Keras models with support for:
- Training on 2D/3D data with proper tensor formatting
- Model compilation with loss functions and metrics
- Early stopping and callbacks support
- Integration with Optuna for hyperparameter tuning
- Model persistence and prediction storage

Matches TensorFlow/Keras model objects and model configurations.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nirs4all.dataset.predictions import Predictions

from ..models.base_model_controller import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.model_utils import ModelUtils, TaskType

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@register_controller
class TensorFlowModelController(BaseModelController):
    """Controller for TensorFlow/Keras models."""

    priority = 20  # Same priority as sklearn

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match TensorFlow/Keras models and model configurations."""
        if not TF_AVAILABLE:
            return False

        # Check if step contains a TensorFlow model or function
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            if isinstance(model, dict) and '_runtime_instance' in model:
                model = model['_runtime_instance']
            return cls._is_tensorflow_model_or_function(model)

        # Check direct TensorFlow objects or functions
        if cls._is_tensorflow_model_or_function(step):
            return True

        # Check operator if provided
        if operator is not None and cls._is_tensorflow_model_or_function(operator):
            return True

        return False

    @classmethod
    def _is_tensorflow_model(cls, obj: Any) -> bool:
        """Check if object is a TensorFlow/Keras model."""
        if not TF_AVAILABLE:
            return False

        try:
            return (isinstance(obj, keras.Model) or
                   isinstance(obj, keras.Sequential) or
                   hasattr(obj, 'fit') and hasattr(obj, 'predict') and
                   hasattr(obj, 'compile'))
        except Exception:
            return False

    @classmethod
    def _is_tensorflow_model_or_function(cls, obj: Any) -> bool:
        """Check if object is a TensorFlow/Keras model or a function decorated with @framework('tensorflow')."""
        if not TF_AVAILABLE:
            return False

        # Check if it's a TensorFlow model instance
        if cls._is_tensorflow_model(obj):
            return True

        # Check if it's a function decorated with @framework('tensorflow')
        if callable(obj) and hasattr(obj, 'framework'):
            return obj.framework == 'tensorflow'

        # Check if it's a serialized function dictionary
        if isinstance(obj, dict) and 'function' in obj:
            function_path = obj['function']
            # Try to import the function and check its framework
            try:
                mod_name, _, func_name = function_path.rpartition(".")
                mod = __import__(mod_name, fromlist=[func_name])
                func = getattr(mod, func_name)
                return hasattr(func, 'framework') and func.framework == 'tensorflow'
            except (ImportError, AttributeError):
                # If we can't import, check the path for tensorflow indicators
                return 'tensorflow' in function_path.lower() or 'tf' in function_path.lower()

        return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create TensorFlow model instance from configuration."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        if 'model_instance' in model_config:
            model = model_config['model_instance']
            if self._is_tensorflow_model(model):
                return model
            elif callable(model):
                # Assume callable models are TensorFlow model factories
                return model

        # If we have a model factory function, call it
        if 'model_factory' in model_config:
            factory = model_config['model_factory']
            factory_params = model_config.get('factory_params', {})
            return factory(**factory_params)

        raise ValueError("Could not create TensorFlow model instance from configuration")

    def _create_model_from_function(self, model_function: Any, input_shape: Tuple, params: Optional[Dict[str, Any]] = None) -> Any:
        """Create a TensorFlow model by calling a model factory function."""
        if params is None:
            params = {}

        # print(f"🏗️ Creating TensorFlow model from function {model_function.__name__} with input_shape={input_shape}")

        # Call the model function with input_shape and params
        try:
            model = model_function(input_shape, params)
            if not self._is_tensorflow_model(model):
                raise ValueError(f"Function {model_function.__name__} did not return a TensorFlow model")
            return model
        except Exception as e:
            raise ValueError(f"Error creating model from function {model_function.__name__}: {e}") from e

    def _train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """Train TensorFlow/Keras model with comprehensive parameter support and score tracking."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        train_params = kwargs  # Use kwargs as train_params
        if train_params is None:
            train_params = {}

        # Handle model factory functions
        if callable(model):
            # This is a model factory function, we need to create the actual model
            input_shape = X_train.shape[1:]  # Get input shape from training data
            model_params = train_params.get('model_params', {})
            # print(f"🏗️ Creating TensorFlow model from function {model.__name__} with input_shape={input_shape}")
            model = self._create_model_from_function(model, input_shape, model_params)

        verbose = train_params.get('verbose', 0)

        # Detect task type and auto-configure loss/metrics
        task_type_str = self._detect_task_type(y_train)

        # Convert string to TaskType enum
        if task_type_str == "regression":
            task_type = TaskType.REGRESSION
        elif task_type_str == "binary_classification":
            task_type = TaskType.BINARY_CLASSIFICATION
        elif task_type_str == "multiclass_classification":
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        else:
            task_type = TaskType.REGRESSION  # Default fallback

        # Auto-configure loss and metrics based on task type
        if 'loss' not in train_params and 'compile' not in train_params:
            default_loss = ModelUtils.get_default_loss(task_type, 'tensorflow')
            train_params['loss'] = default_loss
            if verbose > 1:
                print(f"📊 Auto-detected {task_type.value} task, using loss: {default_loss}")
        elif 'loss' in train_params:
            # Validate provided loss
            provided_loss = train_params['loss']
            if not ModelUtils.validate_loss_compatibility(provided_loss, task_type, 'tensorflow'):
                print(f"⚠️ Warning: Loss '{provided_loss}' may not be compatible with {task_type.value} task")

        if 'metrics' not in train_params and 'compile' not in train_params:
            default_metrics = ModelUtils.get_default_metrics(task_type, 'tensorflow')
            train_params['metrics'] = default_metrics
            if verbose > 1:
                print(f"📈 Using default metrics for {task_type.value}: {default_metrics}")

        # if verbose > 1:
            # print(f"🧠 Training {model.__class__.__name__} with TensorFlow")

        # Show training parameters being used
        # if verbose > 2 and train_params:
            # print(f"🔧 Training parameters: {train_params}")        # The model is already a new instance created from function
        trained_model = model

        # === COMPILATION CONFIGURATION ===
        compile_config = self._prepare_compilation_config(train_params)
        trained_model.compile(**compile_config)
        if verbose > 2:
            print(f"🏗️ Model compiled with: {compile_config}")

        # === TRAINING CONFIGURATION ===
        fit_config = self._prepare_fit_config(train_params, X_val, y_val, verbose)
        validation_data = fit_config.pop('validation_data', None)

        # Show final training configuration
        if verbose > 2:
            self._log_training_config(fit_config, train_params, validation_data)

        # === TRAINING EXECUTION ===
        history = trained_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            **fit_config
        )

        # Store training history in model for reference
        trained_model.history = history

        # === SCORE CALCULATION AND DISPLAY ===
        # Reuse the task_type enum we calculated earlier

        if verbose > 1:
            # Show detailed training scores at verbose > 1
            y_train_pred = self._predict_model(trained_model, X_train)
            train_scores = self._calculate_and_print_scores(
                y_train, y_train_pred, task_type_str, "train",
                trained_model.__class__.__name__, show_detailed_scores=False
            )
            # Display concise training summary
            if train_scores:
                best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
                best_score = train_scores.get(best_metric)
                if best_score is not None:
                    direction = "↑" if higher_is_better else "↓"
                    all_scores_str = ModelUtils.format_scores(train_scores)
                    # print(f"✅ {trained_model.__class__.__name__} - train: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")

            # Validation scores if available
            if X_val is not None and y_val is not None:
                y_val_pred = self._predict_model(trained_model, X_val)
                val_scores = self._calculate_and_print_scores(
                    y_val, y_val_pred, task_type_str, "validation",
                    trained_model.__class__.__name__, show_detailed_scores=False
                )
                # # Display concise validation summary
                # if val_scores:
                #     best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
                #     best_score = val_scores.get(best_metric)
                #     if best_score is not None:
                #         direction = "↑" if higher_is_better else "↓"
                #         all_scores_str = ModelUtils.format_scores(val_scores)
                        # print(f"✅ {trained_model.__class__.__name__} - validation: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")
            elif validation_data is not None:
                # Use validation data from training
                X_val_data, y_val_data = validation_data
                y_val_pred = self._predict_model(trained_model, X_val_data)
                val_scores = self._calculate_and_print_scores(
                    y_val_data, y_val_pred, task_type_str, "validation",
                    trained_model.__class__.__name__, show_detailed_scores=False
                )
                # Display concise validation summary
                if val_scores:
                    best_metric, higher_is_better = ModelUtils.get_best_score_metric(task_type)
                    best_score = val_scores.get(best_metric)
                    if best_score is not None:
                        direction = "↑" if higher_is_better else "↓"
                        all_scores_str = ModelUtils.format_scores(val_scores)
                        # print(f"✅ {trained_model.__class__.__name__} - validation: {best_metric}={best_score:.4f} {direction} ({all_scores_str})")

        return trained_model

    def _prepare_compilation_config(self, train_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model compilation configuration with comprehensive parameter support."""
        # Start with defaults
        compile_config = {
            'optimizer': 'adam',
            'loss': 'mse',
            'metrics': ['mae']
        }

        # Handle nested compile parameters
        if 'compile' in train_params:
            compile_config.update(train_params['compile'])

        # Handle flat parameters (for convenience)
        flat_compile_params = {}
        for key in ['optimizer', 'loss', 'metrics', 'learning_rate', 'lr']:
            if key in train_params:
                if key == 'lr':  # alias for learning_rate
                    flat_compile_params['learning_rate'] = train_params[key]
                else:
                    flat_compile_params[key] = train_params[key]

        compile_config.update(flat_compile_params)

        # Handle optimizer configuration
        compile_config = self._configure_optimizer(compile_config)

        return compile_config

    def _configure_optimizer(self, compile_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure optimizer with advanced options including learning rate."""
        optimizer = compile_config.get('optimizer', 'adam')
        learning_rate = compile_config.pop('learning_rate', None)

        # If optimizer is string and we have learning_rate, create optimizer instance
        if isinstance(optimizer, str) and learning_rate is not None:
            if optimizer.lower() == 'adam':
                compile_config['optimizer'] = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                compile_config['optimizer'] = keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                compile_config['optimizer'] = keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif optimizer.lower() == 'adagrad':
                compile_config['optimizer'] = keras.optimizers.Adagrad(learning_rate=learning_rate)
            else:
                print(f"⚠️ Unknown optimizer {optimizer}, using default with learning_rate={learning_rate}")
                compile_config['optimizer'] = keras.optimizers.Adam(learning_rate=learning_rate)

            # print(f"🔧 Created {compile_config['optimizer'].__class__.__name__} optimizer with lr={learning_rate}")

        return compile_config

    def _prepare_fit_config(self, train_params: Dict[str, Any], X_val: Optional[np.ndarray], y_val: Optional[np.ndarray], verbose: int = 0) -> Dict[str, Any]:
        """Prepare fit configuration with comprehensive callback and parameter support."""
        # Start with defaults
        fit_config = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'verbose': 1
        }

        # Handle nested fit parameters
        if 'fit' in train_params:
            fit_config.update(train_params['fit'])

        # Handle flat parameters (for convenience)
        flat_fit_params = {}
        for param in ['epochs', 'batch_size', 'validation_split', 'verbose']:
            if param in train_params:
                flat_fit_params[param] = train_params[param]

        fit_config.update(flat_fit_params)

        # Handle validation data vs validation split
        if X_val is not None and y_val is not None:
            fit_config['validation_data'] = (X_val, y_val)
            # Remove validation_split if validation_data is provided
            fit_config.pop('validation_split', None)

        # Configure callbacks
        fit_config['callbacks'] = self._configure_callbacks(train_params, fit_config.get('callbacks', []), verbose)

        return fit_config

    def _configure_callbacks(self, train_params: Dict[str, Any], existing_callbacks: List[Any], verbose: int = 0) -> List[Any]:
        """Configure comprehensive callback system including cyclic_lr, early stopping, etc."""
        callbacks = list(existing_callbacks)  # Copy existing callbacks

        # === EARLY STOPPING ===
        if not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks):
            if train_params.get('early_stopping', True):  # Default enabled
                patience = train_params.get('patience', 10)
                monitor = train_params.get('early_stopping_monitor', 'val_loss')
                # Only show early stopping messages if verbose > 1
                callback_verbose = 1 if verbose > 1 else 0
                early_stop = keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    restore_best_weights=True,
                    verbose=callback_verbose
                )
                callbacks.append(early_stop)
                # print(f"🛑 Added EarlyStopping: monitor={monitor}, patience={patience}")

        # === CYCLIC LEARNING RATE ===
        if train_params.get('cyclic_lr', False):
            base_lr = train_params.get('base_lr', 1e-4)
            max_lr = train_params.get('max_lr', 1e-2)
            step_size = train_params.get('step_size', 2000)

            def cyclic_lr_schedule(epoch, lr):
                cycle = np.floor(1 + epoch / (2 * step_size))
                x = np.abs(epoch / step_size - 2 * cycle + 1)
                new_lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
                return float(new_lr)

            # Only show learning rate messages if verbose > 2 (detailed mode)
            callback_verbose = 1 if verbose > 2 else 0
            cyclic_lr_callback = keras.callbacks.LearningRateScheduler(cyclic_lr_schedule, verbose=callback_verbose)
            callbacks.append(cyclic_lr_callback)
            # print(f"🔄 Added CyclicLR: base_lr={base_lr}, max_lr={max_lr}, step_size={step_size}")

        # === REDUCE LR ON PLATEAU ===
        if train_params.get('reduce_lr_on_plateau', False):
            monitor = train_params.get('reduce_lr_monitor', 'val_loss')
            factor = train_params.get('reduce_lr_factor', 0.2)
            patience = train_params.get('reduce_lr_patience', 10)
            # Only show reduce LR messages if verbose > 1
            callback_verbose = 1 if verbose > 1 else 0
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience,
                verbose=callback_verbose
            )
            callbacks.append(reduce_lr)
            # print(f"📉 Added ReduceLROnPlateau: monitor={monitor}, factor={factor}, patience={patience}")

        # === BEST MODEL MEMORY (like legacy system) ===
        if train_params.get('best_model_memory', True):  # Default enabled
            best_model_callback = self._create_best_model_memory_callback(verbose > 1)
            callbacks.append(best_model_callback)
            # print("🏆 Added BestModelMemory callback")

        # === CUSTOM CALLBACKS ===
        if 'custom_callbacks' in train_params:
            custom_callbacks = train_params['custom_callbacks']
            if not isinstance(custom_callbacks, list):
                custom_callbacks = [custom_callbacks]
            callbacks.extend(custom_callbacks)
            # print(f"⚙️ Added {len(custom_callbacks)} custom callback(s)")

        return callbacks

    def _create_best_model_memory_callback(self, verbose: bool = False) -> Any:
        """Create BestModelMemory callback like the legacy system."""
        class BestModelMemory(keras.callbacks.Callback):
            def __init__(self, verbose=False):
                super().__init__()
                self.best_weights = None
                self.best_val_loss = np.inf
                self.verbose = verbose

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get('val_loss')
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = self.model.get_weights()

            def on_train_end(self, logs=None):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose:
                        print(f"🏆 Restored best weights with val_loss={self.best_val_loss:.4f}")

        return BestModelMemory(verbose)

    def _log_training_config(self, fit_config: Dict[str, Any], train_params: Dict[str, Any], validation_data: Any) -> None:
        """Log comprehensive training configuration."""
        print("🏋️ Training configuration:")
        print(f"   - Epochs: {fit_config.get('epochs', 100)}")
        print(f"   - Batch size: {fit_config.get('batch_size', 32)}")

        # Optimizer info
        optimizer_info = train_params.get('optimizer', 'adam')
        if 'learning_rate' in train_params or 'lr' in train_params:
            lr = train_params.get('learning_rate') or train_params.get('lr')
            print(f"   - Optimizer: {optimizer_info} (lr={lr})")
        else:
            print(f"   - Optimizer: {optimizer_info}")

        # Loss and metrics
        print(f"   - Loss: {train_params.get('loss', 'mse')}")
        print(f"   - Metrics: {train_params.get('metrics', ['mae'])}")

        # Validation setup
        if validation_data is not None:
            print("   - Using validation data")
        else:
            print(f"   - Validation split: {fit_config.get('validation_split', 0.2)}")

        # Callback info
        callbacks = fit_config.get('callbacks', [])
        print(f"   - Callbacks: {len(callbacks)} configured")
        for cb in callbacks:
            cb_name = cb.__class__.__name__
            if hasattr(cb, '__class__') and cb.__class__.__name__ == 'EarlyStopping':
                print(f"     * EarlyStopping (patience={getattr(cb, 'patience', 'unknown')})")
            elif hasattr(cb, '__class__') and cb.__class__.__name__ == 'LearningRateScheduler':
                print("     * CyclicLR")
            elif hasattr(cb, '__class__') and cb.__class__.__name__ == 'ReduceLROnPlateau':
                print("     * ReduceLROnPlateau")
            elif hasattr(cb, '__class__') and cb.__class__.__name__ == 'BestModelMemory':
                print("     * BestModelMemory")
            else:
                print(f"     * {cb_name}")

    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate predictions with TensorFlow model."""
        # Prepare data to ensure correct shape for model
        X_prepared, _ = self._prepare_data(X, None, {})

        predictions = model.predict(X_prepared, verbose=0)

        # Ensure predictions are in the correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for TensorFlow (proper tensor formatting)."""
        # Convert to float32 for TensorFlow
        X = X.astype(np.float32)
        if y is not None:
            y = y.astype(np.float32)

        # TensorFlow CNNs typically expect 3D input (batch, time_steps, features)
        # If X is 2D (batch, features), we need to determine if it should be reshaped
        if X.ndim == 2:
            # For 1D CNNs like the NIRS models, we typically want (batch, time_steps, 1)
            # where time_steps is the number of spectral bands
            # print(f"📊 Reshaping 2D input {X.shape} to 3D for TensorFlow CNN")
            X = X.reshape(X.shape[0], X.shape[1], 1)  # Add channel dimension
        elif X.ndim == 3:
            # Check if we have (batch, channels, features) format where channels < features
            # This indicates we need to transpose to (batch, features, channels) for Conv1D
            if X.shape[1] < X.shape[2]:
                # print(f"📊 Transposing 3D input from {X.shape} (batch, channels, features) to (batch, features, channels)")
                X = np.transpose(X, (0, 2, 1))  # (batch, channels, features) -> (batch, features, channels)
        elif X.ndim == 1:
            # Single sample case
            X = X.reshape(1, X.shape[0], 1)

        # Ensure y has proper shape (flatten for most cases)
        if y is not None and y.ndim > 1 and y.shape[1] == 1:
            y = y.flatten()

        # print(f"📊 TensorFlow data prepared: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate TensorFlow model."""
        try:
            # Use model's evaluate method
            loss = model.evaluate(X_val, y_val, verbose=0)

            # If evaluate returns list (loss + metrics), take the loss
            if isinstance(loss, list):
                return loss[0]
            else:
                return loss

        except (ValueError, TypeError, AttributeError) as e:
            print(f"⚠️ Error in TensorFlow model evaluation: {e}")
            try:
                # Fallback: use predictions and calculate MSE
                y_pred = model.predict(X_val, verbose=0)
                mse = np.mean((y_val - y_pred) ** 2)
                return float(mse)
            except (ValueError, TypeError, AttributeError):
                return float('inf')

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for TensorFlow models."""
        return "3d"

    def _clone_model(self, model: Any) -> Any:
        """Clone TensorFlow model, handling model factory functions."""
        if callable(model) and hasattr(model, 'framework') and model.framework == 'tensorflow':
            # Don't clone functions - they will be called later with proper input shape
            # print(f"🔗 Model function {model.__name__} will be instantiated during training")
            return model
        else:
            # Use parent implementation for actual model instances
            return super()._clone_model(model)

    def _extract_model_config(self, step: Any, operator: Any = None) -> Dict[str, Any]:
        """Extract model configuration from step, handling TensorFlow-specific cases."""
        if isinstance(step, dict):
            model_config = {}

            if 'model' in step:
                model = step['model']
                # Handle serialized model functions
                if isinstance(model, dict) and 'function' in model:
                    # Import the function from the serialized form
                    function_path = model['function']
                    try:
                        mod_name, _, func_name = function_path.rpartition(".")
                        mod = __import__(mod_name, fromlist=[func_name])
                        func = getattr(mod, func_name)
                        model_config['model_instance'] = func
                    except (ImportError, AttributeError) as e:
                        raise ValueError(f"Could not import function {function_path}: {e}")
                # Handle runtime instance
                elif isinstance(model, dict) and '_runtime_instance' in model:
                    model_config['model_instance'] = model['_runtime_instance']
                # Handle model factory functions or direct models
                elif callable(model) and hasattr(model, 'framework') and model.framework == 'tensorflow':
                    model_config['model_instance'] = model
                else:
                    model_config['model_instance'] = model

            # Extract other parameters
            for key in ['train_params', 'finetune_params']:
                if key in step:
                    model_config[key] = step[key]

            return model_config
        else:
            # Handle direct model or function
            if callable(step) and hasattr(step, 'framework') and step.framework == 'tensorflow':
                return {'model_instance': step}
            else:
                return {'model_instance': step}

    def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters specific to TensorFlow models."""
        params = super()._sample_hyperparameters(trial, finetune_params)

        # Add TensorFlow-specific parameter handling
        # Handle nested parameters for compile and fit
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

        return tf_params

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
        prediction_store: 'Predictions' = None
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """Execute TensorFlow model controller."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install tensorflow.")

        # Set layout preference for TensorFlow models
        context = context.copy()
        context['layout'] = self.get_preferred_layout()

        # Call parent execute method
        return super().execute(step, operator, dataset, context, runner, source, mode, loaded_binaries, prediction_store)
