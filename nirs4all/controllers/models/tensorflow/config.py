"""
TensorFlow Configuration Management

This module provides classes for managing TensorFlow model configuration:
- Compilation configuration (optimizer, loss, metrics)
- Fit configuration (epochs, batch_size, validation)
- Callback factory for creating various callbacks
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from nirs4all.core.task_type import TaskType
from ..utilities import ModelControllerUtils as ModelUtils
from nirs4all.utils.backend import is_available, require_backend
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

# Fast availability check at module level - no imports
TF_AVAILABLE = is_available('tensorflow')

if TYPE_CHECKING:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        pass


class TensorFlowCompilationConfig:
    """Manages TensorFlow model compilation configuration."""

    @staticmethod
    def prepare(train_params: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
        """Prepare compilation configuration from training parameters.

        Args:
            train_params: Dictionary with training parameters (may include 'compile' key).
            task_type: TaskType enum indicating classification or regression.

        Returns:
            Dictionary with 'optimizer', 'loss', and 'metrics' keys.
        """
        require_backend('tensorflow', feature='TensorFlow compilation')

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
            if key in train_params and key not in ['compile']:
                flat_compile_params[key] = train_params[key]

        compile_config.update(flat_compile_params)

        # Auto-configure loss and metrics based on task type if not explicitly set
        if 'loss' not in train_params and 'compile' not in train_params:
            compile_config['loss'] = ModelUtils.get_default_loss(task_type, framework='tensorflow')

        if 'metrics' not in train_params and 'compile' not in train_params:
            compile_config['metrics'] = ModelUtils.get_default_metrics(task_type, framework='tensorflow')

        # Handle optimizer configuration with learning rate
        compile_config = TensorFlowCompilationConfig._configure_optimizer(compile_config, train_params)

        return compile_config

    @staticmethod
    def _configure_optimizer(compile_config: Dict[str, Any], train_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Configure optimizer with learning rate if provided.

        Args:
            compile_config: Configuration dictionary with 'optimizer' and possibly 'learning_rate'.
            train_params: Training parameters to check for cyclic_lr.

        Returns:
            Updated configuration with optimizer instance if learning_rate was provided.
        """
        if not TF_AVAILABLE:
            return compile_config

        optimizer = compile_config.get('optimizer', 'adam')
        learning_rate = compile_config.pop('learning_rate', None)

        # Also check for 'lr' as alias
        if learning_rate is None:
            learning_rate = compile_config.pop('lr', None)

        # Check if cyclic_lr is enabled
        is_cyclic = train_params and train_params.get('cyclic_lr', False)

        # If optimizer is string and (we have learning_rate OR cyclic_lr is enabled), create optimizer instance
        if isinstance(optimizer, str):
            if learning_rate is not None:
                compile_config['optimizer'] = TensorFlowCompilationConfig.create_optimizer(
                    optimizer, learning_rate
                )
            elif is_cyclic:
                # If cyclic LR is enabled but no initial LR provided, use default (e.g. 0.001)
                # but explicitly create instance so we have a variable LR
                compile_config['optimizer'] = TensorFlowCompilationConfig.create_optimizer(
                    optimizer, 0.001  # Default LR
                )

        return compile_config

    @staticmethod
    def create_optimizer(optimizer_name: str, learning_rate: float) -> 'keras.optimizers.Optimizer':
        """Create optimizer instance with learning rate.

        Args:
            optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop', etc.).
            learning_rate: Learning rate value.

        Returns:
            Configured optimizer instance.
        """
        require_backend('tensorflow', feature='TensorFlow optimizer')
        from tensorflow import keras

        optimizer_classes = {
            'adam': keras.optimizers.Adam,
            'sgd': keras.optimizers.SGD,
            'rmsprop': keras.optimizers.RMSprop,
            'adagrad': keras.optimizers.Adagrad,
            'adadelta': keras.optimizers.Adadelta,
            'adamax': keras.optimizers.Adamax,
            'nadam': keras.optimizers.Nadam,
        }

        optimizer_class = optimizer_classes.get(optimizer_name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer_class(learning_rate=learning_rate)


class TensorFlowFitConfig:
    """Manages TensorFlow model fit configuration."""

    @staticmethod
    def prepare(
        train_params: Dict[str, Any],
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: int = 0
    ) -> Dict[str, Any]:
        """Prepare fit configuration including validation setup.

        Args:
            train_params: Dictionary with training parameters (may include 'fit' key).
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            verbose: Verbosity level for logging.

        Returns:
            Dictionary with fit parameters including 'callbacks'.
        """
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
            if param in train_params and param not in ['fit', 'compile']:
                flat_fit_params[param] = train_params[param]

        fit_config.update(flat_fit_params)

        # Handle validation data vs validation split
        if X_val is not None and y_val is not None:
            fit_config['validation_data'] = (X_val, y_val)
            fit_config.pop('validation_split', None)

        # Configure callbacks
        fit_config['callbacks'] = TensorFlowCallbackFactory.create_callbacks(
            train_params,
            fit_config.get('callbacks', []),
            verbose
        )

        return fit_config


class TensorFlowCallbackFactory:
    """Factory for creating TensorFlow callbacks."""

    @staticmethod
    def create_callbacks(
        train_params: Dict[str, Any],
        existing_callbacks: List[Any],
        verbose: int = 0
    ) -> List[Any]:
        """Create comprehensive callback system.

        Args:
            train_params: Training parameters with callback configuration.
            existing_callbacks: List of existing callback instances.
            verbose: Verbosity level for logging.

        Returns:
            List of callback instances.
        """
        if not TF_AVAILABLE:
            return existing_callbacks

        from tensorflow import keras

        callbacks = list(existing_callbacks)

        # === EARLY STOPPING ===
        if not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks):
            early_stopping_params = train_params.get('early_stopping', {})
            if early_stopping_params or train_params.get('patience'):
                callbacks.append(
                    TensorFlowCallbackFactory.create_early_stopping(train_params, verbose)
                )

        # === CYCLIC LEARNING RATE ===
        if train_params.get('cyclic_lr', False):
            callbacks.append(
                TensorFlowCallbackFactory.create_cyclic_lr(train_params, verbose)
            )

        # === REDUCE LR ON PLATEAU ===
        if train_params.get('reduce_lr_on_plateau', False):
            callbacks.append(
                TensorFlowCallbackFactory.create_reduce_lr_on_plateau(train_params, verbose)
            )

        # === BEST MODEL MEMORY ===
        if train_params.get('best_model_memory', True):
            callbacks.append(
                TensorFlowCallbackFactory.create_best_model_memory(verbose)
            )

        # === CUSTOM CALLBACKS ===
        if 'custom_callbacks' in train_params:
            custom_cbs = train_params['custom_callbacks']
            if not isinstance(custom_cbs, list):
                custom_cbs = [custom_cbs]
            callbacks.extend(custom_cbs)

        return callbacks

    @staticmethod
    def create_early_stopping(train_params: Dict[str, Any], verbose: int = 0) -> 'keras.callbacks.EarlyStopping':
        """Create early stopping callback.

        Args:
            train_params: Training parameters with early_stopping config.
            verbose: Verbosity level.

        Returns:
            EarlyStopping callback instance.
        """
        require_backend('tensorflow', feature='TensorFlow callbacks')
        from tensorflow import keras

        early_stopping_params = train_params.get('early_stopping', {})

        # Handle flat patience parameter
        if 'patience' in train_params and 'patience' not in early_stopping_params:
            early_stopping_params['patience'] = train_params['patience']

        es_config = {
            'monitor': 'val_loss',
            'patience': 10,
            'restore_best_weights': True,
            'verbose': 1 if verbose > 0 else 0
        }
        es_config.update(early_stopping_params)

        return keras.callbacks.EarlyStopping(**es_config)

    @staticmethod
    def create_cyclic_lr(train_params: Dict[str, Any], verbose: int = 0) -> 'keras.callbacks.Callback':
        """Create cyclic learning rate callback.

        Args:
            train_params: Training parameters with cyclic_lr config.
            verbose: Verbosity level.

        Returns:
            Custom cyclic LR callback instance.
        """
        require_backend('tensorflow', feature='TensorFlow callbacks')
        from tensorflow import keras

        cyclic_lr_params = train_params.get('cyclic_lr_params', {})

        class CyclicLR(keras.callbacks.Callback):
            def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, mode='triangular', verbose=0):
                super().__init__()
                self.base_lr = base_lr
                self.max_lr = max_lr
                self.step_size = step_size
                self.mode = mode
                self.verbose = verbose
                self.clr_iterations = 0
                self.history = {}

            def on_train_begin(self, logs=None):
                if self.verbose > 0:
                    logger.debug(f"Cyclic LR: base={self.base_lr}, max={self.max_lr}, step_size={self.step_size}")

            def on_batch_end(self, batch, logs=None):
                self.clr_iterations += 1
                cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
                x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
                lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

                try:
                    if hasattr(self.model.optimizer, 'learning_rate'):
                        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                except (AttributeError, TypeError) as e:
                    if self.verbose > 0 and self.clr_iterations == 1:
                        logger.warning(f"Could not set learning rate for CyclicLR: {e}")

        return CyclicLR(
            base_lr=cyclic_lr_params.get('base_lr', 0.001),
            max_lr=cyclic_lr_params.get('max_lr', 0.006),
            step_size=cyclic_lr_params.get('step_size', 2000),
            mode=cyclic_lr_params.get('mode', 'triangular'),
            verbose=verbose
        )

    @staticmethod
    def create_reduce_lr_on_plateau(train_params: Dict[str, Any], verbose: int = 0) -> 'keras.callbacks.ReduceLROnPlateau':
        """Create reduce LR on plateau callback.

        Args:
            train_params: Training parameters with reduce_lr_on_plateau config.
            verbose: Verbosity level.

        Returns:
            ReduceLROnPlateau callback instance.
        """
        require_backend('tensorflow', feature='TensorFlow callbacks')
        from tensorflow import keras

        reduce_lr_params = train_params.get('reduce_lr_on_plateau_params', {})

        rlr_config = {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-7,
            'verbose': 1 if verbose > 0 else 0
        }
        rlr_config.update(reduce_lr_params)

        return keras.callbacks.ReduceLROnPlateau(**rlr_config)

    @staticmethod
    def create_best_model_memory(verbose: int = 0) -> 'keras.callbacks.Callback':
        """Create best model memory callback (saves best weights during training).

        Args:
            verbose: Verbosity level.

        Returns:
            Custom best model memory callback instance.
        """
        require_backend('tensorflow', feature='TensorFlow callbacks')
        from tensorflow import keras

        class BestModelMemory(keras.callbacks.Callback):
            def __init__(self, verbose=0):
                super().__init__()
                self.best_weights = None
                self.best_loss = float('inf')
                self.verbose = verbose

            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get('val_loss', logs.get('loss'))
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_weights = self.model.get_weights()
                    if self.verbose > 1:
                        logger.debug(f"Best model saved at epoch {epoch + 1} with loss {current_loss:.4f}")

            def on_train_end(self, logs=None):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose > 0:
                        logger.info(f"Restored best model with loss {self.best_loss:.4f}")

        return BestModelMemory(verbose)
