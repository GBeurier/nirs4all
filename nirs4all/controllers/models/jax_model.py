"""
JAX Model Controller - Controller for JAX/Flax models

This controller handles JAX models (specifically Flax) with support for:
- Training on JAX arrays
- Custom training loops with Optax optimizers
- Integration with Optuna for hyperparameter tuning
- Model persistence and prediction storage

Matches Flax Module objects and model configurations.

Lazy loading pattern: JAX is only imported when actually needed
for training or prediction, not at module import time.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import copy

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.core.logging import get_logger
from nirs4all.utils.backend import is_available, require_backend, is_gpu_available

logger = get_logger(__name__)

# Fast availability check at module level - no imports
JAX_AVAILABLE = is_available('jax')

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        import optax
        from flax.training import train_state
    except ImportError:
        pass


# Lazy-loaded module cache
_jax_modules: Dict[str, Any] = {}


def _get_jax():
    """Lazy load JAX with caching."""
    if 'jax' not in _jax_modules:
        require_backend('jax', feature='JAX/Flax neural networks')
        import jax
        import jax.numpy as jnp
        _jax_modules['jax'] = jax
        _jax_modules['jnp'] = jnp
    return _jax_modules['jax']


def _get_flax():
    """Lazy load Flax with caching."""
    if 'flax' not in _jax_modules:
        _get_jax()  # Ensure JAX is loaded first
        import flax.linen as nn
        _jax_modules['flax'] = nn
    return _jax_modules['flax']


def _get_optax():
    """Lazy load Optax with caching."""
    if 'optax' not in _jax_modules:
        _get_jax()  # Ensure JAX is loaded first
        import optax
        _jax_modules['optax'] = optax
    return _jax_modules['optax']


@register_controller
class JaxModelController(BaseModelController):
    """Controller for JAX/Flax models.

    Uses lazy loading pattern - JAX is only imported when
    training or prediction is actually performed.
    """

    priority = 4  # Higher priority than Sklearn (6)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match JAX models and model configurations."""
        if not JAX_AVAILABLE:
            return False

        # Check if step contains a JAX model
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            if cls._is_jax_model(model):
                return True
            # Handle dictionary config for model
            if isinstance(model, dict) and 'class' in model:
                class_name = model['class']
                if isinstance(class_name, str) and ('jax' in class_name or 'flax' in class_name):
                    return True

        # Check direct JAX objects
        if cls._is_jax_model(step):
            return True

        # Check operator if provided
        if operator is not None and cls._is_jax_model(operator):
            return True

        return False

    @classmethod
    def _is_jax_model(cls, obj: Any) -> bool:
        """Check if object is a JAX/Flax model.

        Uses module introspection first to avoid importing JAX
        for non-JAX objects.
        """
        if not JAX_AVAILABLE:
            return False

        if obj is None:
            return False

        # Check for framework attribute first (no import needed)
        if hasattr(obj, 'framework') and obj.framework == 'jax':
            return True

        # Check for dict format from deserialize_component
        if isinstance(obj, dict) and obj.get('type') == 'function' and obj.get('framework') == 'jax':
            return True

        # Quick check via module name (no import needed)
        module = getattr(type(obj), '__module__', '')
        if 'jax' not in module and 'flax' not in module:
            return False

        try:
            nn = _get_flax()
            return isinstance(obj, nn.Module)
        except Exception:
            return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create JAX model instance from configuration."""
        require_backend('jax', feature='JAX/Flax models')

        # Import factory here to avoid circular imports at module level
        from .factory import ModelFactory

        return ModelFactory.build_single_model(
            model_config,
            dataset,
            force_params or {}
        )

    def _create_train_state(self, rng, model, input_shape, learning_rate):
        """Create initial training state."""
        jnp = _jax_modules['jnp']
        optax = _get_optax()
        from flax.training import train_state

        class TrainState(train_state.TrainState):
            batch_stats: Any

        variables = model.init(rng, jnp.ones(input_shape))
        params = variables['params']
        batch_stats = variables.get('batch_stats')

        tx = optax.adam(learning_rate)
        return TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
        )

    def _train_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Train JAX model with custom training loop."""
        require_backend('jax', feature='JAX/Flax training')

        # Import JAX modules here (lazy loading)
        jax = _get_jax()
        jnp = _jax_modules['jnp']
        optax = _get_optax()

        # Import wrapper here (lazy)
        from .jax_wrapper import JaxModelWrapper

        train_params = kwargs
        verbose = train_params.get('verbose', 0)

        if not is_gpu_available('jax') and verbose > 0:
            logger.warning("No GPU detected. Training JAX model on CPU may be slow.")

        epochs = train_params.get('epochs', 100)
        batch_size = train_params.get('batch_size', 32)
        learning_rate = train_params.get('lr', train_params.get('learning_rate', 0.001))

        # Initialize RNG
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)

        # Create TrainState
        # Input shape: (1, features) or (1, features, channels)
        input_shape = (1,) + X_train.shape[1:]
        state = self._create_train_state(init_rng, model, input_shape, learning_rate)

        # Define loss function (MSE for regression, CrossEntropy for classification)
        task_type = train_params.get('task_type')
        is_classification = task_type and task_type.is_classification

        @jax.jit
        def train_step(state, batch_X, batch_y, rng):
            dropout_rng = rng

            def loss_fn(params):
                variables = {'params': params}
                if state.batch_stats is not None:
                    variables['batch_stats'] = state.batch_stats

                mutable = ['batch_stats'] if state.batch_stats is not None else []
                rngs = {'dropout': dropout_rng}

                if mutable:
                    logits, new_model_state = state.apply_fn(
                        variables, batch_X, train=True, mutable=mutable, rngs=rngs
                    )
                else:
                    logits = state.apply_fn(
                        variables, batch_X, train=True, rngs=rngs
                    )
                    new_model_state = None

                if is_classification:
                    # Handle classification loss
                    if batch_y.ndim == 1 or (batch_y.ndim == 2 and batch_y.shape[1] == 1):
                         # Integer labels
                         labels = batch_y.squeeze().astype(jnp.int32)
                         loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                    else:
                         # One-hot labels
                         loss = optax.softmax_cross_entropy(logits, batch_y)
                    loss = jnp.mean(loss)
                else:
                    # Simple MSE loss for regression
                    loss = jnp.mean((logits - batch_y) ** 2)
                return loss, new_model_state

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, new_model_state), grads = grad_fn(state.params)

            new_batch_stats = state.batch_stats
            if new_model_state is not None and 'batch_stats' in new_model_state:
                new_batch_stats = new_model_state['batch_stats']

            state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
            return state, loss

        @jax.jit
        def eval_step(state, batch_X, batch_y):
            variables = {'params': state.params}
            if state.batch_stats is not None:
                variables['batch_stats'] = state.batch_stats

            logits = state.apply_fn(variables, batch_X, train=False)
            if is_classification:
                if batch_y.ndim == 1 or (batch_y.ndim == 2 and batch_y.shape[1] == 1):
                        labels = batch_y.squeeze().astype(jnp.int32)
                        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                else:
                        loss = optax.softmax_cross_entropy(logits, batch_y)
                loss = jnp.mean(loss)
            else:
                loss = jnp.mean((logits - batch_y) ** 2)
            return loss

        # Training loop
        n_samples = X_train.shape[0]
        steps_per_epoch = n_samples // batch_size

        best_val_loss = float('inf')
        best_params = None
        best_batch_stats = None
        patience = train_params.get('patience', 10)
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            rng, shuffle_rng = jax.random.split(rng)
            perms = jax.random.permutation(shuffle_rng, n_samples)
            X_train_shuffled = X_train[perms]
            y_train_shuffled = y_train[perms]

            epoch_loss = 0.0
            for i in range(steps_per_epoch):
                batch_idx = slice(i * batch_size, (i + 1) * batch_size)
                batch_X = X_train_shuffled[batch_idx]
                batch_y = y_train_shuffled[batch_idx]

                rng, step_rng = jax.random.split(rng)
                state, loss = train_step(state, batch_X, batch_y, step_rng)
                epoch_loss += loss

            epoch_loss /= steps_per_epoch

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = eval_step(state, X_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = state.params
                    best_batch_stats = state.batch_stats
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose > 1 and (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

                if patience_counter >= patience:
                    if verbose > 0:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose > 1 and (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

        # Restore best params
        if best_params is not None:
            state = state.replace(params=best_params, batch_stats=best_batch_stats)

        # Attach state to model wrapper for prediction
        # Since Flax models are stateless, we need to return a wrapper that holds the state
        return JaxModelWrapper(model, state)

    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions with JAX model."""
        # Import wrapper here (lazy)
        from .jax_wrapper import JaxModelWrapper

        if isinstance(model, JaxModelWrapper):
            preds = model.predict(X)

            # Handle multiclass classification (convert logits/probs to labels)
            if preds.ndim == 2 and preds.shape[1] > 1:
                 return np.argmax(preds, axis=-1).reshape(-1, 1)

            # Ensure 2D shape for regression/binary
            if preds.ndim == 1:
                return preds.reshape(-1, 1)

            return preds
        else:
            raise ValueError("Model must be a JaxModelWrapper instance for prediction")

    def _predict_proba_model(self, model: Any, X: Any) -> Optional[np.ndarray]:
        """Get class probabilities from JAX classification model.

        Returns softmax probabilities for classification models.

        Args:
            model: Trained JAX model (JaxModelWrapper).
            X: Input features.

        Returns:
            Class probabilities as (n_samples, n_classes) array,
            or None for regression models.
        """
        jax = _get_jax()
        import jax.nn as jnn

        # Import wrapper here (lazy)
        from .jax_wrapper import JaxModelWrapper

        if not isinstance(model, JaxModelWrapper):
            return None

        # Get raw model outputs (logits)
        preds = model.predict(X)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        if preds.shape[1] == 1:
            # Binary classification with single output
            # Apply sigmoid and convert to 2-column format
            probs = jnn.sigmoid(preds)
            probs = np.asarray(probs)
            return np.column_stack([1 - probs, probs])
        else:
            # Multiclass: apply softmax
            probs = jnn.softmax(preds, axis=-1)
            return np.asarray(probs)

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: 'ExecutionContext'
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare data for JAX."""
        # Import here to avoid loading JAX at module import time
        from .jax.data_prep import JaxDataPreparation
        return JaxDataPreparation.prepare_data(X, y)

    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate JAX model."""
        # Import wrapper here (lazy)
        from .jax_wrapper import JaxModelWrapper

        if isinstance(model, JaxModelWrapper):
            predictions = model.predict(X_val)
            # Calculate MSE manually on numpy arrays
            mse = np.mean((predictions - y_val) ** 2)
            return float(mse)
        return float('inf')

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for JAX models.

        Flax Dense layers expect (batch, features).
        Flax Conv layers expect (batch, length, features) i.e. (N, L, C).
        So '3d_transpose' is suitable for Conv1D.
        """
        return "3d_transpose"

    def _clone_model(self, model: Any) -> Any:
        """Clone JAX model."""
        # Flax models are immutable dataclasses, so we can just return the model definition
        # The state is created fresh in _train_model
        return model

    def process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process hyperparameters for JAX model tuning."""
        # JAX implementation is simple, no complex nesting needed yet
        return params

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
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute JAX model controller."""
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Please install with: "
                "pip install nirs4all[jax]"
            )

        # Set layout preference (force_layout overrides preferred)
        context = context.with_layout(self.get_effective_layout(step_info))

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)

