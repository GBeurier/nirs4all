"""
JAX Model Controller - Controller for JAX/Flax models

This controller handles JAX models (specifically Flax) with support for:
- Training on JAX arrays
- Custom training loops with Optax optimizers
- Integration with Optuna for hyperparameter tuning
- Model persistence and prediction storage

Matches Flax Module objects and model configurations.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import copy

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.emoji import WARNING
from .utilities import ModelControllerUtils as ModelUtils
from .factory import ModelFactory
from .jax.data_prep import JaxDataPreparation
from nirs4all.utils.backend import JAX_AVAILABLE, check_backend_available, is_gpu_available

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


@register_controller
class JaxModelController(BaseModelController):
    """Controller for JAX/Flax models."""

    priority = 20  # Same priority as other ML frameworks

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
        """Check if object is a JAX/Flax model."""
        if not JAX_AVAILABLE:
            return False

        try:
            import flax.linen as nn
            if isinstance(obj, nn.Module):
                return True

            # Check for framework attribute (added by @framework decorator)
            if hasattr(obj, 'framework') and obj.framework == 'jax':
                return True

            # Check for dict format from deserialize_component
            if isinstance(obj, dict) and obj.get('type') == 'function' and obj.get('framework') == 'jax':
                return True

            return False
        except Exception:
            return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> Any:
        """Create JAX model instance from configuration."""
        check_backend_available('jax')

        return ModelFactory.build_single_model(
            model_config,
            dataset,
            force_params or {}
        )

    def _create_train_state(self, rng, model, input_shape, learning_rate):
        """Create initial training state."""
        import jax.numpy as jnp
        import optax
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
        check_backend_available('jax')
        import jax
        import jax.numpy as jnp
        import optax

        train_params = kwargs
        verbose = train_params.get('verbose', 0)

        if not is_gpu_available() and verbose > 0:
            print(f"{WARNING} No GPU detected. Training JAX model on CPU may be slow.")

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
                    print(f"   Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"   Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose > 1 and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

        # Restore best params
        if best_params is not None:
            state = state.replace(params=best_params, batch_stats=best_batch_stats)

        # Attach state to model wrapper for prediction
        # Since Flax models are stateless, we need to return a wrapper that holds the state
        return JaxModelWrapper(model, state)

    def _predict_model(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions with JAX model."""
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

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: 'ExecutionContext'
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare data for JAX."""
        return JaxDataPreparation.prepare_data(X, y)

    def _evaluate_model(self, model: Any, X_val: Any, y_val: Any) -> float:
        """Evaluate JAX model."""
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
        check_backend_available('jax')

        # Set layout preference
        context = context.with_layout(self.get_preferred_layout())

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)


class JaxModelWrapper:
    """Wrapper to hold Flax model definition and trained state."""
    def __init__(self, model, state):
        self.model = model
        self.state = state

    def predict(self, X):
        variables = {'params': self.state.params}
        if self.state.batch_stats is not None:
            variables['batch_stats'] = self.state.batch_stats

        logits = self.state.apply_fn(variables, X, train=False)
        return np.array(logits)

    def __getstate__(self):
        # For pickling
        return {'model': self.model, 'state': self.state}

    def __setstate__(self, state):
        self.model = state['model']
        self.state = state['state']

