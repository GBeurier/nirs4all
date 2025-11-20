"""
PyTorch Model Controller - Controller for PyTorch models

This controller handles PyTorch models with support for:
- Training on tensor data with proper device management (CPU/GPU)
- Custom training loops with loss functions and optimizers
- Learning rate scheduling and model checkpointing
- Integration with Optuna for hyperparameter tuning
- Model persistence and prediction storage

Matches PyTorch nn.Module objects and model configurations.
"""

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import copy

from ..models.base_model import BaseModelController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.emoji import WARNING
from .utilities import ModelControllerUtils as ModelUtils
from .factory import ModelFactory
from .torch.data_prep import PyTorchDataPreparation
from nirs4all.utils.backend import TORCH_AVAILABLE, check_backend_available, is_gpu_available

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.config.context import ExecutionContext
    from nirs4all.pipeline.steps.parser import ParsedStep
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        pass


@register_controller
class PyTorchModelController(BaseModelController):
    """Controller for PyTorch models."""

    priority = 20  # Same priority as other ML frameworks

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match PyTorch models and model configurations."""
        if not TORCH_AVAILABLE:
            return False

        # Check if step contains a PyTorch model
        if isinstance(step, dict) and 'model' in step:
            model = step['model']
            if cls._is_pytorch_model(model):
                return True
            # Handle dictionary config for model
            if isinstance(model, dict) and 'class' in model:
                class_name = model['class']
                if isinstance(class_name, str) and 'torch' in class_name:
                    return True

        # Check direct PyTorch objects
        if cls._is_pytorch_model(step):
            return True

        # Check operator if provided
        if operator is not None and cls._is_pytorch_model(operator):
            return True

        return False

    @classmethod
    def _is_pytorch_model(cls, obj: Any) -> bool:
        """Check if object is a PyTorch model."""
        if not TORCH_AVAILABLE:
            return False

        try:
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                return True

            # Check for framework attribute (added by @framework decorator)
            if hasattr(obj, 'framework') and obj.framework == 'pytorch':
                return True

            # Check for dict format from deserialize_component
            if isinstance(obj, dict) and obj.get('type') == 'function' and obj.get('framework') == 'pytorch':
                return True

            return False
        except Exception:
            return False

    def _get_model_instance(self, dataset: 'SpectroDataset', model_config: Dict[str, Any], force_params: Optional[Dict[str, Any]] = None) -> 'nn.Module':
        """Create PyTorch model instance from configuration."""
        check_backend_available('torch')

        return ModelFactory.build_single_model(
            model_config,
            dataset,
            force_params or {}
        )

    def _train_model(
        self,
        model: 'nn.Module',
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **kwargs
    ) -> 'nn.Module':
        """Train PyTorch model with custom training loop."""
        check_backend_available('torch')
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        train_params = kwargs
        verbose = train_params.get('verbose', 0)

        if not is_gpu_available() and verbose > 0:
            print(f"{WARNING} No GPU detected. Training PyTorch model on CPU may be slow.")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Data is already prepared as tensors by _prepare_data, just move to device
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        if X_val is not None:
            X_val = X_val.to(device)
        if y_val is not None:
            y_val = y_val.to(device)

        # Setup optimizer
        optimizer_config = train_params.get('optimizer', 'Adam')
        lr = train_params.get('lr', train_params.get('learning_rate', 0.001))

        if isinstance(optimizer_config, str):
            optimizer_class = getattr(optim, optimizer_config)
            optimizer = optimizer_class(model.parameters(), lr=lr)
        elif isinstance(optimizer_config, dict):
            opt_type = optimizer_config.pop('type', 'Adam')
            optimizer_class = getattr(optim, opt_type)
            optimizer = optimizer_class(model.parameters(), **optimizer_config)
        else:
            optimizer = optimizer_config

        # Setup loss function
        loss_fn_config = train_params.get('loss', 'MSELoss')
        if isinstance(loss_fn_config, str):
            # Handle common loss names
            if loss_fn_config.lower() == 'mse':
                loss_fn = nn.MSELoss()
            elif loss_fn_config.lower() == 'mae':
                loss_fn = nn.L1Loss()
            elif loss_fn_config.lower() == 'crossentropy':
                loss_fn = nn.CrossEntropyLoss()
            elif hasattr(nn, loss_fn_config):
                loss_fn = getattr(nn, loss_fn_config)()
            else:
                loss_fn = nn.MSELoss() # Default
        else:
            loss_fn = loss_fn_config

        # Training parameters
        epochs = train_params.get('epochs', 100)
        batch_size = train_params.get('batch_size', 32)
        patience = train_params.get('patience', 10)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = loss_fn(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose > 1 and (epoch + 1) % 10 == 0:
                    print(f"   Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"   Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose > 1 and (epoch + 1) % 10 == 0:
                    print(f"   Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")

        # Load best model weights if we have them
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model

    def _predict_model(self, model: 'nn.Module', X: Any) -> np.ndarray:
        """Generate predictions with PyTorch model."""
        import torch
        device = next(model.parameters()).device

        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
             X = PyTorchDataPreparation.prepare_features(X, device)
        else:
             X = X.to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.cpu().numpy()

        # Handle multiclass classification (convert logits/probs to labels)
        if predictions.ndim == 2 and predictions.shape[1] > 1:
             # Multi-output: likely multiclass classification with softmax/logits
             # Convert probabilities to class predictions (encoded labels 0-N)
             predictions = np.argmax(predictions, axis=1).reshape(-1, 1).astype(np.float32)
        elif predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: 'ExecutionContext'
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare data for PyTorch (convert to tensors)."""
        return PyTorchDataPreparation.prepare_data(X, y)

    def _evaluate_model(self, model: 'nn.Module', X_val: Any, y_val: Any) -> float:
        """Evaluate PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            device = next(model.parameters()).device
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            model.eval()
            with torch.no_grad():
                predictions = model(X_val)
                mse_loss = nn.MSELoss()
                loss = mse_loss(predictions, y_val)
                return loss.item()

        except Exception as e:
            print(f"{WARNING}Error in PyTorch model evaluation: {e}")
            return float('inf')

    def get_preferred_layout(self) -> str:
        """Return the preferred data layout for PyTorch models.

        PyTorch typically expects (samples, channels, features) for 1D convs.
        We use '3d' which gives (samples, processings, features) -> (N, C, L).
        """
        return "3d"

    def _clone_model(self, model: 'nn.Module') -> 'nn.Module':
        """Clone PyTorch model."""
        # For PyTorch, we can use deepcopy to get a fresh model with same architecture
        # But we need to reset parameters to ensure fresh weights
        cloned = copy.deepcopy(model)

        # Reset parameters
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        cloned.apply(weight_reset)
        return cloned

    def process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process hyperparameters for PyTorch model tuning."""
        torch_params = {}

        for key, value in params.items():
            if key.startswith('optimizer_'):
                # Parameters for optimizer
                opt_key = key.replace('optimizer_', '')
                if 'optimizer' not in torch_params:
                    torch_params['optimizer'] = {}
                torch_params['optimizer'][opt_key] = value
            else:
                # Model or training parameters
                torch_params[key] = value

        return torch_params if torch_params else params

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
        """Execute PyTorch model controller."""
        check_backend_available('torch')

        # Set layout preference
        context = context.with_layout(self.get_preferred_layout())

        # Call parent execute method
        return super().execute(step_info, dataset, context, runtime_context, source, mode, loaded_binaries, prediction_store)
