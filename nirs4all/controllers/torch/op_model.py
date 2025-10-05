# """
# PyTorch Model Controller - Controller for PyTorch models

# This controller handles PyTorch models with support for:
# - Training on tensor data with proper device management (CPU/GPU)
# - Custom training loops with loss functions and optimizers
# - Learning rate scheduling and model checkpointing
# - Integration with Optuna for hyperparameter tuning
# - Model persistence and prediction storage

# Matches PyTorch nn.Module objects and model configurations.
# """

# from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
# import numpy as np

# from ..models.base_model_controller import BaseModelController
# from nirs4all.controllers.registry import register_controller

# if TYPE_CHECKING:
#     from nirs4all.pipeline.runner import PipelineRunner
#     from nirs4all.dataset.dataset import SpectroDataset

# # Try to import PyTorch
# try:
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     from torch.utils.data import DataLoader, TensorDataset
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False


# @register_controller
# class PyTorchModelController(BaseModelController):
#     """Controller for PyTorch models."""

#     priority = 20  # Same priority as other ML frameworks

#     @classmethod
#     def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
#         """Match PyTorch models and model configurations."""
#         if not TORCH_AVAILABLE:
#             return False

#         # Check if step contains a PyTorch model
#         if isinstance(step, dict) and 'model' in step:
#             model = step['model']
#             if isinstance(model, dict) and '_runtime_instance' in model:
#                 model = model['_runtime_instance']
#             return cls._is_pytorch_model(model)

#         # Check direct PyTorch objects
#         if cls._is_pytorch_model(step):
#             return True

#         # Check operator if provided
#         if operator is not None and cls._is_pytorch_model(operator):
#             return True

#         return False

#     @classmethod
#     def _is_pytorch_model(cls, obj: Any) -> bool:
#         """Check if object is a PyTorch model."""
#         if not TORCH_AVAILABLE:
#             return False

#         try:
#             return isinstance(obj, nn.Module)
#         except Exception:
#             return False

#     def _get_model_instance(self, dataset: SpectroDataset, model_config: Dict[str, Any]) -> nn.Module:
#         """Create PyTorch model instance from configuration."""
#         if not TORCH_AVAILABLE:
#             raise ImportError("PyTorch is not available")

#         if 'model_instance' in model_config:
#             model = model_config['model_instance']
#             if isinstance(model, nn.Module):
#                 return model

#         # If we have a model factory function, call it
#         if 'model_factory' in model_config:
#             factory = model_config['model_factory']
#             factory_params = model_config.get('factory_params', {})
#             return factory(**factory_params)

#         raise ValueError("Could not create PyTorch model instance from configuration")

#     def _train_model(
#         self,
#         model: nn.Module,
#         X_train: torch.Tensor,
#         y_train: torch.Tensor,
#         X_val: Optional[torch.Tensor] = None,
#         y_val: Optional[torch.Tensor] = None,
#         train_params: Optional[Dict[str, Any]] = None
#     ) -> nn.Module:
#         """Train PyTorch model with custom training loop."""
#         if not TORCH_AVAILABLE:
#             raise ImportError("PyTorch is not available")

#         if train_params is None:
#             train_params = {}

#         print(f"⚡ Training {model.__class__.__name__} with PyTorch")

#         # Setup device
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)
#         X_train = X_train.to(device)
#         y_train = y_train.to(device)

#         if X_val is not None:
#             X_val = X_val.to(device)
#         if y_val is not None:
#             y_val = y_val.to(device)

#         # Setup optimizer
#         optimizer_config = train_params.get('optimizer', {'type': 'Adam', 'lr': 0.001})
#         if isinstance(optimizer_config, dict):
#             opt_type = optimizer_config.pop('type', 'Adam')
#             optimizer_class = getattr(optim, opt_type)
#             optimizer = optimizer_class(model.parameters(), **optimizer_config)
#         else:
#             optimizer = optimizer_config  # Assume it's already an optimizer instance

#         # Setup loss function
#         loss_fn_config = train_params.get('loss', 'MSELoss')
#         if isinstance(loss_fn_config, str):
#             loss_fn = getattr(nn, loss_fn_config)()
#         else:
#             loss_fn = loss_fn_config  # Assume it's already a loss function

#         # Training parameters
#         epochs = train_params.get('epochs', 100)
#         batch_size = train_params.get('batch_size', 32)
#         patience = train_params.get('patience', 10)
#         verbose = train_params.get('verbose', True)

#         # Create data loaders
#         train_dataset = TensorDataset(X_train, y_train)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#         val_loader = None
#         if X_val is not None and y_val is not None:
#             val_dataset = TensorDataset(X_val, y_val)
#             val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         # Training loop with early stopping
#         best_val_loss = float('inf')
#         best_model_state = None
#         patience_counter = 0

#         for epoch in range(epochs):
#             # Training phase
#             model.train()
#             train_loss = 0.0

#             for batch_X, batch_y in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(batch_X)
#                 loss = loss_fn(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()

#             train_loss /= len(train_loader)

#             # Validation phase
#             val_loss = 0.0
#             if val_loader is not None:
#                 model.eval()
#                 with torch.no_grad():
#                     for batch_X, batch_y in val_loader:
#                         outputs = model(batch_X)
#                         loss = loss_fn(outputs, batch_y)
#                         val_loss += loss.item()
#                 val_loss /= len(val_loader)

#                 # Early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     best_model_state = model.state_dict().copy()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1

#                 if verbose and (epoch + 1) % 10 == 0:
#                     print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#                 if patience_counter >= patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
#             else:
#                 if verbose and (epoch + 1) % 10 == 0:
#                     print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")

#         # Load best model weights if we have them
#         if best_model_state is not None:
#             model.load_state_dict(best_model_state)

#         return model

#     def _predict_model(self, model: nn.Module, X: torch.Tensor) -> np.ndarray:
#         """Generate predictions with PyTorch model."""
#         device = next(model.parameters()).device
#         X = X.to(device)

#         model.eval()
#         with torch.no_grad():
#             predictions = model(X)
#             predictions = predictions.cpu().numpy()

#         # Ensure predictions are in the correct shape
#         if predictions.ndim == 1:
#             predictions = predictions.reshape(-1, 1)

#         return predictions

#     def _prepare_data(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#         context: Dict[str, Any]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Prepare data for PyTorch (convert to tensors)."""
#         # Convert to tensors with appropriate dtype
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32)

#         # Ensure y has correct shape
#         if y_tensor.ndim == 1:
#             y_tensor = y_tensor.unsqueeze(1)

#         return X_tensor, y_tensor

#     def _evaluate_model(self, model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
#         """Evaluate PyTorch model."""
#         try:
#             device = next(model.parameters()).device
#             X_val = X_val.to(device)
#             y_val = y_val.to(device)

#             model.eval()
#             with torch.no_grad():
#                 predictions = model(X_val)
#                 mse_loss = nn.MSELoss()
#                 loss = mse_loss(predictions, y_val)
#                 return loss.item()

#         except Exception as e:
#             print(f"⚠️ Error in PyTorch model evaluation: {e}")
#             return float('inf')

#     def _sample_hyperparameters(self, trial, finetune_params: Dict[str, Any]) -> Dict[str, Any]:
#         """Sample hyperparameters specific to PyTorch models."""
#         params = super()._sample_hyperparameters(trial, finetune_params)

#         # Add PyTorch-specific parameter handling
#         # Handle nested parameters for optimizer and training
#         torch_params = {}

#         for key, value in params.items():
#             if key.startswith('optimizer_'):
#                 # Parameters for optimizer
#                 opt_key = key.replace('optimizer_', '')
#                 if 'optimizer' not in torch_params:
#                     torch_params['optimizer'] = {}
#                 torch_params['optimizer'][opt_key] = value
#             else:
#                 # Model or training parameters
#                 torch_params[key] = value

#         return torch_params

#     def execute(
#         self,
#         step: Any,
#         operator: Any,
#         dataset: 'SpectroDataset',
#         context: Dict[str, Any],
#         runner: 'PipelineRunner',
#         source: int = -1,
#         mode: str = "train",
#         loaded_binaries: Optional[List[Tuple[str, bytes]]] = None
#     ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
#         """Execute PyTorch model controller."""
#         if not TORCH_AVAILABLE:
#             raise ImportError("PyTorch is not available. Please install torch.")

#         print(f"⚡ Executing PyTorch model controller")

#         # Call parent execute method
#         return super().execute(step, operator, dataset, context, runner, source, mode, loaded_binaries)