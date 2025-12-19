"""
PyTorch Data Preparation

This module handles PyTorch-specific data preparation and tensor formatting.
"""

from typing import Any, Optional, Tuple
import numpy as np
from nirs4all.utils.backend import require_backend

class PyTorchDataPreparation:
    """Handles PyTorch-specific data preparation and tensor formatting."""

    @staticmethod
    def prepare_features(X: np.ndarray, device: Optional[str] = None) -> Any:
        """Prepare features for PyTorch (proper tensor formatting).

        Handles conversion to float32 and proper shape formatting:
        - 2D: (samples, features) -> (samples, features)
        - 3D: (samples, channels, features) -> (samples, channels, features)

        Args:
            X: Input features array.
            device: Device to move tensor to (optional).

        Returns:
            Prepared features tensor in float32.
        """
        require_backend('pytorch', feature='PyTorch data preparation')

        import torch

        # Convert to float32
        X = X.astype(np.float32, copy=False)

        # Convert to tensor
        X_tensor = torch.from_numpy(X)

        # Move to device if specified
        if device:
            X_tensor = X_tensor.to(device)

        return X_tensor

    @staticmethod
    def prepare_targets(y: Optional[np.ndarray], device: Optional[str] = None) -> Any:
        """Prepare targets for PyTorch.

        Converts to float32 and ensures correct shape.

        Args:
            y: Target values array (optional).
            device: Device to move tensor to (optional).

        Returns:
            Prepared targets tensor in float32, or None if input was None.
        """
        require_backend('pytorch', feature='PyTorch data preparation')

        import torch

        if y is None:
            return None

        y = y.astype(np.float32, copy=False)

        # Ensure y has correct shape (samples, targets)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_tensor = torch.from_numpy(y)

        if device:
            y_tensor = y_tensor.to(device)

        return y_tensor

    @staticmethod
    def prepare_data(
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: Any = None,
        device: Optional[str] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare both features and targets for PyTorch.

        Args:
            X: Input features array.
            y: Target values array (optional).
            context: Execution context (unused).
            device: Device to move tensors to.

        Returns:
            Tuple of (prepared_X, prepared_y).
        """
        X_prepared = PyTorchDataPreparation.prepare_features(X, device)
        y_prepared = PyTorchDataPreparation.prepare_targets(y, device)

        return X_prepared, y_prepared
