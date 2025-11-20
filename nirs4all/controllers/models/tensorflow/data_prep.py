"""
TensorFlow Data Preparation

This module handles TensorFlow-specific data preparation and tensor formatting.
"""

from typing import Any, Optional, Tuple
import numpy as np


class TensorFlowDataPreparation:
    """Handles TensorFlow-specific data preparation and reshaping."""

    @staticmethod
    def prepare_features(X: np.ndarray) -> np.ndarray:
        """Prepare features for TensorFlow (proper tensor formatting).

        Handles conversion to float32 and proper shape formatting:
        - 2D: (samples, features) -> reshape to (samples, features, 1) for Conv1D
        - 3D: Only transpose if needed (channels < features), ensuring Conv1D gets (samples, timesteps, channels)

        Args:
            X: Input features array.

        Returns:
            Prepared features array in float32.
        """
        # Convert to float32 for TensorFlow
        X = X.astype(np.float32, copy=False)

        # Handle 2D data: add channel dimension for Conv1D
        # (samples, features) -> (samples, features, 1)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)

        # Handle 3D data: transpose only if we have (batch, channels, features) where channels < features
        # This ensures Conv1D receives (batch, timesteps, channels) format
        # For example: (batch, 3, 200) -> (batch, 200, 3) for 200 wavelengths and 3 processings
        elif X.ndim == 3:
            if X.shape[1] < X.shape[2]:
                X = np.transpose(X, (0, 2, 1))

        return X

    @staticmethod
    def prepare_targets(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare targets for TensorFlow.

        Converts to float32 and flattens if needed.

        Args:
            y: Target values array (optional).

        Returns:
            Prepared targets array in float32, or None if input was None.
        """
        if y is None:
            return None

        y = y.astype(np.float32, copy=False)

        # Flatten if 2D with single column
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        return y

    @staticmethod
    def prepare_data(
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: Any = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare both features and targets for TensorFlow.

        Args:
            X: Input features array.
            y: Target values array (optional).
            context: Execution context (currently unused but kept for interface compatibility).

        Returns:
            Tuple of (prepared_X, prepared_y).
        """
        X_prepared = TensorFlowDataPreparation.prepare_features(X)
        y_prepared = TensorFlowDataPreparation.prepare_targets(y)

        return X_prepared, y_prepared
