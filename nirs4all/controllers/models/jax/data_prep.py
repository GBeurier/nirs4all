"""
JAX Data Preparation

This module handles JAX-specific data preparation and array formatting.
"""

from typing import Any, Optional, Tuple
import numpy as np
from nirs4all.utils.backend import require_backend

class JaxDataPreparation:
    """Handles JAX-specific data preparation and array formatting."""

    @staticmethod
    def prepare_features(X: np.ndarray) -> Any:
        """Prepare features for JAX (proper array formatting).

        Handles conversion to float32 and proper shape formatting:
        - 2D: (samples, features) -> (samples, features)
        - 3D: (samples, channels, features) -> (samples, channels, features)

        Args:
            X: Input features array.

        Returns:
            Prepared features array in float32 (JAX array).
        """
        require_backend('jax', feature='JAX data preparation')

        import jax.numpy as jnp

        # Convert to float32
        X = X.astype(np.float32, copy=False)

        # Convert to JAX array
        X_jax = jnp.array(X)

        return X_jax

    @staticmethod
    def prepare_targets(y: Optional[np.ndarray]) -> Any:
        """Prepare targets for JAX.

        Converts to float32 and ensures correct shape.

        Args:
            y: Target values array (optional).

        Returns:
            Prepared targets array in float32, or None if input was None.
        """
        require_backend('jax', feature='JAX data preparation')

        import jax.numpy as jnp

        if y is None:
            return None

        y = y.astype(np.float32, copy=False)

        # Ensure y has correct shape (samples, targets)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_jax = jnp.array(y)

        return y_jax

    @staticmethod
    def prepare_data(
        X: np.ndarray,
        y: Optional[np.ndarray],
        context: Any = None
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare both features and targets for JAX.

        Args:
            X: Input features array.
            y: Target values array (optional).
            context: Execution context (unused).

        Returns:
            Tuple of (prepared_X, prepared_y).
        """
        X_prepared = JaxDataPreparation.prepare_features(X)
        y_prepared = JaxDataPreparation.prepare_targets(y)

        return X_prepared, y_prepared
