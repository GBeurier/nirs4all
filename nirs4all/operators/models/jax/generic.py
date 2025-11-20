from typing import Sequence, Optional, Any
import flax.linen as nn
import jax.numpy as jnp

class JaxMLPRegressor(nn.Module):
    """Simple MLP Regressor using Flax."""
    features: Sequence[int]
    input_shape: Optional[Any] = None  # Ignored, but kept for compatibility with factory

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Flatten input if needed (batch, features...) -> (batch, flat_features)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        # Output layer for regression (1 output)
        x = nn.Dense(1)(x)
        return x

class JaxMLPClassifier(nn.Module):
    """Simple MLP Classifier using Flax."""
    features: Sequence[int]
    num_classes: int
    input_shape: Optional[Any] = None # Ignored

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Flatten input if needed
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        # Output layer for classification (num_classes outputs)
        x = nn.Dense(self.num_classes)(x)
        return x
