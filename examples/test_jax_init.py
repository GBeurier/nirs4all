
from typing import Sequence, Optional, Any
import flax.linen as nn
import jax.numpy as jnp

class JaxMLPRegressor(nn.Module):
    """Simple MLP Regressor using Flax."""
    features: Sequence[int]
    input_shape: Optional[Any] = None

    @nn.compact
    def __call__(self, x):
        pass

try:
    model = JaxMLPRegressor(features=[64, 32])
    print("Instantiation successful")
    print(model)
except Exception as e:
    print(f"Instantiation failed: {e}")
