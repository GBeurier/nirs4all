import pytest
import numpy as np
from nirs4all.utils.backend import JAX_AVAILABLE

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJaxModels:
    def test_mlp_regressor(self):
        from nirs4all.operators.models.jax import JaxMLPRegressor
        import jax
        import jax.numpy as jnp

        model = JaxMLPRegressor(features=[10, 5])
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 20))
        params = model.init(key, x)
        y = model.apply(params, x)
        assert y.shape == (1, 1)

    def test_mlp_classifier(self):
        from nirs4all.operators.models.jax import JaxMLPClassifier
        import jax
        import jax.numpy as jnp

        model = JaxMLPClassifier(features=[10, 5], num_classes=3)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 20))
        params = model.init(key, x)
        y = model.apply(params, x)
        assert y.shape == (1, 3)
