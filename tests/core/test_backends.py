import pytest
import numpy as np
from nirs4all.utils.backend_utils import (
    is_tensorflow_available, is_torch_available, 
    is_keras_available, is_jax_available
)

def test_sklearn_model():
    """Test basique qui s'exécute toujours car il utilise seulement scikit-learn."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=5)
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10)
    clf.fit(X, y)
    assert hasattr(clf, 'predict')

@pytest.mark.tensorflow
def test_tensorflow_model():
    """Test qui s'exécute uniquement si TensorFlow est installé."""
    assert is_tensorflow_available()
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(4,))
    ])
    assert isinstance(model, tf.keras.Model)

@pytest.mark.torch
def test_pytorch_model():
    """Test qui s'exécute uniquement si PyTorch est installé."""
    assert is_torch_available()
    import torch
    import torch.nn as nn
    model = nn.Linear(4, 1)
    assert isinstance(model, nn.Module)

@pytest.mark.keras
def test_keras3_model():
    """Test qui s'exécute uniquement si Keras 3 est installé."""
    assert is_keras_available()
    import keras
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(4,))
    ])
    assert isinstance(model, keras.Model)

@pytest.mark.jax
def test_jax_function():
    """Test qui s'exécute uniquement si JAX est installé."""
    assert is_jax_available()
    import jax
    import jax.numpy as jnp
    def func(x):
        return jnp.sum(x)
    f = jax.jit(func)
    x = jnp.array([1, 2, 3, 4])
    assert f(x) == 10

@pytest.mark.gpu
def test_gpu_availability():
    """Test qui s'exécute uniquement si un GPU est disponible."""

    if is_torch_available():
        import torch
        assert torch.cuda.is_available()
    elif is_tensorflow_available():  
        import tensorflow as tf
        assert len(tf.config.list_physical_devices('GPU')) > 0
    elif is_jax_available():
        import jax
        assert jax.default_backend() == 'gpu'