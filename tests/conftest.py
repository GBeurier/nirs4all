import pytest
import numpy as np
from nirs4all.utils.backend_utils import (
    is_tensorflow_available, is_torch_available, 
    is_keras_available, is_jax_available, is_gpu_available
)

# Définition des marqueurs pour les backends
def pytest_configure(config):
    """Configurer les marqueurs pytest pour les tests conditionnels."""
    config.addinivalue_line("markers", "tensorflow: tests that require TensorFlow")
    config.addinivalue_line("markers", "torch: tests that require PyTorch")
    config.addinivalue_line("markers", "keras: tests that require Keras")
    config.addinivalue_line("markers", "jax: tests that require JAX")
    config.addinivalue_line("markers", "gpu: tests that require GPU")

# Condition pour sauter les tests nécessitant des backends spécifiques
def pytest_runtest_setup(item):
    """Skip tests if required backend is not available."""
    for marker in item.iter_markers():
        if marker.name == "tensorflow" and not is_tensorflow_available():
            pytest.skip("Test requires TensorFlow")
        elif marker.name == "torch" and not is_torch_available():
            pytest.skip("Test requires PyTorch")
        elif marker.name == "keras" and not is_keras_available():
            pytest.skip("Test requires Keras")
        elif marker.name == "jax" and not is_jax_available():
            pytest.skip("Test requires JAX")
        elif marker.name == "gpu" and not is_gpu_available():
            pytest.skip("Test requires GPU")

@pytest.fixture(scope="module")
def simple_data():
    """Fixture to generate simple synthetic data for testing."""
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0]])


@pytest.fixture(scope="module")
def random_data():
    """Fixture to generate random synthetic data for testing."""
    return np.random.rand(10, 100)


