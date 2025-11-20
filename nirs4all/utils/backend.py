"""
Utilitaires pour détecter les backends ML disponibles et permettre des tests conditionnels.
"""
import importlib.util
import warnings

TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
JAX_AVAILABLE = importlib.util.find_spec('jax') is not None
KERAS_AVAILABLE = importlib.util.find_spec('keras') is not None


def framework(framework_name):
    def decorator(func):
        func.framework = framework_name
        return func
    return decorator


def is_tensorflow_available():
    """Vérifie si TensorFlow est installé."""
    return TF_AVAILABLE


def is_torch_available():
    """Vérifie si PyTorch est installé."""
    return TORCH_AVAILABLE


def is_keras_available():
    """Vérifie si Keras 3 est installé."""
    return KERAS_AVAILABLE


def is_jax_available():
    """Vérifie si JAX est installé."""
    return JAX_AVAILABLE


def check_backend_available(backend_name: str):
    """
    Vérifie si un backend est disponible et lève une erreur sinon.

    Args:
        backend_name: Nom du backend ('tensorflow', 'torch', 'jax').

    Raises:
        ImportError: Si le backend n'est pas installé.
    """
    if backend_name == 'tensorflow' and not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. Please install it with `pip install nirs4all[tensorflow]` "
            "or `pip install nirs4all[gpu]`."
        )
    elif backend_name == 'torch' and not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install it with `pip install nirs4all[torch]`."
        )
    elif backend_name == 'jax' and not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install nirs4all[jax]`."
        )


def is_gpu_available():
    """
    Vérifie si un GPU est disponible pour au moins un des frameworks installés.
    """
    # Vérifier la disponibilité de GPU pour PyTorch
    if TORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except Exception:
            pass

    # Vérifier la disponibilité de GPU pour TensorFlow
    if TF_AVAILABLE:
        try:
            import tensorflow as tf
            if len(tf.config.list_physical_devices('GPU')) > 0:
                return True
        except Exception:
            pass

    # Vérifier la disponibilité de GPU pour JAX
    if JAX_AVAILABLE:
        try:
            import jax
            if jax.default_backend() == 'gpu':
                return True
        except Exception:
            pass

    # Aucun backend GPU disponible
    return False
