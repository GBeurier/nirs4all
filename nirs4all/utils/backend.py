"""
Utilitaires pour détecter les backends ML disponibles et permettre des tests conditionnels.
"""
import importlib.util

TF_AVAILABLE = importlib.util.find_spec('tensorflow') is not None
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def framework(framework_name):
    def decorator(func):
        func.framework = framework_name
        return func
    return decorator


def is_tensorflow_available():
    """Vérifie si TensorFlow est installé."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


def is_torch_available():
    """Vérifie si PyTorch est installé."""
    try:
        import torch
        return True
    except ImportError:
        return False


def is_keras_available():
    """Vérifie si Keras 3 est installé."""
    try:
        import keras
        return True
    except ImportError:
        return False


def is_jax_available():
    """Vérifie si JAX est installé."""
    try:
        import jax
        return True
    except ImportError:
        return False


def is_gpu_available():
    """
    Vérifie si un GPU est disponible pour au moins un des frameworks installés.
    """
    # Vérifier la disponibilité de GPU pour PyTorch
    if is_torch_available():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            pass

    # Vérifier la disponibilité de GPU pour TensorFlow
    if is_tensorflow_available():
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except Exception:
            pass

    # Vérifier la disponibilité de GPU pour JAX
    if is_jax_available():
        try:
            import jax
            return jax.default_backend() == 'gpu'
        except Exception:
            pass

    # Aucun backend GPU disponible
    return False
