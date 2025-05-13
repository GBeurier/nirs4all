"""
Utilitaires pour détecter les backends ML disponibles et permettre des tests conditionnels.
"""

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
        import torch
        return torch.cuda.is_available()

    # Vérifier la disponibilité de GPU pour TensorFlow
    if is_tensorflow_available():
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
        
    # Vérifier la disponibilité de GPU pour JAX
    if is_jax_available():
        import jax
        return jax.default_backend() == 'gpu'
    
    # Aucun backend GPU disponible
    return False