"""Reproducibility utilities."""
from typing import Optional
import os
import random
import numpy as np

def init_global_random_state(seed: Optional[int] = None):
    """Initialize global random state for reproducibility.

    Sets random seeds for numpy, Python's random module, TensorFlow, and sklearn
    to ensure reproducible results across runs.

    Args:
        seed: Random seed value. If None, uses default seed of 42 for TensorFlow.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed if seed is not None else 42)
    except ImportError:
        pass

    try:
        from sklearn.utils import check_random_state
        _ = check_random_state(seed)
    except ImportError:
        pass
