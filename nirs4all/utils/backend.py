"""
Backend detection and lazy loading utilities for ML frameworks.

This module provides:
- Lazy detection of framework availability (no imports at module load)
- Cached availability checks for performance
- Clean error messages guiding users to install missing dependencies
- GPU availability detection across all frameworks

Usage:
    from nirs4all.utils.backend import require_backend, is_available, get_backend_info

    # Check availability without importing
    if is_available('tensorflow'):
        # Safe to import tensorflow here
        import tensorflow as tf

    # Require a backend (raises helpful ImportError if missing)
    require_backend('torch')
    import torch
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Dict, Any, Callable, TypeVar


# =============================================================================
# Backend Registry and Detection
# =============================================================================

# Installation instructions for each backend
_INSTALL_INSTRUCTIONS: Dict[str, str] = {
    'tensorflow': 'pip install nirs4all[tensorflow] or pip install nirs4all[gpu]',
    'torch': 'pip install nirs4all[torch]',
    'pytorch': 'pip install nirs4all[torch]',  # Alias for torch
    'jax': 'pip install nirs4all[jax]',
    'keras': 'pip install nirs4all[keras]',
    'autogluon': 'pip install autogluon',
    'xgboost': 'pip install xgboost',
    'lightgbm': 'pip install lightgbm',
    'catboost': 'pip install catboost',
    'optuna': 'pip install optuna',
    'shap': 'pip install shap',
    'ikpls': 'pip install ikpls',
}

# Package name mapping (when import name differs from pip name)
_PACKAGE_MAPPING: Dict[str, str] = {
    'tensorflow': 'tensorflow',
    'torch': 'torch',
    'pytorch': 'torch',  # 'pytorch' is an alias for 'torch' package
    'jax': 'jax',
    'keras': 'keras',
    'autogluon': 'autogluon',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'optuna': 'optuna',
    'shap': 'shap',
    'ikpls': 'ikpls',
}


# =============================================================================
# Cached Availability Detection (No Imports!)
# =============================================================================

# Cache for availability checks - None means not yet checked
_availability_cache: Dict[str, Optional[bool]] = {}


def _check_spec_available(module_name: str) -> bool:
    """Check if a module is available via importlib.util.find_spec.

    This does NOT import the module, making it very fast.

    Args:
        module_name: The module name to check.

    Returns:
        True if the module is available, False otherwise.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def is_available(backend: str) -> bool:
    """Check if a backend is available without importing it.

    Results are cached for performance.

    Args:
        backend: Backend name ('tensorflow', 'torch', 'jax', etc.)

    Returns:
        True if the backend is installed and can be imported.

    Example:
        >>> if is_available('tensorflow'):
        ...     import tensorflow as tf
    """
    backend = backend.lower()

    if backend not in _availability_cache:
        package = _PACKAGE_MAPPING.get(backend, backend)
        _availability_cache[backend] = _check_spec_available(package)

    return _availability_cache[backend]


def clear_availability_cache():
    """Clear the availability cache (useful for testing)."""
    _availability_cache.clear()
    _gpu_cache.clear()


# =============================================================================
# Legacy Compatibility - Module-level Constants (Lazy)
# =============================================================================

class _LazyAvailability:
    """Lazy availability check that only runs when accessed.

    This class allows module-level constants like TF_AVAILABLE to be
    defined without triggering imports at module load time.
    """

    def __init__(self, backend: str):
        self._backend = backend
        self._value: Optional[bool] = None

    def __bool__(self) -> bool:
        if self._value is None:
            self._value = is_available(self._backend)
        return self._value

    def __repr__(self) -> str:
        return str(bool(self))

    def __eq__(self, other) -> bool:
        return bool(self) == other

    def __hash__(self) -> int:
        return hash(bool(self))


# These are now lazy - no import cost at module load
TF_AVAILABLE = _LazyAvailability('tensorflow')
TORCH_AVAILABLE = _LazyAvailability('torch')
JAX_AVAILABLE = _LazyAvailability('jax')
KERAS_AVAILABLE = _LazyAvailability('keras')
IKPLS_AVAILABLE = _LazyAvailability('ikpls')


# =============================================================================
# Requirement Enforcement
# =============================================================================

class BackendNotAvailableError(ImportError):
    """Raised when a required backend is not installed.

    Provides helpful error messages with installation instructions.
    """

    def __init__(self, backend: str, feature: Optional[str] = None):
        self.backend = backend
        self.feature = feature

        install_cmd = _INSTALL_INSTRUCTIONS.get(backend, f'pip install {backend}')

        if feature:
            message = (
                f"The '{feature}' feature requires {backend}, which is not installed.\n"
                f"Install it with: {install_cmd}"
            )
        else:
            message = (
                f"{backend} is not installed.\n"
                f"Install it with: {install_cmd}"
            )

        super().__init__(message)


def require_backend(backend: str, feature: Optional[str] = None) -> None:
    """Require a backend to be available, raising a helpful error if not.

    Args:
        backend: Backend name ('tensorflow', 'torch', 'jax', etc.)
        feature: Optional feature name for better error messages.

    Raises:
        BackendNotAvailableError: If the backend is not installed.

    Example:
        >>> require_backend('tensorflow', feature='NICON neural network')
        >>> import tensorflow as tf  # Safe after require_backend
    """
    if not is_available(backend):
        raise BackendNotAvailableError(backend, feature)


def check_backend_available(backend_name: str) -> None:
    """Check if a backend is available, raising ImportError if not.

    This is a legacy compatibility wrapper for require_backend.
    Use require_backend() for new code.

    Args:
        backend_name: Name of the backend ('tensorflow', 'torch', 'jax').

    Raises:
        BackendNotAvailableError: If the backend is not installed.
    """
    require_backend(backend_name)


# =============================================================================
# Framework Decorator
# =============================================================================

F = TypeVar('F', bound=Callable)


def framework(framework_name: str) -> Callable[[F], F]:
    """Decorator to mark a function/class with its framework.

    This enables automatic framework detection in the model factory.

    Args:
        framework_name: Name of the framework ('tensorflow', 'pytorch', 'jax')

    Returns:
        Decorator function that adds framework attribute.

    Example:
        >>> @framework('tensorflow')
        ... def build_cnn(input_shape, params):
        ...     import tensorflow as tf
        ...     # ... build model
    """
    def decorator(func: F) -> F:
        func.framework = framework_name
        return func
    return decorator


# =============================================================================
# GPU Detection (Lazy)
# =============================================================================

_gpu_cache: Dict[str, Optional[bool]] = {}


def is_gpu_available(backend: Optional[str] = None) -> bool:
    """Check if GPU is available for the specified backend or any backend.

    Results are cached for performance. The first call for each backend
    will import the framework to check GPU availability.

    Args:
        backend: Specific backend to check ('tensorflow', 'torch', 'jax'),
                 or None to check all available backends.

    Returns:
        True if GPU is available for the specified backend(s).

    Example:
        >>> if is_gpu_available('torch'):
        ...     device = 'cuda'
        ... else:
        ...     device = 'cpu'
    """
    if backend:
        return _check_gpu_for_backend(backend)

    # Check all available backends
    for be in ['torch', 'tensorflow', 'jax']:
        if is_available(be) and _check_gpu_for_backend(be):
            return True

    return False


def _check_gpu_for_backend(backend: str) -> bool:
    """Check GPU availability for a specific backend.

    Args:
        backend: Backend name to check.

    Returns:
        True if GPU is available for the backend.
    """
    # Normalize 'pytorch' alias to 'torch'
    if backend == 'pytorch':
        backend = 'torch'

    if backend in _gpu_cache:
        return _gpu_cache[backend]

    result = False

    try:
        if backend == 'torch' and is_available('torch'):
            import torch
            result = torch.cuda.is_available()

        elif backend == 'tensorflow' and is_available('tensorflow'):
            import tensorflow as tf
            result = len(tf.config.list_physical_devices('GPU')) > 0

        elif backend == 'jax' and is_available('jax'):
            import jax
            result = jax.default_backend() == 'gpu'

    except Exception:
        result = False

    _gpu_cache[backend] = result
    return result


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information for all available backends.

    Returns:
        Dictionary with GPU info per backend, including device counts
        and names where available.

    Example:
        >>> info = get_gpu_info()
        >>> if info.get('torch', {}).get('available'):
        ...     print(f"GPU: {info['torch']['device_name']}")
    """
    info = {}

    if is_available('torch'):
        try:
            import torch
            info['torch'] = {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as e:
            info['torch'] = {'available': False, 'error': str(e)}

    if is_available('tensorflow'):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            info['tensorflow'] = {
                'available': len(gpus) > 0,
                'device_count': len(gpus),
                'devices': [g.name for g in gpus],
            }
        except Exception as e:
            info['tensorflow'] = {'available': False, 'error': str(e)}

    if is_available('jax'):
        try:
            import jax
            info['jax'] = {
                'available': jax.default_backend() == 'gpu',
                'backend': jax.default_backend(),
                'device_count': jax.device_count(),
            }
        except Exception as e:
            info['jax'] = {'available': False, 'error': str(e)}

    return info


# =============================================================================
# Backend Info Summary
# =============================================================================

def get_backend_info() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive info about all backends.

    Returns:
        Dictionary with availability and GPU info for each backend.

    Example:
        >>> info = get_backend_info()
        >>> for name, details in info.items():
        ...     status = "✓" if details['available'] else "✗"
        ...     print(f"{status} {name}")
    """
    backends = ['tensorflow', 'torch', 'jax', 'keras', 'autogluon',
                'xgboost', 'lightgbm', 'catboost', 'optuna', 'shap']

    info = {}
    for be in backends:
        info[be] = {
            'available': is_available(be),
            'install': _INSTALL_INSTRUCTIONS.get(be),
        }

        # Add GPU info for relevant backends
        if be in ['tensorflow', 'torch', 'jax'] and is_available(be):
            info[be]['gpu'] = _check_gpu_for_backend(be)

    return info


def print_backend_status():
    """Print a formatted summary of backend availability.

    Displays a table showing which backends are installed and
    whether GPU support is available for deep learning frameworks.
    """
    info = get_backend_info()

    print("\n" + "=" * 50)
    print("NIRS4ALL Backend Status")
    print("=" * 50)

    for backend, details in info.items():
        status = "✓" if details['available'] else "✗"
        gpu_status = ""
        if 'gpu' in details:
            gpu_status = f" (GPU: {'✓' if details['gpu'] else '✗'})"

        print(f"  {status} {backend}{gpu_status}")

    print("=" * 50 + "\n")


# =============================================================================
# Lazy Import Helpers
# =============================================================================

def lazy_import(module_name: str, backend: Optional[str] = None):
    """Create a lazy import that only loads the module when accessed.

    Args:
        module_name: Full module path to import.
        backend: Optional backend to check before importing.

    Returns:
        A proxy object that imports the module on first access.

    Example:
        >>> tf = lazy_import('tensorflow', backend='tensorflow')
        >>> # tensorflow not imported yet
        >>> model = tf.keras.Model()  # Now tensorflow is imported
    """
    class LazyModule:
        _module = None

        def __getattr__(self, name):
            if LazyModule._module is None:
                if backend and not is_available(backend):
                    raise BackendNotAvailableError(backend)
                import importlib
                LazyModule._module = importlib.import_module(module_name)
            return getattr(LazyModule._module, name)

    return LazyModule()


# =============================================================================
# Compatibility Functions
# =============================================================================

def is_tensorflow_available() -> bool:
    """Check if TensorFlow is installed.

    Returns:
        True if TensorFlow is available.
    """
    return is_available('tensorflow')


def is_torch_available() -> bool:
    """Check if PyTorch is installed.

    Returns:
        True if PyTorch is available.
    """
    return is_available('torch')


def is_keras_available() -> bool:
    """Check if Keras is installed.

    Returns:
        True if Keras is available.
    """
    return is_available('keras')


def is_jax_available() -> bool:
    """Check if JAX is installed.

    Returns:
        True if JAX is available.
    """
    return is_available('jax')


def is_ikpls_available() -> bool:
    """Check if ikpls is installed.

    Returns:
        True if ikpls is available.
    """
    return is_available('ikpls')
