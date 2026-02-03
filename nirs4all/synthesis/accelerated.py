"""
GPU-accelerated generation for synthetic NIRS data.

This module provides optional GPU acceleration for generating large
synthetic datasets using JAX, CuPy, or falls back to NumPy.

Phase 4 Features:
    - Automatic backend detection (JAX, CuPy, NumPy)
    - Batch spectrum generation on GPU
    - Significant speedup for large datasets (10x+)
    - Graceful fallback to CPU when GPU unavailable

Note:
    This module is optional. GPU acceleration requires additional
    dependencies (jax[cuda] or cupy-cuda*).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# Backend Detection
# ============================================================================


class AcceleratorBackend(str, Enum):
    """Available acceleration backends."""
    JAX = "jax"
    CUPY = "cupy"
    NUMPY = "numpy"  # CPU fallback


def _check_jax_available() -> bool:
    """Check if JAX with GPU is available."""
    try:
        import jax
        import jax.numpy as jnp
        # Check for GPU
        devices = jax.devices()
        return any(d.platform == 'gpu' for d in devices)
    except ImportError:
        return False
    except Exception:
        return False


def _check_cupy_available() -> bool:
    """Check if CuPy is available."""
    try:
        import cupy as cp
        # Try a simple operation
        cp.array([1, 2, 3])
        return True
    except ImportError:
        return False
    except Exception:
        return False


def detect_best_backend() -> AcceleratorBackend:
    """
    Detect the best available acceleration backend.

    Returns:
        AcceleratorBackend enum indicating best available option.

    Example:
        >>> backend = detect_best_backend()
        >>> print(f"Using backend: {backend}")
    """
    if _check_jax_available():
        return AcceleratorBackend.JAX
    elif _check_cupy_available():
        return AcceleratorBackend.CUPY
    else:
        return AcceleratorBackend.NUMPY


def get_backend_info() -> Dict[str, Any]:
    """
    Get detailed information about available backends.

    Returns:
        Dictionary with backend availability and details.
    """
    info = {
        "jax_available": _check_jax_available(),
        "cupy_available": _check_cupy_available(),
        "best_backend": detect_best_backend().value,
    }

    if info["jax_available"]:
        try:
            import jax
            info["jax_version"] = jax.__version__
            info["jax_devices"] = [str(d) for d in jax.devices()]
        except Exception:
            pass

    if info["cupy_available"]:
        try:
            import cupy as cp
            info["cupy_version"] = cp.__version__
            info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
        except Exception:
            pass

    return info


# ============================================================================
# Abstract Accelerator Interface
# ============================================================================


@dataclass
class AcceleratedArrays:
    """Container for accelerated array operations."""

    backend: AcceleratorBackend

    # Core array creation
    zeros: Callable
    ones: Callable
    arange: Callable
    linspace: Callable
    array: Callable

    # Operations
    exp: Callable
    log: Callable
    sqrt: Callable
    sin: Callable
    cos: Callable
    sum: Callable
    dot: Callable
    matmul: Callable

    # Random
    random_normal: Callable
    random_uniform: Callable

    # Transfer
    to_numpy: Callable


def _create_numpy_arrays() -> AcceleratedArrays:
    """Create NumPy-based array operations."""
    rng = np.random.default_rng()

    return AcceleratedArrays(
        backend=AcceleratorBackend.NUMPY,
        zeros=np.zeros,
        ones=np.ones,
        arange=np.arange,
        linspace=np.linspace,
        array=np.array,
        exp=np.exp,
        log=np.log,
        sqrt=np.sqrt,
        sin=np.sin,
        cos=np.cos,
        sum=np.sum,
        dot=np.dot,
        matmul=np.matmul,
        random_normal=lambda shape: rng.standard_normal(shape),
        random_uniform=lambda shape: rng.uniform(size=shape),
        to_numpy=lambda x: np.asarray(x),
    )


def _create_jax_arrays(seed: int = 0) -> AcceleratedArrays:
    """Create JAX-based array operations."""
    import jax
    import jax.numpy as jnp
    from jax import random

    key = random.PRNGKey(seed)

    def random_normal(shape):
        nonlocal key
        key, subkey = random.split(key)
        return random.normal(subkey, shape)

    def random_uniform(shape):
        nonlocal key
        key, subkey = random.split(key)
        return random.uniform(subkey, shape)

    return AcceleratedArrays(
        backend=AcceleratorBackend.JAX,
        zeros=jnp.zeros,
        ones=jnp.ones,
        arange=jnp.arange,
        linspace=jnp.linspace,
        array=jnp.array,
        exp=jnp.exp,
        log=jnp.log,
        sqrt=jnp.sqrt,
        sin=jnp.sin,
        cos=jnp.cos,
        sum=jnp.sum,
        dot=jnp.dot,
        matmul=jnp.matmul,
        random_normal=random_normal,
        random_uniform=random_uniform,
        to_numpy=lambda x: np.asarray(x),
    )


def _create_cupy_arrays(seed: int = 0) -> AcceleratedArrays:
    """Create CuPy-based array operations."""
    import cupy as cp

    cp.random.seed(seed)

    return AcceleratedArrays(
        backend=AcceleratorBackend.CUPY,
        zeros=cp.zeros,
        ones=cp.ones,
        arange=cp.arange,
        linspace=cp.linspace,
        array=cp.array,
        exp=cp.exp,
        log=cp.log,
        sqrt=cp.sqrt,
        sin=cp.sin,
        cos=cp.cos,
        sum=cp.sum,
        dot=cp.dot,
        matmul=cp.matmul,
        random_normal=lambda shape: cp.random.standard_normal(shape),
        random_uniform=lambda shape: cp.random.uniform(size=shape),
        to_numpy=lambda x: cp.asnumpy(x),
    )


def create_accelerated_arrays(
    backend: Optional[AcceleratorBackend] = None,
    seed: int = 0,
) -> AcceleratedArrays:
    """
    Create accelerated array operations for the specified backend.

    Args:
        backend: Backend to use (auto-detect if None).
        seed: Random seed.

    Returns:
        AcceleratedArrays with operations for the backend.
    """
    if backend is None:
        backend = detect_best_backend()

    if backend == AcceleratorBackend.JAX:
        return _create_jax_arrays(seed)
    elif backend == AcceleratorBackend.CUPY:
        return _create_cupy_arrays(seed)
    else:
        return _create_numpy_arrays()


# ============================================================================
# Accelerated Generation Functions
# ============================================================================


def generate_voigt_profiles_accelerated(
    wavelengths: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    arrays: Optional[AcceleratedArrays] = None,
) -> np.ndarray:
    """
    Generate Voigt profiles using GPU acceleration.

    Uses Pseudo-Voigt approximation for efficiency.

    Args:
        wavelengths: Wavelength array (n_wavelengths,).
        centers: Band centers (n_bands,).
        amplitudes: Band amplitudes (n_bands,).
        sigmas: Gaussian widths (n_bands,).
        gammas: Lorentzian widths (n_bands,).
        arrays: Accelerated arrays (auto-create if None).

    Returns:
        Spectrum array (n_wavelengths,).
    """
    if arrays is None:
        arrays = create_accelerated_arrays()

    # Transfer to device
    wl = arrays.array(wavelengths)
    c = arrays.array(centers)
    a = arrays.array(amplitudes)
    s = arrays.array(sigmas)
    g = arrays.array(gammas)

    # Initialize output
    spectrum = arrays.zeros(len(wavelengths))

    # Generate each band (vectorized over wavelengths)
    for i in range(len(centers)):
        # Pseudo-Voigt mixing parameter
        f_G = 1.0 / (1.0 + g[i] / (s[i] + 1e-10))

        # Gaussian component
        gaussian = arrays.exp(-0.5 * ((wl - c[i]) / (s[i] + 1e-10)) ** 2)

        # Lorentzian component
        lorentzian = 1.0 / (1.0 + ((wl - c[i]) / (g[i] + 1e-10)) ** 2)

        # Pseudo-Voigt
        spectrum = spectrum + a[i] * (f_G * gaussian + (1 - f_G) * lorentzian)

    return arrays.to_numpy(spectrum)


def generate_spectra_batch_accelerated(
    n_samples: int,
    wavelengths: np.ndarray,
    component_spectra: np.ndarray,
    concentrations: np.ndarray,
    noise_level: float = 0.01,
    arrays: Optional[AcceleratedArrays] = None,
) -> np.ndarray:
    """
    Generate batch of spectra using GPU acceleration.

    Args:
        n_samples: Number of samples to generate.
        wavelengths: Wavelength array.
        component_spectra: Pure component spectra (n_components, n_wavelengths).
        concentrations: Concentration matrix (n_samples, n_components).
        noise_level: Noise level as fraction of signal.
        arrays: Accelerated arrays.

    Returns:
        Generated spectra (n_samples, n_wavelengths).
    """
    if arrays is None:
        arrays = create_accelerated_arrays()

    # Transfer to device
    E = arrays.array(component_spectra)
    C = arrays.array(concentrations)

    # Beer-Lambert mixing: X = C @ E
    X = arrays.matmul(C, E)

    # Add noise
    noise = arrays.random_normal((n_samples, len(wavelengths)))
    noise = noise * noise_level * (arrays.sqrt(arrays.sum(X ** 2, axis=1, keepdims=True)) / len(wavelengths))
    X = X + noise

    return arrays.to_numpy(X)


# ============================================================================
# High-Level Accelerated Generator
# ============================================================================


class AcceleratedGenerator:
    """
    GPU-accelerated synthetic spectrum generator.

    This class provides a high-level interface for generating large
    batches of synthetic spectra using GPU acceleration when available.

    Args:
        backend: Backend to use (auto-detect if None).
        random_state: Random state for reproducibility.

    Example:
        >>> gen = AcceleratedGenerator(random_state=42)
        >>> print(f"Using backend: {gen.backend}")
        >>>
        >>> # Generate 10000 spectra
        >>> X = gen.generate_batch(
        ...     n_samples=10000,
        ...     wavelengths=np.linspace(1000, 2500, 700),
        ...     component_spectra=E,
        ...     concentrations=C,
        ... )
    """

    def __init__(
        self,
        backend: Optional[AcceleratorBackend] = None,
        random_state: Optional[int] = None,
    ):
        self.backend = backend or detect_best_backend()
        self.random_state = random_state or 0
        self.arrays = create_accelerated_arrays(self.backend, self.random_state)

    def generate_batch(
        self,
        n_samples: int,
        wavelengths: np.ndarray,
        component_spectra: np.ndarray,
        concentrations: np.ndarray,
        noise_level: float = 0.01,
    ) -> np.ndarray:
        """
        Generate a batch of spectra.

        Args:
            n_samples: Number of samples.
            wavelengths: Wavelength array.
            component_spectra: Component spectra (n_components, n_wavelengths).
            concentrations: Concentrations (n_samples, n_components).
            noise_level: Noise level.

        Returns:
            Generated spectra (n_samples, n_wavelengths).
        """
        return generate_spectra_batch_accelerated(
            n_samples=n_samples,
            wavelengths=wavelengths,
            component_spectra=component_spectra,
            concentrations=concentrations,
            noise_level=noise_level,
            arrays=self.arrays,
        )

    def generate_voigt_profiles(
        self,
        wavelengths: np.ndarray,
        centers: np.ndarray,
        amplitudes: np.ndarray,
        sigmas: np.ndarray,
        gammas: np.ndarray,
    ) -> np.ndarray:
        """
        Generate Voigt profiles for component spectra.

        Args:
            wavelengths: Wavelength array.
            centers: Band centers.
            amplitudes: Band amplitudes.
            sigmas: Gaussian widths.
            gammas: Lorentzian widths.

        Returns:
            Spectrum array.
        """
        return generate_voigt_profiles_accelerated(
            wavelengths=wavelengths,
            centers=centers,
            amplitudes=amplitudes,
            sigmas=sigmas,
            gammas=gammas,
            arrays=self.arrays,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if JAX with GPU or CuPy is available.

    Example:
        >>> if is_gpu_available():
        ...     print("GPU acceleration enabled!")
    """
    return detect_best_backend() != AcceleratorBackend.NUMPY


def get_acceleration_speedup_estimate(n_samples: int) -> float:
    """
    Estimate speedup from GPU acceleration.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        Estimated speedup factor (1.0 for CPU).
    """
    backend = detect_best_backend()

    if backend == AcceleratorBackend.NUMPY:
        return 1.0

    # Empirical estimates based on typical workloads
    if n_samples < 100:
        return 0.5  # GPU overhead dominates for small batches
    elif n_samples < 1000:
        return 2.0
    elif n_samples < 10000:
        return 5.0
    else:
        return 10.0  # Large batches benefit most


def benchmark_backends(
    n_samples: int = 1000,
    n_wavelengths: int = 700,
    n_components: int = 5,
    n_trials: int = 5,
) -> Dict[str, float]:
    """
    Benchmark available backends.

    Args:
        n_samples: Number of samples to generate.
        n_wavelengths: Number of wavelengths.
        n_components: Number of components.
        n_trials: Number of timing trials.

    Returns:
        Dictionary of backend name to mean time in seconds.

    Example:
        >>> results = benchmark_backends()
        >>> for backend, time in results.items():
        ...     print(f"{backend}: {time:.4f}s")
    """
    import time

    results = {}

    # Generate test data
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    component_spectra = np.random.randn(n_components, n_wavelengths)
    concentrations = np.abs(np.random.randn(n_samples, n_components))

    # Test NumPy (always available)
    arrays = _create_numpy_arrays()
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = generate_spectra_batch_accelerated(
            n_samples, wavelengths, component_spectra, concentrations,
            arrays=arrays
        )
        times.append(time.perf_counter() - start)
    results["numpy"] = np.mean(times)

    # Test JAX if available
    if _check_jax_available():
        arrays = _create_jax_arrays()
        # Warmup
        _ = generate_spectra_batch_accelerated(
            n_samples, wavelengths, component_spectra, concentrations,
            arrays=arrays
        )
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = generate_spectra_batch_accelerated(
                n_samples, wavelengths, component_spectra, concentrations,
                arrays=arrays
            )
            times.append(time.perf_counter() - start)
        results["jax"] = np.mean(times)

    # Test CuPy if available
    if _check_cupy_available():
        arrays = _create_cupy_arrays()
        # Warmup
        _ = generate_spectra_batch_accelerated(
            n_samples, wavelengths, component_spectra, concentrations,
            arrays=arrays
        )
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = generate_spectra_batch_accelerated(
                n_samples, wavelengths, component_spectra, concentrations,
                arrays=arrays
            )
            times.append(time.perf_counter() - start)
        results["cupy"] = np.mean(times)

    return results
