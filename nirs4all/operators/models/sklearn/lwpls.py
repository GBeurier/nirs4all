"""Locally-Weighted Partial Least Squares (LWPLS) model operator.

This module provides a sklearn-compatible LWPLS implementation for nirs4all.
The core algorithm is adapted from the original implementation by Hiromasa Kaneko
(https://github.com/hkaneko1985/lwpls), licensed under MIT License.

LWPLS builds just-in-time local PLS models near each query sample, which is
useful when dealing with drift, local nonlinearity, or heterogeneous data.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
- Kim, S., Kano, M., Nakagawa, H., & Hasebe, S. (2011).
  Estimation of active pharmaceutical ingredient content using
  locally weighted partial least squares and statistical wavelength
  selection. International Journal of Pharmaceutics, 421(2), 269-274.
- https://datachemeng.com/locallyweightedpartialleastsquares/

License
-------
Original lwpls.py by Hiromasa Kaneko is MIT licensed.
"""

from __future__ import annotations

from functools import partial
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


def _check_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _lwpls_predict(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    max_component_number: int,
    lambda_in_similarity: float,
) -> NDArray[np.floating]:
    """Core LWPLS prediction algorithm (memory-optimized).

    Builds a locally-weighted PLS model for each test sample using
    Gaussian kernel weights based on Euclidean distance.

    This implementation avoids creating O(n²) diagonal matrices by using
    element-wise weighted operations instead of matrix multiplications
    with diagonal similarity matrices.

    Parameters
    ----------
    x_train : ndarray of shape (n_train, n_features)
        Autoscaled training X data.
    y_train : ndarray of shape (n_train,) or (n_train, 1)
        Autoscaled training y data.
    x_test : ndarray of shape (n_test, n_features)
        Autoscaled test X data.
    max_component_number : int
        Maximum number of PLS components to extract.
    lambda_in_similarity : float
        Parameter controlling the kernel width. Smaller values give
        more localized models; larger values approach global PLS.

    Returns
    -------
    estimated_y_test : ndarray of shape (n_test, max_component_number)
        Predictions for each number of components (column i contains
        predictions using i+1 components).

    Notes
    -----
    The algorithm:
    1. For each test sample, compute distances to all training samples
    2. Convert distances to similarities using Gaussian kernel
    3. Compute weighted mean of X and Y
    4. Build weighted PLS components iteratively
    5. Predict Y by accumulating component contributions

    Memory optimization: Instead of creating (n_train, n_train) diagonal
    matrices for weighted operations, we use element-wise multiplication:
    - X.T @ diag(w) @ y  →  X.T @ (w[:, None] * y)  [O(n*p) vs O(n²)]
    - t.T @ diag(w) @ t  →  sum(w * t²)  [O(n) vs O(n²)]
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_train = np.reshape(y_train, (len(y_train), 1))
    x_test = np.asarray(x_test, dtype=np.float64)

    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    n_features = x_train.shape[1]

    estimated_y_test = np.zeros((n_test, max_component_number))

    # Precompute distance matrix for efficiency
    # Shape: (n_train, n_test) - acceptable memory usage
    distance_matrix = cdist(x_train, x_test, metric='euclidean')

    for test_idx in range(n_test):
        query_x_test = x_test[test_idx:test_idx + 1, :]

        # Get distances and compute similarities
        distance = distance_matrix[:, test_idx]
        distance_std = distance.std(ddof=1) if distance.std(ddof=1) > 0 else 1.0

        # Gaussian kernel weights - 1D array, NOT a diagonal matrix
        w = np.exp(-distance / distance_std / lambda_in_similarity)
        w_sum = w.sum()

        if w_sum < 1e-10:
            # All samples too far away; use uniform weights
            w = np.ones(n_train) / n_train
            w_sum = 1.0

        # Weighted means using element-wise operations (O(n) not O(n²))
        y_w = np.dot(w, y_train[:, 0]) / w_sum
        x_w = (w @ x_train) / w_sum  # shape: (n_features,)

        # Center data
        centered_y = y_train[:, 0] - y_w  # 1D array (n_train,)
        centered_x = x_train - x_w  # shape: (n_train, n_features)
        centered_query = query_x_test[0] - x_w  # 1D array (n_features,)

        # Initialize prediction with weighted mean
        estimated_y_test[test_idx, :] = y_w

        # Build PLS components
        for comp_num in range(max_component_number):
            # Weighted loading direction: X.T @ diag(w) @ y = X.T @ (w * y)
            # Equivalent to sum of (x_i * w_i * y_i) for each feature
            numerator = centered_x.T @ (w * centered_y)  # O(n*p)
            norm_val = np.linalg.norm(numerator)

            if norm_val < 1e-10:
                # Degenerate case - no more variance to explain
                break

            w_a = numerator / norm_val  # Loading weight vector

            # Scores: t = X @ w_a
            t_a = centered_x @ w_a  # shape: (n_train,)

            # Weighted denominator: t.T @ diag(w) @ t = sum(w * t²)
            denom = np.dot(w * t_a, t_a)  # O(n)
            if denom < 1e-10:
                break

            # Loadings: p = (X.T @ diag(w) @ t) / denom
            p_a = (centered_x.T @ (w * t_a)) / denom  # O(n*p)
            # q = (y.T @ diag(w) @ t) / denom
            q_a = np.dot(w * centered_y, t_a) / denom  # O(n)

            # Query score
            t_q_a = np.dot(centered_query, w_a)  # scalar

            # Accumulate prediction for this and all subsequent components
            estimated_y_test[test_idx, comp_num:] += t_q_a * q_a

            # Deflate for next component
            if comp_num < max_component_number - 1:
                centered_x = centered_x - np.outer(t_a, p_a)  # O(n*p)
                centered_y = centered_y - t_a * q_a  # O(n)
                centered_query = centered_query - t_q_a * p_a  # O(p)

    return estimated_y_test


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_lwpls_functions():
    """Lazy import and create JAX LWPLS functions.

    Returns the JAX-accelerated prediction function. This is done lazily
    to avoid importing JAX unless needed.

    Returns
    -------
    lwpls_predict_jax : callable
        JAX-accelerated LWPLS prediction function with batching support.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    def _lwpls_single_query(
        x_train: jax.Array,
        y_train: jax.Array,
        query_x: jax.Array,
        max_components: int,
        lambda_sim: float,
    ) -> jax.Array:
        """LWPLS prediction for a single query sample.

        Parameters
        ----------
        x_train : jax.Array of shape (n_train, n_features)
            Training X data.
        y_train : jax.Array of shape (n_train, 1)
            Training y data.
        query_x : jax.Array of shape (n_features,)
            Single query sample.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : jax.Array of shape (max_components,)
            Predictions for each number of components.
        """
        n_train, n_features = x_train.shape

        # Compute Euclidean distances from query to all training samples
        diff = x_train - query_x[jnp.newaxis, :]
        distances = jnp.sqrt(jnp.sum(diff ** 2, axis=1))

        # Compute distance std (with Bessel correction, matching NumPy)
        dist_mean = jnp.mean(distances)
        dist_std = jnp.sqrt(jnp.sum((distances - dist_mean) ** 2) / (n_train - 1))
        dist_std = jnp.maximum(dist_std, 1e-10)  # Avoid division by zero

        # Gaussian kernel weights
        weights = jnp.exp(-distances / dist_std / lambda_sim)
        weight_sum = jnp.sum(weights)

        # Handle degenerate case
        weights = lax.cond(
            weight_sum < 1e-10,
            lambda w: jnp.ones(n_train) / n_train,
            lambda w: w,
            weights,
        )
        weight_sum = lax.cond(
            weight_sum < 1e-10,
            lambda _: 1.0,
            lambda ws: ws,
            weight_sum,
        )

        # Weighted means
        y_w = jnp.sum(y_train[:, 0] * weights) / weight_sum
        x_w = jnp.sum(x_train * weights[:, jnp.newaxis], axis=0) / weight_sum

        # Center data
        centered_x = x_train - x_w[jnp.newaxis, :]
        centered_y = y_train - y_w
        centered_query = query_x - x_w

        # Initialize predictions with weighted mean
        predictions = jnp.full(max_components, y_w)

        # Build PLS components using lax.fori_loop for JIT compatibility
        def component_step(comp_idx, carry):
            centered_x, centered_y, centered_query, predictions, weights = carry

            # Weighted loading direction: X^T @ W @ y
            # W is diagonal, so X^T @ W @ y = sum(x_i * w_i * y_i)
            numerator = jnp.sum(
                centered_x * (weights * centered_y[:, 0])[:, jnp.newaxis],
                axis=0,
            )
            norm_val = jnp.linalg.norm(numerator)

            # Safe normalization
            w_a = lax.cond(
                norm_val < 1e-10,
                lambda n: jnp.zeros(n_features),
                lambda n: numerator / norm_val,
                numerator,
            )

            # Scores: t = X @ w
            t_a = centered_x @ w_a  # shape: (n_train,)

            # Weighted denominator: t^T @ W @ t
            denom = jnp.sum(t_a ** 2 * weights)
            denom = jnp.maximum(denom, 1e-10)

            # Loadings
            # p = (X^T @ W @ t) / denom
            p_a = jnp.sum(centered_x * (weights * t_a)[:, jnp.newaxis], axis=0) / denom
            # q = (y^T @ W @ t) / denom
            q_a = jnp.sum(centered_y[:, 0] * weights * t_a) / denom

            # Query score
            t_q = jnp.dot(centered_query, w_a)

            # Update predictions for this and all subsequent components
            contribution = t_q * q_a
            # Add contribution to predictions[comp_idx:]
            mask = jnp.arange(max_components) >= comp_idx
            predictions = predictions + contribution * mask

            # Deflate for next component
            centered_x = centered_x - jnp.outer(t_a, p_a)
            centered_y = centered_y - (t_a * q_a)[:, jnp.newaxis]
            centered_query = centered_query - t_q * p_a

            return (centered_x, centered_y, centered_query, predictions, weights)

        # Run the component loop
        init_carry = (centered_x, centered_y, centered_query, predictions, weights)
        _, _, _, predictions, _ = lax.fori_loop(
            0, max_components, component_step, init_carry
        )

        return predictions

    # Vectorize over test samples using vmap
    _lwpls_batch = jax.vmap(
        _lwpls_single_query,
        in_axes=(None, None, 0, None, None),  # Vectorize over query samples
    )

    @partial(jax.jit, static_argnums=(3,))
    def _lwpls_predict_batch_jit(
        x_train: jax.Array,
        y_train: jax.Array,
        x_test: jax.Array,
        max_components: int,
        lambda_sim: float,
    ) -> jax.Array:
        """JIT-compiled LWPLS prediction for a batch of test samples.

        Parameters
        ----------
        x_train : jax.Array of shape (n_train, n_features)
            Training X data.
        y_train : jax.Array of shape (n_train, 1)
            Training y data.
        x_test : jax.Array of shape (batch_size, n_features)
            Batch of test samples.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : jax.Array of shape (batch_size, max_components)
            Predictions for each test sample in the batch.
        """
        return _lwpls_batch(x_train, y_train, x_test, max_components, lambda_sim)

    def lwpls_predict_jax(
        x_train: jax.Array,
        y_train: jax.Array,
        x_test: jax.Array,
        max_components: int,
        lambda_sim: float,
        batch_size: int = 64,
    ) -> jax.Array:
        """Batched LWPLS prediction to control memory usage.

        Processes test samples in batches to avoid OOM on large datasets.

        Parameters
        ----------
        x_train : jax.Array of shape (n_train, n_features)
            Training X data.
        y_train : jax.Array of shape (n_train, 1)
            Training y data.
        x_test : jax.Array of shape (n_test, n_features)
            Test X data.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.
        batch_size : int, default=64
            Number of test samples to process at once.

        Returns
        -------
        predictions : jax.Array of shape (n_test, max_components)
            Predictions for each test sample and number of components.
        """
        n_test = x_test.shape[0]

        if n_test <= batch_size:
            # Small enough to process in one go
            return _lwpls_predict_batch_jit(
                x_train, y_train, x_test, max_components, lambda_sim
            )

        # Process in batches to control memory
        results = []
        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, n_test)
            batch = x_test[start_idx:end_idx]
            batch_pred = _lwpls_predict_batch_jit(
                x_train, y_train, batch, max_components, lambda_sim
            )
            results.append(batch_pred)

        return jnp.concatenate(results, axis=0)

    return lwpls_predict_jax


# Cache the JAX function to avoid re-creating it
_JAX_LWPLS_FUNC = None


# =============================================================================
# PyTorch Backend Implementation
# =============================================================================

def _get_torch_lwpls_functions():
    """Lazy import and create PyTorch LWPLS functions.

    Returns the PyTorch-accelerated prediction function. This is done lazily
    to avoid importing PyTorch unless needed.

    Returns
    -------
    lwpls_predict_torch : callable
        PyTorch-accelerated LWPLS prediction function with batching support.
    """
    import torch

    def _lwpls_single_query_torch(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        query_x: torch.Tensor,
        max_components: int,
        lambda_sim: float,
    ) -> torch.Tensor:
        """LWPLS prediction for a single query sample using PyTorch.

        Parameters
        ----------
        x_train : torch.Tensor of shape (n_train, n_features)
            Training X data.
        y_train : torch.Tensor of shape (n_train, 1)
            Training y data.
        query_x : torch.Tensor of shape (n_features,)
            Single query sample.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : torch.Tensor of shape (max_components,)
            Predictions for each number of components.
        """
        n_train, n_features = x_train.shape
        device = x_train.device
        dtype = x_train.dtype

        # Compute Euclidean distances from query to all training samples
        diff = x_train - query_x.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=1))

        # Compute distance std (with Bessel correction, matching NumPy)
        dist_mean = torch.mean(distances)
        dist_std = torch.sqrt(torch.sum((distances - dist_mean) ** 2) / (n_train - 1))
        dist_std = torch.clamp(dist_std, min=1e-10)  # Avoid division by zero

        # Gaussian kernel weights
        weights = torch.exp(-distances / dist_std / lambda_sim)
        weight_sum = torch.sum(weights)

        # Handle degenerate case
        if weight_sum < 1e-10:
            weights = torch.ones(n_train, device=device, dtype=dtype) / n_train
            weight_sum = torch.tensor(1.0, device=device, dtype=dtype)

        # Weighted means
        y_w = torch.sum(y_train[:, 0] * weights) / weight_sum
        x_w = torch.sum(x_train * weights.unsqueeze(1), dim=0) / weight_sum

        # Center data
        centered_x = x_train - x_w.unsqueeze(0)
        centered_y = y_train - y_w
        centered_query = query_x - x_w

        # Initialize predictions with weighted mean
        predictions = torch.full((max_components,), y_w.item(), device=device, dtype=dtype)

        # Build PLS components
        for comp_idx in range(max_components):
            # Weighted loading direction: X^T @ W @ y
            numerator = torch.sum(
                centered_x * (weights * centered_y[:, 0]).unsqueeze(1),
                dim=0,
            )
            norm_val = torch.linalg.norm(numerator)

            if norm_val < 1e-10:
                break

            w_a = numerator / norm_val

            # Scores: t = X @ w
            t_a = centered_x @ w_a  # shape: (n_train,)

            # Weighted denominator: t^T @ W @ t
            denom = torch.sum(t_a ** 2 * weights)
            if denom < 1e-10:
                break

            # Loadings
            p_a = torch.sum(centered_x * (weights * t_a).unsqueeze(1), dim=0) / denom
            q_a = torch.sum(centered_y[:, 0] * weights * t_a) / denom

            # Query score
            t_q = torch.dot(centered_query, w_a)

            # Update predictions for this and all subsequent components
            contribution = t_q * q_a
            predictions[comp_idx:] = predictions[comp_idx:] + contribution

            # Deflate for next component
            if comp_idx < max_components - 1:
                centered_x = centered_x - torch.outer(t_a, p_a)
                centered_y = centered_y - (t_a * q_a).unsqueeze(1)
                centered_query = centered_query - t_q * p_a

        return predictions

    def _lwpls_batch_torch(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        max_components: int,
        lambda_sim: float,
    ) -> torch.Tensor:
        """Batched LWPLS prediction for multiple test samples.

        Parameters
        ----------
        x_train : torch.Tensor of shape (n_train, n_features)
            Training X data.
        y_train : torch.Tensor of shape (n_train, 1)
            Training y data.
        x_test : torch.Tensor of shape (batch_size, n_features)
            Batch of test samples.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : torch.Tensor of shape (batch_size, max_components)
            Predictions for each test sample in the batch.
        """
        batch_size = x_test.shape[0]
        device = x_train.device
        dtype = x_train.dtype

        results = torch.zeros((batch_size, max_components), device=device, dtype=dtype)

        for i in range(batch_size):
            results[i] = _lwpls_single_query_torch(
                x_train, y_train, x_test[i], max_components, lambda_sim
            )

        return results

    def lwpls_predict_torch(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        max_components: int,
        lambda_sim: float,
        batch_size: int = 64,
        device: str = 'auto',
    ) -> torch.Tensor:
        """Batched LWPLS prediction to control memory usage.

        Processes test samples in batches to avoid OOM on large datasets.

        Parameters
        ----------
        x_train : torch.Tensor of shape (n_train, n_features)
            Training X data.
        y_train : torch.Tensor of shape (n_train, 1)
            Training y data.
        x_test : torch.Tensor of shape (n_test, n_features)
            Test X data.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.
        batch_size : int, default=64
            Number of test samples to process at once.
        device : str, default='auto'
            Device to use ('auto', 'cpu', 'cuda', 'mps').
            'auto' will use CUDA if available, otherwise CPU.

        Returns
        -------
        predictions : torch.Tensor of shape (n_test, max_components)
            Predictions for each test sample and number of components.
        """
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        torch_device = torch.device(device)

        # Move data to device
        x_train_t = x_train.to(torch_device)
        y_train_t = y_train.to(torch_device)
        x_test_t = x_test.to(torch_device)

        n_test = x_test_t.shape[0]

        if n_test <= batch_size:
            # Small enough to process in one go
            return _lwpls_batch_torch(
                x_train_t, y_train_t, x_test_t, max_components, lambda_sim
            )

        # Process in batches to control memory
        results = []
        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, n_test)
            batch = x_test_t[start_idx:end_idx]
            batch_pred = _lwpls_batch_torch(
                x_train_t, y_train_t, batch, max_components, lambda_sim
            )
            results.append(batch_pred)

        return torch.cat(results, dim=0)

    return lwpls_predict_torch


# Cache the PyTorch function to avoid re-creating it
_TORCH_LWPLS_FUNC = None


def _lwpls_predict_jax(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    max_component_number: int,
    lambda_in_similarity: float,
    batch_size: int = 64,
) -> NDArray[np.floating]:
    """JAX-accelerated LWPLS prediction with batching.

    Same interface as _lwpls_predict but uses JAX for GPU/TPU acceleration.
    Processes test samples in batches to avoid OOM on large datasets.

    Parameters
    ----------
    x_train : ndarray of shape (n_train, n_features)
        Autoscaled training X data.
    y_train : ndarray of shape (n_train,) or (n_train, 1)
        Autoscaled training y data.
    x_test : ndarray of shape (n_test, n_features)
        Autoscaled test X data.
    max_component_number : int
        Maximum number of PLS components to extract.
    lambda_in_similarity : float
        Parameter controlling the kernel width.
    batch_size : int, default=64
        Number of test samples to process per batch.
        Reduce this if running out of memory.

    Returns
    -------
    estimated_y_test : ndarray of shape (n_test, max_component_number)
        Predictions for each number of components.
    """
    global _JAX_LWPLS_FUNC

    if _JAX_LWPLS_FUNC is None:
        _JAX_LWPLS_FUNC = _get_jax_lwpls_functions()

    import jax.numpy as jnp

    # Convert to JAX arrays
    x_train_jax = jnp.asarray(x_train, dtype=jnp.float64)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float64)
    if y_train_jax.ndim == 1:
        y_train_jax = y_train_jax.reshape(-1, 1)
    x_test_jax = jnp.asarray(x_test, dtype=jnp.float64)

    # Run JAX prediction with batching
    predictions_jax = _JAX_LWPLS_FUNC(
        x_train_jax,
        y_train_jax,
        x_test_jax,
        max_component_number,
        lambda_in_similarity,
        batch_size,
    )

    # Convert back to NumPy
    return np.asarray(predictions_jax)


def _lwpls_predict_torch(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    max_component_number: int,
    lambda_in_similarity: float,
    batch_size: int = 64,
    device: str = 'auto',
) -> NDArray[np.floating]:
    """PyTorch-accelerated LWPLS prediction with batching.

    Same interface as _lwpls_predict but uses PyTorch for GPU acceleration.
    Processes test samples in batches to avoid OOM on large datasets.

    Parameters
    ----------
    x_train : ndarray of shape (n_train, n_features)
        Autoscaled training X data.
    y_train : ndarray of shape (n_train,) or (n_train, 1)
        Autoscaled training y data.
    x_test : ndarray of shape (n_test, n_features)
        Autoscaled test X data.
    max_component_number : int
        Maximum number of PLS components to extract.
    lambda_in_similarity : float
        Parameter controlling the kernel width.
    batch_size : int, default=64
        Number of test samples to process per batch.
        Reduce this if running out of memory.
    device : str, default='auto'
        Device to use ('auto', 'cpu', 'cuda', 'mps').
        'auto' will use CUDA if available, otherwise CPU.

    Returns
    -------
    estimated_y_test : ndarray of shape (n_test, max_component_number)
        Predictions for each number of components.
    """
    global _TORCH_LWPLS_FUNC

    if _TORCH_LWPLS_FUNC is None:
        _TORCH_LWPLS_FUNC = _get_torch_lwpls_functions()

    import torch

    # Convert to PyTorch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    if y_train_t.ndim == 1:
        y_train_t = y_train_t.reshape(-1, 1)
    x_test_t = torch.tensor(x_test, dtype=torch.float64)

    # Run PyTorch prediction with batching
    predictions_torch = _TORCH_LWPLS_FUNC(
        x_train_t,
        y_train_t,
        x_test_t,
        max_component_number,
        lambda_in_similarity,
        batch_size,
        device,
    )

    # Convert back to NumPy
    return predictions_torch.cpu().numpy()


class LWPLS(BaseEstimator, RegressorMixin):
    """Locally-Weighted Partial Least Squares (LWPLS) regressor.

    LWPLS builds a local PLS model for each query sample, weighting
    training samples by their similarity (proximity) to the query.
    This approach is useful for:

    - Data with local nonlinearity
    - Drifting processes where the relationship changes over time
    - Heterogeneous data where a single global model is inadequate

    The similarity is computed using a Gaussian kernel based on
    Euclidean distance, controlled by the `lambda_in_similarity` parameter.

    Parameters
    ----------
    n_components : int, default=10
        Maximum number of PLS components to extract for each local model.
    lambda_in_similarity : float, default=1.0
        Kernel width parameter. Smaller values create more localized models
        (more weight on nearby samples), larger values approach global PLS.
        Typical values range from 2^-9 to 2^5 depending on the data.
    scale : bool, default=True
        Whether to standardize X and y before fitting. Strongly recommended
        as LWPLS uses Euclidean distances.
    backend : str, default='numpy'
        Computational backend to use. Options are:
        - 'numpy': NumPy backend (CPU only, default).
        - 'jax': JAX backend (supports GPU/TPU acceleration).
        - 'torch': PyTorch backend (supports GPU acceleration).
        JAX backend requires JAX to be installed: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``
        PyTorch backend requires PyTorch: ``pip install torch``
        For GPU support: ``pip install torch`` with CUDA.
    batch_size : int, default=64
        Number of test samples to process per batch (JAX/torch backends).
        Reduce this if running out of GPU memory on large datasets.
        Ignored for NumPy backend.

    Attributes
    ----------
    n_features_in\_ : int
        Number of features seen during fit.
    n_components\_ : int
        Actual number of components used (limited by data dimensions).
    X_train\_ : ndarray of shape (n_samples, n_features)
        Stored training X data (standardized if scale=True).
    y_train\_ : ndarray of shape (n_samples,)
        Stored training y data (standardized if scale=True).
    x_scaler\_ : StandardScaler or None
        Fitted scaler for X (if scale=True).
    y_scaler\_ : StandardScaler or None
        Fitted scaler for y (if scale=True).

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.lwpls import LWPLS
    >>> import numpy as np
    >>> # Nonlinear data
    >>> np.random.seed(42)
    >>> X = 5 * np.random.rand(100, 2)
    >>> y = 3 * X[:, 0]**2 + 10 * np.log(X[:, 1] + 0.1) + np.random.randn(100)
    >>> # Split data
    >>> X_train, X_test = X[:70], X[70:]
    >>> y_train, y_test = y[:70], y[70:]
    >>> # Fit LWPLS with NumPy backend (default)
    >>> model = LWPLS(n_components=5, lambda_in_similarity=0.25)
    >>> model.fit(X_train, y_train)
    LWPLS(n_components=5, lambda_in_similarity=0.25)
    >>> y_pred = model.predict(X_test)
    >>> # Use JAX backend for GPU acceleration
    >>> model_jax = LWPLS(n_components=5, lambda_in_similarity=0.25, backend='jax')
    >>> model_jax.fit(X_train, y_train)
    >>> y_pred_jax = model_jax.predict(X_test)
    >>> # Use PyTorch backend for GPU acceleration
    >>> model_torch = LWPLS(n_components=5, lambda_in_similarity=0.25, backend='torch')
    >>> model_torch.fit(X_train, y_train)
    >>> y_pred_torch = model_torch.predict(X_test)

    Notes
    -----
    LWPLS is computationally more expensive than standard PLS because
    it builds a separate weighted model for each prediction. The training
    data must be stored for prediction.

    The JAX backend provides significant speedups on GPU by:
    - Vectorizing the per-sample loop using ``jax.vmap``
    - JIT-compiling the prediction function
    - Running on GPU/TPU when available

    The PyTorch backend provides GPU acceleration by:
    - Running tensor operations on CUDA or MPS devices
    - Batched processing to control memory usage
    - Automatic device selection when device='auto'

    The optimal `lambda_in_similarity` should be tuned via cross-validation.
    Typical search range is 2^k for k in [-9, 6].

    This implementation is adapted from the original code by Hiromasa Kaneko
    (https://github.com/hkaneko1985/lwpls), licensed under MIT License.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard global PLS.
    IKPLS : Fast PLS implementation.

    References
    ----------
    - Kim, S., et al. (2011). Estimation of active pharmaceutical
      ingredient content using locally weighted partial least squares.
      International Journal of Pharmaceutics, 421(2), 269-274.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        lambda_in_similarity: float = 1.0,
        scale: bool = True,
        backend: str = 'numpy',
        batch_size: int = 64,
    ):
        """Initialize LWPLS regressor.

        Parameters
        ----------
        n_components : int, default=10
            Maximum number of PLS components.
        lambda_in_similarity : float, default=1.0
            Kernel width parameter for similarity computation.
        scale : bool, default=True
            Whether to standardize X and y.
        backend : str, default='numpy'
            Computational backend ('numpy', 'jax', or 'torch').
        batch_size : int, default=64
            Batch size for JAX/torch backend to control memory usage.
        """
        self.n_components = n_components
        self.lambda_in_similarity = lambda_in_similarity
        self.scale = scale
        self.backend = backend
        self.batch_size = batch_size

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "LWPLS":
        """Fit the LWPLS model.

        This stores the training data and fits scalers if requested.
        Actual model building happens lazily at prediction time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : LWPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is not 'numpy', 'jax', or 'torch'.
        ImportError
            If backend is 'jax' and JAX is not installed, or
            if backend is 'torch' and PyTorch is not installed.
        """
        # Validate backend
        if self.backend not in ('numpy', 'jax', 'torch'):
            raise ValueError(
                f"backend must be 'numpy', 'jax', or 'torch', got '{self.backend}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for LWPLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        if self.backend == 'torch' and not _check_torch_available():
            raise ImportError(
                "PyTorch is required for LWPLS with backend='torch'. "
                "Install it with: pip install torch\n"
                "For GPU support, see: https://pytorch.org/get-started/locally/"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.n_features_in_ = X.shape[1]

        # Limit components by data dimensions
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components_ = min(self.n_components, max_components)

        if self.scale:
            self.x_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()

            self.X_train_ = self.x_scaler_.fit_transform(X)
            self.y_train_ = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            self.x_scaler_ = None
            self.y_scaler_ = None
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()

        # Store original data for reference
        self._n_train_samples = X.shape[0]

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: Union[int, None] = None,
    ) -> NDArray[np.floating]:
        """Predict using the LWPLS model.

        Builds a local weighted PLS model for each test sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use for prediction.
            If None, uses ``n_components_`` (all fitted components).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'n_components_'])

        X = np.asarray(X, dtype=np.float64)

        if n_components is None:
            n_components = self.n_components_
        else:
            n_components = min(n_components, self.n_components_)

        # Scale input if needed
        if self.scale and self.x_scaler_ is not None:
            X_scaled = self.x_scaler_.transform(X)
        else:
            X_scaled = X

        # Get predictions for all component numbers using appropriate backend
        if self.backend == 'jax':
            all_predictions = _lwpls_predict_jax(
                self.X_train_,
                self.y_train_,
                X_scaled,
                n_components,
                self.lambda_in_similarity,
                self.batch_size,
            )
        elif self.backend == 'torch':
            all_predictions = _lwpls_predict_torch(
                self.X_train_,
                self.y_train_,
                X_scaled,
                n_components,
                self.lambda_in_similarity,
                self.batch_size,
            )
        else:
            all_predictions = _lwpls_predict(
                self.X_train_,
                self.y_train_,
                X_scaled,
                n_components,
                self.lambda_in_similarity,
            )

        # Take prediction from the requested number of components
        y_pred_scaled = all_predictions[:, n_components - 1]

        # Inverse transform if needed
        if self.scale and self.y_scaler_ is not None:
            y_pred = self.y_scaler_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
        else:
            y_pred = y_pred_scaled

        return y_pred

    def predict_all_components(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Predict with all component numbers (for component selection).

        Returns predictions for each number of components, which can be
        used for cross-validation to select the optimal n_components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred_all : ndarray of shape (n_samples, n_components)
            Predictions where column i contains predictions using i+1 components.
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'n_components_'])

        X = np.asarray(X, dtype=np.float64)

        # Scale input if needed
        if self.scale and self.x_scaler_ is not None:
            X_scaled = self.x_scaler_.transform(X)
        else:
            X_scaled = X

        # Get predictions for all component numbers using appropriate backend
        if self.backend == 'jax':
            all_predictions = _lwpls_predict_jax(
                self.X_train_,
                self.y_train_,
                X_scaled,
                self.n_components_,
                self.lambda_in_similarity,
                self.batch_size,
            )
        elif self.backend == 'torch':
            all_predictions = _lwpls_predict_torch(
                self.X_train_,
                self.y_train_,
                X_scaled,
                self.n_components_,
                self.lambda_in_similarity,
                self.batch_size,
            )
        else:
            all_predictions = _lwpls_predict(
                self.X_train_,
                self.y_train_,
                X_scaled,
                self.n_components_,
                self.lambda_in_similarity,
            )

        # Inverse transform if needed
        if self.scale and self.y_scaler_ is not None:
            # Need to inverse transform each column
            y_pred_all = np.zeros_like(all_predictions)
            for i in range(all_predictions.shape[1]):
                y_pred_all[:, i] = self.y_scaler_.inverse_transform(
                    all_predictions[:, i : i + 1]
                ).ravel()
        else:
            y_pred_all = all_predictions

        return y_pred_all

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_components': self.n_components,
            'lambda_in_similarity': self.lambda_in_similarity,
            'scale': self.scale,
            'backend': self.backend,
            'batch_size': self.batch_size,
        }

    def set_params(self, **params) -> "LWPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : LWPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LWPLS(n_components={self.n_components}, "
            f"lambda_in_similarity={self.lambda_in_similarity}, "
            f"scale={self.scale}, backend='{self.backend}', "
            f"batch_size={self.batch_size})"
        )
