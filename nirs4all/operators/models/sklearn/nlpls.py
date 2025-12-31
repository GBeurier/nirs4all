"""Nonlinear PLS (NL-PLS / Kernel PLS) regressor for nirs4all.

A sklearn-compatible implementation of Nonlinear PLS using kernel methods.
This approach maps the data into a higher-dimensional feature space using
a kernel function (e.g., RBF) and then fits a standard PLS model on the
kernel matrix.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

Two implementations are provided:

1. KernelPLS (KPLS) - Simple Kernel PLS
   Maps X into kernel space using a nonlinear kernel (RBF, polynomial, etc.)
   and fits PLS on the kernel matrix K = kernel(X, X).

2. MIRPLS - Monotonic Inner Relation PLS (experimental)
   Implements the MIR-PLS algorithm from Zheng et al. (2024) which uses
   monotonic cubic spline piecewise regression for the inner model.

References
----------
- Rosipal, R., & Trejo, L. J. (2001). Kernel partial least squares
  regression in reproducing kernel hilbert space. Journal of Machine
  Learning Research, 2, 97-123.
- Zheng, X., Nie, B., Du, J., et al. (2024). A non-linear partial
  least squares based on monotonic inner relation. Frontiers in
  Physiology, 15, 1369165. doi:10.3389/fphys.2024.1369165
- Qin, S. J., & McAvoy, T. J. (1992). Nonlinear PLS modeling using
  neural networks. Computers & Chemical Engineering, 16(4), 379-391.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax
        return True
    except ImportError:
        return False


# =============================================================================
# Kernel Functions (NumPy)
# =============================================================================

def _linear_kernel(X: NDArray, Y: NDArray = None) -> NDArray:
    """Compute linear kernel: K(x, y) = x^T y."""
    if Y is None:
        Y = X
    return X @ Y.T


def _rbf_kernel(X: NDArray, Y: NDArray = None, gamma: float = None) -> NDArray:
    """Compute RBF (Gaussian) kernel: K(x, y) = exp(-gamma ||x - y||^2).

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        First input array.
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second input array. If None, uses X.
    gamma : float, optional
        Kernel coefficient. If None, uses 1/n_features.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Efficient computation of ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = X_norm_sq + Y_norm_sq.T - 2 * (X @ Y.T)
    sq_dists = np.maximum(sq_dists, 0)  # Ensure non-negative

    return np.exp(-gamma * sq_dists)


def _polynomial_kernel(
    X: NDArray,
    Y: NDArray = None,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
) -> NDArray:
    """Compute polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        First input array.
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second input array. If None, uses X.
    degree : int, default=3
        Degree of the polynomial kernel.
    gamma : float, optional
        Kernel coefficient. If None, uses 1/n_features.
    coef0 : float, default=1.0
        Independent term in kernel function.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return (gamma * (X @ Y.T) + coef0) ** degree


def _sigmoid_kernel(
    X: NDArray,
    Y: NDArray = None,
    gamma: float = None,
    coef0: float = 1.0,
) -> NDArray:
    """Compute sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0).

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        First input array.
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second input array. If None, uses X.
    gamma : float, optional
        Kernel coefficient. If None, uses 1/n_features.
    coef0 : float, default=1.0
        Independent term in kernel function.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return np.tanh(gamma * (X @ Y.T) + coef0)


def _normalize_kernel_train(K: NDArray) -> tuple[NDArray, NDArray]:
    """Normalize kernel matrix to have unit diagonal (cosine normalization).

    Computes K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))

    This bounds the kernel values to [-1, 1] for polynomial/sigmoid kernels,
    making them comparable in scale to RBF kernels.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Kernel matrix.

    Returns
    -------
    K_normalized : ndarray of shape (n_samples, n_samples)
        Normalized kernel matrix.
    diag_sqrt : ndarray of shape (n_samples,)
        Square root of diagonal elements (for normalizing test kernels).
    """
    diag = np.diag(K).copy()
    diag = np.maximum(diag, 1e-12)  # Avoid division by zero
    diag_sqrt = np.sqrt(diag)
    K_normalized = K / np.outer(diag_sqrt, diag_sqrt)
    return K_normalized, diag_sqrt


def _normalize_kernel_test(
    K_test: NDArray,
    diag_sqrt_test: NDArray,
    diag_sqrt_train: NDArray,
) -> NDArray:
    """Normalize test kernel matrix using training normalization factors.

    Parameters
    ----------
    K_test : ndarray of shape (n_test, n_train)
        Test kernel matrix K(X_test, X_train).
    diag_sqrt_test : ndarray of shape (n_test,)
        Square root of diagonal elements K(X_test, X_test).
    diag_sqrt_train : ndarray of shape (n_train,)
        Square root of training kernel diagonal.

    Returns
    -------
    K_normalized : ndarray of shape (n_test, n_train)
        Normalized test kernel matrix.
    """
    # Normalize: K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))
    K_normalized = K_test / np.outer(diag_sqrt_test, diag_sqrt_train)
    return K_normalized


def _compute_kernel_diagonal(
    X: NDArray,
    kernel: str,
    gamma: float = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> NDArray:
    """Compute diagonal of kernel matrix K(X, X) efficiently.

    For most kernels, the diagonal can be computed without forming
    the full kernel matrix, which is O(n) instead of O(n^2).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    kernel : str
        Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'.
    gamma : float, optional
        Kernel coefficient.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term for polynomial/sigmoid kernels.

    Returns
    -------
    diag : ndarray of shape (n_samples,)
        Diagonal elements K(x_i, x_i).
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    if kernel == 'linear':
        # K(x,x) = x^T x
        diag = np.sum(X ** 2, axis=1)
    elif kernel == 'rbf':
        # K(x,x) = exp(-gamma * 0) = 1
        diag = np.ones(X.shape[0], dtype=np.float64)
    elif kernel == 'poly':
        # K(x,x) = (gamma * x^T x + coef0)^degree
        x_norm_sq = np.sum(X ** 2, axis=1)
        diag = (gamma * x_norm_sq + coef0) ** degree
    elif kernel == 'sigmoid':
        # K(x,x) = tanh(gamma * x^T x + coef0)
        x_norm_sq = np.sum(X ** 2, axis=1)
        diag = np.tanh(gamma * x_norm_sq + coef0)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

    return diag


# =============================================================================
# Kernel Centering Functions
# =============================================================================

def _center_kernel_train(K: NDArray) -> NDArray:
    """Center kernel matrix for training data.

    Centering in feature space: K_c = K - 1_n K - K 1_n + 1_n K 1_n
    where 1_n is n x n matrix of 1/n.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Kernel matrix.

    Returns
    -------
    K_centered : ndarray of shape (n_samples, n_samples)
        Centered kernel matrix.
    """
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n

    K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n
    return K_centered


def _center_kernel_test(K_test: NDArray, K_train: NDArray) -> NDArray:
    """Center test kernel matrix using training kernel statistics.

    Parameters
    ----------
    K_test : ndarray of shape (n_test, n_train)
        Test kernel matrix K(X_test, X_train).
    K_train : ndarray of shape (n_train, n_train)
        Training kernel matrix K(X_train, X_train).

    Returns
    -------
    K_test_centered : ndarray of shape (n_test, n_train)
        Centered test kernel matrix.
    """
    n_test = K_test.shape[0]
    n_train = K_train.shape[0]

    ones_test = np.ones((n_test, n_train)) / n_train
    ones_train = np.ones((n_train, n_train)) / n_train

    # K_test_c = K_test - 1_m K_train - K_test 1_n + 1_m K_train 1_n
    K_centered = (
        K_test
        - ones_test @ K_train
        - K_test @ ones_train
        + ones_test @ K_train @ ones_train
    )

    return K_centered


# =============================================================================
# NumPy Kernel PLS Implementation
# =============================================================================

def _simpls_fit_numpy(
    K: NDArray,
    Y: NDArray,
    n_components: int,
) -> tuple[
    NDArray,  # T (K scores)
    NDArray,  # U (Y scores)
    NDArray,  # W (K weights)
    NDArray,  # P (K loadings)
    NDArray,  # Q (Y loadings)
    NDArray,  # B (regression coefficients)
]:
    """Fit SIMPLS on kernel matrix.

    This is a SIMPLS-style algorithm for PLS on a kernel matrix K.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Centered kernel matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered target matrix.
    n_components : int
        Number of PLS components.

    Returns
    -------
    T, U, W, P, Q, B : tuple of ndarrays
        Fitted model parameters.
    """
    n_samples = K.shape[0]
    n_targets = Y.shape[1]

    # Storage for components
    T = np.zeros((n_samples, n_components), dtype=np.float64)
    U = np.zeros((n_samples, n_components), dtype=np.float64)
    W = np.zeros((n_samples, n_components), dtype=np.float64)
    P = np.zeros((n_samples, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    B = np.zeros((n_components, n_samples, n_targets), dtype=np.float64)

    # V for orthogonalization
    V = np.zeros((n_samples, n_components), dtype=np.float64)

    # SIMPLS covariance matrix in kernel space
    S = K @ Y  # (n_samples, n_targets)

    for a in range(n_components):
        # Get weight as dominant left singular vector of S
        if n_targets == 1:
            w = S[:, 0].copy()
        else:
            u_svd, s_svd, vh = np.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

        # Normalize weight
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # K score
        t = K @ w

        # Loadings
        tt = t.T @ t
        if tt < 1e-14:
            break

        p = (K @ t) / tt  # K loading
        q = (Y.T @ t) / tt  # Y loading

        # Y score
        u = Y @ q
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-14:
            u = u / u_norm

        # Store
        T[:, a] = t
        U[:, a] = u.ravel()
        W[:, a] = w
        P[:, a] = p
        Q[:, a] = q.ravel()

        # Orthogonalize v against previous v's
        v = p.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p)

        v_norm = np.linalg.norm(v)
        if v_norm > 1e-14:
            v = v / v_norm
        V[:, a] = v

        # Deflate S
        S = S - np.outer(v, v @ S)

        # Compute B for this component count
        W_a = W[:, :a+1]
        P_a = P[:, :a+1]
        Q_a = Q[:, :a+1]

        PtW_a = P_a.T @ W_a
        try:
            R_a = W_a @ np.linalg.inv(PtW_a)
        except np.linalg.LinAlgError:
            R_a = W_a @ np.linalg.pinv(PtW_a)

        B[a] = R_a @ Q_a.T

    return T, U, W, P, Q, B


def _kernel_pls_predict_numpy(
    K_test: NDArray,
    B: NDArray,
    n_components: int,
) -> NDArray:
    """Predict using Kernel PLS model.

    Parameters
    ----------
    K_test : ndarray of shape (n_test, n_train)
        Centered test kernel matrix.
    B : ndarray of shape (n_components, n_train, n_targets)
        Regression coefficients for each component.
    n_components : int
        Number of components to use.

    Returns
    -------
    Y_pred : ndarray of shape (n_test, n_targets)
        Predicted values.
    """
    B_coef = B[n_components - 1]
    Y_pred = K_test @ B_coef
    return Y_pred


# =============================================================================
# JAX Kernel PLS Functions
# =============================================================================

def _get_jax_kernel_pls_functions():
    """Lazy import and create JAX Kernel PLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def rbf_kernel_jax(X, Y, gamma):
        """JAX JIT-compiled RBF kernel."""
        X_norm_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
        Y_norm_sq = jnp.sum(Y ** 2, axis=1, keepdims=True)
        sq_dists = X_norm_sq + Y_norm_sq.T - 2 * (X @ Y.T)
        sq_dists = jnp.maximum(sq_dists, 0)
        return jnp.exp(-gamma * sq_dists)

    @jax.jit
    def poly_kernel_jax(X, Y, degree, gamma, coef0):
        """JAX JIT-compiled polynomial kernel."""
        return (gamma * (X @ Y.T) + coef0) ** degree

    @jax.jit
    def sigmoid_kernel_jax(X, Y, gamma, coef0):
        """JAX JIT-compiled sigmoid kernel."""
        return jnp.tanh(gamma * (X @ Y.T) + coef0)

    @jax.jit
    def linear_kernel_jax(X, Y):
        """JAX JIT-compiled linear kernel."""
        return X @ Y.T

    @jax.jit
    def center_kernel_train_jax(K):
        """JAX JIT-compiled kernel centering for training."""
        n = K.shape[0]
        ones_n = jnp.ones((n, n)) / n
        return K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n

    @jax.jit
    def center_kernel_test_jax(K_test, K_train):
        """JAX JIT-compiled kernel centering for test."""
        n_test = K_test.shape[0]
        n_train = K_train.shape[0]

        ones_test = jnp.ones((n_test, n_train)) / n_train
        ones_train = jnp.ones((n_train, n_train)) / n_train

        return (
            K_test
            - ones_test @ K_train
            - K_test @ ones_train
            + ones_test @ K_train @ ones_train
        )

    @partial(jax.jit, static_argnums=(2,))
    def simpls_fit_jax(K, Y, n_components):
        """JAX JIT-compiled SIMPLS fit on kernel matrix."""
        n_samples = K.shape[0]
        n_targets = Y.shape[1]

        S = K @ Y

        def component_step(a, carry):
            S, T, U, W, P, Q, V = carry

            # Get weight as dominant left singular vector
            u_svd, s_svd, vh = jnp.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

            # Normalize
            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-14, w / w_norm, w)

            # K score
            t = K @ w

            # Loadings
            tt = t.T @ t
            tt_safe = jnp.maximum(tt, 1e-14)

            p = (K @ t) / tt_safe
            q = (Y.T @ t) / tt_safe

            # Y score
            u = Y @ q
            u_norm = jnp.linalg.norm(u)
            u = jnp.where(u_norm > 1e-14, u / u_norm, u)

            # Store
            T = T.at[:, a].set(t)
            U = U.at[:, a].set(u.ravel())
            W = W.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q.ravel())

            # Orthogonalize v
            v = p.copy()
            prev_V = V * jnp.where(jnp.arange(n_components) < a, 1.0, 0.0)
            v = v - prev_V @ (prev_V.T @ p)

            v_norm = jnp.linalg.norm(v)
            v = jnp.where(v_norm > 1e-14, v / v_norm, v)
            V = V.at[:, a].set(v)

            # Deflate S
            S = S - jnp.outer(v, v @ S)

            return S, T, U, W, P, Q, V

        # Initialize
        T = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        U = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        W = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        V = jnp.zeros((n_samples, n_components), dtype=jnp.float64)

        init_carry = (S, T, U, W, P, Q, V)
        _, T, U, W, P, Q, _ = lax.fori_loop(0, n_components, component_step, init_carry)

        # Compute B for each component count
        def compute_B(a, B):
            mask = jnp.where(jnp.arange(n_components) <= a, 1.0, 0.0)
            W_a = W * mask
            P_a = P * mask
            Q_a = Q * mask

            PtW_a = P_a.T @ W_a
            R_a = W_a @ jnp.linalg.pinv(PtW_a)
            B_a = R_a @ Q_a.T
            B = B.at[a].set(B_a)
            return B

        B = jnp.zeros((n_components, n_samples, n_targets), dtype=jnp.float64)
        B = lax.fori_loop(0, n_components, compute_B, B)

        return T, U, W, P, Q, B

    @jax.jit
    def kernel_pls_predict_jax(K_test, B, n_components):
        """JAX JIT-compiled Kernel PLS prediction."""
        B_coef = B[n_components - 1]
        return K_test @ B_coef

    return {
        'rbf_kernel': rbf_kernel_jax,
        'poly_kernel': poly_kernel_jax,
        'sigmoid_kernel': sigmoid_kernel_jax,
        'linear_kernel': linear_kernel_jax,
        'center_kernel_train': center_kernel_train_jax,
        'center_kernel_test': center_kernel_test_jax,
        'simpls_fit': simpls_fit_jax,
        'kernel_pls_predict': kernel_pls_predict_jax,
    }


# Cache for JAX functions
_JAX_KERNEL_PLS_FUNCS = None


def _get_cached_jax_kernel_pls():
    """Get cached JAX Kernel PLS functions."""
    global _JAX_KERNEL_PLS_FUNCS
    if _JAX_KERNEL_PLS_FUNCS is None:
        _JAX_KERNEL_PLS_FUNCS = _get_jax_kernel_pls_functions()
    return _JAX_KERNEL_PLS_FUNCS


# =============================================================================
# KernelPLS Estimator Class
# =============================================================================

class KernelPLS(BaseEstimator, RegressorMixin):
    """Nonlinear PLS using Kernel Methods (Kernel PLS / NL-PLS).

    Kernel PLS maps the input data X into a higher-dimensional feature space
    using a kernel function (RBF, polynomial, sigmoid) and then fits a PLS
    model on the kernel matrix K(X, X). This allows capturing nonlinear
    relationships between X and Y while retaining the interpretability of PLS.

    The algorithm:
    1. Compute kernel matrix K = kernel(X_train, X_train)
    2. Center the kernel matrix
    3. Fit PLS on K with target Y
    4. For prediction: K_test = kernel(X_test, X_train), center, predict

    This is a simple and effective approach for nonlinear regression that
    combines the power of kernel methods with PLS dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    kernel : {'rbf', 'linear', 'poly', 'sigmoid'}, default='rbf'
        Kernel function to use:
        - 'rbf': Radial basis function K(x,y) = exp(-gamma ||x-y||^2)
        - 'linear': Linear kernel K(x,y) = x^T y (equivalent to standard PLS)
        - 'poly': Polynomial kernel K(x,y) = (gamma * x^T y + coef0)^degree
        - 'sigmoid': Sigmoid kernel K(x,y) = tanh(gamma * x^T y + coef0)
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
        If None, defaults to 1/n_features.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in polynomial and sigmoid kernels.
    center_kernel : bool, default=True
        Whether to center the kernel matrix. Recommended for most cases.
    scale_y : bool, default=True
        Whether to center and scale Y to zero mean and unit variance.
    backend : str, default='numpy'
        Computational backend to use:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU acceleration).

    Attributes
    ----------
    n_features_in\_ : int
        Number of features seen during fit.
    n_components\_ : int
        Actual number of components used.
    X_train\_ : ndarray of shape (n_train, n_features)
        Training data (stored for kernel computation at predict time).
    K_train\_ : ndarray of shape (n_train, n_train)
        Raw (uncentered) training kernel matrix.
    y_mean\_ : ndarray of shape (n_targets,)
        Mean of Y (if scale_y=True).
    y_std\_ : ndarray of shape (n_targets,)
        Standard deviation of Y (if scale_y=True).
    x_scores\_ : ndarray of shape (n_train, n_components)
        X scores in kernel space (T).
    y_scores\_ : ndarray of shape (n_train, n_components)
        Y scores (U).
    coef\_ : ndarray of shape (n_train, n_targets)
        Kernel regression coefficients.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.nlpls import KernelPLS
    >>> import numpy as np
    >>> # Generate nonlinear data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = np.sin(X[:, :5].sum(axis=1)) + 0.1 * np.random.randn(100)
    >>> # Fit Kernel PLS with RBF kernel
    >>> model = KernelPLS(n_components=10, kernel='rbf', gamma=0.1)
    >>> model.fit(X, y)
    KernelPLS(...)
    >>> predictions = model.predict(X)
    >>> print(f"R^2 score: {model.score(X, y):.4f}")

    Notes
    -----
    Kernel PLS is particularly useful when:
    - The relationship between X and Y is nonlinear
    - Standard linear PLS gives poor predictions
    - You want to use kernel methods but need PLS-style dimensionality reduction

    The choice of kernel and gamma parameter significantly affects performance.
    Cross-validation is recommended for hyperparameter tuning.

    For NIRS data, the RBF kernel with small gamma often works well for
    capturing nonlinear spectral-property relationships.

    See Also
    --------
    KOPLS : Kernel OPLS with orthogonal variation filtering.
    sklearn.cross_decomposition.PLSRegression : Standard linear PLS.

    References
    ----------
    - Rosipal, R., & Trejo, L. J. (2001). Kernel partial least squares
      regression in reproducing kernel hilbert space. Journal of Machine
      Learning Research, 2, 97-123.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        kernel: Literal['rbf', 'linear', 'poly', 'sigmoid'] = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        center_kernel: bool = True,
        scale_y: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize Kernel PLS regressor."""
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.center_kernel = center_kernel
        self.scale_y = scale_y
        self.backend = backend

    def _compute_kernel(
        self,
        X: NDArray,
        Y: NDArray = None,
    ) -> NDArray:
        """Compute kernel matrix between X and Y.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First input.
        Y : ndarray of shape (n_samples_Y, n_features), optional
            Second input. If None, uses X.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        gamma = self.gamma
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        if self.backend == 'jax':
            import jax.numpy as jnp

            jax_funcs = _get_cached_jax_kernel_pls()

            X_jax = jnp.asarray(X)
            Y_jax = jnp.asarray(Y) if Y is not None else X_jax

            if self.kernel == 'linear':
                K = jax_funcs['linear_kernel'](X_jax, Y_jax)
            elif self.kernel == 'rbf':
                K = jax_funcs['rbf_kernel'](X_jax, Y_jax, gamma)
            elif self.kernel == 'poly':
                K = jax_funcs['poly_kernel'](X_jax, Y_jax, self.degree, gamma, self.coef0)
            elif self.kernel == 'sigmoid':
                K = jax_funcs['sigmoid_kernel'](X_jax, Y_jax, gamma, self.coef0)
            else:
                raise ValueError(f"Unknown kernel '{self.kernel}'")

            return np.asarray(K)
        else:
            if self.kernel == 'linear':
                return _linear_kernel(X, Y)
            elif self.kernel == 'rbf':
                return _rbf_kernel(X, Y, gamma)
            elif self.kernel == 'poly':
                return _polynomial_kernel(X, Y, self.degree, gamma, self.coef0)
            elif self.kernel == 'sigmoid':
                return _sigmoid_kernel(X, Y, gamma, self.coef0)
            else:
                raise ValueError(f"Unknown kernel '{self.kernel}'")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "KernelPLS":
        """Fit the Kernel PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : KernelPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is not 'numpy' or 'jax'.
            If kernel is not one of the supported types.
        ImportError
            If backend is 'jax' and JAX is not installed.
        """
        # Validate parameters
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.kernel not in ('rbf', 'linear', 'poly', 'sigmoid'):
            raise ValueError(
                f"kernel must be 'rbf', 'linear', 'poly', or 'sigmoid', got '{self.kernel}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for KernelPLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Handle 1D y
        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features

        # Store training data for kernel computation at predict time
        self.X_train_ = X.copy()

        # Limit components
        max_components = min(n_samples - 1, n_samples)
        self.n_components_ = min(self.n_components, max_components)

        # Center and scale Y
        if self.scale_y:
            self.y_mean_ = y.mean(axis=0)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.y_mean_ = np.zeros(n_targets, dtype=np.float64)
            self.y_std_ = np.ones(n_targets, dtype=np.float64)

        Y_centered = (y - self.y_mean_) / self.y_std_

        # Compute kernel matrix
        K = self._compute_kernel(X)

        # Normalize kernel for non-RBF kernels (unit-diagonal normalization)
        if self.kernel != 'rbf':
            K, self._kdiag_sqrt_train_ = _normalize_kernel_train(K)
        else:
            self._kdiag_sqrt_train_ = None

        self.K_train_ = K.copy()  # Store kernel before centering (but after normalization)

        # Center kernel
        if self.center_kernel:
            if self.backend == 'jax':
                import jax.numpy as jnp
                jax_funcs = _get_cached_jax_kernel_pls()
                K = np.asarray(jax_funcs['center_kernel_train'](jnp.asarray(K)))
            else:
                K = _center_kernel_train(K)

        self._K_train_centered = K.copy()

        # Fit PLS on kernel matrix
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_kernel_pls()

            K_jax = jnp.asarray(K)
            Y_jax = jnp.asarray(Y_centered)

            T, U, W, P, Q, B = jax_funcs['simpls_fit'](K_jax, Y_jax, self.n_components_)

            self.x_scores_ = np.asarray(T)
            self.y_scores_ = np.asarray(U)
            self._W = np.asarray(W)
            self._P = np.asarray(P)
            self._Q = np.asarray(Q)
            self._B = np.asarray(B)
        else:
            T, U, W, P, Q, B = _simpls_fit_numpy(K, Y_centered, self.n_components_)

            self.x_scores_ = T
            self.y_scores_ = U
            self._W = W
            self._P = P
            self._Q = Q
            self._B = B

        # Store final regression coefficients
        self.coef_ = self._B[self.n_components_ - 1]

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: Union[int, None] = None,
    ) -> NDArray[np.floating]:
        """Predict using the Kernel PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use for prediction.
            If None, uses all fitted components.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['X_train_', 'K_train_', '_B'])

        X = np.asarray(X, dtype=np.float64)

        if n_components is None:
            n_components = self.n_components_
        else:
            n_components = min(n_components, self.n_components_)

        # Compute test kernel
        K_test = self._compute_kernel(X, self.X_train_)

        # Normalize test kernel for non-RBF kernels
        if self.kernel != 'rbf':
            # Compute diagonal of K(X_test, X_test) efficiently
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            diag_test = _compute_kernel_diagonal(
                X, self.kernel, gamma, self.degree, self.coef0
            )
            diag_test = np.maximum(diag_test, 1e-12)
            diag_sqrt_test = np.sqrt(diag_test)
            # Type ignore: _kdiag_sqrt_train_ is set in fit() when kernel != 'rbf'
            K_test = _normalize_kernel_test(K_test, diag_sqrt_test, self._kdiag_sqrt_train_)  # type: ignore[arg-type]

        # Center test kernel
        if self.center_kernel:
            if self.backend == 'jax':
                import jax.numpy as jnp
                jax_funcs = _get_cached_jax_kernel_pls()
                K_test = np.asarray(
                    jax_funcs['center_kernel_test'](
                        jnp.asarray(K_test),
                        jnp.asarray(self.K_train_)
                    )
                )
            else:
                K_test = _center_kernel_test(K_test, self.K_train_)

        # Predict
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_kernel_pls()

            y_pred_std = np.asarray(
                jax_funcs['kernel_pls_predict'](
                    jnp.asarray(K_test),
                    jnp.asarray(self._B),
                    n_components
                )
            )
        else:
            y_pred_std = _kernel_pls_predict_numpy(K_test, self._B, n_components)

        # De-standardize
        y_pred = y_pred_std * self.y_std_ + self.y_mean_

        if self._y_1d:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Transform X to kernel PLS score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components_)
            X scores in kernel space.
        """
        check_is_fitted(self, ['X_train_', 'K_train_', '_W'])

        X = np.asarray(X, dtype=np.float64)

        # Compute test kernel
        K_test = self._compute_kernel(X, self.X_train_)

        # Normalize test kernel for non-RBF kernels
        if self.kernel != 'rbf':
            # Compute diagonal of K(X_test, X_test) efficiently
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            diag_test = _compute_kernel_diagonal(
                X, self.kernel, gamma, self.degree, self.coef0
            )
            diag_test = np.maximum(diag_test, 1e-12)
            diag_sqrt_test = np.sqrt(diag_test)
            # Type ignore: _kdiag_sqrt_train_ is set in fit() when kernel != 'rbf'
            K_test = _normalize_kernel_test(K_test, diag_sqrt_test, self._kdiag_sqrt_train_)  # type: ignore[arg-type]

        # Center test kernel
        if self.center_kernel:
            if self.backend == 'jax':
                import jax.numpy as jnp
                jax_funcs = _get_cached_jax_kernel_pls()
                K_test = np.asarray(
                    jax_funcs['center_kernel_test'](
                        jnp.asarray(K_test),
                        jnp.asarray(self.K_train_)
                    )
                )
            else:
                K_test = _center_kernel_test(K_test, self.K_train_)

        # Compute scores: T = K_test @ W
        T = K_test @ self._W

        return T

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
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'center_kernel': self.center_kernel,
            'scale_y': self.scale_y,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "KernelPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : KernelPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"KernelPLS(n_components={self.n_components}, "
            f"kernel='{self.kernel}', gamma={self.gamma}, "
            f"degree={self.degree}, coef0={self.coef0}, "
            f"center_kernel={self.center_kernel}, "
            f"scale_y={self.scale_y}, "
            f"backend='{self.backend}')"
        )


# Alias for backward compatibility and user convenience
NLPLS = KernelPLS
KPLS = KernelPLS
