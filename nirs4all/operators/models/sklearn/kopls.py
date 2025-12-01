"""Kernel Orthogonal PLS (K-OPLS) regressor for nirs4all.

A sklearn-compatible implementation of K-OPLS that combines kernel methods
with Orthogonal PLS to handle nonlinear relationships in the data. K-OPLS
separates Y-predictive variation from Y-orthogonal variation in kernel space.

This implementation is based on the ConsensusOPLS R package algorithm from
https://github.com/sib-swiss/ConsensusOPLS, which itself is based on the
original K-OPLS algorithm by Bylesjo, Rantalainen, et al.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
.. [1] Bylesjo, M., Rantalainen, M., Cloarec, O., Nicholson, J. K.,
       Holmes, E., & Trygg, J. (2006). OPLS discriminant analysis:
       combining the strengths of PLS-DA and SIMCA classification.
       Journal of Chemometrics, 20(8-10), 341-351.

.. [2] Rantalainen, M., Bylesjo, M., Cloarec, O., Nicholson, J. K.,
       Holmes, E., & Trygg, J. (2007). Kernel-based orthogonal
       projections to latent structures (K-OPLS). Journal of
       Chemometrics, 21(7-9), 376-385.

.. [3] ConsensusOPLS R package: https://github.com/sib-swiss/ConsensusOPLS
"""

from __future__ import annotations

from typing import Literal

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
    """Compute RBF (Gaussian) kernel: K(x, y) = exp(-gamma ||x - y||^2)."""
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = X_norm_sq + Y_norm_sq.T - 2 * (X @ Y.T)
    sq_dists = np.maximum(sq_dists, 0)

    return np.exp(-gamma * sq_dists)


def _polynomial_kernel(
    X: NDArray,
    Y: NDArray = None,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
) -> NDArray:
    """Compute polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree."""
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return (gamma * (X @ Y.T) + coef0) ** degree


# =============================================================================
# Kernel Centering Functions (following ConsensusOPLS exactly)
# =============================================================================

def _center_kernel_train(K: NDArray) -> NDArray:
    """Center training kernel matrix K = <phi(Xtr), phi(Xtr)>.

    This is equivalent to R's: scale(t(scale(K, scale=F)), scale=F)
    which double-centers the kernel matrix.
    """
    # Center columns then rows (double centering)
    K_centered = K - K.mean(axis=0, keepdims=True)
    K_centered = K_centered - K_centered.mean(axis=1, keepdims=True)
    return K_centered


def _center_kernel_test_train(KteTr: NDArray, KtrTr: NDArray) -> NDArray:
    """Center hybrid test/training kernel KteTr = <phi(Xte), phi(Xtr)>.

    Following ConsensusOPLS koplsCenterKTeTr function.
    """
    n_train = KtrTr.shape[0]
    n_test = KteTr.shape[0]

    # scaling_matrix = (1/n_train) * ones(n_test) @ ones(n_train).T
    ones_test = np.ones((n_test, 1))
    ones_train = np.ones((1, n_train))
    scaling_matrix = (1.0 / n_train) * (ones_test @ ones_train)

    # KteTr = (KteTr - scaling_matrix @ KtrTr) @ (I - (1/n_train) * ones @ ones.T)
    I_train = np.eye(n_train)
    center_right = I_train - (1.0 / n_train) * (ones_train.T @ ones_train)

    KteTr_centered = (KteTr - scaling_matrix @ KtrTr) @ center_right

    return KteTr_centered


def _center_kernel_test_test(KteTe: NDArray, KteTr: NDArray, KtrTr: NDArray) -> NDArray:
    """Center test kernel KteTe = <phi(Xte), phi(Xte)>.

    Following ConsensusOPLS koplsCenterKTeTe function.
    """
    n_train = KtrTr.shape[0]
    n_test = KteTr.shape[0]

    ones_test = np.ones((n_test, 1))
    ones_train = np.ones((1, n_train))
    scaling_matrix = (1.0 / n_train) * (ones_test @ ones_train)

    # KteTe = KteTe - scaling_matrix @ KteTr.T - KteTr @ scaling_matrix.T
    #         + scaling_matrix @ KtrTr @ scaling_matrix.T
    KteTe_centered = (
        KteTe
        - scaling_matrix @ KteTr.T
        - KteTr @ scaling_matrix.T
        + scaling_matrix @ KtrTr @ scaling_matrix.T
    )

    return KteTe_centered


# =============================================================================
# NumPy K-OPLS Implementation (following ConsensusOPLS algorithm exactly)
# =============================================================================

def _kopls_fit_numpy(
    K: NDArray,
    Y: NDArray,
    n_predictive: int,
    n_orthogonal: int,
) -> dict:
    """Fit K-OPLS model using NumPy following ConsensusOPLS algorithm.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Centered kernel matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered target matrix.
    n_predictive : int
        Number of predictive components (A in the paper).
    n_orthogonal : int
        Number of orthogonal components to remove (nox in the paper).

    Returns
    -------
    model_params : dict
        Dictionary containing all fitted parameters.
    """
    n_samples = K.shape[0]
    n_targets = Y.shape[1]

    # Limit predictive components
    A = min(n_predictive, max(n_targets - 1, 1))

    # Initialize identity matrix
    I = np.eye(n_samples)

    # Store deflated kernels for each orthogonal component
    # Kdeflate[i,j] stores the deflated kernel at iteration (i,j)
    Kdeflate = {}
    Kdeflate[(0, 0)] = K.copy()

    # =========================================================================
    # Step 1: SVD of Y' @ K @ Y to get Y loadings (Cp) and scaling (Sp)
    # =========================================================================
    YtKY = Y.T @ K @ Y  # (n_targets, n_targets)

    # SVD: Y'KY = U @ S @ V'
    U, S, Vt = np.linalg.svd(YtKY, full_matrices=False)

    # Extract first A components
    Cp = U[:, :A]  # Y loadings (n_targets, A)
    # Handle case where some singular values might be zero or very small
    S_safe = np.maximum(S[:A], 1e-10)
    Sp = np.diag(S_safe ** (-0.5))  # Inverse sqrt of singular values (A, A)

    # =========================================================================
    # Step 2: Compute Y scores Up = Y @ Cp
    # =========================================================================
    Up = Y @ Cp  # (n_samples, A)

    # =========================================================================
    # Step 3-11: Loop to extract orthogonal components
    # =========================================================================
    to_list = []      # Y-orthogonal score vectors
    co_list = []      # Y-orthogonal loading vectors
    so_list = []      # Eigenvalues for orthogonal components
    toNorm_list = []  # Norms of orthogonal scores before normalization
    Tp_list = []      # Predictive scores at each iteration
    Bt_list = []      # T-U regression coefficients

    for i in range(n_orthogonal):
        # Step 4: Compute predictive score Tp = K[0,i] @ Up @ Sp
        Tp_i = Kdeflate[(0, i)] @ Up @ Sp  # (n_samples, A)
        Tp_list.append(Tp_i)

        # Step 4b: Compute T-U regression coefficients Bt = (Tp'Tp)^-1 @ Tp' @ Up
        TpTp_inv = np.linalg.pinv(Tp_i.T @ Tp_i)
        Bt_i = TpTp_inv @ Tp_i.T @ Up
        Bt_list.append(Bt_i)

        # Step 5: SVD of Tp' @ (K[i,i] - Tp @ Tp') @ Tp to get orthogonal loading
        # This finds the direction in kernel space orthogonal to Tp
        K_ii = Kdeflate[(i, i)]
        TpTpt = Tp_i @ Tp_i.T  # (n_samples, n_samples)
        K_orth = K_ii - TpTpt  # Residual kernel

        # Compute Tp' @ K_orth @ Tp
        TpKorthTp = Tp_i.T @ K_orth @ Tp_i  # (A, A)

        # SVD to get first orthogonal direction
        U_orth, S_orth, Vt_orth = np.linalg.svd(TpKorthTp, full_matrices=False)

        co_i = U_orth[:, 0:1]  # (A, 1) - orthogonal loading in Tp space
        so_i = max(S_orth[0], 1e-10)  # scalar eigenvalue

        co_list.append(co_i)
        so_list.append(so_i)

        # Step 6: Compute orthogonal score to = (K[i,i] - Tp @ Tp') @ Tp @ co / sqrt(so)
        to_i = K_orth @ Tp_i @ co_i / np.sqrt(so_i)  # (n_samples, 1)

        # Step 7: Compute norm before normalization
        toNorm_i = np.sqrt(to_i.T @ to_i).item()
        toNorm_i = max(toNorm_i, 1e-10)
        toNorm_list.append(toNorm_i)

        # Step 8: Normalize orthogonal score
        to_i = to_i / toNorm_i
        to_list.append(to_i)

        # Step 9-10: Update/deflate kernel matrices
        # scale_matrix = I - to @ to'
        scale_matrix = I - to_i @ to_i.T

        # K[0, i+1] = K[0, i] @ scale_matrix'
        Kdeflate[(0, i + 1)] = Kdeflate[(0, i)] @ scale_matrix.T

        # K[i+1, i+1] = scale_matrix @ K[i,i] @ scale_matrix'
        Kdeflate[(i + 1, i + 1)] = scale_matrix @ K_ii @ scale_matrix.T

    # =========================================================================
    # Step 12: Final predictive scores Tp[nox] = K[0, nox] @ Up @ Sp
    # =========================================================================
    Tp_final = Kdeflate[(0, n_orthogonal)] @ Up @ Sp  # (n_samples, A)
    Tp_list.append(Tp_final)

    # =========================================================================
    # Step 13: Final T-U regression coefficients
    # =========================================================================
    TpTp_inv = np.linalg.pinv(Tp_final.T @ Tp_final)
    Bt_final = TpTp_inv @ Tp_final.T @ Up
    Bt_list.append(Bt_final)

    # =========================================================================
    # Combine orthogonal scores into matrix
    # =========================================================================
    if n_orthogonal > 0:
        To = np.hstack(to_list)  # (n_samples, n_orthogonal)
    else:
        To = None

    # =========================================================================
    # Compute R2 statistics
    # =========================================================================
    sstot_K = np.trace(Kdeflate[(0, 0)])
    sstot_Y = np.sum(Y ** 2)

    # R2X for each model size (with i orthogonal components)
    R2X = []
    R2Yhat = []
    for i in range(n_orthogonal + 1):
        # Residual = K[i,i] - Tp_final @ Tp_final'
        K_ii = Kdeflate[(i, i)]
        resid = K_ii - Tp_final @ Tp_final.T
        R2X.append(1.0 - np.trace(resid) / (sstot_K + 1e-10))

        # Yhat = Tp[i] @ Bt[i] @ Cp'
        Yhat_i = Tp_list[i] @ Bt_list[i] @ Cp.T
        R2Yhat.append(1.0 - np.sum((Yhat_i - Y) ** 2) / (sstot_Y + 1e-10))

    return {
        'Cp': Cp,           # Y loadings (n_targets, A)
        'Sp': Sp,           # Inverse sqrt singular values (A, A)
        'Up': Up,           # Y scores (n_samples, A)
        'Tp': Tp_list,      # List of predictive scores at each iteration
        'scoresP': Tp_final,  # Final predictive scores (n_samples, A)
        'scoresO': To,      # Orthogonal scores (n_samples, n_orthogonal)
        'co': co_list,      # Orthogonal loadings
        'so': so_list,      # Orthogonal eigenvalues
        'to': to_list,      # Orthogonal scores (list of column vectors)
        'toNorm': toNorm_list,  # Norms before normalization
        'Bt': Bt_list,      # T-U regression coefficients
        'Kdeflate': Kdeflate,  # Deflated kernel matrices
        'n_predictive': A,
        'n_orthogonal': n_orthogonal,
        'sstot_K': sstot_K,
        'sstot_Y': sstot_Y,
        'R2X': R2X,
        'R2Yhat': R2Yhat,
    }


def _kopls_predict_numpy(
    KteTr: NDArray,
    KteTe: NDArray,
    model: dict,
) -> NDArray:
    """Predict using K-OPLS model with NumPy.

    Following ConsensusOPLS koplsPredict algorithm exactly.

    Parameters
    ----------
    KteTr : ndarray of shape (n_test, n_train)
        Centered hybrid test/training kernel.
    KteTe : ndarray of shape (n_test, n_test)
        Centered test kernel (used for deflation).
    model : dict
        Fitted model parameters from _kopls_fit_numpy.

    Returns
    -------
    Yhat : ndarray of shape (n_test, n_targets)
        Predicted values (standardized, needs de-centering/scaling).
    """
    n_orthogonal = model['n_orthogonal']

    Up = model['Up']
    Sp = model['Sp']
    Cp = model['Cp']

    # Initialize deflated kernels for prediction
    KteTrdeflate = {}
    KteTedeflate = {}
    KteTrdeflate[(0, 0)] = KteTr.copy()
    KteTedeflate[(0, 0)] = KteTe.copy()

    if n_orthogonal > 0:
        for i in range(n_orthogonal):
            # Step 2.2: Predicted predictive score
            Tp_i = KteTrdeflate[(i, 0)] @ Up @ Sp

            # Step 2.3: Predicted Y-orthogonal score
            model_Tp_i = model['Tp'][i]

            # Need KteTrdeflate[i,i]
            if (i, i) not in KteTrdeflate:
                KteTrdeflate[(i, i)] = KteTrdeflate[(0, 0)].copy()

            K_diff = KteTrdeflate[(i, i)] - Tp_i @ model_Tp_i.T

            to_i = K_diff @ model_Tp_i @ model['co'][i] / np.sqrt(model['so'][i])
            to_i = to_i / model['toNorm'][i]

            # Step 2.4.5: Deflate KteTedeflate
            model_to_i = model['to'][i]
            model_K_ii = model['Kdeflate'][(i, i)]

            term1 = KteTrdeflate[(i, i)] @ model_to_i @ to_i.T
            term2 = to_i @ model_to_i.T @ KteTrdeflate[(i, i)].T
            term3 = to_i @ model_to_i.T @ model_K_ii @ model_to_i @ to_i.T
            KteTedeflate[(i + 1, i + 1)] = KteTedeflate[(i, i)] - term1 - term2 + term3

            # Step 2.5: Update KteTrdeflate[i+1, 0]
            model_K_0i = model['Kdeflate'][(0, i)]
            KteTrdeflate[(i + 1, 0)] = KteTrdeflate[(i, 0)] - to_i @ model_to_i.T @ model_K_0i.T

            # Step 2.6: Update KteTrdeflate[i+1, i+1]
            term1 = KteTrdeflate[(i, i)] @ model_to_i @ model_to_i.T
            term2 = to_i @ model_to_i.T @ model_K_ii
            term3 = to_i @ model_to_i.T @ model_K_ii @ model_to_i @ model_to_i.T
            KteTrdeflate[(i + 1, i + 1)] = KteTrdeflate[(i, i)] - term1 - term2 + term3

    # Final predictive score
    final_idx = n_orthogonal
    Tp_final = KteTrdeflate[(final_idx, 0)] @ Up @ Sp

    # Prediction: Yhat = Tp_final @ Bt_final @ Cp'
    Bt_final = model['Bt'][final_idx]
    Yhat = Tp_final @ Bt_final @ Cp.T

    return Yhat


def _kopls_transform_numpy(
    KteTr: NDArray,
    model: dict,
) -> NDArray:
    """Transform new samples to K-OPLS score space.

    Parameters
    ----------
    KteTr : ndarray of shape (n_test, n_train)
        Centered hybrid test/training kernel.
    model : dict
        Fitted model parameters.

    Returns
    -------
    T : ndarray of shape (n_test, n_predictive)
        Scores in the filtered kernel PLS space.
    """
    n_orthogonal = model['n_orthogonal']
    Up = model['Up']
    Sp = model['Sp']

    KteTrdeflate = {}
    KteTrdeflate[(0, 0)] = KteTr.copy()

    if n_orthogonal > 0:
        for i in range(n_orthogonal):
            Tp_i = KteTrdeflate[(i, 0)] @ Up @ Sp

            model_Tp_i = model['Tp'][i]

            if (i, i) not in KteTrdeflate:
                KteTrdeflate[(i, i)] = KteTrdeflate[(0, 0)].copy()

            K_diff = KteTrdeflate[(i, i)] - Tp_i @ model_Tp_i.T
            to_i = K_diff @ model_Tp_i @ model['co'][i] / np.sqrt(model['so'][i])
            to_i = to_i / model['toNorm'][i]

            model_to_i = model['to'][i]
            model_K_0i = model['Kdeflate'][(0, i)]
            model_K_ii = model['Kdeflate'][(i, i)]

            KteTrdeflate[(i + 1, 0)] = KteTrdeflate[(i, 0)] - to_i @ model_to_i.T @ model_K_0i.T

            term1 = KteTrdeflate[(i, i)] @ model_to_i @ model_to_i.T
            term2 = to_i @ model_to_i.T @ model_K_ii
            term3 = to_i @ model_to_i.T @ model_K_ii @ model_to_i @ model_to_i.T
            KteTrdeflate[(i + 1, i + 1)] = KteTrdeflate[(i, i)] - term1 - term2 + term3

    T = KteTrdeflate[(n_orthogonal, 0)] @ Up @ Sp

    return T


# =============================================================================
# JAX K-OPLS Implementation
# =============================================================================

def _get_jax_kopls_functions():
    """Lazy import and create JAX K-OPLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def linear_kernel_jax(X, Y):
        return X @ Y.T

    @jax.jit
    def rbf_kernel_jax(X, Y, gamma):
        X_norm_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
        Y_norm_sq = jnp.sum(Y ** 2, axis=1, keepdims=True)
        sq_dists = X_norm_sq + Y_norm_sq.T - 2 * (X @ Y.T)
        sq_dists = jnp.maximum(sq_dists, 0)
        return jnp.exp(-gamma * sq_dists)

    @jax.jit
    def poly_kernel_jax(X, Y, degree, gamma, coef0):
        return (gamma * (X @ Y.T) + coef0) ** degree

    @jax.jit
    def center_kernel_train_jax(K):
        """Double-center kernel matrix."""
        K_centered = K - jnp.mean(K, axis=0, keepdims=True)
        K_centered = K_centered - jnp.mean(K_centered, axis=1, keepdims=True)
        return K_centered

    @jax.jit
    def center_kernel_test_train_jax(KteTr, KtrTr):
        """Center hybrid test/training kernel."""
        n_train = KtrTr.shape[0]
        n_test = KteTr.shape[0]

        ones_test = jnp.ones((n_test, 1))
        ones_train = jnp.ones((1, n_train))
        scaling_matrix = (1.0 / n_train) * (ones_test @ ones_train)

        I_train = jnp.eye(n_train)
        center_right = I_train - (1.0 / n_train) * (ones_train.T @ ones_train)

        return (KteTr - scaling_matrix @ KtrTr) @ center_right

    @jax.jit
    def center_kernel_test_test_jax(KteTe, KteTr, KtrTr):
        """Center test kernel."""
        n_train = KtrTr.shape[0]
        n_test = KteTr.shape[0]

        ones_test = jnp.ones((n_test, 1))
        ones_train = jnp.ones((1, n_train))
        scaling_matrix = (1.0 / n_train) * (ones_test @ ones_train)

        return (
            KteTe
            - scaling_matrix @ KteTr.T
            - KteTr @ scaling_matrix.T
            + scaling_matrix @ KtrTr @ scaling_matrix.T
        )

    return {
        'linear_kernel': linear_kernel_jax,
        'rbf_kernel': rbf_kernel_jax,
        'poly_kernel': poly_kernel_jax,
        'center_kernel_train': center_kernel_train_jax,
        'center_kernel_test_train': center_kernel_test_train_jax,
        'center_kernel_test_test': center_kernel_test_test_jax,
    }


_JAX_KOPLS_FUNCS = None


def _get_cached_jax_kopls():
    """Get cached JAX K-OPLS functions."""
    global _JAX_KOPLS_FUNCS
    if _JAX_KOPLS_FUNCS is None:
        _JAX_KOPLS_FUNCS = _get_jax_kopls_functions()
    return _JAX_KOPLS_FUNCS


# =============================================================================
# KOPLS Estimator Class
# =============================================================================

class KOPLS(BaseEstimator, RegressorMixin):
    """Kernel Orthogonal PLS (K-OPLS) regressor.

    K-OPLS combines kernel methods with Orthogonal PLS to handle nonlinear
    relationships in the data. It first removes Y-orthogonal variation from
    the kernel matrix, then fits a kernel PLS model on the filtered kernel.

    This implementation follows the algorithm from ConsensusOPLS R package,
    which is based on the original K-OPLS algorithm by Rantalainen et al.

    Parameters
    ----------
    n_components : int, default=5
        Number of predictive PLS components.
    n_ortho_components : int, default=1
        Number of orthogonal components to remove. These represent
        Y-orthogonal variation that would hurt prediction.
    kernel : str, default='rbf'
        Kernel function to use:
        - 'linear': Linear kernel K(x,y) = x^T y
        - 'rbf': Radial basis function K(x,y) = exp(-gamma ||x-y||^2)
        - 'poly': Polynomial kernel K(x,y) = (gamma x^T y + coef0)^degree
    gamma : float, optional
        Kernel coefficient for 'rbf' and 'poly' kernels.
        If None, uses 1/n_features.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in polynomial kernel.
    center : bool, default=True
        Whether to center the kernel matrix.
    scale : bool, default=True
        Whether to scale Y to unit variance.
    backend : str, default='numpy'
        Computational backend to use:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU acceleration).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of predictive components used.
    n_ortho_components_ : int
        Actual number of orthogonal components used.
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data (stored for kernel computation at predict time).
    y_mean_ : ndarray of shape (n_targets,)
        Mean of Y.
    y_std_ : ndarray of shape (n_targets,)
        Standard deviation of Y.
    x_scores_ : ndarray of shape (n_samples, n_components_)
        X scores from filtered kernel PLS (T).
    y_scores_ : ndarray of shape (n_samples, n_components_)
        Y scores (U).
    y_loadings_ : ndarray of shape (n_targets, n_components_)
        Y loadings (C).
    ortho_scores_ : ndarray of shape (n_samples, n_ortho_components_)
        Orthogonal scores (T_ortho).

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.kopls import KOPLS
    >>> import numpy as np
    >>> # Generate nonlinear data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = np.sin(X[:, :5].sum(axis=1)) + 0.1 * np.random.randn(100)
    >>> # Fit K-OPLS with RBF kernel
    >>> model = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf')
    >>> model.fit(X, y)
    KOPLS(...)
    >>> predictions = model.predict(X)
    >>> # Transform to score space
    >>> T = model.transform(X)
    >>> print(T.shape)
    (100, 5)

    References
    ----------
    .. [1] Rantalainen, M., Bylesjo, M., Cloarec, O., Nicholson, J. K.,
           Holmes, E., & Trygg, J. (2007). Kernel-based orthogonal
           projections to latent structures (K-OPLS). Journal of
           Chemometrics, 21(7-9), 376-385.

    .. [2] ConsensusOPLS R package: https://github.com/sib-swiss/ConsensusOPLS
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 5,
        n_ortho_components: int = 1,
        kernel: Literal['linear', 'rbf', 'poly'] = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        center: bool = True,
        scale: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize K-OPLS regressor."""
        self.n_components = n_components
        self.n_ortho_components = n_ortho_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.center = center
        self.scale = scale
        self.backend = backend

    def _compute_kernel(self, X: NDArray, Y: NDArray = None) -> NDArray:
        """Compute kernel matrix between X and Y."""
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_kopls()

            X_jax = jnp.asarray(X)
            Y_jax = jnp.asarray(Y) if Y is not None else X_jax
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]

            if self.kernel == 'linear':
                K = jax_funcs['linear_kernel'](X_jax, Y_jax)
            elif self.kernel == 'rbf':
                K = jax_funcs['rbf_kernel'](X_jax, Y_jax, gamma)
            elif self.kernel == 'poly':
                K = jax_funcs['poly_kernel'](X_jax, Y_jax, self.degree, gamma, self.coef0)
            else:
                raise ValueError(f"Unknown kernel '{self.kernel}'")

            return np.asarray(K)
        else:
            gamma = self.gamma
            if self.kernel == 'linear':
                return _linear_kernel(X, Y)
            elif self.kernel == 'rbf':
                return _rbf_kernel(X, Y, gamma)
            elif self.kernel == 'poly':
                return _polynomial_kernel(X, Y, self.degree, gamma, self.coef0)
            else:
                raise ValueError(f"Unknown kernel '{self.kernel}'")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KOPLS":
        """Fit the K-OPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : KOPLS
            Fitted estimator.
        """
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(f"backend must be 'numpy' or 'jax', got '{self.backend}'")

        if self.kernel not in ('linear', 'rbf', 'poly'):
            raise ValueError(f"kernel must be 'linear', 'rbf', or 'poly', got '{self.kernel}'")

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for KOPLS with backend='jax'. "
                "Install it with: pip install jax"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features
        self.X_train_ = X.copy()

        # Limit components - n_components should not be limited by n_targets for the API
        # The internal algorithm limits A = min(n_components, max(n_targets-1, 1))
        # but externally we store the user's requested value (limited by n_samples)
        max_components = n_samples - 1
        self.n_components_ = min(self.n_components, max(max_components, 1))
        self.n_ortho_components_ = min(self.n_ortho_components, n_samples - self.n_components_ - 1)
        self.n_ortho_components_ = max(0, self.n_ortho_components_)

        # Center and scale Y
        if self.scale:
            self.y_mean_ = y.mean(axis=0)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.y_mean_ = np.zeros(n_targets, dtype=np.float64)
            self.y_std_ = np.ones(n_targets, dtype=np.float64)

        Y_centered = (y - self.y_mean_) / self.y_std_

        # Compute and center kernel
        K = self._compute_kernel(X)
        self._K_train_raw = K.copy()  # Store raw kernel for prediction

        if self.center:
            K = _center_kernel_train(K)

        self._K_train = K.copy()

        # Fit model using NumPy (JAX backend uses NumPy for fit, JAX for kernels)
        model = _kopls_fit_numpy(K, Y_centered, self.n_components_, self.n_ortho_components_)

        self._model = model
        self.x_scores_ = model['scoresP']
        self.y_scores_ = model['Up']
        self.y_loadings_ = model['Cp']
        self.ortho_scores_ = model['scoresO']
        # Create ortho_loadings_ from co list (each co[i] is (A, 1) array)
        if self.n_ortho_components_ > 0 and model['co']:
            self.ortho_loadings_ = np.hstack(model['co'])  # (A, n_ortho_components)
        else:
            self.ortho_loadings_ = None

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict using the K-OPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['X_train_', 'y_mean_', 'y_std_', '_model'])

        X = np.asarray(X, dtype=np.float64)

        # Compute hybrid test/training kernel
        KteTr = self._compute_kernel(X, self.X_train_)
        # Compute test kernel
        KteTe = self._compute_kernel(X, X)

        # Center kernels using training statistics
        if self.center:
            KteTr = _center_kernel_test_train(KteTr, self._K_train_raw)
            KteTe = _center_kernel_test_test(KteTe, KteTr, self._K_train_raw)

        # Predict
        y_pred_std = _kopls_predict_numpy(KteTr, KteTe, self._model)

        # De-standardize
        y_pred = y_pred_std * self.y_std_ + self.y_mean_

        if self._y_1d:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Transform X to K-OPLS score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components_)
            X scores in the filtered kernel PLS space.
        """
        check_is_fitted(self, ['X_train_', '_model'])

        X = np.asarray(X, dtype=np.float64)

        # Compute kernel
        KteTr = self._compute_kernel(X, self.X_train_)

        # Center kernel
        if self.center:
            KteTr = _center_kernel_test_train(KteTr, self._K_train_raw)

        # Transform
        T = _kopls_transform_numpy(KteTr, self._model)

        return T

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'n_ortho_components': self.n_ortho_components,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'center': self.center,
            'scale': self.scale,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "KOPLS":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"KOPLS(n_components={self.n_components}, "
            f"n_ortho_components={self.n_ortho_components}, "
            f"kernel='{self.kernel}', gamma={self.gamma}, "
            f"degree={self.degree}, coef0={self.coef0}, "
            f"center={self.center}, scale={self.scale}, "
            f"backend='{self.backend}')"
        )
