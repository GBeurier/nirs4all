"""Fractional Convolutional Kernel PLS (FCK-PLS) regressor for nirs4all.

A sklearn-compatible implementation of FCK-PLS that uses fractional order
convolutional filters to build spectral features, then applies PLS regression.
This approach is particularly suited for NIRS data where derivative-like
features at various fractional orders can capture different spectral signatures.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
.. [1] Podlubny, I. (1999). Fractional differential equations. Academic Press.

.. [2] Chen, Y., Petras, I., & Xue, D. (2009). Fractional order control -
       A tutorial. In Proc. American Control Conference (pp. 1397-1411).

Mathematical formulation
------------------------
Let

- X ∈ ℝ^{n×p} be the input matrix of n samples and p features
  (e.g. NIRS spectra, treated as 1D signals over wavelength),
- Y ∈ ℝ^{n×q} be the response matrix.

FCK-PLS builds an explicit feature map Φ : ℝ^p → ℝ^{D} by convolving each
spectrum with a bank of L fractional filters { h_ℓ }_{ℓ=1,…,L}, and then
applies PLS in this expanded feature space.

Fractional filter bank
----------------------
Each filter h_ℓ ∈ ℝ^{k} (k odd) is defined by parameters (α_ℓ, σ_ℓ) that
control its fractional "order" and scale. Conceptually, h_ℓ approximates
a 1D operator whose frequency response has the form

    H_ℓ(ω) ∝ |ω|^{α_ℓ} exp(−σ_ℓ ω²),

so that:

- α_ℓ ≈ 0 corresponds to a smoothing operator,
- α_ℓ ≈ 1 to a first-derivative-like operator,
- α_ℓ ≈ 2 to a second-derivative-like operator,

with intermediate values giving fractional-order behavior. In practice,
h_ℓ is implemented as a discrete, symmetric 1D kernel constructed from
(α_ℓ, σ_ℓ, k) and normalized for numerical stability.

For a single spectrum x ∈ ℝ^p, the convolution with filter ℓ is

    f_ℓ = x * h_ℓ,          f_ℓ ∈ ℝ^{p′},

where * denotes 1D discrete convolution along the wavelength axis (with
either "same" or "valid" output length p′).

The feature map Φ stacks all convolved signals:

    Φ(x) = [ f_1ᵀ, f_2ᵀ, …, f_Lᵀ ]ᵀ ∈ ℝ^{D},
    D = L · p′.

Collecting all samples, we form the feature-expanded matrix

    X_feat ∈ ℝ^{n×D},     row i = Φ(x_i)ᵀ.

PLS in feature space
--------------------
On X_feat and Y, FCK-PLS applies a standard PLS regression:

- find loading matrix W_feat ∈ ℝ^{D×r},
- scores T = X_feat W_feat ∈ ℝ^{n×r},
- regression matrix C ∈ ℝ^{r×q},

such that:

1) the covariance between T and Y is maximized (PLS objective), and
2) the regression Y ≈ T C is well fitted in least-squares sense.

Equivalently, one can define a kernel in the original input space

    K_{ij} = Φ(x_i)ᵀ Φ(x_j),

and view FCK-PLS as a (linear) Kernel PLS in the feature space induced
by the fractional convolutional map Φ. In this implementation, the
feature map is explicit (X_feat is computed directly) and a standard
PLSRegression is applied.

Prediction for new data X* proceeds as follows:

1) apply the same preprocessing and fractional convolutional featurizer
   to get X*_feat,
2) compute scores T* = X*_feat W_feat,
3) output Ŷ* = T* C (with inverse scaling if standardization is used).

By tuning { (α_ℓ, σ_ℓ) } and the number of components r, FCK-PLS can
adaptively emphasize different fractional smooth/derivative behaviors
and scales in the spectra, providing a flexible family of
preprocessing+PLS models specialized to 1D spectral data.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax
        return True
    except ImportError:
        return False


# =============================================================================
# Fractional Filter Construction
# =============================================================================

def fractional_kernel_1d(
    alpha: float,
    sigma: float,
    kernel_size: int,
) -> NDArray[np.floating]:
    """Build a 1D discrete kernel for fractional smoothing/derivative.

    This kernel approximates fractional order operators by combining a
    Gaussian envelope with a fractional power modulation. The result
    captures derivative-like behavior at non-integer orders.

    Parameters
    ----------
    alpha : float
        Fractional order in [0, 2]:
        - 0: Pure smoothing (Gaussian-like)
        - 1: First-derivative-like behavior
        - 2: Second-derivative-like behavior
        Intermediate values provide fractional derivatives.
    sigma : float
        Scale parameter controlling the width of the kernel.
        Larger sigma = wider filter, more smoothing.
    kernel_size : int
        Number of points in the kernel (should be odd).

    Returns
    -------
    h : ndarray of shape (kernel_size,)
        Normalized discrete filter.

    Notes
    -----
    This is a heuristic approximation suitable for spectral data.
    For mathematically rigorous fractional derivatives, use FFT-based
    implementations with frequency domain multiplication by |ω|^α.
    """
    k = kernel_size
    idx = np.arange(k) - (k - 1) / 2.0  # Symmetric indices centered at 0

    # Gaussian envelope for spatial localization
    sigma_safe = max(sigma, 1e-6)
    gaussian = np.exp(-0.5 * (idx / sigma_safe) ** 2)

    # Handle alpha = 0 case (pure smoothing)
    if alpha < 1e-10:
        h = gaussian.copy()
    else:
        # Fractional power modulation with sign preservation
        sign = np.sign(idx)
        frac = np.abs(idx) ** alpha
        frac[np.abs(idx) < 1e-10] = 0.0  # Handle center point

        # Combine: Gaussian envelope * signed fractional power
        h = gaussian * (frac * sign)

    # Zero-mean for derivative-like behavior (when alpha > 0)
    if alpha > 0.1:
        h = h - h.mean()

    # L1 normalization for stability
    norm = np.sum(np.abs(h))
    if norm > 1e-12:
        h = h / norm

    return h


def fractional_kernel_grrunwald_letnikov(
    alpha: float,
    kernel_size: int,
) -> NDArray[np.floating]:
    """Build Grünwald-Letnikov fractional derivative kernel.

    This is a more mathematically rigorous approximation of the fractional
    derivative operator using the Grünwald-Letnikov definition.

    Parameters
    ----------
    alpha : float
        Fractional order (can be any real number).
    kernel_size : int
        Number of points in the kernel.

    Returns
    -------
    h : ndarray of shape (kernel_size,)
        Grünwald-Letnikov coefficients.

    Notes
    -----
    The Grünwald-Letnikov definition:
    D^α f(x) ≈ lim_{h→0} (1/h^α) Σ_{j=0}^{n} (-1)^j C(α,j) f(x - jh)

    where C(α,j) = Γ(α+1) / (Γ(j+1) * Γ(α-j+1))
    """
    from scipy.special import gammaln

    coeffs = np.zeros(kernel_size, dtype=np.float64)

    for j in range(kernel_size):
        # Binomial-like coefficient: (-1)^j * C(α, j)
        # C(α, j) = Γ(α+1) / (Γ(j+1) * Γ(α-j+1))
        # Using log-gamma for numerical stability
        arg1 = alpha + 1
        arg2 = j + 1
        arg3 = alpha - j + 1

        # Check for valid gamma arguments (no non-positive integers)
        args_valid = True
        for arg in [arg1, arg2, arg3]:
            if arg <= 0 and float(arg).is_integer():
                args_valid = False
                break

        if args_valid:
            try:
                # Use log-gamma to avoid overflow
                log_coeff = gammaln(arg1) - gammaln(arg2) - gammaln(arg3)
                coeff = np.exp(log_coeff)
                if np.isfinite(coeff):
                    coeffs[j] = ((-1) ** j) * coeff
                else:
                    coeffs[j] = 0.0
            except (ValueError, OverflowError, RuntimeWarning):
                coeffs[j] = 0.0
        else:
            coeffs[j] = 0.0

    # Ensure no NaN or Inf
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    norm = np.sum(np.abs(coeffs))
    if norm > 1e-12:
        coeffs = coeffs / norm

    return coeffs


# =============================================================================
# Fractional Convolutional Featurizer
# =============================================================================

class FractionalConvFeaturizer(BaseEstimator, TransformerMixin):
    """Convolutional featurizer using a bank of fractional filters.

    Builds features by convolving input spectra with multiple fractional
    order filters at different scales. This captures derivative-like
    information at various fractional orders, which can be useful for
    identifying spectral features.

    Parameters
    ----------
    alphas : sequence of float, default=(0.0, 0.5, 1.0, 1.5, 2.0)
        Fractional orders for the filter bank.
        - 0: Smoothing/identity-like
        - 0.5: Half-derivative
        - 1: First derivative
        - 1.5: Fractional between 1st and 2nd derivative
        - 2: Second derivative
    sigmas : sequence of float, default=(2.0,)
        Scale parameters. If single value, same sigma for all alphas.
        If same length as alphas, pairs (alpha[i], sigma[i]).
    kernel_size : int, default=15
        Size of convolution kernels (should be odd).
    mode : str, default='same'
        Convolution mode:
        - 'same': Output same length as input
        - 'valid': Output shorter (no padding)
    kernel_type : str, default='heuristic'
        Type of fractional kernel:
        - 'heuristic': Gaussian-modulated fractional power
        - 'grunwald': Grünwald-Letnikov coefficients

    Attributes
    ----------
    kernels_ : list of ndarray
        Precomputed convolution kernels.
    n_kernels_ : int
        Number of kernels in the filter bank.
    """

    def __init__(
        self,
        alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
        sigmas: Sequence[float] = (2.0,),
        kernel_size: int = 15,
        mode: Literal['same', 'valid'] = 'same',
        kernel_type: Literal['heuristic', 'grunwald'] = 'heuristic',
    ):
        """Initialize the fractional convolutional featurizer."""
        self.alphas = list(alphas)
        self.sigmas = list(sigmas)
        self.kernel_size = kernel_size
        self.mode = mode
        self.kernel_type = kernel_type

    def fit(self, X, y=None):
        """Precompute convolution kernels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (used only for validation).
        y : ignored

        Returns
        -------
        self : FractionalConvFeaturizer
        """
        # Ensure odd kernel size
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        # Handle sigma broadcasting
        if len(self.sigmas) == 1 and len(self.alphas) > 1:
            sigmas = [self.sigmas[0]] * len(self.alphas)
        else:
            sigmas = self.sigmas

        if len(sigmas) != len(self.alphas):
            raise ValueError(
                "sigmas must have length 1 or same length as alphas. "
                f"Got {len(sigmas)} sigmas and {len(self.alphas)} alphas."
            )

        # Build kernels
        self.kernels_ = []
        for alpha, sigma in zip(self.alphas, sigmas):
            if self.kernel_type == 'heuristic':
                h = fractional_kernel_1d(alpha, sigma, self.kernel_size)
            elif self.kernel_type == 'grunwald':
                h = fractional_kernel_grrunwald_letnikov(alpha, self.kernel_size)
            else:
                raise ValueError(
                    f"kernel_type must be 'heuristic' or 'grunwald', "
                    f"got '{self.kernel_type}'"
                )
            self.kernels_.append(h)

        self.n_kernels_ = len(self.kernels_)

        return self

    def transform(self, X):
        """Apply fractional convolution filter bank.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.

        Returns
        -------
        X_feat : ndarray of shape (n_samples, n_features_out)
            Convolved features. n_features_out depends on mode:
            - 'same': n_features * n_kernels
            - 'valid': (n_features - kernel_size + 1) * n_kernels
        """
        check_is_fitted(self, ['kernels_'])

        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Compute output length
        if self.mode == 'same':
            out_len = n_features
        elif self.mode == 'valid':
            out_len = n_features - self.kernel_size + 1
            if out_len <= 0:
                raise ValueError(
                    f"Input has {n_features} features but kernel_size "
                    f"is {self.kernel_size}. Use smaller kernel or 'same' mode."
                )
        else:
            raise ValueError(f"mode must be 'same' or 'valid', got '{self.mode}'")

        # Allocate output
        X_feat = np.empty(
            (n_samples, self.n_kernels_ * out_len),
            dtype=np.float64
        )

        # Apply each kernel
        for i in range(n_samples):
            x = X[i]
            feats = []
            for h in self.kernels_:
                conv = fftconvolve(x, h, mode=self.mode)
                feats.append(conv)
            X_feat[i, :] = np.concatenate(feats)

        return X_feat

    def get_kernel_info(self) -> dict:
        """Get information about the filter bank.

        Returns
        -------
        info : dict
            Dictionary containing kernel parameters and shapes.
        """
        check_is_fitted(self, ['kernels_'])

        return {
            'n_kernels': self.n_kernels_,
            'alphas': self.alphas,
            'sigmas': self.sigmas if len(self.sigmas) == len(self.alphas)
                     else [self.sigmas[0]] * len(self.alphas),
            'kernel_size': self.kernel_size,
            'kernel_type': self.kernel_type,
            'mode': self.mode,
        }


# =============================================================================
# JAX Backend for Fractional Convolution
# =============================================================================

def _get_jax_fckpls_functions():
    """Lazy import and create JAX FCK-PLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def convolve_1d_jax(x, kernel):
        """1D convolution using JAX."""
        # Use lax.conv_general_dilated for 1D convolution
        # Reshape for conv: (batch, spatial, channels)
        x_reshaped = x.reshape(1, -1, 1)
        kernel_reshaped = kernel.reshape(-1, 1, 1)

        # Padding for 'same' mode
        pad = kernel.shape[0] // 2

        result = lax.conv_general_dilated(
            x_reshaped,
            kernel_reshaped,
            window_strides=(1,),
            padding=((pad, pad),),
            dimension_numbers=('NWC', 'WIO', 'NWC'),
        )

        return result.ravel()

    def apply_filter_bank_jax(X, kernels):
        """Apply filter bank using JAX.

        Note: This function is not JIT-compiled because the number
        of kernels is dynamic.
        """
        X = jnp.asarray(X)
        n_samples, n_features = X.shape
        n_kernels = len(kernels)

        results = []
        for h in kernels:
            h_jax = jnp.asarray(h)
            # Apply to each sample
            convolved = jax.vmap(lambda x: convolve_1d_jax(x, h_jax))(X)
            results.append(convolved)

        return jnp.hstack(results)

    return {
        'convolve_1d': convolve_1d_jax,
        'apply_filter_bank': apply_filter_bank_jax,
    }


_JAX_FCKPLS_FUNCS = None


def _get_cached_jax_fckpls():
    """Get cached JAX FCK-PLS functions."""
    global _JAX_FCKPLS_FUNCS
    if _JAX_FCKPLS_FUNCS is None:
        _JAX_FCKPLS_FUNCS = _get_jax_fckpls_functions()
    return _JAX_FCKPLS_FUNCS


# =============================================================================
# FCK-PLS Estimator Class
# =============================================================================

class FCKPLS(BaseEstimator, RegressorMixin):
    """Fractional Convolutional Kernel PLS (FCK-PLS).

    FCK-PLS builds spectral features by convolving input spectra with a bank
    of fractional order filters, then applies PLS regression on the expanded
    feature space. This approach captures derivative-like information at
    various fractional orders.

    The pipeline is:
    1. Optional standardization of X and Y
    2. FractionalConvFeaturizer: X -> X_feat (feature expansion)
    3. PLSRegression: X_feat, Y -> predictions

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    alphas : sequence of float, default=(0.0, 0.5, 1.0, 1.5, 2.0)
        Fractional orders for the filter bank.
    sigmas : sequence of float, default=(2.0,)
        Scale parameters for fractional kernels.
    kernel_size : int, default=15
        Size of convolution kernels (must be odd).
    mode : str, default='same'
        Convolution mode: 'same' or 'valid'.
    kernel_type : str, default='heuristic'
        Fractional kernel type: 'heuristic' or 'grunwald'.
    standardize : bool, default=True
        Whether to standardize X and Y before fitting.
    backend : str, default='numpy'
        Computational backend:
        - 'numpy': NumPy/SciPy backend (CPU)
        - 'jax': JAX backend (supports GPU/TPU)

    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    n_features_out_ : int
        Number of features after convolution.
    featurizer_ : FractionalConvFeaturizer
        The fitted fractional featurizer.
    pls_ : PLSRegression
        The fitted PLS model.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.fckpls import FCKPLS
    >>> import numpy as np
    >>> # Generate spectral data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 200)  # 100 samples, 200 wavelengths
    >>> y = X[:, 50:60].mean(axis=1) + 0.1 * np.random.randn(100)
    >>> # Fit FCK-PLS with default fractional orders
    >>> model = FCKPLS(n_components=10, alphas=(0.0, 0.5, 1.0, 1.5, 2.0))
    >>> model.fit(X, y)
    FCKPLS(...)
    >>> predictions = model.predict(X)
    >>> # Use specific fractional orders
    >>> model2 = FCKPLS(n_components=10, alphas=(0.0, 1.0, 2.0), sigmas=(3.0,))
    >>> model2.fit(X, y)

    Notes
    -----
    The fractional order α controls the type of spectral feature extracted:
    - α ≈ 0: Smoothed spectrum (low-pass filtering)
    - α ≈ 1: First derivative-like (highlights slopes)
    - α ≈ 2: Second derivative-like (highlights peaks/valleys)
    - Fractional α: Intermediate behavior

    The sigma parameter controls the scale of the filter. Larger sigma
    captures broader spectral features; smaller sigma captures local details.

    FCK-PLS can be computationally expensive with many filters and large
    spectra. Consider using the JAX backend for GPU acceleration.

    See Also
    --------
    SIMPLS : Standard PLS without feature expansion.
    IntervalPLS : PLS with wavelength interval selection.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
        sigmas: Sequence[float] = (2.0,),
        kernel_size: int = 15,
        mode: Literal['same', 'valid'] = 'same',
        kernel_type: Literal['heuristic', 'grunwald'] = 'heuristic',
        standardize: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize FCK-PLS regressor."""
        self.n_components = n_components
        self.alphas = alphas
        self.sigmas = sigmas
        self.kernel_size = kernel_size
        self.mode = mode
        self.kernel_type = kernel_type
        self.standardize = standardize
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "FCKPLS":
        """Fit the FCK-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : FCKPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid.
        ImportError
            If JAX backend requested but not installed.
        """
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for FCKPLS with backend='jax'. "
                "Install with: pip install jax"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features

        # Standardization
        if self.standardize:
            self.x_scaler_ = StandardScaler(with_mean=True, with_std=True)
            self.y_scaler_ = StandardScaler(with_mean=True, with_std=True)
        else:
            self.x_scaler_ = None
            self.y_scaler_ = None

        X_proc = X.copy()
        Y_proc = y.copy()

        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.fit_transform(X_proc)
        if self.y_scaler_ is not None:
            Y_proc = self.y_scaler_.fit_transform(Y_proc)

        # Build featurizer
        self.featurizer_ = FractionalConvFeaturizer(
            alphas=self.alphas,
            sigmas=self.sigmas,
            kernel_size=self.kernel_size,
            mode=self.mode,
            kernel_type=self.kernel_type,
        )
        self.featurizer_.fit(X_proc)

        # Apply featurizer
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_fckpls()
            X_feat = np.asarray(
                jax_funcs['apply_filter_bank'](X_proc, self.featurizer_.kernels_)
            )
        else:
            X_feat = self.featurizer_.transform(X_proc)

        self.n_features_out_ = X_feat.shape[1]

        # Limit components
        max_components = min(n_samples - 1, X_feat.shape[1])
        n_comp = min(self.n_components, max_components)

        # Fit PLS on expanded features
        self.pls_ = PLSRegression(n_components=n_comp)
        self.pls_.fit(X_feat, Y_proc)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict using the FCK-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['featurizer_', 'pls_'])

        X = np.asarray(X, dtype=np.float64)

        # Preprocess
        X_proc = X.copy()
        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.transform(X_proc)

        # Apply featurizer
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_fckpls()
            X_feat = np.asarray(
                jax_funcs['apply_filter_bank'](X_proc, self.featurizer_.kernels_)
            )
        else:
            X_feat = self.featurizer_.transform(X_proc)

        # PLS predict
        Y_proc = self.pls_.predict(X_feat)

        # Inverse transform
        if self.y_scaler_ is not None:
            Y = self.y_scaler_.inverse_transform(Y_proc)
        else:
            Y = Y_proc

        if self._y_1d:
            Y = Y.ravel()

        return Y

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Transform X to PLS score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components)
            PLS scores in the feature-expanded space.
        """
        check_is_fitted(self, ['featurizer_', 'pls_'])

        X = np.asarray(X, dtype=np.float64)

        X_proc = X.copy()
        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.transform(X_proc)

        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_fckpls()
            X_feat = np.asarray(
                jax_funcs['apply_filter_bank'](X_proc, self.featurizer_.kernels_)
            )
        else:
            X_feat = self.featurizer_.transform(X_proc)

        return self.pls_.transform(X_feat)

    def get_filter_info(self) -> dict:
        """Get information about the fractional filter bank.

        Returns
        -------
        info : dict
            Dictionary containing filter parameters.
        """
        check_is_fitted(self, ['featurizer_'])
        return self.featurizer_.get_kernel_info()

    def get_fractional_features(self, X: ArrayLike) -> NDArray[np.floating]:
        """Get the fractional convolution features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra.

        Returns
        -------
        X_feat : ndarray of shape (n_samples, n_features_out)
            Fractional convolution features.
        """
        check_is_fitted(self, ['featurizer_'])

        X = np.asarray(X, dtype=np.float64)

        X_proc = X.copy()
        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.transform(X_proc)

        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_fckpls()
            X_feat = np.asarray(
                jax_funcs['apply_filter_bank'](X_proc, self.featurizer_.kernels_)
            )
        else:
            X_feat = self.featurizer_.transform(X_proc)

        return X_feat

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'alphas': self.alphas,
            'sigmas': self.sigmas,
            'kernel_size': self.kernel_size,
            'mode': self.mode,
            'kernel_type': self.kernel_type,
            'standardize': self.standardize,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "FCKPLS":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FCKPLS(n_components={self.n_components}, "
            f"alphas={self.alphas}, "
            f"sigmas={self.sigmas}, "
            f"kernel_size={self.kernel_size}, "
            f"mode='{self.mode}', "
            f"kernel_type='{self.kernel_type}', "
            f"backend='{self.backend}')"
        )


# Aliases
FractionalPLS = FCKPLS
