"""Fractional Convolutional Kernel (FCK) static transformer.

A small, fixed bank of normalised fractional-derivative filters applied as
1D convolutions across the wavelength axis. Each filter is built from a
Gaussian envelope multiplied by a signed fractional-power profile, then
zero-meaned (for ``alpha > 0``) and L1-normalised.

The bank is the Cartesian product of ``alphas × scales × kernel_sizes``.
With the defaults the bank has 16 filters
(``{0.5, 1.0, 1.5, 2.0} × {1, 2} × {15, 31}``); the output is the
horizontally concatenated convolution responses, shape ``(n, K * L)``.

The transformer is stateless: ``fit`` only validates the input. The
filters depend solely on the constructor hyperparameters, so no information
leaks from training data into the bank.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse
from scipy.ndimage import convolve1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array


def _build_kernel(
    alpha: float,
    scale: float,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """Build a single normalised fractional-derivative kernel.

    Parameters
    ----------
    alpha : float
        Fractional order. ``alpha < 0.1`` returns a pure Gaussian smoother.
    scale : float
        Multiplier on the index axis. Larger ``scale`` widens the receptive
        field of the same-sized kernel.
    kernel_size : int
        Odd kernel length.
    sigma : float
        Gaussian envelope standard deviation, in raw index units before
        the ``scale`` multiplier.

    Returns
    -------
    np.ndarray
        1D kernel of length ``kernel_size``, L1-normalised, zero-mean for
        ``alpha > 0``.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    half = kernel_size // 2
    idx = np.arange(-half, half + 1, dtype=np.float64) * float(scale)
    gauss = np.exp(-0.5 * (idx / float(sigma)) ** 2)
    if alpha < 0.1:
        kernel = gauss
    else:
        # signed fractional derivative: gauss * sign(idx) * |idx|**alpha
        eps = 1e-8
        frac = np.sign(idx) * np.power(np.abs(idx) + eps, float(alpha))
        kernel = gauss * frac
        kernel = kernel - kernel.mean()
    norm = float(np.sum(np.abs(kernel)) + 1e-8)
    return np.asarray(kernel / norm, dtype=np.float64)


class FCKStaticTransformer(TransformerMixin, BaseEstimator):
    """Static fractional convolutional kernel bank for 1D spectra.

    Parameters
    ----------
    alphas : sequence of float, default=(0.5, 1.0, 1.5, 2.0)
        Fractional orders for the filter bank.
    scales : sequence of float, default=(1, 2)
        Index-axis multipliers.
    kernel_sizes : sequence of int, default=(15, 31)
        Odd kernel lengths.
    sigma : float, default=3.0
        Gaussian envelope sigma. Fixed (not searched) to keep the bank
        small.
    mode : {'nearest', 'reflect', 'mirror', 'constant', 'wrap'}, default='nearest'
        Boundary mode passed to :func:`scipy.ndimage.convolve1d`.
    flatten : bool, default=True
        If ``True`` (default), output shape is ``(n_samples, K * n_features)``
        so the transformer composes directly with sklearn estimators that
        expect 2D input. If ``False``, output is ``(n_samples, K, n_features)``.
    copy : bool, default=True
        Honoured by the sklearn transformer protocol; the implementation
        always allocates a new output array.

    Attributes
    ----------
    kernels_ : np.ndarray
        Bank of shape ``(K, kernel_size_max)`` after fitting. Variable-length
        kernels are right-padded with zeros for storage; the actual
        per-kernel length is recovered from ``kernel_specs_``.
    kernel_specs_ : list of tuple[float, float, int]
        ``(alpha, scale, kernel_size)`` for each filter in bank order.
    n_kernels_ : int
        Number of filters in the bank.

    Examples
    --------
    >>> import numpy as np
    >>> from nirs4all.operators.transforms import FCKStaticTransformer
    >>> X = np.random.RandomState(0).randn(10, 200)
    >>> fck = FCKStaticTransformer().fit(X)
    >>> fck.transform(X).shape
    (10, 3200)
    """

    _webapp_meta = {
        "category": "feature-extraction",
        "tier": "experimental",
        "tags": ["fck", "fractional-derivative", "convolution", "feature-extraction"],
    }

    _stateless = True

    def __init__(
        self,
        alphas: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
        scales: Sequence[float] = (1, 2),
        kernel_sizes: Sequence[int] = (15, 31),
        sigma: float = 3.0,
        mode: str = "nearest",
        flatten: bool = True,
        *,
        copy: bool = True,
    ):
        self.alphas = alphas
        self.scales = scales
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.mode = mode
        self.flatten = flatten
        self.copy = copy

    def _build_bank(self) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        specs: list[tuple[float, float, int]] = []
        kernels: list[np.ndarray] = []
        for ks in self.kernel_sizes:
            if int(ks) != ks or ks < 3:
                raise ValueError(f"kernel_size must be an integer >= 3, got {ks}")
        for alpha in self.alphas:
            for scale in self.scales:
                if scale <= 0:
                    raise ValueError(f"scale must be > 0, got {scale}")
                for kernel_size in self.kernel_sizes:
                    kernels.append(_build_kernel(float(alpha), float(scale), int(kernel_size), float(self.sigma)))
                    specs.append((float(alpha), float(scale), int(kernel_size)))
        if not kernels:
            raise ValueError(
                "Empty filter bank: alphas, scales, and kernel_sizes must each be non-empty."
            )
        max_len = max(k.shape[0] for k in kernels)
        bank = np.zeros((len(kernels), max_len), dtype=np.float64)
        for i, k in enumerate(kernels):
            bank[i, : k.shape[0]] = k
        return bank, specs

    def fit(self, X, y=None):
        """Validate the input and build the (data-independent) filter bank.

        Building the bank in ``fit`` rather than ``__init__`` makes
        :func:`sklearn.base.clone` and ``set_params`` work as expected:
        the bank is rebuilt whenever hyperparameters change.
        """
        if scipy.sparse.issparse(X):
            raise TypeError("FCKStaticTransformer does not support scipy.sparse input")
        check_array(X, dtype=FLOAT_DTYPES, copy=False, ensure_2d=True)
        self.kernels_, self.kernel_specs_ = self._build_bank()
        self.n_kernels_ = self.kernels_.shape[0]
        return self

    def transform(self, X, y=None):
        """Apply each filter as a 1D convolution along the wavelength axis."""
        if scipy.sparse.issparse(X):
            raise TypeError("FCKStaticTransformer does not support scipy.sparse input")
        if not hasattr(self, "kernels_"):
            raise ValueError("FCKStaticTransformer must be fitted before transform.")
        X = check_array(X, dtype=FLOAT_DTYPES, copy=False, ensure_2d=True)
        n_samples, n_features = X.shape
        responses = np.empty((n_samples, self.n_kernels_, n_features), dtype=np.float64)
        for i, (_, _, ks) in enumerate(self.kernel_specs_):
            kernel = self.kernels_[i, :ks]
            responses[:, i, :] = convolve1d(X, kernel, axis=1, mode=self.mode)
        if self.flatten:
            return responses.reshape(n_samples, self.n_kernels_ * n_features)
        return responses

    def _more_tags(self):
        return {"allow_nan": False}
