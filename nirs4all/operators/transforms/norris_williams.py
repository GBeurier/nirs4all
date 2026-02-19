import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin


def _segment_smooth(spectra: np.ndarray, segment: int) -> np.ndarray:
    """Apply centered moving average smoothing along the feature axis.

    Args:
        spectra (numpy.ndarray): Data matrix (n_samples, n_features).
        segment (int): Smoothing window size (must be odd, >= 1).

    Returns:
        numpy.ndarray: Smoothed spectra with same shape as input.
    """
    if segment <= 1:
        return spectra
    half = segment // 2
    n_features = spectra.shape[1]
    padded = np.pad(spectra, ((0, 0), (half, half)), mode="edge")
    smoothed = np.empty_like(spectra)
    for j in range(n_features):
        smoothed[:, j] = np.mean(padded[:, j:j + segment], axis=1)
    return smoothed

def _gap_derivative(spectra: np.ndarray, gap: int, delta: float) -> np.ndarray:
    """Compute gap derivative along the feature axis.

    Args:
        spectra (numpy.ndarray): Data matrix (n_samples, n_features).
        gap (int): Gap size in data points.
        delta (float): Sampling interval.

    Returns:
        numpy.ndarray: Gap derivative with same shape as input.
    """
    padded = np.pad(spectra, ((0, 0), (gap, gap)), mode="edge")
    return (padded[:, 2 * gap:] - padded[:, :spectra.shape[1]]) / (2 * gap * delta)

def norris_williams(spectra: np.ndarray, gap: int = 5, segment: int = 5, deriv: int = 1, delta: float = 1.0) -> np.ndarray:
    """Norris-Williams gap derivative for spectral data.

    Applies segment smoothing followed by gap differentiation, a standard
    preprocessing technique in NIR spectroscopy.

    Args:
        spectra (numpy.ndarray): NIRS data matrix (n_samples, n_features).
        gap (int): Gap size in data points.
        segment (int): Segment size for smoothing (must be odd, >= 1; 1 = no smoothing).
        deriv (int): Derivative order (1 or 2).
        delta (float): Sampling interval.

    Returns:
        numpy.ndarray: Derivative spectra with same shape as input.

    Raises:
        ValueError: If deriv is not 1 or 2, or segment is not odd.
    """
    if deriv not in (1, 2):
        raise ValueError(f"deriv must be 1 or 2, got {deriv}")
    if segment < 1 or segment % 2 == 0:
        raise ValueError(f"segment must be odd and >= 1, got {segment}")

    result = spectra
    for _ in range(deriv):
        result = _segment_smooth(result, segment)
        result = _gap_derivative(result, gap, delta)
    return result

class NorrisWilliams(TransformerMixin, BaseEstimator):
    """Norris-Williams gap derivative transform.

    Applies segment smoothing followed by gap differentiation, a standard
    preprocessing technique in NIR spectroscopy for computing smooth
    derivatives that are more robust to noise than simple finite differences.

    Parameters
    ----------
    gap : int, default=5
        Gap size in data points for the derivative computation.
    segment : int, default=5
        Segment size for smoothing (must be odd, >= 1; 1 = no smoothing).
    deriv : int, default=1
        Derivative order (1 or 2).
    delta : float, default=1.0
        Sampling interval.
    copy : bool, default=True
        Whether to copy input data.

    References
    ----------
    Norris, K.H. and Williams, P.C. (1984). Optimization of mathematical
    treatments of raw near-infrared signal in the measurement of protein
    in hard red spring wheat. Cereal Chemistry, 61, 158-165.
    """

    _webapp_meta = {
        "category": "derivatives",
        "tier": "standard",
        "tags": ["derivative", "gap-derivative", "norris-williams", "smoothing"],
    }

    _stateless = True

    def __init__(self, gap: int = 5, segment: int = 5, deriv: int = 1, delta: float = 1.0, *, copy: bool = True):
        self.gap = gap
        self.segment = segment
        self.deriv = deriv
        self.delta = delta
        self.copy = copy

    def _reset(self):
        pass

    def fit(self, X, y=None):
        """Validate input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
        y : None
            Ignored.

        Raises
        ------
        ValueError
            If the input X is a sparse matrix, or parameters are invalid.

        Returns
        -------
        NorrisWilliams
            The fitted object.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("NorrisWilliams does not support scipy.sparse input")
        if self.deriv not in (1, 2):
            raise ValueError(f"deriv must be 1 or 2, got {self.deriv}")
        if self.segment < 1 or self.segment % 2 == 0:
            raise ValueError(f"segment must be odd and >= 1, got {self.segment}")
        return self

    def transform(self, X, copy=None):
        """Apply Norris-Williams gap derivative to the data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
        copy : bool or None, optional
            Whether to copy the input data.

        Returns
        -------
        numpy.ndarray
            The transformed data.
        """
        if scipy.sparse.issparse(X):
            raise ValueError("Sparse matrices not supported!")
        return norris_williams(X, gap=self.gap, segment=self.segment, deriv=self.deriv, delta=self.delta)

    def _more_tags(self):
        return {"allow_nan": False}
