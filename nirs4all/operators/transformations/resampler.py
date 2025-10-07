"""
Wavelength resampling operators for NIRS spectral data.

This module provides resampling functionality to interpolate spectral data
to new wavelength grids using various scipy interpolation methods.
"""

import numpy as np
from typing import Optional, Union, Tuple, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import warnings


class Resampler(TransformerMixin, BaseEstimator):
    """
    Resample spectral data to new wavelength grid using interpolation.

    This transformer interpolates NIRS spectral data from the original wavelength
    grid to a target wavelength grid using scipy interpolation methods.

    Parameters
    ----------
    target_wavelengths : array-like
        Target wavelengths for resampling. Must be 1D array.
    method : str, default='linear'
        Interpolation method. Supported methods:
        - 'linear': Linear interpolation
        - 'nearest': Nearest neighbor interpolation
        - 'cubic': Cubic spline interpolation
        - 'quadratic': Quadratic spline interpolation
        - 'slinear': Linear spline (order 1)
        - 'zero': Zero-order spline (piecewise constant)
        Future: May support additional scipy methods
    crop_range : tuple of (float, float) or None, default=None
        Optional (min_wavelength, max_wavelength) to crop original data before resampling.
    fill_value : float or 'extrapolate', default=0.0
        Value to use for target wavelengths outside the original range.
        - float: Use this constant value for extrapolation
        - 'extrapolate': Extrapolate using the interpolation method
        - 0.0: Default padding with zeros (safe choice)
    bounds_error : bool, default=False
        If True, raise error when target wavelengths are outside original range.
        If False, use fill_value for out-of-bounds points.
    copy : bool, default=True
        Whether to copy input data or modify in place.

    Attributes
    ----------
    original_wavelengths_ : ndarray of shape (n_features,)
        Original wavelength grid from fit data
    n_features_in_ : int
        Number of features (wavelengths) in input data
    n_features_out_ : int
        Number of features (wavelengths) in output data
    interpolator_params_ : dict
        Stored interpolation parameters for reconstruction

    Examples
    --------
    >>> from nirs4all.operators.transformations import Resampler
    >>> import numpy as np
    >>>
    >>> # Original data at 1000-2500 nm with 200 points
    >>> X = np.random.randn(100, 200)
    >>> original_wl = np.linspace(1000, 2500, 200)
    >>>
    >>> # Resample to 100 evenly-spaced wavelengths
    >>> target_wl = np.linspace(1000, 2500, 100)
    >>> resampler = Resampler(target_wavelengths=target_wl, method='cubic')
    >>> resampler.fit(X, wavelengths=original_wl)
    >>> X_resampled = resampler.transform(X)
    >>> X_resampled.shape
    (100, 100)

    Notes
    -----
    - Wavelengths must be strictly increasing
    - Warns if target wavelengths extend beyond original range
    - Raises error if no wavelengths overlap between original and target
    """

    def __init__(
        self,
        target_wavelengths: np.ndarray,
        method: Literal['linear', 'nearest', 'cubic', 'quadratic', 'slinear', 'zero'] = 'linear',
        crop_range: Optional[Tuple[float, float]] = None,
        fill_value: Union[float, str] = 0.0,
        bounds_error: bool = False,
        copy: bool = True
    ):
        self.target_wavelengths = target_wavelengths
        self.method = method
        self.crop_range = crop_range
        self.fill_value = fill_value
        self.bounds_error = bounds_error
        self.copy = copy

    def _validate_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Validate and convert wavelengths to 1D float array."""
        wavelengths = np.asarray(wavelengths, dtype=float)

        if wavelengths.ndim != 1:
            raise ValueError(f"Wavelengths must be 1D array, got shape {wavelengths.shape}")

        # Check if strictly monotonic (increasing or decreasing)
        diffs = np.diff(wavelengths)
        is_increasing = np.all(diffs > 0)
        is_decreasing = np.all(diffs < 0)

        if not (is_increasing or is_decreasing):
            # Provide detailed error message
            non_monotonic_indices = np.where(diffs[:-1] * diffs[1:] <= 0)[0]
            raise ValueError(
                f"Wavelengths must be strictly monotonic (all increasing or all decreasing). "
                f"Found non-monotonic behavior at indices: {non_monotonic_indices[:5]}. "
                f"Wavelengths range: [{wavelengths.min():.2f}, {wavelengths.max():.2f}]. "
                f"First 10 values: {wavelengths[:10]}. "
                f"First 10 diffs: {diffs[:10]}"
            )

        return wavelengths

    def _check_wavelength_overlap(self, original_wl: np.ndarray, target_wl: np.ndarray):
        """Check overlap between original and target wavelengths."""
        orig_min, orig_max = original_wl.min(), original_wl.max()
        target_min, target_max = target_wl.min(), target_wl.max()

        # Check if there's any overlap
        if target_max < orig_min or target_min > orig_max:
            raise ValueError(
                f"No overlap between original wavelengths ({orig_min:.1f}-{orig_max:.1f}) "
                f"and target wavelengths ({target_min:.1f}-{target_max:.1f})"
            )

        # Warn if target extends beyond original range
        if target_min < orig_min or target_max > orig_max:
            out_of_bounds = []
            if target_min < orig_min:
                out_of_bounds.append(f"below {orig_min:.1f}")
            if target_max > orig_max:
                out_of_bounds.append(f"above {orig_max:.1f}")

            warnings.warn(
                f"Target wavelengths extend {' and '.join(out_of_bounds)} original range. "
                f"Using fill_value={self.fill_value} for extrapolation.",
                UserWarning
            )

    def fit(self, X, y=None, wavelengths: Optional[np.ndarray] = None):
        """
        Fit the resampler by storing original wavelength grid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
        wavelengths : array-like of shape (n_features,), optional
            Original wavelength grid. If None, will be extracted from dataset headers
            by the controller.

        Returns
        -------
        self : Resampler
            Fitted resampler.
        """
        X = check_array(X, dtype=np.float64, force_all_finite='allow-nan', copy=self.copy)

        if wavelengths is None:
            raise ValueError(
                "Wavelengths must be provided to fit(). "
                "The controller should extract these from dataset.headers()."
            )

        # Validate wavelengths
        self.original_wavelengths_ = self._validate_wavelengths(wavelengths)

        if len(self.original_wavelengths_) != X.shape[1]:
            raise ValueError(
                f"Number of wavelengths ({len(self.original_wavelengths_)}) "
                f"must match number of features ({X.shape[1]})"
            )

        # Store original number of features before cropping
        original_n_features = X.shape[1]

        # Apply crop range if specified
        if self.crop_range is not None:
            crop_min, crop_max = self.crop_range
            self.crop_mask_ = (self.original_wavelengths_ >= crop_min) & (self.original_wavelengths_ <= crop_max)

            # Store info about whether we need to crop features (done by controller for raw data)
            # For preprocessed data, controller will pad with 0 instead
            self.wavelengths_after_crop_ = self.original_wavelengths_[self.crop_mask_]

            if len(self.wavelengths_after_crop_) == 0:
                raise ValueError(f"Crop range {self.crop_range} excludes all wavelengths")

            # Use cropped wavelengths for interpolation
            self.original_wavelengths_ = self.wavelengths_after_crop_
        else:
            self.crop_mask_ = None
            self.wavelengths_after_crop_ = self.original_wavelengths_

        # Validate target wavelengths
        target_wl = self._validate_wavelengths(self.target_wavelengths)

        # Check for overlap and warn if needed
        self._check_wavelength_overlap(self.original_wavelengths_, target_wl)

        # Store dimensions
        self.n_features_in_ = original_n_features  # Original before crop
        self.n_features_out_ = len(target_wl)

        # Store interpolation parameters for serialization
        self.interpolator_params_ = {
            'target_wavelengths': target_wl,
            'method': self.method,
            'fill_value': self.fill_value,
            'bounds_error': self.bounds_error
        }

        return self

    def transform(self, X):
        """
        Resample spectral data to target wavelength grid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectral data to resample. Should have same number of features as training data.

        Returns
        -------
        X_resampled : ndarray of shape (n_samples, n_features_out_)
            Resampled spectral data.
        """
        check_is_fitted(self, ['original_wavelengths_', 'interpolator_params_'])

        X = check_array(X, dtype=np.float64, force_all_finite='allow-nan', copy=self.copy)

        # Check that input matches the original shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but resampler was fitted with {self.n_features_in_} features"
            )

        # Apply crop if it was specified during fit
        if self.crop_mask_ is not None:
            X = X[:, self.crop_mask_]

        # Perform interpolation
        from scipy.interpolate import interp1d

        target_wl = self.interpolator_params_['target_wavelengths']
        n_samples = X.shape[0]
        X_resampled = np.zeros((n_samples, self.n_features_out_), dtype=X.dtype)

        for i in range(n_samples):
            # Create interpolator for this sample
            interpolator = interp1d(
                self.original_wavelengths_,
                X[i, :],
                kind=self.method,
                fill_value=self.fill_value,
                bounds_error=self.bounds_error,
                assume_sorted=False
            )

            # Interpolate to target wavelengths
            X_resampled[i, :] = interpolator(target_wl)

        return X_resampled

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (target wavelengths as strings).

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        feature_names_out : ndarray of str
            Target wavelengths as strings.
        """
        check_is_fitted(self, 'interpolator_params_')
        target_wl = self.interpolator_params_['target_wavelengths']
        return np.array([f"{wl:.2f}" for wl in target_wl])

    def __repr__(self):
        """String representation of the resampler."""
        if hasattr(self, 'n_features_in_'):
            return (f"Resampler(method='{self.method}', "
                    f"n_in={self.n_features_in_}, n_out={self.n_features_out_})")
        else:
            target_len = len(self.target_wavelengths) if hasattr(self.target_wavelengths, '__len__') else '?'
            return f"Resampler(method='{self.method}', n_out={target_len}, unfitted)"
