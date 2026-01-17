"""
SpectraTransformerMixin base class for wavelength-aware transformations.

This module provides a base class for spectral transformations that require
wavelength information. The controller automatically provides wavelengths
from the dataset when available and when the operator declares it needs them.
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    """
    Base class for spectral transformations that require wavelength information.

    This mixin extends sklearn's TransformerMixin to support wavelength-aware
    transformations. The controller automatically provides wavelengths from the
    dataset when available and when the operator declares it needs them.

    Subclasses must implement `transform_with_wavelengths()` instead of `transform()`.

    Parameters
    ----------
    None - this is a mixin class. Subclasses define their own parameters.

    Attributes
    ----------
    _requires_wavelengths : bool
        Class-level flag indicating whether this operator requires wavelengths.
        If True (default), transform() will raise ValueError if wavelengths
        are not provided. Subclasses can set this to False if wavelengths
        are optional.

    Examples
    --------
    >>> class TemperatureAugmenter(SpectraTransformerMixin):
    ...     def __init__(self, temperature_delta: float = 5.0):
    ...         self.temperature_delta = temperature_delta
    ...
    ...     def transform_with_wavelengths(
    ...         self, X: np.ndarray, wavelengths: np.ndarray
    ...     ) -> np.ndarray:
    ...         # Apply temperature-dependent spectral changes
    ...         # ... implementation ...
    ...         return X_transformed

    Notes
    -----
    The controller detects `SpectraTransformerMixin` instances via:

    .. code-block:: python

        needs_wavelengths = (
            isinstance(op, SpectraTransformerMixin) and
            getattr(op, '_requires_wavelengths', False)
        )

    Wavelengths are extracted from the dataset using `dataset.wavelengths_nm(source)`.
    """

    _requires_wavelengths: bool = True

    def fit(self, X, y=None, **fit_params):
        """
        Fit is a no-op for most spectral transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or None, default=None
            Target values (unused).
        **fit_params : dict
            Additional fit parameters. May include 'wavelengths' for
            operators that need to fit using wavelength information.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X, wavelengths: Optional[np.ndarray] = None):
        """
        Transform method that delegates to transform_with_wavelengths.

        If wavelengths are not provided and the operator requires them,
        this will raise a ValueError.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra array.
        wavelengths : ndarray of shape (n_features,) or None, default=None
            Wavelength array in nm. Required if _requires_wavelengths is True.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra array.

        Raises
        ------
        ValueError
            If wavelengths are not provided and _requires_wavelengths is True.
        """
        if wavelengths is None and self._requires_wavelengths:
            raise ValueError(
                f"{self.__class__.__name__} requires wavelengths but none were provided. "
                "Ensure the dataset has wavelength headers or pass wavelengths explicitly."
            )
        return self.transform_with_wavelengths(X, wavelengths)

    @abstractmethod
    def transform_with_wavelengths(
        self, X: np.ndarray, wavelengths: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply the transformation using wavelength information.

        Subclasses must implement this method to perform the actual transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,) or None
            Wavelength array in nm. May be None if _requires_wavelengths is False.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform_with_wavelengths()"
        )

    def _more_tags(self):
        """
        Provide additional sklearn estimator tags.

        Returns
        -------
        dict
            Additional tags for sklearn compatibility.
        """
        return {"allow_nan": False, "requires_wavelengths": self._requires_wavelengths}
