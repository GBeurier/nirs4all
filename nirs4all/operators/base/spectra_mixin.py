"""
SpectraTransformerMixin base class for wavelength-aware transformations.

This module provides a base class for spectral transformations that can use
wavelength information. The controller automatically provides wavelengths
from the dataset when available and when the operator declares it needs them.
"""

from abc import abstractmethod
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SpectraTransformerMixin(TransformerMixin, BaseEstimator):
    """
    Mixin for operators that can use wavelength information.

    Wavelengths are passed via kwargs for full sklearn compatibility::

        op.fit(X, wavelengths=wl)
        op.transform(X, wavelengths=wl)

    Set ``_requires_wavelengths`` to control behavior:

    - ``True``: wavelengths required, raise if not provided
    - ``False``: wavelengths ignored
    - ``"optional"``: use if provided, fallback to ``None`` otherwise

    Subclasses must implement ``_transform_impl()`` instead of ``transform()``.

    Parameters
    ----------
    None - this is a mixin class. Subclasses define their own parameters.

    Attributes
    ----------
    _requires_wavelengths : Union[bool, str]
        Class-level flag indicating whether this operator requires wavelengths.

    Examples
    --------
    >>> class TemperatureAugmenter(SpectraTransformerMixin):
    ...     def __init__(self, temperature_delta: float = 5.0):
    ...         self.temperature_delta = temperature_delta
    ...
    ...     def _transform_impl(
    ...         self, X: np.ndarray, wavelengths: np.ndarray
    ...     ) -> np.ndarray:
    ...         # Apply temperature-dependent spectral changes
    ...         # ... implementation ...
    ...         return X_transformed

    Notes
    -----
    The controller detects ``SpectraTransformerMixin`` instances via:

    .. code-block:: python

        needs_wavelengths = (
            isinstance(op, SpectraTransformerMixin) and
            getattr(op, '_requires_wavelengths', False)
        )

    Wavelengths are extracted from the dataset using ``dataset.wavelengths_nm(source)``.
    """

    _requires_wavelengths: Union[bool, str] = True

    def fit(self, X, y=None, **kwargs):
        """
        Fit the transformer. Override in subclass if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or None, default=None
            Target values (unused).
        **kwargs : dict
            Additional fit parameters. May include 'wavelengths' for
            operators that need to fit using wavelength information.

        Returns
        -------
        self : object
            Returns self.
        """
        wavelengths = kwargs.get('wavelengths')
        self._validate_wavelengths(wavelengths, X.shape[1])
        self._wavelengths = wavelengths
        return self

    def transform(self, X, **kwargs):
        """
        Transform X. Subclasses must implement _transform_impl.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectra array.
        **kwargs : dict
            Additional parameters. May include 'wavelengths' as ndarray
            of shape (n_features,) in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra array.

        Raises
        ------
        ValueError
            If wavelengths are not provided and _requires_wavelengths is True.
        """
        wavelengths = kwargs.get('wavelengths', getattr(self, '_wavelengths', None))
        self._validate_wavelengths(wavelengths, X.shape[1])
        return self._transform_impl(X, wavelengths)

    def _validate_wavelengths(self, wavelengths, n_features):
        """
        Validate wavelengths against the requirement and feature count.

        Parameters
        ----------
        wavelengths : ndarray or None
            Wavelength array to validate.
        n_features : int
            Expected number of features.

        Raises
        ------
        ValueError
            If wavelengths are required but not provided, or if length mismatches.
        """
        if self._requires_wavelengths is True and wavelengths is None:
            raise ValueError(
                f"{self.__class__.__name__} requires wavelengths but none were provided. "
                "Ensure the dataset has wavelength headers or pass wavelengths explicitly."
            )
        if wavelengths is not None and len(wavelengths) != n_features:
            raise ValueError(
                f"wavelengths length {len(wavelengths)} != features {n_features}"
            )

    @abstractmethod
    def _transform_impl(self, X: np.ndarray, wavelengths) -> np.ndarray:
        """
        Implement transformation logic. wavelengths may be None if optional.

        Subclasses must implement this method to perform the actual transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,) or None
            Wavelength array in nm. May be None if _requires_wavelengths
            is False or "optional".

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _transform_impl()"
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
