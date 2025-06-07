import abc
import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES


class Augmenter(TransformerMixin, BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for data augmentation transformers."""

    def __init__(self, apply_on="samples", random_state=None, *, copy=True):
        """
        Initialize the augmenter.

        Parameters
        ----------
        apply_on : str
            The level at which augmentation is applied.
            Can be one of 'samples', 'features', 'subsets', or 'global'.
            Defaults to 'samples'.
        random_state : int or None
            Seed for the random number generator.
            If None, no random seed is set. Defaults to None.
        copy : bool
            Whether to make a copy of the input data.
            Defaults to True.
        """
        self.copy = copy
        self.apply_on = apply_on
        self.random_state = random_state
        self.random_gen = np.random.default_rng(random_state)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data and transform it.

        Parameters
        ----------
        X : array-like
            Input data to fit and transform.
        y : array-like or None
            Target variable (unused).
        **fit_params : dict
            Additional fitting parameters (unused).

        Returns
        -------
        array-like
            Transformed data.
        """
        return self.transform(X)

    def fit(self, X, y=None):
        """
        Fit to data.

        Parameters
        ----------
        X : array-like
            Input data to fit.
        y : array-like or None
            Target variable (unused).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    @abc.abstractmethod
    def augment(self, X, apply_on="samples"):
        """
        Perform data augmentation.

        Parameters
        ----------
        X : array-like
            Input data to augment.
        apply_on : str
            The level at which augmentation is applied.
            Can be one of 'samples', 'features', 'subsets', or 'global'.
            Defaults to 'samples'.

        Returns
        -------
        array-like
            Augmented data.
        """
        pass

    def transform(self, X):
        """
        Transform the input data by applying data augmentation.

        Parameters
        ----------
        X : array-like
            Input data to transform.

        Returns
        -------
        array-like
            Transformed data after augmentation.
        """
        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        return self.augment(X, self.apply_on)

    def _more_tags(self):
        """
        Provide additional tags for the estimator.

        Returns
        -------
        dict
            Additional tags.
        """
        return {"allow_nan": False}


class IdentityAugmenter(Augmenter):
    """An augmenter that returns the input data without any changes."""

    def augment(self, X, _):
        """
        Perform identity augmentation.

        Parameters
        ----------
        X : array-like
            Input data to augment.
        _ : str
            Placeholder for unused parameter.

        Returns
        -------
        array-like
            Augmented data (same as input data).
        """
        return X
