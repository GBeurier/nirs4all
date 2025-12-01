"""Orthogonal PLS Discriminant Analysis (OPLS-DA) classifier for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .plsda import PLSDA

def _check_pyopls_available():
    """Check if pyopls package is available."""
    try:
        import pyopls
        return True
    except ImportError:
        return False

class OPLSDA(BaseEstimator, ClassifierMixin):
    """Orthogonal PLS Discriminant Analysis (OPLS-DA) classifier.

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingClassifier)
    _estimator_type = "classifier"

    OPLS-DA combines OPLS filtering with PLS-DA classification.
    It removes Y-orthogonal variation from X before applying PLS-DA,
    improving class separation and model interpretability.

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove.
    pls_components : int, default=5
        Number of PLS components for the discriminant model.
    scale : bool, default=True
        Whether to scale X before fitting.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    opls_ : pyopls.OPLS
        Fitted OPLS transformer.
    plsda_ : PLSDA
        Fitted PLS-DA model on filtered data.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import OPLSDA
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=50, n_classes=2,
    ...                            n_informative=10, random_state=42)
    >>> model = OPLSDA(n_components=1, pls_components=5)
    >>> model.fit(X, y)
    OPLSDA(n_components=1, pls_components=5)
    >>> predictions = model.predict(X)

    Notes
    -----
    Requires the `pyopls` package: ``pip install pyopls``

    See Also
    --------
    PLSDA : Standard PLS-DA without orthogonal filtering.
    OPLS : OPLS for regression tasks.

    References
    ----------
    .. [1] Bylesjö, M., et al. (2006). OPLS discriminant analysis: combining
           the strengths of PLS-DA and SIMCA classification. Journal of
           Chemometrics, 20(8-10), 341-351.
    """

    def __init__(
        self,
        n_components: int = 1,
        pls_components: int = 5,
        scale: bool = True,
    ):
        """Initialize OPLSDA classifier.

        Parameters
        ----------
        n_components : int, default=1
            Number of orthogonal components to remove.
        pls_components : int, default=5
            Number of PLS components for the discriminant model.
        scale : bool, default=True
            Whether to scale X before fitting.
        """
        self.n_components = n_components
        self.pls_components = pls_components
        self.scale = scale

    def fit(self, X, y):
        """Fit the OPLS-DA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : OPLSDA
            Fitted estimator.

        Raises
        ------
        ImportError
            If pyopls package is not installed.
        """
        if not _check_pyopls_available():
            raise ImportError(
                "pyopls package is required for OPLSDA. "
                "Install it with: pip install pyopls"
            )

        from pyopls import OPLS as PyOPLS

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Encode y for OPLS fitting (use numeric encoding)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        # Limit components
        max_ortho = min(self.n_components, X.shape[1] - 1, X.shape[0] - 2)
        n_ortho = max(1, max_ortho)

        # Fit OPLS transformer
        self.opls_ = PyOPLS(n_components=n_ortho, scale=self.scale)
        X_filtered = self.opls_.fit_transform(X, y_encoded)

        # Fit PLS-DA on filtered data
        self.plsda_ = PLSDA(n_components=self.pls_components)
        self.plsda_.fit(X_filtered, y)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)

        # Transform X to remove orthogonal variation
        X_filtered = self.opls_.transform(X)

        # Predict with PLS-DA
        return self.plsda_.predict(X_filtered)

    def predict_proba(self, X):
        """Return pseudo-probabilities (PLS responses).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Pseudo-probability estimates.
        """
        X = np.asarray(X)

        # Transform X to remove orthogonal variation
        X_filtered = self.opls_.transform(X)

        # Get probabilities from PLS-DA
        return self.plsda_.predict_proba(X_filtered)

    def transform(self, X):
        """Transform X by removing orthogonal variation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            Transformed samples with orthogonal variation removed.
        """
        X = np.asarray(X)
        return self.opls_.transform(X)

    def get_params(self, deep=True):
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
            'pls_components': self.pls_components,
            'scale': self.scale,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : OPLSDA
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
