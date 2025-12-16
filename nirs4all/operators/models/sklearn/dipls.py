"""Dynamic PLS (DiPLS) regressor for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_trendfitter_available():
    """Check if trendfitter package is available."""
    try:
        import trendfitter
        return True
    except ImportError:
        return False

class DiPLS(BaseEstimator, RegressorMixin):
    """Dynamic PLS (DiPLS) regressor.

    DiPLS extends PLS to handle dynamic systems by including time-lagged
    variables. It uses the `trendfitter` package.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent variables to extract.
    lags : int, default=1
        Number of time lags to consider (s parameter in DiPLS).
    cv_splits : int, default=7
        Number of cross-validation splits for automatic component selection.
    tol : float, default=1e-8
        Convergence tolerance.
    max_iter : int, default=1000
        Maximum number of iterations.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import DiPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = DiPLS(n_components=5, lags=2)
    >>> model.fit(X, y)
    DiPLS(n_components=5, lags=2)
    >>> predictions = model.predict(X)

    Notes
    -----
    Requires the `trendfitter` package: ``pip install trendfitter``

    DiPLS is particularly useful for:
    - Process monitoring with temporal dependencies
    - NIR data collected over time
    - Batch process analytics

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard PLS without dynamics.

    References
    ----------
    .. [1] Dong, Y., & Qin, S. J. (2018). A novel dynamic PLS soft sensor
           based on moving-window modeling. Chemical Engineering Research
           and Design, 131, 509-519.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 5,
        lags: int = 1,
        cv_splits: int = 7,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ):
        """Initialize DiPLS regressor.

        Parameters
        ----------
        n_components : int, default=5
            Number of latent variables to extract.
        lags : int, default=1
            Number of time lags to consider.
        cv_splits : int, default=7
            Number of cross-validation splits.
        tol : float, default=1e-8
            Convergence tolerance.
        max_iter : int, default=1000
            Maximum number of iterations.
        """
        self.n_components = n_components
        self.lags = lags
        self.cv_splits = cv_splits
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit the DiPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (time-ordered measurements).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : DiPLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If trendfitter package is not installed.
        """
        if not _check_trendfitter_available():
            raise ImportError(
                "trendfitter package is required for DiPLS. "
                "Install it with: pip install trendfitter"
            )

        from trendfitter.models import DiPLS as TFDiPLS

        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Create and fit trendfitter DiPLS
        self._model = TFDiPLS(
            cv_splits_number=self.cv_splits,
            tol=self.tol,
            loop_limit=self.max_iter,
        )

        # Fit with specified components and lags
        self._model.fit(
            X, y,
            latent_variables=self.n_components,
            s=self.lags,
        )

        self.n_components_ = self.n_components

        return self

    def predict(self, X):
        """Predict using the DiPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.

        Notes
        -----
        DiPLS uses Hankelization which may produce fewer predictions than
        input samples. This implementation pads the beginning with the first
        predicted value to maintain compatibility with sklearn cross-validation.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        y_pred = self._model.predict(X)

        # DiPLS may return fewer predictions due to Hankelization
        # Pad beginning with first prediction to match input length for sklearn compatibility
        n_pred = y_pred.shape[0]
        if n_pred < n_samples:
            n_pad = n_samples - n_pred
            if y_pred.ndim == 1:
                pad_value = y_pred[0]
                y_pred = np.concatenate([np.full(n_pad, pad_value), y_pred])
            else:
                pad_value = y_pred[0:1]
                y_pred = np.concatenate([np.tile(pad_value, (n_pad, 1)), y_pred], axis=0)

        # Flatten if single target
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

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
            'lags': self.lags,
            'cv_splits': self.cv_splits,
            'tol': self.tol,
            'max_iter': self.max_iter,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DiPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
