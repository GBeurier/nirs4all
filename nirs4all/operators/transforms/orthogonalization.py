"""Orthogonal Signal Correction (OSC) and External Parameter Orthogonalization (EPO).

This module provides advanced preprocessing methods for removing unwanted variation
from spectral data while preserving information relevant to the target variable.

Classes:
    OSC: Orthogonal Signal Correction (Direct OSC / DOSC)
    EPO: External Parameter Orthogonalization

References:
    - Wold, S., et al. (1998). Orthogonal signal correction of near-infrared spectra.
      Chemometrics and Intelligent Laboratory Systems, 44(1-2), 175-185.
    - Westerhuis, J.A., et al. (2001). Direct orthogonal signal correction.
      Chemometrics and Intelligent Laboratory Systems, 56(1), 13-25.
    - Roger, J.M., et al. (2003). EPO-PLS external parameter orthogonalisation of PLS
      application to temperature-independent measurement of sugar content of intact fruits.
      Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.
"""

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class OSC(TransformerMixin, BaseEstimator):
    """Orthogonal Signal Correction (Direct OSC / DOSC).

    Removes systematic variation in X that is orthogonal (unrelated) to Y, improving
    model interpretability and potentially prediction performance. OSC filters out
    variation unrelated to the target while preserving Y-predictive information.

    This implementation uses the Direct OSC (DOSC) algorithm, which is mathematically
    more rigorous than the original iterative Wold's OSC. DOSC provides a theoretically
    exact solution using simple least squares operations.

    Scientific Basis:
        OSC is particularly useful for NIRS data where systematic variations (temperature,
        scattering effects, instrumental drift) can obscure chemical information. By removing
        Y-orthogonal variation, OSC can:

        - Reduce model complexity (fewer PLS components needed)
        - Improve model interpretability (focus on Y-relevant features)
        - Enhance calibration transfer between instruments
        - Potentially improve prediction performance (though not guaranteed)

    Algorithm:
        1. Center and scale X and Y
        2. Compute Y-predictive direction: w_pls = X^T @ y / ||X^T @ y||
        3. For each orthogonal component:
           a. Compute PLS direction on residual X
           b. Orthogonalize to Y-direction using Gram-Schmidt projection
           c. Compute scores and loadings
           d. Deflate X by removing orthogonal component
        4. Store orthogonal weights and loadings for transform

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove. Typical values are 1-3.
        Larger values remove more variation but risk removing Y-relevant information.
        Use cross-validation to select optimal value.
    scale : bool, default=True
        Whether to center and scale X and Y before OSC. Recommended for spectral data.
    method : str, default='dosc'
        OSC algorithm variant. Currently only 'dosc' (Direct OSC) is supported.
        Reserved for future extensions (e.g., 'wold', 'posc').
    copy : bool, default=True
        Whether to copy input data. If False, input arrays may be modified.

    Attributes
    ----------
    W_ortho_ : ndarray of shape (n_features, n_components)
        Orthogonal weight vectors. These define the directions in X space that are
        orthogonal to Y and contain systematic variation to be removed.
    P_ortho_ : ndarray of shape (n_features, n_components)
        Orthogonal loading vectors. These define how the orthogonal scores relate
        back to the original features.
    X_mean_ : ndarray of shape (n_features,)
        Mean of X from training data (if scale=True).
    X_std_ : ndarray of shape (n_features,)
        Standard deviation of X from training data (if scale=True).
    y_mean_ : float or ndarray
        Mean of y from training data (if scale=True).
    y_std_ : float or ndarray
        Standard deviation of y from training data (if scale=True).
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used (may be less than requested if limited by data dimensions).

    References
    ----------
    .. [1] Wold, S., Antti, H., Lindgren, F., & Öhman, J. (1998). Orthogonal signal
           correction of near-infrared spectra. Chemometrics and Intelligent Laboratory
           Systems, 44(1-2), 175-185.
    .. [2] Westerhuis, J.A., de Jong, S., & Smilde, A.K. (2001). Direct orthogonal signal
           correction. Chemometrics and Intelligent Laboratory Systems, 56(1), 13-25.
    .. [3] Fearn, T. (2000). On orthogonal signal correction. Chemometrics and Intelligent
           Laboratory Systems, 50(1), 47-52.

    Examples
    --------
    >>> from nirs4all.operators.transforms import OSC
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, :5].mean(axis=1) + np.random.randn(100) * 0.1
    >>> # Apply OSC
    >>> osc = OSC(n_components=2)
    >>> X_filtered = osc.fit_transform(X, y)
    >>> # Use in pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ('osc', OSC(n_components=2)),
    ...     ('pls', PLSRegression(n_components=5))
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[...])

    Notes
    -----
    - OSC is a supervised preprocessing method that requires Y during fit
    - OSC does not guarantee improved prediction performance; use cross-validation
    - Removing too many components can remove Y-relevant information
    - For temperature/batch correction with known external parameters, consider EPO instead
    """

    _webapp_meta = {
        "category": "orthogonalization",
        "tier": "advanced",
        "tags": ["osc", "orthogonalization", "supervised", "dosc", "preprocessing"],
    }

    _stateless = False  # Learns from data (stores W_ortho_, P_ortho_)

    def __init__(self, n_components: int = 1, scale: bool = True, method: str = "dosc", *, copy: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.method = method
        self.copy = copy

    def _reset(self):
        """Reset fitted attributes."""
        attrs = ["W_ortho_", "P_ortho_", "X_mean_", "X_std_", "y_mean_", "y_std_", "n_features_in_", "n_components_"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def fit(self, X, y):
        """Fit OSC filter to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra or feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. OSC uses Y to determine which variation to preserve.

        Returns
        -------
        self : OSC
            Fitted transformer instance.

        Raises
        ------
        TypeError
            If X is a sparse matrix (not supported).
        ValueError
            If n_components is invalid or data dimensions are incompatible.
        """
        self._reset()

        # Validate input
        if scipy.sparse.issparse(X):
            raise TypeError("OSC does not support scipy.sparse input")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: X has {X.shape[0]} samples, y has {y.shape[0]}")

        self.n_features_in_ = n_features

        # Limit n_components by data dimensions
        max_components = min(n_features - 1, n_samples - 2)
        if self.n_components > max_components:
            self.n_components_ = max_components
        else:
            self.n_components_ = self.n_components

        if self.n_components_ < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")

        # Center and scale X
        if self.scale:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_ = np.std(X, axis=0, ddof=1)
            self.X_std_[self.X_std_ < 1e-10] = 1.0  # Avoid division by zero
            X_centered = (X - self.X_mean_) / self.X_std_
        else:
            self.X_mean_ = np.zeros(n_features)
            self.X_std_ = np.ones(n_features)
            X_centered = X.copy() if self.copy else X

        # Center and scale Y
        if self.scale:
            self.y_mean_ = np.mean(y, axis=0)
            self.y_std_ = np.std(y, axis=0, ddof=1)
            self.y_std_[self.y_std_ < 1e-10] = 1.0
            y_centered = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = np.zeros(n_targets)
            self.y_std_ = np.ones(n_targets)
            y_centered = y.copy()

        # Use first target if multiple targets
        y_vec = y_centered[:, 0]

        # Compute initial Y-predictive direction (PLS weight)
        w_pls = X_centered.T @ y_vec
        w_pls_norm = np.linalg.norm(w_pls)
        if w_pls_norm < 1e-10:
            raise ValueError("Y has no correlation with X; OSC cannot be applied")
        w_pls = w_pls / w_pls_norm

        # Initialize storage for orthogonal components
        self.W_ortho_ = np.zeros((n_features, self.n_components_), dtype=np.float64)
        self.P_ortho_ = np.zeros((n_features, self.n_components_), dtype=np.float64)

        # Residual matrix (will be deflated)
        X_res = X_centered.copy()

        # Extract orthogonal components iteratively
        for i in range(self.n_components_):
            # Compute PLS weight and loading for current residual
            w = X_res.T @ y_vec
            w_norm = np.linalg.norm(w)

            if w_norm < 1e-10:
                # No more variance to explain
                break

            w = w / w_norm

            # Compute PLS scores and loadings
            t = X_res @ w
            t_norm_sq = np.dot(t, t)

            if t_norm_sq < 1e-10:
                break

            p = X_res.T @ t / t_norm_sq

            # Orthogonalize loading p to Y-predictive direction using Gram-Schmidt
            # This is the key step: we orthogonalize the loading, not the weight
            projection = np.dot(p, w_pls) * w_pls
            w_ortho = p - projection

            w_ortho_norm = np.linalg.norm(w_ortho)
            if w_ortho_norm < 1e-10:
                # p is parallel to w_pls (no orthogonal component)
                break

            w_ortho = w_ortho / w_ortho_norm

            # Compute orthogonal scores using orthogonalized weight
            t_ortho = X_res @ w_ortho

            # Compute orthogonal loadings
            t_ortho_norm_sq = np.dot(t_ortho, t_ortho)
            if t_ortho_norm_sq < 1e-10:
                break

            p_ortho = X_res.T @ t_ortho / t_ortho_norm_sq

            # Deflate X by removing orthogonal component
            X_res = X_res - np.outer(t_ortho, p_ortho)

            # Store weights and loadings
            self.W_ortho_[:, i] = w_ortho
            self.P_ortho_[:, i] = p_ortho

        return self

    def transform(self, X):
        """Apply OSC filter to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectra or feature matrix to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            OSC-filtered data with orthogonal components removed.

        Raises
        ------
        NotFittedError
            If transform is called before fit.
        ValueError
            If X has incompatible number of features.
        """
        check_is_fitted(self, ["W_ortho_", "P_ortho_"])

        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but OSC was fitted with {self.n_features_in_} features")

        # Copy if requested
        X_out = X.copy() if self.copy else X

        # Center and scale
        if self.scale:
            X_out = (X_out - self.X_mean_) / self.X_std_

        # Apply orthogonal deflation for each component
        for i in range(self.n_components_):
            t_ortho = X_out @ self.W_ortho_[:, i]
            X_out = X_out - np.outer(t_ortho, self.P_ortho_[:, i])

        # Unscale back to original scale
        if self.scale:
            X_out = X_out * self.X_std_ + self.X_mean_

        return X_out

    def fit_transform(self, X, y):
        """Fit OSC and transform X in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            OSC-filtered training data.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"requires_y": True, "allow_nan": False}

class EPO(TransformerMixin, BaseEstimator):
    """External Parameter Orthogonalization (EPO).

    Removes variation in X that is correlated with external parameters (e.g., temperature,
    humidity, batch effects, instrumental drift) without using the target variable Y. EPO
    is useful when you know specific sources of unwanted variation and want to remove them
    while retaining spectral directions orthogonal to the learned interference.

    Scientific Basis:
        EPO differs from OSC in that it uses external parameter measurements rather than
        the target Y. This makes EPO a semi-supervised method that can remove known
        interference sources. Signal sharing those spectral directions is also removed.

        Common applications include:
        - Temperature-independent sugar content measurement in fruits
        - Removing batch effects in manufacturing processes
        - Correcting for instrumental drift over time
        - Handling moisture effects in soil spectroscopy

    Algorithm:
        1. Center X and external parameter d
        2. Regress X on d to estimate interference loadings in feature space
        3. Store the orthogonal complement of those loadings
        4. Transform applies learned orthogonalization without needing d

    Parameters
    ----------
    scale : bool, default=True
        Whether to center X and external parameter before EPO.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    projection_ : ndarray of shape (n_features, n_features)
        Orthogonal projector onto the complement of the fitted interference directions.
    projection_coefs_ : ndarray
        Regression coefficients for orthogonalizing each feature against external parameter(s).
    X_mean_ : ndarray of shape (n_features,)
        Mean of X from training data (if scale=True).
    d_mean_ : float or ndarray
        Mean of external parameter(s) from training data (if scale=True).
    n_features_in_ : int
        Number of features seen during fit.

    References
    ----------
    .. [1] Roger, J.M., Chauchard, F., & Bellon-Maurel, V. (2003). EPO-PLS external
           parameter orthogonalisation of PLS application to temperature-independent
           measurement of sugar content of intact fruits. Chemometrics and Intelligent
           Laboratory Systems, 66(2), 191-204.

    Examples
    --------
    >>> from nirs4all.operators.transforms import EPO
    >>> import numpy as np
    >>> # Generate synthetic data with temperature effect
    >>> X = np.random.randn(100, 50)
    >>> temperature = np.random.randn(100)
    >>> X = X + np.outer(temperature, np.ones(50)) * 0.5  # Add temperature effect
    >>> # Apply EPO to remove temperature effect
    >>> epo = EPO()
    >>> X_filtered = epo.fit_transform(X, temperature)
    >>> # Temperature effect is removed

    Notes
    -----
    - EPO uses external parameter during fit, not Y
    - The second argument to fit() is the external parameter (d), not the target (y)
    - Variation sharing the interference subspace is removed even when it is Y-related
    - For purely Y-orthogonal variation removal, use OSC instead
    """

    _webapp_meta = {
        "category": "orthogonalization",
        "tier": "advanced",
        "tags": ["epo", "orthogonalization", "batch-correction", "external-parameter", "preprocessing"],
    }

    _stateless = False  # Learns from data

    def __init__(self, scale: bool = True, *, copy: bool = True):
        self.scale = scale
        self.copy = copy

    def _reset(self):
        """Reset fitted attributes."""
        attrs = ["projection_coefs_", "projection_", "X_mean_", "d_mean_", "n_features_in_"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def fit(self, X, d):
        """Fit EPO filter using external parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra or feature matrix.
        d : array-like of shape (n_samples,) or (n_samples, n_params)
            External parameter(s) to orthogonalize against (e.g., temperature, batch ID).
            This is NOT the target variable Y.

        Returns
        -------
        self : EPO
            Fitted transformer instance.

        Raises
        ------
        TypeError
            If X is a sparse matrix.
        ValueError
            If X and d have incompatible shapes.
        """
        self._reset()

        # Validate input
        if scipy.sparse.issparse(X):
            raise TypeError("EPO does not support scipy.sparse input")

        X = np.asarray(X, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)

        if d.ndim == 1:
            d = d.reshape(-1, 1)

        n_samples, n_features = X.shape

        if X.shape[0] != d.shape[0]:
            raise ValueError(f"X and d have incompatible shapes: X has {X.shape[0]} samples, d has {d.shape[0]}")

        self.n_features_in_ = n_features

        # Center X
        if self.scale:
            self.X_mean_ = np.mean(X, axis=0)
            X_centered = X - self.X_mean_
        else:
            self.X_mean_ = np.zeros(n_features)
            X_centered = X.copy() if self.copy else X

        # Center external parameter
        if self.scale:
            self.d_mean_ = np.mean(d, axis=0)
            d_centered = d - self.d_mean_
        else:
            self.d_mean_ = np.zeros(d.shape[1])
            d_centered = d.copy()

        self.projection_coefs_ = np.linalg.lstsq(d_centered, X_centered, rcond=None)[0]
        # The row space of the regression loadings is the learned spectral
        # interference subspace. Its complement can be applied without d.
        _, singular_values, directions = np.linalg.svd(self.projection_coefs_, full_matrices=False)
        tolerance = max(self.projection_coefs_.shape) * np.finfo(float).eps * singular_values.max(initial=0.0)
        directions = directions[singular_values > tolerance]
        self.projection_ = np.eye(n_features) - directions.T @ directions

        return self

    def transform(self, X):
        """Apply EPO filter to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectra or feature matrix to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            EPO-filtered data with external parameter effects removed.

        Raises
        ------
        NotFittedError
            If transform is called before fit.
        ValueError
            If X has incompatible number of features.

        Notes
        -----
        The transform applies the learned orthogonalization without requiring
        the external parameter. The projection learned during fit is applied
        to remove the systematic variation pattern identified in the calibration set.
        """
        check_is_fitted(self, ["projection_", "X_mean_"])
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but EPO was fitted with {self.n_features_in_} features")
        X_out = X.copy() if self.copy else X
        X_out -= self.X_mean_
        X_out[:] = X_out @ self.projection_
        X_out += self.X_mean_
        return X_out

    def fit_transform(self, X, d):
        """Fit on external parameters and apply the reusable spectral projection."""
        return self.fit(X, d).transform(X)

    def _more_tags(self):
        return {"requires_y": False, "allow_nan": False}
