"""
Feature selection operators for NIRS spectral data.

This module provides wavelength/variable selection methods commonly used
in chemometrics for NIRS data:
- CARS (Competitive Adaptive Reweighted Sampling)
- MC-UVE (Monte-Carlo Uninformative Variable Elimination)

These selectors identify informative wavelengths and reduce feature
dimensionality while preserving predictive performance.
"""

import numpy as np
from typing import Optional, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cross_decomposition import PLSRegression
import warnings


class CARS(TransformerMixin, BaseEstimator):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for wavelength selection.

    CARS is a variable selection method that iteratively selects important
    wavelengths by:
    1. Fitting PLS models on subsets of samples
    2. Calculating variable importance weights from regression coefficients
    3. Using exponentially decreasing function to reduce variable count
    4. Applying adaptive reweighted sampling based on importance

    The method was introduced by Li et al. (2009) and is widely used for
    NIRS wavelength selection.

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components for the internal PLS model.
    n_sampling_runs : int, default=50
        Number of Monte-Carlo sampling runs.
    n_variables_ratio_start : float, default=1.0
        Starting ratio of variables to keep (1.0 = all variables).
    n_variables_ratio_end : float, default=0.1
        Ending ratio of variables to keep.
    cv_folds : int, default=5
        Number of cross-validation folds for RMSECV calculation.
    subset_ratio : float, default=0.8
        Ratio of samples to use in each Monte-Carlo run.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    selected_indices_ : ndarray of shape (n_selected,)
        Indices of selected features/wavelengths.
    selection_mask_ : ndarray of shape (n_features,)
        Boolean mask indicating selected features.
    n_features_in_ : int
        Number of features in input data.
    n_features_out_ : int
        Number of selected features.
    rmsecv_history_ : ndarray of shape (n_sampling_runs,)
        RMSECV values at each iteration.
    n_variables_history_ : ndarray of shape (n_sampling_runs,)
        Number of variables at each iteration.
    optimal_run_idx_ : int
        Index of the run with minimum RMSECV.

    Examples
    --------
    >>> from nirs4all.operators.transforms import CARS
    >>> import numpy as np
    >>>
    >>> # Spectral data with 200 wavelengths
    >>> X = np.random.randn(100, 200)
    >>> y = np.random.randn(100)
    >>>
    >>> # Select informative wavelengths
    >>> cars = CARS(n_components=10, n_sampling_runs=30)
    >>> cars.fit(X, y)
    >>> X_selected = cars.transform(X)
    >>> print(f"Selected {X_selected.shape[1]} from {X.shape[1]} wavelengths")

    References
    ----------
    Li, H., Liang, Y., Xu, Q., & Cao, D. (2009). Key wavelengths screening
    using competitive adaptive reweighted sampling method for multivariate
    calibration. Analytica Chimica Acta, 648(1), 77-84.

    Notes
    -----
    - CARS works best with standardized/scaled data
    - The exponential decay function ensures smooth variable reduction
    - Final selection is based on minimum cross-validated RMSECV
    """

    def __init__(
        self,
        n_components: int = 10,
        n_sampling_runs: int = 50,
        n_variables_ratio_start: float = 1.0,
        n_variables_ratio_end: float = 0.1,
        cv_folds: int = 5,
        subset_ratio: float = 0.8,
        random_state: Optional[int] = None
    ):
        self.n_components = n_components
        self.n_sampling_runs = n_sampling_runs
        self.n_variables_ratio_start = n_variables_ratio_start
        self.n_variables_ratio_end = n_variables_ratio_end
        self.cv_folds = cv_folds
        self.subset_ratio = subset_ratio
        self.random_state = random_state

    def _calculate_rmsecv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int,
        rng: np.random.Generator
    ) -> float:
        """Calculate RMSECV using k-fold cross-validation."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        shuffled_indices = rng.permutation(indices)

        fold_size = n_samples // self.cv_folds
        errors = []

        for fold in range(self.cv_folds):
            # Split indices
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else n_samples
            val_indices = shuffled_indices[start_idx:end_idx]
            train_indices = np.concatenate([shuffled_indices[:start_idx], shuffled_indices[end_idx:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # Fit PLS
            n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
            if n_comp < 1:
                return np.inf

            pls = PLSRegression(n_components=n_comp)
            try:
                pls.fit(X_train, y_train)
                y_pred = pls.predict(X_val)
                errors.extend((y_val.ravel() - y_pred.ravel()) ** 2)
            except Exception:
                return np.inf

        return np.sqrt(np.mean(errors))

    def fit(self, X, y=None, wavelengths: Optional[np.ndarray] = None):
        """
        Fit the CARS selector to identify important wavelengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values. Required for CARS.
        wavelengths : array-like of shape (n_features,), optional
            Original wavelength grid. Stored for reference but not required.

        Returns
        -------
        self : CARS
            Fitted selector.
        """
        if y is None:
            raise ValueError("CARS requires y (target values) for fitting")

        X = check_array(X, dtype=np.float64, ensure_all_finite='allow-nan')
        y = np.asarray(y).ravel()

        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")

        self.n_features_in_ = X.shape[1]
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Store wavelengths if provided
        self.original_wavelengths_ = wavelengths

        # Initialize random generator
        rng = np.random.default_rng(self.random_state)

        # Calculate exponential decay ratio for variable count
        # r^N = end_ratio/start_ratio => r = (end_ratio/start_ratio)^(1/N)
        ratio = (self.n_variables_ratio_end / self.n_variables_ratio_start)
        decay_rate = ratio ** (1.0 / (self.n_sampling_runs - 1)) if self.n_sampling_runs > 1 else 1.0

        # Track all selected variable sets and their RMSECV
        rmsecv_history = []
        n_variables_history = []
        selection_history = []

        # Initialize: all variables selected
        current_indices = np.arange(n_features)
        current_X = X.copy()

        for run_idx in range(self.n_sampling_runs):
            # Calculate target number of variables for this iteration
            target_ratio = self.n_variables_ratio_start * (decay_rate ** run_idx)
            target_n_vars = max(1, int(n_features * target_ratio))

            # Sample a subset of data (Monte-Carlo)
            n_subset = max(2, int(n_samples * self.subset_ratio))
            subset_indices = rng.choice(n_samples, size=n_subset, replace=False)
            X_subset = current_X[subset_indices]
            y_subset = y[subset_indices]

            # Fit PLS on subset
            n_comp = min(self.n_components, X_subset.shape[1], X_subset.shape[0] - 1)
            if n_comp < 1:
                break

            pls = PLSRegression(n_components=n_comp)
            try:
                pls.fit(X_subset, y_subset)
            except Exception:
                break

            # Calculate variable importance (absolute regression coefficients)
            coefs = np.abs(pls.coef_.ravel())

            # Adaptive reweighted sampling: probability proportional to coefficient magnitude
            weights = coefs / (coefs.sum() + 1e-10)

            # Select variables using weighted sampling without replacement
            n_to_select = min(target_n_vars, len(current_indices))

            if n_to_select < len(current_indices):
                # Weighted sampling
                selected_local = rng.choice(
                    len(current_indices),
                    size=n_to_select,
                    replace=False,
                    p=weights
                )
                selected_local = np.sort(selected_local)  # Keep order

                # Update current indices and data
                current_indices = current_indices[selected_local]
                current_X = X[:, current_indices]

            # Calculate RMSECV for current variable set
            rmsecv = self._calculate_rmsecv(current_X, y, self.n_components, rng)

            rmsecv_history.append(rmsecv)
            n_variables_history.append(len(current_indices))
            selection_history.append(current_indices.copy())

        # Find optimal selection (minimum RMSECV)
        self.rmsecv_history_ = np.array(rmsecv_history)
        self.n_variables_history_ = np.array(n_variables_history)
        self.optimal_run_idx_ = int(np.argmin(self.rmsecv_history_))

        # Get selected indices at optimal point
        self.selected_indices_ = selection_history[self.optimal_run_idx_]

        # Create boolean mask
        self.selection_mask_ = np.zeros(n_features, dtype=bool)
        self.selection_mask_[self.selected_indices_] = True

        self.n_features_out_ = len(self.selected_indices_)

        return self

    def transform(self, X):
        """
        Transform data by selecting only the important wavelengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected)
            Data with only selected wavelengths.
        """
        check_is_fitted(self, ['selected_indices_', 'selection_mask_'])

        X = check_array(X, dtype=np.float64, ensure_all_finite='allow-nan')

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but CARS was fitted with {self.n_features_in_}"
            )

        return X[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        """
        Get a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices instead of boolean mask.

        Returns
        -------
        support : ndarray
            Boolean mask or indices of selected features.
        """
        check_is_fitted(self, ['selected_indices_', 'selection_mask_'])

        if indices:
            return self.selected_indices_
        return self.selection_mask_

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (selected wavelengths as strings).

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, uses indices.

        Returns
        -------
        feature_names_out : ndarray of str
            Selected feature names.
        """
        check_is_fitted(self, 'selected_indices_')

        if input_features is not None:
            return np.array(input_features)[self.selected_indices_]

        if hasattr(self, 'original_wavelengths_') and self.original_wavelengths_ is not None:
            return np.array([f"{wl:.2f}" for wl in self.original_wavelengths_[self.selected_indices_]])

        return np.array([f"x{i}" for i in self.selected_indices_])

    def __repr__(self):
        """String representation of the selector."""
        if hasattr(self, 'n_features_out_'):
            return (f"CARS(n_components={self.n_components}, "
                    f"n_in={self.n_features_in_}, n_out={self.n_features_out_})")
        return f"CARS(n_components={self.n_components}, unfitted)"


class MCUVE(TransformerMixin, BaseEstimator):
    """
    Monte-Carlo Uninformative Variable Elimination (MC-UVE) for wavelength selection.

    MC-UVE identifies uninformative variables by comparing the stability of
    regression coefficients between real variables and random noise variables.
    Variables with low stability (similar to noise) are eliminated.

    The method works by:
    1. Augmenting X with noise variables (same distribution as X)
    2. Performing multiple PLS fits on bootstrap samples
    3. Calculating stability (mean/std) of regression coefficients
    4. Selecting variables with stability significantly higher than noise

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components for the internal PLS model.
    n_iterations : int, default=100
        Number of Monte-Carlo iterations (bootstrap samples).
    subset_ratio : float, default=0.8
        Ratio of samples to use in each bootstrap iteration.
    n_noise_variables : int or None, default=None
        Number of noise variables to add. If None, uses n_features.
    threshold_method : {'percentile', 'fixed', 'auto'}, default='auto'
        Method to determine selection threshold:
        - 'percentile': Use percentile of noise stability as threshold
        - 'fixed': Use fixed stability threshold
        - 'auto': Automatically select based on noise distribution
    threshold_percentile : float, default=99
        Percentile of noise stability used as threshold (for 'percentile' method).
    threshold_value : float, default=2.0
        Fixed stability threshold value (for 'fixed' method).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    selected_indices_ : ndarray of shape (n_selected,)
        Indices of selected features/wavelengths.
    selection_mask_ : ndarray of shape (n_features,)
        Boolean mask indicating selected features.
    n_features_in_ : int
        Number of features in input data.
    n_features_out_ : int
        Number of selected features.
    stability_ : ndarray of shape (n_features,)
        Stability values for each real variable.
    noise_stability_ : ndarray of shape (n_noise_variables,)
        Stability values for noise variables.
    threshold_ : float
        Threshold value used for selection.
    mean_coefs_ : ndarray of shape (n_features,)
        Mean regression coefficients across iterations.
    std_coefs_ : ndarray of shape (n_features,)
        Standard deviation of coefficients across iterations.

    Examples
    --------
    >>> from nirs4all.operators.transforms import MCUVE
    >>> import numpy as np
    >>>
    >>> # Spectral data with 200 wavelengths
    >>> X = np.random.randn(100, 200)
    >>> y = np.random.randn(100)
    >>>
    >>> # Select informative wavelengths
    >>> mcuve = MCUVE(n_components=10, n_iterations=100)
    >>> mcuve.fit(X, y)
    >>> X_selected = mcuve.transform(X)
    >>> print(f"Selected {X_selected.shape[1]} from {X.shape[1]} wavelengths")

    References
    ----------
    Cai, W., Li, Y., & Shao, X. (2008). A variable selection method based on
    uninformative variable elimination for multivariate calibration of
    near-infrared spectra. Chemometrics and Intelligent Laboratory Systems,
    90(2), 188-194.

    Notes
    -----
    - MC-UVE is robust against random noise
    - Higher stability indicates more informative variables
    - The noise comparison ensures a principled selection threshold
    """

    def __init__(
        self,
        n_components: int = 10,
        n_iterations: int = 100,
        subset_ratio: float = 0.8,
        n_noise_variables: Optional[int] = None,
        threshold_method: Literal['percentile', 'fixed', 'auto'] = 'auto',
        threshold_percentile: float = 99,
        threshold_value: float = 2.0,
        random_state: Optional[int] = None
    ):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.subset_ratio = subset_ratio
        self.n_noise_variables = n_noise_variables
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.threshold_value = threshold_value
        self.random_state = random_state

    def fit(self, X, y=None, wavelengths: Optional[np.ndarray] = None):
        """
        Fit the MC-UVE selector to identify important wavelengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values. Required for MC-UVE.
        wavelengths : array-like of shape (n_features,), optional
            Original wavelength grid. Stored for reference but not required.

        Returns
        -------
        self : MCUVE
            Fitted selector.
        """
        if y is None:
            raise ValueError("MC-UVE requires y (target values) for fitting")

        X = check_array(X, dtype=np.float64, ensure_all_finite='allow-nan')
        y = np.asarray(y).ravel()

        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")

        self.n_features_in_ = X.shape[1]
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Store wavelengths if provided
        self.original_wavelengths_ = wavelengths

        # Initialize random generator
        rng = np.random.default_rng(self.random_state)

        # Number of noise variables
        n_noise = self.n_noise_variables if self.n_noise_variables else n_features

        # Generate noise variables with same distribution as X
        X_noise = rng.standard_normal((n_samples, n_noise)) * X.std(axis=0).mean()

        # Augmented matrix: [X, X_noise]
        X_augmented = np.hstack([X, X_noise])

        # Store coefficients from all iterations
        all_coefs = []

        for _ in range(self.n_iterations):
            # Bootstrap sampling
            n_subset = max(2, int(n_samples * self.subset_ratio))
            subset_indices = rng.choice(n_samples, size=n_subset, replace=True)

            X_subset = X_augmented[subset_indices]
            y_subset = y[subset_indices]

            # Fit PLS
            n_comp = min(self.n_components, X_subset.shape[1], X_subset.shape[0] - 1)
            if n_comp < 1:
                continue

            pls = PLSRegression(n_components=n_comp)
            try:
                pls.fit(X_subset, y_subset)
                coefs = pls.coef_.ravel()
                all_coefs.append(coefs)
            except Exception:
                continue

        if len(all_coefs) < 10:
            warnings.warn("MC-UVE: Too few successful iterations. Results may be unreliable.")
            # Fall back to selecting all variables
            self.selected_indices_ = np.arange(n_features)
            self.selection_mask_ = np.ones(n_features, dtype=bool)
            self.n_features_out_ = n_features
            self.stability_ = np.zeros(n_features)
            self.noise_stability_ = np.zeros(n_noise)
            self.threshold_ = 0.0
            self.mean_coefs_ = np.zeros(n_features)
            self.std_coefs_ = np.ones(n_features)
            return self

        # Stack coefficients
        coefs_matrix = np.array(all_coefs)

        # Calculate stability (mean / std) for each variable
        mean_coefs = coefs_matrix.mean(axis=0)
        std_coefs = coefs_matrix.std(axis=0) + 1e-10  # Avoid division by zero

        stability = mean_coefs / std_coefs

        # Separate real and noise variables
        self.stability_ = stability[:n_features]
        self.noise_stability_ = np.abs(stability[n_features:])
        self.mean_coefs_ = mean_coefs[:n_features]
        self.std_coefs_ = std_coefs[:n_features]

        # Determine threshold
        if self.threshold_method == 'percentile':
            self.threshold_ = np.percentile(self.noise_stability_, self.threshold_percentile)
        elif self.threshold_method == 'fixed':
            self.threshold_ = self.threshold_value
        else:  # 'auto'
            # Use max of noise stability as threshold
            self.threshold_ = np.max(self.noise_stability_) * 1.1  # 10% margin

        # Select variables with absolute stability above threshold
        abs_stability = np.abs(self.stability_)
        self.selection_mask_ = abs_stability > self.threshold_
        self.selected_indices_ = np.where(self.selection_mask_)[0]

        # Ensure at least 1 variable is selected
        if len(self.selected_indices_) == 0:
            warnings.warn(
                "MC-UVE: No variables passed threshold. Selecting top 10% by stability."
            )
            n_select = max(1, n_features // 10)
            top_indices = np.argsort(abs_stability)[-n_select:]
            self.selected_indices_ = np.sort(top_indices)
            self.selection_mask_ = np.zeros(n_features, dtype=bool)
            self.selection_mask_[self.selected_indices_] = True

        self.n_features_out_ = len(self.selected_indices_)

        return self

    def transform(self, X):
        """
        Transform data by selecting only the important wavelengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected)
            Data with only selected wavelengths.
        """
        check_is_fitted(self, ['selected_indices_', 'selection_mask_'])

        X = check_array(X, dtype=np.float64, ensure_all_finite='allow-nan')

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but MC-UVE was fitted with {self.n_features_in_}"
            )

        return X[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        """
        Get a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices instead of boolean mask.

        Returns
        -------
        support : ndarray
            Boolean mask or indices of selected features.
        """
        check_is_fitted(self, ['selected_indices_', 'selection_mask_'])

        if indices:
            return self.selected_indices_
        return self.selection_mask_

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (selected wavelengths as strings).

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, uses indices.

        Returns
        -------
        feature_names_out : ndarray of str
            Selected feature names.
        """
        check_is_fitted(self, 'selected_indices_')

        if input_features is not None:
            return np.array(input_features)[self.selected_indices_]

        if hasattr(self, 'original_wavelengths_') and self.original_wavelengths_ is not None:
            return np.array([f"{wl:.2f}" for wl in self.original_wavelengths_[self.selected_indices_]])

        return np.array([f"x{i}" for i in self.selected_indices_])

    def __repr__(self):
        """String representation of the selector."""
        if hasattr(self, 'n_features_out_'):
            return (f"MCUVE(n_components={self.n_components}, "
                    f"n_in={self.n_features_in_}, n_out={self.n_features_out_})")
        return f"MCUVE(n_components={self.n_components}, unfitted)"
