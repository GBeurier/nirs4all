"""TabPFN NIRS regressor with a fixed, empirically-validated preprocessing.

This module provides :class:`TabPFNNIRSRegressor`, a sklearn-compatible NIRS
regressor that pairs a fixed preprocessing chain with the TabPFN tabular
foundation model. The recipe avoids per-dataset HPO entirely.

Recipe
------
``SG(window_length=11, polyorder=2, deriv=1)`` ▶ ``OSC()`` ▶
centering (``with_std=False``) ▶ ``TabPFNRegressor(n_estimators=16)``.

If the post-OSC matrix has more than ``max_features`` columns, an evenly
spaced ``linspace`` subsample brings it back below the TabPFN v3 hard limit
of 2000 features.

Empirical results on a 57-dataset NIRS cohort (regression, paper splits)
versus the TabPFN paper's 72-chain per-dataset HPO (``TabPFN_opt``):

- median RMSE delta: ~+2.5%
- worst case: +47% on one dataset (Brix_spxy70 at n=35)
- ~21 of 57 datasets beat ``TabPFN_opt`` outright

Cost is a single TabPFN training per dataset (no HPO, no stacking, no CV
selector). The recipe was chosen by extracting the most-frequent winning
preprocessing motif from the TabPFN paper logs (deriv-1 SG with OSC
correction) and a small ``n_estimators`` sweep (4/8/16/32) that pinned
``n_estimators=16`` as the sweet spot.

Requirements
------------
- ``tabpfn`` (``pip install tabpfn``). Imported lazily inside :meth:`fit`,
  so importing this module does not require the optional dependency.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms import SavitzkyGolay as _SG
from nirs4all.operators.transforms.orthogonalization import OSC as _OSC


class TabPFNNIRSRegressor(BaseEstimator, RegressorMixin):
    """NIRS-tuned TabPFN regressor with a fixed AGG preprocessing pipeline.

    Parameters
    ----------
    n_estimators : int, default=16
        TabPFN ensemble size at final fit. Selected empirically.
    max_features : int, default=2000
        TabPFN v3 hard limit on input features. If the preprocessed matrix
        exceeds it, columns are uniformly subsampled via ``np.linspace``.
    sg_window_length : int, default=11
    sg_polyorder : int, default=2
    sg_deriv : int, default=1
        Savitzky-Golay smoothing parameters.
    osc_n_components : int, default=1
        Number of orthogonal-signal-correction components to remove.
    random_state : int, default=0
        Random seed for TabPFN.
    device : str, default="auto"
        TabPFN device. ``"auto"``, ``"cpu"``, or ``"cuda"``.
    model_path : str, default="auto"
        TabPFN checkpoint path. ``"auto"`` loads the default v3 checkpoint.

    Attributes
    ----------
    sg_ : SavitzkyGolay
        Fitted SG transformer.
    osc_ : OSC
        Fitted OSC transformer.
    scaler_ : StandardScaler
        Fitted centering scaler.
    model_ : TabPFNRegressor
        Fitted TabPFN model.
    n_features_in_ : int
        Number of features seen at fit time.
    n_features_used_ : int
        Number of features actually fed to TabPFN (after subsample cap).
    subsample_idx_ : ndarray or None
        Column indices used. ``None`` when no subsample was applied.

    Notes
    -----
    On a 57-dataset NIRS cohort (paper splits) the fixed recipe lands at
    median RMSE ~+2.5% versus TabPFN_opt's 72-chain HPO and beats it on
    ~37% of datasets, at a single TabPFN training per dataset.

    The recipe is intentionally fixed. For per-dataset preprocessing
    selection, run an external CV loop over candidate chains and feed the
    chosen ones into this class.

    Examples
    --------
    >>> from nirs4all.operators.models import TabPFNNIRSRegressor
    >>> est = TabPFNNIRSRegressor(n_estimators=16)  # doctest: +SKIP
    >>> est.fit(X_train, y_train)                   # doctest: +SKIP
    >>> y_pred = est.predict(X_test)                # doctest: +SKIP
    """

    def __init__(
        self,
        n_estimators: int = 16,
        max_features: int = 2000,
        sg_window_length: int = 11,
        sg_polyorder: int = 2,
        sg_deriv: int = 1,
        osc_n_components: int = 1,
        random_state: int = 0,
        device: str = "auto",
        model_path: str = "auto",
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_features = int(max_features)
        self.sg_window_length = int(sg_window_length)
        self.sg_polyorder = int(sg_polyorder)
        self.sg_deriv = int(sg_deriv)
        self.osc_n_components = int(osc_n_components)
        self.random_state = int(random_state)
        self.device = device
        self.model_path = model_path

    def _apply_column_cap(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
        """Return ``X`` subsampled to at most ``max_features`` columns.

        At fit time, builds ``subsample_idx_`` from a ``np.linspace`` grid.
        At predict time, reuses the cached indices (or returns ``X``
        unchanged if no subsampling was needed at fit).
        """
        if fitting:
            if X.shape[1] <= self.max_features:
                self.subsample_idx_ = None
                return X
            idx = np.linspace(0, X.shape[1] - 1, self.max_features, dtype=np.int64)
            self.subsample_idx_ = np.unique(idx)
            return X[:, self.subsample_idx_]
        if self.subsample_idx_ is None:
            return X
        return X[:, self.subsample_idx_]

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabPFNNIRSRegressor:
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]

        self.sg_ = _SG(
            window_length=self.sg_window_length,
            polyorder=self.sg_polyorder,
            deriv=self.sg_deriv,
        )
        X_sg = self.sg_.fit_transform(X)

        self.osc_ = _OSC(n_components=self.osc_n_components)
        self.osc_.fit(X_sg, y)
        X_osc = np.asarray(self.osc_.transform(X_sg), dtype=np.float64)

        X_osc = self._apply_column_cap(X_osc, fitting=True)
        self.n_features_used_ = X_osc.shape[1]

        self.scaler_ = StandardScaler(with_mean=True, with_std=False).fit(X_osc)
        X_final = self.scaler_.transform(X_osc)

        kwargs: dict = {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "device": self.device,
            "ignore_pretraining_limits": True,
        }
        if self.model_path and self.model_path != "auto":
            kwargs["model_path"] = self.model_path
        self.model_ = TabPFNRegressor(**kwargs)
        self.model_.fit(X_final, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_sg = self.sg_.transform(X)
        X_osc = np.asarray(self.osc_.transform(X_sg), dtype=np.float64)
        X_osc = self._apply_column_cap(X_osc, fitting=False)
        X_final = self.scaler_.transform(X_osc)
        return np.asarray(self.model_.predict(X_final)).ravel()


__all__ = ["TabPFNNIRSRegressor"]
