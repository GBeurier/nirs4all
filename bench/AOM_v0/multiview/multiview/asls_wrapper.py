"""ASLS-outer-preproc wrapper.

Applies ASLSBaseline (asymmetric least-squares baseline correction) to X
before passing to a downstream multi-view estimator. Mirrors the
`_PreprocAOMWrapper` pattern from `bench/AOM_v0/benchmarks/run_aompls_benchmark.py`
but is sklearn-compatible.

ASLS (lambda=1e6, p=0.01) was the secret sauce of the AOM_v0 champion variant
(`ASLS-AOM-compact-cv5-numpy` with median rel-RMSEP 0.96). Applying it
upstream of multi-view AOM-MoE / block-sparse should provide similar gains.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from aompls.preprocessing import ASLSBaseline


class ASLSPreprocWrapper(BaseEstimator, RegressorMixin):
    """Wrap any regressor with an outer ASLS baseline correction.

    Parameters
    ----------
    estimator : BaseEstimator
        Inner regressor to fit on ASLS-corrected X.
    lam : float
        ASLS smoothness parameter (default 1e6).
    p : float
        ASLS asymmetry parameter (default 0.01 — favors low values).
    max_iter : int
        ASLS max iterations (default 50).
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        lam: float = 1e6,
        p: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        self.estimator = estimator
        self.lam = lam
        self.p = p
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ASLSPreprocWrapper":
        if self.estimator is None:
            raise ValueError("estimator must be provided")
        start = time.perf_counter()
        self._asls = ASLSBaseline(lam=self.lam, p=self.p, max_iter=self.max_iter)
        X_corr = self._asls.fit_transform(np.asarray(X, dtype=float), y)
        self._inner = clone(self.estimator)
        self._inner.fit(X_corr, y)
        self.fit_time_s_ = float(time.perf_counter() - start)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_inner"):
            raise RuntimeError("Estimator not fitted")
        X_corr = self._asls.transform(np.asarray(X, dtype=float))
        return np.asarray(self._inner.predict(X_corr)).ravel()
