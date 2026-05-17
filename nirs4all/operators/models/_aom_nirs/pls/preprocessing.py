"""Standalone preprocessors applied upstream of AOM-PLS / POP-PLS.

`StandardNormalVariate` and `MultiplicativeScatterCorrection` are
non-linear (per-sample normalisation) and therefore cannot live in the
strict-linear operator bank. `OrthogonalSignalCorrection` (OSC) is a
supervised linear projector that *can* in principle live in the bank
once a fold-aware fit/refit is wired in, but the simpler use case is
upstream of AOM-PLS, applied once on the training set with `y` and then
replayed at predict time with the stored projection.

`ExtendedMSC` is a self-contained EMSC implementation (vendored from
the original `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`).
`ASLSBaseline` wraps `pybaselines.whittaker.asls` directly, so the
AOM_v0 benchmark can mirror the TabPFN paper's full pipeline-search
configuration (5 normalisations x 2 baselines x 4 OSC settings).

These pre-processors are used as the "non-linear pipeline" baselines in
the benchmark: SNV->AOM-PLS, MSC->AOM-PLS, OSC->AOM-PLS, and
combinations. They follow a sklearn-compatible `fit(X, y) /
transform(X)` interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
from sklearn.utils.validation import check_is_fitted


class StandardNormalVariate(BaseEstimator, TransformerMixin):
    """Standard Normal Variate normalization (per-sample, non-linear)."""

    def fit(self, X: np.ndarray, y=None) -> StandardNormalVariate:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        return np.asarray((X - mean) / std, dtype=float)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """MSC normalization fit on a reference spectrum (per-sample affine fit)."""

    def __init__(self) -> None:
        self.reference_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> MultiplicativeScatterCorrection:
        X = np.asarray(X, dtype=float)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.reference_ is None:
            self.fit(X)
        ref = np.asarray(self.reference_, dtype=float)
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            slope, intercept = np.polyfit(ref, X[i], 1)
            if abs(slope) > 1e-12:
                out[i] = (X[i] - intercept) / slope
            else:
                out[i] = X[i] - intercept
        return out

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)


class OrthogonalSignalCorrection(BaseEstimator, TransformerMixin):
    """Direct OSC (DOSC) — supervised linear projector.

    OSC removes the variance in `X` that is orthogonal to `y`. For each
    of `n_components` orthogonal components, we extract a PLS-style score
    that has maximum X-variance, then orthogonalise its loading against
    the y-direction so that the removed component is uncorrelated with
    `y`. The projection matrix `P_o = [p_orth_1, ..., p_orth_K]` is
    stored at fit time and reused at predict time:

        X_filtered = X (I - P_o P_o^T)

    The transform is therefore strictly linear at apply time. The
    operator is "supervised linear": the projection matrix depends on
    `y` (used at fit only).

    The implementation matches the production `_opls_prefilter` in
    `nirs4all/operators/models/sklearn/aom_pls.py`, line 773.
    """

    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        self.P_orth_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> OrthogonalSignalCorrection:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.x_mean_ = X.mean(axis=0)
        Xc = X - self.x_mean_
        n, p = Xc.shape
        eps = 1e-12
        P_orth = np.zeros((p, self.n_components), dtype=np.float64)
        X_filt = Xc.copy()
        actual = 0
        for i in range(self.n_components):
            w = X_filt.T @ y
            w_norm = np.linalg.norm(w)
            if w_norm < eps:
                break
            w = w / w_norm
            t = X_filt @ w
            tt = t @ t
            if tt < eps:
                break
            p_vec = X_filt.T @ t / tt
            p_orth = p_vec - w * (w @ p_vec)
            p_orth_norm = np.linalg.norm(p_orth)
            if p_orth_norm < eps:
                break
            p_orth = p_orth / p_orth_norm
            t_orth = X_filt @ p_orth
            tt_orth = t_orth @ t_orth
            if tt_orth < eps:
                break
            X_filt = X_filt - np.outer(t_orth, t_orth @ X_filt / tt_orth)
            P_orth[:, i] = p_orth
            actual += 1
        self.P_orth_ = P_orth[:, :actual]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.P_orth_ is None or self.x_mean_ is None:
            raise RuntimeError("OrthogonalSignalCorrection must be fit before transform")
        Xc = X - self.x_mean_
        if self.P_orth_.size == 0:
            return np.asarray(Xc, dtype=float)
        return np.asarray(Xc - (Xc @ self.P_orth_) @ self.P_orth_.T, dtype=float)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


class PreprocessingPipeline:
    """Sequential pipeline of `(fit_transform, transform)` preprocessors.

    Lightweight wrapper to compose SNV/MSC/OSC chains and ship them as a
    single estimator-friendly preprocessor. Each step's `fit` sees the
    output of the previous steps, then `transform` is replayed at
    predict time in the same order.
    """

    def __init__(self, steps) -> None:
        self.steps = list(steps)

    def fit(self, X: np.ndarray, y=None) -> PreprocessingPipeline:
        Xt = np.asarray(X, dtype=float)
        for step in self.steps:
            try:
                step.fit(Xt, y)
            except TypeError:
                step.fit(Xt)
            Xt = step.transform(Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=float)
        for step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class ExtendedMSC(TransformerMixin, BaseEstimator):
    """Extended Multiplicative Scatter Correction (EMSC).

    Vendored from `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`.
    EMSC extends MSC by including polynomial terms to model chemical
    and physical light-scattering effects.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial used to model interference.
    scale : bool, default=True
        Whether to mean-center the data before correction (with_std=False).
    copy : bool, default=True
        Whether to copy input data.
    """

    def __init__(self, degree: int = 2, scale: bool = True, *, copy: bool = True):
        self.degree = int(degree)
        self.scale = bool(scale)
        self.copy = bool(copy)

    def _reset(self):
        for attr in ("scaler_", "reference_", "wavelengths_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        tmp_x = X.copy() if self.copy else X
        if self.scale:
            scaler = _SklearnStandardScaler(with_std=False)
            scaler.fit(X)
            self.scaler_ = scaler
            tmp_x = scaler.transform(tmp_x)
        self.reference_ = np.mean(tmp_x, axis=0)
        self.wavelengths_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_transformed = X.copy() if self.copy else X
        if self.scale:
            X_transformed = self.scaler_.transform(X_transformed)

        for i in range(X_transformed.shape[0]):
            design_matrix = np.column_stack([
                self.reference_,
                *[self.wavelengths_ ** d for d in range(1, self.degree + 1)],
            ])
            coeffs, _, _, _ = np.linalg.lstsq(design_matrix, X_transformed[i], rcond=None)
            polynomial_part = sum(
                coeffs[d] * (self.wavelengths_ ** d) for d in range(1, self.degree + 1)
            )
            X_transformed[i] = (X_transformed[i] - polynomial_part) / coeffs[0]
        return X_transformed

    def _more_tags(self):
        return {"allow_nan": False}


class ASLSBaseline(TransformerMixin, BaseEstimator):
    """Asymmetric Least Squares (AsLS) baseline correction.

    Direct wrapper around `pybaselines.whittaker.asls`. Iterative
    re-weighted-least-squares baseline removal driven by an asymmetry
    parameter ``p`` and a smoothness ``lambda``. Stateless (per-sample
    baseline subtracted at ``transform`` time), so safe to apply
    upstream of AOM-PLS without leaking y information.

    Parameters
    ----------
    lam : float, default=1e6
        Smoothness parameter.
    p : float, default=0.01
        Asymmetry parameter (0 < p < 1).
    max_iter : int, default=50
        Maximum number of iterations.
    tol : float, default=1e-3
        Convergence tolerance.

    References
    ----------
    Eilers and Boelens (2005), "Baseline Correction with Asymmetric
    Least Squares Smoothing".
    """

    _stateless = True

    def __init__(self, lam: float = 1e6, p: float = 0.01, max_iter: int = 50, tol: float = 1e-3):
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from pybaselines.whittaker import asls as _asls
        except ImportError as exc:
            raise ImportError(
                "pybaselines is required for ASLSBaseline. "
                "Install with: pip install pybaselines"
            ) from exc

        corrected = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            baseline, _ = _asls(
                X[i],
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            corrected[i] = X[i] - baseline
        return corrected

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def _more_tags(self):
        return {"allow_nan": False, "stateless": True}


class LocalSNV(BaseEstimator, TransformerMixin):
    """Windowed Standard Normal Variate (per-sample, per-window).

    Standard SNV normalises each spectrum globally (one mean / std for the
    full wavelength range). Local SNV partitions the spectrum into
    consecutive windows of size `window` and normalises each window
    independently. This captures localised scatter that a global SNV
    cannot. Typical NIR window sizes: 31, 51, 101 wavelengths.

    Like SNV, this is per-sample non-linear (no training-time state).
    """

    def __init__(self, window: int = 51) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        self.window = int(window)

    def fit(self, X: np.ndarray, y=None) -> "LocalSNV":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        out = np.empty_like(X)
        w = self.window
        for start in range(0, p, w):
            end = min(start + w, p)
            block = X[:, start:end]
            mean = block.mean(axis=1, keepdims=True)
            std = block.std(axis=1, keepdims=True)
            std = np.where(std > 1e-12, std, 1.0)
            out[:, start:end] = (block - mean) / std
        return np.asarray(out, dtype=float)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)
