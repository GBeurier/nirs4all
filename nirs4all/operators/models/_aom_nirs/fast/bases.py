"""Nonlinear base transforms for FastAOM.

A *base* is a (possibly non-linear) preprocessing of the spectra
``B(X) ∈ R^{n x p}`` that is applied **before** any linear chain. Because
most useful spectral pre-treatments are non-linear (SNV, MSC, EMSC,
absorbance, ASLS baseline removal), and because the screening / low-rank
machinery only works on the linear part, we keep the number of bases
very small (≤ ~6) and explore the linear chain space inside each base.

Each base implements a minimal fold-aware interface::

    base.fit(X_train, y_train=None) -> self
    base.transform(X) -> ndarray (n, p)
    base.fit_transform(X_train, y_train=None) -> ndarray
    base.signature -> str

The first four (raw / absorbance / SNV / MSC) are stateless or
row-statistic-based; EMSC and ASLS / Whittaker fit a reference (or
baseline parameters) on the *training fold* only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solveh_banded

from nirs4all.operators.models._aom_nirs.pls.preprocessing import (
    StandardNormalVariate as _AompStandardNormalVariate,
    MultiplicativeScatterCorrection as _AompMSC,
    OrthogonalSignalCorrection as _AompOSC,
)


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class BaseTransform:
    """Protocol for fold-aware nonlinear base transforms."""

    name: str

    @property
    def signature(self) -> str:  # pragma: no cover - thin alias
        return self.name

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseTransform":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)


# ---------------------------------------------------------------------------
# Raw / absorbance
# ---------------------------------------------------------------------------


class RawBase(BaseTransform):
    """Identity base: ``B(X) = X``."""

    name = "raw"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RawBase":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


class AbsorbanceBase(BaseTransform):
    """Absorbance from reflectance: ``B(X) = -log10(clip(X, eps, 1))``.

    Reflectance is expected to lie in (0, 1]. If the data is already in
    absorbance space (e.g. negative values, magnitudes far outside
    [0, 1]) the transform falls back to identity to avoid producing NaN
    values that would break downstream models — the fallback is recorded
    in :attr:`fallback_to_identity_`.
    """

    name = "absorbance"

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.fallback_to_identity_: bool = False

    @property
    def signature(self) -> str:
        return f"absorbance(eps={self.eps:g})"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AbsorbanceBase":
        X = np.asarray(X, dtype=float)
        # Heuristic: reflectance should be in (0, 1] with most mass in [0.05, 1].
        # Reject if there are negative values or extreme values that indicate
        # we are already in absorbance / log-reflectance / derivative space.
        if X.size and (np.min(X) < 0.0 or np.max(X) > 5.0):
            self.fallback_to_identity_ = True
        else:
            self.fallback_to_identity_ = False
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.fallback_to_identity_:
            return X.copy()
        clipped = np.clip(X, self.eps, None)
        return -np.log10(clipped)


# ---------------------------------------------------------------------------
# SNV / MSC / EMSC
# ---------------------------------------------------------------------------


class SNVBase(BaseTransform):
    """Standard Normal Variate, row-wise centring and scaling.

    SNV is independent of the training fold (purely row-based), but we
    keep the fit/transform interface for API symmetry.
    """

    name = "snv"

    def __init__(self) -> None:
        self._impl = _AompStandardNormalVariate()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SNVBase":
        self._impl.fit(X, y=y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._impl.transform(X), dtype=float)


class MSCBase(BaseTransform):
    """Multiplicative Scatter Correction with training-fold mean as reference."""

    name = "msc"

    def __init__(self) -> None:
        self._impl = _AompMSC()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MSCBase":
        self._impl.fit(X, y=y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._impl.transform(X), dtype=float)


class EMSCBase(BaseTransform):
    """Extended MSC with polynomial wavelength terms.

    For each row ``x`` and a reference mean ``m`` (fit on the training
    fold), fit ``a + b * m + c1 * t + c2 * t^2 + ... + cD * t^D ≈ x``
    and return ``(x - a - c1 t - ... - cD t^D) / b``. Reduces to MSC
    when ``degree == 0``.
    """

    name = "emsc"

    def __init__(self, degree: int = 2) -> None:
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self.reference_: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._basis: Optional[np.ndarray] = None
        self._pinv: Optional[np.ndarray] = None

    @property
    def signature(self) -> str:
        return f"emsc(d={self.degree})"

    def _build_basis(self, p: int) -> None:
        t = np.linspace(-1.0, 1.0, p)
        ref = self.reference_
        assert ref is not None
        cols = [np.ones(p), ref]
        for k in range(1, self.degree + 1):
            cols.append(t**k)
        basis = np.column_stack(cols)  # shape (p, 2 + degree)
        self._basis = basis
        # Least-squares pseudo-inverse for the design matrix.
        self._pinv = np.linalg.pinv(basis)
        self._t = t

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EMSCBase":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("EMSCBase expects a 2D array")
        self.reference_ = X.mean(axis=0)
        self._build_basis(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None or self._basis is None or self._pinv is None:
            raise RuntimeError("EMSCBase.transform called before fit")
        X = np.asarray(X, dtype=float)
        # coeffs shape: (n, 2 + degree); columns are [a, b, c1, ..., cD]
        coeffs = X @ self._pinv.T
        a = coeffs[:, 0]
        b = coeffs[:, 1]
        # Reconstruct the polynomial baseline (without the reference term).
        if self.degree > 0:
            poly = coeffs[:, 2:] @ self._basis[:, 2:].T
        else:
            poly = np.zeros_like(X)
        denom = np.where(np.abs(b) < 1e-8, 1.0, b)
        return (X - a[:, None] - poly) / denom[:, None]


# ---------------------------------------------------------------------------
# ASLS / Whittaker baseline corrections (fold-aware via deterministic params)
# ---------------------------------------------------------------------------


def _asls_baseline(y: np.ndarray, lam: float, p: float, max_iter: int = 20, tol: float = 1e-3) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens 2005).

    Solves ``(W + lam * D.T D) z = W y`` where ``D`` is the second-difference
    operator. ``D.T D`` is symmetric pentadiagonal, so we use banded
    Cholesky (`scipy.linalg.solveh_banded`) for O(n) per solve instead of
    O(n^3) — critical for NIRS spectra with p ≥ 1000.
    """
    n = y.shape[0]
    if n < 3:
        return y.copy()

    # DTD diagonals (second-difference, stencil [1, -2, 1]):
    #   main:    [1, 5, 6, 6, ..., 6, 5, 1]
    #   1st sup: [-2, -4, -4, ..., -4, -2]   (length n-1)
    #   2nd sup: [1, 1, ..., 1]               (length n-2)
    main_dtd = np.full(n, 6.0)
    main_dtd[0] = main_dtd[-1] = 1.0
    main_dtd[1] = main_dtd[-2] = 5.0
    super1_dtd = np.full(n - 1, -4.0)
    super1_dtd[0] = super1_dtd[-1] = -2.0

    # Upper-banded layout for solveh_banded (lower=False):
    #   ab[2, j] = main diagonal at column j
    #   ab[1, j] = first super-diagonal entry (j-1, j); ab[1, 0] unused
    #   ab[0, j] = second super-diagonal entry (j-2, j); ab[0, 0:2] unused
    ab = np.zeros((3, n))
    ab[1, 1:] = lam * super1_dtd
    ab[0, 2:] = lam  # second super of DTD is all-ones

    w = np.ones(n)
    z = y.copy()
    for _ in range(max_iter):
        ab[2, :] = w + lam * main_dtd
        z_new = solveh_banded(ab, w * y, lower=False, check_finite=False)
        w_new = np.where(y > z_new, p, 1.0 - p)
        if np.linalg.norm(w_new - w) / max(1e-12, np.linalg.norm(w)) < tol:
            w = w_new
            z = z_new
            break
        w = w_new
        z = z_new
    return z


class ASLSBase(BaseTransform):
    """ASLS baseline removal, applied independently to each spectrum.

    Strictly speaking ASLS is non-linear because the weights depend on the
    spectrum. We keep the lambda/p as hyperparameters so that the
    transformation is deterministic given those parameters; the fit step
    is a no-op because there is no per-fold parameter to learn.
    """

    def __init__(self, lam: float = 1e5, p: float = 0.01, max_iter: int = 20, tol: float = 1e-3) -> None:
        if lam <= 0:
            raise ValueError("lam must be > 0")
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    @property
    def name(self) -> str:
        return f"asls_l{self.lam:g}_p{self.p:g}"

    @property
    def signature(self) -> str:
        return self.name

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ASLSBase":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            base = _asls_baseline(X[i], lam=self.lam, p=self.p, max_iter=self.max_iter, tol=self.tol)
            out[i] = X[i] - base
        return out


class OSCBase(BaseTransform):
    """Orthogonal Signal Correction (supervised linear).

    Removes ``n_components`` directions of ``X`` orthogonal to ``y`` via
    the production-style OSC routine from ``aompls.preprocessing``. Used as
    a *base* because it must see ``y`` at fit time and replays a stored
    projection at predict time — exactly the protocol required for
    fold-aware NIRS preprocessing.
    """

    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        self._impl = _AompOSC(n_components=self.n_components)

    @property
    def name(self) -> str:
        return f"osc_n{self.n_components}"

    @property
    def signature(self) -> str:
        return self.name

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OSCBase":
        if y is None:
            raise ValueError("OSCBase is supervised; ``y`` is required at fit time")
        self._impl.fit(np.asarray(X), np.asarray(y).ravel())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._impl.transform(X), dtype=float)


class SNVOSCBase(BaseTransform):
    """SNV row-normalisation followed by OSC (supervised)."""

    def __init__(self, n_components: int = 2) -> None:
        self.snv = SNVBase()
        self.osc = OSCBase(n_components=n_components)

    @property
    def name(self) -> str:
        return f"snv_osc_n{self.osc.n_components}"

    @property
    def signature(self) -> str:
        return self.name

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SNVOSCBase":
        Xsnv = self.snv.fit_transform(X)
        self.osc.fit(Xsnv, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.osc.transform(self.snv.transform(X))


class WhittakerBaseLine(BaseTransform):
    """Whittaker smoother used as a baseline estimator: ``B(X) = X - smooth(X)``.

    The smoother is *linear* given lambda but is plugged in here as a base
    because the **subtraction** of the smoothed signal is what we want to
    feed the linear chains downstream. The smoother itself shares the
    parent ``aompls.WhittakerOperator``.
    """

    def __init__(self, lam: float = 1e5) -> None:
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.lam = float(lam)
        self._op = None

    @property
    def name(self) -> str:
        return f"whittaker_baseline_l{self.lam:g}"

    @property
    def signature(self) -> str:
        return self.name

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "WhittakerBaseLine":
        # Lazy import to avoid circular dependencies during package init.
        from nirs4all.operators.models._aom_nirs.pls.operators import WhittakerOperator

        self._op = WhittakerOperator(lam=self.lam)
        self._op.fit(np.asarray(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._op is None:
            raise RuntimeError("WhittakerBaseLine.transform called before fit")
        X = np.asarray(X, dtype=float)
        return X - self._op.transform(X)


# ---------------------------------------------------------------------------
# Default bank
# ---------------------------------------------------------------------------


def build_base_bank(
    use_raw: bool = True,
    use_absorbance: bool = True,
    use_snv: bool = True,
    use_msc: bool = True,
    use_emsc: bool = False,
    asls_grid: Optional[Sequence[Tuple[float, float]]] = None,
    osc_components: Optional[Sequence[int]] = None,
    use_snv_osc: bool = False,
    use_whittaker_baseline: Optional[Sequence[float]] = None,
) -> List[BaseTransform]:
    """Construct a default bank of nonlinear / supervised bases.

    Args:
        use_raw / use_absorbance / use_snv / use_msc / use_emsc: toggle
            simple normalisation bases.
        asls_grid: optional sequence of ``(lam, p)`` pairs to add as
            :class:`ASLSBase` baseline-corrected variants.
        osc_components: optional sequence of OSC component counts (each
            added as an :class:`OSCBase`). Supervised — uses ``y`` at fit
            time and replays at predict.
        use_snv_osc: include an SNV+OSC compound base (typical NIRS
            production pipeline).
        use_whittaker_baseline: optional sequence of ``lam`` for
            :class:`WhittakerBaseLine` bases.
    """
    bank: List[BaseTransform] = []
    if use_raw:
        bank.append(RawBase())
    if use_absorbance:
        bank.append(AbsorbanceBase())
    if use_snv:
        bank.append(SNVBase())
    if use_msc:
        bank.append(MSCBase())
    if use_emsc:
        bank.append(EMSCBase(degree=2))
    if asls_grid:
        for lam, p in asls_grid:
            bank.append(ASLSBase(lam=lam, p=p))
    if osc_components:
        for n in osc_components:
            bank.append(OSCBase(n_components=int(n)))
    if use_snv_osc:
        bank.append(SNVOSCBase(n_components=2))
    if use_whittaker_baseline:
        for lam in use_whittaker_baseline:
            bank.append(WhittakerBaseLine(lam=float(lam)))
    if not bank:
        raise ValueError("build_base_bank produced an empty bank")
    return bank
