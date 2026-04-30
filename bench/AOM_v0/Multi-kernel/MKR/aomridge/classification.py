"""AOM-Ridge classifier (PLS-DA-style).

This estimator wraps :class:`AOMRidgeRegressor` on a class-balanced one-hot
encoding of ``y`` and calibrates ``predict_proba`` via logistic regression on
the latent regression scores. The protocol mirrors AOM-PLS-DA:

1. Encode ``y`` as ``Y_ic = 1 / sqrt(pi_c)`` if ``y_i == c`` else ``0``,
   where ``pi_c`` is the empirical training class prior.
2. Fit the dual Ridge regressor on the encoded ``Y`` (multi-output).
3. Compute training latent scores ``T = X @ coef_ + intercept_`` (shape
   ``(n, n_classes)``).
4. Fit ``LogisticRegression(class_weight="balanced", max_iter=2000)`` on
   ``(T_train, y_train_int)``. ``predict_proba`` runs through the calibrator.
5. Fallback: if logistic calibration fails, fit a single-temperature softmax
   on the training scores.

All :class:`AOMRidgeRegressor` parameters are forwarded to the underlying
estimator, so the five fold-local CV selection modes (``superblock``,
``global``, ``active_superblock``, ``mkl``, ``branch_global``) work
unchanged. The classifier inherits the AOM-Ridge no-leak invariants because
it never bypasses the regressor's selection / refit pipeline.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

import numpy as np
from aompls.operators import LinearSpectralOperator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .estimators import AOMRidgeRegressor

OperatorBankSpec = str | Sequence[LinearSpectralOperator]


def _class_balanced_encode(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return the class-balanced one-hot encoding of ``y``.

    ``Y_ic = 1 / sqrt(pi_c)`` if ``y_i == classes[c]``, else ``0``. ``pi_c``
    is the empirical class prior on the training set, so rare classes get a
    larger column magnitude to balance their contribution to the
    cross-covariance during the dual Ridge fit.
    """
    n = y.shape[0]
    K = classes.shape[0]
    Y = np.zeros((n, K), dtype=float)
    for c, cls in enumerate(classes):
        mask = y == cls
        prior = float(mask.sum()) / float(n)
        if prior <= 0.0:
            continue
        Y[mask, c] = 1.0 / np.sqrt(prior)
    return Y


def _align_proba(
    proba: np.ndarray,
    calib_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    """Reorder columns of ``proba`` to match ``target_classes`` ordering.

    Necessary because ``LogisticRegression.classes_`` may be ordered
    differently from the encoder's class order when the calibrator drops or
    re-orders classes internally.
    """
    out = np.zeros((proba.shape[0], target_classes.shape[0]), dtype=float)
    cls_to_idx = {c: i for i, c in enumerate(calib_classes.tolist())}
    for j, cls in enumerate(target_classes.tolist()):
        if cls in cls_to_idx:
            out[:, j] = proba[:, cls_to_idx[cls]]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return np.asarray(out / s)


def _fit_temperature(
    scores: np.ndarray, y: np.ndarray, classes: np.ndarray
) -> float:
    """Fit a single positive temperature scalar by golden-section search.

    Minimises mean class log loss of ``softmax(scores / T)`` on the supplied
    training data. ``T`` is bracketed in ``[1e-2, 5.0]``.
    """
    eps = 1e-12

    def loss(T: float) -> float:
        scaled = scores / max(T, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        proba = e / e.sum(axis=1, keepdims=True)
        out = []
        for i, yi in enumerate(y):
            idx = int(np.where(classes == yi)[0][0])
            out.append(-np.log(max(float(proba[i, idx]), eps)))
        return float(np.mean(out))

    a, b = 1e-2, 5.0
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    res_phi = 1.0 / phi
    c = b - res_phi * (b - a)
    d = a + res_phi * (b - a)
    for _ in range(40):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c = b - res_phi * (b - a)
        d = a + res_phi * (b - a)
    return 0.5 * (a + b)


class AOMRidgeClassifier(BaseEstimator, ClassifierMixin):
    """AOM-Ridge classifier (PLS-DA-style).

    Wraps :class:`AOMRidgeRegressor` on a class-balanced one-hot encoding of
    ``y`` and calibrates ``predict_proba`` via a logistic regression fitted
    on the latent regression scores. All AOM-Ridge selection modes
    (``superblock``, ``global``, ``active_superblock``, ``mkl``,
    ``branch_global``) are available through the ``selection`` parameter.

    Parameters
    ----------
    selection, operator_bank, alphas, alpha_grid_size, alpha_grid_low,
    alpha_grid_high, alpha, cv, scoring, block_scaling, center, scale,
    active_top_m, active_diversity_threshold, random_state, solver,
    scale_power, adaptive_alpha_grid, max_grid_expansions, x_scale,
    active_score_method, active_max_per_family, global_per_operator_grid,
    selection_rule, mkl_top_k, mkl_mode, branches
        Forwarded to the underlying :class:`AOMRidgeRegressor`. See its
        docstring for the full contract.
    calibrator : str
        ``"logistic"`` (default) fits ``LogisticRegression`` on training
        latent scores and uses it for ``predict_proba``. ``"temperature"``
        skips logistic calibration and uses the temperature-scaled softmax
        of the latent scores. The logistic path automatically falls back to
        temperature scaling when logistic fitting fails.
    calibration_max_iter : int
        ``max_iter`` for the logistic calibrator (default 2000).

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted unique class labels seen during ``fit``.
    regressor_ : AOMRidgeRegressor
        The fitted underlying regressor (after ``fit``).
    calibrator_ : LogisticRegression or None
        Fitted calibrator, ``None`` when the temperature fallback is active.
    calibrator_kind_ : str
        ``"logistic"`` or ``"temperature"`` depending on which path was used.
    temperature_ : float or None
        Fitted temperature (only when ``calibrator_kind_ == "temperature"``).
    diagnostics_ : dict
        Selection / calibration diagnostics.
    """

    def __init__(
        self,
        selection: str = "superblock",
        operator_bank: OperatorBankSpec = "compact",
        alphas: str | Sequence[float] = "auto",
        alpha_grid_size: int = 50,
        alpha_grid_low: float = -6.0,
        alpha_grid_high: float = 6.0,
        alpha: float | None = None,
        cv: int | object = 5,
        scoring: str = "rmse",
        block_scaling: str = "rms",
        center: bool = True,
        scale: bool = False,
        active_top_m: int = 20,
        active_diversity_threshold: float = 0.98,
        random_state: int | None = 0,
        solver: str = "auto",
        scale_power: float = 1.0,
        adaptive_alpha_grid: bool = True,
        max_grid_expansions: int = 2,
        x_scale: str = "center",
        active_score_method: str = "norm",
        active_max_per_family: int | None = None,
        global_per_operator_grid: bool = True,
        selection_rule: str = "min",
        mkl_top_k: int = 6,
        mkl_mode: str = "alignment",
        branches: Sequence[str] = ("none", "snv", "msc"),
        calibrator: str = "logistic",
        calibration_max_iter: int = 2000,
    ) -> None:
        self.selection = selection
        self.operator_bank = operator_bank
        self.alphas = alphas
        self.alpha_grid_size = alpha_grid_size
        self.alpha_grid_low = alpha_grid_low
        self.alpha_grid_high = alpha_grid_high
        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.block_scaling = block_scaling
        self.center = center
        self.scale = scale
        self.active_top_m = active_top_m
        self.active_diversity_threshold = active_diversity_threshold
        self.random_state = random_state
        self.solver = solver
        self.scale_power = scale_power
        self.adaptive_alpha_grid = adaptive_alpha_grid
        self.max_grid_expansions = max_grid_expansions
        self.x_scale = x_scale
        self.active_score_method = active_score_method
        self.active_max_per_family = active_max_per_family
        self.global_per_operator_grid = global_per_operator_grid
        self.selection_rule = selection_rule
        self.mkl_top_k = mkl_top_k
        self.mkl_mode = mkl_mode
        self.branches = branches
        self.calibrator = calibrator
        self.calibration_max_iter = calibration_max_iter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _regressor_kwargs(self) -> dict:
        """Return the keyword arguments forwarded to ``AOMRidgeRegressor``.

        The classifier-only parameters (``calibrator`` and
        ``calibration_max_iter``) are excluded.
        """
        return {
            "selection": self.selection,
            "operator_bank": self.operator_bank,
            "alphas": self.alphas,
            "alpha_grid_size": self.alpha_grid_size,
            "alpha_grid_low": self.alpha_grid_low,
            "alpha_grid_high": self.alpha_grid_high,
            "alpha": self.alpha,
            "cv": self.cv,
            "scoring": self.scoring,
            "block_scaling": self.block_scaling,
            "center": self.center,
            "scale": self.scale,
            "active_top_m": self.active_top_m,
            "active_diversity_threshold": self.active_diversity_threshold,
            "random_state": self.random_state,
            "solver": self.solver,
            "scale_power": self.scale_power,
            "adaptive_alpha_grid": self.adaptive_alpha_grid,
            "max_grid_expansions": self.max_grid_expansions,
            "x_scale": self.x_scale,
            "active_score_method": self.active_score_method,
            "active_max_per_family": self.active_max_per_family,
            "global_per_operator_grid": self.global_per_operator_grid,
            "selection_rule": self.selection_rule,
            "mkl_top_k": self.mkl_top_k,
            "mkl_mode": self.mkl_mode,
            "branches": self.branches,
        }

    def _latent_scores(self, X: np.ndarray) -> np.ndarray:
        """Return the regressor's latent scores at ``X``, shape ``(n, K)``.

        The regressor's ``predict`` already produces the multi-output
        (encoded-Y) prediction, which is exactly the latent score vector
        used as input to the calibrator. We delegate to it so the branched
        path (non-trivial ``branch_global`` selection) is handled
        identically to regression.
        """
        Y_pred = self.regressor_.predict(X)
        Y_pred = np.asarray(Y_pred, dtype=float)
        if Y_pred.ndim == 1:
            # Should not happen for K >= 2 (encoded Y is always 2D), but
            # handle the degenerate single-class case defensively.
            Y_pred = Y_pred.reshape(-1, 1)
        return Y_pred

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgeClassifier:
        if self.calibrator not in ("logistic", "temperature"):
            raise ValueError(
                f"unknown calibrator {self.calibrator!r}; expected "
                "'logistic' or 'temperature'"
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        t0 = time.perf_counter()
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError(
                "AOMRidgeClassifier requires at least two distinct classes "
                f"in y; got {self.classes_.tolist()}"
            )

        # Class-balanced one-hot encoding
        Y_bal = _class_balanced_encode(y, self.classes_)

        # Fit the dual Ridge regressor on encoded Y. The regressor handles
        # all selection / fold-local CV / refit logic.
        self.regressor_ = AOMRidgeRegressor(**self._regressor_kwargs())
        self.regressor_.fit(X, Y_bal)

        # Compute training latent scores (shape (n, K)) for calibration.
        T_train = self._latent_scores(X)

        self._fit_calibrator(T_train, y)
        self._fit_time_s = float(time.perf_counter() - t0)
        self.diagnostics_ = self._build_diagnostics()
        return self

    def _fit_calibrator(self, T_train: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic calibrator with temperature fallback."""
        self.calibrator_ = None
        self.temperature_ = None
        self.calibrator_kind_ = self.calibrator
        if self.calibrator == "logistic":
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    calib = LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(self.calibration_max_iter),
                        random_state=self.random_state,
                    )
                    calib.fit(T_train, y)
                self.calibrator_ = calib
                self.calibrator_kind_ = "logistic"
                return
            except Exception:
                self.calibrator_kind_ = "temperature"
        # Temperature fallback (also used when ``calibrator == "temperature"``)
        self.temperature_ = _fit_temperature(T_train, y, self.classes_)
        self.calibrator_kind_ = "temperature"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("predict_proba called before fit")
        T = self._latent_scores(X)
        if self.calibrator_kind_ == "logistic" and self.calibrator_ is not None:
            proba = self.calibrator_.predict_proba(T)
            return _align_proba(proba, self.calibrator_.classes_, self.classes_)
        # Temperature-scaled softmax
        T_arr = np.asarray(T, dtype=float)
        temperature = float(self.temperature_) if self.temperature_ is not None else 1.0
        scaled = T_arr / max(temperature, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        return np.asarray(e / e.sum(axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[idx])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw latent scores (regressor output on encoded Y)."""
        return self._latent_scores(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return balanced accuracy on ``(X, y)``."""
        from aompls.metrics import balanced_accuracy

        return float(balanced_accuracy(np.asarray(y), self.predict(X)))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        return dict(self.diagnostics_)

    def get_selected_operators(self) -> list[str]:
        return list(self.regressor_.selected_operators_)

    def _build_diagnostics(self) -> dict:
        reg_diag = self.regressor_.get_diagnostics()
        diag: dict = {
            "model": "AOMRidgeClassifier",
            "n_classes": int(self.classes_.shape[0]),
            "classes": [c.item() if hasattr(c, "item") else c for c in self.classes_],
            "calibrator": self.calibrator,
            "calibrator_kind": self.calibrator_kind_,
            "calibration_max_iter": int(self.calibration_max_iter),
            "temperature": (
                None if self.temperature_ is None else float(self.temperature_)
            ),
            "fit_time_s": float(getattr(self, "_fit_time_s", 0.0)),
        }
        # Forward the regressor's selection diagnostics under a clear prefix.
        for k in (
            "selection",
            "operator_bank",
            "alpha",
            "alpha_index",
            "alpha_at_boundary",
            "alphas",
            "cv",
            "cv_min_score",
            "selection_rule",
            "grid_expansions",
            "block_scaling",
            "scale_power",
            "x_scale",
            "selected_operator_names",
            "selected_operator_indices",
            "operator_scores",
            "block_scales",
            "block_importance",
            "chosen_branch",
            "mkl_weights",
            "mkl_operator_weights",
            "active_operator_names",
            "active_operator_indices",
            "active_operator_scores",
            "active_pruned_count",
        ):
            if k in reg_diag:
                diag[k] = reg_diag[k]
        return diag
