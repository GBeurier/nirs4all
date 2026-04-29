"""Selection criteria for operator-adaptive PLS.

A scorer evaluates a candidate (`engine`, `operator_index`, `prefix_k`) tuple
and returns a real-valued score. Lower is better. The selection module then
picks the argmin (or the prefix that minimises the score for `auto`).

Implemented criteria:

- `covariance`: maximise covariance norm (proxy, fast). Returns `-||A_b S||`.
- `cv`: leakage-safe K-fold CV RMSE / log-loss.
- `approx_press`: approximate PRESS using the predictive residuals on the
  training set with leverage correction (when feasible).
- `hybrid`: covariance prescreening + CV refinement on the top-m operators.
- `holdout`: legacy single train/validation split (debug only, never default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold


@dataclass
class CriterionConfig:
    """Configuration for the selection criterion.

    Attributes:
        kind: One of `"covariance"`, `"cv"`, `"approx_press"`, `"hybrid"`,
            `"holdout"`.
        cv: Number of folds for `cv` and `hybrid`. Default 5.
        prescreen_top_m: Number of operators kept by the covariance proxy
            for the `hybrid` criterion. Default 5.
        random_state: Random seed for fold construction (CV / hybrid).
        task: `"regression"` (RMSE) or `"classification"` (balanced log loss).
        holdout_fraction: Validation fraction for the legacy holdout
            criterion. Default 0.2 (matches the production AOM-PLS).
        holdout_seed: Fixed seed for the holdout split. Defaults to 42 to
            match the production deployment's hard-coded
            `RandomState(42)`; this keeps the comparison apples-to-apples
            and removes a source of small-n variance that is otherwise
            user-supplied.
        repeats: Number of CV repeats with different seeds. Default 1
            (single CV pass). Setting `repeats=3` averages CV across
            three independent K-fold splits, lowering selection variance
            by sqrt(repeats). Only used when `kind == "cv"`.
        one_se_rule: If True, after picking the argmin of (b, k), apply
            the one-standard-error rule: among all candidates within
            `best_score + std/sqrt(n_folds)` of the best, pick the one
            with the smallest k (preferring identity-like operators when
            the bank starts with `IdentityOperator`). Default False.
    """

    kind: str = "cv"
    cv: int = 5
    prescreen_top_m: int = 5
    random_state: int = 0
    task: str = "regression"
    holdout_fraction: float = 0.2
    holdout_seed: int = 42
    repeats: int = 1
    one_se_rule: bool = False


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_true - y_pred).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def _balanced_log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Class-balanced log loss for classification CV."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    classes = np.unique(y_true)
    losses = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        p_cls = np.clip(proba[mask, list(classes).index(cls)], eps, 1.0 - eps)
        losses.append(-np.mean(np.log(p_cls)))
    return float(np.mean(losses)) if losses else float("inf")


# ---------------------------------------------------------------------------
# Covariance proxy
# ---------------------------------------------------------------------------


def covariance_score(S: np.ndarray) -> float:
    """Return the negative spectral / Frobenius norm of `S` (lower is better)."""
    if S.ndim == 1:
        return -float(np.linalg.norm(S))
    return -float(np.linalg.norm(S, ord="fro"))


# ---------------------------------------------------------------------------
# CV-based scorer
# ---------------------------------------------------------------------------


def cv_score_regression(
    Xc: np.ndarray,
    yc: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int,
    random_state: int,
) -> float:
    """Compute K-fold CV RMSE on already-centered `Xc, yc`.

    The `fit_predict` callable receives `(X_train, y_train, X_val)` and
    returns the predicted `y_val_hat`. Centering is the caller's
    responsibility for the fold split: the fold splitter sees the centered
    matrix, and the callable should re-center inside each fold.

    Note: this helper passes the *raw* centered training matrices to
    `fit_predict`, which is responsible for any per-fold (re)centering.
    Standard practice in PLS is to re-center within each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses = []
    for train_idx, val_idx in kf.split(Xc):
        X_tr, X_va = Xc[train_idx], Xc[val_idx]
        y_tr, y_va = yc[train_idx], yc[val_idx]
        try:
            y_va_hat = fit_predict(X_tr, y_tr, X_va)
        except Exception:
            return float("inf")
        rmses.append(_rmse(y_va, y_va_hat))
    return float(np.mean(rmses))


def cv_score_classification(
    Xc: np.ndarray,
    y: np.ndarray,
    fit_predict_proba: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int,
    random_state: int,
) -> float:
    """Compute K-fold balanced log loss for classification."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses = []
    for train_idx, val_idx in skf.split(Xc, y):
        X_tr, X_va = Xc[train_idx], Xc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        try:
            proba_va = fit_predict_proba(X_tr, y_tr, X_va)
        except Exception:
            return float("inf")
        losses.append(_balanced_log_loss(y_va, proba_va))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Approximate PRESS
# ---------------------------------------------------------------------------


def approx_press_regression(
    Xc: np.ndarray,
    yc: np.ndarray,
    coef_per_prefix: Sequence[np.ndarray],
) -> List[float]:
    """Return the approximate PRESS for each prefix length.

    The approximation evaluates training-set residuals adjusted for leverage
    via a hat-matrix proxy `h = diag(X (X^T X)^+ X^T)`. We compute the hat
    once on the centered `Xc` and reuse it for all prefix lengths.
    """
    Xc = np.asarray(Xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    if yc.ndim == 1:
        yc = yc.reshape(-1, 1)
    n, p = Xc.shape
    # Hat matrix proxy via SVD: H = U U^T where U are left singular vectors of Xc.
    U, _S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    h = np.einsum("ij,ij->i", U, U)
    h = np.clip(h, 0.0, 1.0 - 1e-9)
    out: List[float] = []
    for B in coef_per_prefix:
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        y_hat = Xc @ B
        residuals = (yc - y_hat) / (1.0 - h.reshape(-1, 1))
        press = float(np.sum(residuals * residuals))
        out.append(press)
    return out


# ---------------------------------------------------------------------------
# Holdout (legacy)
# ---------------------------------------------------------------------------


def holdout_score_regression(
    Xc: np.ndarray,
    yc: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    fraction: float,
    random_state: int,
) -> float:
    """Single train/val split RMSE used for legacy holdout selection."""
    rng = np.random.default_rng(random_state)
    n = Xc.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * fraction)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_tr, X_va = Xc[train_idx], Xc[val_idx]
    y_tr, y_va = yc[train_idx], yc[val_idx]
    try:
        y_va_hat = fit_predict(X_tr, y_tr, X_va)
    except Exception:
        return float("inf")
    return _rmse(y_va, y_va_hat)
