"""Fold-local CV utilities for AOM-Ridge.

All routines here recompute fold-local means, operator clones, operator fits,
block scales, and kernels. Nothing is sliced from a globally centered kernel.

The CV interface is intentionally generic: ``cv`` may be an integer (interpreted
as ``KFold(n_splits=cv, shuffle=True, random_state=random_state)``) or any
sklearn-compatible splitter exposing ``split(X, y)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from aompls.operators import IdentityOperator, LinearSpectralOperator
from sklearn.model_selection import KFold

from .kernels import (
    clone_operator_bank,
    compute_block_scales_from_xt,
    fit_operator_bank,
    linear_operator_kernel_cross,
    linear_operator_kernel_train,
)
from .preprocessing import apply_feature_scaler, fit_feature_scaler
from .solvers import solve_dual_ridge_path_eigh

CVSpec = int | object


# ----------------------------------------------------------------------
# CV resolution
# ----------------------------------------------------------------------


def resolve_cv(
    cv: CVSpec, random_state: int | None = None
) -> object:
    """Resolve ``cv`` to an sklearn-compatible splitter.

    Integers are mapped to a shuffled ``KFold``. Any object that exposes
    ``split(X, y)`` is returned unchanged.
    """
    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("integer cv must be >= 2")
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    if hasattr(cv, "split"):
        return cv
    raise TypeError(
        "cv must be an integer or an sklearn-compatible splitter with `split(X, y)`"
    )


# ----------------------------------------------------------------------
# Fold scoring
# ----------------------------------------------------------------------


def _rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    diff = Y_true - Y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _fold_local_kernels(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    Y_tr: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    block_scaling: str,
    center: bool,
    scale_power: float = 1.0,
    x_scale: str = "center",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute fold-local kernels and scaled targets.

    Returns ``K_tr``, ``K_va``, ``Yc_tr``, ``y_mean_f``, ``x_mean_f``,
    ``U_tr``, ``block_scales``, ``x_scale_f``.

    ``x_scale``: ``"center"`` (default, current behavior — only subtract
    mean), ``"none"``, ``"feature_std"``, or ``"feature_rms"``.
    """
    # Reconcile center=False with x_scale: center=False means "no centering",
    # in which case x_scale must be "none".
    if not center and x_scale not in ("none",):
        x_scale = "none"
    mode = x_scale
    x_mean_f, x_scale_f = fit_feature_scaler(X_tr, mode=mode)
    Xc_tr = apply_feature_scaler(X_tr, x_mean_f, x_scale_f)
    Xc_va = apply_feature_scaler(X_va, x_mean_f, x_scale_f)
    if center:
        y_mean_f = Y_tr.mean(axis=0)
    else:
        y_mean_f = np.zeros(Y_tr.shape[1])
    Yc_tr = Y_tr - y_mean_f
    ops_f = clone_operator_bank(operators_template, p=Xc_tr.shape[1])
    fit_operator_bank(ops_f, Xc_tr)
    scales_f = compute_block_scales_from_xt(
        Xc_tr.T, ops_f, block_scaling=block_scaling, scale_power=scale_power,
    )
    K_tr, U_tr = linear_operator_kernel_train(Xc_tr, ops_f, scales_f)
    K_va = linear_operator_kernel_cross(Xc_va, U_tr)
    return K_tr, K_va, Yc_tr, y_mean_f, x_mean_f, U_tr, scales_f, x_scale_f


def cv_score_alphas(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "rms",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
) -> np.ndarray:
    """Compute mean validation RMSE per alpha for one operator subset.

    Returns an array of shape ``(len(alphas),)``. ``operators_template`` is the
    list of operator instances to use as the *superblock* for every fold.
    Folds clone and fit them locally.
    """
    rmse_acc = np.zeros((len(alphas),), dtype=float)
    n_folds = 0
    for train_idx, valid_idx in cv.split(X, Y):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        K_tr, K_va, Yc_tr, y_mean_f, _, _, _, _ = _fold_local_kernels(
            X_tr, X_va, Y_tr, operators_template, block_scaling, center,
            scale_power=scale_power, x_scale=x_scale,
        )
        # Solve for all alphas via single eigendecomposition
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        # Predict and accumulate RMSE
        for i in range(alphas.size):
            Y_pred = K_va @ Cs[i] + y_mean_f
            rmse_acc[i] += _rmse(Y_va, Y_pred)
        n_folds += 1
    if n_folds == 0:
        raise ValueError("cv produced no folds")
    return rmse_acc / n_folds


def select_alpha_superblock(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "rms",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
) -> tuple[float, np.ndarray]:
    """Return ``(alpha_star, rmse_per_alpha)`` for the superblock model.

    The selected alpha minimises mean validation RMSE over the supplied folds.
    """
    rmse = cv_score_alphas(
        X, Y, operators_template, alphas, cv,
        block_scaling=block_scaling, center=center, scale_power=scale_power,
        x_scale=x_scale,
    )
    if not np.all(np.isfinite(rmse)):
        raise FloatingPointError("non-finite RMSE encountered during CV")
    idx = int(np.argmin(rmse))
    return float(alphas[idx]), rmse


# ----------------------------------------------------------------------
# Global hard selection over (operator, alpha)
# ----------------------------------------------------------------------


def select_global(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "rms",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    per_operator_alpha_grids: list[np.ndarray] | None = None,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Select ``(operator_idx, alpha)`` minimising mean validation RMSE.

    Returns the index of the chosen operator, the chosen alpha, the 2D RMSE
    table with shape ``(len(operators_template), len(alphas))``, and the
    per-operator alpha grid actually used (so callers can recover the alpha
    value when ``per_operator_alpha_grids`` differs across operators).

    When ``per_operator_alpha_grids`` is provided, the b-th row of
    ``rmse_table`` is scored against ``per_operator_alpha_grids[b]`` instead
    of the shared ``alphas`` argument.
    """
    n_ops = len(operators_template)
    n_alpha = len(alphas)
    rmse_table = np.empty((n_ops, n_alpha), dtype=float)
    grids_used = np.empty((n_ops, n_alpha), dtype=float)
    for b, op in enumerate(operators_template):
        op_alphas = per_operator_alpha_grids[b] if per_operator_alpha_grids else alphas
        if len(op_alphas) != n_alpha:
            raise ValueError(
                "per_operator_alpha_grids[b] must have the same length as alphas"
            )
        rmse_b = cv_score_alphas(
            X,
            Y,
            [op],
            op_alphas,
            cv,
            block_scaling=block_scaling,
            center=center,
            scale_power=scale_power,
            x_scale=x_scale,
        )
        rmse_table[b] = rmse_b
        grids_used[b] = op_alphas
    if not np.all(np.isfinite(rmse_table)):
        raise FloatingPointError("non-finite RMSE encountered during global selection")
    flat_idx = int(np.argmin(rmse_table))
    b_star, a_star = np.unravel_index(flat_idx, rmse_table.shape)
    return int(b_star), float(grids_used[b_star, a_star]), rmse_table, grids_used


# ----------------------------------------------------------------------
# Active superblock screening
# ----------------------------------------------------------------------


def _normalized_score(
    Xc_tr: np.ndarray, Yc_tr: np.ndarray, op: LinearSpectralOperator, scale: float
) -> float:
    """Compute ``||s_b A_b Xc^T Yc||_F^2`` for one operator."""
    S = Xc_tr.T @ Yc_tr                # (p, q)
    R = op.apply_cov(S)                # A_b S
    return float(scale) ** 2 * float(np.linalg.norm(R, "fro")) ** 2


def _kta_score(
    Xc_tr: np.ndarray, Yc_tr: np.ndarray, op: LinearSpectralOperator, scale: float,
) -> float:
    """Kernel-target alignment ``<K_b, Y Y^T>_F / (||K_b||_F * ||Y Y^T||_F)``.

    Scale-invariant by construction; complementary to ``_normalized_score``
    which is sensitive to magnitude.
    """
    AXt = op.apply_cov(Xc_tr.T)                 # (p, n) — same shape as Xc^T
    AtAXt = op.adjoint_vec(AXt)                 # (p, n)
    K = float(scale) ** 2 * (Xc_tr @ AtAXt)     # (n, n)
    K_norm = float(np.linalg.norm(K, "fro"))
    YYt = Yc_tr @ Yc_tr.T
    Y_norm = float(np.linalg.norm(YYt, "fro"))
    if K_norm < 1e-30 or Y_norm < 1e-30:
        return 0.0
    return float(np.sum(K * YYt) / (K_norm * Y_norm))


def _operator_family(name: str) -> str:
    """Map operator name to a coarse family for quota-balanced screening.

    Mirrors the family heuristic in ``aompls.banks.family_pruned_default``.
    """
    if name == "identity":
        return "identity"
    if name.startswith("compose"):
        return "compose"
    if name.startswith("sg_smooth"):
        return "sg_smooth"
    if name.startswith("sg_d1"):
        return "sg_d1"
    if name.startswith("sg_d2"):
        return "sg_d2"
    if name.startswith("nw"):
        return "nw"
    if name.startswith("detrend"):
        return "detrend"
    if name.startswith("fd"):
        return "fd"
    if name.startswith("whittaker"):
        return "whittaker"
    return "other"


def _response_signature(
    Xc_tr: np.ndarray, Yc_tr: np.ndarray, op: LinearSpectralOperator, scale: float
) -> np.ndarray:
    """Compute the flattened response signature ``s_b A_b Xc^T Yc`` for cosine pruning."""
    S = Xc_tr.T @ Yc_tr
    R = op.apply_cov(S)
    return float(scale) * R.ravel()


def screen_active_operators(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    block_scaling: str = "rms",
    center: bool = True,
    top_m: int = 20,
    diversity_threshold: float = 0.98,
    keep_identity: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    score_method: str = "norm",                  # "norm", "kta", or "blend"
    max_per_family: int | None = None,           # cap operators per family
) -> tuple[list[int], list[float], int]:
    """Screen and prune operators on the supplied (training) data.

    Returns ``(active_indices, active_scores, pruned_count)``. The screening
    fits operators and computes block scales on the supplied data only — the
    caller is responsible for passing fold-local or full-calibration data.

    ``top_m`` is a hard cap on the returned subset size, including identity
    when ``keep_identity=True``.
    """
    if top_m < 1:
        raise ValueError("top_m must be >= 1")
    mode = x_scale if center else "none"
    x_mean, x_scale_arr = fit_feature_scaler(X_tr, mode=mode)
    if center:
        y_mean = Y_tr.mean(axis=0)
    else:
        y_mean = np.zeros(Y_tr.shape[1])
    Xc = apply_feature_scaler(X_tr, x_mean, x_scale_arr)
    Yc = Y_tr - y_mean
    ops = clone_operator_bank(operators_template, p=Xc.shape[1])
    fit_operator_bank(ops, Xc)
    scales = compute_block_scales_from_xt(
        Xc.T, ops, block_scaling=block_scaling, scale_power=scale_power,
    )
    if score_method == "norm":
        scores = np.array(
            [_normalized_score(Xc, Yc, op, s)
             for op, s in zip(ops, scales, strict=False)],
            dtype=float,
        )
    elif score_method == "kta":
        scores = np.array(
            [_kta_score(Xc, Yc, op, s)
             for op, s in zip(ops, scales, strict=False)],
            dtype=float,
        )
    elif score_method == "blend":
        s_norm = np.array(
            [_normalized_score(Xc, Yc, op, s)
             for op, s in zip(ops, scales, strict=False)],
            dtype=float,
        )
        s_kta = np.array(
            [_kta_score(Xc, Yc, op, s)
             for op, s in zip(ops, scales, strict=False)],
            dtype=float,
        )
        # Min-max normalise each then sum: gives operators that are both
        # high-magnitude and well-aligned with Y a boost.
        def _norm_to_01(arr):
            lo, hi = float(arr.min()), float(arr.max())
            return (arr - lo) / (hi - lo + 1e-30) if hi > lo else np.zeros_like(arr)
        scores = _norm_to_01(s_norm) + _norm_to_01(s_kta)
    else:
        raise ValueError("score_method must be 'norm', 'kta', or 'blend'")
    order = np.argsort(-scores)            # descending
    identity_indices = [
        i for i, op in enumerate(operators_template) if isinstance(op, IdentityOperator)
    ]
    active: list[int] = []
    active_signatures: list[np.ndarray] = []
    family_counts: dict[str, int] = {}
    pruned = 0

    def _try_add(idx: int) -> bool:
        """Attempt to add operator idx; return True if added, False if pruned."""
        sig = _response_signature(Xc, Yc, ops[idx], scales[idx])
        sig_norm = sig / (np.linalg.norm(sig) + 1e-30)
        for prev in active_signatures:
            if abs(float(sig_norm @ prev)) >= diversity_threshold:
                return False
        if max_per_family is not None:
            family = _operator_family(operators_template[idx].name)
            if family_counts.get(family, 0) >= max_per_family:
                return False
            family_counts[family] = family_counts.get(family, 0) + 1
        active.append(idx)
        active_signatures.append(sig_norm)
        return True

    if keep_identity and identity_indices:
        idx = identity_indices[0]
        active.append(idx)
        sig = _response_signature(Xc, Yc, ops[idx], scales[idx])
        active_signatures.append(sig / (np.linalg.norm(sig) + 1e-30))
        if max_per_family is not None:
            family_counts["identity"] = 1
    if len(active) >= top_m:
        return active[:top_m], [float(scores[i]) for i in active[:top_m]], pruned
    for idx in order:
        idx = int(idx)
        if idx in active:
            continue
        if not _try_add(idx):
            pruned += 1
            continue
        if len(active) >= top_m:
            break
    active_scores = [float(scores[i]) for i in active]
    return active, active_scores, pruned


# ----------------------------------------------------------------------
# Fold-local active CV: screen operators inside every fold to avoid leak
# ----------------------------------------------------------------------


def cv_score_active_alphas(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "rms",
    center: bool = True,
    active_top_m: int = 20,
    active_diversity_threshold: float = 0.98,
    scale_power: float = 1.0,
    x_scale: str = "center",
    score_method: str = "norm",
    max_per_family: int | None = None,
) -> np.ndarray:
    """Mean validation RMSE per alpha for active-superblock selection.

    Inside every fold the active subset is screened from the *training* fold
    only, so validation rows never participate in the operator-selection
    decision.
    """
    rmse_acc = np.zeros((len(alphas),), dtype=float)
    n_folds = 0
    for train_idx, valid_idx in cv.split(X, Y):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        active_idx, _, _ = screen_active_operators(
            X_tr,
            Y_tr,
            operators_template,
            block_scaling=block_scaling,
            center=center,
            top_m=active_top_m,
            diversity_threshold=active_diversity_threshold,
            keep_identity=True,
            scale_power=scale_power,
            x_scale=x_scale,
            score_method=score_method,
            max_per_family=max_per_family,
        )
        active_subset = [operators_template[i] for i in active_idx]
        K_tr, K_va, Yc_tr, y_mean_f, _, _, _, _ = _fold_local_kernels(
            X_tr, X_va, Y_tr, active_subset, block_scaling, center,
            scale_power=scale_power, x_scale=x_scale,
        )
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        for i in range(alphas.size):
            Y_pred = K_va @ Cs[i] + y_mean_f
            rmse_acc[i] += _rmse(Y_va, Y_pred)
        n_folds += 1
    if n_folds == 0:
        raise ValueError("cv produced no folds")
    return rmse_acc / n_folds


def select_alpha_active(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "rms",
    center: bool = True,
    active_top_m: int = 20,
    active_diversity_threshold: float = 0.98,
    scale_power: float = 1.0,
    x_scale: str = "center",
    score_method: str = "norm",
    max_per_family: int | None = None,
) -> tuple[float, np.ndarray]:
    """Return ``(alpha_star, rmse_per_alpha)`` with fold-local active screening."""
    rmse = cv_score_active_alphas(
        X,
        Y,
        operators_template,
        alphas,
        cv,
        block_scaling=block_scaling,
        center=center,
        active_top_m=active_top_m,
        active_diversity_threshold=active_diversity_threshold,
        scale_power=scale_power,
        x_scale=x_scale,
        score_method=score_method,
        max_per_family=max_per_family,
    )
    if not np.all(np.isfinite(rmse)):
        raise FloatingPointError("non-finite RMSE encountered during active CV")
    idx = int(np.argmin(rmse))
    return float(alphas[idx]), rmse
