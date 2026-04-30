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

from .branches import VALID_BRANCHES, apply_branch_transform, make_branch_preproc
from .kernels import (
    clone_operator_bank,
    compute_block_scales_from_xt,
    fit_operator_bank,
    linear_operator_kernel_cross,
    linear_operator_kernel_train,
)
from .mkl import learn_block_weights, mkl_kernel_cross, mkl_kernel_train
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


def _sum_squared_error(Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[float, int]:
    """Return (sum_of_squared_errors, n_elements) for pooled-MSE scoring."""
    diff = Y_true - Y_pred
    return float(np.sum(diff * diff)), int(diff.size)


def _summarise_alpha_scores(
    rmse_per_fold: np.ndarray, sse_per_fold: np.ndarray, count_per_fold: np.ndarray,
    scoring: str,
) -> np.ndarray:
    """Reduce per-fold per-alpha scores to a 1D summary used for selection.

    ``scoring`` is ``"rmse_mean"`` (mean of fold RMSEs, default) or
    ``"mse_pooled"`` (sqrt of total SSE divided by total element count, i.e.
    the global RMSE pooled across folds).
    """
    if scoring == "rmse_mean":
        return rmse_per_fold.mean(axis=0)
    if scoring == "mse_pooled":
        total_sse = sse_per_fold.sum(axis=0)
        total_n = count_per_fold.sum(axis=0)
        total_n = np.where(total_n > 0, total_n, 1)
        return np.sqrt(total_sse / total_n)
    raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")


def select_alpha_with_rule(
    rmse_per_fold: np.ndarray,
    alphas: np.ndarray,
    rule: str,
    summary: np.ndarray | None = None,
) -> int:
    """Pick the alpha index from a ``(n_folds, n_alphas)`` per-fold matrix.

    Parameters
    ----------
    rmse_per_fold : ndarray of shape ``(n_folds, n_alphas)``
        Per-fold per-alpha validation scores (lower is better).
    alphas : ndarray of shape ``(n_alphas,)``
        Alpha grid; only used to validate the column count.
    rule : str
        ``"min"`` (argmin of the summary) or ``"1se"`` (most-regularised alpha
        whose summary lies within one SE of the minimum).
    summary : ndarray of shape ``(n_alphas,)``, optional
        Pooled / external summary used for both the argmin (``"min"``) and the
        within-SE comparison (``"1se"``). Defaults to ``rmse_per_fold.mean(axis=0)``.

    Returns
    -------
    idx : int
        The selected column index.
    """
    rmse_arr = np.asarray(rmse_per_fold, dtype=float)
    if rmse_arr.ndim != 2:
        raise ValueError("rmse_per_fold must be a 2D (n_folds, n_alphas) array")
    if rmse_arr.shape[1] != alphas.size:
        raise ValueError(
            "rmse_per_fold and alphas must agree on the alpha count"
        )
    if rule not in ("min", "1se"):
        raise ValueError("rule must be 'min' or '1se'")
    if summary is None:
        summary_arr = rmse_arr.mean(axis=0)
    else:
        summary_arr = np.asarray(summary, dtype=float)
        if summary_arr.shape != (rmse_arr.shape[1],):
            raise ValueError(
                "summary must have shape (n_alphas,) matching rmse_per_fold"
            )
    if rule == "min":
        return int(np.argmin(summary_arr))
    # 1-SE rule: standard error of the per-alpha row-mean column at argmin.
    n_folds = rmse_arr.shape[0]
    if n_folds < 2:
        return int(np.argmin(summary_arr))
    idx_min = int(np.argmin(summary_arr))
    se = float(rmse_arr[:, idx_min].std(ddof=1)) / float(np.sqrt(n_folds))
    threshold = float(summary_arr[idx_min]) + se
    # Most-regularised (largest alpha) within the threshold band.
    candidates = np.where(summary_arr <= threshold + 1e-15)[0]
    if candidates.size == 0:
        return idx_min
    return int(candidates.max())


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

    ``x_scale``: ``"center"`` (default, current behavior - only subtract
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
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute mean validation RMSE per alpha for one operator subset.

    Returns an array of shape ``(len(alphas),)`` (the summary score per alpha).
    With ``return_per_fold=True`` returns a tuple ``(summary, per_fold)`` where
    ``per_fold`` has shape ``(n_folds, len(alphas))``.

    ``operators_template`` is the list of operator instances to use as the
    *superblock* for every fold. Folds clone and fit them locally.

    ``scoring`` is ``"rmse_mean"`` (mean of fold RMSEs, default) or
    ``"mse_pooled"`` (pooled RMSE across folds).
    """
    folds = list(cv.split(X, Y))
    if not folds:
        raise ValueError("cv produced no folds")
    n_folds = len(folds)
    n_alphas = alphas.size
    rmse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    sse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    count_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        K_tr, K_va, Yc_tr, y_mean_f, _, _, _, _ = _fold_local_kernels(
            X_tr, X_va, Y_tr, operators_template, block_scaling, center,
            scale_power=scale_power, x_scale=x_scale,
        )
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        for i in range(n_alphas):
            Y_pred = K_va @ Cs[i] + y_mean_f
            rmse_per_fold[fold_idx, i] = _rmse(Y_va, Y_pred)
            sse, count = _sum_squared_error(Y_va, Y_pred)
            sse_per_fold[fold_idx, i] = sse
            count_per_fold[fold_idx, i] = count
    summary = _summarise_alpha_scores(
        rmse_per_fold, sse_per_fold, count_per_fold, scoring,
    )
    if return_per_fold:
        return summary, rmse_per_fold
    return summary


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
    scoring: str = "rmse_mean",
    selection_rule: str = "min",
) -> tuple[float, np.ndarray]:
    """Return ``(alpha_star, rmse_per_alpha)`` for the superblock model.

    The selected alpha minimises mean validation RMSE over the supplied folds
    (``selection_rule="min"``) or follows the 1-SE rule
    (``selection_rule="1se"``).
    """
    summary, per_fold = cv_score_alphas(
        X, Y, operators_template, alphas, cv,
        block_scaling=block_scaling, center=center, scale_power=scale_power,
        x_scale=x_scale, scoring=scoring, return_per_fold=True,
    )
    if not np.all(np.isfinite(summary)):
        raise FloatingPointError("non-finite RMSE encountered during CV")
    idx = select_alpha_with_rule(per_fold, alphas, rule=selection_rule, summary=summary)
    return float(alphas[idx]), summary


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
    scoring: str = "rmse_mean",
    selection_rule: str = "min",
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Select ``(operator_idx, alpha)`` minimising mean validation RMSE.

    Returns the index of the chosen operator, the chosen alpha, the 2D RMSE
    table with shape ``(len(operators_template), len(alphas))``, and the
    per-operator alpha grid actually used (so callers can recover the alpha
    value when ``per_operator_alpha_grids`` differs across operators).

    When ``per_operator_alpha_grids`` is provided, the b-th row of
    ``rmse_table`` is scored against ``per_operator_alpha_grids[b]`` instead
    of the shared ``alphas`` argument.

    With ``selection_rule="1se"`` the chosen operator is the one whose row
    minimises the summary; the alpha is the most-regularised value within one
    SE of that minimum *inside the chosen operator's row*. Comparing alphas
    across rows is unsafe because per-operator grids may sit on different
    scales when ``per_operator_alpha_grids`` is set.
    """
    n_ops = len(operators_template)
    n_alpha = len(alphas)
    rmse_table = np.empty((n_ops, n_alpha), dtype=float)
    grids_used = np.empty((n_ops, n_alpha), dtype=float)
    per_fold_table: list[np.ndarray] = []
    for b, op in enumerate(operators_template):
        op_alphas = per_operator_alpha_grids[b] if per_operator_alpha_grids else alphas
        if len(op_alphas) != n_alpha:
            raise ValueError(
                "per_operator_alpha_grids[b] must have the same length as alphas"
            )
        summary, per_fold = cv_score_alphas(
            X,
            Y,
            [op],
            np.asarray(op_alphas, dtype=float),
            cv,
            block_scaling=block_scaling,
            center=center,
            scale_power=scale_power,
            x_scale=x_scale,
            scoring=scoring,
            return_per_fold=True,
        )
        rmse_table[b] = summary
        grids_used[b] = op_alphas
        per_fold_table.append(per_fold)
    if not np.all(np.isfinite(rmse_table)):
        raise FloatingPointError("non-finite RMSE encountered during global selection")
    # Pick the operator whose best row-summary is smallest, then choose the
    # alpha *inside that row* using the configured rule.
    row_min = rmse_table.min(axis=1)
    b_star = int(np.argmin(row_min))
    alpha_idx = select_alpha_with_rule(
        per_fold_table[b_star],
        np.asarray(grids_used[b_star], dtype=float),
        rule=selection_rule,
        summary=rmse_table[b_star],
    )
    return int(b_star), float(grids_used[b_star, alpha_idx]), rmse_table, grids_used


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
    AXt = op.apply_cov(Xc_tr.T)                 # (p, n) - same shape as Xc^T
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
    fits operators and computes block scales on the supplied data only - the
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
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Mean validation RMSE per alpha for active-superblock selection.

    Inside every fold the active subset is screened from the *training* fold
    only, so validation rows never participate in the operator-selection
    decision.
    """
    folds = list(cv.split(X, Y))
    if not folds:
        raise ValueError("cv produced no folds")
    n_folds = len(folds)
    n_alphas = alphas.size
    rmse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    sse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    count_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
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
        for i in range(n_alphas):
            Y_pred = K_va @ Cs[i] + y_mean_f
            rmse_per_fold[fold_idx, i] = _rmse(Y_va, Y_pred)
            sse, count = _sum_squared_error(Y_va, Y_pred)
            sse_per_fold[fold_idx, i] = sse
            count_per_fold[fold_idx, i] = count
    summary = _summarise_alpha_scores(
        rmse_per_fold, sse_per_fold, count_per_fold, scoring,
    )
    if return_per_fold:
        return summary, rmse_per_fold
    return summary


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
    scoring: str = "rmse_mean",
    selection_rule: str = "min",
) -> tuple[float, np.ndarray]:
    """Return ``(alpha_star, rmse_per_alpha)`` with fold-local active screening."""
    summary, per_fold = cv_score_active_alphas(
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
        scoring=scoring,
        return_per_fold=True,
    )
    if not np.all(np.isfinite(summary)):
        raise FloatingPointError("non-finite RMSE encountered during active CV")
    idx = select_alpha_with_rule(per_fold, alphas, rule=selection_rule, summary=summary)
    return float(alphas[idx]), summary


# ----------------------------------------------------------------------
# Fold-local MKL CV: weights learned per fold from kernel-target alignment
# ----------------------------------------------------------------------


def cv_score_alphas_mkl(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "none",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    mkl_top_k: int = 6,
    mkl_mode: str = "alignment",
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Mean validation RMSE per alpha for the MKL-weighted superblock.

    Inside every fold, block weights are *re-learned* from kernel-target
    alignment on the *training* slice only - validation rows never enter
    the weight-learning step. The combined kernel is

    ``K_mkl = sum_b w_b * K_b``    (linear in weights, no squaring)

    where ``K_b = (X A_b^T)(X A_b^T)^T`` is built with **unit per-block scales**
    (the ``s_b`` factor is *not* applied here, regardless of the user-facing
    ``block_scaling``). The MKL math is linear in ``w``; absorbing ``s_b``
    inside ``K_b`` would re-introduce an ``s_b^2`` factor that breaks the
    documented ``K = sum_b w_b K_b`` identity.
    """
    folds = list(cv.split(X, Y))
    if not folds:
        raise ValueError("cv produced no folds")
    n_folds = len(folds)
    n_alphas = alphas.size
    rmse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    sse_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    count_per_fold = np.zeros((n_folds, n_alphas), dtype=float)
    # ``block_scaling`` is preserved for API symmetry but the MKL kernel is
    # built with unit scales: see the docstring for the math invariant.
    _ = block_scaling
    _ = scale_power
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        # Fold-local centering / feature scaling.
        mode = x_scale if center else "none"
        x_mean_f, x_scale_f = fit_feature_scaler(X_tr, mode=mode)
        Xc_tr = apply_feature_scaler(X_tr, x_mean_f, x_scale_f)
        Xc_va = apply_feature_scaler(X_va, x_mean_f, x_scale_f)
        if center:
            y_mean_f = Y_tr.mean(axis=0)
        else:
            y_mean_f = np.zeros(Y_tr.shape[1])
        Yc_tr = Y_tr - y_mean_f
        # Fold-local operator clones, unit scales, weights.
        ops_f = clone_operator_bank(operators_template, p=Xc_tr.shape[1])
        fit_operator_bank(ops_f, Xc_tr)
        scales_f = np.ones(len(ops_f), dtype=float)
        weights_f = learn_block_weights(
            ops_f, Xc_tr, Yc_tr, scales_f, top_k=mkl_top_k, mode=mkl_mode,
        )
        K_tr, U_tr = mkl_kernel_train(Xc_tr, ops_f, weights_f, scales=scales_f)
        K_va = mkl_kernel_cross(Xc_va, U_tr)
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        for i in range(n_alphas):
            Y_pred = K_va @ Cs[i] + y_mean_f
            rmse_per_fold[fold_idx, i] = _rmse(Y_va, Y_pred)
            sse, count = _sum_squared_error(Y_va, Y_pred)
            sse_per_fold[fold_idx, i] = sse
            count_per_fold[fold_idx, i] = count
    summary = _summarise_alpha_scores(
        rmse_per_fold, sse_per_fold, count_per_fold, scoring,
    )
    if return_per_fold:
        return summary, rmse_per_fold
    return summary


def select_alpha_mkl(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    block_scaling: str = "none",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    mkl_top_k: int = 6,
    mkl_mode: str = "alignment",
    scoring: str = "rmse_mean",
    selection_rule: str = "min",
) -> tuple[float, np.ndarray]:
    """Return ``(alpha_star, rmse_per_alpha)`` for the MKL-weighted model."""
    summary, per_fold = cv_score_alphas_mkl(
        X, Y, operators_template, alphas, cv,
        block_scaling=block_scaling,
        center=center,
        scale_power=scale_power,
        x_scale=x_scale,
        mkl_top_k=mkl_top_k,
        mkl_mode=mkl_mode,
        scoring=scoring,
        return_per_fold=True,
    )
    if not np.all(np.isfinite(summary)):
        raise FloatingPointError("non-finite RMSE encountered during MKL CV")
    idx = select_alpha_with_rule(per_fold, alphas, rule=selection_rule, summary=summary)
    return float(alphas[idx]), summary


# ----------------------------------------------------------------------
# Fold-local branch x operator x alpha selection
# ----------------------------------------------------------------------


def cv_score_branch_global(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    branches: Sequence[str] = ("none", "snv", "msc"),
    block_scaling: str = "none",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold-local CV over ``(branch, operator, alpha)`` triples.

    For every CV fold and every branch in ``branches``, a fresh branch
    transformer is fitted on the training fold only and applied to both the
    training and validation rows. The transformed views are then scored
    against every operator in ``operators_template`` and every alpha in
    ``alphas`` via the standard fold-local kernel + dual Ridge path.

    Parameters
    ----------
    branches : sequence of str
        Branch labels. Each must be one of ``VALID_BRANCHES``.
    scoring : str
        ``"rmse_mean"`` (default) or ``"mse_pooled"``.
    return_per_fold : bool
        When True, also return the ``(n_folds, n_branches, n_ops, n_alpha)``
        per-fold table of validation RMSEs.

    Returns
    -------
    rmse_table : ndarray, shape ``(len(branches), len(operators_template), len(alphas))``
        Per-cell summary score (lower is better).
    grids_used : ndarray, shape ``(len(branches), len(operators_template), len(alphas))``
        The alpha grid used per cell. Currently constant across cells (the
        shared ``alphas`` argument), but materialised for caller convenience.
    per_fold : ndarray, shape ``(n_folds, n_branches, n_ops, n_alpha)``
        Only returned when ``return_per_fold=True``.
    """
    for b in branches:
        if b not in VALID_BRANCHES:
            raise ValueError(
                f"unknown branch {b!r}; expected one of {VALID_BRANCHES}"
            )
    folds = list(cv.split(X, Y))
    if not folds:
        raise ValueError("cv produced no folds")
    n_folds = len(folds)
    n_branches = len(branches)
    n_ops = len(operators_template)
    n_alpha = len(alphas)
    rmse_per_fold = np.zeros((n_folds, n_branches, n_ops, n_alpha), dtype=float)
    sse_per_fold = np.zeros((n_folds, n_branches, n_ops, n_alpha), dtype=float)
    count_per_fold = np.zeros((n_folds, n_branches, n_ops, n_alpha), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        for bi, branch in enumerate(branches):
            preproc = make_branch_preproc(branch)
            if preproc is None:
                X_tr_b = np.asarray(X_tr, dtype=float)
                X_va_b = np.asarray(X_va, dtype=float)
            else:
                # Some branches (e.g. OSC) need supervised fit; pass y_tr.
                try:
                    X_tr_b = np.asarray(
                        preproc.fit_transform(np.asarray(X_tr, dtype=float), Y_tr),
                        dtype=float,
                    )
                except TypeError:
                    X_tr_b = np.asarray(
                        preproc.fit_transform(np.asarray(X_tr, dtype=float)),
                        dtype=float,
                    )
                X_va_b = apply_branch_transform(preproc, X_va)
            for oi in range(n_ops):
                op_template = [operators_template[oi]]
                K_tr, K_va, Yc_tr, y_mean_f, _, _, _, _ = _fold_local_kernels(
                    X_tr_b,
                    X_va_b,
                    Y_tr,
                    op_template,
                    block_scaling,
                    center,
                    scale_power=scale_power,
                    x_scale=x_scale,
                )
                Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
                for ai in range(n_alpha):
                    Y_pred = K_va @ Cs[ai] + y_mean_f
                    rmse_per_fold[fold_idx, bi, oi, ai] = _rmse(Y_va, Y_pred)
                    sse, count = _sum_squared_error(Y_va, Y_pred)
                    sse_per_fold[fold_idx, bi, oi, ai] = sse
                    count_per_fold[fold_idx, bi, oi, ai] = count
    if scoring == "rmse_mean":
        rmse_table = rmse_per_fold.mean(axis=0)
    elif scoring == "mse_pooled":
        total_sse = sse_per_fold.sum(axis=0)
        total_n = count_per_fold.sum(axis=0)
        total_n = np.where(total_n > 0, total_n, 1)
        rmse_table = np.sqrt(total_sse / total_n)
    else:
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    grids_used = np.broadcast_to(alphas, (n_branches, n_ops, n_alpha)).copy()
    if return_per_fold:
        return rmse_table, grids_used, rmse_per_fold
    return rmse_table, grids_used


def select_branch_global(
    X: np.ndarray,
    Y: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    alphas: np.ndarray,
    cv: object,
    branches: Sequence[str] = ("none", "snv", "msc"),
    block_scaling: str = "none",
    center: bool = True,
    scale_power: float = 1.0,
    x_scale: str = "center",
    scoring: str = "rmse_mean",
    selection_rule: str = "min",
) -> tuple[str, int, float, np.ndarray]:
    """Pick the ``(branch, operator, alpha)`` triple minimising fold-local CV RMSE.

    Returns ``(branch_name, operator_index, alpha, rmse_table)``. The
    ``rmse_table`` has shape ``(len(branches), len(operators_template), len(alphas))``
    and is suitable for diagnostics / further analysis.

    With ``selection_rule="1se"`` the ``(branch, operator)`` pair is chosen
    by argmin over the cell summary; the alpha is then chosen with the 1-SE
    rule inside that cell's per-fold column.
    """
    rmse_table, _grids, per_fold = cv_score_branch_global(
        X,
        Y,
        operators_template,
        alphas,
        cv,
        branches=branches,
        block_scaling=block_scaling,
        center=center,
        scale_power=scale_power,
        x_scale=x_scale,
        scoring=scoring,
        return_per_fold=True,
    )
    if not np.all(np.isfinite(rmse_table)):
        raise FloatingPointError(
            "non-finite RMSE encountered during branch_global selection"
        )
    # Pick the (branch, op) cell whose alpha-row minimum is smallest, then
    # apply the configured selection rule on that cell's per-fold matrix.
    cell_min = rmse_table.min(axis=2)
    flat_cell_idx = int(np.argmin(cell_min))
    bi, oi = np.unravel_index(flat_cell_idx, cell_min.shape)
    cell_per_fold = per_fold[:, int(bi), int(oi), :]
    cell_summary = rmse_table[int(bi), int(oi), :]
    ai = select_alpha_with_rule(
        cell_per_fold, np.asarray(alphas, dtype=float),
        rule=selection_rule, summary=cell_summary,
    )
    return (
        str(branches[int(bi)]),
        int(oi),
        float(alphas[int(ai)]),
        rmse_table,
    )
