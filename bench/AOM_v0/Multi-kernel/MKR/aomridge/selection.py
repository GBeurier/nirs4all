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

from .branches import (
    VALID_BRANCHES,
    apply_branch_transform,
    fit_transform_branch,
    make_branch_preproc,
)
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
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the validation score per alpha for one operator subset.

    Returns an array of shape ``(len(alphas),)``. ``operators_template`` is the
    list of operator instances to use as the *superblock* for every fold.
    Folds clone and fit them locally.

    ``scoring``:

    - ``"rmse_mean"`` (default): mean of per-fold RMSE values.
    - ``"mse_pooled"``: pooled mean squared error over all validation rows
      (more accurate when folds have unequal sizes), then sqrt at the end.

    When ``return_per_fold=True``, also returns the full RMSE matrix of shape
    ``(n_folds, n_alphas)`` so the caller can compute fold-level statistics
    (e.g. the standard error required for the 1-SE selection rule).
    """
    if scoring not in ("rmse_mean", "mse_pooled"):
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    rmse_per_fold: list[np.ndarray] = []
    if scoring == "rmse_mean":
        rmse_acc = np.zeros((len(alphas),), dtype=float)
        n_folds = 0
    else:
        sse_acc = np.zeros((len(alphas),), dtype=float)
        total_count = 0
    for train_idx, valid_idx in cv.split(X, Y):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        K_tr, K_va, Yc_tr, y_mean_f, _, _, _, _ = _fold_local_kernels(
            X_tr, X_va, Y_tr, operators_template, block_scaling, center,
            scale_power=scale_power, x_scale=x_scale,
        )
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        fold_rmse = np.zeros((alphas.size,), dtype=float)
        for i in range(alphas.size):
            Y_pred = K_va @ Cs[i] + y_mean_f
            r = _rmse(Y_va, Y_pred)
            fold_rmse[i] = r
            if scoring == "rmse_mean":
                rmse_acc[i] += r
            else:
                diff = (Y_va - Y_pred).ravel()
                sse_acc[i] += float(diff @ diff)
        rmse_per_fold.append(fold_rmse)
        if scoring == "rmse_mean":
            n_folds += 1
        else:
            total_count += int(np.size(Y_va))
    if scoring == "rmse_mean":
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        summary = rmse_acc / n_folds
    else:
        if total_count == 0:
            raise ValueError("cv produced no validation rows")
        summary = np.sqrt(sse_acc / total_count)
    if return_per_fold:
        return summary, np.asarray(rmse_per_fold, dtype=float)
    return summary


# ----------------------------------------------------------------------
# Selection rules (min / 1-SE)
# ----------------------------------------------------------------------


def select_alpha_with_rule(
    rmse_per_fold: np.ndarray,
    alphas: np.ndarray,
    rule: str = "min",
    summary: np.ndarray | None = None,
) -> int:
    """Pick an alpha index from a ``(n_folds, n_alphas)`` RMSE matrix.

    Parameters
    ----------
    rmse_per_fold
        Matrix of per-fold RMSE values, shape ``(n_folds, n_alphas)``.
    alphas
        The alpha grid; only its values matter for the 1-SE tie-break.
    rule
        - ``"min"``: argmin of the summary score.
        - ``"1se"``: among alphas whose summary score is within one standard
          error of the minimum, return the most-regularised one (largest
          alpha). The standard error is the per-fold standard deviation of
          RMSE at the chosen-by-min alpha, divided by ``sqrt(n_folds)``.
    summary
        Optional precomputed per-alpha summary score (shape ``(n_alphas,)``).
        When provided, this score (e.g. pooled-MSE-derived RMSE) drives the
        selection instead of the row-wise mean of ``rmse_per_fold``. The
        1-SE band is still computed from per-fold dispersion since pooled
        MSE has no fold-level decomposition.

    Returns the chosen index into ``alphas``. The caller is responsible for
    looking up ``alphas[idx]``.
    """
    rmse_per_fold = np.asarray(rmse_per_fold, dtype=float)
    if rmse_per_fold.ndim != 2:
        raise ValueError("rmse_per_fold must be a 2D array of shape (n_folds, n_alphas)")
    n_folds, n_alphas = rmse_per_fold.shape
    if n_alphas != len(alphas):
        raise ValueError("alphas length must match rmse_per_fold.shape[1]")
    if summary is None:
        score = rmse_per_fold.mean(axis=0)
    else:
        score = np.asarray(summary, dtype=float)
        if score.shape != (n_alphas,):
            raise ValueError(
                "summary must have shape (n_alphas,) matching rmse_per_fold.shape[1]"
            )
    if rule == "min":
        return int(np.argmin(score))
    if rule == "1se":
        best_idx = int(np.argmin(score))
        if n_folds <= 1:
            return best_idx
        # Standard error at the best alpha (sample std / sqrt(n_folds)).
        se = float(rmse_per_fold[:, best_idx].std(ddof=1) / np.sqrt(n_folds))
        threshold = float(score[best_idx]) + se
        # Pick the most regularised alpha (largest value) within the band.
        candidates = np.where(score <= threshold)[0]
        if candidates.size == 0:
            return best_idx
        # alphas is monotonic-ish but we do not assume sorting: pick by value.
        chosen_pos = int(candidates[int(np.argmax(np.asarray(alphas, dtype=float)[candidates]))])
        return chosen_pos
    raise ValueError("rule must be 'min' or '1se'")


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

    ``selection_rule``:

    - ``"min"`` (default): pick the alpha with the lowest mean CV score.
    - ``"1se"``: pick the most-regularised alpha within one standard error
      of the minimum (computed from the per-fold RMSE matrix).
    """
    if selection_rule not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    rmse, rmse_per_fold = cv_score_alphas(
        X, Y, operators_template, alphas, cv,
        block_scaling=block_scaling, center=center, scale_power=scale_power,
        x_scale=x_scale, scoring=scoring, return_per_fold=True,
    )
    if not np.all(np.isfinite(rmse)):
        raise FloatingPointError("non-finite RMSE encountered during CV")
    idx = select_alpha_with_rule(
        rmse_per_fold, alphas, rule=selection_rule, summary=rmse,
    )
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

    ``selection_rule``:

    - ``"min"`` (default): pick the operator-alpha pair with the lowest mean
      CV score.
    - ``"1se"``: among (operator, alpha) pairs whose mean RMSE is within one
      standard error of the global minimum, pick the most-regularised alpha
      (largest value) and, if tied, the lowest operator index.
    """
    if selection_rule not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    n_ops = len(operators_template)
    n_alpha = len(alphas)
    rmse_table = np.empty((n_ops, n_alpha), dtype=float)
    grids_used = np.empty((n_ops, n_alpha), dtype=float)
    rmse_per_fold_table: list[np.ndarray] = []
    for b, op in enumerate(operators_template):
        op_alphas = per_operator_alpha_grids[b] if per_operator_alpha_grids else alphas
        if len(op_alphas) != n_alpha:
            raise ValueError(
                "per_operator_alpha_grids[b] must have the same length as alphas"
            )
        rmse_b, rmse_b_per_fold = cv_score_alphas(
            X,
            Y,
            [op],
            op_alphas,
            cv,
            block_scaling=block_scaling,
            center=center,
            scale_power=scale_power,
            x_scale=x_scale,
            scoring=scoring,
            return_per_fold=True,
        )
        rmse_table[b] = rmse_b
        grids_used[b] = op_alphas
        rmse_per_fold_table.append(rmse_b_per_fold)
    if not np.all(np.isfinite(rmse_table)):
        raise FloatingPointError("non-finite RMSE encountered during global selection")
    if selection_rule == "min":
        flat_idx = int(np.argmin(rmse_table))
        b_star, a_star = np.unravel_index(flat_idx, rmse_table.shape)
    else:
        # 1-SE rule. Per-operator alpha grids may be on incomparable scales
        # (each grid is trace-relative to its own operator), so picking the
        # "most regularised alpha" by raw value across the joint grid is
        # not meaningful. We therefore:
        #   1. Choose the best operator by minimum row mean RMSE.
        #   2. Apply the standard 1-SE rule *inside* that operator's row.
        # This restores the per-operator regularisation comparison and
        # matches what users expect from "1-SE" (pick the most-regularised
        # alpha within one SE of the operator's best score).
        op_best = rmse_table.min(axis=1)
        b_min = int(np.argmin(op_best))
        a_min_in_row = int(np.argmin(rmse_table[b_min]))
        per_fold_at_min = rmse_per_fold_table[b_min][:, a_min_in_row]
        n_folds = per_fold_at_min.shape[0]
        if n_folds <= 1:
            b_star, a_star = b_min, a_min_in_row
        else:
            se = float(per_fold_at_min.std(ddof=1) / np.sqrt(n_folds))
            threshold = float(rmse_table[b_min, a_min_in_row]) + se
            row = rmse_table[b_min]
            row_alphas = grids_used[b_min]
            cand_idx = np.where(row <= threshold)[0]
            if cand_idx.size == 0:
                a_star = a_min_in_row
            else:
                a_star = int(
                    cand_idx[int(np.argmax(row_alphas[cand_idx]))]
                )
            b_star = b_min
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
    scoring: str = "rmse_mean",
    return_per_fold: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Mean validation RMSE per alpha for active-superblock selection.

    Inside every fold the active subset is screened from the *training* fold
    only, so validation rows never participate in the operator-selection
    decision.

    When ``return_per_fold=True``, also returns the full RMSE matrix of shape
    ``(n_folds, n_alphas)``.
    """
    if scoring not in ("rmse_mean", "mse_pooled"):
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    rmse_per_fold: list[np.ndarray] = []
    if scoring == "rmse_mean":
        rmse_acc = np.zeros((len(alphas),), dtype=float)
        n_folds = 0
    else:
        sse_acc = np.zeros((len(alphas),), dtype=float)
        total_count = 0
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
        fold_rmse = np.zeros((alphas.size,), dtype=float)
        for i in range(alphas.size):
            Y_pred = K_va @ Cs[i] + y_mean_f
            r = _rmse(Y_va, Y_pred)
            fold_rmse[i] = r
            if scoring == "rmse_mean":
                rmse_acc[i] += r
            else:
                diff = (Y_va - Y_pred).ravel()
                sse_acc[i] += float(diff @ diff)
        rmse_per_fold.append(fold_rmse)
        if scoring == "rmse_mean":
            n_folds += 1
        else:
            total_count += int(np.size(Y_va))
    if scoring == "rmse_mean":
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        summary = rmse_acc / n_folds
    else:
        if total_count == 0:
            raise ValueError("cv produced no validation rows")
        summary = np.sqrt(sse_acc / total_count)
    if return_per_fold:
        return summary, np.asarray(rmse_per_fold, dtype=float)
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
    """Return ``(alpha_star, rmse_per_alpha)`` with fold-local active screening.

    ``selection_rule``:

    - ``"min"`` (default): pick the alpha with the lowest mean CV score.
    - ``"1se"``: pick the most-regularised alpha within one standard error
      of the minimum.
    """
    if selection_rule not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    rmse, rmse_per_fold = cv_score_active_alphas(
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
    if not np.all(np.isfinite(rmse)):
        raise FloatingPointError("non-finite RMSE encountered during active CV")
    idx = select_alpha_with_rule(
        rmse_per_fold, alphas, rule=selection_rule, summary=rmse,
    )
    return float(alphas[idx]), rmse


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

    ``scoring``:

    - ``"rmse_mean"`` (default): mean of per-fold RMSE values.
    - ``"mse_pooled"``: pooled MSE over all validation rows, then sqrt.

    When ``return_per_fold=True``, also returns the full RMSE per fold tensor
    of shape ``(n_folds, n_branches, n_ops, n_alphas)``.

    Returns
    -------
    rmse_table : ndarray, shape ``(len(branches), len(operators_template), len(alphas))``
        Aggregated validation RMSE per ``(branch, operator, alpha)`` cell.
    grids_used : ndarray, same shape as ``rmse_table``
        The alpha grid used per cell (constant across cells; materialised
        for caller convenience).
    rmse_per_fold : ndarray, optional
        Only when ``return_per_fold=True``. Shape
        ``(n_folds, n_branches, n_ops, n_alphas)``.
    """
    if scoring not in ("rmse_mean", "mse_pooled"):
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    for b in branches:
        if b not in VALID_BRANCHES:
            raise ValueError(
                f"unknown branch {b!r}; expected one of {VALID_BRANCHES}"
            )
    n_branches = len(branches)
    n_ops = len(operators_template)
    n_alpha = len(alphas)
    rmse_per_fold_list: list[np.ndarray] = []
    if scoring == "rmse_mean":
        rmse_acc = np.zeros((n_branches, n_ops, n_alpha), dtype=float)
        n_folds = 0
    else:
        sse_acc = np.zeros((n_branches, n_ops, n_alpha), dtype=float)
        total_count = 0
    for train_idx, valid_idx in cv.split(X, Y):
        X_tr, X_va = X[train_idx], X[valid_idx]
        Y_tr, Y_va = Y[train_idx], Y[valid_idx]
        fold_rmse = np.zeros((n_branches, n_ops, n_alpha), dtype=float)
        for bi, branch in enumerate(branches):
            preproc = make_branch_preproc(branch)
            if preproc is None:
                X_tr_b = np.asarray(X_tr, dtype=float)
                X_va_b = np.asarray(X_va, dtype=float)
            else:
                # Supervised branches (OSC and OSC-containing pipelines)
                # need ``y`` at fit time; unsupervised branches ignore it.
                # ``fit_transform_branch`` reconciles both signatures.
                X_tr_b = fit_transform_branch(preproc, X_tr, Y_tr)
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
                    r = _rmse(Y_va, Y_pred)
                    fold_rmse[bi, oi, ai] = r
                    if scoring == "rmse_mean":
                        rmse_acc[bi, oi, ai] += r
                    else:
                        diff = (Y_va - Y_pred).ravel()
                        sse_acc[bi, oi, ai] += float(diff @ diff)
        rmse_per_fold_list.append(fold_rmse)
        if scoring == "rmse_mean":
            n_folds += 1
        else:
            total_count += int(np.size(Y_va))
    if scoring == "rmse_mean":
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        rmse_table = rmse_acc / n_folds
    else:
        if total_count == 0:
            raise ValueError("cv produced no validation rows")
        rmse_table = np.sqrt(sse_acc / total_count)
    grids_used = np.broadcast_to(alphas, (n_branches, n_ops, n_alpha)).copy()
    if return_per_fold:
        return rmse_table, grids_used, np.asarray(rmse_per_fold_list, dtype=float)
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

    ``selection_rule``:

    - ``"min"`` (default): pick the cell with the lowest mean CV score.
    - ``"1se"``: among cells whose mean RMSE is within one standard error of
      the global minimum, pick the most-regularised alpha (largest value);
      tie-break by lowest branch index, then lowest operator index.

    Returns ``(branch_name, operator_index, alpha, rmse_table)``. The
    ``rmse_table`` has shape ``(len(branches), len(operators_template), len(alphas))``.
    """
    if selection_rule not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    rmse_table, grids_used, rmse_per_fold = cv_score_branch_global(
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
    n_branches, n_ops, n_alpha = rmse_table.shape
    if selection_rule == "min":
        flat_idx = int(np.argmin(rmse_table))
        bi, oi, ai = np.unravel_index(flat_idx, rmse_table.shape)
    else:
        # 1-SE across the joint (branch, operator, alpha) grid.
        flat_idx = int(np.argmin(rmse_table))
        bi_min, oi_min, ai_min = np.unravel_index(flat_idx, rmse_table.shape)
        per_fold_at_min = rmse_per_fold[:, int(bi_min), int(oi_min), int(ai_min)]
        n_folds = per_fold_at_min.shape[0]
        if n_folds <= 1:
            bi, oi, ai = int(bi_min), int(oi_min), int(ai_min)
        else:
            se = float(per_fold_at_min.std(ddof=1) / np.sqrt(n_folds))
            threshold = float(rmse_table[bi_min, oi_min, ai_min]) + se
            mask = rmse_table <= threshold
            best = (int(bi_min), int(oi_min), int(ai_min))
            best_alpha = float(grids_used[bi_min, oi_min, ai_min])
            for bb in range(n_branches):
                for oo in range(n_ops):
                    for aa in range(n_alpha):
                        if not mask[bb, oo, aa]:
                            continue
                        cand_alpha = float(grids_used[bb, oo, aa])
                        if cand_alpha > best_alpha or (
                            cand_alpha == best_alpha
                            and (bb < best[0] or (bb == best[0] and oo < best[1]))
                        ):
                            best = (bb, oo, aa)
                            best_alpha = cand_alpha
            bi, oi, ai = best
    return (
        str(branches[int(bi)]),
        int(oi),
        float(alphas[int(ai)]),
        rmse_table,
    )


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
    alignment on the *training* slice only — validation rows never enter
    the weight-learning step. The combined kernel is

    ``K_mkl = sum_b w_b * K_b``    (linear in weights, no squaring)

    where ``K_b`` already absorbs the per-block scale ``s_b``.

    ``scoring``:

    - ``"rmse_mean"`` (default): mean of per-fold RMSE values.
    - ``"mse_pooled"``: pooled MSE over all validation rows, then sqrt.

    When ``return_per_fold=True``, also returns the full ``(n_folds, n_alphas)``
    RMSE matrix so the caller can compute the 1-SE rule.
    """
    if scoring not in ("rmse_mean", "mse_pooled"):
        raise ValueError("scoring must be 'rmse_mean' or 'mse_pooled'")
    rmse_per_fold: list[np.ndarray] = []
    if scoring == "rmse_mean":
        rmse_acc = np.zeros((len(alphas),), dtype=float)
        n_folds = 0
    else:
        sse_acc = np.zeros((len(alphas),), dtype=float)
        total_count = 0
    for train_idx, valid_idx in cv.split(X, Y):
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
        # Fold-local operator clones. For MKL the documented kernel math is
        # ``K_mkl = sum_b w_b K_b`` (linear in weights). The per-block scales
        # ``s_b`` are absorbed into the learned weights, so internally we pin
        # ``scales = 1`` regardless of the user-facing ``block_scaling`` —
        # otherwise the combined kernel would be ``sum_b w_b s_b^2 K_b``,
        # which is not what the documented MKL math says.
        ops_f = clone_operator_bank(operators_template, p=Xc_tr.shape[1])
        fit_operator_bank(ops_f, Xc_tr)
        scales_f = np.ones(len(ops_f), dtype=float)
        weights_f = learn_block_weights(
            ops_f, Xc_tr, Yc_tr, scales_f, top_k=mkl_top_k, mode=mkl_mode,
        )
        K_tr, U_tr = mkl_kernel_train(Xc_tr, ops_f, weights_f, scales=scales_f)
        K_va = mkl_kernel_cross(Xc_va, U_tr)
        Cs = solve_dual_ridge_path_eigh(K_tr, Yc_tr, alphas)
        fold_rmse = np.zeros((alphas.size,), dtype=float)
        for i in range(alphas.size):
            Y_pred = K_va @ Cs[i] + y_mean_f
            r = _rmse(Y_va, Y_pred)
            fold_rmse[i] = r
            if scoring == "rmse_mean":
                rmse_acc[i] += r
            else:
                diff = (Y_va - Y_pred).ravel()
                sse_acc[i] += float(diff @ diff)
        rmse_per_fold.append(fold_rmse)
        if scoring == "rmse_mean":
            n_folds += 1
        else:
            total_count += int(np.size(Y_va))
    if scoring == "rmse_mean":
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        summary = rmse_acc / n_folds
    else:
        if total_count == 0:
            raise ValueError("cv produced no validation rows")
        summary = np.sqrt(sse_acc / total_count)
    if return_per_fold:
        return summary, np.asarray(rmse_per_fold, dtype=float)
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
    """Return ``(alpha_star, rmse_per_alpha)`` for the MKL-weighted model.

    ``selection_rule``:

    - ``"min"`` (default): pick the alpha with the lowest mean CV score.
    - ``"1se"``: pick the most-regularised alpha within one standard error
      of the minimum.
    """
    if selection_rule not in ("min", "1se"):
        raise ValueError("selection_rule must be 'min' or '1se'")
    rmse, rmse_per_fold = cv_score_alphas_mkl(
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
    if not np.all(np.isfinite(rmse)):
        raise FloatingPointError("non-finite RMSE encountered during MKL CV")
    idx = select_alpha_with_rule(
        rmse_per_fold, alphas, rule=selection_rule, summary=rmse,
    )
    return float(alphas[idx]), rmse
