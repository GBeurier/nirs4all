"""Local Ridge in AOM score space.

For each test sample, find the k nearest training samples (in the
superblock-AOM kernel-induced metric — optionally after a row-wise SNV or a
training-fitted MSC pre-transform), fit a Ridge restricted to those k
neighbours, and predict.

The selected (branch, k, alpha) triple — and optionally a blend weight beta
that mixes the local prediction with a global AOM-Ridge prediction — is
chosen by fold-local cross-validation. All preprocessing (SNV is row-local,
MSC reference fits on the training fold) and all kernel scaling are
recomputed inside every fold so validation rows never participate in the
neighbour search nor in the local fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from aompls.operators import LinearSpectralOperator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .kernels import (
    as_2d_y,
    clone_operator_bank,
    fit_operator_bank,
    resolve_operator_bank,
)
from .preprocessing import apply_feature_scaler, fit_feature_scaler
from .selection import resolve_cv
from .solvers import solve_dual_ridge

OperatorBankSpec = str | Sequence[LinearSpectralOperator]
LocalWeightBeta = float | str

VALID_BRANCHES = ("none", "snv", "msc")


# ----------------------------------------------------------------------
# Branch pre-transforms
# ----------------------------------------------------------------------


@dataclass
class _BranchState:
    """State produced by fitting a branch on a training fold.

    ``msc_reference`` is the per-feature mean spectrum used by MSC; for SNV
    and ``none`` it is None. ``apply`` is bound at fit time to a callable
    that pre-transforms a test matrix using only the branch's training-fold
    state.
    """

    name: str
    msc_reference: np.ndarray | None


def _snv_transform(X: np.ndarray) -> np.ndarray:
    """Row-wise standard normal variate.

    Each spectrum is independently standardised, so SNV introduces no
    train/test leakage.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / std


def _fit_msc_reference(X: np.ndarray) -> np.ndarray:
    """Fit the MSC reference spectrum from training rows."""
    return X.mean(axis=0)


def _msc_transform(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply MSC using the supplied reference spectrum.

    Each spectrum is regressed against the reference and the linear scatter
    component is removed.
    """
    n, _ = X.shape
    ref_mean = reference.mean()
    ref_centered = reference - ref_mean
    ref_norm_sq = float(ref_centered @ ref_centered)
    if ref_norm_sq <= 1e-30:
        return X.copy()
    out = np.empty_like(X, dtype=float)
    for i in range(n):
        row = X[i]
        row_mean = row.mean()
        b = float((row - row_mean) @ ref_centered) / ref_norm_sq
        a = row_mean - b * ref_mean
        out[i] = (row - a) / b
    return out


def _fit_branch(branch: str, X_tr: np.ndarray) -> _BranchState:
    """Build the branch state from training-fold rows only."""
    if branch == "none":
        return _BranchState(name="none", msc_reference=None)
    if branch == "snv":
        return _BranchState(name="snv", msc_reference=None)
    if branch == "msc":
        return _BranchState(name="msc", msc_reference=_fit_msc_reference(X_tr))
    raise ValueError(f"unknown branch {branch!r}; expected one of {VALID_BRANCHES}")


def _apply_branch(state: _BranchState, X: np.ndarray) -> np.ndarray:
    if state.name == "none":
        return X.astype(float, copy=False)
    if state.name == "snv":
        return _snv_transform(X)
    if state.name == "msc":
        if state.msc_reference is None:
            raise RuntimeError("msc branch missing reference spectrum")
        return _msc_transform(X, state.msc_reference)
    raise ValueError(f"unknown branch state {state.name!r}")


# ----------------------------------------------------------------------
# Per-branch kernel cache
# ----------------------------------------------------------------------


@dataclass
class _BranchKernel:
    """Pre-computed kernels for a single (branch, training set) pair.

    ``K_tr`` is the symmetric ``(n_tr, n_tr)`` train-train kernel.
    ``K_cross`` is ``(n_query, n_tr)`` for the query points (validation or
    test rows).
    """

    branch: _BranchState
    K_tr: np.ndarray
    K_cross: np.ndarray
    Yc_tr: np.ndarray
    y_mean_tr: np.ndarray
    x_mean_tr: np.ndarray
    x_scale_tr: np.ndarray


def _build_branch_kernels(
    branch: str,
    X_tr: np.ndarray,
    X_query: np.ndarray,
    Y_tr: np.ndarray,
    operators_template: Sequence[LinearSpectralOperator],
    block_scaling: str,
    center: bool,
) -> _BranchKernel:
    """Fit a branch on training rows and produce its train + cross kernels.

    All branch and kernel state is fitted on the training rows alone; the
    query rows are projected through the same fitted state.

    The kernel is accumulated per operator as ``K_b = Xb @ Xb.T`` where
    ``Xb = op.transform(Xc)``. This calls each operator only twice
    (training + query) and avoids the column-by-column ``adjoint_vec``
    Python loop in the upstream operator base class — which dominates the
    runtime for high-feature-count datasets.
    """
    branch_state = _fit_branch(branch, X_tr)
    Xb_tr = _apply_branch(branch_state, X_tr)
    Xb_q = _apply_branch(branch_state, X_query)

    if center:
        x_mean, x_scale_arr = fit_feature_scaler(Xb_tr, mode="center")
        y_mean = Y_tr.mean(axis=0)
    else:
        x_mean = np.zeros(Xb_tr.shape[1])
        x_scale_arr = np.ones(Xb_tr.shape[1])
        y_mean = np.zeros(Y_tr.shape[1])
    Xc_tr = apply_feature_scaler(Xb_tr, x_mean, x_scale_arr)
    Xc_q = apply_feature_scaler(Xb_q, x_mean, x_scale_arr)
    Yc_tr = Y_tr - y_mean

    ops = clone_operator_bank(operators_template, p=Xc_tr.shape[1])
    fit_operator_bank(ops, Xc_tr)

    # Per-operator transformed views (one call per (op, fold-branch))
    Xb_tr_views: list[np.ndarray] = []
    Xb_q_views: list[np.ndarray] = []
    for op in ops:
        Xb_tr_views.append(op.transform(Xc_tr))
        Xb_q_views.append(op.transform(Xc_q))

    # Block scales from the transformed views (avoids re-running operators)
    block_scales = _block_scales_from_views(
        Xb_tr_views, block_scaling=block_scaling,
    )

    n_tr = Xc_tr.shape[0]
    n_q = Xc_q.shape[0]
    K_tr = np.zeros((n_tr, n_tr), dtype=float)
    K_cross = np.zeros((n_q, n_tr), dtype=float)
    for Xb_tr_v, Xb_q_v, s in zip(Xb_tr_views, Xb_q_views, block_scales, strict=False):
        s2 = float(s) ** 2
        K_tr += s2 * (Xb_tr_v @ Xb_tr_v.T)
        K_cross += s2 * (Xb_q_v @ Xb_tr_v.T)
    K_tr = 0.5 * (K_tr + K_tr.T)
    return _BranchKernel(
        branch=branch_state,
        K_tr=K_tr,
        K_cross=K_cross,
        Yc_tr=Yc_tr,
        y_mean_tr=y_mean,
        x_mean_tr=x_mean,
        x_scale_tr=x_scale_arr,
    )


def _block_scales_from_views(
    Xb_tr_views: Sequence[np.ndarray], block_scaling: str, eps: float = 1e-12,
) -> np.ndarray:
    """Compute per-operator block scales from the transformed training views.

    Mirrors ``compute_block_scales_from_xt`` but works on ``Xb = X A_b^T``
    directly (so we never need ``apply_cov`` again). ``Xb`` has shape
    ``(n, p)`` and ``||Xb||_F = ||A_b X^T||_F``.
    """
    if block_scaling == "none":
        return np.ones(len(Xb_tr_views), dtype=float)
    if block_scaling not in ("rms", "scale_power"):
        raise ValueError("block_scaling must be 'rms', 'none', or 'scale_power'")
    n, p = Xb_tr_views[0].shape
    denom = max(np.sqrt(float(n) * float(p)), 1.0)
    rms_b = np.array([np.linalg.norm(Xb, "fro") / denom for Xb in Xb_tr_views],
                     dtype=float)
    return 1.0 / (rms_b + eps)


# ----------------------------------------------------------------------
# Local Ridge core math
# ----------------------------------------------------------------------


def _local_alpha_grid(K_tr: np.ndarray, n_grid: int) -> np.ndarray:
    """Build a log alpha grid relative to ``median(diag(K))``.

    The local fit is on a much smaller submatrix than the full kernel, so
    using the median of the diagonal as the trace-equivalent base keeps the
    grid centred on the right magnitude across branches and folds.
    """
    base = float(np.median(np.diag(K_tr)))
    base = max(base, 1e-12)
    return base * np.logspace(-4.0, 4.0, int(n_grid))


def _solve_local_ridge_path(
    K_sub: np.ndarray, Y_sub: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """Solve ``(K_sub + alpha I) C = Y_sub`` for every alpha via one eigh.

    ``K_sub`` is the small ``(k, k)`` neighbour kernel and ``Y_sub`` is its
    matching ``(k, q)`` target slice. Returns an array of shape
    ``(len(alphas), k, q)``.
    """
    K_sym = 0.5 * (K_sub + K_sub.T)
    lam, V = np.linalg.eigh(K_sym)
    lam = np.clip(lam, 0.0, None)
    rhs = V.T @ Y_sub
    out = np.empty((alphas.size, *Y_sub.shape), dtype=float)
    for i, a in enumerate(alphas):
        out[i] = V @ (rhs * (1.0 / (lam + a))[:, None])
    return out


def _topk_indices_per_row(K_cross: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k column indices per row of ``K_cross`` by similarity.

    Output shape is ``(n_query, k)``; columns are not sorted within each row,
    which is fine because we slice symmetric submatrices anyway.
    """
    n_tr = K_cross.shape[1]
    if k >= n_tr:
        return np.broadcast_to(np.arange(n_tr), (K_cross.shape[0], n_tr)).copy()
    return np.argpartition(-K_cross, k - 1, axis=1)[:, :k]


def _batched_local_ridge_path(
    K_tr: np.ndarray,
    K_cross: np.ndarray,
    Yc_tr: np.ndarray,
    y_mean_tr: np.ndarray,
    k: int,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """Vectorised local-Ridge predictions for every (alpha, query) at fixed k.

    Builds the ``(n_query, k, k)`` neighbour-kernel tensor and uses one
    batched linear solve per alpha. NumPy's batched ``solve`` is well
    vectorised across the leading axis, while batched ``eigh`` is not — so
    we pay ``n_alpha`` solves but each is two orders of magnitude faster
    than a single batched ``eigh`` on the same shape.

    Returns predictions of shape ``(n_alpha, n_query, q)``.
    """
    n_query, n_tr = K_cross.shape
    q = Yc_tr.shape[1]
    n_alpha = alpha_grid.size
    k_eff = min(int(k), n_tr)
    idx = _topk_indices_per_row(K_cross, k_eff)         # (n_query, k_eff)
    # Gather submatrices: K_sub[i] = K_tr[idx[i], :][:, idx[i]] -> (n_q, k, k)
    K_sub = K_tr[idx[:, :, None], idx[:, None, :]]
    K_sub = 0.5 * (K_sub + np.swapaxes(K_sub, -1, -2))
    Y_sub = Yc_tr[idx]                                  # (n_query, k_eff, q)
    sims_sub = np.take_along_axis(K_cross, idx, axis=1)  # (n_query, k_eff)
    eye_k = np.eye(k_eff, dtype=float)
    out = np.empty((n_alpha, n_query, q), dtype=float)
    for ai, alpha in enumerate(alpha_grid):
        # Batched solve: (K_sub + alpha I) C = Y_sub, broadcasts over n_query
        C = np.linalg.solve(K_sub + float(alpha) * eye_k, Y_sub)
        # Per-query prediction: sims_sub[i] @ C[i] -> (q,)
        out[ai] = np.einsum("ij,ijl->il", sims_sub, C) + y_mean_tr[None, :]
    return out


def _local_predict(
    K_tr: np.ndarray,
    K_cross: np.ndarray,
    Yc_tr: np.ndarray,
    y_mean_tr: np.ndarray,
    k: int,
    alpha: float,
) -> np.ndarray:
    """Predict every query row by a local Ridge on its k nearest neighbours.

    The neighbour set for query ``i`` is the top-k columns of
    ``K_cross[i, :]`` ranked by similarity (largest first). Implemented via
    the batched alpha path with a 1-element alpha grid.
    """
    preds = _batched_local_ridge_path(
        K_tr, K_cross, Yc_tr, y_mean_tr, int(k), np.asarray([float(alpha)])
    )
    return preds[0]


def _local_predict_path(
    K_tr: np.ndarray,
    K_cross: np.ndarray,
    Yc_tr: np.ndarray,
    y_mean_tr: np.ndarray,
    k_grid: Sequence[int],
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """Predict every query row for every (k, alpha) on the supplied grids.

    Returns an array with shape ``(len(k_grid), len(alpha_grid), n_query, q)``.
    Per fixed k we use a single batched eigh across all queries.
    """
    out = np.empty(
        (len(k_grid), alpha_grid.size, K_cross.shape[0], Yc_tr.shape[1]),
        dtype=float,
    )
    for ki, k in enumerate(k_grid):
        out[ki] = _batched_local_ridge_path(
            K_tr, K_cross, Yc_tr, y_mean_tr, int(k), alpha_grid,
        )
    return out


def _global_predict(
    K_tr: np.ndarray,
    K_cross: np.ndarray,
    Yc_tr: np.ndarray,
    y_mean_tr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Global AOM-Ridge prediction on the full training kernel."""
    C = solve_dual_ridge(K_tr, Yc_tr, alpha=alpha, method="cholesky")
    return K_cross @ C + y_mean_tr


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


@dataclass
class _FittedBranchState:
    """Frozen training-data state that ``predict`` reuses verbatim.

    ``Xb_views`` holds one ``(n_train, p)`` matrix per operator — the
    transformed training spectra. ``predict`` reuses these to project test
    rows through the same operators without re-running the slow upstream
    ``adjoint_vec`` path.
    """

    branch: _BranchState
    K_tr: np.ndarray
    Yc_tr: np.ndarray
    y_mean_tr: np.ndarray
    x_mean_tr: np.ndarray
    x_scale_tr: np.ndarray
    operators: list[LinearSpectralOperator]
    block_scales: np.ndarray
    Xb_views: list[np.ndarray]


class AOMLocalRidge(BaseEstimator, RegressorMixin):
    """Local Ridge in AOM score space.

    For each test sample, find the k nearest training samples (in the
    superblock-AOM kernel-induced metric, optionally after an SNV or MSC
    pre-transform), fit a local Ridge on those k neighbours, predict.

    The 'distance' branch and k are chosen by fold-local CV:
        branches in {"none", "snv", "msc"}
        k_grid = [10, 20, 50, 100]
        alpha_grid is logarithmic relative to ``median(diag(K))``.

    Optionally blend the local prediction with a global AOM-Ridge prediction:
        y_hat = beta * y_local + (1-beta) * y_global, beta in [0, 1].
    """

    def __init__(
        self,
        operator_bank: OperatorBankSpec = "compact",
        distance_branches: Sequence[str] = ("none", "snv", "msc"),
        k_grid: Sequence[int] = (10, 20, 50, 100),
        alpha_grid_size: int = 15,
        cv: int | object = 3,
        local_weight_beta: LocalWeightBeta = "auto",
        random_state: int | None = 0,
        block_scaling: str = "none",
        center: bool = True,
    ) -> None:
        self.operator_bank = operator_bank
        self.distance_branches = distance_branches
        self.k_grid = k_grid
        self.alpha_grid_size = alpha_grid_size
        self.cv = cv
        self.local_weight_beta = local_weight_beta
        self.random_state = random_state
        self.block_scaling = block_scaling
        self.center = center

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params_for_fit(self) -> None:
        for b in self.distance_branches:
            if b not in VALID_BRANCHES:
                raise ValueError(
                    f"unknown distance_branch {b!r}; expected one of {VALID_BRANCHES}"
                )
        if not self.distance_branches:
            raise ValueError("distance_branches must not be empty")
        if self.alpha_grid_size < 1:
            raise ValueError("alpha_grid_size must be >= 1")
        if not self.k_grid:
            raise ValueError("k_grid must not be empty")
        for k in self.k_grid:
            if int(k) < 1:
                raise ValueError("each k in k_grid must be >= 1")
        if self.block_scaling not in ("rms", "none", "scale_power"):
            raise ValueError("block_scaling must be 'rms', 'none', or 'scale_power'")
        if isinstance(self.local_weight_beta, str):
            if self.local_weight_beta != "auto":
                raise ValueError(
                    "local_weight_beta string must be 'auto' or a float in [0, 1]"
                )
        else:
            beta = float(self.local_weight_beta)
            if not (0.0 <= beta <= 1.0):
                raise ValueError("local_weight_beta must lie in [0, 1]")

    # ------------------------------------------------------------------
    # CV-driven hyperparameter selection
    # ------------------------------------------------------------------

    def _cv_select(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ops_template: Sequence[LinearSpectralOperator],
    ) -> dict:
        """Pick the best (branch, k, alpha, beta) by mean validation RMSE."""
        cv_obj = resolve_cv(self.cv, random_state=self.random_state)
        branches = list(self.distance_branches)
        k_grid = [int(k) for k in self.k_grid]
        n_alpha = int(self.alpha_grid_size)
        tune_beta = isinstance(self.local_weight_beta, str)
        beta_grid = (
            np.linspace(0.0, 1.0, 11) if tune_beta
            else np.array([float(self.local_weight_beta)])
        )
        # Per-branch RMSE table accumulated over folds:
        # rmse_acc[branch_idx, k_idx, alpha_idx, beta_idx]
        rmse_acc = np.zeros(
            (len(branches), len(k_grid), n_alpha, beta_grid.size), dtype=float
        )
        # Each fold builds its own trace-relative alpha grid per branch. The
        # *index* of the chosen alpha is what we vote on; alpha values differ
        # per fold but live on a comparable scale, so we report the median.
        per_fold_alpha_grids: list[list[np.ndarray]] = []
        n_folds = 0
        for train_idx, valid_idx in cv_obj.split(X, Y):
            X_tr, X_va = X[train_idx], X[valid_idx]
            Y_tr, Y_va = Y[train_idx], Y[valid_idx]
            fold_alpha_grids: list[np.ndarray] = []
            for b_idx, branch in enumerate(branches):
                bk = _build_branch_kernels(
                    branch=branch,
                    X_tr=X_tr,
                    X_query=X_va,
                    Y_tr=Y_tr,
                    operators_template=ops_template,
                    block_scaling=self.block_scaling,
                    center=self.center,
                )
                alpha_grid = _local_alpha_grid(bk.K_tr, n_alpha)
                fold_alpha_grids.append(alpha_grid)
                # Local predictions for the full (k, alpha) grid
                local_preds = _local_predict_path(
                    bk.K_tr, bk.K_cross, bk.Yc_tr, bk.y_mean_tr,
                    k_grid, alpha_grid,
                )  # shape (n_k, n_alpha, n_va, q)
                # Global prediction at every alpha (one eigh on full K_tr)
                global_path = _solve_local_ridge_path(
                    bk.K_tr, bk.Yc_tr, alpha_grid
                )  # (n_alpha, n_tr, q)
                global_preds = np.empty(
                    (n_alpha, X_va.shape[0], bk.Yc_tr.shape[1]), dtype=float
                )
                for ai in range(n_alpha):
                    global_preds[ai] = bk.K_cross @ global_path[ai] + bk.y_mean_tr
                # Score each (k, alpha, beta) combo
                Y_va_2d = Y_va.reshape(Y_va.shape[0], -1)
                for ki in range(len(k_grid)):
                    for ai in range(n_alpha):
                        loc = local_preds[ki, ai]
                        glo = global_preds[ai]
                        for bi, beta in enumerate(beta_grid):
                            blended = beta * loc + (1.0 - beta) * glo
                            diff = Y_va_2d - blended
                            rmse_acc[b_idx, ki, ai, bi] += float(
                                np.sqrt(np.mean(diff * diff))
                            )
            per_fold_alpha_grids.append(fold_alpha_grids)
            n_folds += 1
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        rmse_mean = rmse_acc / n_folds

        # Pick the optimum jointly across (branch, k, alpha_index, beta).
        flat_idx = int(np.argmin(rmse_mean))
        b_star, k_star, a_star, beta_star = np.unravel_index(
            flat_idx, rmse_mean.shape
        )
        chosen_branch = branches[b_star]
        chosen_k = k_grid[k_star]
        # Chosen alpha is averaged across folds at the chosen alpha index, on
        # the trace-relative scale; this is consistent because the grid
        # spacing is identical across folds (only the base shifts).
        chosen_alpha = float(np.median(
            [per_fold_alpha_grids[f][b_star][a_star] for f in range(n_folds)]
        ))
        chosen_beta = float(beta_grid[beta_star])
        return {
            "branch": chosen_branch,
            "k": chosen_k,
            "alpha": chosen_alpha,
            "beta": chosen_beta,
            "alpha_index": int(a_star),
            "beta_grid": beta_grid,
            "rmse_table": rmse_mean,
            "branches": branches,
            "k_grid": k_grid,
            "n_alpha": n_alpha,
        }

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMLocalRidge:
        self._validate_params_for_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        Y2, was_1d = as_2d_y(y)
        if Y2.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n, p = X.shape
        self._was_1d_y = was_1d
        self._n_train = n

        ops_template = resolve_operator_bank(self.operator_bank, p=p)
        selection = self._cv_select(X, Y2, ops_template)

        self.selected_branch_ = selection["branch"]
        self.selected_k_ = selection["k"]
        self.alpha_ = selection["alpha"]
        self.local_weight_beta_ = selection["beta"]
        self.diagnostics_ = {
            "model": "AOMLocalRidge",
            "branches": selection["branches"],
            "k_grid": selection["k_grid"],
            "alpha_grid_size": selection["n_alpha"],
            "beta_grid": [float(b) for b in selection["beta_grid"]],
            "selected_branch": self.selected_branch_,
            "selected_k": int(self.selected_k_),
            "selected_alpha": float(self.alpha_),
            "selected_beta": float(self.local_weight_beta_),
            "selected_alpha_index": int(selection["alpha_index"]),
            "operator_bank": (
                self.operator_bank if isinstance(self.operator_bank, str)
                else "custom"
            ),
        }

        # Refit the chosen branch on the full training data and freeze state.
        branch_state = _fit_branch(self.selected_branch_, X)
        Xb = _apply_branch(branch_state, X)
        if self.center:
            x_mean, x_scale_arr = fit_feature_scaler(Xb, mode="center")
            y_mean = Y2.mean(axis=0)
        else:
            x_mean = np.zeros(p)
            x_scale_arr = np.ones(p)
            y_mean = np.zeros(Y2.shape[1])
        Xc = apply_feature_scaler(Xb, x_mean, x_scale_arr)
        Yc = Y2 - y_mean

        ops_final = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops_final, Xc)
        Xb_views = [op.transform(Xc) for op in ops_final]
        scales = _block_scales_from_views(Xb_views, block_scaling=self.block_scaling)
        K_tr = np.zeros((n, n), dtype=float)
        for Xb_v, s in zip(Xb_views, scales, strict=False):
            K_tr += float(s) ** 2 * (Xb_v @ Xb_v.T)
        K_tr = 0.5 * (K_tr + K_tr.T)
        self._fit_state = _FittedBranchState(
            branch=branch_state,
            K_tr=K_tr,
            Yc_tr=Yc,
            y_mean_tr=y_mean,
            x_mean_tr=x_mean,
            x_scale_tr=x_scale_arr,
            operators=ops_final,
            block_scales=scales,
            Xb_views=Xb_views,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_fit_state")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        st = self._fit_state
        Xb = _apply_branch(st.branch, X)
        Xc = apply_feature_scaler(Xb, st.x_mean_tr, st.x_scale_tr)
        # Project each test row through every operator and accumulate the
        # cross kernel as a sum of per-operator outer products with the
        # frozen training views.
        n_test = Xc.shape[0]
        n_train = st.K_tr.shape[0]
        K_cross = np.zeros((n_test, n_train), dtype=float)
        for op, Xb_tr_v, s in zip(
            st.operators, st.Xb_views, st.block_scales, strict=False
        ):
            Xb_te = op.transform(Xc)
            K_cross += float(s) ** 2 * (Xb_te @ Xb_tr_v.T)
        local = _local_predict(
            st.K_tr, K_cross, st.Yc_tr, st.y_mean_tr,
            k=int(self.selected_k_), alpha=float(self.alpha_),
        )
        beta = float(self.local_weight_beta_)
        if beta >= 1.0:
            Y_pred = local
        elif beta <= 0.0:
            Y_pred = _global_predict(
                st.K_tr, K_cross, st.Yc_tr, st.y_mean_tr, alpha=float(self.alpha_),
            )
        else:
            globalp = _global_predict(
                st.K_tr, K_cross, st.Yc_tr, st.y_mean_tr, alpha=float(self.alpha_),
            )
            Y_pred = beta * local + (1.0 - beta) * globalp
        if self._was_1d_y:
            return Y_pred.ravel()
        return Y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        Y2, was_1d = as_2d_y(y)
        Y_pred = self.predict(X)
        if was_1d:
            Y_pred = np.asarray(Y_pred).reshape(-1, 1)
        return float(r2_score(Y2, Y_pred, multioutput="uniform_average"))

    def get_diagnostics(self) -> dict:
        check_is_fitted(self, "_fit_state")
        return dict(self.diagnostics_)
