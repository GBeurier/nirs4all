"""AOM Ridge-PLS estimator.

Implements the model described in ``bench/AOM_v0/Ridge/Ridge-PLS.md``:

```text
1. Build superblock Z = [s_1 Z_1, ..., s_B Z_B]   (Z_b = T_b(X), s_b = sqrt(n)/||Z_b||_F)
2. Center (and optionally scale) Z and y on training rows only.
3. Fit PLSRegression(scale=False) on (Zs, Ys) with n_components=H.
4. Get x_scores T = Zs @ R   (R = pls.x_rotations_).
5. Ridge on scores: C = (T^T T + alpha I)^-1 T^T Ys.
6. coef_z = R @ C.
7. predict(X) -> Z_test -> Zs_test -> T_test -> Yhat = T_test @ C, then de-standardise.
```

Crucial: the final regression is the *closed-form ridge on the PLS scores*; we
never call ``pls_.predict`` (which would yield classic PLS, not Ridge-PLS).

Phase 1 (MVP) and Phase 2 (math diagnostics) are implemented inside
``AOMRidgePLS``. Phase 3 (CV over ``n_components`` and ``ridge_alpha``) is
implemented by the companion ``AOMRidgePLSCV`` wrapper.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from aompls.operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WhittakerOperator,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from .kernels import (
    as_2d_y,
    clone_operator_bank,
    fit_operator_bank,
    resolve_operator_bank,
)

OperatorBankSpec = str | Sequence[LinearSpectralOperator]

# Operators whose action is a fixed linear map ``A_b`` (the same for any input).
# For these we can back-project the superblock coefficient into the original
# spectral space as ``sum_b s_b A_b^T coef_block_b``.
_LINEAR_OPERATOR_TYPES: tuple[type, ...] = (
    IdentityOperator,
    SavitzkyGolayOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    NorrisWilliamsOperator,
    WhittakerOperator,
    ComposedOperator,
)


def _is_linear_operator(op: LinearSpectralOperator) -> bool:
    """Return True iff ``op`` is a strict-linear primitive with a fixed matrix.

    ``ComposedOperator`` qualifies only if every stage is itself a fixed linear
    primitive (recursive check).
    """
    if isinstance(op, ComposedOperator):
        return all(_is_linear_operator(stage) for stage in op.stages)
    return isinstance(op, _LINEAR_OPERATOR_TYPES)


# ----------------------------------------------------------------------
# Superblock construction (used by both the estimator and tests)
# ----------------------------------------------------------------------


def _frobenius_block_scales(blocks: Sequence[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    """Compute ``s_b = sqrt(n) / ||Z_b||_F`` per block (one scalar per block)."""
    n = blocks[0].shape[0]
    sqrt_n = float(np.sqrt(n))
    out = np.empty(len(blocks), dtype=float)
    for i, Zb in enumerate(blocks):
        norm = float(np.linalg.norm(Zb, ord="fro"))
        out[i] = sqrt_n / max(norm, eps)
    return out


def _build_superblock(
    X: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    block_scaling: str,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Apply each operator to ``X``, scale, concatenate.

    Returns ``(Z, block_scales, block_slices)`` where ``Z`` has shape
    ``(n, B*p)`` (each block is shape ``(n, p)`` for strict-linear operators)
    and ``block_slices[b]`` indexes the columns of ``Z`` belonging to block
    ``b``.
    """
    blocks: list[np.ndarray] = []
    for op in operators:
        Zb = op.transform(X)
        blocks.append(np.asarray(Zb, dtype=float))
    if block_scaling == "frobenius":
        scales = _frobenius_block_scales(blocks)
    elif block_scaling == "none":
        scales = np.ones(len(operators), dtype=float)
    else:
        raise ValueError(f"unknown block_scaling: {block_scaling!r}")
    scaled = [s * Zb for s, Zb in zip(scales, blocks, strict=True)]
    slices: list[slice] = []
    start = 0
    for Zb in scaled:
        stop = start + Zb.shape[1]
        slices.append(slice(start, stop))
        start = stop
    Z = np.concatenate(scaled, axis=1)
    return Z, scales, slices


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


class AOMRidgePLS(BaseEstimator, RegressorMixin):
    """Adaptive Operator-Mixture Ridge-PLS regressor.

    The model is a PLS decomposition on a multi-block AOM superblock followed
    by a closed-form ridge regression on the resulting scores. The two
    hyperparameters are ``n_components`` (the latent dimensionality ``H``) and
    ``ridge_alpha`` (the L2 penalty on the score-space coefficients).

    Parameters
    ----------
    operator_bank : str or sequence of LinearSpectralOperator
        Operator bank preset name (resolved via ``aompls.banks.bank_by_name``)
        or an explicit sequence.
    n_components : int
        Number of PLS components (``H``).
    ridge_alpha : float
        Absolute ridge penalty on the score-space coefficients. Must be
        non-negative; values of ``0`` reduce to vanilla PLS-on-superblock.
    ridge_alpha_mode : str
        Either ``"absolute"`` (use ``ridge_alpha`` directly) or
        ``"relative_to_score_variance"`` (multiply ``ridge_alpha`` by the
        median of ``diag(T^T T)`` measured on the train fold). Default is
        ``"absolute"``.
    block_scaling : str
        Either ``"frobenius"`` (``s_b = sqrt(n) / ||Z_b||_F``) or ``"none"``
        (raw blocks, ``s_b = 1``). Default ``"frobenius"``.
    column_scaling : bool
        If True, divide the superblock columns by their per-column std.
    center_y : bool
        If True, center the target (default).
    scale_y : bool
        If True, divide the centered target by its std.
    cv : int or splitter
        Reserved for the ``AOMRidgePLSCV`` wrapper. Ignored by ``AOMRidgePLS``
        itself but kept on the constructor so user-supplied dictionaries can
        be passed through unchanged.
    random_state : int
        Random state for the optional CV splitter.
    max_iter, tol : forwarded to ``sklearn.cross_decomposition.PLSRegression``.

    Attributes
    ----------
    coef_z_ : ndarray of shape (B*p, q)
        Coefficient in the standardised superblock space.
    coef_in_original_space_ : ndarray of shape (p, q) or None
        Linear-only back-projection of ``coef_z_`` into the original feature
        space. ``None`` when the bank contains a non-fixed-linear operator.
    coef_by_block_ : list of ndarray
        One ``(p, q)`` slice of ``coef_z_`` per operator block.
    pls_ : sklearn.cross_decomposition.PLSRegression
        The fitted PLS object (used only for its rotations / scores).
    rotations_ : ndarray of shape (B*p, H)
        The PLS rotation matrix ``R`` used to compute scores.
    score_diag_ : ndarray of shape (H,)
        Diagonal of ``T^T T`` (one value per latent component).
    shrinkage_factors_ : ndarray of shape (H,)
        Per-component shrinkage ``d_h / (d_h + alpha)``.
    effective_components_ : float
        Sum of the shrinkage factors (continuous analogue of ``H``).
    block_importance_ : ndarray of shape (B,)
        Aggregated rotation L2 norm per block, weighted by per-component
        shrinkage. Higher = more contribution to predictions.
    block_scales_ : ndarray of shape (B,)
        Per-block scaling ``s_b`` learned on the training fold.
    block_slices_ : list of slice
        Column slice in the standardised superblock for each operator.
    """

    def __init__(
        self,
        operator_bank: OperatorBankSpec = "compact",
        n_components: int = 10,
        ridge_alpha: float = 1.0,
        ridge_alpha_mode: str = "absolute",
        block_scaling: str = "frobenius",
        column_scaling: bool = False,
        center_y: bool = True,
        scale_y: bool = False,
        cv: int | object = 5,
        random_state: int = 0,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.operator_bank = operator_bank
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha
        self.ridge_alpha_mode = ridge_alpha_mode
        self.block_scaling = block_scaling
        self.column_scaling = column_scaling
        self.center_y = center_y
        self.scale_y = scale_y
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if float(self.ridge_alpha) < 0.0:
            raise ValueError("ridge_alpha must be >= 0")
        if self.ridge_alpha_mode not in ("absolute", "relative_to_score_variance"):
            raise ValueError(
                "ridge_alpha_mode must be 'absolute' or "
                "'relative_to_score_variance'"
            )
        if self.block_scaling not in ("frobenius", "none"):
            raise ValueError("block_scaling must be 'frobenius' or 'none'")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgePLS:
        self._validate_params()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        Y2, was_1d = as_2d_y(y)
        if Y2.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n, p = X.shape
        self._was_1d_y = bool(was_1d)
        t0 = time.perf_counter()

        # Resolve / clone / fit the operator bank on this training fold only.
        ops_template = resolve_operator_bank(self.operator_bank, p=p)
        ops = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops, X)

        # Build the superblock with fold-local Frobenius scales.
        Z, block_scales, block_slices = _build_superblock(
            X, ops, block_scaling=self.block_scaling
        )

        # Center / scale Z (train-fold only).
        z_mean = Z.mean(axis=0)
        Zc = Z - z_mean
        if self.column_scaling:
            z_scale = Zc.std(axis=0, ddof=0)
            z_scale = np.where(z_scale > 1e-12, z_scale, 1.0)
        else:
            z_scale = np.ones(Z.shape[1], dtype=float)
        Zs = Zc / z_scale

        # Center / scale y.
        if self.center_y:
            y_mean = Y2.mean(axis=0)
        else:
            y_mean = np.zeros(Y2.shape[1], dtype=float)
        Yc = Y2 - y_mean
        if self.scale_y:
            y_scale = Yc.std(axis=0, ddof=0)
            y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
        else:
            y_scale = np.ones(Y2.shape[1], dtype=float)
        Ys = Yc / y_scale

        # Fit PLS with internal scaling disabled (we already centred / scaled).
        h_max = max(1, min(int(self.n_components), Zs.shape[1], n - 1))
        pls = PLSRegression(
            n_components=h_max,
            scale=False,
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        )
        pls.fit(Zs, Ys)
        R = np.asarray(pls.x_rotations_, dtype=float)            # (B*p, H)
        T = Zs @ R                                                # (n, H)

        # Resolve ridge alpha (absolute vs relative-to-score-variance).
        d = np.einsum("ij,ij->j", T, T)                           # diag(T^T T)
        d = np.maximum(d, 0.0)
        if self.ridge_alpha_mode == "relative_to_score_variance":
            base = float(np.median(d)) if d.size else 0.0
            alpha_eff = float(self.ridge_alpha) * base
        else:
            alpha_eff = float(self.ridge_alpha)

        # Closed-form ridge on the PLS scores.
        H = T.shape[1]
        G = T.T @ T + alpha_eff * np.eye(H)
        rhs = T.T @ Ys
        C = np.linalg.solve(G, rhs)                              # (H, q)

        coef_z = R @ C                                            # (B*p, q)

        # Diagnostics.
        shrink = d / (d + alpha_eff) if alpha_eff > 0.0 else np.ones_like(d)
        # Per-block rotation importance, optionally weighted by shrinkage.
        # I_b = sum_h shrink_h * ||R[slice_b, h]||^2
        block_importance = np.zeros(len(block_slices), dtype=float)
        for b, sl in enumerate(block_slices):
            R_b = R[sl, :]
            comp_norms = np.einsum("ij,ij->j", R_b, R_b)
            block_importance[b] = float(np.dot(shrink, comp_norms))

        # Linear back-projection ``beta = sum_b s_b A_b^T C_b`` (column-rescaled
        # if column_scaling is enabled). Available only when every operator has
        # a fixed linear matrix.
        coef_in_original = None
        if all(_is_linear_operator(op) for op in ops):
            coef_in_original = np.zeros((p, Y2.shape[1]), dtype=float)
            for op, s, sl in zip(ops, block_scales, block_slices, strict=True):
                C_b_std = coef_z[sl, :]
                C_b = C_b_std / z_scale[sl, None]                # un-do column scaling
                A = op.matrix(p)                                 # (p, p)
                coef_in_original += float(s) * (A.T @ C_b)
            # ``coef_in_original`` is in y-standardised space; rescale back.
            coef_in_original = coef_in_original * y_scale

        # Persist state.
        self._operators_ = ops
        self.block_slices_ = block_slices
        self.block_scales_ = block_scales
        self.z_mean_ = z_mean
        self.z_scale_ = z_scale
        self.y_mean_ = y_mean
        self.y_scale_ = y_scale
        self.pls_ = pls
        self.rotations_ = R
        self.x_scores_train_ = T
        self.coef_z_ = coef_z
        self.coef_by_block_ = [coef_z[sl, :] for sl in block_slices]
        self.coef_in_original_space_ = coef_in_original
        self.score_diag_ = d
        self.alpha_effective_ = alpha_eff
        self.shrinkage_factors_ = shrink
        self.effective_components_ = float(np.sum(shrink))
        self.block_importance_ = block_importance
        self.C_ = C
        self.n_components_ = int(H)
        self._operator_names_ = [op.name for op in ops]
        self._fit_time_s = float(time.perf_counter() - t0)
        self._predict_time_s: float | None = None
        return self

    # ------------------------------------------------------------------
    # Predict / score
    # ------------------------------------------------------------------

    def _superblock_test(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted operators to a test matrix and rebuild Zs."""
        scaled_blocks: list[np.ndarray] = []
        for op, s in zip(self._operators_, self.block_scales_, strict=True):
            Zb = np.asarray(op.transform(X), dtype=float)
            scaled_blocks.append(float(s) * Zb)
        Z = np.concatenate(scaled_blocks, axis=1)
        return (Z - self.z_mean_) / self.z_scale_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_z_"):
            raise RuntimeError("predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        t0 = time.perf_counter()
        Zs = self._superblock_test(X)
        T_test = Zs @ self.rotations_
        Yhat_std = T_test @ self.C_                              # (n, q)
        Yhat = Yhat_std * self.y_scale_ + self.y_mean_
        self._predict_time_s = float(time.perf_counter() - t0)
        if self._was_1d_y:
            return Yhat.ravel()
        return Yhat

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        Y2, was_1d = as_2d_y(y)
        Y_pred = self.predict(X)
        if was_1d:
            Y_pred = np.asarray(Y_pred).reshape(-1, 1)
        return float(r2_score(Y2, Y_pred, multioutput="uniform_average"))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return a JSON-serialisable summary of the fitted model."""
        return {
            "model": "AOMRidgePLS",
            "n_components": int(self.n_components_),
            "ridge_alpha": float(self.alpha_effective_),
            "ridge_alpha_mode": self.ridge_alpha_mode,
            "block_scaling": self.block_scaling,
            "operator_bank": (
                self.operator_bank if isinstance(self.operator_bank, str) else "custom"
            ),
            "operator_names": list(self._operator_names_),
            "block_scales": [float(s) for s in self.block_scales_],
            "score_diag": [float(x) for x in self.score_diag_],
            "shrinkage_factors": [float(x) for x in self.shrinkage_factors_],
            "effective_components": float(self.effective_components_),
            "block_importance": [float(x) for x in self.block_importance_],
            "fit_time_s": float(self._fit_time_s),
            "predict_time_s": (
                None if self._predict_time_s is None else float(self._predict_time_s)
            ),
        }


# ----------------------------------------------------------------------
# CV wrapper (Sprint 2)
# ----------------------------------------------------------------------


def _resolve_cv(cv: int | object, random_state: int | None) -> object:
    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("integer cv must be >= 2")
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    if hasattr(cv, "split"):
        return cv
    raise TypeError("cv must be an integer or an sklearn-compatible splitter")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


class AOMRidgePLSCV(BaseEstimator, RegressorMixin):
    """K-fold CV wrapper that selects ``(n_components, ridge_alpha)`` for AOMRidgePLS.

    The wrapper performs a fold-local grid search over the cartesian product of
    ``n_components_grid`` and ``ridge_alpha_grid``. Each fold refits the full
    chain (operator bank -> superblock scaling -> PLS -> ridge), so no leakage
    can flow between folds. The best ``(H, alpha)`` pair (lowest mean RMSE) is
    used to refit on the full training set.

    Parameters
    ----------
    operator_bank : str or sequence
    n_components_grid : sequence of int
        Candidate values for ``n_components``. Default ``(2, 3, 5, 7, 10, 15, 20)``.
    ridge_alpha_grid : sequence of float
        Candidate values for ``ridge_alpha``. Default
        ``np.logspace(-4, 4, 9)``.
    ridge_alpha_mode : str
        Forwarded to the inner estimator.
    block_scaling, column_scaling, center_y, scale_y, max_iter, tol :
        Forwarded to the inner estimator.
    cv : int or splitter
        K-fold CV strategy. Integer ``k`` -> shuffled ``KFold(k)``.
    random_state : int

    Attributes
    ----------
    best_n_components_ : int
    best_ridge_alpha_ : float
    best_score_ : float
        Mean CV RMSE at the chosen ``(H, alpha)``.
    cv_results_ : dict
        Contains ``"n_components"``, ``"ridge_alpha"``, and ``"mean_rmse"``
        arrays describing the full grid.
    estimator_ : AOMRidgePLS
        The refit estimator on the full training data.
    """

    def __init__(
        self,
        operator_bank: OperatorBankSpec = "compact",
        n_components_grid: Sequence[int] = (2, 3, 5, 7, 10, 15, 20),
        ridge_alpha_grid: Sequence[float] | None = None,
        ridge_alpha_mode: str = "absolute",
        block_scaling: str = "frobenius",
        column_scaling: bool = False,
        center_y: bool = True,
        scale_y: bool = False,
        cv: int | object = 5,
        random_state: int = 0,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.operator_bank = operator_bank
        self.n_components_grid = n_components_grid
        self.ridge_alpha_grid = ridge_alpha_grid
        self.ridge_alpha_mode = ridge_alpha_mode
        self.block_scaling = block_scaling
        self.column_scaling = column_scaling
        self.center_y = center_y
        self.scale_y = scale_y
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    # --- helpers ---

    def _resolved_alpha_grid(self) -> np.ndarray:
        if self.ridge_alpha_grid is None:
            return np.logspace(-4.0, 4.0, 9)
        arr = np.asarray(self.ridge_alpha_grid, dtype=float)
        if arr.ndim != 1 or arr.size == 0 or np.any(arr < 0.0):
            raise ValueError(
                "ridge_alpha_grid must be a non-empty 1D sequence of >=0 values"
            )
        return arr

    def _make_inner(self, n_components: int, ridge_alpha: float) -> AOMRidgePLS:
        return AOMRidgePLS(
            operator_bank=self.operator_bank,
            n_components=int(n_components),
            ridge_alpha=float(ridge_alpha),
            ridge_alpha_mode=self.ridge_alpha_mode,
            block_scaling=self.block_scaling,
            column_scaling=self.column_scaling,
            center_y=self.center_y,
            scale_y=self.scale_y,
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    # --- fit / predict ---

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgePLSCV:
        X = np.asarray(X, dtype=float)
        Y2, was_1d = as_2d_y(y)
        n_grid = list(self.n_components_grid)
        if not n_grid:
            raise ValueError("n_components_grid must be non-empty")
        a_grid = self._resolved_alpha_grid()
        cv_obj = _resolve_cv(self.cv, random_state=self.random_state)

        mean_rmse = np.full((len(n_grid), len(a_grid)), np.inf, dtype=float)
        for i_h, h in enumerate(n_grid):
            for j_a, alpha in enumerate(a_grid):
                fold_scores: list[float] = []
                for tr, va in cv_obj.split(X, Y2):
                    inner = self._make_inner(h, alpha)
                    inner.fit(X[tr], Y2[tr] if Y2.ndim == 2 else Y2[tr])
                    pred = inner.predict(X[va])
                    fold_scores.append(_rmse(Y2[va].ravel() if was_1d else Y2[va], pred))
                mean_rmse[i_h, j_a] = float(np.mean(fold_scores))

        best_i, best_j = np.unravel_index(int(np.argmin(mean_rmse)), mean_rmse.shape)
        best_h = int(n_grid[best_i])
        best_alpha = float(a_grid[best_j])
        self.cv_results_ = {
            "n_components": np.asarray(n_grid, dtype=int),
            "ridge_alpha": a_grid,
            "mean_rmse": mean_rmse,
        }
        self.best_n_components_ = best_h
        self.best_ridge_alpha_ = best_alpha
        self.best_score_ = float(mean_rmse[best_i, best_j])

        self.estimator_ = self._make_inner(best_h, best_alpha)
        # Restore the original 1D shape on the inner refit.
        self.estimator_.fit(X, Y2.ravel() if was_1d else Y2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("predict called before fit")
        return self.estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.estimator_.score(X, y)

    def get_diagnostics(self) -> dict:
        diag = dict(self.estimator_.get_diagnostics())
        diag.update({
            "best_n_components": int(self.best_n_components_),
            "best_ridge_alpha": float(self.best_ridge_alpha_),
            "best_score": float(self.best_score_),
        })
        return diag
