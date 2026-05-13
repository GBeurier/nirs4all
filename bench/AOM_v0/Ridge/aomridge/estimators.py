"""Sklearn-style AOM-Ridge regressor.

The estimator implements three selection policies:

- ``"superblock"`` (primary): one dual Ridge model on the union of strict
  linear operator views, with alpha selected by fold-local CV.
- ``"global"``: hard ``(operator, alpha)`` selection by fold-local CV.
- ``"active_superblock"``: superblock Ridge restricted to a pruned set of
  high-relevance operators chosen on the calibration fold.

All policies operate on strict-linear operators only (Phase 1-6). Nonlinear
branches are out of scope.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from aompls.operators import LinearSpectralOperator
from sklearn.base import BaseEstimator, RegressorMixin

from .kernels import (
    as_2d_y,
    clone_operator_bank,
    compute_block_scales_from_xt,
    fit_operator_bank,
    linear_operator_kernel_train,
    resolve_operator_bank,
)
from .selection import (
    resolve_cv,
    screen_active_operators,
    select_alpha_active,
    select_alpha_superblock,
    select_global,
)
from .solvers import make_alpha_grid, solve_dual_ridge

OperatorBankSpec = str | Sequence[LinearSpectralOperator]


class AOMRidgeRegressor(BaseEstimator, RegressorMixin):
    """Adaptive Operator-Mixture Ridge regressor (dual / kernel).

    The Ridge solution is computed via a dual / kernel formulation that never
    materialises the wide superblock feature matrix. The fitted ``coef_`` has
    shape ``(p, q)`` and lives in the original feature space, so the estimator
    is a drop-in replacement for ``sklearn.linear_model.Ridge``.

    Parameters
    ----------
    selection : str
        ``"superblock"`` (default), ``"global"``, or ``"active_superblock"``.
    operator_bank : str or sequence
        Bank preset name (resolved by ``aompls.banks.bank_by_name``) or an
        explicit sequence of ``LinearSpectralOperator`` instances. Identity is
        always present (added if missing).
    alphas : str or sequence
        ``"auto"`` (trace-relative log grid) or an explicit sequence of
        positive scalars.
    alpha_grid_size, alpha_grid_low, alpha_grid_high
        Parameters for the auto grid (only used when ``alphas == "auto"``).
    alpha : float, optional
        If provided, skip alpha CV and use this fixed alpha.
    cv : int or splitter
        Integer ``KFold`` size or any sklearn-compatible splitter (e.g.
        ``SPXYFold``).
    block_scaling : str
        ``"rms"`` (default) or ``"none"``.
    center : bool
        If ``True``, center ``X`` and ``Y`` before computing kernels.
    scale : bool
        Reserved; ``True`` is not implemented and raises.
    active_top_m : int
        Maximum active operators in active-superblock mode.
    active_diversity_threshold : float
        Cosine threshold for response-based pruning.
    random_state : int, optional
        Seed for the default ``KFold`` shuffle when ``cv`` is an integer.
    solver : str
        ``"auto"``, ``"cholesky"``, or ``"eigh"``.

    Attributes
    ----------
    coef_, intercept_, alpha_, alphas_, dual_coef_, x_mean_, y_mean_,
    block_scales_, selected_operators_, selected_operator_indices_,
    diagnostics_.
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_params_for_fit(self) -> None:
        if self.scale:
            raise NotImplementedError("scale=True is not implemented")
        if self.selection not in ("superblock", "global", "active_superblock"):
            raise ValueError(
                f"unknown selection {self.selection!r}; expected 'superblock', "
                "'global', or 'active_superblock'"
            )
        if self.block_scaling not in ("rms", "none", "scale_power"):
            raise ValueError("block_scaling must be 'rms', 'none', or 'scale_power'")
        if self.solver not in ("auto", "cholesky", "eigh"):
            raise ValueError("solver must be 'auto', 'cholesky', or 'eigh'")
        if not (0.0 <= float(self.scale_power) <= 2.0):
            raise ValueError("scale_power must be in [0, 2]")
        if self.max_grid_expansions < 0:
            raise ValueError("max_grid_expansions must be >= 0")
        if self.x_scale not in ("none", "center", "feature_std", "feature_rms"):
            raise ValueError(
                "x_scale must be one of: none, center, feature_std, feature_rms"
            )
        if self.active_score_method not in ("norm", "kta", "blend"):
            raise ValueError("active_score_method must be 'norm', 'kta', or 'blend'")

    def _resolve_alpha_grid(self, K_full: np.ndarray) -> np.ndarray:
        if isinstance(self.alphas, str):
            if self.alphas != "auto":
                raise ValueError("alphas string must be 'auto'")
            return make_alpha_grid(
                K_full,
                n_grid=self.alpha_grid_size,
                low=self.alpha_grid_low,
                high=self.alpha_grid_high,
            )
        arr = np.asarray(self.alphas, dtype=float)
        if arr.ndim != 1 or arr.size == 0 or np.any(arr <= 0.0):
            raise ValueError("alphas must be a non-empty 1D sequence of positive values")
        return arr

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_active_indices(
        self, X: np.ndarray, Y: np.ndarray, ops_template: list[LinearSpectralOperator]
    ) -> list[int]:
        active, active_scores, pruned = screen_active_operators(
            X,
            Y,
            ops_template,
            block_scaling=self.block_scaling,
            center=self.center,
            top_m=self.active_top_m,
            diversity_threshold=self.active_diversity_threshold,
            keep_identity=True,
            scale_power=self.scale_power,
            x_scale=self.x_scale,
            score_method=self.active_score_method,
            max_per_family=self.active_max_per_family,
        )
        self._active_scores = active_scores
        self._active_pruned = pruned
        return active

    def _select_alpha_with_expansion(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ops_template: Sequence[LinearSpectralOperator],
        cv_obj: object,
        active: bool,
    ) -> tuple[float, np.ndarray, np.ndarray, dict]:
        """Select alpha by fold-local CV, expanding the grid if the optimum
        sits at a boundary. Returns ``(alpha, rmse_per_alpha, alpha_grid, info)``.
        """
        from .solvers import alpha_at_boundary

        low = float(self.alpha_grid_low)
        high = float(self.alpha_grid_high)
        size = int(self.alpha_grid_size)
        info = {"expansions": 0, "boundary_hit": []}
        for _ in range(self.max_grid_expansions + 1):
            alpha_grid = self._build_alpha_grid_from_data(X, Y, ops_template,
                                                          low=low, high=high,
                                                          size=size)
            if active:
                alpha_star, rmse_per_alpha = select_alpha_active(
                    X, Y, ops_template, alpha_grid, cv_obj,
                    block_scaling=self.block_scaling,
                    center=self.center,
                    active_top_m=self.active_top_m,
                    active_diversity_threshold=self.active_diversity_threshold,
                    scale_power=self.scale_power,
                    x_scale=self.x_scale,
                    score_method=self.active_score_method,
                    max_per_family=self.active_max_per_family,
                )
            else:
                alpha_star, rmse_per_alpha = select_alpha_superblock(
                    X, Y, ops_template, alpha_grid, cv_obj,
                    block_scaling=self.block_scaling,
                    center=self.center,
                    scale_power=self.scale_power,
                    x_scale=self.x_scale,
                )
            hit = alpha_at_boundary(rmse_per_alpha, edge_tolerance=2)
            info["boundary_hit"].append(bool(hit))
            if not (self.adaptive_alpha_grid and hit):
                break
            # Expand: shift one decade outward on the side that hit
            idx = int(np.argmin(rmse_per_alpha))
            if idx <= 2:
                low -= 3.0
            else:
                high += 3.0
            info["expansions"] += 1
        return float(alpha_star), rmse_per_alpha, alpha_grid, info

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMRidgeRegressor:
        self._validate_params_for_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        Y2, was_1d = as_2d_y(y)
        if Y2.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n, p = X.shape
        q = Y2.shape[1]
        self._was_1d_y = was_1d

        # Resolve and clone the bank once for the estimator. Per-fold and
        # active screening clone again so they never share state.
        ops_template = resolve_operator_bank(self.operator_bank, p=p)
        all_operator_names = [op.name for op in ops_template]

        # Determine selected operator subset and the alpha used for the final fit.
        cv_obj = resolve_cv(self.cv, random_state=self.random_state)

        t0 = time.perf_counter()

        self._grid_info = {"expansions": 0, "boundary_hit": []}
        if self.selection == "global":
            # Build per-operator alpha grids when auto: each operator scales
            # its own grid by its own kernel trace, so derivative blocks with
            # very different magnitudes still get a well-centred sweep.
            if isinstance(self.alphas, str) and self.global_per_operator_grid:
                per_grids = [
                    self._build_alpha_grid_from_data(X, Y2, [op])
                    for op in ops_template
                ]
                alpha_grid = per_grids[0]   # for diagnostics; alpha is reported separately
            else:
                alpha_grid = self._build_alpha_grid_from_data(
                    X, Y2, [ops_template[0]]
                )
                per_grids = None
            b_star, alpha_star, rmse_table, grids_used = select_global(
                X,
                Y2,
                ops_template,
                alpha_grid,
                cv_obj,
                block_scaling=self.block_scaling,
                center=self.center,
                scale_power=self.scale_power,
                x_scale=self.x_scale,
                per_operator_alpha_grids=per_grids,
            )
            selected_indices = [b_star]
            self._selection_rmse_table = rmse_table
            self._operator_scores = [
                {
                    "index": int(i),
                    "name": all_operator_names[i],
                    "best_rmse": float(rmse_table[i].min()),
                    "best_alpha": float(grids_used[i, int(np.argmin(rmse_table[i]))]),
                }
                for i in range(len(ops_template))
            ]
            # Keep alpha_grid for diagnostics — use the row of the chosen op
            alpha_grid = grids_used[b_star]
        elif self.selection == "active_superblock":
            # Phase A — alpha CV must screen the active subset *inside each
            # fold* (Codex-flagged leak otherwise). Use the full bank as the
            # screening pool and let every fold pick its own subset.
            if self.alpha is not None:
                alpha_grid = self._build_alpha_grid_from_data(X, Y2, ops_template)
                alpha_star = float(self.alpha)
                self._selection_rmse_per_alpha = None
            else:
                (alpha_star, rmse_per_alpha, alpha_grid,
                 self._grid_info) = self._select_alpha_with_expansion(
                    X, Y2, ops_template, cv_obj, active=True,
                )
                self._selection_rmse_per_alpha = rmse_per_alpha
            # Phase B — final active subset for refit comes from the full
            # calibration set (no leak: training data only at this point).
            selected_indices = self._select_active_indices(X, Y2, ops_template)
        else:  # superblock
            selected_indices = list(range(len(ops_template)))
            if self.alpha is not None:
                alpha_grid = self._build_alpha_grid_from_data(X, Y2, ops_template)
                alpha_star = float(self.alpha)
                self._selection_rmse_per_alpha = None
            else:
                (alpha_star, rmse_per_alpha, alpha_grid,
                 self._grid_info) = self._select_alpha_with_expansion(
                    X, Y2, ops_template, cv_obj, active=False,
                )
                self._selection_rmse_per_alpha = rmse_per_alpha

        self._selected_indices = list(selected_indices)
        self._selected_operator_names = [all_operator_names[i] for i in selected_indices]
        self.alphas_ = alpha_grid
        self.alpha_ = float(alpha_star)

        # ------------------------------------------------------------------
        # Final refit on full calibration data with fresh-cloned operators.
        # ------------------------------------------------------------------

        from .preprocessing import apply_feature_scaler, fit_feature_scaler

        if self.center:
            x_mean, x_scale_arr = fit_feature_scaler(X, mode=self.x_scale)
            y_mean = Y2.mean(axis=0)
        else:
            x_mean = np.zeros(p)
            x_scale_arr = np.ones(p)
            y_mean = np.zeros(q)
        Xc = apply_feature_scaler(X, x_mean, x_scale_arr)
        Yc = Y2 - y_mean
        active_template = [ops_template[i] for i in selected_indices]
        ops_final = clone_operator_bank(active_template, p=p)
        fit_operator_bank(ops_final, Xc)
        block_scales = compute_block_scales_from_xt(
            Xc.T, ops_final, block_scaling=self.block_scaling,
            scale_power=self.scale_power,
        )
        K, U = linear_operator_kernel_train(Xc, ops_final, block_scales)
        method = "eigh" if self.solver == "eigh" else "cholesky"
        if self.solver == "auto":
            method = "cholesky"
        C = solve_dual_ridge(K, Yc, alpha=self.alpha_, method=method)

        # `coef_proc` lives in the processed feature space; map back to the
        # original feature space by dividing by the per-feature scale so the
        # estimator predicts directly from raw X without remembering scales.
        coef_proc = U @ C                        # shape (p, q)
        self.coef_ = coef_proc / x_scale_arr[:, None] if coef_proc.ndim == 2 else coef_proc / x_scale_arr
        self.intercept_ = y_mean - x_mean @ self.coef_
        self.dual_coef_ = C
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale_arr
        self.y_mean_ = y_mean
        self.block_scales_ = block_scales
        self.selected_operators_ = list(self._selected_operator_names)
        self.selected_operator_indices_ = list(selected_indices)

        if was_1d:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = float(self.intercept_.ravel()[0])
            self.dual_coef_ = self.dual_coef_.ravel()

        self._fit_time_s = float(time.perf_counter() - t0)
        self._predict_time_s = None
        self._all_operator_names = all_operator_names
        self.diagnostics_ = self._build_diagnostics()
        return self

    def _build_alpha_grid_from_data(
        self,
        X: np.ndarray,
        Y2: np.ndarray,
        operators_template: Sequence[LinearSpectralOperator],
        low: float | None = None,
        high: float | None = None,
        size: int | None = None,
    ) -> np.ndarray:
        """Construct the alpha grid from a centred train kernel.

        We center on the full calibration set here only to get the trace
        scaling for the auto grid; the *selection* and *refit* paths build
        their own fold-local kernels for actual model fitting.
        """
        if not isinstance(self.alphas, str):
            return self._resolve_alpha_grid(K_full=np.zeros((1, 1)))
        low_ = self.alpha_grid_low if low is None else low
        high_ = self.alpha_grid_high if high is None else high
        size_ = self.alpha_grid_size if size is None else size
        from .preprocessing import apply_feature_scaler, fit_feature_scaler

        if self.center:
            x_mean, x_scale_arr = fit_feature_scaler(X, mode=self.x_scale)
        else:
            x_mean = np.zeros(X.shape[1])
            x_scale_arr = np.ones(X.shape[1])
        Xc = apply_feature_scaler(X, x_mean, x_scale_arr)
        ops = clone_operator_bank(operators_template, p=Xc.shape[1])
        fit_operator_bank(ops, Xc)
        scales = compute_block_scales_from_xt(
            Xc.T, ops, block_scaling=self.block_scaling, scale_power=self.scale_power,
        )
        K, _ = linear_operator_kernel_train(Xc, ops, scales)
        return make_alpha_grid(K, n_grid=size_, low=low_, high=high_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        t0 = time.perf_counter()
        # intercept_ already absorbs -x_mean @ coef_, so apply it to raw X
        Y_pred = X @ self.coef_ + self.intercept_
        self._predict_time_s = float(time.perf_counter() - t0)
        return Y_pred

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
        return dict(self.diagnostics_)

    def get_selected_operators(self) -> list[str]:
        return list(self.selected_operators_)

    def _build_diagnostics(self) -> dict:
        # Index of the chosen alpha within the (possibly expanded) grid
        alpha_idx = (
            int(np.argmin(np.abs(np.asarray(self.alphas_) - self.alpha_)))
            if hasattr(self, "alpha_")
            else None
        )
        n_alphas = len(self.alphas_) if hasattr(self, "alphas_") else 0
        boundary = bool(
            alpha_idx is not None
            and (alpha_idx <= 1 or alpha_idx >= n_alphas - 2)
        )
        cv_min_score = (
            float(np.min(self._selection_rmse_per_alpha))
            if getattr(self, "_selection_rmse_per_alpha", None) is not None
            else None
        )
        diag: dict = {
            "model": "AOMRidgeRegressor",
            "selection": self.selection,
            "operator_bank": self.operator_bank if isinstance(self.operator_bank, str)
            else "custom",
            "alpha": float(self.alpha_),
            "alpha_index": alpha_idx,
            "alpha_at_boundary": boundary,
            "alphas": [float(a) for a in self.alphas_],
            "cv": self.cv if isinstance(self.cv, int) else type(self.cv).__name__,
            "cv_min_score": cv_min_score,
            "grid_expansions": int(getattr(self, "_grid_info", {}).get("expansions", 0)),
            "block_scaling": self.block_scaling,
            "scale_power": float(self.scale_power),
            "x_scale": self.x_scale,
            "block_scales": [float(s) for s in self.block_scales_],
            "selected_operator_names": list(self.selected_operators_),
            "selected_operator_indices": list(self.selected_operator_indices_),
            "operator_scores": getattr(self, "_operator_scores", []),
            "block_importance": self._compute_block_importance(),
            "fit_time_s": float(getattr(self, "_fit_time_s", 0.0)),
            "predict_time_s": (
                None if self._predict_time_s is None else float(self._predict_time_s)
            ),
            "coef_available": True,
            "original_feature_space": True,
        }
        if self.selection == "active_superblock":
            diag.update(
                {
                    "active_top_m": int(self.active_top_m),
                    "active_diversity_threshold": float(self.active_diversity_threshold),
                    "active_operator_names": list(self.selected_operators_),
                    "active_operator_indices": list(self.selected_operator_indices_),
                    "active_operator_scores": {
                        name: float(score)
                        for name, score in zip(
                            self.selected_operators_,
                            getattr(self, "_active_scores", []), strict=False,
                        )
                    },
                    "active_pruned_count": int(getattr(self, "_active_pruned", 0)),
                }
            )
        return diag

    def _compute_block_importance(self) -> dict[str, float]:
        """Block-wise importance ``s_b * ||A_b Xc^T||_F`` per selected operator.

        This is a cheap, fold-independent diagnostic that shows the relative
        signal each block contributes. Empty when no kernel has been fit.
        """
        if not hasattr(self, "block_scales_"):
            return {}
        names = list(self.selected_operators_)
        scales = list(self.block_scales_)
        return {n: float(s) for n, s in zip(names, scales, strict=False)}
