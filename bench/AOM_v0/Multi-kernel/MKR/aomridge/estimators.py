"""Sklearn-style AOM-Ridge regressor.

The estimator implements five selection policies:

- ``"superblock"`` (primary): one dual Ridge model on the union of strict
  linear operator views, with alpha selected by fold-local CV.
- ``"global"``: hard ``(operator, alpha)`` selection by fold-local CV.
- ``"active_superblock"``: superblock Ridge restricted to a pruned set of
  high-relevance operators chosen on the calibration fold.
- ``"mkl"``: MKL-light supervised block weights learned fold-locally from
  kernel-target alignment, plugged into the dual Ridge as
  ``K = sum_b w_b K_b`` (linear in weights). The user-facing
  ``block_scaling`` parameter is ignored in this mode: the learned weights
  ``w_b`` already absorb the role of the per-block scales, so the combined
  kernel is built from raw block kernels (``s_b = 1``).
- ``"branch_global"``: fold-local hard selection over
  ``(branch, operator, alpha)`` triples. The branch transformer (SNV/MSC)
  is fitted on the training fold only. ``coef_`` is unavailable for
  non-trivial branches because the branch transform is non-linear.
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
from .mkl import learn_block_weights, mkl_kernel_train
from .selection import (
    resolve_cv,
    screen_active_operators,
    select_alpha_active,
    select_alpha_mkl,
    select_alpha_superblock,
    select_branch_global,
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
        ``"superblock"`` (default), ``"global"``, ``"active_superblock"``,
        ``"mkl"`` (MKL-light supervised block weights), or
        ``"branch_global"`` (fold-local branch x operator x alpha selection).
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
        selection_rule: str = "min",
        mkl_top_k: int = 6,
        mkl_mode: str = "alignment",
        branches: Sequence[str] = ("none", "snv", "msc"),
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_params_for_fit(self) -> None:
        if self.scale:
            raise NotImplementedError("scale=True is not implemented")
        if self.selection not in (
            "superblock", "global", "active_superblock", "mkl", "branch_global",
        ):
            raise ValueError(
                f"unknown selection {self.selection!r}; expected 'superblock', "
                "'global', 'active_superblock', 'mkl', or 'branch_global'"
            )
        if self.mkl_top_k < 1:
            raise ValueError("mkl_top_k must be >= 1")
        if self.mkl_mode != "alignment":
            raise ValueError("mkl_mode must be 'alignment' (only mode supported)")
        if self.selection == "branch_global":
            from .branches import VALID_BRANCHES
            for b in self.branches:
                if b not in VALID_BRANCHES:
                    raise ValueError(
                        f"unknown branch {b!r}; expected one of {VALID_BRANCHES}"
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
        if self.scoring not in ("rmse", "rmse_mean", "mse_pooled"):
            raise ValueError("scoring must be 'rmse', 'rmse_mean', or 'mse_pooled'")
        if self.selection_rule not in ("min", "1se"):
            raise ValueError("selection_rule must be 'min' or '1se'")

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

    def _scoring_metric(self) -> str:
        """Map the user-facing ``scoring`` to the CV metric name."""
        if self.scoring in ("rmse", "rmse_mean"):
            return "rmse_mean"
        return "mse_pooled"

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
        mode: str,
    ) -> tuple[float, np.ndarray, np.ndarray, dict]:
        """Select alpha by fold-local CV, expanding the grid if the optimum
        sits at a boundary. Returns ``(alpha, rmse_per_alpha, alpha_grid, info)``.

        ``mode`` is one of ``"superblock"``, ``"active"``, or ``"mkl"``.
        """
        from .solvers import alpha_at_boundary

        if mode not in ("superblock", "active", "mkl"):
            raise ValueError(
                f"unknown alpha-CV mode {mode!r}; expected superblock/active/mkl"
            )
        low = float(self.alpha_grid_low)
        high = float(self.alpha_grid_high)
        size = int(self.alpha_grid_size)
        info = {"expansions": 0, "boundary_hit": []}
        # We always evaluate the initial grid; up to ``max_grid_expansions``
        # additional evaluations are performed when the optimum sits at a
        # boundary. Counting matches the number of *evaluated* expansions.
        for attempt in range(self.max_grid_expansions + 1):
            alpha_grid = self._build_alpha_grid_from_data(X, Y, ops_template,
                                                          low=low, high=high,
                                                          size=size)
            scoring = self._scoring_metric()
            if mode == "active":
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
                    scoring=scoring,
                    selection_rule=self.selection_rule,
                )
            elif mode == "mkl":
                alpha_star, rmse_per_alpha = select_alpha_mkl(
                    X, Y, ops_template, alpha_grid, cv_obj,
                    block_scaling=self.block_scaling,
                    center=self.center,
                    scale_power=self.scale_power,
                    x_scale=self.x_scale,
                    mkl_top_k=self.mkl_top_k,
                    mkl_mode=self.mkl_mode,
                    scoring=scoring,
                    selection_rule=self.selection_rule,
                )
            else:  # superblock
                alpha_star, rmse_per_alpha = select_alpha_superblock(
                    X, Y, ops_template, alpha_grid, cv_obj,
                    block_scaling=self.block_scaling,
                    center=self.center,
                    scale_power=self.scale_power,
                    x_scale=self.x_scale,
                    scoring=scoring,
                    selection_rule=self.selection_rule,
                )
            if attempt > 0:
                info["expansions"] = attempt
            # The boundary check must respect ``selection_rule``: under
            # ``"1se"`` the selected alpha may differ from ``argmin(rmse)``,
            # so a 1-SE selection sitting on a grid edge is invisible to a
            # plain argmin probe. We locate the actually-chosen alpha index
            # in the current grid and probe its position.
            chosen_idx = int(
                np.argmin(np.abs(np.asarray(alpha_grid, dtype=float) - float(alpha_star)))
            )
            hit_argmin = alpha_at_boundary(rmse_per_alpha, edge_tolerance=2)
            n_alpha = int(alpha_grid.size)
            hit_chosen = chosen_idx <= 2 or chosen_idx >= n_alpha - 1 - 2
            hit = bool(hit_argmin or hit_chosen)
            info["boundary_hit"].append(hit)
            if not (self.adaptive_alpha_grid and hit):
                break
            if attempt >= self.max_grid_expansions:
                # Already at the cap — do not expand further; keep the last
                # evaluated grid.
                break
            # Expand: shift one decade outward on the side that hit. Use the
            # *chosen* index when the 1-SE rule moved the optimum off argmin.
            idx = chosen_idx if hit_chosen and not hit_argmin else int(
                np.argmin(rmse_per_alpha)
            )
            if idx <= 2:
                low -= 3.0
            else:
                high += 3.0
        return float(alpha_star), rmse_per_alpha, alpha_grid, info

    def _run_global_with_expansion(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ops_template: Sequence[LinearSpectralOperator],
        alpha_grid: np.ndarray,
        cv_obj: object,
        per_operator_alpha_grids: list[np.ndarray] | None,
    ) -> tuple[int, float, np.ndarray, np.ndarray]:
        """Run ``select_global`` and expand the alpha grid on boundary hits.

        Mirrors ``_select_alpha_with_expansion`` for the 2-D
        ``(operator, alpha)`` table: if the chosen ``(operator, alpha)`` pair
        sits within ``edge_tolerance`` of a grid boundary, the per-operator
        alpha grids are extended by 3 decades on the relevant side and the CV
        is re-run, up to ``max_grid_expansions`` times. When ``alpha`` is
        fixed (single-alpha grid) no expansion is performed.
        """
        low = float(self.alpha_grid_low)
        high = float(self.alpha_grid_high)
        size = int(self.alpha_grid_size)
        n_alpha = int(np.asarray(alpha_grid).size)
        # Single-alpha grid (e.g. fixed ``alpha``): nothing to expand.
        fixed_grid = n_alpha == 1
        cur_alpha_grid = np.asarray(alpha_grid, dtype=float)
        cur_per_grids = (
            None if per_operator_alpha_grids is None
            else [np.asarray(g, dtype=float) for g in per_operator_alpha_grids]
        )
        info = {"expansions": 0, "boundary_hit": []}
        for attempt in range(self.max_grid_expansions + 1):
            b_star, alpha_star, rmse_table, grids_used = select_global(
                X,
                Y,
                ops_template,
                cur_alpha_grid,
                cv_obj,
                block_scaling=self.block_scaling,
                center=self.center,
                scale_power=self.scale_power,
                x_scale=self.x_scale,
                per_operator_alpha_grids=cur_per_grids,
                scoring=self._scoring_metric(),
                selection_rule=self.selection_rule,
            )
            if attempt > 0:
                info["expansions"] = attempt
            chosen_alpha_row = grids_used[int(b_star)]
            chosen_idx = int(
                np.argmin(np.abs(chosen_alpha_row - float(alpha_star)))
            )
            n_grid = int(chosen_alpha_row.size)
            hit = (
                not fixed_grid
                and (chosen_idx <= 2 or chosen_idx >= n_grid - 1 - 2)
            )
            info["boundary_hit"].append(bool(hit))
            if not (self.adaptive_alpha_grid and hit):
                break
            if attempt >= self.max_grid_expansions:
                break
            if chosen_idx <= 2:
                low -= 3.0
            else:
                high += 3.0
            # Rebuild per-operator grids over the new (low, high) bracket.
            if isinstance(self.alphas, str) and self.global_per_operator_grid:
                cur_per_grids = [
                    self._build_alpha_grid_from_data(
                        X, Y, [op], low=low, high=high, size=size
                    )
                    for op in ops_template
                ]
                cur_alpha_grid = cur_per_grids[0]
            else:
                cur_alpha_grid = self._build_alpha_grid_from_data(
                    X, Y, [ops_template[0]], low=low, high=high, size=size,
                )
                cur_per_grids = None
        # Stash for diagnostics. ``grids_used`` already reflects the
        # final per-operator grids actually evaluated.
        prev = getattr(self, "_grid_info", None)
        if isinstance(prev, dict):
            prev_expansions = int(prev.get("expansions", 0))
        else:
            prev_expansions = 0
        info["expansions"] = max(prev_expansions, int(info["expansions"]))
        self._grid_info = info
        return int(b_star), float(alpha_star), rmse_table, grids_used

    def _run_branch_global_with_expansion(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ops_template: Sequence[LinearSpectralOperator],
        alpha_grid: np.ndarray,
        cv_obj: object,
    ) -> tuple[str, int, float, np.ndarray, np.ndarray]:
        """Run ``select_branch_global`` with adaptive alpha expansion.

        The branch_global table is 3-D ``(branch, operator, alpha)``. We
        check whether the chosen alpha index sits at a boundary of the
        shared ``alpha_grid`` and extend the grid by 3 decades on that side.
        Single-alpha grids (e.g. fixed ``alpha``) skip expansion.

        Returns ``(branch_name, op_idx, alpha_star, rmse_table, alpha_grid)``;
        the returned ``alpha_grid`` reflects the *final* grid that was
        actually evaluated (so callers can store it as ``alphas_``).
        """
        low = float(self.alpha_grid_low)
        high = float(self.alpha_grid_high)
        size = int(self.alpha_grid_size)
        n_alpha = int(np.asarray(alpha_grid).size)
        fixed_grid = n_alpha == 1
        cur_alpha_grid = np.asarray(alpha_grid, dtype=float)
        info = {"expansions": 0, "boundary_hit": []}
        for attempt in range(self.max_grid_expansions + 1):
            branch_name, op_idx, alpha_star, rmse_table = select_branch_global(
                X,
                Y,
                ops_template,
                cur_alpha_grid,
                cv_obj,
                branches=tuple(self.branches),
                block_scaling=self.block_scaling,
                center=self.center,
                scale_power=self.scale_power,
                x_scale=self.x_scale,
                scoring=self._scoring_metric(),
                selection_rule=self.selection_rule,
            )
            if attempt > 0:
                info["expansions"] = attempt
            chosen_idx = int(
                np.argmin(np.abs(cur_alpha_grid - float(alpha_star)))
            )
            n_grid = int(cur_alpha_grid.size)
            hit = (
                not fixed_grid
                and (chosen_idx <= 2 or chosen_idx >= n_grid - 1 - 2)
            )
            info["boundary_hit"].append(bool(hit))
            if not (self.adaptive_alpha_grid and hit):
                break
            if attempt >= self.max_grid_expansions:
                break
            if chosen_idx <= 2:
                low -= 3.0
            else:
                high += 3.0
            cur_alpha_grid = self._build_alpha_grid_from_data(
                X, Y, [ops_template[0]], low=low, high=high, size=size,
            )
        prev = getattr(self, "_grid_info", None)
        if isinstance(prev, dict):
            prev_expansions = int(prev.get("expansions", 0))
        else:
            prev_expansions = 0
        info["expansions"] = max(prev_expansions, int(info["expansions"]))
        self._grid_info = info
        return (
            str(branch_name),
            int(op_idx),
            float(alpha_star),
            rmse_table,
            cur_alpha_grid,
        )

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
        # Default branch state ("none" — no branch transformer is applied).
        # branch_global selection sets this to the chosen branch.
        chosen_branch = "none"
        if self.selection == "branch_global":
            # Build the alpha grid from the raw + identity view (a stable
            # reference scale that does not depend on which branch wins).
            # When the user fixed `alpha`, score only that single value so
            # branch_global still selects the best (branch, operator) pair.
            if self.alpha is not None:
                alpha_grid = np.asarray([float(self.alpha)], dtype=float)
            else:
                alpha_grid = self._build_alpha_grid_from_data(
                    X, Y2, [ops_template[0]],
                )
            (
                branch_name,
                op_idx,
                alpha_star,
                rmse_table,
                alpha_grid,
            ) = self._run_branch_global_with_expansion(
                X, Y2, ops_template, alpha_grid, cv_obj,
            )
            chosen_branch = branch_name
            selected_indices = [int(op_idx)]
            self._selection_rmse_table = rmse_table
            self._branch_global_branches = list(self.branches)
            self._branch_global_chosen_branch = branch_name
            self._branch_global_chosen_op_idx = int(op_idx)
            self._operator_scores = [
                {
                    "branch": str(self.branches[bi]),
                    "index": int(oi),
                    "name": all_operator_names[oi],
                    "best_rmse": float(rmse_table[bi, oi].min()),
                    "best_alpha": float(
                        alpha_grid[int(np.argmin(rmse_table[bi, oi]))]
                    ),
                }
                for bi in range(len(self.branches))
                for oi in range(len(ops_template))
            ]
            # branch_global picks across (branch, op, alpha); there is no
            # single 1D rmse-per-alpha curve, so this diagnostic is None.
            self._selection_rmse_per_alpha = None
        elif self.selection == "global":
            # Build per-operator alpha grids when auto: each operator scales
            # its own grid by its own kernel trace, so derivative blocks with
            # very different magnitudes still get a well-centred sweep.
            # When ``alpha`` is fixed by the user, we still need to *pick the
            # best operator*, so we score every operator at the single user
            # alpha (a one-element grid) and take the minimum-RMSE operator.
            if self.alpha is not None:
                fixed_alpha_grid = np.asarray([float(self.alpha)], dtype=float)
                per_grids = [fixed_alpha_grid for _ in ops_template]
                alpha_grid = fixed_alpha_grid
            elif isinstance(self.alphas, str) and self.global_per_operator_grid:
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
            (
                b_star,
                alpha_star,
                rmse_table,
                grids_used,
            ) = self._run_global_with_expansion(
                X,
                Y2,
                ops_template,
                alpha_grid,
                cv_obj,
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
                    X, Y2, ops_template, cv_obj, mode="active",
                )
                self._selection_rmse_per_alpha = rmse_per_alpha
            # Phase B — final active subset for refit comes from the full
            # calibration set (no leak: training data only at this point).
            selected_indices = self._select_active_indices(X, Y2, ops_template)
        elif self.selection == "mkl":
            # MKL-light: weights are learned fold-locally inside the alpha CV
            # (so validation rows never see the weight learning), then
            # re-learned on full training data for the final refit.
            selected_indices = list(range(len(ops_template)))
            if self.alpha is not None:
                alpha_grid = self._build_alpha_grid_from_data(X, Y2, ops_template)
                alpha_star = float(self.alpha)
                self._selection_rmse_per_alpha = None
            else:
                (alpha_star, rmse_per_alpha, alpha_grid,
                 self._grid_info) = self._select_alpha_with_expansion(
                    X, Y2, ops_template, cv_obj, mode="mkl",
                )
                self._selection_rmse_per_alpha = rmse_per_alpha
        else:  # superblock
            selected_indices = list(range(len(ops_template)))
            if self.alpha is not None:
                alpha_grid = self._build_alpha_grid_from_data(X, Y2, ops_template)
                alpha_star = float(self.alpha)
                self._selection_rmse_per_alpha = None
            else:
                (alpha_star, rmse_per_alpha, alpha_grid,
                 self._grid_info) = self._select_alpha_with_expansion(
                    X, Y2, ops_template, cv_obj, mode="superblock",
                )
                self._selection_rmse_per_alpha = rmse_per_alpha

        self._selected_indices = list(selected_indices)
        self._selected_operator_names = [all_operator_names[i] for i in selected_indices]
        self.alphas_ = alpha_grid
        self.alpha_ = float(alpha_star)

        # ------------------------------------------------------------------
        # Final refit on full calibration data with fresh-cloned operators.
        # ------------------------------------------------------------------

        from .branches import fit_transform_branch, make_branch_preproc
        from .preprocessing import apply_feature_scaler, fit_feature_scaler

        # If branch_global selected a non-trivial branch, fit a fresh branch
        # transformer on the full calibration set and apply it to X before
        # the standard centering path. Supervised branches (OSC and pipelines
        # that contain it) consume ``y`` at fit time.
        if chosen_branch != "none":
            branch_preproc = make_branch_preproc(chosen_branch)
            X_branched = fit_transform_branch(branch_preproc, X, Y2)
        else:
            branch_preproc = None
            X_branched = X

        if self.center:
            x_mean, x_scale_arr = fit_feature_scaler(X_branched, mode=self.x_scale)
            y_mean = Y2.mean(axis=0)
        else:
            x_mean = np.zeros(p)
            x_scale_arr = np.ones(p)
            y_mean = np.zeros(q)
        Xc = apply_feature_scaler(X_branched, x_mean, x_scale_arr)
        Yc = Y2 - y_mean
        active_template = [ops_template[i] for i in selected_indices]
        ops_final = clone_operator_bank(active_template, p=p)
        fit_operator_bank(ops_final, Xc)
        if self.selection == "mkl":
            # MKL kernel math: ``K_mkl = sum_b w_b K_b``. The weights absorb
            # the role of per-block scales ``s_b`` (any user-facing
            # ``block_scaling`` would otherwise apply ``s_b^2`` *inside* the
            # kernel and break the linear-in-weight identity). We keep the
            # learning + final kernel on raw block kernels (scales = 1).
            block_scales = np.ones(len(ops_final), dtype=float)
            mkl_weights = learn_block_weights(
                ops_final, Xc, Yc, block_scales,
                top_k=self.mkl_top_k, mode=self.mkl_mode,
            )
            K, U = mkl_kernel_train(Xc, ops_final, mkl_weights, scales=block_scales)
            self.mkl_weights_ = mkl_weights
        else:
            block_scales = compute_block_scales_from_xt(
                Xc.T, ops_final, block_scaling=self.block_scaling,
                scale_power=self.scale_power,
            )
            K, U = linear_operator_kernel_train(Xc, ops_final, block_scales)
            self.mkl_weights_ = None
        method = "eigh" if self.solver == "eigh" else "cholesky"
        if self.solver == "auto":
            method = "cholesky"
        C = solve_dual_ridge(K, Yc, alpha=self.alpha_, method=method)

        # `coef_proc` lives in the processed feature space (post-branch,
        # post-centering, post-scaling). When no branch transform is applied
        # we can map it back to the original feature space and expose `coef_`
        # for direct prediction. With a non-trivial branch the relationship
        # between raw and processed inputs is no longer linear, so `coef_` is
        # not available — predict() applies the branch transformer first.
        coef_proc = U @ C                        # shape (p, q)
        self._coef_proc_ = coef_proc
        self._branch_preproc_ = branch_preproc
        self._chosen_branch_ = chosen_branch
        if branch_preproc is None:
            self.coef_ = (
                coef_proc / x_scale_arr[:, None]
                if coef_proc.ndim == 2 else coef_proc / x_scale_arr
            )
            self.intercept_ = y_mean - x_mean @ self.coef_
        else:
            # No raw-space coef; predict() handles the chain explicitly.
            self.coef_ = None
            self.intercept_ = y_mean.copy()
        self.dual_coef_ = C
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale_arr
        self.y_mean_ = y_mean
        self.block_scales_ = block_scales
        self.selected_operators_ = list(self._selected_operator_names)
        self.selected_operator_indices_ = list(selected_indices)

        if was_1d:
            if self.coef_ is not None:
                self.coef_ = self.coef_.ravel()
                self.intercept_ = float(np.asarray(self.intercept_).ravel()[0])
            else:
                self.intercept_ = float(np.asarray(self.intercept_).ravel()[0])
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
        # For MKL the documented kernel math is ``K = sum_b w_b K_b`` with
        # raw block kernels (s_b = 1); using ``block_scaling`` here would
        # produce a different trace and a slightly different alpha grid
        # depending on the user-facing setting, breaking the invariance
        # ``"rms" == "none"`` we just established in the selection path.
        if self.selection == "mkl":
            scales = np.ones(len(ops), dtype=float)
        else:
            scales = compute_block_scales_from_xt(
                Xc.T, ops, block_scaling=self.block_scaling,
                scale_power=self.scale_power,
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
        if getattr(self, "_branch_preproc_", None) is None:
            # intercept_ already absorbs -x_mean @ coef_, so apply it to raw X.
            Y_pred = X @ self.coef_ + self.intercept_
        else:
            # Branched path: apply stored branch transformer, then center +
            # scale, then predict with the proc-space coefficient. The
            # intercept stored is the y_mean (no x-side correction needed
            # because the centered features are subtracted explicitly here).
            from .preprocessing import apply_feature_scaler

            X_branched = np.asarray(
                self._branch_preproc_.transform(X), dtype=float,
            )
            Xc = apply_feature_scaler(X_branched, self.x_mean_, self.x_scale_)
            coef_proc = self._coef_proc_
            if self._was_1d_y:
                coef_proc_2d = coef_proc.reshape(coef_proc.shape[0], -1)
                Y_pred = (Xc @ coef_proc_2d).ravel() + float(self.intercept_)
            else:
                Y_pred = Xc @ coef_proc + self.intercept_
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
        # CV score: use per-alpha (superblock/active/mkl) or full table
        # (global / branch_global).
        rmse_per_alpha = getattr(self, "_selection_rmse_per_alpha", None)
        rmse_table = getattr(self, "_selection_rmse_table", None)
        if rmse_per_alpha is not None:
            cv_min_score = float(np.min(rmse_per_alpha))
        elif rmse_table is not None:
            cv_min_score = float(np.min(rmse_table))
        else:
            cv_min_score = None
        chosen_branch = getattr(self, "_chosen_branch_", "none")
        coef_available = chosen_branch == "none"
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
            "selection_rule": self.selection_rule,
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
            "coef_available": coef_available,
            "original_feature_space": coef_available,
            "chosen_branch": chosen_branch,
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
        if self.selection == "mkl":
            mkl_w = getattr(self, "mkl_weights_", None)
            diag.update(
                {
                    "mkl_top_k": int(self.mkl_top_k),
                    "mkl_mode": self.mkl_mode,
                    "mkl_weights": (
                        [float(w) for w in mkl_w] if mkl_w is not None else []
                    ),
                    "mkl_operator_weights": (
                        {
                            name: float(w)
                            for name, w in zip(
                                self.selected_operators_, mkl_w, strict=False,
                            )
                        }
                        if mkl_w is not None
                        else {}
                    ),
                }
            )
        if self.selection == "branch_global":
            diag.update(
                {
                    "branches": list(self.branches),
                    "chosen_branch": chosen_branch,
                    "chosen_operator_index": int(
                        getattr(self, "_branch_global_chosen_op_idx", -1)
                    ),
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
