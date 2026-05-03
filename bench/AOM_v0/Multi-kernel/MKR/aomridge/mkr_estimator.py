"""AOMMultiKernelRidge — multi-kernel Ridge with explicit per-block weights.

This estimator is the **mkR** model. It composes ``AOMKernelizer``
(centred + trace-normalised block kernels) with a per-block weight
strategy from ``weights`` (uniform / manual / kta / softmax_cv) and a dual
Ridge solve.

Strict-linear bank only in v1: original-space ``coef_`` is exposed; nonlinear
branch preprocessing is deferred to Phase 6.

API (sklearn-compatible):

```python
model = AOMMultiKernelRidge(
    operator_bank="compact",
    weight_strategy="uniform",
    weight_top_k=None,
    weight_init=None,
    weight_n_restarts=3,
    lambda_eta=1e-3,
    weight_max_iter=50,
    alphas="auto",
    alpha_grid_size=50,
    alpha_low=-6.0, alpha_high=6.0,
    alpha_cv_n_splits=5,
    kernel_center=True,
    kernel_normalize="trace",
    feature_scaling="center",
    one_se_rule=False,
    random_state=0,
    verbose=0,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Stored attributes after ``fit``:

- ``eta_``                   simplex weights, shape (B,)
- ``alpha_``                 selected ridge regulariser
- ``dual_coef_``             dual coefficient C, shape (n_train,)
- ``coef_``                  original-space coefficient, shape (p,) (or None)
- ``intercept_``             scalar (= y_mean)
- ``block_names_``           list of operator names
- ``kernel_alignment_max_``  max off-diagonal alignment
- ``inner_cv_rmse_``         softmax_cv inner CV RMSE (NaN otherwise)
- ``weight_diagnostics_``    dict from weight learner
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .kernelizer import AOMKernelizer, kernel_alignment_matrix
from .solvers import make_alpha_grid, solve_dual_ridge_path_eigh
from .weights import (
    WeightLearningResult,
    kta_simplex_weights,
    manual_weights,
    softmax_cv_weights,
    uniform_weights,
)

__all__ = ["AOMMultiKernelRidge"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_K_eta(K_blocks: Sequence[np.ndarray], eta: np.ndarray) -> np.ndarray:
    K = np.zeros_like(K_blocks[0])
    for K_b, w in zip(K_blocks, eta, strict=False):
        if w == 0.0:
            continue
        K += float(w) * K_b
    return 0.5 * (K + K.T)


def _inner_cv_rmse_alpha(
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    eta: np.ndarray,
    alphas: np.ndarray,
    cv: KFold,
) -> np.ndarray:
    """Mean validation RMSE per alpha at a fixed ``eta``."""
    n = y.shape[0]
    rmse_sums = np.zeros(alphas.size, dtype=float)
    n_folds = 0
    indices = np.arange(n)
    for tr_idx, va_idx in cv.split(indices, y):
        K_eta_full = _build_K_eta(K_blocks, eta)
        K_tr = K_eta_full[np.ix_(tr_idx, tr_idx)]
        K_va = K_eta_full[np.ix_(va_idx, tr_idx)]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        try:
            C_path = solve_dual_ridge_path_eigh(K_tr, y_tr, alphas)  # (A, n_tr)
        except np.linalg.LinAlgError:
            return np.full(alphas.size, np.inf)
        rmse_sums += np.array([
            float(np.sqrt(np.mean((y_va - K_va @ C_path[a]) ** 2)))
            for a in range(alphas.size)
        ])
        n_folds += 1
    return rmse_sums / max(n_folds, 1)


def _select_alpha_with_one_se(
    rmse_per_alpha: np.ndarray, alphas: np.ndarray, one_se: bool
) -> tuple[float, int]:
    """Return ``(alpha_star, idx_star)``.

    If ``one_se`` is True, pick the largest alpha within one-standard-error
    of the minimum (more regularised, more parsimonious).
    """
    idx = int(np.argmin(rmse_per_alpha))
    if not one_se:
        return float(alphas[idx]), idx
    best = float(rmse_per_alpha[idx])
    se = float(np.std(rmse_per_alpha) / max(np.sqrt(rmse_per_alpha.size), 1.0))
    threshold = best + se
    # Larger alpha = more regularisation (alphas are increasing log-spaced).
    candidates = np.where(rmse_per_alpha <= threshold)[0]
    idx_se = int(candidates.max())
    return float(alphas[idx_se]), idx_se


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


class AOMMultiKernelRidge(RegressorMixin, BaseEstimator):
    """Multi-kernel Ridge with explicit per-block weights (mkR).

    Parameters
    ----------
    operator_bank : str | sequence
        Name of bank or explicit operator list. See ``aompls.banks``.
    weight_strategy : {"uniform", "manual", "kta", "softmax_cv"}
        How to choose ``eta``.
    weight_init : sequence of B floats, optional
        For ``"manual"``, this *is* the (pre-projection) eta vector.
        For ``"softmax_cv"``, ignored (random Dirichlet starts).
    weight_top_k : int, optional
        For ``"kta"`` and ``"softmax_cv"`` (via inner-CV alpha grid),
        keep only the top-k highest-aligned blocks. ``None`` means all.
    weight_n_restarts : int
        Random restarts for ``"softmax_cv"``.
    lambda_eta : float
        KL-to-uniform regulariser for ``"softmax_cv"``.
    weight_max_iter : int
        L-BFGS-B max iterations per restart for ``"softmax_cv"``.
    alphas : "auto" or sequence of positive floats
        Alpha grid. ``"auto"`` builds a trace-relative grid.
    alpha_grid_size, alpha_low, alpha_high : grid params (when "auto").
    alpha_cv_n_splits : int
        K-fold splits for inner alpha CV (and softmax_cv inner CV).
    kernel_center, kernel_normalize, kernel_eps : passed to AOMKernelizer.
    feature_scaling : {"center"} (only "center" in v1).
    one_se_rule : bool
        Pick largest alpha within one SE of minimum CV RMSE.
    random_state : int | None
    verbose : int
    """

    def __init__(
        self,
        operator_bank: object = "compact",
        *,
        weight_strategy: str = "uniform",
        weight_init: Sequence[float] | None = None,
        weight_top_k: int | None = None,
        weight_top_k_post: int | None = None,
        weight_top_k_post_retune_alpha: bool = True,
        weight_n_restarts: int = 3,
        lambda_eta: float = 1e-3,
        weight_max_iter: int = 50,
        alphas: object = "auto",
        alpha_grid_size: int = 50,
        alpha_low: float = -6.0,
        alpha_high: float = 6.0,
        alpha_cv_n_splits: int = 5,
        kernel_center: bool = True,
        kernel_normalize: str = "trace",
        kernel_eps: float = 1e-12,
        kernel_top_k_active: int | None = None,
        kernel_screen_score_method: str = "norm",
        feature_scaling: str = "center",
        branch_preproc: str = "none",
        add_rbf: bool = False,
        rbf_gammas: list | None = None,
        one_se_rule: bool = False,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        self.operator_bank = operator_bank
        self.weight_strategy = weight_strategy
        self.weight_init = weight_init
        self.weight_top_k = weight_top_k
        self.weight_top_k_post = weight_top_k_post
        self.weight_top_k_post_retune_alpha = weight_top_k_post_retune_alpha
        self.weight_n_restarts = weight_n_restarts
        self.lambda_eta = lambda_eta
        self.weight_max_iter = weight_max_iter
        self.alphas = alphas
        self.alpha_grid_size = alpha_grid_size
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.alpha_cv_n_splits = alpha_cv_n_splits
        self.kernel_center = kernel_center
        self.kernel_normalize = kernel_normalize
        self.kernel_eps = kernel_eps
        self.kernel_top_k_active = kernel_top_k_active
        self.kernel_screen_score_method = kernel_screen_score_method
        self.feature_scaling = feature_scaling
        self.branch_preproc = branch_preproc
        self.add_rbf = add_rbf
        self.rbf_gammas = rbf_gammas
        self.one_se_rule = one_se_rule
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMultiKernelRidge":
        X_arr, y_arr = self._validate_inputs(X, y)
        n, p = X_arr.shape

        # Centre y.
        self._y_mean_ = float(y_arr.mean())
        y_c = y_arr - self._y_mean_

        # Branch preprocessor (fitted on training only). Supports
        # single-branch ``str`` or multi-branch ``list[str]``. Multi-branch
        # mode concatenates the kernels from each branch into one big block
        # list, then learns weights jointly across all blocks.
        if isinstance(self.branch_preproc, (list, tuple)):
            branches = list(self.branch_preproc)
        else:
            branches = [self.branch_preproc or "none"]
        self._branches_ = []  # list of (branch_name, preproc, kernelizer)
        self._is_multi_branch_ = len(branches) > 1
        K_blocks: list[np.ndarray] = []
        block_names_all: list[str] = []
        from .branches import apply_branch_train
        for branch in branches:
            if branch == "none":
                preproc = None
                X_b = X_arr.copy()
            else:
                preproc, X_b = apply_branch_train(branch, X_arr, y_arr)
            kernelizer_b = AOMKernelizer(
                operator_bank=self.operator_bank,
                center=self.kernel_center,
                normalize=self.kernel_normalize,
                eps=self.kernel_eps,
                top_k_active=self.kernel_top_k_active,
                screen_score_method=self.kernel_screen_score_method,
            )
            K_blocks_b = kernelizer_b.fit_transform(X_b, y_arr)
            K_blocks.extend(K_blocks_b)
            for nm in (kernelizer_b.block_names_ or []):
                if self._is_multi_branch_:
                    block_names_all.append(f"{branch}::{nm}")
                else:
                    block_names_all.append(nm)
            # Optional RBF block (DKL-light): adds non-linear similarity
            # alongside the linear AOM operators. Kernel is double-centred and
            # trace-normalised to match the AOM kernel API.
            if self.add_rbf:
                from sklearn.metrics.pairwise import pairwise_distances
                d2_train = pairwise_distances(X_b, X_b, metric='sqeuclidean')
                gamma = self.rbf_gammas[0] if self.rbf_gammas else 1.0 / float(np.median(d2_train[d2_train > 0]) or 1.0)
                K_rbf_raw = np.exp(-gamma * d2_train)
                n_b = K_rbf_raw.shape[0]
                mu_b = K_rbf_raw.mean(axis=1)              # (n,)
                nu_b = float(K_rbf_raw.mean())              # scalar
                # Double-centre: K - mu mu.T-style via subtraction
                K_rbf_c = K_rbf_raw - mu_b[:, None] - mu_b[None, :] + nu_b
                tau_b = n_b / max(float(np.trace(K_rbf_c)), 1e-30)
                K_rbf_norm = tau_b * K_rbf_c
                K_blocks.append(K_rbf_norm)
                nm_rbf = f"rbf_g{gamma:.3g}"
                block_names_all.append(f"{branch}::{nm_rbf}" if self._is_multi_branch_ else nm_rbf)
                if not hasattr(self, '_rbf_per_branch_'):
                    self._rbf_per_branch_ = []
                self._rbf_per_branch_.append({
                    'branch': branch, 'gamma': gamma, 'tau': tau_b,
                    'X_train': X_b.copy(), 'mu': mu_b, 'nu': nu_b,
                })
            self._branches_.append((branch, preproc, kernelizer_b))
        # Single-branch back-compat: keep _branch_ and kernelizer_ scalars
        if not self._is_multi_branch_:
            self._branch_ = self._branches_[0][1]
            self.kernelizer_ = self._branches_[0][2]
            X_arr = X_arr if branches[0] == "none" else apply_branch_train(branches[0], X_arr, y_arr)[1]
        else:
            self._branch_ = None
            self.kernelizer_ = None
        B = len(K_blocks)
        if B < 1:
            raise ValueError("operator bank produced no blocks")
        self.block_names_ = block_names_all
        self.B_ = B

        # Build alpha grid from a representative kernel (use the uniform-mean
        # kernel as reference so the grid is meaningful regardless of weight
        # strategy).
        K_uniform = _build_K_eta(K_blocks, uniform_weights(B))
        if self.alphas == "auto":
            alpha_grid = make_alpha_grid(
                K_uniform,
                n_grid=self.alpha_grid_size,
                low=self.alpha_low,
                high=self.alpha_high,
            )
        else:
            alpha_grid = np.asarray(self.alphas, dtype=float).ravel()
            if alpha_grid.size == 0 or np.any(alpha_grid <= 0.0):
                raise ValueError("alphas must be a non-empty positive sequence")
        self._alpha_grid_ = alpha_grid

        # Choose eta + alpha.
        cv = KFold(
            n_splits=self.alpha_cv_n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        eta, alpha_star, weight_diag = self._choose_eta_alpha(
            K_blocks, y_c, alpha_grid, cv, B
        )
        self.eta_ = eta
        self.alpha_ = float(alpha_star)
        self.weight_diagnostics_ = weight_diag

        # Final fit.
        K_eta = _build_K_eta(K_blocks, eta)
        Ka = K_eta + self.alpha_ * np.eye(n)
        Ka = 0.5 * (Ka + Ka.T)
        cf = cho_factor(Ka, lower=True, check_finite=False)
        C = cho_solve(cf, y_c, check_finite=False)
        self.dual_coef_ = C

        # Recover original-space coef (strict-linear bank only).
        # Single-branch: standard primal coef. Multi-branch: defer to dual
        # prediction (no scalar coef_ defined; predict() routes to dual path).
        if not self._is_multi_branch_:
            kernelizer = self.kernelizer_
            Xc = X_arr - kernelizer.x_mean_
            operators = list(kernelizer.operators_ or [])
            stats = list(kernelizer.block_stats_ or [])
            U_eta = np.zeros(p, dtype=float)
            for op, stat, w in zip(operators, stats, eta, strict=False):
                if w == 0.0:
                    continue
                AXt = op.apply_cov(Xc.T)
                AtAXt = op.adjoint_vec(AXt)
                U_eta += float(w) * float(stat.tau) * (AtAXt @ C)
            self.coef_ = U_eta
            self.intercept_ = float(self._y_mean_ - kernelizer.x_mean_ @ U_eta)
        else:
            # Store per-branch coefs for primal prediction across branches.
            # y_pred = sum_branch (X_branch_c @ coef_branch) + y_mean
            self.coef_ = None
            self.intercept_ = float(self._y_mean_)
            self._coef_branches_: list[tuple[str, object, np.ndarray, np.ndarray]] = []
            cursor = 0
            for branch, preproc, kernelizer_b in self._branches_:
                stats_b = list(kernelizer_b.block_stats_ or [])
                ops_b = list(kernelizer_b.operators_ or [])
                Bb = len(ops_b)
                if branch == "none":
                    X_b = X_arr.copy()
                else:
                    X_b = apply_branch_train(branch, X_arr, y_arr)[1]
                Xc_b = X_b - kernelizer_b.x_mean_
                U_branch = np.zeros(Xc_b.shape[1], dtype=float)
                for j, (op, stat) in enumerate(zip(ops_b, stats_b, strict=False)):
                    w = float(eta[cursor + j])
                    if w == 0.0:
                        continue
                    AXt = op.apply_cov(Xc_b.T)
                    AtAXt = op.adjoint_vec(AXt)
                    U_branch += w * float(stat.tau) * (AtAXt @ C)
                # store the primal coef for this branch (in branch feature space)
                self._coef_branches_.append((branch, preproc, U_branch, kernelizer_b.x_mean_))
                cursor += Bb

        # Diagnostics.
        A = kernel_alignment_matrix(K_blocks)
        if A.shape[0] >= 2:
            off = A - np.eye(A.shape[0])
            self.kernel_alignment_max_ = float(np.max(np.abs(off)))
            self.kernel_alignment_mean_ = float(
                np.sum(np.abs(off)) / max(A.shape[0] * (A.shape[0] - 1), 1)
            )
        else:
            self.kernel_alignment_max_ = 0.0
            self.kernel_alignment_mean_ = 0.0
        self.kernel_alignment_matrix_ = A

        # Inner CV RMSE (final eta+alpha).
        self.inner_cv_rmse_ = float(
            _inner_cv_rmse_alpha(
                K_blocks, y_c, eta, np.array([self.alpha_]), cv
            )[0]
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        # RBF blocks are non-linear; primal coef recovery is not defined for
        # them. Route to dual prediction when add_rbf is set.
        if getattr(self, "add_rbf", False):
            return self.predict_dual(X_arr)
        if getattr(self, "_is_multi_branch_", False):
            # Multi-branch primal: sum per-branch contributions, then add y_mean
            from .branches import apply_branch_transform
            y_pred = np.full(X_arr.shape[0], self._y_mean_, dtype=float)
            for branch, preproc, coef_branch, x_mean_branch in self._coef_branches_:
                X_b = X_arr if preproc is None else apply_branch_transform(preproc, X_arr)
                X_bc = X_b - x_mean_branch
                y_pred += X_bc @ coef_branch
            return y_pred
        # Single-branch primal
        if self._branch_ is not None:
            from .branches import apply_branch_transform
            X_arr = apply_branch_transform(self._branch_, X_arr)
        return X_arr @ self.coef_ + self.intercept_

    def predict_dual(self, X: np.ndarray) -> np.ndarray:
        """Predict via the dual form (cross kernels). Used in tests for
        agreement with primal."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        from .branches import apply_branch_transform
        K_eta_cross = None
        cursor = 0
        rbf_index = 0
        rbf_metas = list(getattr(self, '_rbf_per_branch_', []) or [])
        for branch_idx, (branch, preproc, kernelizer_b) in enumerate(self._branches_):
            X_b = X_arr if preproc is None else apply_branch_transform(preproc, X_arr)
            K_blocks_cross_b = kernelizer_b.transform(X_b)
            if K_eta_cross is None:
                K_eta_cross = np.zeros_like(K_blocks_cross_b[0])
            for K_b, w in zip(K_blocks_cross_b, self.eta_[cursor:cursor + len(K_blocks_cross_b)], strict=False):
                if float(w) == 0.0:
                    continue
                K_eta_cross = K_eta_cross + float(w) * K_b
            cursor += len(K_blocks_cross_b)
            # Optional RBF block per-branch
            if getattr(self, 'add_rbf', False) and rbf_index < len(rbf_metas):
                meta = rbf_metas[rbf_index]
                from sklearn.metrics.pairwise import pairwise_distances
                d2_cross = pairwise_distances(X_b, meta['X_train'], metric='sqeuclidean')
                K_rbf_cross_raw = np.exp(-meta['gamma'] * d2_cross)
                # Centre using training mu, nu
                r_star = K_rbf_cross_raw.mean(axis=1)
                K_rbf_cross_c = K_rbf_cross_raw - r_star[:, None] - meta['mu'][None, :] + meta['nu']
                K_rbf_cross = meta['tau'] * K_rbf_cross_c
                w_rbf = float(self.eta_[cursor])
                if w_rbf != 0.0:
                    K_eta_cross = K_eta_cross + w_rbf * K_rbf_cross
                cursor += 1
                rbf_index += 1
        return K_eta_cross @ self.dual_coef_ + self._y_mean_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(r2_score(y, self.predict(X)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_raw = np.asarray(y, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        # Reject multi-output BEFORE ravelling, otherwise a (n, q>1) target
        # collapses into a confusing length mismatch.
        if y_raw.ndim == 2 and y_raw.shape[1] > 1:
            raise ValueError(
                "AOMMultiKernelRidge does not support multi-output y in v1"
            )
        y_arr = y_raw.ravel()
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D in v1 (multi-output deferred)")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if self.feature_scaling != "center":
            raise NotImplementedError(
                "Only feature_scaling='center' is supported in v1"
            )
        return X_arr, y_arr

    def _check_fitted(self) -> None:
        # Multi-branch sets coef_ to None and uses _coef_branches_ instead.
        if getattr(self, "_is_multi_branch_", False):
            if not hasattr(self, "_coef_branches_"):
                raise RuntimeError("AOMMultiKernelRidge must be fitted first")
            return
        if not hasattr(self, "coef_") or self.coef_ is None:
            raise RuntimeError("AOMMultiKernelRidge must be fitted first")

    def _choose_eta_alpha(
        self,
        K_blocks: Sequence[np.ndarray],
        y_c: np.ndarray,
        alpha_grid: np.ndarray,
        cv: KFold,
        B: int,
    ) -> tuple[np.ndarray, float, dict]:
        strategy = self.weight_strategy
        diag: dict = {"strategy": strategy}
        if strategy == "uniform":
            eta = uniform_weights(B)
            rmse_per_alpha = _inner_cv_rmse_alpha(
                K_blocks, y_c, eta, alpha_grid, cv
            )
            alpha_star, idx = _select_alpha_with_one_se(
                rmse_per_alpha, alpha_grid, self.one_se_rule
            )
            diag.update({
                "alpha_index": int(idx),
                "cv_min_score": float(np.min(rmse_per_alpha)),
                "cv_score_at_selected": float(rmse_per_alpha[idx]),
            })
            return eta, alpha_star, diag
        if strategy == "manual":
            if self.weight_init is None:
                raise ValueError("weight_init must be supplied for 'manual'")
            eta = manual_weights(self.weight_init, B)
            rmse_per_alpha = _inner_cv_rmse_alpha(
                K_blocks, y_c, eta, alpha_grid, cv
            )
            alpha_star, idx = _select_alpha_with_one_se(
                rmse_per_alpha, alpha_grid, self.one_se_rule
            )
            diag.update({
                "alpha_index": int(idx),
                "cv_min_score": float(np.min(rmse_per_alpha)),
            })
            return eta, alpha_star, diag
        if strategy == "kta":
            eta = kta_simplex_weights(K_blocks, y_c, top_k=self.weight_top_k)
            rmse_per_alpha = _inner_cv_rmse_alpha(
                K_blocks, y_c, eta, alpha_grid, cv
            )
            alpha_star, idx = _select_alpha_with_one_se(
                rmse_per_alpha, alpha_grid, self.one_se_rule
            )
            diag.update({
                "alpha_index": int(idx),
                "cv_min_score": float(np.min(rmse_per_alpha)),
            })
            return eta, alpha_star, diag
        if strategy == "pop_greedy":
            eta, alpha_star, pop_diag = self._pop_greedy_select(
                K_blocks, y_c, alpha_grid, cv, B
            )
            diag.update(pop_diag)
            return eta, alpha_star, diag
        if strategy == "softmax_cv":
            res: WeightLearningResult = softmax_cv_weights(
                K_blocks,
                y_c,
                alphas=alpha_grid,
                cv_n_splits=self.alpha_cv_n_splits,
                cv_shuffle=True,
                n_restarts=self.weight_n_restarts,
                lambda_eta=self.lambda_eta,
                max_iter=self.weight_max_iter,
                random_state=self.random_state,
                top_k_post=self.weight_top_k_post,
                top_k_post_retune_alpha=self.weight_top_k_post_retune_alpha,
            )
            diag.update({
                "n_iterations": int(res.n_iterations),
                "converged": bool(res.converged),
                "inner_cv_rmse": float(res.inner_cv_rmse),
                **res.diagnostics,
            })
            return res.eta, res.alpha, diag
        raise ValueError(
            f"unknown weight_strategy {strategy!r}; expected "
            "'uniform', 'manual', 'kta', 'softmax_cv', or 'pop_greedy'"
        )

    def _pop_greedy_select(
        self,
        K_blocks: Sequence[np.ndarray],
        y_c: np.ndarray,
        alpha_grid: np.ndarray,
        cv: KFold,
        B: int,
    ) -> tuple[np.ndarray, float, dict]:
        """Greedy forward selection of kernel blocks (POP-PLS style).

        At each step, find the (block_b, weight_w, alpha) that minimises
        inner-CV RMSE when added to the current accumulated kernel
        ``K_eta = sum_b eta_b K_b``. Stop when relative improvement falls
        below ``pop_tol`` or after ``pop_max_steps`` selections.
        """
        max_steps = int(getattr(self, "pop_max_steps", 5) or 5)
        weight_grid = np.array([0.1, 0.3, 1.0, 3.0])
        tol = 1e-3
        eta = np.zeros(B, dtype=float)
        K_current = np.zeros_like(K_blocks[0])
        # Initial best: uniform over all blocks (sanity reference).
        best_alpha = float(alpha_grid[len(alpha_grid) // 2])
        best_rmse = float("inf")
        selected: list[int] = []
        history: list[tuple[int, float, float, float]] = []
        for step in range(min(max_steps, B)):
            step_best = (None, None, None, float("inf"))
            for b in range(B):
                if b in selected:
                    continue
                for w in weight_grid:
                    K_test = K_current + float(w) * K_blocks[b]
                    rmses = _inner_cv_rmse_alpha(
                        [K_test], y_c, np.array([1.0]), alpha_grid, cv
                    )
                    a_idx = int(np.argmin(rmses))
                    rmse_b = float(rmses[a_idx])
                    if rmse_b < step_best[3]:
                        step_best = (b, float(w), float(alpha_grid[a_idx]), rmse_b)
            if step_best[0] is None:
                break
            improvement = (best_rmse - step_best[3]) / max(best_rmse, 1e-12)
            if step > 0 and improvement < tol:
                break
            best_b, best_w, best_alpha, best_rmse = step_best
            selected.append(best_b)
            eta[best_b] += best_w
            K_current = K_current + best_w * K_blocks[best_b]
            history.append(step_best)
        if eta.sum() <= 0.0:
            eta = uniform_weights(B)
        else:
            eta = eta / eta.sum()
        return eta, best_alpha, {
            "n_selected": len(selected),
            "selected_blocks": selected,
            "best_inner_rmse": float(best_rmse),
        }
