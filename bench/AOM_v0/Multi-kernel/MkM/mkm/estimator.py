"""AOMMultiKernelMixedModel — REML / ML mixed model with AOM block kernels.

API:

```python
from mkm import AOMMultiKernelMixedModel

model = AOMMultiKernelMixedModel(
    operator_bank="compact",
    method="reml",
    n_random_restarts=5,
    random_state=0,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("variance components:", model.sigma2_blocks_)
print("residual variance: ", model.sigma2_residual_)
print("relative contrib: ", model.relative_contributions_)
```

Stored attributes after ``fit``:

- ``sigma2_blocks_``           per-block variance components, shape (B,)
- ``sigma2_residual_``         residual variance (scalar)
- ``relative_contributions_``  ``sigma_b^2 / total`` plus residual share
- ``beta_fixed_``              GLS fixed-effect estimate, shape (p_f,)
- ``alpha_dual_``              ``V^-1 (y - X_f beta_hat)``, shape (n_train,)
- ``log_likelihood_``          ``-neg_log_lik`` (REML or ML)
- ``converged_``               bool
- ``boundary_components_``     list[int] of theta indices at lower bound
- ``optimisation_diagnostics_`` dict
- ``kernel_alignment_max_``    float

Multi-output ``y`` is not supported in v1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from .kernelizer import AOMKernelizer, kernel_alignment_matrix
from .likelihood import compute_neg_log_reml, fit_fixed_effects
from .optimisation import fit_variance_components

__all__ = ["AOMMultiKernelMixedModel"]


def _intercept_design(n: int) -> np.ndarray:
    return np.ones((n, 1), dtype=float)


class AOMMultiKernelMixedModel(RegressorMixin, BaseEstimator):
    """REML / ML multi-kernel mixed model with AOM block kernels."""

    def __init__(
        self,
        operator_bank: object = "compact",
        *,
        method: str = "reml",
        n_random_restarts: int = 5,
        bounds_log_var: tuple[float, float] = (-15.0, 15.0),
        max_iter: int = 200,
        tol_grad: float = 1e-5,
        jitter: float = 1e-8,
        kernel_center: bool = True,
        kernel_normalize: str = "trace",
        kernel_eps: float = 1e-12,
        kernel_top_k_active: int | None = None,
        kernel_screen_score_method: str = "norm",
        zero_trace_policy: str = "raise",
        feature_scaling: str = "center",
        branch_preproc: str = "none",
        sigma2_top_k_post: int | None = None,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        self.operator_bank = operator_bank
        self.method = method
        self.n_random_restarts = n_random_restarts
        self.bounds_log_var = bounds_log_var
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.jitter = jitter
        self.kernel_center = kernel_center
        self.kernel_normalize = kernel_normalize
        self.kernel_eps = kernel_eps
        self.kernel_top_k_active = kernel_top_k_active
        self.kernel_screen_score_method = kernel_screen_score_method
        self.zero_trace_policy = zero_trace_policy
        self.feature_scaling = feature_scaling
        self.branch_preproc = branch_preproc
        self.sigma2_top_k_post = sigma2_top_k_post
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMultiKernelMixedModel":
        X_arr, y_arr = self._validate_inputs(X, y)
        n, p = X_arr.shape

        # Centre y inside the model — fixed-effect intercept absorbs it
        # without that step, but centring keeps numerical scales tidy.
        self._y_mean_ = float(y_arr.mean())

        # Branch preprocessor (fitted on training only).
        self._branch_ = None
        if self.branch_preproc and self.branch_preproc != "none":
            # Import lazily; the branch lib lives in the MKR package.
            try:
                from aomridge.branches import (
                    fit_transform_branch,
                    make_branch_preproc,
                )
            except ImportError as exc:
                raise ImportError(
                    "branch_preproc requires aomridge to be on PYTHONPATH "
                    "(see Multi-kernel/MKR)"
                ) from exc
            self._branch_ = make_branch_preproc(self.branch_preproc)
            X_arr = fit_transform_branch(self._branch_, X_arr, y_arr)
            n, p = X_arr.shape

        # We pass un-centred y through, with X_f = ones(n, 1); the GLS
        # estimate of beta will be the GLS-mean, equivalent to centring.
        # This avoids confusion about whether the intercept is in beta or
        # added back in predict.
        kernelizer = AOMKernelizer(
            operator_bank=self.operator_bank,
            center=self.kernel_center,
            normalize=self.kernel_normalize,
            eps=self.kernel_eps,
            zero_trace_policy=self.zero_trace_policy,
            top_k_active=self.kernel_top_k_active,
            screen_score_method=self.kernel_screen_score_method,
        )
        K_blocks = kernelizer.fit_transform(X_arr, y_arr)
        B = len(K_blocks)
        if B < 1:
            raise ValueError("operator bank produced no blocks")
        self.kernelizer_ = kernelizer
        self.block_names_ = list(kernelizer.block_names_ or [])
        self.B_ = B

        X_f_raw = _intercept_design(n)
        X_f, p_f = fit_fixed_effects(X_f_raw)
        self._X_f_used_ = X_f
        self.p_f_ = p_f

        opt_res = fit_variance_components(
            K_blocks,
            y_arr,
            X_f,
            method=self.method,
            n_random_restarts=self.n_random_restarts,
            bounds=self.bounds_log_var,
            max_iter=self.max_iter,
            tol_grad=self.tol_grad,
            jitter=self.jitter,
            random_state=self.random_state,
        )

        # Store best variance components.
        sigma2 = np.exp(opt_res.theta[:-1])
        sigma2_e = float(np.exp(opt_res.theta[-1]))
        total = float(sigma2.sum() + sigma2_e)
        if total <= 0.0:
            total = 1e-30

        self.sigma2_blocks_ = sigma2
        self.sigma2_residual_ = sigma2_e
        self.relative_contributions_ = {
            **{name: float(s2 / total)
               for name, s2 in zip(self.block_names_, sigma2, strict=False)},
            "_residual": float(sigma2_e / total),
        }
        self.theta_ = opt_res.theta
        self.log_likelihood_ = -float(opt_res.neg_log_lik)
        self.converged_ = bool(opt_res.converged)
        self.boundary_components_ = list(opt_res.boundary_components)
        self.optimisation_diagnostics_ = dict(opt_res.diagnostics)

        # Optional post-REML sparsification of variance components.
        # Keep only the top-k components by magnitude; zero out the rest;
        # re-solve V and alpha_dual at the sparsified theta.
        if self.sigma2_top_k_post is not None and 1 <= self.sigma2_top_k_post < B:
            keep_idx = np.argsort(-sigma2)[: self.sigma2_top_k_post]
            theta_sparse = opt_res.theta.copy()
            for j in range(B):
                if j not in keep_idx:
                    theta_sparse[j] = -50.0  # effectively zero variance
            self.theta_ = theta_sparse
            sigma2 = np.exp(theta_sparse[:-1])
            sigma2_e = float(np.exp(theta_sparse[-1]))
            self.sigma2_blocks_ = sigma2
            self.sigma2_residual_ = sigma2_e
            self.sparse_active_count_ = int(self.sigma2_top_k_post)

        # Recompute final solve for prediction state.
        final_res = compute_neg_log_reml(
            self.theta_, K_blocks, y_arr, X_f, jitter=self.jitter
        )
        self.beta_fixed_ = final_res.beta_hat
        self.alpha_dual_ = final_res.alpha_dual

        # Diagnostics.
        A = kernel_alignment_matrix(K_blocks)
        self.kernel_alignment_matrix_ = A
        if A.shape[0] >= 2:
            off = A - np.eye(A.shape[0])
            self.kernel_alignment_max_ = float(np.max(np.abs(off)))
        else:
            self.kernel_alignment_max_ = 0.0

        # Cache training kernel norm for prediction reuse.
        self._n_train_ = n
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        if self._branch_ is not None:
            from aomridge.branches import apply_branch_transform
            X_arr = apply_branch_transform(self._branch_, X_arr)
        n_test = X_arr.shape[0]
        K_blocks_cross = self.kernelizer_.transform(X_arr)
        # E-BLUP mean prediction:
        #   hat y_* = X_f_* beta_hat + sum_b sigma_b^2 K_b_cross alpha_dual
        X_f_test = _intercept_design(n_test)
        # If we used a rank-reduced X_f at fit time, project test-side too.
        # In v1, X_f is intercept-only so this is identity.
        if self._X_f_used_.shape[1] != X_f_test.shape[1]:
            # Use the same SVD orthonormalisation the fit step applied.
            # For intercept-only this branch is unreachable.
            X_f_test = X_f_test @ np.eye(X_f_test.shape[1], self._X_f_used_.shape[1])
        fixed_part = X_f_test @ self.beta_fixed_
        random_part = np.zeros(n_test, dtype=float)
        for K_b, s2 in zip(K_blocks_cross, self.sigma2_blocks_, strict=False):
            random_part += float(s2) * (K_b @ self.alpha_dual_)
        return fixed_part + random_part

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
        if y_raw.ndim == 2 and y_raw.shape[1] > 1:
            raise ValueError("MKM does not support multi-output y in v1")
        y_arr = y_raw.ravel()
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if self.feature_scaling != "center":
            raise NotImplementedError("Only feature_scaling='center' supported in v1")
        if self.method not in ("reml", "ml"):
            raise ValueError("method must be 'reml' or 'ml'")
        return X_arr, y_arr

    def _check_fitted(self) -> None:
        if not hasattr(self, "alpha_dual_"):
            raise RuntimeError("AOMMultiKernelMixedModel must be fitted first")
