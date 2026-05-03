"""AOMMultiKernelBLUP — E-BLUP wrapper around AOMMultiKernelMixedModel.

Adds per-block prediction decomposition:

```python
from blup import AOMMultiKernelBLUP

model = AOMMultiKernelBLUP(operator_bank="compact", method="reml")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
components = model.predict_components(X_test)
# {"fixed": (n_test,), "random": OrderedDict[block_name, (n_test,)],
#  "total": (n_test,)}
```

Decomposition identity (must hold within fp tolerance):

```text
predict_components(X)["total"] == predict(X)
```

The estimator delegates fitting to ``AOMMultiKernelMixedModel`` and adds
the decomposition layer. Variance components, fixed effects, dual_alpha,
kernel statistics — all live in the MKM instance.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from mkm.estimator import AOMMultiKernelMixedModel

__all__ = ["AOMMultiKernelBLUP"]


class AOMMultiKernelBLUP(RegressorMixin, BaseEstimator):
    """E-BLUP with AOM block kernels and per-block prediction decomposition.

    Parameters mirror :class:`AOMMultiKernelMixedModel`. Fit delegates to the
    wrapped MKM instance; predict and predict_components reuse the stored
    ``alpha_dual_`` (V^-1 (y - X_f beta_hat)) without refactorising V.
    """

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
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AOMMultiKernelBLUP":
        self._mkm_ = AOMMultiKernelMixedModel(
            operator_bank=self.operator_bank,
            method=self.method,
            n_random_restarts=self.n_random_restarts,
            bounds_log_var=self.bounds_log_var,
            max_iter=self.max_iter,
            tol_grad=self.tol_grad,
            jitter=self.jitter,
            kernel_center=self.kernel_center,
            kernel_normalize=self.kernel_normalize,
            kernel_eps=self.kernel_eps,
            kernel_top_k_active=self.kernel_top_k_active,
            kernel_screen_score_method=self.kernel_screen_score_method,
            zero_trace_policy=self.zero_trace_policy,
            feature_scaling=self.feature_scaling,
            branch_preproc=self.branch_preproc,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._mkm_.fit(X, y)
        # Mirror useful attributes on self (sklearn convention).
        self.sigma2_blocks_ = self._mkm_.sigma2_blocks_
        self.sigma2_residual_ = self._mkm_.sigma2_residual_
        self.relative_contributions_ = self._mkm_.relative_contributions_
        self.beta_fixed_ = self._mkm_.beta_fixed_
        self.alpha_dual_ = self._mkm_.alpha_dual_
        self.block_names_ = self._mkm_.block_names_
        self.B_ = self._mkm_.B_
        self.log_likelihood_ = self._mkm_.log_likelihood_
        self.converged_ = self._mkm_.converged_
        self.boundary_components_ = self._mkm_.boundary_components_
        self.kernel_alignment_max_ = self._mkm_.kernel_alignment_max_
        self.kernel_alignment_matrix_ = self._mkm_.kernel_alignment_matrix_
        self._train_X_ = np.asarray(X, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._mkm_.predict(X)

    def predict_components(self, X: np.ndarray) -> dict:
        """Return per-block prediction decomposition.

        Returns
        -------
        dict
            ``{"fixed": ndarray (n_test,),
               "random": OrderedDict[block_name, ndarray (n_test,)],
               "total": ndarray (n_test,)}``.
            ``total`` equals ``predict(X)`` to floating-point tolerance.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        # Apply the same branch preprocessor that the wrapped MKM used at
        # fit time, so the cross kernels are consistent with the training
        # kernel space. We forward via the MKM's stored branch.
        if self._mkm_._branch_ is not None:
            from aomridge.branches import apply_branch_transform
            X_arr = apply_branch_transform(self._mkm_._branch_, X_arr)
        n_test = X_arr.shape[0]
        K_blocks_cross = self._mkm_.kernelizer_.transform(X_arr)
        # Fixed-effect contribution (intercept-only in v1).
        X_f_test = np.ones((n_test, 1), dtype=float)
        if self._mkm_._X_f_used_.shape[1] != X_f_test.shape[1]:
            X_f_test = X_f_test @ np.eye(
                X_f_test.shape[1], self._mkm_._X_f_used_.shape[1]
            )
        fixed = X_f_test @ self.beta_fixed_
        # Accumulate total INDEPENDENTLY of the dict so that duplicate block
        # names do not collapse the sum. Disambiguate keys with ``__k``.
        random: OrderedDict[str, np.ndarray] = OrderedDict()
        seen: dict[str, int] = {}
        total = fixed.copy()
        for K_b, s2, name in zip(
            K_blocks_cross, self.sigma2_blocks_, self.block_names_, strict=False,
        ):
            contribution = float(s2) * (K_b @ self.alpha_dual_)
            total = total + contribution
            unique_name = name
            if name in seen:
                seen[name] += 1
                unique_name = f"{name}__{seen[name]}"
            else:
                seen[name] = 1
            random[unique_name] = contribution
        return {"fixed": fixed, "random": random, "total": total}

    def train_decompose(self) -> dict:
        """Per-block decomposition on training data."""
        return self.predict_components(self._train_X_)

    def contribution_table(self, X: np.ndarray):
        """Return a pandas DataFrame of per-individual contributions.

        Columns: ``sample_index``, ``component_type`` ("fixed"/"random"),
        ``block_name``, ``contribution``, ``contribution_norm``,
        ``contribution_relative``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("contribution_table requires pandas") from exc
        comps = self.predict_components(X)
        rows = []
        total = comps["total"]
        for i, v in enumerate(comps["fixed"]):
            rows.append({
                "sample_index": i,
                "component_type": "fixed",
                "block_name": "_intercept",
                "contribution": float(v),
                "contribution_norm": float(abs(v)),
                "contribution_relative": float(abs(v)) / (abs(float(total[i])) + 1e-30),
            })
        for name, vec in comps["random"].items():
            for i, v in enumerate(vec):
                rows.append({
                    "sample_index": i,
                    "component_type": "random",
                    "block_name": name,
                    "contribution": float(v),
                    "contribution_norm": float(abs(v)),
                    "contribution_relative": float(abs(v)) / (abs(float(total[i])) + 1e-30),
                })
        return pd.DataFrame(rows)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(r2_score(y, self.predict(X)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "_mkm_"):
            raise RuntimeError("AOMMultiKernelBLUP must be fitted first")
