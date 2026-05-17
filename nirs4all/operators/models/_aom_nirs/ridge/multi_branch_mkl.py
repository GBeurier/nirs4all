"""Soft Multi-Branch Kernel for AOM-Ridge (Codex Phase H4).

Extends MKL-light over operators to MKL across *branches*. A branch is a
preprocessing transformer applied upstream of the operator bank (e.g. SNV,
MSC, ASLS baseline correction, EMSC2). For each branch ``b`` and operator
``a`` in the compact bank, build the kernel

```text
K_{b,a} = (Z_b A_a^T) (Z_b A_a^T)^T,   Z_b = T_b(X)
```

The combined kernel is

```text
K_total = sum_b w_b * sum_a (1/B_a) * K_{b,a}
```

with non-negative branch weights ``w_b`` summing to one. Weights are learned
fold-locally by kernel-target alignment (KTA) on the training fold, then
shrunk towards the identity-only weight ``w_none = 1`` to protect small-n
datasets from overfitting high-gain branches such as EMSC2.

Public functions
----------------

- ``kta_branch_score(K_b, YYt)``: alignment between a branch kernel and the
  outer-product target.
- ``learn_branch_weights(branch_kernels, YYt, shrinkage_to_identity)``:
  return the simplex weights blended towards identity.
- ``multi_branch_kernel(X_train, branches, operator_bank, branch_weights)``:
  compose the combined kernel from per-branch operator kernels.

Branch transformers must be cloned per fold; the helpers in this module
never share fitted state across folds.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from nirs4all.operators.models._aom_nirs.pls.operators import LinearSpectralOperator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from .kernels import (
    as_2d_y,
    clone_operator_bank,
    fit_operator_bank,
    resolve_operator_bank,
)
from .selection import resolve_cv
from .solvers import make_alpha_grid, solve_dual_ridge, solve_dual_ridge_path_eigh

BranchSpec = str
BranchTransformer = Any   # any object with fit(X[, y]) / transform(X)


# ----------------------------------------------------------------------
# Branch transformer factory
# ----------------------------------------------------------------------


def _make_branch_transformer(name: str) -> BranchTransformer | None:
    """Instantiate a branch transformer by name.

    The ``"none"`` branch is represented by ``None`` and means "use ``X`` as
    is" (the identity-like reference point for the simplex shrinkage). All
    other transformers follow the sklearn ``fit(X[, y]) / transform(X)``
    contract.
    """
    name = name.lower()
    if name == "none":
        return None
    from nirs4all.operators.models._aom_nirs.pls.preprocessing import (
        ASLSBaseline,
        ExtendedMSC,
        MultiplicativeScatterCorrection,
        StandardNormalVariate,
    )

    if name == "snv":
        return StandardNormalVariate()
    if name == "msc":
        return MultiplicativeScatterCorrection()
    if name == "asls":
        return ASLSBaseline()
    if name in ("emsc2", "emsc"):
        return ExtendedMSC(degree=2)
    raise ValueError(f"unknown branch: {name!r}")


def _fit_branch_transformer(
    name: str, X: np.ndarray, y: np.ndarray | None = None
) -> BranchTransformer | None:
    """Fit a fresh branch transformer on ``X`` (and optionally ``y``)."""
    transformer = _make_branch_transformer(name)
    if transformer is None:
        return None
    try:
        transformer.fit(X, y)
    except TypeError:
        transformer.fit(X)
    return transformer


def _apply_branch(transformer: BranchTransformer | None, X: np.ndarray) -> np.ndarray:
    """Apply a (possibly None) branch transformer."""
    if transformer is None:
        return np.asarray(X, dtype=float)
    return np.asarray(transformer.transform(X), dtype=float)


# ----------------------------------------------------------------------
# Kernel construction
# ----------------------------------------------------------------------


def _branch_kernel(
    Z: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Build ``sum_a (1/B) K_{b,a}`` for one branch.

    Each ``K_{b,a} = (Z A_a^T)(Z A_a^T)^T = Z A_a^T A_a Z^T`` is computed via
    the operator's ``apply_cov`` / ``adjoint_vec`` primitives so the wide
    feature view is never materialized.

    When ``normalize=True``, the kernel is rescaled by ``n / trace(K_b)`` so
    every branch contributes a kernel with unit average diagonal regardless
    of the magnitude of ``Z``. This is necessary because branches like EMSC2
    can produce wildly different-scale outputs and would otherwise dominate
    the simplex mixture and the auto alpha grid. Returns ``(K_b_norm,
    norm_factor)`` so the caller can replay the scaling on cross kernels.
    """
    n = Z.shape[0]
    K_b = np.zeros((n, n), dtype=float)
    inv_B = 1.0 / float(len(operators))
    Zt = Z.T
    for op in operators:
        AZt = op.apply_cov(Zt)               # (p, n)
        AtAZt = op.adjoint_vec(AZt)          # (p, n)
        K_b += inv_B * (Z @ AtAZt)
    K_b = 0.5 * (K_b + K_b.T)
    if not normalize:
        return K_b, 1.0
    trace = float(np.trace(K_b))
    if trace <= 1e-30:
        return K_b, 1.0
    factor = float(n) / trace
    return K_b * factor, factor


def multi_branch_kernel(
    X_train: np.ndarray,
    branches: Sequence[str],
    operator_bank: str | Sequence[LinearSpectralOperator],
    branch_weights: dict[str, float],
    fitted_transformers: dict[str, BranchTransformer | None] | None = None,
    branch_norm_factors: dict[str, float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Return the combined kernel ``K_total`` of shape ``(n, n)``.

    Parameters
    ----------
    X_train
        Training spectra (raw, ``(n, p)``).
    branches
        Sequence of branch names (e.g. ``("none", "snv", "msc", "asls", "emsc2")``).
    operator_bank
        Bank preset name (resolved by ``aompls.banks.bank_by_name``) or an
        explicit sequence of operators.
    branch_weights
        Mapping ``branch_name -> w_b`` (need not sum to one for this
        function; the caller is expected to pass a simplex point but no
        normalisation is enforced here).
    fitted_transformers
        Optional pre-fit branch transformers. When ``None`` the transformers
        are fit on ``X_train`` itself; pass fold-local fitted transformers
        from ``learn_branch_weights`` to guarantee anti-leakage.
    branch_norm_factors
        Optional pre-computed trace-normalisation factors per branch. Used
        when reusing factors learned on the training fold for cross
        kernels. If ``None`` and ``normalize=True``, factors are recomputed
        on ``X_train``.
    normalize
        When True, each branch kernel is rescaled to unit average diagonal
        before mixing, so the simplex weights are scale-invariant across
        branches.
    """
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D")
    n, p = X_train.shape
    ops_template = resolve_operator_bank(operator_bank, p=p)
    K_total = np.zeros((n, n), dtype=float)
    for b in branches:
        w_b = float(branch_weights.get(b, 0.0))
        if w_b == 0.0:
            continue
        if fitted_transformers is not None and b in fitted_transformers:
            transformer = fitted_transformers[b]
        else:
            transformer = _fit_branch_transformer(b, X_train)
        Z = _apply_branch(transformer, X_train)
        # Operators are bound to `p`; the branch transformers we use here
        # never change the wavelength count (per-sample normalisations and
        # baseline removals all preserve the second dimension), so the same
        # `p` clone is correct for every branch.
        if Z.shape[1] != p:
            raise ValueError(
                f"branch {b!r} changed feature dim from {p} to {Z.shape[1]};"
                " multi-branch MKL requires per-sample branches that preserve p"
            )
        ops_b = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops_b, Z)
        if branch_norm_factors is not None and b in branch_norm_factors:
            K_b, _ = _branch_kernel(Z, ops_b, normalize=False)
            K_b = K_b * branch_norm_factors[b]
        else:
            K_b, _ = _branch_kernel(Z, ops_b, normalize=normalize)
        K_total += w_b * K_b
    return 0.5 * (K_total + K_total.T)


# ----------------------------------------------------------------------
# Branch-weight learning (KTA + simplex shrinkage)
# ----------------------------------------------------------------------


def kta_branch_score(K_b: np.ndarray, YYt: np.ndarray) -> float:
    """Kernel-target alignment ``<K_b, YYt>_F / (||K_b||_F * ||YYt||_F)``.

    Returns 0.0 when either kernel has near-zero Frobenius norm.
    """
    if K_b.ndim != 2 or K_b.shape[0] != K_b.shape[1]:
        raise ValueError("K_b must be a square 2D matrix")
    if YYt.shape != K_b.shape:
        raise ValueError("YYt must have the same shape as K_b")
    K_norm = float(np.linalg.norm(K_b, "fro"))
    Y_norm = float(np.linalg.norm(YYt, "fro"))
    if K_norm < 1e-30 or Y_norm < 1e-30:
        return 0.0
    return float(np.sum(K_b * YYt) / (K_norm * Y_norm))


def learn_branch_weights(
    branch_kernels: dict[str, np.ndarray],
    YYt: np.ndarray,
    shrinkage_to_identity: float = 0.3,
) -> dict[str, float]:
    """Learn non-negative simplex weights from per-branch kernels via KTA.

    The raw weight for branch ``b`` is ``max(KTA(K_b, YYt), 0)``, normalised
    to the simplex over branches with positive raw alignment. The returned
    weight is then blended towards the identity branch:

    ```text
    w_b = alpha * raw_b + (1 - alpha) * (1 if b == 'none' else 0)
    ```

    where ``alpha = shrinkage_to_identity``. The semantics match the
    contract: ``shrinkage_to_identity = 0`` means "always use the identity
    branch only" (full shrinkage to none), ``shrinkage_to_identity = 1``
    means "use the data-driven raw KTA weights", with intermediate values
    interpolating linearly. The returned weights are non-negative and sum
    to one.
    """
    if not 0.0 <= shrinkage_to_identity <= 1.0:
        raise ValueError("shrinkage_to_identity must be in [0, 1]")
    if not branch_kernels:
        raise ValueError("branch_kernels must contain at least one branch")
    raw_scores: dict[str, float] = {}
    for name, K_b in branch_kernels.items():
        raw_scores[name] = max(kta_branch_score(K_b, YYt), 0.0)
    total = sum(raw_scores.values())
    if total > 0.0:
        raw_weights = {name: s / total for name, s in raw_scores.items()}
    else:
        # All KTAs were zero/negative — fall back to identity-only.
        raw_weights = {name: (1.0 if name == "none" else 0.0)
                       for name in branch_kernels}
    identity_weights = {name: (1.0 if name == "none" else 0.0)
                        for name in branch_kernels}
    if "none" not in branch_kernels:
        raise ValueError(
            "the 'none' branch must be present so shrinkage has an "
            "identity-only reference point"
        )
    blended = {
        name: shrinkage_to_identity * raw_weights[name]
        + (1.0 - shrinkage_to_identity) * identity_weights[name]
        for name in branch_kernels
    }
    return blended


# ----------------------------------------------------------------------
# Fold-local helper: fit branches on training rows, build branch kernels
# ----------------------------------------------------------------------


def fit_branches_and_kernels(
    X_tr: np.ndarray,
    branches: Sequence[str],
    operator_bank: str | Sequence[LinearSpectralOperator],
    y_tr: np.ndarray | None = None,
    normalize: bool = True,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, BranchTransformer | None],
    dict[str, float],
]:
    """Fit branch transformers on the training fold and build per-branch kernels.

    Returns ``(branch_kernels, fitted_transformers, branch_norm_factors)``.
    The kernels are ``sum_a (1/B) K_{b,a}`` for each branch on the training
    rows (trace-normalised when ``normalize=True``); the transformers are
    returned so the caller can replay them on the validation / test rows;
    the normalisation factors are returned so cross kernels are rescaled
    consistently.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    if X_tr.ndim != 2:
        raise ValueError("X_tr must be 2D")
    p = X_tr.shape[1]
    ops_template = resolve_operator_bank(operator_bank, p=p)
    fitted: dict[str, BranchTransformer | None] = {}
    K_per_branch: dict[str, np.ndarray] = {}
    norm_factors: dict[str, float] = {}
    for b in branches:
        transformer = _fit_branch_transformer(b, X_tr, y_tr)
        Z = _apply_branch(transformer, X_tr)
        if Z.shape[1] != p:
            raise ValueError(
                f"branch {b!r} changed feature dim from {p} to {Z.shape[1]}"
            )
        ops_b = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops_b, Z)
        K_b, factor = _branch_kernel(Z, ops_b, normalize=normalize)
        K_per_branch[b] = K_b
        fitted[b] = transformer
        norm_factors[b] = factor
    return K_per_branch, fitted, norm_factors


def cross_branch_kernel(
    X_left: np.ndarray,
    X_train: np.ndarray,
    branches: Sequence[str],
    operator_bank: str | Sequence[LinearSpectralOperator],
    branch_weights: dict[str, float],
    fitted_transformers: dict[str, BranchTransformer | None],
    branch_norm_factors: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the cross kernel ``K_cross[i, j] = phi(x_left_i)^T phi(x_train_j)``.

    Mirrors :func:`multi_branch_kernel` but produces a non-square matrix of
    shape ``(n_left, n_train)``. Each branch's transformer is replayed on
    the left rows; operators are then applied to both left and train
    branches and the cross kernel is accumulated with the same weights.

    ``branch_norm_factors`` (when provided) rescale each branch's cross
    kernel by the same factor used on its train kernel, so the dual
    regression coefficients map back to consistent feature views at
    predict time.
    """
    X_left = np.asarray(X_left, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    if X_left.ndim != 2 or X_train.ndim != 2:
        raise ValueError("inputs must be 2D")
    if X_left.shape[1] != X_train.shape[1]:
        raise ValueError("X_left and X_train must share the same feature dim")
    p = X_train.shape[1]
    ops_template = resolve_operator_bank(operator_bank, p=p)
    K_cross = np.zeros((X_left.shape[0], X_train.shape[0]), dtype=float)
    inv_B = 1.0 / float(len(ops_template))
    for b in branches:
        w_b = float(branch_weights.get(b, 0.0))
        if w_b == 0.0:
            continue
        transformer = fitted_transformers.get(b)
        if b != "none" and transformer is None:
            raise ValueError(
                f"branch {b!r} is missing a fitted transformer; refit before scoring"
            )
        Z_tr = _apply_branch(transformer, X_train)
        Z_left = _apply_branch(transformer, X_left)
        ops_b = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops_b, Z_tr)
        Z_tr_t = Z_tr.T
        K_cross_b = np.zeros((X_left.shape[0], X_train.shape[0]), dtype=float)
        for op in ops_b:
            AZ_tr_t = op.apply_cov(Z_tr_t)         # (p, n_tr)
            AtAZ_tr_t = op.adjoint_vec(AZ_tr_t)    # (p, n_tr)
            K_cross_b += inv_B * (Z_left @ AtAZ_tr_t)
        if branch_norm_factors is not None:
            K_cross_b = K_cross_b * float(branch_norm_factors.get(b, 1.0))
        K_cross += w_b * K_cross_b
    return K_cross


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------


class AOMMultiBranchMKL(BaseEstimator, RegressorMixin):
    """Soft Multi-Branch Kernel Ridge regressor (AOM-Ridge Phase H4).

    For each branch ``b`` in ``branches`` and each operator ``a`` in
    ``operator_bank``, the estimator forms the kernel

    ```text
    K_{b,a} = (Z_b A_a^T) (Z_b A_a^T)^T,   Z_b = T_b(X)
    ```

    and combines them as

    ```text
    K_total = sum_b w_b * sum_a (1/B) * K_{b,a}
    ```

    The branch weights ``w_b`` are learned fold-locally by KTA on the
    training fold (max-then-normalised) and shrunk towards
    identity-only weight ``w_none`` controlled by
    ``shrinkage_to_identity``. The dual Ridge is solved on
    ``K_total + alpha I`` with the alpha selected by cross-validation on
    the training fold.

    Parameters
    ----------
    branches
        Sequence of branch names (must include ``"none"`` for the
        identity reference point).
    operator_bank
        Bank preset name or sequence of operators.
    block_scaling
        Reserved for compatibility with the rest of AOM-Ridge; only
        ``"none"`` is implemented in the multi-branch path.
    shrinkage_to_identity
        Blend factor in ``[0, 1]``: ``w_b = alpha * raw + (1-alpha) *
        identity_b``. Larger values let the data drive the weights;
        smaller values protect small-n datasets by forcing reliance on
        the identity branch.
    alphas
        ``"auto"`` or an explicit grid.
    cv
        Integer ``KFold`` size or sklearn-compatible splitter.
    random_state
        Seed for the default ``KFold`` shuffle.

    Attributes
    ----------
    branch_weights_, branch_kta_scores_, dual_coef_, alpha_, alphas_,
    fitted_branch_transformers_, x_mean_, y_mean_, n_train_,
    diagnostics_.
    """

    def __init__(
        self,
        branches: Sequence[str] = ("none", "snv", "msc", "asls", "emsc2"),
        operator_bank: str | Sequence[LinearSpectralOperator] = "compact",
        block_scaling: str = "none",
        shrinkage_to_identity: float = 0.3,
        alphas: str | Sequence[float] = "auto",
        alpha_grid_size: int = 50,
        alpha_grid_low: float = -6.0,
        alpha_grid_high: float = 6.0,
        cv: int | object = 3,
        center: bool = True,
        random_state: int | None = 0,
    ) -> None:
        self.branches = branches
        self.operator_bank = operator_bank
        self.block_scaling = block_scaling
        self.shrinkage_to_identity = shrinkage_to_identity
        self.alphas = alphas
        self.alpha_grid_size = alpha_grid_size
        self.alpha_grid_low = alpha_grid_low
        self.alpha_grid_high = alpha_grid_high
        self.cv = cv
        self.center = center
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params_for_fit(self) -> None:
        if self.block_scaling != "none":
            raise NotImplementedError(
                "AOMMultiBranchMKL only supports block_scaling='none'"
            )
        if not 0.0 <= float(self.shrinkage_to_identity) <= 1.0:
            raise ValueError("shrinkage_to_identity must be in [0, 1]")
        names = [b.lower() for b in self.branches]
        if "none" not in names:
            raise ValueError(
                "branches must include 'none' (the identity reference branch)"
            )
        if len(set(names)) != len(names):
            raise ValueError("branches must be unique")

    def _resolve_alpha_grid(self, K_total: np.ndarray) -> np.ndarray:
        if isinstance(self.alphas, str):
            if self.alphas != "auto":
                raise ValueError("alphas string must be 'auto'")
            return make_alpha_grid(
                K_total,
                n_grid=self.alpha_grid_size,
                low=self.alpha_grid_low,
                high=self.alpha_grid_high,
            )
        arr = np.asarray(self.alphas, dtype=float)
        if arr.ndim != 1 or arr.size == 0 or np.any(arr <= 0.0):
            raise ValueError("alphas must be a non-empty 1D sequence of positive values")
        return arr

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> AOMMultiBranchMKL:
        self._validate_params_for_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        Y2, was_1d = as_2d_y(y)
        if Y2.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n = X.shape[0]
        self._was_1d_y = was_1d
        branches = tuple(b.lower() for b in self.branches)

        t0 = time.perf_counter()

        # Final-refit centering uses full-train mean (consistent with the
        # existing AOM-Ridge estimator). Fold-local CV in the alpha sweep
        # re-centers on training rows only.
        if self.center:
            x_mean = X.mean(axis=0)
            y_mean = Y2.mean(axis=0)
        else:
            x_mean = np.zeros(X.shape[1])
            y_mean = np.zeros(Y2.shape[1])
        Yc = Y2 - y_mean
        # NOTE: branch transformers see the *uncentered* X — SNV/MSC/EMSC2
        # are per-sample normalisations and must operate on the raw
        # spectra to be physically meaningful.

        # Step 1: fold-local CV — per fold learn branch weights, build
        # kernel, score the alpha grid. Pre-compute alpha_grid from the
        # *full-train* combined kernel so every fold scores the same
        # alphas.
        K_per_branch_full, fitted_full, norm_factors_full = (
            fit_branches_and_kernels(
                X, branches, self.operator_bank, y_tr=Y2,
            )
        )
        YYt_full = Yc @ Yc.T
        weights_full = learn_branch_weights(
            K_per_branch_full, YYt_full,
            shrinkage_to_identity=float(self.shrinkage_to_identity),
        )
        K_total_full = np.zeros((n, n), dtype=float)
        for b in branches:
            K_total_full += weights_full[b] * K_per_branch_full[b]
        K_total_full = 0.5 * (K_total_full + K_total_full.T)
        alpha_grid = self._resolve_alpha_grid(K_total_full)

        # Step 2: alpha CV — fold-local rebuild of branches/weights/kernels
        cv_obj = resolve_cv(self.cv, random_state=self.random_state)
        rmse_acc = np.zeros((alpha_grid.size,), dtype=float)
        n_folds = 0
        for train_idx, valid_idx in cv_obj.split(X, Y2):
            X_tr, X_va = X[train_idx], X[valid_idx]
            Y_tr, Y_va = Y2[train_idx], Y2[valid_idx]
            if self.center:
                y_mean_f = Y_tr.mean(axis=0)
            else:
                y_mean_f = np.zeros(Y_tr.shape[1])
            Yc_tr = Y_tr - y_mean_f
            K_per_branch_f, fitted_f, norm_factors_f = (
                fit_branches_and_kernels(
                    X_tr, branches, self.operator_bank, y_tr=Y_tr,
                )
            )
            YYt_f = Yc_tr @ Yc_tr.T
            weights_f = learn_branch_weights(
                K_per_branch_f, YYt_f,
                shrinkage_to_identity=float(self.shrinkage_to_identity),
            )
            K_tr_f = np.zeros((X_tr.shape[0], X_tr.shape[0]), dtype=float)
            for b in branches:
                K_tr_f += weights_f[b] * K_per_branch_f[b]
            K_tr_f = 0.5 * (K_tr_f + K_tr_f.T)
            K_va_f = cross_branch_kernel(
                X_va, X_tr, branches, self.operator_bank,
                weights_f, fitted_f, branch_norm_factors=norm_factors_f,
            )
            Cs = solve_dual_ridge_path_eigh(K_tr_f, Yc_tr, alpha_grid)
            for i in range(alpha_grid.size):
                Y_pred = K_va_f @ Cs[i] + y_mean_f
                diff = Y_va - Y_pred
                rmse_acc[i] += float(np.sqrt(np.mean(diff * diff)))
            n_folds += 1
        if n_folds == 0:
            raise ValueError("cv produced no folds")
        rmse_per_alpha = rmse_acc / n_folds
        if not np.all(np.isfinite(rmse_per_alpha)):
            raise FloatingPointError("non-finite RMSE during multi-branch CV")
        alpha_star = float(alpha_grid[int(np.argmin(rmse_per_alpha))])

        # Step 3: final refit on full-train with the full-train weights.
        C = solve_dual_ridge(K_total_full, Yc, alpha=alpha_star, method="cholesky")

        # Store fitted state
        self.branch_weights_ = dict(weights_full)
        self.branch_kta_scores_ = {
            b: kta_branch_score(K_per_branch_full[b], YYt_full)
            for b in branches
        }
        self.fitted_branch_transformers_ = fitted_full
        self.branch_norm_factors_ = norm_factors_full
        self.dual_coef_ = C.ravel() if was_1d else C
        self.alpha_ = alpha_star
        self.alphas_ = alpha_grid
        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        self.n_train_ = int(n)
        self._X_train = X.copy()
        self._cv_rmse_per_alpha = rmse_per_alpha
        self._fit_time_s = float(time.perf_counter() - t0)
        self._predict_time_s: float | None = None
        self.diagnostics_ = self._build_diagnostics()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dual_coef_"):
            raise NotFittedError(
                "AOMMultiBranchMKL must be fitted before calling predict()"
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self._X_train.shape[1]}"
            )
        t0 = time.perf_counter()
        branches = tuple(b.lower() for b in self.branches)
        K_cross = cross_branch_kernel(
            X, self._X_train, branches, self.operator_bank,
            self.branch_weights_, self.fitted_branch_transformers_,
            branch_norm_factors=self.branch_norm_factors_,
        )
        if self._was_1d_y:
            Y_pred = K_cross @ self.dual_coef_ + self.y_mean_
            self._predict_time_s = float(time.perf_counter() - t0)
            return Y_pred.ravel()
        Y_pred = K_cross @ self.dual_coef_ + self.y_mean_
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

    def _build_diagnostics(self) -> dict:
        alpha_idx = int(np.argmin(np.abs(self.alphas_ - self.alpha_)))
        n_alphas = self.alphas_.size
        boundary = bool(alpha_idx <= 1 or alpha_idx >= n_alphas - 2)
        return {
            "model": "AOMMultiBranchMKL",
            "selection": "multi_branch_mkl",
            "operator_bank": self.operator_bank
            if isinstance(self.operator_bank, str) else "custom",
            "branches": list(self.branches),
            "branch_weights": {b: float(w) for b, w in self.branch_weights_.items()},
            "branch_kta_scores": {
                b: float(s) for b, s in self.branch_kta_scores_.items()
            },
            "shrinkage_to_identity": float(self.shrinkage_to_identity),
            "alpha": float(self.alpha_),
            "alpha_index": alpha_idx,
            "alpha_at_boundary": boundary,
            "alphas": [float(a) for a in self.alphas_],
            "cv": self.cv if isinstance(self.cv, int) else type(self.cv).__name__,
            "cv_min_score": float(np.min(self._cv_rmse_per_alpha))
            if hasattr(self, "_cv_rmse_per_alpha") else None,
            "block_scaling": self.block_scaling,
            "n_train": int(self.n_train_),
            "fit_time_s": float(getattr(self, "_fit_time_s", 0.0)),
            "predict_time_s": (
                None if self._predict_time_s is None else float(self._predict_time_s)
            ),
        }
