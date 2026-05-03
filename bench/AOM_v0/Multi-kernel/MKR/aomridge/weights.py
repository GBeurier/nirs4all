"""Per-block weight strategies for mkR.

All strategies always return a **simplex** weight vector
(`eta_b >= 0, sum_b eta_b = 1`).

Strategies
----------

- **uniform** — ``eta_b = 1/B``.
- **manual**  — caller supplies ``eta``; we clip negatives, reject all-zero,
  and project onto the simplex.
- **kta**     — closed-form simplex weights from kernel-target alignment;
  works on **centred + trace-normalised** block kernels (see
  ``kernelizer.py``). Identity is *not* force-kept.
- **softmax_cv** — gradient-based optimisation of
  ``eta = softmax(theta)`` jointly with a single ``alpha`` over an inner
  CV split. Multi-start with random Dirichlet init, regularisation toward
  uniform.

**softmax_cv leakage caveat (v1)**: this v1 implementation passes the
already-centred + trace-normalised block kernels into the inner CV
*sliced by index*. Strictly speaking, the kernels know all training rows
(including those that will be inner-validation in any given fold) through
the centring/normalisation moments. This is a controlled simplification:
the **outer** test set is held out and the kernelizer is fitted on the
outer training set only, so test-set leakage is impossible. Inner-CV
weight estimation may slightly overfit; we recommend reporting the
inner-CV vs outer-CV gap. A v2 mode that refits the kernelizer per inner
fold is reserved for Round 2 (slower; documented in the implementation
log).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.model_selection import KFold

__all__ = [
    "uniform_weights",
    "manual_weights",
    "kta_simplex_weights",
    "softmax_cv_weights",
    "WeightLearningResult",
]


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WeightLearningResult:
    """Result of a supervised weight-learning call."""

    eta: np.ndarray              # (B,) simplex weights
    alpha: float                 # selected ridge regulariser (NaN if N/A)
    inner_cv_rmse: float         # best inner CV RMSE (NaN if N/A)
    n_iterations: int            # optimiser iterations (0 if closed form)
    converged: bool              # optimiser converged
    method: str                  # "uniform" / "manual" / "kta" / "softmax_cv"
    diagnostics: dict            # method-specific extra info


# ----------------------------------------------------------------------
# Closed-form / trivial strategies
# ----------------------------------------------------------------------


def uniform_weights(B: int) -> np.ndarray:
    """Return ``ones(B) / B``."""
    if B < 1:
        raise ValueError("B must be >= 1")
    return np.full(B, 1.0 / B, dtype=float)


def manual_weights(values: Sequence[float], B: int) -> np.ndarray:
    """Project user-supplied values onto the simplex.

    Negative values are clipped to zero. The result is renormalised to sum to 1.
    Raises if all values are zero or negative.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape != (B,):
        raise ValueError(f"manual weights must have shape ({B},); got {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0.0:
        raise ValueError("manual weights are all <= 0; cannot normalise")
    return arr / s


def kta_simplex_weights(
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    *,
    top_k: int | None = None,
) -> np.ndarray:
    """Closed-form KTA-aligned simplex weights.

    For centred kernels and centred ``y``:

    ```text
    align_b = <K_b, yc yc^T>_F / (||K_b||_F * ||yc yc^T||_F)
    eta_b = max(align_b, 0) / sum_b max(align_b, 0)
    ```

    If ``top_k`` is supplied, only the top-k highest-aligned blocks receive
    positive weight. If no block has positive alignment we fall back to a
    uniform weight on the top-k by ``|alignment|`` (or all blocks if
    ``top_k`` is None / >= B).
    """
    B = len(K_blocks)
    if B == 0:
        raise ValueError("K_blocks must be non-empty")
    yc = np.asarray(y, dtype=float).ravel()
    yc = yc - yc.mean()
    YYt = np.outer(yc, yc)
    Y_norm = float(np.linalg.norm(YYt, ord="fro"))
    aligns = np.zeros(B, dtype=float)
    if Y_norm < 1e-30:
        # y is constant — all alignments are 0; use uniform.
        return uniform_weights(B)
    for b, K_b in enumerate(K_blocks):
        K_norm = float(np.linalg.norm(K_b, ord="fro"))
        if K_norm < 1e-30:
            aligns[b] = 0.0
        else:
            aligns[b] = float(np.sum(K_b * YYt) / (K_norm * Y_norm))

    pos = np.maximum(aligns, 0.0)
    if top_k is not None and top_k >= 1 and top_k < B:
        cutoff_idx = np.argsort(-aligns)[:top_k]
        mask = np.zeros(B, dtype=bool)
        mask[cutoff_idx] = True
        pos = np.where(mask, pos, 0.0)
    total = float(pos.sum())
    if total > 0.0:
        return pos / total
    # Fallback: uniform on top_k by |aligns|
    eta = np.zeros(B, dtype=float)
    if top_k is None or top_k >= B:
        eta[:] = 1.0 / B
    else:
        cutoff_idx = np.argsort(-np.abs(aligns))[:top_k]
        eta[cutoff_idx] = 1.0 / top_k
    return eta


# ----------------------------------------------------------------------
# Softmax-CV (gradient on inner CV RMSE)
# ----------------------------------------------------------------------


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = theta - np.max(theta)
    e = np.exp(z)
    return e / e.sum()


def _kl_to_uniform(eta: np.ndarray) -> float:
    """KL divergence from softmax(theta) to uniform."""
    B = eta.size
    p = np.clip(eta, 1e-30, None)
    return float(np.sum(p * (np.log(p) + np.log(B))))


def _solve_alpha_for_kernel(
    K_train: np.ndarray, y_train: np.ndarray, alpha: float
) -> np.ndarray:
    """Solve ``(K + alpha I) c = y`` via Cholesky."""
    n = K_train.shape[0]
    Ka = K_train + alpha * np.eye(n)
    Ka = 0.5 * (Ka + Ka.T)
    cf = cho_factor(Ka, lower=True, check_finite=False)
    return cho_solve(cf, y_train, check_finite=False)


def _build_K_eta_train(
    K_blocks: Sequence[np.ndarray], eta: np.ndarray
) -> np.ndarray:
    """Build ``sum_b eta_b K_b`` for training kernels."""
    n = K_blocks[0].shape[0]
    K = np.zeros((n, n), dtype=float)
    for K_b, w in zip(K_blocks, eta, strict=False):
        if w == 0.0:
            continue
        K += w * K_b
    return 0.5 * (K + K.T)


def _build_K_eta_cross(
    K_blocks_cross: Sequence[np.ndarray], eta: np.ndarray
) -> np.ndarray:
    """Build ``sum_b eta_b K_b_cross``."""
    K = np.zeros_like(K_blocks_cross[0])
    for K_b, w in zip(K_blocks_cross, eta, strict=False):
        if w == 0.0:
            continue
        K += w * K_b
    return K


def _inner_cv_rmse(
    K_blocks_train: Sequence[np.ndarray],
    y_train: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    cv: KFold,
) -> float:
    """Mean RMSE across ``cv`` folds with a fixed ``(eta, alpha)``."""
    n = y_train.shape[0]
    rmse_folds = []
    indices = np.arange(n)
    for tr_idx, va_idx in cv.split(indices, y_train):
        # Slice the precomputed training kernels into fold-train and fold-val.
        # IMPORTANT: each K_b was already built on the full training set; for
        # the fold inner CV we further restrict to tr_idx for the train sub-
        # kernel and use rows va_idx, cols tr_idx for the cross. We do NOT
        # recompute centring here (this is inner CV; outer CV protects against
        # global centring leakage). The kernels were centred on the **outer**
        # training set, so this inner CV approximates fold-local CV with a
        # frozen kernelizer. This is acceptable for weight learning when the
        # outer test set is held out and the kernelizer is fitted only on
        # the outer training set.
        K_tr_blocks = [K_b[np.ix_(tr_idx, tr_idx)] for K_b in K_blocks_train]
        K_va_blocks = [K_b[np.ix_(va_idx, tr_idx)] for K_b in K_blocks_train]
        K_tr = _build_K_eta_train(K_tr_blocks, eta)
        K_va = _build_K_eta_cross(K_va_blocks, eta)
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]
        try:
            c = _solve_alpha_for_kernel(K_tr, y_tr, alpha)
        except np.linalg.LinAlgError:
            return float("inf")
        y_pred = K_va @ c
        rmse_folds.append(float(np.sqrt(np.mean((y_va - y_pred) ** 2))))
    if not rmse_folds:
        return float("inf")
    return float(np.mean(rmse_folds))


def softmax_cv_weights(
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    *,
    alphas: Sequence[float] | np.ndarray,
    cv_n_splits: int = 3,
    cv_shuffle: bool = True,
    n_restarts: int = 3,
    lambda_eta: float = 1e-3,
    max_iter: int = 50,
    random_state: int | None = 0,
    top_k_post: int | None = None,
    top_k_post_retune_alpha: bool = True,
) -> WeightLearningResult:
    """Optimise ``eta = softmax(theta)`` and ``alpha`` jointly on inner CV RMSE.

    Parameters
    ----------
    K_blocks : list of (n, n) ndarrays
        Centred + trace-normalised training block kernels.
    y : (n,) ndarray
        Centred training targets (the caller must subtract ``y_mean``).
    alphas : sequence of positive floats
        Candidate ridge regularisers; the optimiser picks the best one.
    cv_n_splits : int
        Inner CV K-fold splits (default 3).
    cv_shuffle : bool
        Shuffle inner CV (default True).
    n_restarts : int
        Random Dirichlet initialisations of ``theta`` (default 3).
    lambda_eta : float
        Regularisation toward uniform softmax via KL divergence.
    max_iter : int
        L-BFGS-B max iterations per restart.
    random_state : int | None
        Seed for inner CV and restart initialisation.
    top_k_post : int | None
        Post-hoc sparsification: keep only the top-``k`` weights of the
        optimised eta and renormalise. The alpha is then re-screened on the
        same grid at the sparse eta, and the inner-CV RMSE is recomputed.
        ``None`` (default) leaves eta dense.
    top_k_post_retune_alpha : bool
        When ``top_k_post`` is set, re-search ``alpha`` on the grid at the
        sparsified eta. When ``False`` the dense optimum alpha is reused
        unchanged (used for the iter7 ablation that isolates structural
        pruning from alpha rotation).

    Returns
    -------
    WeightLearningResult
    """
    B = len(K_blocks)
    if B < 1:
        raise ValueError("K_blocks must be non-empty")
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.shape[0] != K_blocks[0].shape[0]:
        raise ValueError("y must have the same length as K_blocks[0]")
    alpha_grid = np.asarray(alphas, dtype=float)
    if alpha_grid.ndim != 1 or alpha_grid.size == 0:
        raise ValueError("alphas must be a non-empty 1D sequence")
    if np.any(alpha_grid <= 0.0):
        raise ValueError("all alphas must be > 0")

    rng = np.random.default_rng(random_state)
    cv = KFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_state)

    def loss_at_theta(theta_alpha: np.ndarray) -> float:
        theta = theta_alpha[:B]
        log_alpha = theta_alpha[B]
        eta = _softmax(theta)
        rmse = _inner_cv_rmse(K_blocks, y_arr, eta, float(np.exp(log_alpha)), cv)
        return rmse + lambda_eta * _kl_to_uniform(eta)

    # Grid screen on alpha at uniform eta to seed log_alpha.
    eta_uniform = uniform_weights(B)
    alpha_seeds = []
    for a in alpha_grid:
        rmse = _inner_cv_rmse(K_blocks, y_arr, eta_uniform, float(a), cv)
        alpha_seeds.append((rmse, a))
    alpha_seeds.sort(key=lambda x: x[0])
    best_alpha_uniform = float(alpha_seeds[0][1])
    log_alpha_low = float(np.log(alpha_grid.min()))
    log_alpha_high = float(np.log(alpha_grid.max()))

    best_loss = float("inf")
    best_eta = eta_uniform
    best_alpha = best_alpha_uniform
    best_iter = 0
    best_converged = False

    bounds = [(-50.0, 50.0)] * B + [(log_alpha_low, log_alpha_high)]

    for restart in range(n_restarts):
        if restart == 0:
            # First restart: uniform softmax, best alpha from grid screen.
            theta_init = np.zeros(B)
        else:
            # Random Dirichlet-style init.
            dirichlet = rng.dirichlet(np.ones(B))
            theta_init = np.log(np.clip(dirichlet, 1e-6, None))
            theta_init -= theta_init.mean()
        log_alpha_init = float(np.log(best_alpha_uniform))
        x0 = np.concatenate([theta_init, [log_alpha_init]])
        try:
            res = minimize(
                loss_at_theta,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-7, "gtol": 1e-5},
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            continue
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_eta = _softmax(res.x[:B])
            best_alpha = float(np.exp(res.x[B]))
            best_iter = int(res.nit)
            best_converged = bool(res.success)

    sparse_meta: dict = {}
    if top_k_post is not None and 1 <= top_k_post < B:
        sorted_idx = np.argsort(-best_eta)[:top_k_post]
        sparse_eta = np.zeros(B, dtype=float)
        sparse_eta[sorted_idx] = best_eta[sorted_idx]
        s = float(sparse_eta.sum())
        if s > 0.0:
            sparse_eta = sparse_eta / s
            if top_k_post_retune_alpha:
                best_sparse_alpha = best_alpha
                best_sparse_rmse = float("inf")
                for a in alpha_grid:
                    rmse_a = _inner_cv_rmse(
                        K_blocks, y_arr, sparse_eta, float(a), cv
                    )
                    if rmse_a < best_sparse_rmse:
                        best_sparse_rmse = float(rmse_a)
                        best_sparse_alpha = float(a)
                best_eta = sparse_eta
                best_alpha = best_sparse_alpha
                inner_rmse = best_sparse_rmse
            else:
                best_eta = sparse_eta
                inner_rmse = _inner_cv_rmse(
                    K_blocks, y_arr, best_eta, best_alpha, cv
                )
            sparse_meta = {
                "top_k_post": int(top_k_post),
                "sparse_active_count": int(np.sum(best_eta > 0)),
                "post_hoc_rmse": float(inner_rmse),
                "alpha_retuned": bool(top_k_post_retune_alpha),
            }
        else:
            inner_rmse = _inner_cv_rmse(K_blocks, y_arr, best_eta, best_alpha, cv)
    else:
        inner_rmse = _inner_cv_rmse(K_blocks, y_arr, best_eta, best_alpha, cv)

    return WeightLearningResult(
        eta=best_eta,
        alpha=best_alpha,
        inner_cv_rmse=float(inner_rmse),
        n_iterations=best_iter,
        converged=best_converged,
        method="softmax_cv",
        diagnostics={
            "alpha_grid_min": float(alpha_grid.min()),
            "alpha_grid_max": float(alpha_grid.max()),
            "lambda_eta": float(lambda_eta),
            "n_restarts": int(n_restarts),
            "best_loss": float(best_loss),
            **sparse_meta,
        },
    )
