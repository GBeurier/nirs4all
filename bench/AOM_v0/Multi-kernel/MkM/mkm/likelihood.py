"""REML / ML log-likelihood and analytic gradient for the MKM model.

Model:

```text
y ~ N(X_f beta, V),    V = sum_b sigma_b^2 K_b + sigma_e^2 I_n
```

Variances are parameterised on the log scale:

```text
theta = (theta_1, ..., theta_B, theta_e)
sigma_b^2 = exp(theta_b),  sigma_e^2 = exp(theta_e)
```

Per-evaluation recipe (single Cholesky of V reused for all derivatives):

```text
L              = cholesky(V)
Q              = V^-1 X_f       = cho_solve(L, X_f)
M              = X_f^T Q
L_M            = cholesky(M)    (Cholesky of X_f^T V^-1 X_f)
beta_hat       = cho_solve(L_M, X_f^T cho_solve(L, y))
resid          = y - X_f beta_hat
a              = V^-1 resid     = P y    (since resid is in the GLS null space)
S              = V^-1           = cho_solve(L, I_n)    (one (n, n) solve)
P              = S - Q M^-1 Q^T

logdet V       = 2 sum log diag(L)
logdet M       = 2 sum log diag(L_M)
ell_REML       = -0.5 (logdet V + logdet M + resid^T a + (n - p_f) log 2*pi)

dV / dtheta_b  = sigma_b^2 K_b
dV / dtheta_e  = sigma_e^2 I_n
g_j            = 0.5 (tr(P dV/dtheta_j) - a^T dV/dtheta_j a)
```

The negative-log-likelihood objective (for minimisation) is `-ell_REML`,
with gradient `+g`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = [
    "MKMSolveResult",
    "compute_neg_log_reml",
    "compute_neg_log_ml",
    "compute_neg_log_reml_grad",
    "fit_fixed_effects",
]


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MKMSolveResult:
    """Per-evaluation cache returned by ``compute_neg_log_reml``.

    Attributes
    ----------
    neg_log_lik : float
        Negative log-(REML or ML) likelihood (the optimisation target).
    beta_hat : ndarray (p_f,)
        GLS fixed-effect estimate.
    resid : ndarray (n,)
        ``y - X_f beta_hat``.
    alpha_dual : ndarray (n,)
        ``V^-1 resid`` (= ``P y`` for the REML residual).
    P : ndarray (n, n)
        Projection ``V^-1 - V^-1 X_f M^-1 X_f^T V^-1``.
    log_det_V : float
        Log-determinant of ``V`` (positive when V is well-conditioned).
    log_det_M : float
        Log-determinant of ``X_f^T V^-1 X_f``.
    p_f : int
        Rank of ``X_f``.
    """

    neg_log_lik: float
    beta_hat: np.ndarray
    resid: np.ndarray
    alpha_dual: np.ndarray
    P: np.ndarray
    log_det_V: float
    log_det_M: float
    p_f: int


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _validate_inputs(
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    X_f: np.ndarray,
    theta: np.ndarray,
) -> tuple[int, int, int]:
    if len(K_blocks) < 1:
        raise ValueError("K_blocks must be non-empty")
    n = K_blocks[0].shape[0]
    for K_b in K_blocks:
        if K_b.shape != (n, n):
            raise ValueError("all K_blocks must be (n, n)")
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y must be (n,) with n={n}")
    if X_f.ndim != 2 or X_f.shape[0] != n:
        raise ValueError(f"X_f must be (n, p_f) with n={n}")
    p_f = X_f.shape[1]
    if theta.ndim != 1 or theta.shape[0] != len(K_blocks) + 1:
        raise ValueError(
            f"theta must be (B+1,) with B={len(K_blocks)}; got {theta.shape}"
        )
    return n, p_f, len(K_blocks)


def _build_V(
    K_blocks: Sequence[np.ndarray],
    theta: np.ndarray,
    jitter: float,
) -> np.ndarray:
    """Build ``V = sum_b sigma_b^2 K_b + sigma_e^2 I + jitter I``."""
    n = K_blocks[0].shape[0]
    sigma2 = np.exp(theta[:-1])
    sigma2_e = float(np.exp(theta[-1]))
    V = np.zeros((n, n), dtype=float)
    for K_b, s2 in zip(K_blocks, sigma2, strict=False):
        V += float(s2) * K_b
    V += sigma2_e * np.eye(n)
    if jitter > 0.0:
        V += jitter * np.eye(n)
    return 0.5 * (V + V.T)


def _safe_cho_factor(
    V: np.ndarray, base_jitter: float
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Cholesky-factor V, with adaptive jitter on failure.

    Returns ``(L, lower, V_used)`` where ``V_used`` includes any added
    jitter.
    """
    jitter = 0.0
    for _ in range(6):
        if jitter > 0.0:
            V_try = V + jitter * np.eye(V.shape[0])
        else:
            V_try = V
        try:
            cf = cho_factor(V_try, lower=True, check_finite=False)
            return cf[0], True, V_try
        except np.linalg.LinAlgError:
            jitter = max(base_jitter, 10.0 * jitter)
    raise np.linalg.LinAlgError(
        f"Cholesky failed after adaptive jitter (final jitter={jitter:.3e})"
    )


# ----------------------------------------------------------------------
# Fixed-effect rank handling
# ----------------------------------------------------------------------


def fit_fixed_effects(X_f: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, int]:
    """Drop near-collinear columns from ``X_f`` and return ``(X_f_used, p_f)``.

    Uses an SVD-based rank check (relative threshold ``tol``).
    """
    if X_f.size == 0:
        return X_f, 0
    U, sv, Vt = np.linalg.svd(X_f, full_matrices=False)
    sv_max = sv.max() if sv.size > 0 else 0.0
    rank = int(np.sum(sv > tol * sv_max))
    if rank == X_f.shape[1]:
        return X_f, rank
    # Project to leading SVD directions (orthonormal columns).
    return U[:, :rank], rank


def _rank_of(X_f: np.ndarray, tol: float = 1e-9) -> int:
    """Numerical rank using an SVD threshold (used as a defensive guard)."""
    if X_f.size == 0:
        return 0
    sv = np.linalg.svd(X_f, full_matrices=False, compute_uv=False)
    sv_max = sv.max() if sv.size > 0 else 0.0
    return int(np.sum(sv > tol * sv_max))


# ----------------------------------------------------------------------
# Negative log-likelihood (REML / ML)
# ----------------------------------------------------------------------


def compute_neg_log_reml(
    theta: np.ndarray,
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    X_f: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> MKMSolveResult:
    """Negative REML log-likelihood at ``theta`` plus per-eval cache.

    See module docstring for the algorithm. Returns the value as
    ``result.neg_log_lik`` and the cache for gradient computation.
    """
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X_f = np.asarray(X_f, dtype=float)
    n, p_f, B = _validate_inputs(K_blocks, y, X_f, theta)
    # Defensive rank check: a rank-deficient X_f silently produces a wrong
    # logdet(M) and wrong (n - p_f) degrees of freedom. The estimator's fit
    # path runs ``fit_fixed_effects`` first; this guard catches direct
    # callers (e.g. tests, benchmarks) that pass raw X_f.
    p_f_actual = _rank_of(X_f)
    if p_f_actual < p_f:
        raise ValueError(
            f"X_f is rank-deficient: rank={p_f_actual}, columns={p_f}. "
            "Run fit_fixed_effects(X_f) first to project onto the leading "
            "SVD directions, or pass an already-rank-corrected X_f."
        )
    V = _build_V(K_blocks, theta, jitter=jitter)
    L, lower, _ = _safe_cho_factor(V, base_jitter=jitter)
    cf = (L, lower)
    # Solves
    Q = cho_solve(cf, X_f, check_finite=False)              # (n, p_f)
    M = X_f.T @ Q                                            # (p_f, p_f)
    M = 0.5 * (M + M.T)
    cf_M = cho_factor(M, lower=True, check_finite=False)
    L_M = cf_M[0]
    Vinv_y = cho_solve(cf, y, check_finite=False)
    beta_hat = cho_solve(cf_M, X_f.T @ Vinv_y, check_finite=False)
    resid = y - X_f @ beta_hat
    alpha_dual = cho_solve(cf, resid, check_finite=False)
    # P = V^-1 - V^-1 X_f M^-1 X_f^T V^-1 = S - Q M^-1 Q^T
    S = cho_solve(cf, np.eye(n), check_finite=False)         # (n, n) — one solve
    M_inv = cho_solve(cf_M, np.eye(p_f), check_finite=False)
    P = S - Q @ M_inv @ Q.T
    P = 0.5 * (P + P.T)
    log_det_V = 2.0 * float(np.sum(np.log(np.abs(np.diag(L)))))
    log_det_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(L_M)))))
    quad = float(resid @ alpha_dual)
    neg_log_reml = 0.5 * (
        log_det_V + log_det_M + quad + (n - p_f) * math.log(2.0 * math.pi)
    )
    return MKMSolveResult(
        neg_log_lik=neg_log_reml,
        beta_hat=beta_hat,
        resid=resid,
        alpha_dual=alpha_dual,
        P=P,
        log_det_V=log_det_V,
        log_det_M=log_det_M,
        p_f=p_f,
    )


def compute_neg_log_ml(
    theta: np.ndarray,
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    X_f: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> MKMSolveResult:
    """Negative ML log-likelihood (no `logdet M` correction)."""
    res = compute_neg_log_reml(theta, K_blocks, y, X_f, jitter=jitter)
    n = K_blocks[0].shape[0]
    p_f = res.p_f
    # Rebuild ML objective from the REML cache.
    quad = float(res.resid @ res.alpha_dual)
    neg_log_ml = 0.5 * (
        res.log_det_V + quad + n * math.log(2.0 * math.pi)
    )
    # Replace neg_log_lik field; keep the rest of the cache.
    return MKMSolveResult(
        neg_log_lik=neg_log_ml,
        beta_hat=res.beta_hat,
        resid=res.resid,
        alpha_dual=res.alpha_dual,
        P=res.P,
        log_det_V=res.log_det_V,
        log_det_M=res.log_det_M,
        p_f=p_f,
    )


# ----------------------------------------------------------------------
# Gradient of negative log-likelihood (REML or ML, identical form using P)
# ----------------------------------------------------------------------


def compute_neg_log_reml_grad(
    theta: np.ndarray,
    K_blocks: Sequence[np.ndarray],
    res: MKMSolveResult,
) -> np.ndarray:
    """Analytic gradient of the **negative** log-REML at ``theta``.

    Derivation (from module docstring):

    ```text
    g_j = 0.5 (tr(P dotV_j) - a^T dotV_j a)
    ```

    For a block ``b``: ``dotV_b = sigma_b^2 K_b``. For the residual
    ``theta_e``: ``dotV_e = sigma_e^2 I``.

    The gradient of the **negative** log-REML is ``-`` of the gradient of
    the (positive) log-REML; using the form above (which is the gradient
    of the positive log-REML's negative, see derivation), we return ``g``
    directly.
    """
    theta = np.asarray(theta, dtype=float)
    sigma2 = np.exp(theta[:-1])
    sigma2_e = float(np.exp(theta[-1]))
    a = res.alpha_dual                                       # (n,)
    P = res.P
    grad = np.zeros_like(theta)
    for b, K_b in enumerate(K_blocks):
        dotV_b = float(sigma2[b]) * K_b                       # (n, n)
        tr_P_dotV = float(np.sum(P * dotV_b))
        a_dotV_a = float(a @ dotV_b @ a)
        grad[b] = 0.5 * (tr_P_dotV - a_dotV_a)
    # Residual block: dotV_e = sigma_e^2 I -> tr(P) * sigma_e^2 - sigma_e^2 a^T a
    tr_P = float(np.trace(P))
    a2 = float(a @ a)
    grad[-1] = 0.5 * sigma2_e * (tr_P - a2)
    return grad
