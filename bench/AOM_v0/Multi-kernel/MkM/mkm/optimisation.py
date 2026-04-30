"""Multi-start L-BFGS-B optimisation of REML / ML for MKM.

Approach:

- Deterministic seeds: uniform variance, residual-only, each single-block
  active, plus random perturbations around ``log(var(y) / (B + 1))``.
- L-BFGS-B with bounds on log-variances.
- Best by lowest negative log-likelihood (NOT gradient norm).
- Endpoint variance reported to flag multimodality.
- Boundary detection: KKT-style (projected gradient near zero AND
  ``theta_b - lower_bound < tol`` AND relative contribution < eps).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .likelihood import (
    MKMSolveResult,
    compute_neg_log_ml,
    compute_neg_log_reml,
    compute_neg_log_reml_grad,
)

__all__ = [
    "MKMOptimisationResult",
    "fit_variance_components",
]


@dataclass(frozen=True)
class MKMOptimisationResult:
    """Result of multi-start REML/ML fitting."""

    theta: np.ndarray             # (B + 1,) best theta
    neg_log_lik: float            # at best theta
    converged: bool               # best run converged
    n_iterations: int             # iterations of best run
    boundary_components: list[int]  # indices of theta entries near boundary
    restart_endpoints: np.ndarray  # (n_restarts, B+1) endpoint of each restart
    restart_neg_log_liks: np.ndarray  # (n_restarts,)
    method: str                   # "reml" or "ml"
    diagnostics: dict


# ----------------------------------------------------------------------
# Deterministic + random starts
# ----------------------------------------------------------------------


def _deterministic_starts(
    var_y: float, B: int, n_random: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Generate (n_deterministic + n_random) starting points.

    Deterministic:
    - uniform variance: log(var_y / (B+1)) on every component.
    - residual-only: log(var_y) on theta_e, log(eps) on each theta_b.
    - per-block active: log(var_y / 2) on one block, log(eps) elsewhere
      (for each b in {0, ..., B-1}).
    """
    starts: list[np.ndarray] = []
    log_unif = float(np.log(var_y / max(B + 1, 1)))
    base = np.full(B + 1, log_unif, dtype=float)
    starts.append(base.copy())
    # residual-only
    res_only = np.full(B + 1, np.log(1e-6), dtype=float)
    res_only[-1] = float(np.log(var_y))
    starts.append(res_only)
    # per-block active
    for b in range(B):
        s = np.full(B + 1, np.log(1e-6), dtype=float)
        s[b] = float(np.log(var_y / 2.0))
        s[-1] = float(np.log(var_y / 2.0))
        starts.append(s)
    # random perturbations around uniform
    for _ in range(n_random):
        pert = rng.normal(0.0, 1.0, size=B + 1)
        starts.append(base + pert)
    return starts


# ----------------------------------------------------------------------
# Boundary detection
# ----------------------------------------------------------------------


def _detect_boundary(
    theta: np.ndarray,
    grad: np.ndarray,
    bounds: tuple[float, float],
    contrib: np.ndarray,
    *,
    bound_tol: float = 0.5,
    grad_tol: float = 1e-3,
    contrib_tol: float = 1e-3,
) -> list[int]:
    """KKT-style boundary detection.

    A component ``j`` is at the lower bound when
      (a) ``theta_j - lower_bound < bound_tol``,
      (b) projected gradient (i.e. gradient when not free to decrease)
          is near zero, and
      (c) its relative contribution is below ``contrib_tol``.
    """
    lb = float(bounds[0])
    out: list[int] = []
    for j in range(theta.size):
        at_lb = (theta[j] - lb) < bound_tol
        if not at_lb:
            continue
        # Projected gradient at lower bound: only outward direction matters.
        proj_g = max(grad[j], 0.0)
        if proj_g < grad_tol and contrib[j] < contrib_tol:
            out.append(j)
    return out


def _relative_contributions(theta: np.ndarray) -> np.ndarray:
    sigma2 = np.exp(theta)
    total = float(sigma2.sum()) + 1e-30
    return sigma2 / total


# ----------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------


def fit_variance_components(
    K_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    X_f: np.ndarray,
    *,
    method: str = "reml",
    n_random_restarts: int = 5,
    bounds: tuple[float, float] = (-15.0, 15.0),
    max_iter: int = 200,
    tol_grad: float = 1e-5,
    jitter: float = 1e-8,
    random_state: int | None = 0,
) -> MKMOptimisationResult:
    """Multi-start L-BFGS-B fit for ``theta`` minimising ``-ell_REML`` / ``-ell_ML``."""
    if method not in ("reml", "ml"):
        raise ValueError("method must be 'reml' or 'ml'")
    y_arr = np.asarray(y, dtype=float).ravel()
    X_f_arr = np.asarray(X_f, dtype=float)
    if X_f_arr.ndim != 2:
        raise ValueError("X_f must be 2D")
    B = len(K_blocks)
    if B < 1:
        raise ValueError("K_blocks must be non-empty")

    var_y = float(np.var(y_arr))
    if var_y <= 0.0:
        raise ValueError("var(y) must be > 0")

    rng = np.random.default_rng(random_state)
    starts = _deterministic_starts(var_y, B, n_random_restarts, rng)

    bounds_list = [bounds] * (B + 1)

    obj_fn: Callable[..., MKMSolveResult]
    obj_fn = compute_neg_log_reml if method == "reml" else compute_neg_log_ml

    def loss_and_grad(theta_arr: np.ndarray) -> tuple[float, np.ndarray]:
        # Clip to bounds for numerical safety even though L-BFGS-B enforces them.
        theta_arr = np.clip(theta_arr, bounds[0], bounds[1])
        try:
            res = obj_fn(theta_arr, K_blocks, y_arr, X_f_arr, jitter=jitter)
        except np.linalg.LinAlgError:
            return float("inf"), np.zeros_like(theta_arr)
        # Gradient is the same for REML and ML when using P (we use P for both
        # — for ML, `P` should be replaced by `V^-1`, but the cache stores P).
        # For ML, recompute V^-1 path. We only compute analytic gradient for
        # REML in v1; for ML, fall back to finite differences via the
        # optimiser's own approximation. Here we always pass the REML gradient
        # for the REML branch and let L-BFGS-B handle ML by FD.
        if method == "reml":
            grad = compute_neg_log_reml_grad(theta_arr, K_blocks, res)
            return float(res.neg_log_lik), grad
        return float(res.neg_log_lik), np.zeros_like(theta_arr)

    # Run all restarts; record endpoints and losses.
    endpoints: list[np.ndarray] = []
    losses: list[float] = []
    iter_counts: list[int] = []
    converged_flags: list[bool] = []
    final_results: list[MKMSolveResult | None] = []

    use_grad = method == "reml"

    for start in starts:
        try:
            res = minimize(
                loss_and_grad if use_grad else lambda t: float(loss_and_grad(t)[0]),
                start,
                jac=use_grad,
                method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": max_iter, "gtol": tol_grad, "ftol": 1e-9},
            )
            theta_end = np.array(res.x, dtype=float)
            endpoints.append(theta_end)
            losses.append(float(res.fun))
            iter_counts.append(int(res.nit))
            converged_flags.append(bool(res.success))
            try:
                solve_res = obj_fn(theta_end, K_blocks, y_arr, X_f_arr, jitter=jitter)
                final_results.append(solve_res)
            except np.linalg.LinAlgError:
                final_results.append(None)
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            # Record failure as inf loss so this restart loses.
            endpoints.append(np.array(start, dtype=float))
            losses.append(float("inf"))
            iter_counts.append(0)
            converged_flags.append(False)
            final_results.append(None)
            continue

    # Pick the best.
    losses_arr = np.asarray(losses)
    best_idx = int(np.argmin(losses_arr))
    best_theta = endpoints[best_idx]
    best_res = final_results[best_idx]
    if best_res is None:
        raise RuntimeError(
            "all restarts failed; check K_blocks symmetry, conditioning, jitter"
        )

    # Boundary diagnostics.
    contrib = _relative_contributions(best_theta)
    grad_at_best = compute_neg_log_reml_grad(best_theta, K_blocks, best_res)
    boundary_components = _detect_boundary(
        best_theta, grad_at_best, bounds, contrib
    )

    diag = {
        "endpoint_neg_log_lik_min": float(np.min(losses_arr)),
        "endpoint_neg_log_lik_max": float(np.max(losses_arr[np.isfinite(losses_arr)]))
        if np.any(np.isfinite(losses_arr)) else float("inf"),
        "endpoint_neg_log_lik_std": float(np.std(losses_arr[np.isfinite(losses_arr)]))
        if np.any(np.isfinite(losses_arr)) else 0.0,
        "n_restarts_total": int(len(starts)),
        "n_restarts_finite": int(np.sum(np.isfinite(losses_arr))),
        "method": method,
    }

    return MKMOptimisationResult(
        theta=best_theta,
        neg_log_lik=float(losses[best_idx]),
        converged=bool(converged_flags[best_idx]),
        n_iterations=int(iter_counts[best_idx]),
        boundary_components=boundary_components,
        restart_endpoints=np.asarray(endpoints, dtype=float),
        restart_neg_log_liks=losses_arr,
        method=method,
        diagnostics=diag,
    )
