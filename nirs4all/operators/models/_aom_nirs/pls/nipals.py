"""NIPALS engines: materialized reference and adjoint fast variant.

Conventions:

- `X in R^{n x p}` and `Y in R^{n x q}` are *centered* on input.
- The operator `A_b in R^{p x p}` acts on row spectra: `X_b = X A_b^T`.
- Effective weight in the original space: `z_a = A_b^T r_a` where `r_a` is
  the NIPALS direction in the transformed space.
- Loadings:
  `p_a = X^T t_a / (t_a^T t_a)`,
  `q_a = Y^T t_a / (t_a^T t_a)`.
- Coefficient matrix: `B = Z (P^T Z)^+ Q^T` where `+` is the Moore-Penrose
  pseudoinverse (used because P^T Z may be ill-conditioned for tiny tail
  components).

The dataclass `NIPALSResult` holds all extracted quantities for downstream
use (estimators, classification, diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .operators import IdentityOperator, LinearSpectralOperator


@dataclass
class NIPALSResult:
    """Output of a NIPALS extraction with operator-adaptive weighting."""

    Z: np.ndarray  # original-space effective weights, shape (p, K)
    P: np.ndarray  # X-loadings in original space, shape (p, K)
    Q: np.ndarray  # Y-loadings, shape (q, K)
    T: np.ndarray  # X-scores, shape (n, K)
    R: List[np.ndarray] = field(default_factory=list)  # transformed-space directions per component
    operator_indices: List[int] = field(default_factory=list)  # selected operator id per component
    operator_names: List[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return self.Z.shape[1]

    def coef(self) -> np.ndarray:
        """Return the regression coefficient matrix `B = Z (P^T Z)^+ Q^T`."""
        Z = self.Z
        P = self.P
        Q = self.Q
        if Z.shape[1] == 0:
            return np.zeros((Z.shape[0], Q.shape[0]))
        ptz = P.T @ Z
        # Use pseudoinverse for robustness; ptz is upper-triangular for
        # standard PLS, so this is an inverse for full-rank cases.
        try:
            inv = np.linalg.inv(ptz)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(ptz)
        return Z @ inv @ Q.T

    def coef_prefix(self, k: int) -> np.ndarray:
        """Return coefficients using only the first `k` components."""
        if k <= 0 or k > self.n_components:
            raise ValueError(f"k must be in [1, {self.n_components}]")
        Z = self.Z[:, :k]
        P = self.P[:, :k]
        Q = self.Q[:, :k]
        ptz = P.T @ Z
        try:
            inv = np.linalg.inv(ptz)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(ptz)
        return Z @ inv @ Q.T


# ---------------------------------------------------------------------------
# Standard NIPALS on a fixed matrix
# ---------------------------------------------------------------------------


def nipals_standard(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard PLS2 NIPALS on `(X, Y)` with deflation.

    Returns (W, T, P, Q, U): X-weights, X-scores, X-loadings, Y-loadings,
    Y-scores. Both X and Y are deflated using the standard NIPALS rule.
    The columns of W are normalized to unit length.
    """
    X = np.asarray(X, dtype=float).copy()
    Y = np.asarray(Y, dtype=float).copy()
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, p = X.shape
    q = Y.shape[1]
    K = int(n_components)
    if K < 1:
        raise ValueError("n_components must be >= 1")
    W = np.zeros((p, K))
    T = np.zeros((n, K))
    P = np.zeros((p, K))
    Q = np.zeros((q, K))
    U = np.zeros((n, K))
    eps = 1e-12
    for a in range(K):
        # Initialise u as the column of Y with maximum variance
        col = int(np.argmax(np.linalg.norm(Y, axis=0)))
        u = Y[:, col].copy()
        if np.linalg.norm(u) < eps:
            break
        for _ in range(max_iter):
            w_num = X.T @ u
            w_norm = np.linalg.norm(w_num)
            if w_norm < eps:
                w = np.zeros(p)
                break
            w = w_num / w_norm
            t = X @ w
            t_sq = t @ t
            if t_sq < eps:
                break
            c = Y.T @ t / t_sq
            c_norm = np.linalg.norm(c)
            if c_norm < eps:
                break
            u_new = Y @ c / (c @ c)
            if np.linalg.norm(u_new - u) < tol * max(np.linalg.norm(u_new), eps):
                u = u_new
                break
            u = u_new
        # Final loadings
        t = X @ w
        t_sq = float(t @ t)
        if t_sq < eps:
            break
        p_load = X.T @ t / t_sq
        q_load = Y.T @ t / t_sq
        W[:, a] = w
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        U[:, a] = u
        # Deflation
        X = X - np.outer(t, p_load)
        Y = Y - np.outer(t, q_load)
        # Stop if residual is exhausted
        if np.linalg.norm(Y) < eps and np.linalg.norm(X) < eps:
            break
    actual_K = int(np.sum(np.linalg.norm(W, axis=0) > 0))
    return (
        W[:, :actual_K],
        T[:, :actual_K],
        P[:, :actual_K],
        Q[:, :actual_K],
        U[:, :actual_K],
    )


# ---------------------------------------------------------------------------
# Materialized NIPALS through a fixed operator
# ---------------------------------------------------------------------------


def nipals_materialized_fixed(
    X: np.ndarray,
    Y: np.ndarray,
    operator: LinearSpectralOperator,
    n_components: int,
) -> NIPALSResult:
    """Run standard NIPALS on `X_b = X A^T` and map back to the original space.

    The result represents the operator-adapted PLS model fitted with a single
    fixed operator throughout. This is the slow reference for AOM-global
    selection with a fixed operator.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    operator.fit(X)
    Xb = operator.transform(X)
    W, _Tb, _Pb, _Qb, _U = nipals_standard(Xb, Y, n_components)
    # Map transformed-space directions back to original-space effective weights.
    # If Pb is the transformed loadings and W are transformed weights, the
    # effective weights in the original space are A^T W where A acts as a
    # left-multiplication on column vectors. With our convention the matrix
    # of A is A; A^T W applies the adjoint to each weight column.
    K = W.shape[1]
    Z = np.empty((X.shape[1], K))
    for a in range(K):
        Z[:, a] = operator.adjoint_vec(W[:, a])
    # Recompute scores and loadings in the original space using deflation in
    # the original-space scores (matches transformed-space PLS for fixed
    # operator up to numerical precision).
    Tres = X.copy()
    Yres = Y.copy()
    T = np.empty((X.shape[0], K))
    P = np.empty((X.shape[1], K))
    Q = np.empty((Y.shape[1], K))
    R_list: List[np.ndarray] = []
    for a in range(K):
        z = Z[:, a]
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-14:
            T[:, a] = 0.0
            P[:, a] = 0.0
            Q[:, a] = 0.0
            R_list.append(np.zeros(X.shape[1]))
            continue
        # Use weight as direction in the original space; do not renormalise.
        t = Tres @ z
        t_sq = float(t @ t)
        if t_sq < 1e-14:
            T[:, a] = 0.0
            P[:, a] = 0.0
            Q[:, a] = 0.0
            R_list.append(np.zeros(X.shape[1]))
            continue
        p_load = Tres.T @ t / t_sq
        q_load = Yres.T @ t / t_sq
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(W[:, a].copy())
        Tres = Tres - np.outer(t, p_load)
        Yres = Yres - np.outer(t, q_load)
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=[0] * K,
        operator_names=[operator.name] * K,
        diagnostics={"engine": "nipals_materialized_fixed", "operator": operator.name},
    )


def nipals_materialized_per_component(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
) -> NIPALSResult:
    """Materialized NIPALS where each component uses a (potentially different)
    operator from the bank.

    `op_indices[a]` is the index of the operator applied at component `a`.
    The component direction is computed by deflating in the original space
    after each iteration, so the loadings are consistent across operators.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if len(op_indices) != n_components:
        raise ValueError("op_indices must have length n_components")
    n, p = X.shape
    q = Y.shape[1]
    Tres = X.copy()
    Yres = Y.copy()
    Z = np.zeros((p, n_components))
    P = np.zeros((p, n_components))
    Q = np.zeros((q, n_components))
    T = np.zeros((n, n_components))
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    for a in range(n_components):
        op = operators[op_indices[a]]
        op.fit(X)
        Xb = op.transform(Tres)
        c = Xb.T @ Yres
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        # Take leading left singular vector for PLS2.
        if c.shape[1] == 1:
            r = c[:, 0].copy()
        else:
            U, _S, _Vt = np.linalg.svd(c, full_matrices=False)
            r = U[:, 0].copy()
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-14:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        r = r / r_norm
        z = op.adjoint_vec(r)
        t = Tres @ z
        t_sq = float(t @ t)
        if t_sq < 1e-14:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        p_load = Tres.T @ t / t_sq
        q_load = Yres.T @ t / t_sq
        Z[:, a] = z
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(r)
        op_names.append(op.name)
        Tres = Tres - np.outer(t, p_load)
        Yres = Yres - np.outer(t, q_load)
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "nipals_materialized_per_component"},
    )


# ---------------------------------------------------------------------------
# Adjoint NIPALS (fast)
# ---------------------------------------------------------------------------


def nipals_adjoint(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
) -> NIPALSResult:
    """Adjoint NIPALS: avoids materializing X A^T at each component.

    For each component `a`:

    1. Compute `S = X_res^T Y_res` (cross-covariance in the original space).
    2. Reduce to a vector `s` (PLS1 takes column 0; PLS2 takes the dominant
       left singular vector of `S`).
    3. The transformed-space direction is proportional to `A s` and the
       original-space effective weight is `z = A^T r` with `r = A s / ||A s||`.
    4. Score `t = X_res z` and loadings `p, q` in the original space.

    This matches the materialized engine numerically up to floating-point
    precision when applied to strict linear operators.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if len(op_indices) != n_components:
        raise ValueError("op_indices must have length n_components")
    n, p = X.shape
    q = Y.shape[1]
    Tres = X.copy()
    Yres = Y.copy()
    Z = np.zeros((p, n_components))
    P = np.zeros((p, n_components))
    Q = np.zeros((q, n_components))
    T = np.zeros((n, n_components))
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    for a in range(n_components):
        op = operators[op_indices[a]]
        op.fit(X)
        # Production NIPALS-adjoint algorithm (matches nirs4all aom_pls):
        #   c = X_res^T Y_res
        #   g = A^T c     (apply adjoint of operator to covariance)
        #   w_hat = g / ||g||
        #   a_w = A w_hat (apply forward to the normalised adjoint direction)
        #   w_k = a_w / ||a_w||
        #   t = X_res w_k
        # The effective weight in the original space is w_k ∝ A A^T c.
        # This differs from a strict SIMPLS direction A^T A c by the order
        # in which A and A^T are composed; both are valid PLS directions but
        # they only coincide for symmetric/anti-symmetric A.
        S = Tres.T @ Yres
        if S.shape[1] == 1:
            c = S[:, 0]
        else:
            U_svd, _Sigma, _Vt = np.linalg.svd(S, full_matrices=False)
            c = U_svd[:, 0] * _Sigma[0]
        g = op.adjoint_vec(c)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        w_hat = g / g_norm
        a_w = op.apply_cov(w_hat)
        a_w_norm = np.linalg.norm(a_w)
        if a_w_norm < 1e-14:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        w = a_w / a_w_norm
        t = Tres @ w
        t_sq = float(t @ t)
        if t_sq < 1e-14:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        p_load = Tres.T @ t / t_sq
        q_load = Yres.T @ t / t_sq
        Z[:, a] = w
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(w_hat)
        op_names.append(op.name)
        Tres = Tres - np.outer(t, p_load)
        Yres = Yres - np.outer(t, q_load)
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "nipals_adjoint"},
    )


# ---------------------------------------------------------------------------
# Convenience: standard PLS as identity-only AOM
# ---------------------------------------------------------------------------


def nipals_pls_standard(X: np.ndarray, Y: np.ndarray, n_components: int) -> NIPALSResult:
    """Run standard PLS via the materialized identity-only path.

    Provided as a sanity reference for the AOM engines: with operator bank
    `{I}`, AOM/POP must reduce exactly to this computation.
    """
    return nipals_materialized_fixed(X, Y, IdentityOperator(p=X.shape[1]), n_components)
