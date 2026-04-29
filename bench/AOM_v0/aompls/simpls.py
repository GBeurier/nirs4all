"""SIMPLS engines: materialized reference, covariance-space, and superblock.

Conventions match `nipals.py`:

- `X in R^{n x p}` and `Y in R^{n x q}` centered.
- Operator covariance identity `(X A^T)^T Y = A X^T Y`.
- Effective weights `z_a` live in the original space; coefficients are
  `B = Z (P^T Z)^+ Q^T`.

Two SIMPLS variants are implemented:

1. `simpls_materialized_fixed`: build `X_b = X A^T` and run de Jong's SIMPLS
   on the transformed matrix, then map the transformed weights back to the
   original space. This is the slow reference.

2. `simpls_covariance`: compute `S = X^T Y` once, then for each component
   evaluate operators in covariance space via `S_b = A_b S`, choose the
   dominant direction, and update an original-space orthogonal basis. This
   is the fast variant used in benchmarks.

Both engines support per-component operator sequences (POP) and a single
fixed operator (AOM).

The superblock variant concatenates all operator views into one wide matrix
and runs standard SIMPLS on top.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .nipals import NIPALSResult
from .operators import IdentityOperator, LinearSpectralOperator


def _dominant_direction(S: np.ndarray) -> np.ndarray:
    """Return the dominant left singular direction of `S` (or column 0 for PLS1)."""
    if S.ndim == 1:
        return S.copy()
    if S.shape[1] == 1:
        return S[:, 0].copy()
    U, _Sv, _Vt = np.linalg.svd(S, full_matrices=False)
    return U[:, 0].copy()


# ---------------------------------------------------------------------------
# Materialized SIMPLS through a fixed operator (reference)
# ---------------------------------------------------------------------------


def simpls_standard(X: np.ndarray, Y: np.ndarray, n_components: int) -> NIPALSResult:
    """de Jong's SIMPLS reference on a single matrix `(X, Y)`.

    Returns a NIPALSResult-like object with original-space weights `Z`,
    loadings `P, Q`, and scores `T`. The implementation follows the
    formulation:

    ```
    S = X^T Y
    for a in 1..K:
        r = u_1(S)             # leading left singular vector
        t = X r;  norm t
        p = X^T t
        q = Y^T t
        deflate S: S <- (I - V V^T) S where V is the Gram-Schmidt basis of P
        accumulate
    ```
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, p = X.shape
    q = Y.shape[1]
    K = int(n_components)
    Z = np.zeros((p, K))
    P = np.zeros((p, K))
    Q = np.zeros((q, K))
    T = np.zeros((n, K))
    V = np.zeros((p, K))  # Gram-Schmidt basis of loadings
    R_list: List[np.ndarray] = []
    S = X.T @ Y
    eps = 1e-14
    for a in range(K):
        r = _dominant_direction(S)
        t = X @ r
        t_norm = np.linalg.norm(t)
        if t_norm < eps:
            break
        t = t / t_norm
        r = r / t_norm
        p_load = X.T @ t
        q_load = Y.T @ t
        # Orthogonalise loading against previous basis
        v = p_load.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm < eps:
            break
        v = v / v_norm
        # Deflate S
        S = S - np.outer(v, v.T @ S)
        Z[:, a] = r
        P[:, a] = p_load
        Q[:, a] = q_load
        T[:, a] = t
        V[:, a] = v
        R_list.append(r.copy())
    actual_K = int(np.sum(np.linalg.norm(Z, axis=0) > 0))
    return NIPALSResult(
        Z=Z[:, :actual_K],
        P=P[:, :actual_K],
        Q=Q[:, :actual_K],
        T=T[:, :actual_K],
        R=R_list[:actual_K],
        operator_indices=[0] * actual_K,
        operator_names=["identity"] * actual_K,
        diagnostics={"engine": "simpls_standard", "basis": V[:, :actual_K]},
    )


def simpls_materialized_fixed(
    X: np.ndarray,
    Y: np.ndarray,
    operator: LinearSpectralOperator,
    n_components: int,
) -> NIPALSResult:
    """Materialized SIMPLS through a single fixed operator.

    Build `Xb = X A^T`, run standard SIMPLS, then map transformed weights
    `r_a` back to original space `z_a = A^T r_a`. Loadings are recomputed in
    the original space so prediction uses `(X - x_mean) B + intercept`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    operator.fit(X)
    Xb = operator.transform(X)
    res_b = simpls_standard(Xb, Y, n_components)
    K = res_b.Z.shape[1]
    p = X.shape[1]
    Z = np.empty((p, K))
    for a in range(K):
        Z[:, a] = operator.adjoint_vec(res_b.Z[:, a])
    # Recompute loadings in original space using V-deflated SIMPLS update.
    P = np.empty((p, K))
    Q = np.empty((Y.shape[1], K))
    T = np.empty((X.shape[0], K))
    V = np.zeros((p, K))
    R_list: List[np.ndarray] = []
    eps = 1e-14
    for a in range(K):
        z = Z[:, a]
        t = X @ z
        t_norm = np.linalg.norm(t)
        if t_norm < eps:
            P[:, a] = 0.0
            Q[:, a] = 0.0
            T[:, a] = 0.0
            R_list.append(np.zeros(p))
            continue
        t = t / t_norm
        z = z / t_norm
        Z[:, a] = z
        p_load = X.T @ t
        q_load = Y.T @ t
        v = p_load.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm < eps:
            P[:, a] = 0.0
            Q[:, a] = 0.0
            T[:, a] = 0.0
            R_list.append(np.zeros(p))
            continue
        V[:, a] = v / v_norm
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(res_b.Z[:, a].copy())
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=[0] * K,
        operator_names=[operator.name] * K,
        diagnostics={"engine": "simpls_materialized_fixed", "operator": operator.name, "basis": V[:, :K]},
    )


def simpls_materialized_per_component(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
    orthogonalization: str = "original",
) -> NIPALSResult:
    """SIMPLS with a (potentially different) operator per component.

    For each component `a` with selected operator `b_a`:

    1. Build the deflated `X_res^T Y_res` covariance.
    2. Apply operator `A_{b_a}` to the covariance: `S_b = A_b S`.
    3. Take `r = u_1(S_b)`.
    4. The original-space effective weight is `z = A_b^T r` after deflation
       in the chosen orthogonalization mode.

    Two modes:
        - `original`: orthogonalize using the original-space loadings (default
          when operators vary across components).
        - `transformed`: deflate in the transformed space of the current
          operator (only valid for fixed-operator runs; raises otherwise).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if len(op_indices) != n_components:
        raise ValueError("op_indices must have length n_components")
    if orthogonalization not in ("original", "transformed"):
        raise ValueError("orthogonalization must be 'original' or 'transformed'")
    if orthogonalization == "transformed" and len(set(op_indices)) > 1:
        raise ValueError("orthogonalization='transformed' requires a single fixed operator")
    n, p = X.shape
    q = Y.shape[1]
    K = int(n_components)
    Tres = X.copy()
    Yres = Y.copy()
    Z = np.zeros((p, K))
    P = np.zeros((p, K))
    Q = np.zeros((q, K))
    T = np.zeros((n, K))
    V = np.zeros((p, K))
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    eps = 1e-14
    for a in range(K):
        op = operators[op_indices[a]]
        op.fit(X)
        S = Tres.T @ Yres
        S_b = op.apply_cov(S)
        if S_b.ndim == 1:
            S_b = S_b.reshape(-1, 1)
        r = _dominant_direction(S_b)
        r_norm = np.linalg.norm(r)
        if r_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        r = r / r_norm
        z = op.adjoint_vec(r)
        t = X @ z if orthogonalization == "original" else Tres @ z
        # Use original X for loadings to ensure prediction works on (X - x_mean) only.
        t_orig = X @ z
        t_norm = np.linalg.norm(t_orig)
        if t_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        t_orig = t_orig / t_norm
        z = z / t_norm
        p_load = X.T @ t_orig
        q_load = Y.T @ t_orig
        v = p_load.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        V[:, a] = v / v_norm
        Z[:, a] = z
        T[:, a] = t_orig
        P[:, a] = p_load
        Q[:, a] = q_load
        # Deflate residuals in the original space using projection on V.
        # Tres <- (I - v v^T) Tres-flow:
        Tres = Tres - np.outer(Tres @ V[:, a], V[:, a])
        Yres = Yres - np.outer(t_orig, q_load)
        R_list.append(r.copy())
        op_names.append(op.name)
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "simpls_materialized_per_component", "basis": V[:, :K]},
    )


# ---------------------------------------------------------------------------
# Covariance-space SIMPLS (fast)
# ---------------------------------------------------------------------------


def simpls_covariance(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
    orthogonalization: str = "original",
) -> NIPALSResult:
    """Covariance-space SIMPLS using the identity `S_b = A_b X^T Y`.

    The cross-covariance `S = X^T Y` is computed once. Operators are applied
    directly to `S` to evaluate candidates in covariance space, avoiding the
    `X A_b^T` materialization. Component-wise deflation uses the original-space
    Gram-Schmidt basis of the loadings.

    When `orthogonalization="transformed"` is requested (single fixed operator
    only), the engine delegates to `simpls_materialized_fixed` so that the
    transformed-space basis is built honestly.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if len(op_indices) != n_components:
        raise ValueError("op_indices must have length n_components")
    if orthogonalization not in ("original", "transformed"):
        raise ValueError("orthogonalization must be 'original' or 'transformed'")
    if orthogonalization == "transformed" and len(set(op_indices)) > 1:
        raise ValueError("orthogonalization='transformed' requires a single fixed operator")
    if orthogonalization == "transformed":
        # Delegate to materialized fixed which honestly orthogonalises in the
        # transformed space (Codex math review, HIGH).
        res = simpls_materialized_fixed(X, Y, operators[op_indices[0]], n_components)
        res.diagnostics["engine"] = "simpls_covariance:delegated_to_materialized_fixed"
        return res
    n, p = X.shape
    q = Y.shape[1]
    K = int(n_components)
    S = X.T @ Y
    Z = np.zeros((p, K))
    P = np.zeros((p, K))
    Q = np.zeros((q, K))
    T = np.zeros((n, K))
    V = np.zeros((p, K))
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    eps = 1e-14
    for a in range(K):
        op = operators[op_indices[a]]
        op.fit(X)
        S_b = op.apply_cov(S)
        if S_b.ndim == 1:
            S_b = S_b.reshape(-1, 1)
        r = _dominant_direction(S_b)
        r_norm = np.linalg.norm(r)
        if r_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        r = r / r_norm
        z = op.adjoint_vec(r)
        t = X @ z
        t_norm = np.linalg.norm(t)
        if t_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        t = t / t_norm
        z = z / t_norm
        p_load = X.T @ t
        q_load = Y.T @ t
        v = p_load.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ v)
        v_norm = np.linalg.norm(v)
        if v_norm < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        V[:, a] = v / v_norm
        # Deflate covariance: S <- (I - v v^T) S
        S = S - np.outer(V[:, a], V[:, a].T @ S)
        Z[:, a] = z
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(r.copy())
        op_names.append(op.name)
    return NIPALSResult(
        Z=Z,
        P=P,
        Q=Q,
        T=T,
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "simpls_covariance", "basis": V[:, :K]},
    )


# ---------------------------------------------------------------------------
# Superblock SIMPLS
# ---------------------------------------------------------------------------


def superblock_simpls(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    n_components: int,
    block_weights: Optional[Sequence[float]] = None,
) -> Tuple[NIPALSResult, np.ndarray]:
    """Concatenate operator-transformed views and run SIMPLS on the wide matrix.

    Optional `block_weights[b]` scales block `b` of the concatenated matrix
    so that high-gain operators do not dominate solely by amplitude.

    Returns a NIPALSResult mapped back to the **original** feature space
    (so the predictor accepts plain `(n, p)` matrices) and the per-column
    group-membership vector. The wide-space loadings are recoverable from
    `result.diagnostics["wide_basis"]`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, p = X.shape
    q = Y.shape[1]
    if block_weights is None:
        block_weights = [1.0] * len(operators)
    if len(block_weights) != len(operators):
        raise ValueError("block_weights must match number of operators")
    block_weights = np.asarray(block_weights, dtype=float)
    blocks: List[np.ndarray] = []
    groups: List[int] = []
    op_names: List[str] = []
    for b, op in enumerate(operators):
        op.fit(X)
        Xb = op.transform(X) * float(block_weights[b])
        blocks.append(Xb)
        groups.extend([b] * Xb.shape[1])
        op_names.append(op.name)
    Xwide = np.hstack(blocks)
    res_wide = simpls_standard(Xwide, Y, n_components)
    K = res_wide.Z.shape[1]
    # Map wide-space weights back to original (n, p) feature space.
    # Z_orig[:, a] = sum_b alpha_b * A_b^T Z_wide_block_b[:, a]
    Z_orig = np.zeros((p, K))
    groups_arr = np.asarray(groups, dtype=int)
    for b, op in enumerate(operators):
        cols = np.where(groups_arr == b)[0]
        if cols.size == 0:
            continue
        Z_block = res_wide.Z[cols, :]
        for a in range(K):
            Z_orig[:, a] += float(block_weights[b]) * op.adjoint_vec(Z_block[:, a])
    # Recompute scores and loadings in the original space.
    T = X @ Z_orig
    P_orig = np.zeros((p, K))
    Q_orig = np.zeros((q, K))
    for a in range(K):
        t = T[:, a]
        t_sq = float(t @ t)
        if t_sq < 1e-14:
            continue
        P_orig[:, a] = X.T @ t / t_sq
        Q_orig[:, a] = Y.T @ t / t_sq
    res_orig = NIPALSResult(
        Z=Z_orig,
        P=P_orig,
        Q=Q_orig,
        T=T,
        R=list(res_wide.R),
        operator_indices=list(range(len(operators))),
        operator_names=op_names,
        diagnostics={
            "engine": "superblock_simpls",
            "operators": op_names,
            "block_sizes": [b.shape[1] for b in blocks],
            "block_weights": block_weights.tolist(),
            "wide_basis": res_wide.diagnostics.get("basis"),
            "original_feature_space": True,
        },
    )
    return res_orig, groups_arr


# ---------------------------------------------------------------------------
# PLS standard via covariance SIMPLS (identity-only check)
# ---------------------------------------------------------------------------


def simpls_pls_standard(X: np.ndarray, Y: np.ndarray, n_components: int) -> NIPALSResult:
    """Identity-only AOM via covariance SIMPLS, used as a sanity reference."""
    return simpls_covariance(
        X,
        Y,
        [IdentityOperator(p=X.shape[1])],
        [0] * n_components,
        n_components,
        orthogonalization="original",
    )
