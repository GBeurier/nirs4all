"""PyTorch backend for adjoint NIPALS, covariance SIMPLS, and superblock SIMPLS.

Mirrors the NumPy reference engines but operates on PyTorch tensors so that
GPU acceleration is available when CUDA is present. Falls back to CPU when
CUDA is unavailable. The interface accepts NumPy arrays and returns NumPy
arrays so that tests and downstream code remain backend-agnostic.

The Torch engines reuse the operator protocol's NumPy `apply_cov` and
`adjoint_vec` for cheap operators (kernels and projections); the heavy
matrix multiplications `X @ z`, `X.T @ Y`, and the SVDs run on Torch tensors.

This is a parity layer rather than a feature-complete backend: it implements
the engines required by section 5 of `Prompt.md`:

- `nipals_adjoint`
- `simpls_covariance`
- `superblock_simpls`
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import-time guard
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .nipals import NIPALSResult
from .operators import LinearSpectralOperator


def torch_available() -> bool:
    return _TORCH_AVAILABLE


def _device(device: str | None = None):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _as_tensor(x: np.ndarray, device, dtype=None):
    if dtype is None:
        dtype = torch.float64
    return torch.as_tensor(np.asarray(x, dtype=float), device=device, dtype=dtype)


def _dominant_direction_torch(S):
    """Leading left singular direction of `S`."""
    if S.ndim == 1:
        return S
    if S.shape[1] == 1:
        return S[:, 0]
    U, _Sv, _Vt = torch.linalg.svd(S, full_matrices=False)
    return U[:, 0]


# ---------------------------------------------------------------------------
# Operator helpers (route through NumPy implementations for cheap ops)
# ---------------------------------------------------------------------------


def _operator_apply_cov_torch(op: LinearSpectralOperator, S, device, dtype):
    """Apply `A` to a Torch tensor `S` of shape `(p, q)` or `(p,)`."""
    S_np = S.detach().cpu().numpy()
    out_np = op.apply_cov(S_np)
    return torch.as_tensor(out_np, device=device, dtype=dtype)


def _operator_adjoint_vec_torch(op: LinearSpectralOperator, v, device, dtype):
    v_np = v.detach().cpu().numpy()
    out_np = op.adjoint_vec(v_np)
    return torch.as_tensor(out_np, device=device, dtype=dtype)


def _operator_transform_torch(op: LinearSpectralOperator, X, device, dtype):
    X_np = X.detach().cpu().numpy()
    out_np = op.transform(X_np)
    return torch.as_tensor(out_np, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Adjoint NIPALS (Torch)
# ---------------------------------------------------------------------------


def nipals_adjoint_torch(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
    device: str | None = None,
    dtype: str = "float64",
) -> NIPALSResult:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    dev = _device(device)
    th_dtype = torch.float64 if dtype == "float64" else torch.float32
    Xt = _as_tensor(X, dev, th_dtype)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    Yt = _as_tensor(Y_arr, dev, th_dtype)
    n, p = Xt.shape
    q = Yt.shape[1]
    Z = torch.zeros((p, n_components), device=dev, dtype=th_dtype)
    P = torch.zeros((p, n_components), device=dev, dtype=th_dtype)
    Q = torch.zeros((q, n_components), device=dev, dtype=th_dtype)
    T = torch.zeros((n, n_components), device=dev, dtype=th_dtype)
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    Tres = Xt.clone()
    Yres = Yt.clone()
    eps = 1e-14
    for a in range(n_components):
        op = operators[op_indices[a]]
        op.fit(np.asarray(X, dtype=float))
        S = Tres.T @ Yres
        if S.shape[1] == 1:
            s = S[:, 0]
        else:
            s = _dominant_direction_torch(S)
        a_s = _operator_apply_cov_torch(op, s, dev, th_dtype)
        a_s_norm = torch.linalg.norm(a_s)
        if float(a_s_norm) < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        r = a_s / a_s_norm
        z = _operator_adjoint_vec_torch(op, r, dev, th_dtype)
        t = Tres @ z
        t_sq = float(t @ t)
        if t_sq < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        p_load = Tres.T @ t / t_sq
        q_load = Yres.T @ t / t_sq
        Z[:, a] = z
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(r.detach().cpu().numpy())
        op_names.append(op.name)
        Tres = Tres - torch.outer(t, p_load)
        Yres = Yres - torch.outer(t, q_load)
    return NIPALSResult(
        Z=Z.detach().cpu().numpy(),
        P=P.detach().cpu().numpy(),
        Q=Q.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "nipals_adjoint", "backend": "torch", "device": str(dev)},
    )


# ---------------------------------------------------------------------------
# Covariance SIMPLS (Torch)
# ---------------------------------------------------------------------------


def simpls_covariance_torch(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    op_indices: Sequence[int],
    n_components: int,
    orthogonalization: str = "original",
    device: str | None = None,
    dtype: str = "float64",
) -> NIPALSResult:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    if orthogonalization == "transformed" and len(set(op_indices)) > 1:
        raise ValueError("orthogonalization='transformed' requires a single fixed operator")
    dev = _device(device)
    th_dtype = torch.float64 if dtype == "float64" else torch.float32
    Xt = _as_tensor(X, dev, th_dtype)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    Yt = _as_tensor(Y_arr, dev, th_dtype)
    n, p = Xt.shape
    q = Yt.shape[1]
    K = int(n_components)
    S = Xt.T @ Yt
    Z = torch.zeros((p, K), device=dev, dtype=th_dtype)
    P = torch.zeros((p, K), device=dev, dtype=th_dtype)
    Q = torch.zeros((q, K), device=dev, dtype=th_dtype)
    T = torch.zeros((n, K), device=dev, dtype=th_dtype)
    V = torch.zeros((p, K), device=dev, dtype=th_dtype)
    R_list: List[np.ndarray] = []
    op_names: List[str] = []
    eps = 1e-14
    for a in range(K):
        op = operators[op_indices[a]]
        op.fit(np.asarray(X, dtype=float))
        S_b = _operator_apply_cov_torch(op, S, dev, th_dtype)
        if S_b.ndim == 1:
            S_b = S_b.unsqueeze(1)
        r = _dominant_direction_torch(S_b)
        r_norm = torch.linalg.norm(r)
        if float(r_norm) < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        r = r / r_norm
        z = _operator_adjoint_vec_torch(op, r, dev, th_dtype)
        t = Xt @ z
        t_norm = torch.linalg.norm(t)
        if float(t_norm) < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        t = t / t_norm
        z = z / t_norm
        p_load = Xt.T @ t
        q_load = Yt.T @ t
        v = p_load.clone()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ v)
        v_norm = torch.linalg.norm(v)
        if float(v_norm) < eps:
            R_list.append(np.zeros(p))
            op_names.append(op.name)
            continue
        V[:, a] = v / v_norm
        S = S - torch.outer(V[:, a], (V[:, a] @ S))
        Z[:, a] = z
        T[:, a] = t
        P[:, a] = p_load
        Q[:, a] = q_load
        R_list.append(r.detach().cpu().numpy())
        op_names.append(op.name)
    return NIPALSResult(
        Z=Z.detach().cpu().numpy(),
        P=P.detach().cpu().numpy(),
        Q=Q.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        R=R_list,
        operator_indices=list(op_indices),
        operator_names=op_names,
        diagnostics={"engine": "simpls_covariance", "backend": "torch", "device": str(dev)},
    )


# ---------------------------------------------------------------------------
# Superblock SIMPLS (Torch)
# ---------------------------------------------------------------------------


def superblock_simpls_torch(
    X: np.ndarray,
    Y: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    n_components: int,
    device: str | None = None,
    dtype: str = "float64",
) -> Tuple[NIPALSResult, np.ndarray]:
    """Concatenate operator views and run SIMPLS on the wide matrix in Torch."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    dev = _device(device)
    th_dtype = torch.float64 if dtype == "float64" else torch.float32
    blocks = []
    groups: List[int] = []
    op_names: List[str] = []
    X_arr = np.asarray(X, dtype=float)
    for b, op in enumerate(operators):
        op.fit(X_arr)
        Xb = op.transform(X_arr)
        blocks.append(Xb)
        groups.extend([b] * Xb.shape[1])
        op_names.append(op.name)
    Xwide = np.hstack(blocks)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    # Run torch SIMPLS-standard on the wide matrix via covariance-form with
    # identity bank (the operator was already applied by concatenation).
    from .operators import IdentityOperator
    identity = IdentityOperator(p=Xwide.shape[1])
    res = simpls_covariance_torch(
        Xwide, Y_arr, [identity], [0] * n_components, n_components,
        orthogonalization="original", device=device, dtype=dtype,
    )
    res.diagnostics["engine"] = "superblock_simpls"
    res.diagnostics["operators"] = op_names
    res.diagnostics["block_sizes"] = [b.shape[1] for b in blocks]
    return res, np.asarray(groups, dtype=int)
