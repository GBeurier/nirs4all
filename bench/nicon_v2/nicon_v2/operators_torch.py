"""AOM linear operators as frozen / learnable PyTorch layers.

Each AOM operator `A ∈ R^{p × p}` acts on row spectra as `X_b = X A^T`.
For shift-invariant operators (Identity, SG, Gaussian, FD, Norris-Williams)
this is a 1-D convolution with a fixed kernel; for shift-variant operators
(Detrend, Whittaker) we materialise `A^T` once and apply a single matmul.

This module exposes a unified `OperatorLayer` API so a multi-branch CNN can
mix shift-invariant and shift-variant operators without bespoke wiring.

By default operators are **frozen** (`requires_grad=False` on the kernel /
matrix). Setting ``trainable=True`` makes the kernel learnable; an optional
L2 penalty against the initial kernel is exposed via `regularisation_loss()`
so the trunk training loop can add it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    kind: str  # "conv" (shift-invariant) or "matrix" (shift-variant)
    builder: Callable[[int], np.ndarray]  # returns kernel (1-D for "conv") or full matrix (p x p)


# ---------------------------------------------------------------------------
# Builders for AOM compact-bank-style operators.
# ---------------------------------------------------------------------------


def _sg_kernel(window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    """Savitzky-Golay 1-D kernel; the "convolution" direction of scipy.savgol_filter.

    Reverses scipy's coefficient vector so that ``F.conv1d`` (cross-correlation)
    matches scipy's convolution direction. Identical to `nicon_v2.preprocessing.FixedSavGol1D`.
    """
    from scipy.signal import savgol_coeffs

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv).astype(np.float64)
    return np.ascontiguousarray(coeffs[::-1].copy(), dtype=np.float32)


def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(np.ceil(truncate * sigma))
    if radius < 1:
        radius = 1
    if radius % 2 == 0:
        radius += 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)


def _identity_kernel() -> np.ndarray:
    return np.array([1.0], dtype=np.float32)


def _finite_difference_kernel(order: int = 1) -> np.ndarray:
    if order == 1:
        return np.array([0.5, 0.0, -0.5], dtype=np.float32)  # centred (for `F.conv1d` we still flip)
    if order == 2:
        return np.array([1.0, -2.0, 1.0], dtype=np.float32)
    raise ValueError(f"unsupported FD order {order}")


def _norris_williams_kernel(gap: int, smoothing: int = 1, order: int = 1) -> np.ndarray:
    """Norris-Williams gap-derivative kernel (smoothed finite-difference across `gap`)."""
    smooth = np.ones(smoothing, dtype=np.float64) / max(1, smoothing)
    half = gap
    fd = np.zeros(2 * half + 1, dtype=np.float64)
    fd[0] = -1.0
    fd[-1] = +1.0
    fd /= 2.0 * half
    out = np.convolve(fd, smooth, mode="same")
    return out.astype(np.float32)


def _whittaker_smoother_matrix(p: int, lam: float = 10.0, order: int = 2) -> np.ndarray:
    """Whittaker smoother as an explicit (p × p) matrix.

    Solves `(I + λ D^T D) z = y` for the smoothed `z`. The matrix `(I + λ D^T D)^{-1}`
    is the Whittaker smoother operator. For NIRS we typically use `order=2`
    (curvature penalty).
    """
    D = np.diff(np.eye(p), n=order, axis=0)
    A = np.linalg.inv(np.eye(p) + lam * (D.T @ D))
    return A.astype(np.float32)


def _low_rank_factor(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD factorisation: A ≈ U @ V^T with U: (p, k), V^T: (k, p).

    Used by `LowRankMatrixOperator` (V2H — Codex round 7 Q2 follow-up): keeps
    the flexibility of a learnable (p × p) matrix while reducing parameter
    count from O(p²) to O(p · k). Initialised from the truncated-SVD of the
    closed-form AOM operator (Detrend / Whittaker), so the prior is preserved.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(rank, S.shape[0])
    U_k = (U[:, :k] * S[:k]).astype(np.float32)        # (p, k) — absorbs the singular values
    V_k = Vt[:k].astype(np.float32)                     # (k, p)
    return U_k, V_k


# ---------------------------------------------------------------------------
# RMS normalisation per branch — mirrors AOM-Ridge `compute_block_scales_from_xt`.
# ---------------------------------------------------------------------------


class RMSBranchNorm(nn.Module):
    """Divide the post-branch tensor by its training-time RMS.

    AOM-Ridge applies block scaling `s_b = 1 / sqrt(trace(K_b) / n)` so that all
    operator views contribute on a common scale before the superblock kernel is
    formed. The CNN analogue is to divide the per-branch feature tensor by its
    mean RMS over the training set; we estimate this on-the-fly during the first
    forward pass in `train()` mode and freeze it after `freeze()` is called or on
    transition to `eval()` mode if `freeze_on_eval=True`.

    When ``learnable=True`` (V2L), the inverse-RMS scale is stored as a
    `nn.Parameter` initialised from the first-batch estimate and trained by
    gradient — this lets the network refine per-branch weighting through
    training instead of locking it at one snapshot.
    """

    def __init__(self, freeze_on_eval: bool = True, eps: float = 1e-8,
                 learnable: bool = False, init_mode: str = "inverse_rms"):
        """``init_mode`` (Codex round 9 diagnostic):
        * ``"inverse_rms"`` — scale starts at `1 / RMS(branch)` from first batch (V2L default).
        * ``"unit"``        — scale starts at 1.0 (no data-dependent init); ablation control.
        """
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(0))
        self.freeze_on_eval = freeze_on_eval
        self.eps = eps
        self.learnable = learnable
        if init_mode not in ("inverse_rms", "unit"):
            raise ValueError(f"unknown init_mode {init_mode!r}; expected inverse_rms | unit")
        self.init_mode = init_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.fitted.item() == 0:
            with torch.no_grad():
                if self.init_mode == "inverse_rms":
                    rms = torch.sqrt((x * x).mean()).clamp(min=self.eps)
                    target = 1.0 / rms
                else:
                    target = torch.tensor(1.0, device=x.device)
                if isinstance(self.scale, nn.Parameter):
                    self.scale.data.copy_(target)
                else:
                    self.scale.copy_(target)
                self.fitted.fill_(1)
        return x * self.scale

    @property
    def rms(self) -> torch.Tensor:  # back-compat with pre-V2L tests
        return 1.0 / self.scale.detach()


def _detrend_matrix(p: int, degree: int = 1) -> np.ndarray:
    """Detrend = (I − P) where P is the projection onto polynomials of `degree`.
    Returns the (p × p) matrix `A` so that `X_b = X A^T = X (I − P)^T`."""
    t = np.linspace(-1.0, 1.0, p)
    B = np.vander(t, degree + 1, increasing=True)
    # Projection P = B (B^T B)^-1 B^T.
    BtB_inv = np.linalg.pinv(B.T @ B)
    P = B @ BtB_inv @ B.T
    return (np.eye(p) - P).astype(np.float32)


def _snv_apply_inplace(x: torch.Tensor) -> torch.Tensor:
    """SNV is row-wise; not strictly linear so we implement as a layer (not a matrix)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-12)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Conv-style frozen operator layer
# ---------------------------------------------------------------------------


class FrozenConvOperator(nn.Module):
    """Frozen 1-D convolution applying a fixed kernel along the wavelength axis.

    Input shape: (N, 1, L). Output shape: (N, 1, L) with reflect-padding so that
    output length equals input length.

    The kernel is stored as a non-trainable `Parameter` if `trainable=False`,
    otherwise as a `Parameter` with grad. An L2-from-init penalty is available
    via `regularisation_loss()` so callers can add it to the training loss.
    """

    def __init__(self, kernel_init: np.ndarray, name: str = "op", trainable: bool = False,
                 reg_lambda: float = 0.0):
        super().__init__()
        kernel = torch.as_tensor(kernel_init, dtype=torch.float32).view(1, 1, -1)
        if trainable:
            self.kernel = nn.Parameter(kernel.clone())
        else:
            self.register_buffer("kernel", kernel)
        self.register_buffer("kernel_init", kernel.clone())
        self.window_length = kernel.shape[-1]
        self.name = name
        self.reg_lambda = float(reg_lambda)
        self.trainable = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FrozenConvOperator expects (N, C, L); got {tuple(x.shape)}")
        n, c, L = x.shape
        # Apply per-channel by reshaping to (N*C, 1, L).
        x_flat = x.reshape(n * c, 1, L)
        pad = self.window_length // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
        out = F.conv1d(x_pad, self.kernel)
        return out.reshape(n, c, L)

    def regularisation_loss(self) -> torch.Tensor:
        if self.trainable and self.reg_lambda > 0:
            return self.reg_lambda * ((self.kernel - self.kernel_init) ** 2).sum()
        return torch.tensor(0.0, device=self.kernel.device)


# ---------------------------------------------------------------------------
# Matrix-style frozen operator (Detrend, future Whittaker)
# ---------------------------------------------------------------------------


class FrozenMatrixOperator(nn.Module):
    """Apply a fixed `(p × p)` operator matrix as `X_b = X A^T`.

    Input shape: (N, 1, L). Output: (N, 1, L). Useful for shift-variant operators
    (Detrend, Whittaker) where the convolution model does not apply.
    """

    def __init__(self, matrix_init: np.ndarray, name: str = "op", trainable: bool = False,
                 reg_lambda: float = 0.0):
        super().__init__()
        if matrix_init.ndim != 2 or matrix_init.shape[0] != matrix_init.shape[1]:
            raise ValueError(f"matrix must be square; got {matrix_init.shape}")
        mat = torch.as_tensor(matrix_init, dtype=torch.float32)
        if trainable:
            self.matrix = nn.Parameter(mat.clone())
        else:
            self.register_buffer("matrix", mat)
        self.register_buffer("matrix_init", mat.clone())
        self.p = mat.shape[0]
        self.name = name
        self.reg_lambda = float(reg_lambda)
        self.trainable = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FrozenMatrixOperator expects (N, C, L); got {tuple(x.shape)}")
        n, c, L = x.shape
        if L != self.p:
            raise ValueError(f"FrozenMatrixOperator(p={self.p}) called with L={L}")
        # X_b = X @ A^T (rows: spectra)
        x2 = x.reshape(n * c, L)
        out = x2 @ self.matrix.transpose(0, 1)
        return out.reshape(n, c, L)

    def regularisation_loss(self) -> torch.Tensor:
        if self.trainable and self.reg_lambda > 0:
            return self.reg_lambda * ((self.matrix - self.matrix_init) ** 2).sum()
        return torch.tensor(0.0, device=self.matrix.device)


# ---------------------------------------------------------------------------
# Low-rank matrix operator — V2H (Codex round 7 Q2 follow-up)
# ---------------------------------------------------------------------------


class LowRankMatrixOperator(nn.Module):
    """Apply a low-rank approximation `A ≈ U V^T` of the (p × p) operator matrix.

    Initialised from the truncated-SVD (Eckart-Young best rank-k approximation)
    of the AOM matrix (Detrend, Whittaker). At rank `k = p` the approximation
    is exact; at smaller ranks it is a **valid chemometric prior**, not an
    exact replication of the closed-form operator (Codex round 8 F1):

    * **Detrend(degree=1)**: full matrix `A = I − P_B` has rank `p − 2`. At
      `k < p − 2` the trend subspace is still annihilated (its near-zero
      singular values survive truncation), but `p − 2 − k` signal-orthogonal
      directions are lost. Detrending fidelity degrades gracefully.
    * **Whittaker** `(I + λ D^T D)^{-1}`: full matrix is rank `p`. Truncation
      to rank `k` converts the smooth-shrinkage prior into hard removal of
      the `p − k` low-eigenvalue modes — a stronger smoothing than the
      closed-form operator at high-frequency modes.

    Trainable parameters: `U ∈ R^{p × k}`, `V ∈ R^{k × p}` — total `2 · p · k`
    instead of `p²`. With `p=2151, k=16`, this is `68,832` instead of
    `4,626,801` (98.5 % reduction). Empirically (round 8) the rank constraint
    acts as implicit regularisation that improves cohort-median rmsep over
    full-rank trainable matrices.
    """

    def __init__(self, matrix_init: np.ndarray, rank: int = 16, name: str = "lowrank",
                 trainable: bool = True, reg_lambda: float = 0.0):
        super().__init__()
        if matrix_init.ndim != 2 or matrix_init.shape[0] != matrix_init.shape[1]:
            raise ValueError(f"matrix must be square; got {matrix_init.shape}")
        U_init, V_init = _low_rank_factor(matrix_init, rank=rank)
        if trainable:
            self.U = nn.Parameter(torch.as_tensor(U_init))
            self.V = nn.Parameter(torch.as_tensor(V_init))
        else:
            self.register_buffer("U", torch.as_tensor(U_init))
            self.register_buffer("V", torch.as_tensor(V_init))
        self.register_buffer("U_init", torch.as_tensor(U_init))
        self.register_buffer("V_init", torch.as_tensor(V_init))
        self.p = matrix_init.shape[0]
        self.rank = U_init.shape[1]
        self.name = name
        self.reg_lambda = float(reg_lambda)
        self.trainable = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LowRankMatrixOperator expects (N, C, L); got {tuple(x.shape)}")
        n, c, L = x.shape
        if L != self.p:
            raise ValueError(f"LowRankMatrixOperator(p={self.p}) called with L={L}")
        # X_b = X @ A^T = X @ (U V^T)^T = (X @ V^T) @ U^T
        x2 = x.reshape(n * c, L)
        x_proj = x2 @ self.V.transpose(0, 1)            # (N*C, k)
        out = x_proj @ self.U.transpose(0, 1)           # (N*C, p)
        return out.reshape(n, c, L)

    def regularisation_loss(self) -> torch.Tensor:
        if not self.trainable or self.reg_lambda <= 0:
            return torch.tensor(0.0, device=self.U.device)
        return self.reg_lambda * (
            ((self.U - self.U_init) ** 2).sum() + ((self.V - self.V_init) ** 2).sum()
        )


# ---------------------------------------------------------------------------
# Stateful operators: SNV / MSC
# ---------------------------------------------------------------------------


class SNVOperator(nn.Module):
    """Row-wise standard-normal-variate. Stateless."""

    def __init__(self, name: str = "snv"):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _snv_apply_inplace(x)

    def regularisation_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)


class MSCOperator(nn.Module):
    """Multiplicative-scatter-correction with a learnable (or pre-fit) reference spectrum.

    Default behaviour: the reference is initialised to the first batch's mean and
    is **trainable**. To emulate AOM-Ridge's behaviour (reference fitted on train
    only, frozen at predict time) call `freeze()` after the first forward pass on
    the training fold.
    """

    def __init__(self, p: int, trainable: bool = True, reg_lambda: float = 0.0, name: str = "msc"):
        super().__init__()
        self.p = p
        self.name = name
        ref = torch.zeros(p)
        if trainable:
            self.reference = nn.Parameter(ref)
        else:
            self.register_buffer("reference", ref)
        self.register_buffer("reference_init", ref.clone())
        self.trainable = trainable
        self.reg_lambda = float(reg_lambda)
        self._fitted = False

    def fit(self, X: torch.Tensor | np.ndarray) -> "MSCOperator":
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        if X.dim() == 3:
            X = X[:, 0, :]
        with torch.no_grad():
            ref = X.mean(dim=0).to(self.reference.device)
            if self.trainable:
                self.reference.data.copy_(ref)
            else:
                self.reference.copy_(ref)
            self.reference_init.copy_(ref)
        self._fitted = True
        return self

    def freeze(self) -> "MSCOperator":
        if isinstance(self.reference, nn.Parameter):
            self.reference.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MSCOperator expects (N, C, L); got {tuple(x.shape)}")
        ref = self.reference
        ref_centered = ref - ref.mean()
        denom = (ref_centered * ref_centered).sum().clamp(min=1e-12)
        x_centered = x - x.mean(dim=-1, keepdim=True)
        b = (x_centered * ref_centered).sum(dim=-1, keepdim=True) / denom
        a = x.mean(dim=-1, keepdim=True) - b * ref.mean()
        b = b.clamp(min=1e-6)
        return (x - a) / b

    def regularisation_loss(self) -> torch.Tensor:
        if self.trainable and self.reg_lambda > 0:
            return self.reg_lambda * ((self.reference - self.reference_init) ** 2).sum()
        return torch.tensor(0.0, device=self.reference.device)


# ---------------------------------------------------------------------------
# Factories — mirror AOM-PLS compact_bank()
# ---------------------------------------------------------------------------


def aom_compact_branches_torch(
    p: int,
    trainable: bool = False,
    reg_lambda: float = 0.0,
    matrix_trainable: bool | None = None,
) -> list[nn.Module]:
    """Mirror of `aompls.banks.compact_bank()` as PyTorch layers.

    Identity + 5 SG variants + 2 Detrend + 1 FD = 9 operators. Detrend is a
    (p × p) matrix; everything else is a 1-D convolution.
    """
    if matrix_trainable is None:
        matrix_trainable = trainable
    layers: list[nn.Module] = [
        FrozenConvOperator(_identity_kernel(), name="identity", trainable=False, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 0), name="sg_smooth_w11p2",  trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(21, 3, 0), name="sg_smooth_w21p3",  trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 1), name="sg_d1_w11p2",      trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(21, 3, 1), name="sg_d1_w21p3",      trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 2), name="sg_d2_w11p2",      trainable=trainable, reg_lambda=reg_lambda),
        FrozenMatrixOperator(_detrend_matrix(p, 1), name="detrend_d1",    trainable=matrix_trainable, reg_lambda=reg_lambda),
        FrozenMatrixOperator(_detrend_matrix(p, 2), name="detrend_d2",    trainable=matrix_trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_finite_difference_kernel(1), name="fd_o1",    trainable=trainable, reg_lambda=reg_lambda),
    ]
    return layers


def aom_extended_strict_linear_branches_torch(
    p: int, trainable: bool = False, reg_lambda: float = 0.0,
    whittaker_lambda: float = 10.0,
    matrix_trainable: bool | None = None,
) -> list[nn.Module]:
    """Strict-linear AOM-style bank, extended with Norris-Williams + Whittaker.

    Per Codex round 5 review: keep this bank chemometrically aligned with the
    AOM operator grammar (no SNV / MSC / EMSC / OSC). Adds Norris-Williams gap
    derivative and Whittaker smoother (order 2) to the compact bank for a total
    of 11 branches.
    """
    if matrix_trainable is None:
        matrix_trainable = trainable
    layers = aom_compact_branches_torch(
        p, trainable=trainable, reg_lambda=reg_lambda, matrix_trainable=matrix_trainable
    )
    layers.append(FrozenConvOperator(_norris_williams_kernel(gap=5, smoothing=5, order=1),
                                     name="nw_g5_s5", trainable=trainable, reg_lambda=reg_lambda))
    layers.append(FrozenMatrixOperator(_whittaker_smoother_matrix(p, lam=whittaker_lambda, order=2),
                                       name="whittaker_l10", trainable=matrix_trainable, reg_lambda=reg_lambda))
    return layers


def cnn_only_extra_branches_torch(p: int, trainable: bool = False, reg_lambda: float = 0.0) -> list[nn.Module]:
    """Branches that go BEYOND the strict-linear AOM grammar.

    SNV / MSC / Gaussian smoothing are CNN-only additions: SNV/MSC are sample-wise
    fitted (so they break the strict-linear AOM kernel contract); Gaussian smooth
    is strict-linear but absent from AOM `compact`.
    """
    return [
        FrozenConvOperator(_gaussian_kernel(1.5), name="gauss_s1.5",
                           trainable=trainable, reg_lambda=reg_lambda),
        SNVOperator(),
        MSCOperator(p=p, trainable=trainable, reg_lambda=reg_lambda),
    ]


def full_branches_torch(
    p: int,
    trainable: bool = False,
    reg_lambda: float = 0.0,
    matrix_trainable: bool | None = None,
) -> list[nn.Module]:
    """Strict-linear AOM-extended + CNN-only extras (Gaussian, SNV, MSC) = ~14 branches."""
    return aom_extended_strict_linear_branches_torch(
        p, trainable=trainable, reg_lambda=reg_lambda, matrix_trainable=matrix_trainable
    ) \
        + cnn_only_extra_branches_torch(p, trainable=trainable, reg_lambda=reg_lambda)


# Back-compat alias — old name kept until phase 6 publication run is locked in.
extended_branches_torch = full_branches_torch


def aom_extended_lowrank_branches_torch(
    p: int,
    trainable: bool = True,
    reg_lambda: float = 0.0,
    whittaker_lambda: float = 10.0,
    rank: int = 16,
) -> list[nn.Module]:
    """V2H bank: same 11 strict-linear AOM ops as `aom_extended_strict_linear_branches_torch`,
    but Detrend/Whittaker are stored as **low-rank** A ≈ U V^T (rank=`rank`).

    Convolutional ops are trainable (default). Low-rank matrix ops are also
    trainable; their parameter count is `2 · p · rank` instead of `p²`. With
    `p=2151, rank=16` this is 68 832 instead of 4 626 801 (98.5 % reduction).
    """
    layers: list[nn.Module] = [
        FrozenConvOperator(_identity_kernel(), name="identity", trainable=False, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 0), name="sg_smooth_w11p2",  trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(21, 3, 0), name="sg_smooth_w21p3",  trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 1), name="sg_d1_w11p2",      trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(21, 3, 1), name="sg_d1_w21p3",      trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_sg_kernel(11, 2, 2), name="sg_d2_w11p2",      trainable=trainable, reg_lambda=reg_lambda),
        LowRankMatrixOperator(_detrend_matrix(p, 1), rank=rank, name="detrend_d1_lr", trainable=trainable, reg_lambda=reg_lambda),
        LowRankMatrixOperator(_detrend_matrix(p, 2), rank=rank, name="detrend_d2_lr", trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_finite_difference_kernel(1), name="fd_o1",    trainable=trainable, reg_lambda=reg_lambda),
        FrozenConvOperator(_norris_williams_kernel(gap=5, smoothing=5, order=1),
                           name="nw_g5_s5", trainable=trainable, reg_lambda=reg_lambda),
        LowRankMatrixOperator(_whittaker_smoother_matrix(p, lam=whittaker_lambda, order=2),
                              rank=rank, name="whittaker_l10_lr", trainable=trainable, reg_lambda=reg_lambda),
    ]
    return layers
