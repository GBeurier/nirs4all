# oom_pls_torch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin


# ----------------------------
# Utils
# ----------------------------

def _odd_round(x: float) -> int:
    w = int(round(x))
    if w % 2 == 0:
        w += 1
    return max(w, 3)

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax (Martins & Astudillo, 2016) projection on simplex.
    Works on last dim by default.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(
        *((1,) * (z.dim() - 1)), -1
    )
    # condition: z_k > (cumsum_k - 1) / k
    tau = (z_cumsum - 1) / k
    support = z_sorted > tau
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    # tau_star = (sum_{i<=k_z} z_i - 1) / k_z
    tau_star = (z_cumsum.gather(dim, k_z - 1) - 1) / k_z
    return torch.clamp(z - tau_star, min=0)

def conv1d_same_corr(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 1, L)
    w: (Cout, 1, K)  (cross-correlation weights like torch conv1d)
    returns: (B, Cout, L) with zero padding ("same")
    """
    k = w.shape[-1]
    pad = (k - 1) // 2
    return F.conv1d(x, w, padding=pad)

def conv1d_same_corr_single(x: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 1, L)
    k1: (K,) correlation kernel -> use weight (1,1,K)
    returns: (B, 1, L)
    """
    w = k1.view(1, 1, -1)
    return conv1d_same_corr(x, w)

def conv1d_full_conv_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Full 1D convolution (not correlation) for two vectors.
    Output length = len(a) + len(b) - 1.
    """
    x = a.view(1, 1, -1)
    # torch.conv1d performs correlation, so flip b to obtain convolution.
    w = torch.flip(b, dims=[0]).view(1, 1, -1)
    return F.conv1d(x, w, padding=b.numel() - 1).view(-1)


# ----------------------------
# Kernel builders (SG, Norris-Williams)
# ----------------------------

def savgol_coeffs_torch(window: int, polyorder: int, deriv: int, delta: float, device, dtype) -> torch.Tensor:
    """
    Compute Savitzky-Golay coefficients for cross-correlation form:
      y[i] = sum_{m=-h..h} coeff[m] * x[i+m]
    Returned tensor shape: (window,)
    """
    assert window % 2 == 1 and window > polyorder
    h = (window - 1) // 2
    x = torch.arange(-h, h + 1, device=device, dtype=torch.float64)  # compute in float64
    # Vandermonde: (window, polyorder+1)
    V = torch.stack([x ** p for p in range(polyorder + 1)], dim=1)  # [W, P+1]
    # pinv(V): [P+1, W]
    V_pinv = torch.linalg.pinv(V)
    # derivative at 0: deriv! * a_deriv ; a = pinv(V) y
    fac = math.factorial(deriv) / (delta ** deriv)
    coeff = fac * V_pinv[deriv]  # [W]
    return coeff.to(dtype=dtype)

def norris_williams_kernel(gap: int, segment: int, deriv: int, delta: float, device, dtype) -> torch.Tensor:
    """
    Build a correlation kernel for Norris-Williams gap derivative with optional segment averaging.
    - segment smoothing: boxcar (odd segment)
    - gap derivative (1st): [-1, 0, ..., +1] / (2*gap*delta)
    - deriv=2: apply gap derivative twice (convolution with itself)
    """
    assert gap >= 1
    assert segment >= 1 and segment % 2 == 1
    assert deriv in (1, 2)

    seg = torch.ones(segment, device=device, dtype=torch.float64) / float(segment)
    gk = torch.zeros(2 * gap + 1, device=device, dtype=torch.float64)
    gk[0] = -1.0 / (2.0 * gap * delta)
    gk[-1] = 1.0 / (2.0 * gap * delta)

    k = conv1d_full_conv_1d(seg, gk)
    if deriv == 2:
        k = conv1d_full_conv_1d(k, gk)
    return k.to(dtype=dtype)


# ----------------------------
# Detrend projector (low-rank, self-adjoint)
# ----------------------------

class DetrendProjector:
    """
    Detrend projection D = I - Q Q^T where Q spans polynomials up to degree.
    Symmetric: D^T = D. Cheap on batches.
    """

    def __init__(self, p: int, degree: int, device, dtype=torch.float32):
        assert degree >= 0
        self.p = p
        self.degree = degree
        t = torch.linspace(-1, 1, p, device=device, dtype=torch.float64)
        V = torch.stack([t ** d for d in range(degree + 1)], dim=1)  # (p, r)
        Q, _ = torch.linalg.qr(V, mode="reduced")  # (p, r), orthonormal cols
        self.Q = Q.to(dtype=dtype)  # store float32
        self.r = degree + 1

    def apply_vec(self, v: torch.Tensor) -> torch.Tensor:
        # v: (p,)
        Q = self.Q
        return v - Q @ (Q.transpose(0, 1) @ v)

    def apply_batch(self, V: torch.Tensor) -> torch.Tensor:
        # V: (B, p)
        Q = self.Q
        return V - (V @ Q) @ Q.transpose(0, 1)


# ----------------------------
# Convolution family bank (Option A: sparse mixture of kernels)
# ----------------------------

class ConvKernelFamily:
    """
    A family of kernels {k_j}. For adjoint scoring we need all k_j^T applied:
      g_j = conv_adj(k_j, c)
    In conv1d terms (cross-correlation forward):
      forward uses k_j as-is,
      adjoint uses flipped kernel.
    """

    def __init__(self, kernels: torch.Tensor):
        """
        kernels: (M, Kmax) padded with zeros to same K for grouped implementation.
        We also store per-kernel true lengths for correct padding if needed.
        Here we assume all are already same K and odd.
        """
        assert kernels.dim() == 2
        self.kernels = kernels  # (M, K)
        self.M, self.K = kernels.shape

    @staticmethod
    def from_list(k_list: List[torch.Tensor], device, dtype) -> "ConvKernelFamily":
        # pad all to same odd length (max)
        maxK = max(int(k.numel()) for k in k_list)
        if maxK % 2 == 0:
            maxK += 1
        padded = []
        for k in k_list:
            k = k.to(device=device, dtype=dtype)
            K = k.numel()
            pad = maxK - K
            if pad > 0:
                left = pad // 2
                right = pad - left
                k = F.pad(k, (left, right), value=0.0)
            padded.append(k)
        kernels = torch.stack(padded, dim=0)  # (M, maxK)
        return ConvKernelFamily(kernels)

    def apply_adj_bank(self, c: torch.Tensor) -> torch.Tensor:
        """
        Apply all adjoint kernels to vector c.
        c: (p,)
        returns G: (M, p)
        """
        # adjoint for cross-correlation is correlation with flipped kernel
        w_adj = torch.flip(self.kernels, dims=[1]).unsqueeze(1)  # (M,1,K)
        x = c.view(1, 1, -1)  # (1,1,p)
        y = conv1d_same_corr(x, w_adj)  # (1,M,p)
        return y.squeeze(0)  # (M,p)

    def apply_fwd_eff(self, v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Apply effective forward kernel k_eff = sum_j alpha_j k_j
        v: (p,)
        alpha: (M,) sparse weights
        returns: (p,)
        """
        k_eff = torch.sum(alpha.view(-1, 1) * self.kernels, dim=0)  # (K,)
        y = conv1d_same_corr_single(v.view(1, 1, -1), k_eff).view(-1)
        return y


# ----------------------------
# Block spec: order of detrend around a family
# ----------------------------

@dataclass(frozen=True)
class BlockSpec:
    family_name: str
    pre_deg: int   # -1 => none
    post_deg: int  # -1 => none

    @property
    def name(self) -> str:
        pre = "none" if self.pre_deg < 0 else f"deg{self.pre_deg}"
        post = "none" if self.post_deg < 0 else f"deg{self.post_deg}"
        return f"{self.family_name}[pre={pre},post={post}]"


# ----------------------------
# OOM-PLS core
# ----------------------------

class OOMPLSRegressorTorch(BaseEstimator, RegressorMixin):
    """
    Operator-Order-Mixture PLS with Option A:
    - Each block = D_post ∘ ConvFamily ∘ D_pre (order-aware)
    - Inside each block, ConvFamily selects a sparse mixture of kernels (alpha) from scores.
    - Across blocks, sparse selection (gamma) per PLS component.

    Notes:
    - This is PLS-like (NIPALS deflation).
    - No per-feature scaling (preserve spectral geometry).
    """
    framework = "sklearn"

    def __init__(
        self,
        n_components: int = 15,
        tau_kernel: float = 0.5,
        tau_block: float = 0.5,
        degrees: Tuple[int, ...] = (0, 1),   # detrend degrees to consider (0=mean,1=linear); add 2 if needed
        families: Tuple[str, ...] = ("SG", "NW"),
        degrees_mode: Optional[int] = None,  # 0->(0,), 1->(0,1), 2->(0,1,2)
        families_mode: Optional[int] = None, # 0->("SG",), 1->("NW",), 2->("SG","NW")
        top_m_kernels: Optional[int] = 8,    # restrict kernel selection inside a family for speed
        top_m_blocks: Optional[int] = 8,     # restrict block selection for speed
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_components = n_components
        self.tau_kernel = tau_kernel
        self.tau_block = tau_block
        self.degrees_mode = degrees_mode
        self.families_mode = families_mode
        if degrees_mode is not None:
            degrees_map = {
                0: (0,),
                1: (0, 1),
                2: (0, 1, 2),
            }
            if degrees_mode not in degrees_map:
                raise ValueError(f"degrees_mode must be in {sorted(degrees_map)}, got {degrees_mode}")
            self.degrees = degrees_map[degrees_mode]
        else:
            self.degrees = tuple(degrees)

        if families_mode is not None:
            families_map = {
                0: ("SG",),
                1: ("NW",),
                2: ("SG", "NW"),
            }
            if families_mode not in families_map:
                raise ValueError(f"families_mode must be in {sorted(families_map)}, got {families_mode}")
            self.families = families_map[families_mode]
        else:
            self.families = tuple(families)

        self.top_m_kernels = top_m_kernels
        self.top_m_blocks = top_m_blocks
        # Keep a plain string for pipeline serialization compatibility.
        # `torch.device` instances trigger serializer introspection errors.
        device_str = str(device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device_str
        self.dtype = dtype

        # learned after fit
        self.x_mean_ = None
        self.y_mean_ = None
        self.W_ = None
        self.P_ = None
        self.Q_ = None
        self.coef_ = None
        self.gamma_ = None              # (K,B)
        self.block_alphas_ = None       # list per component: dict(block_idx -> alpha tensor)
        self.block_names_ = None
        self.blocks_ = None

    def _build_families(
        self,
        p: int,
        wavelengths_nm: Optional[torch.Tensor],
    ) -> Dict[str, ConvKernelFamily]:
        """
        Build kernel families. Windows are defined in nm-widths if wavelengths are provided,
        otherwise in points (fallback).
        """
        device, dtype = self.device, self.dtype

        # delta estimate
        if wavelengths_nm is not None and wavelengths_nm.numel() == p:
            wl = wavelengths_nm.to(device=device, dtype=torch.float64)
            delta = torch.median(wl[1:] - wl[:-1]).item()
        else:
            # fallback: assume unit spacing
            delta = 1.0

        fams: Dict[str, ConvKernelFamily] = {}

        if "SG" in self.families:
            # Define widths in nm that are "scale-stable" across varying p
            # You can tune this ladder; keep it compact.
            widths_nm = [15, 25, 45, 75, 120]  # interpretable scales
            # Convert to odd window lengths in points
            windows = [_odd_round(w / delta) for w in widths_nm]
            # Guard: windows must be > polyorder and <= p
            windows = [w for w in windows if w <= p and w >= 5]

            configs = []
            # smoothing + deriv1 + deriv2
            for w in windows:
                configs.append((w, 2, 0))
                configs.append((w, 2, 1))
                # deriv2 requires polyorder>=2; often better polyorder=3
                configs.append((w, 3, 2))

            k_list = [savgol_coeffs_torch(w, po, d, delta=delta, device=device, dtype=dtype) for (w, po, d) in configs]
            fams["SG"] = ConvKernelFamily.from_list(k_list, device=device, dtype=dtype)

        if "NW" in self.families:
            # gaps/segments in points; convert from nm-ish if wl is known
            # We'll pick gaps roughly matching SG scales.
            if wavelengths_nm is not None and wavelengths_nm.numel() == p:
                gap_nm = [10, 20, 40, 80]  # in nm
                gaps = [max(1, int(round(g / delta))) for g in gap_nm]
            else:
                gaps = [3, 5, 11, 21]

            segments = [1, 5, 9]  # odd
            k_list = []
            for g in gaps:
                for s in segments:
                    k_list.append(norris_williams_kernel(gap=g, segment=s, deriv=1, delta=delta, device=device, dtype=dtype))
                    k_list.append(norris_williams_kernel(gap=g, segment=s, deriv=2, delta=delta, device=device, dtype=dtype))
            fams["NW"] = ConvKernelFamily.from_list(k_list, device=device, dtype=dtype)

        return fams

    def _build_blocks(self) -> List[BlockSpec]:
        blocks: List[BlockSpec] = []
        # detrend degrees: include "none" as -1
        degs = [-1] + list(self.degrees)
        for fam in self.families:
            for pre in degs:
                for post in degs:
                    # keep all four modes (none/pre/post/both). You can prune if needed.
                    blocks.append(BlockSpec(family_name=fam, pre_deg=pre, post_deg=post))
        return blocks

    @staticmethod
    def _norm_sq(v: torch.Tensor) -> torch.Tensor:
        return torch.sum(v * v)

    def fit(self, X: Any, y: Any, wavelengths_nm: Optional[Any] = None) -> "OOMPLSRegressorTorch":
        """
        X: (n,p) torch tensor (cpu or cuda)
        y: (n,) or (n,1)
        wavelengths_nm: optional (p,) to build scale-stable kernels
        """
        device, dtype = self.device, self.dtype
        if not torch.is_tensor(X):
            X = torch.as_tensor(np.asarray(X), device=device, dtype=dtype)
        else:
            X = X.to(device=device, dtype=dtype)

        if not torch.is_tensor(y):
            y = torch.as_tensor(np.asarray(y), device=device, dtype=dtype)
        else:
            y = y.to(device=device, dtype=dtype)

        if y.dim() == 1:
            y = y.view(-1, 1)

        n, p = X.shape
        self.x_mean_ = X.mean(dim=0, keepdim=True)
        self.y_mean_ = y.mean(dim=0, keepdim=True)
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        # build families and blocks
        if wavelengths_nm is not None and not torch.is_tensor(wavelengths_nm):
            wavelengths_nm = torch.as_tensor(np.asarray(wavelengths_nm), device=device, dtype=torch.float32)
        elif wavelengths_nm is not None:
            wavelengths_nm = wavelengths_nm.to(device=device, dtype=torch.float32)

        self.families_ = self._build_families(p, wavelengths_nm)
        self.blocks_ = self._build_blocks()
        self.block_names_ = [b.name for b in self.blocks_]
        B = len(self.blocks_)

        # detrend projectors (cache)
        detrenders: Dict[int, DetrendProjector] = {}
        for deg in self.degrees:
            detrenders[deg] = DetrendProjector(p=p, degree=deg, device=device, dtype=dtype)

        # allocate
        Kmax = min(self.n_components, n - 1, p)
        W = torch.zeros((p, Kmax), device=device, dtype=dtype)
        P = torch.zeros((p, Kmax), device=device, dtype=dtype)
        Q = torch.zeros((1, Kmax), device=device, dtype=dtype)
        Gamma = torch.zeros((Kmax, B), device=device, dtype=dtype)
        block_alphas: List[Dict[int, torch.Tensor]] = []

        Xres = Xc
        yres = yc

        eps = 1e-12

        for k in range(Kmax):
            # cross-cov vector
            c = (Xres.transpose(0, 1) @ yres).view(-1)  # (p,)

            # Per-block g and alpha
            g_blocks = torch.zeros((B, p), device=device, dtype=dtype)
            s_blocks = torch.zeros((B,), device=device, dtype=dtype)
            alphas_k: Dict[int, torch.Tensor] = {}

            # (speed) precompute per family and post-degree: G0 = C^T(D_post(c))
            # We do it naively per block here for readability; optimize later.
            for b_idx, bs in enumerate(self.blocks_):
                fam = self.families_[bs.family_name]

                # apply post detrend to input c (because A^T = D_pre ∘ C^T ∘ D_post)
                c1 = c
                if bs.post_deg >= 0:
                    c1 = detrenders[bs.post_deg].apply_vec(c1)

                # bank adjoint
                G0 = fam.apply_adj_bank(c1)  # (M,p)

                # apply pre detrend to each channel if needed
                if bs.pre_deg >= 0:
                    G0 = detrenders[bs.pre_deg].apply_batch(G0)

                # kernel scoring
                s_kern = torch.sum(G0 * G0, dim=1)  # (M,)

                # optional top-m to stabilize / speed
                if self.top_m_kernels is not None and self.top_m_kernels < s_kern.numel():
                    vals, idxs = torch.topk(s_kern, k=self.top_m_kernels, largest=True)
                    z = vals / (self.tau_kernel * (vals.max() + eps))
                    a_small = sparsemax(z, dim=0)
                    alpha = torch.zeros_like(s_kern)
                    alpha[idxs] = a_small
                else:
                    z = s_kern / (self.tau_kernel * (s_kern.max() + eps))
                    alpha = sparsemax(z, dim=0)

                # combine g = sum_j alpha_j G0_j
                g = torch.sum(alpha.view(-1, 1) * G0, dim=0)  # (p,)

                # block score (normalize to reduce “amplify wins” bias)
                # cheap proxy for ||A||_F^2: kernel energy * p (rough), detrend reduces rank.
                # For now: normalize by kernel energy only (works well enough with sparse gating).
                k_energy = torch.sum(torch.sum(fam.kernels * fam.kernels, dim=1) * alpha)
                nu = (k_energy + eps)
                s_block = torch.sum(g * g) / nu

                g_blocks[b_idx] = g
                s_blocks[b_idx] = s_block
                alphas_k[b_idx] = alpha.detach()  # store for reporting

            # block gating gamma
            if self.top_m_blocks is not None and self.top_m_blocks < B:
                vals, idxs = torch.topk(s_blocks, k=self.top_m_blocks, largest=True)
                z = vals / (self.tau_block * (vals.max() + eps))
                g_small = sparsemax(z, dim=0)
                gamma = torch.zeros_like(s_blocks)
                gamma[idxs] = g_small
            else:
                z = s_blocks / (self.tau_block * (s_blocks.max() + eps))
                gamma = sparsemax(z, dim=0)

            Gamma[k] = gamma
            block_alphas.append(alphas_k)

            # build w from blocks
            w_mix = torch.zeros((p,), device=device, dtype=dtype)
            for b_idx, bs in enumerate(self.blocks_):
                gb = g_blocks[b_idx]
                gb_norm = torch.sqrt(torch.sum(gb * gb) + eps)
                if gb_norm < 1e-10 or gamma[b_idx] <= 0:
                    continue
                w_hat = gb / gb_norm  # in operator space

                fam = self.families_[bs.family_name]
                alpha = alphas_k[b_idx].to(device=device, dtype=dtype)

                # forward: A = D_post ∘ C ∘ D_pre
                v = w_hat
                if bs.pre_deg >= 0:
                    v = detrenders[bs.pre_deg].apply_vec(v)
                v = fam.apply_fwd_eff(v, alpha)
                if bs.post_deg >= 0:
                    v = detrenders[bs.post_deg].apply_vec(v)

                v_norm = torch.sqrt(torch.sum(v * v) + eps)
                if v_norm < 1e-10:
                    continue
                w_b = v / v_norm
                w_mix = w_mix + gamma[b_idx] * w_b

            w_norm = torch.sqrt(torch.sum(w_mix * w_mix) + eps)
            if w_norm < 1e-10:
                # fail-safe: stop extracting components
                Kmax = k
                W = W[:, :Kmax]
                P = P[:, :Kmax]
                Q = Q[:, :Kmax]
                Gamma = Gamma[:Kmax]
                block_alphas = block_alphas[:Kmax]
                break

            w = w_mix / w_norm

            # scores and deflation (NIPALS-style)
            t = Xres @ w  # (n,)
            tt = torch.sum(t * t) + eps
            p_load = (Xres.transpose(0, 1) @ t) / tt  # (p,)
            q_load = (yres.transpose(0, 1) @ t) / tt  # (1,)

            # store
            W[:, k] = w
            P[:, k] = p_load
            Q[:, k] = q_load

            # deflate
            Xres = Xres - t.view(-1, 1) @ p_load.view(1, -1)
            yres = yres - t.view(-1, 1) @ q_load.view(1, -1)

        # final regression coef: B = W (P^T W)^-1 Q^T  (scalar y)
        K = W.shape[1]
        if K > 0:
            PtW = P.transpose(0, 1) @ W  # (K,K)
            R = torch.linalg.pinv(PtW)   # stable
            coef = W @ (R @ Q.transpose(0, 1))  # (p,1)
        else:
            coef = torch.zeros((p, 1), device=device, dtype=dtype)

        self.W_ = W
        self.P_ = P
        self.Q_ = Q
        self.coef_ = coef
        self.gamma_ = Gamma
        self.block_alphas_ = block_alphas
        return self

    @torch.no_grad()
    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None or self.y_mean_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        if not torch.is_tensor(X):
            X = torch.as_tensor(np.asarray(X), device=self.device, dtype=self.dtype)
        else:
            X = X.to(device=self.device, dtype=self.dtype)

        Xc = X - self.x_mean_
        yhat = Xc @ self.coef_ + self.y_mean_
        return yhat.view(-1).detach().cpu().numpy()

    def preprocessing_report(self, topk_blocks: int = 5, topk_kernels: int = 5) -> List[dict]:
        """
        Human-readable report:
        - top blocks per component (gamma)
        - inside each block, top kernels (alpha)
        """
        report = []
        K, B = self.gamma_.shape
        for k in range(K):
            gamma = self.gamma_[k].detach().cpu()
            vals, idxs = torch.topk(gamma, k=min(topk_blocks, B))
            blocks = []
            for v, b_idx in zip(vals.tolist(), idxs.tolist()):
                if v <= 0:
                    continue
                name = self.block_names_[b_idx]
                alpha = self.block_alphas_[k][b_idx].cpu()
                avals, aidxs = torch.topk(alpha, k=min(topk_kernels, alpha.numel()))
                kernels = [{"idx": int(i), "w": float(a)} for a, i in zip(avals.tolist(), aidxs.tolist()) if a > 0]
                blocks.append({"block": name, "gamma": float(v), "kernels": kernels})
            report.append({"component": k + 1, "blocks": blocks})
        return report
