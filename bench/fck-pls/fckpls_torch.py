"""
FCK-PLS Torch: End-to-end learnable Fractional Convolutional Kernel PLS
========================================================================

Prototype implementation based on discussion with ChatGPT.

Two versions:
- V1: Learnable free kernels (recommended for stability)
- V2: Learnable alpha/sigma fractional parametric kernels (more interpretable)

The PLS head is NOT learned as a dense layer but recalculated via closed-form
differentiable operations (SVD/solve). This preserves PLS structure while allowing
backprop to the convolutional front-end.

Usage:
    est = FCKPLSTorch(version="v1", n_kernels=16, n_components=10)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# Utilities
# =============================================================================

def _to_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_torch(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _standardize_fit(x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd, mu, sd


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


# =============================================================================
# Feature Extractors (Convolutional Front-ends)
# =============================================================================

class LearnableKernelBank(nn.Module):
    """
    Version 1: Free learnable 1D convolution kernels.

    The kernels are fully learnable parameters. Regularization priors
    (smoothness, zero-mean) encourage "filter-like" behavior.
    """

    def __init__(
        self,
        n_kernels: int = 16,
        kernel_size: int = 31,
        init_mode: Literal["random", "derivative", "fractional"] = "random",
    ):
        super().__init__()
        self.n_kernels = int(n_kernels)
        self.kernel_size = int(kernel_size)
        self.init_mode = init_mode

        # Initialize kernels
        k = self._init_kernels()
        self.k = nn.Parameter(k)

    def _init_kernels(self) -> torch.Tensor:
        """Initialize kernels based on mode."""
        k = torch.randn(self.n_kernels, 1, self.kernel_size) * 0.01

        if self.init_mode == "derivative":
            # Initialize some kernels as derivative-like
            m = self.kernel_size // 2
            idx = torch.arange(-m, m + 1, dtype=torch.float32)

            for i in range(min(self.n_kernels, 5)):
                order = i * 0.5  # 0, 0.5, 1, 1.5, 2
                if order < 0.1:
                    # Smoothing kernel (Gaussian-like)
                    sigma = 3.0
                    kernel = torch.exp(-0.5 * (idx / sigma) ** 2)
                else:
                    # Derivative-like
                    sigma = 3.0
                    gaussian = torch.exp(-0.5 * (idx / sigma) ** 2)
                    frac = torch.sign(idx) * torch.abs(idx) ** order
                    kernel = gaussian * frac
                    kernel = kernel - kernel.mean()

                kernel = kernel / (kernel.abs().sum() + 1e-8)
                k[i, 0, :] = kernel

        elif self.init_mode == "fractional":
            # Initialize with spread of fractional orders
            m = self.kernel_size // 2
            idx = torch.arange(-m, m + 1, dtype=torch.float32)
            alphas = torch.linspace(0, 2, self.n_kernels)

            for i, alpha in enumerate(alphas):
                sigma = 3.0
                gaussian = torch.exp(-0.5 * (idx / sigma) ** 2)
                if alpha < 0.1:
                    kernel = gaussian
                else:
                    frac = torch.sign(idx) * (torch.abs(idx) + 1e-8) ** alpha
                    kernel = gaussian * frac
                    kernel = kernel - kernel.mean()
                kernel = kernel / (kernel.abs().sum() + 1e-8)
                k[i, 0, :] = kernel

        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel bank convolution.

        Args:
            x: (B, L) input spectra

        Returns:
            z: (B, K, L) convolved features
        """
        x1 = x[:, None, :]  # (B, 1, L)
        pad = self.kernel_size // 2
        z = F.conv1d(x1, self.k, padding=pad)  # (B, K, L)
        return z

    def kernel_regularization(
        self,
        smooth_w: float = 1e-3,
        zeromean_w: float = 1e-3,
        l2_w: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute regularization loss on kernels.

        - smooth: penalize second differences (encourage smooth kernels)
        - zeromean: penalize mean (encourage derivative-like behavior)
        - l2: simple weight decay
        """
        k = self.k.squeeze(1)  # (K, ks)

        # Zero-mean prior
        zeromean = (k.mean(dim=1) ** 2).mean()

        # Smoothness via second differences
        if k.shape[1] >= 3:
            d2 = k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]
            smooth = (d2 ** 2).mean()
        else:
            smooth = torch.zeros((), device=k.device, dtype=k.dtype)

        # L2 regularization
        l2 = (k ** 2).mean()

        return smooth_w * smooth + zeromean_w * zeromean + l2_w * l2

    def get_kernels(self) -> np.ndarray:
        """Get kernels as numpy array."""
        return self.k.squeeze(1).detach().cpu().numpy()


class FractionalKernelBank(nn.Module):
    """
    Version 2: Learn alpha/sigma and rebuild kernels parametrically.

    More interpretable but potentially less stable optimization.
    Uses smooth approximations for non-differentiable operations.
    """

    def __init__(
        self,
        n_kernels: int = 16,
        kernel_size: int = 31,
        alpha_max: float = 2.0,
        alpha_init: Optional[Sequence[float]] = None,
        sigma_init: Optional[Sequence[float]] = None,
        eps: float = 1e-8,
        tau: float = 1.0,  # smoothness for sign approximation
    ):
        super().__init__()
        self.n_kernels = int(n_kernels)
        self.kernel_size = int(kernel_size)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)
        self.tau = float(tau)

        # Raw parameters (transformed to constrained space in forward)
        if alpha_init is not None:
            # Convert target alphas to raw values (inverse sigmoid)
            alphas = torch.tensor(alpha_init, dtype=torch.float32)
            alpha_raw = torch.log(alphas / (self.alpha_max - alphas + self.eps) + self.eps)
        else:
            # Initialize to spread across [0, alpha_max]
            alpha_raw = torch.zeros(self.n_kernels)

        if sigma_init is not None:
            # Inverse softplus
            sigmas = torch.tensor(sigma_init, dtype=torch.float32)
            sigma_raw = torch.log(torch.exp(sigmas) - 1 + self.eps)
        else:
            sigma_raw = torch.zeros(self.n_kernels)  # softplus(0) ≈ 0.693

        self.alpha_raw = nn.Parameter(alpha_raw)
        self.sigma_raw = nn.Parameter(sigma_raw)

        # Index buffer
        m = self.kernel_size // 2
        idx = torch.arange(-m, m + 1, dtype=torch.float32)
        self.register_buffer("idx", idx)

    def build_kernels(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Reconstruct kernels from learned alpha/sigma."""
        idx = self.idx.to(device=device, dtype=dtype)[None, :]  # (1, ks)

        # Constrain parameters
        alpha = (self.alpha_max * torch.sigmoid(self.alpha_raw)).to(device=device, dtype=dtype)[:, None]  # (K, 1)
        sigma = (F.softplus(self.sigma_raw) + self.eps).to(device=device, dtype=dtype)[:, None]  # (K, 1)

        # Gaussian envelope
        gaussian = torch.exp(-(idx ** 2) / (2.0 * sigma ** 2))  # (K, ks)

        # Smooth sign approximation
        smooth_sign = torch.tanh(idx / self.tau)  # (1, ks)

        # Fractional power: |idx|^alpha with smooth handling of zero
        abs_pow = torch.exp(alpha * torch.log(torch.abs(idx) + self.eps))  # (K, ks)

        # Combine
        k = gaussian * smooth_sign * abs_pow  # (K, ks)

        # Zero-mean (derivative-like)
        k = k - k.mean(dim=1, keepdim=True)

        # L1 normalization
        k = k / (k.abs().sum(dim=1, keepdim=True) + self.eps)

        return k[:, None, :]  # (K, 1, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fractional kernel bank convolution."""
        x1 = x[:, None, :]  # (B, 1, L)
        k = self.build_kernels(dtype=x.dtype, device=x.device)
        pad = self.kernel_size // 2
        z = F.conv1d(x1, k, padding=pad)  # (B, K, L)
        return z

    def kernel_regularization(
        self,
        alpha_w: float = 1e-4,
        sigma_w: float = 1e-4,
    ) -> torch.Tensor:
        """Regularization to keep parameters in reasonable ranges."""
        alpha = self.alpha_max * torch.sigmoid(self.alpha_raw)
        sigma = F.softplus(self.sigma_raw)
        # Mild L2 on parameters
        return alpha_w * (alpha ** 2).mean() + sigma_w * (sigma ** 2).mean()

    def get_alphas(self) -> np.ndarray:
        """Get current alpha values."""
        with torch.no_grad():
            return (self.alpha_max * torch.sigmoid(self.alpha_raw)).cpu().numpy()

    def get_sigmas(self) -> np.ndarray:
        """Get current sigma values."""
        with torch.no_grad():
            return (F.softplus(self.sigma_raw) + self.eps).cpu().numpy()

    def get_kernels(self) -> np.ndarray:
        """Get kernels as numpy array."""
        with torch.no_grad():
            k = self.build_kernels(torch.float32, self.idx.device)
            return k.squeeze(1).cpu().numpy()


# =============================================================================
# PLS Solved Head (Differentiable)
# =============================================================================

def ridge_closed_form(
    T: torch.Tensor,
    Y: torch.Tensor,
    lam: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Closed-form ridge regression: B = (T'T + λI)^{-1} T'Y
    """
    r = T.shape[1]
    A = T.T @ T + lam * torch.eye(r, device=T.device, dtype=T.dtype)
    B = T.T @ Y
    return torch.linalg.solve(A, B)  # (r, m)


class PLSSolvedHead(nn.Module):
    """
    Solved PLS head: computes directions via closed-form operations.

    Two modes:
    - "svd": PLS2-like via SVD of cross-covariance (recommended for multi-target)
    - "deflation": Iterative deflation (classic PLS1 style)

    The head is NOT learned but computed from data. Gradients flow through
    the SVD/solve operations back to the feature extractor.
    """

    def __init__(
        self,
        n_components: int = 10,
        ridge_lambda: float = 1e-3,
        mode: Literal["svd", "deflation"] = "svd",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_components = int(n_components)
        self.ridge_lambda = float(ridge_lambda)
        self.mode = str(mode)
        self.eps = float(eps)

    def forward(
        self,
        Zf: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PLS projection and prediction.

        Args:
            Zf: (B, D) flattened features
            Y: (B, m) targets

        Returns:
            Yhat: (B, m) predictions
            aux: dict with W, B, means for caching
        """
        # Center
        z_mean = Zf.mean(dim=0, keepdim=True)
        y_mean = Y.mean(dim=0, keepdim=True)
        Zc = Zf - z_mean
        Yc = Y - y_mean

        if self.mode == "svd":
            # Cross-covariance SVD
            C = Zc.T @ Yc  # (D, m)
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            r = min(self.n_components, U.shape[1])
            W = U[:, :r]  # (D, r)

        elif self.mode == "deflation":
            # Deflation-based for PLS1-ish
            W_list = []
            Zr = Zc
            Yr = Yc
            r = min(self.n_components, Zc.shape[1], max(1, Zc.shape[0] - 1))

            for _ in range(r):
                C = Zr.T @ Yr  # (D, m)
                c = C.sum(dim=1)  # (D,) - sum across targets
                norm_c = torch.norm(c)
                if norm_c < self.eps:
                    # Degenerate case: stop early
                    break
                w = c / (norm_c + self.eps)
                t = Zr @ w[:, None]  # (B, 1)
                # Deflation (X only)
                p = (Zr.T @ t) / (t.T @ t + self.eps)  # (D, 1)
                Zr = Zr - t @ p.T
                W_list.append(w)

            if len(W_list) == 0:
                # Fallback: use identity-like projection
                W = torch.eye(Zc.shape[1], min(self.n_components, Zc.shape[1]),
                             device=Zc.device, dtype=Zc.dtype)
            else:
                W = torch.stack(W_list, dim=1)  # (D, r)
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        # Project and regress
        T = Zc @ W  # (B, r)
        B = ridge_closed_form(T, Yc, lam=self.ridge_lambda, eps=self.eps)  # (r, m)
        Yhat = T @ B + y_mean

        aux = {
            "W": W,
            "B": B,
            "z_mean": z_mean,
            "y_mean": y_mean,
        }
        return Yhat, aux


# =============================================================================
# Full Model
# =============================================================================

class FCKPLSTorchModel(nn.Module):
    """Combined feature extractor + PLS head."""

    def __init__(
        self,
        extractor: nn.Module,
        n_components: int = 10,
        ridge_lambda: float = 1e-3,
        pls_mode: str = "deflation",  # deflation works for single-target, svd only for multi-target
        feature_mode: Literal["interleaved", "concatenated"] = "interleaved",
    ):
        super().__init__()
        self.extractor = extractor
        self.feature_mode = feature_mode
        self.head = PLSSolvedHead(
            n_components=n_components,
            ridge_lambda=ridge_lambda,
            mode=pls_mode,
        )

    def flatten_features(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Flatten conv features according to feature_mode.

        Args:
            Z: (B, K, L) convolved features - K kernels, L wavelengths

        Returns:
            Zf: (B, K*L) flattened features

        Modes:
            - "interleaved": wavelength-major order [k0_w0, k1_w0, ..., kK-1_w0, k0_w1, ...]
              Groups all kernel responses at each wavelength position together.
              Better for PLS as it preserves local spectral structure.
            - "concatenated": kernel-major order [k0_w0, k0_w1, ..., k0_wL-1, k1_w0, ...]
              All responses from kernel 0, then kernel 1, etc.
        """
        if self.feature_mode == "interleaved":
            # (B, K, L) -> (B, L, K) -> (B, L*K)
            return Z.permute(0, 2, 1).flatten(1)
        else:  # concatenated
            # (B, K, L) -> (B, K*L)
            return Z.flatten(1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Full forward pass.

        Returns:
            Yhat: predictions
            aux: head cache for inference
            Z: raw conv features (for analysis)
        """
        Z = self.extractor(X)  # (B, K, L)
        Zf = self.flatten_features(Z)  # (B, K*L) or (B, L*K) depending on mode
        Yhat, aux = self.head(Zf, Y)
        return Yhat, aux, Z


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration for FCK-PLS Torch."""
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: Optional[int] = None  # None => full-batch (recommended)
    early_stopping_patience: int = 40

    # Validation-based training (critical for proper learning!)
    # The PLS head is fit on (1 - val_fraction) of the data,
    # and the kernel loss is computed on val_fraction of the data.
    # This prevents the PLS head from "seeing" the loss samples.
    val_fraction: float = 0.3  # Fraction of training data for kernel validation

    # Regularization
    reg_smooth: float = 1e-3  # kernel smoothness
    reg_zeromean: float = 1e-3  # zero-mean prior
    reg_l2: float = 0.0  # weight decay on kernels
    reg_scale: float = 1.0  # global scale for all regularizations

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: Optional[int] = 42

    # Logging
    verbose: int = 0
    log_every: int = 50


# =============================================================================
# sklearn-compatible Estimator
# =============================================================================

class FCKPLSTorch(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    sklearn-compatible FCK-PLS with PyTorch backend.

    Trains the convolutional front-end via backprop, with a solved PLS head.

    Parameters
    ----------
    version : str, default="v1"
        "v1" for learnable kernels (stable), "v2" for alpha/sigma parametric (interpretable)
    n_kernels : int, default=16
        Number of convolution kernels
    kernel_size : int, default=31
        Size of each kernel (odd)
    n_components : int, default=10
        Number of PLS components
    ridge_lambda : float, default=1e-3
        Ridge regularization for PLS regression
    pls_mode : str, default="deflation"
        "deflation" (works for single-target) or "svd" (multi-target only)
    feature_mode : str, default="interleaved"
        How to arrange flattened features:
        - "interleaved": wavelength-major order, groups kernel responses per wavelength
        - "concatenated": kernel-major order, all of kernel 0 then kernel 1, etc.
    init_mode : str, default="random"
        Kernel initialization: "random", "derivative", "fractional" (v1 only)
    alpha_max : float, default=2.0
        Maximum fractional order (v2 only)
    tau : float, default=1.0
        Smoothness for sign approximation (v2 only)
    train_cfg : TrainConfig, optional
        Training configuration

    Attributes
    ----------
    model_ : FCKPLSTorchModel
        Trained PyTorch model
    head_cache_ : dict
        Cached PLS parameters for inference
    """

    def __init__(
        self,
        version: str = "v1",
        n_kernels: int = 16,
        kernel_size: int = 31,
        n_components: int = 10,
        ridge_lambda: float = 1e-3,
        pls_mode: str = "deflation",  # deflation for single-target, svd for multi-target
        feature_mode: Literal["interleaved", "concatenated"] = "interleaved",
        # v1 params
        init_mode: str = "random",
        # v2 params
        alpha_max: float = 2.0,
        tau: float = 1.0,
        alpha_init: Optional[Sequence[float]] = None,
        sigma_init: Optional[Sequence[float]] = None,
        # training
        train_cfg: Optional[TrainConfig] = None,
    ):
        self.version = version
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.n_components = n_components
        self.ridge_lambda = ridge_lambda
        self.pls_mode = pls_mode
        self.feature_mode = feature_mode
        self.init_mode = init_mode
        self.alpha_max = alpha_max
        self.tau = tau
        self.alpha_init = alpha_init
        self.sigma_init = sigma_init
        self.train_cfg = train_cfg if train_cfg is not None else TrainConfig()

        # Fitted attributes
        self.model_: Optional[FCKPLSTorchModel] = None
        self.head_cache_: Optional[Dict[str, torch.Tensor]] = None
        self.x_mu_: Optional[np.ndarray] = None
        self.x_sd_: Optional[np.ndarray] = None
        self.y_mu_: Optional[np.ndarray] = None
        self.y_sd_: Optional[np.ndarray] = None
        self.n_targets_: Optional[int] = None
        self._y_1d: bool = False
        self.training_history_: List[Dict[str, float]] = []

    def _flatten_features(self, Z: torch.Tensor) -> torch.Tensor:
        """Flatten features according to feature_mode (for use during training)."""
        if self.feature_mode == "interleaved":
            return Z.permute(0, 2, 1).flatten(1)  # (B, K, L) -> (B, L, K) -> (B, L*K)
        else:
            return Z.flatten(1)  # (B, K, L) -> (B, K*L)

    def _build_extractor(self) -> nn.Module:
        if self.version == "v1":
            return LearnableKernelBank(
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                init_mode=self.init_mode,
            )
        elif self.version == "v2":
            return FractionalKernelBank(
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                alpha_max=self.alpha_max,
                alpha_init=self.alpha_init,
                sigma_init=self.sigma_init,
                tau=self.tau,
            )
        else:
            raise ValueError(f"Unknown version={self.version}")

    def _compute_reg(self, extractor: nn.Module) -> torch.Tensor:
        """Compute regularization loss."""
        cfg = self.train_cfg
        if self.version == "v1":
            return extractor.kernel_regularization(
                smooth_w=cfg.reg_smooth,
                zeromean_w=cfg.reg_zeromean,
                l2_w=cfg.reg_l2,
            )
        elif self.version == "v2":
            return extractor.kernel_regularization()
        return torch.zeros(())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FCKPLSTorch":
        """
        Fit the FCK-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : FCKPLSTorch
        """
        cfg = self.train_cfg
        _set_seed(cfg.seed)

        # Prepare data
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._y_1d = y.ndim == 1
        y = _to_2d(y)
        self.n_targets_ = y.shape[1]

        # Standardize
        Xs, self.x_mu_, self.x_sd_ = _standardize_fit(X)
        ys, self.y_mu_, self.y_sd_ = _standardize_fit(y)

        # Split into PLS-fitting (train) and kernel-learning (val) subsets
        # This is critical: PLS head is fit on train, loss computed on val
        # Prevents the solved head from "seeing" the loss samples
        n_samples = Xs.shape[0]
        n_val = max(1, int(n_samples * cfg.val_fraction))
        n_train = n_samples - n_val

        # Shuffle indices for split
        rng = np.random.RandomState(cfg.seed)
        indices = rng.permutation(n_samples)
        idx_train, idx_val = indices[:n_train], indices[n_train:]

        # Move to device
        device = torch.device(cfg.device)
        Xt_all = _as_torch(Xs, device)
        yt_all = _as_torch(ys, device)

        Xt_train = Xt_all[idx_train]
        yt_train = yt_all[idx_train]
        Xt_val = Xt_all[idx_val]
        yt_val = yt_all[idx_val]

        # Build extractor only (not full model, as we need custom forward)
        extractor = self._build_extractor().to(device)
        head = PLSSolvedHead(
            n_components=self.n_components,
            ridge_lambda=self.ridge_lambda,
            mode=self.pls_mode,
        ).to(device)

        # Optimizer (only extractor params are learnable)
        opt = torch.optim.AdamW(
            extractor.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Training loop
        best_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience = 0
        self.training_history_ = []

        for epoch in range(cfg.epochs):
            extractor.train()
            opt.zero_grad(set_to_none=True)

            # Forward on TRAIN data - fit PLS head
            Z_train = self._flatten_features(extractor(Xt_train))
            yhat_train, aux = head(Z_train, yt_train)

            # Forward on VAL data using TRAIN's PLS solution
            Z_val = self._flatten_features(extractor(Xt_val))
            z_mean = aux['z_mean']
            y_mean = aux['y_mean']
            W = aux['W']
            B = aux['B']

            T_val = (Z_val - z_mean) @ W
            yhat_val = T_val @ B + y_mean

            # Loss on VALIDATION (kernels learn to generalize)
            mse = F.mse_loss(yhat_val, yt_val)
            reg = self._compute_reg(extractor) * cfg.reg_scale
            loss = mse + reg

            loss.backward()
            opt.step()

            loss_val = loss.detach().item()
            mse_val = mse.detach().item()

            self.training_history_.append({
                "epoch": epoch,
                "loss": loss_val,
                "mse": mse_val,
                "reg": reg.detach().item(),
            })

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                print(f"[epoch {epoch:04d}] loss={loss_val:.6f} mse={mse_val:.6f}")

            # Early stopping
            if loss_val < best_loss - 1e-8:
                best_loss = loss_val
                best_state = {k: v.detach().clone() for k, v in extractor.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    if cfg.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best state
        if best_state is not None:
            extractor.load_state_dict(best_state)

        # Build full model with trained extractor
        extractor.eval()
        model = FCKPLSTorchModel(
            extractor=extractor,
            n_components=self.n_components,
            ridge_lambda=self.ridge_lambda,
            pls_mode=self.pls_mode,
            feature_mode=self.feature_mode,
        ).to(device)

        self.model_ = model.eval()

        # Cache head parameters using ALL training data for final inference
        self._finalize_head_cache(Xt_all, yt_all)

        return self

    def _finalize_head_cache(self, Xt: torch.Tensor, yt: torch.Tensor) -> None:
        """Cache solved head parameters for inference."""
        with torch.no_grad():
            model = self.model_
            Z = model.extractor(Xt)
            Zf = model.flatten_features(Z)
            _, aux = model.head(Zf, yt)

            self.head_cache_ = {
                "W": aux["W"].detach(),
                "B": aux["B"].detach(),
                "z_mean": aux["z_mean"].detach(),
                "y_mean": aux["y_mean"].detach(),
            }

    def _check_fitted(self):
        if self.model_ is None or self.head_cache_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray
            Predictions
        """
        self._check_fitted()

        X = np.asarray(X, dtype=np.float32)
        Xs = _standardize_apply(X, self.x_mu_, self.x_sd_)

        device = next(self.model_.parameters()).device
        Xt = _as_torch(Xs, device)

        cache = self.head_cache_
        Z = self.model_.extractor(Xt)
        Zf = self.model_.flatten_features(Z)

        Zc = Zf - cache["z_mean"]
        T = Zc @ cache["W"]
        Yhat = T @ cache["B"] + cache["y_mean"]

        yhat = Yhat.cpu().numpy()

        # De-standardize
        yhat = yhat * self.y_sd_ + self.y_mu_

        if self._y_1d:
            yhat = yhat.ravel()

        return yhat

    @torch.no_grad()
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to PLS score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform

        Returns
        -------
        T : ndarray of shape (n_samples, n_components)
            PLS scores
        """
        self._check_fitted()

        X = np.asarray(X, dtype=np.float32)
        Xs = _standardize_apply(X, self.x_mu_, self.x_sd_)

        device = next(self.model_.parameters()).device
        Xt = _as_torch(Xs, device)

        cache = self.head_cache_
        Z = self.model_.extractor(Xt)
        Zf = self.model_.flatten_features(Z)

        Zc = Zf - cache["z_mean"]
        T = Zc @ cache["W"]

        return T.cpu().numpy()

    @torch.no_grad()
    def get_features(self, X: np.ndarray) -> np.ndarray:
        """Get convolved features (before PLS)."""
        self._check_fitted()

        X = np.asarray(X, dtype=np.float32)
        Xs = _standardize_apply(X, self.x_mu_, self.x_sd_)

        device = next(self.model_.parameters()).device
        Xt = _as_torch(Xs, device)

        Z = self.model_.extractor(Xt)
        return Z.cpu().numpy()

    def get_kernels(self) -> np.ndarray:
        """Get learned kernels."""
        self._check_fitted()
        return self.model_.extractor.get_kernels()

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        params = {
            "version": self.version,
            "n_kernels": self.n_kernels,
            "kernel_size": self.kernel_size,
            "n_components": self.n_components,
            "ridge_lambda": self.ridge_lambda,
            "pls_mode": self.pls_mode,
            "feature_mode": self.feature_mode,
            "init_mode": self.init_mode,
            "alpha_max": self.alpha_max,
            "tau": self.tau,
            "alpha_init": self.alpha_init,
            "sigma_init": self.sigma_init,
        }
        if deep:
            params["train_cfg"] = self.train_cfg
        return params

    def set_params(self, **params) -> "FCKPLSTorch":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# Convenience Functions
# =============================================================================

def create_fckpls_v1(
    n_kernels: int = 16,
    kernel_size: int = 31,
    n_components: int = 10,
    epochs: int = 300,
    lr: float = 1e-3,
    verbose: int = 0,
    **kwargs,
) -> FCKPLSTorch:
    """Create a V1 (learnable kernels) FCK-PLS model."""
    cfg = TrainConfig(epochs=epochs, lr=lr, verbose=verbose)
    return FCKPLSTorch(
        version="v1",
        n_kernels=n_kernels,
        kernel_size=kernel_size,
        n_components=n_components,
        train_cfg=cfg,
        **kwargs,
    )


def create_fckpls_v2(
    n_kernels: int = 16,
    kernel_size: int = 31,
    n_components: int = 10,
    alpha_max: float = 2.0,
    epochs: int = 300,
    lr: float = 1e-3,
    verbose: int = 0,
    **kwargs,
) -> FCKPLSTorch:
    """Create a V2 (alpha/sigma parametric) FCK-PLS model."""
    cfg = TrainConfig(epochs=epochs, lr=lr, verbose=verbose)
    return FCKPLSTorch(
        version="v2",
        n_kernels=n_kernels,
        kernel_size=kernel_size,
        n_components=n_components,
        alpha_max=alpha_max,
        train_cfg=cfg,
        **kwargs,
    )


# =============================================================================
# nirs4all-compatible Module Wrapper
# =============================================================================

class FCKPLSModule(nn.Module):
    """
    nirs4all-compatible FCK-PLS module.

    This wrapper makes FCK-PLS compatible with nirs4all's PyTorchModelController
    by providing a standard forward(x) interface while internally managing the
    PLS computation that requires Y.

    The model has two modes:
    - Training: Expects set_targets(y) to be called before forward(x)
    - Inference: Uses cached PLS parameters from the last training batch

    Parameters
    ----------
    input_shape : tuple
        Shape of input (channels, sequence_length) for spectral data
    params : dict
        Configuration parameters:
        - version: "v1" or "v2" (default: "v1")
        - n_kernels: number of conv kernels (default: 16)
        - kernel_size: size of each kernel (default: 31)
        - n_components: PLS components (default: 10)
        - ridge_lambda: ridge regularization (default: 1e-3)
        - pls_mode: "deflation" or "svd" (default: "deflation")
        - feature_mode: "interleaved" or "concatenated" (default: "interleaved")
        - init_mode: kernel init for v1 (default: "random")
        - alpha_max: max fractional order for v2 (default: 2.0)
        - tau: smoothness for v2 (default: 1.0)
    num_classes : int
        Number of output targets (default: 1 for regression)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        params: Optional[Dict[str, Any]] = None,
        num_classes: int = 1,
    ):
        super().__init__()
        params = params or {}

        self.input_shape = input_shape
        self.num_classes = num_classes

        # Extract parameters
        version = params.get("version", "v1")
        n_kernels = params.get("n_kernels", 16)
        kernel_size = params.get("kernel_size", 31)
        n_components = params.get("n_components", 10)
        ridge_lambda = params.get("ridge_lambda", 1e-3)
        pls_mode = params.get("pls_mode", "deflation")
        feature_mode = params.get("feature_mode", "interleaved")
        init_mode = params.get("init_mode", "random")
        alpha_max = params.get("alpha_max", 2.0)
        tau = params.get("tau", 1.0)

        # Store feature mode for flattening
        self.feature_mode = feature_mode

        # Build extractor
        if version == "v1":
            self.extractor = LearnableKernelBank(
                n_kernels=n_kernels,
                kernel_size=kernel_size,
                init_mode=init_mode,
            )
        else:  # v2
            self.extractor = FractionalKernelBank(
                n_kernels=n_kernels,
                kernel_size=kernel_size,
                alpha_max=alpha_max,
                tau=tau,
            )

        # PLS head
        self.head = PLSSolvedHead(
            n_components=n_components,
            ridge_lambda=ridge_lambda,
            mode=pls_mode,
        )

        # Cache for inference
        self._head_cache: Optional[Dict[str, torch.Tensor]] = None
        self._current_targets: Optional[torch.Tensor] = None

        # Store params for regularization (version-specific)
        self._version = version
        if version == "v1":
            self._reg_params = {
                "smooth_w": params.get("reg_smooth", 1e-3),
                "zeromean_w": params.get("reg_zeromean", 1e-3),
                "l2_w": params.get("reg_l2", 0.0),
            }
        else:  # v2
            self._reg_params = {
                "alpha_w": params.get("alpha_w", 1e-4),
                "sigma_w": params.get("sigma_w", 1e-4),
            }

    def set_targets(self, y: torch.Tensor) -> None:
        """Set targets for the next forward pass (training mode)."""
        self._current_targets = y

    def flatten_features(self, Z: torch.Tensor) -> torch.Tensor:
        """Flatten features according to feature_mode."""
        if self.feature_mode == "interleaved":
            return Z.permute(0, 2, 1).flatten(1)  # (B, K, L) -> (B, L, K) -> (B, L*K)
        else:
            return Z.flatten(1)  # (B, K, L) -> (B, K*L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass compatible with nirs4all's training loop.

        During training (when targets are set via set_targets()):
            Computes PLS and returns predictions.

        During inference (eval mode with cached parameters):
            Uses cached PLS parameters for prediction.
        """
        # Handle 3D input (N, C, L) - extract features
        if x.ndim == 2:
            # (N, L) -> (N, 1, L) for conv
            x = x.unsqueeze(1)

        # Apply convolutional feature extraction
        Z = self.extractor(x[:, 0, :] if x.shape[1] == 1 else x.mean(dim=1))  # (B, K, L)
        Zf = self.flatten_features(Z)

        if self.training and self._current_targets is not None:
            # Training mode: compute PLS with current targets
            Y = self._current_targets
            if Y.ndim == 1:
                Y = Y.unsqueeze(1)

            Yhat, aux = self.head(Zf, Y)
            self._head_cache = {k: v.detach().clone() for k, v in aux.items()}
            return Yhat

        elif self._head_cache is not None:
            # Inference mode: use cached parameters
            W = self._head_cache["W"]
            B = self._head_cache["B"]
            z_mean = self._head_cache["z_mean"]
            y_mean = self._head_cache["y_mean"]

            Zc = Zf - z_mean
            T = Zc @ W
            Yhat = T @ B + y_mean
            return Yhat

        else:
            # No cache available - return zeros (shouldn't happen in normal use)
            warnings.warn("FCKPLSModule: No cached parameters available for inference")
            return torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=x.dtype)

    def kernel_regularization(self) -> torch.Tensor:
        """Compute kernel regularization loss."""
        return self.extractor.kernel_regularization(**self._reg_params)

    def get_kernels(self) -> np.ndarray:
        """Get learned kernels as numpy array."""
        return self.extractor.get_kernels()


# =============================================================================
# nirs4all Factory Functions
# =============================================================================

try:
    from nirs4all.utils import framework

    @framework("pytorch")
    def fckpls(input_shape: Tuple[int, int], params: Optional[Dict[str, Any]] = None) -> FCKPLSModule:
        """
        FCK-PLS model factory for nirs4all (regression).

        Creates a Fractional Convolutional Kernel PLS model compatible with
        nirs4all's pipeline system.

        Args:
            input_shape: (channels, sequence_length) tuple
            params: Configuration dict with keys:
                - version: "v1" (learnable kernels) or "v2" (alpha/sigma parametric)
                - n_kernels: number of convolution kernels (default: 16)
                - kernel_size: kernel size (default: 31)
                - n_components: PLS components (default: 10)
                - ridge_lambda: ridge regularization (default: 1e-3)
                - pls_mode: "deflation" or "svd" (default: "deflation")
                - init_mode: kernel initialization for v1 (default: "random")
                - alpha_max: max fractional order for v2 (default: 2.0)

        Returns:
            FCKPLSModule compatible with nirs4all's PyTorchModelController
        """
        return FCKPLSModule(input_shape, params or {}, num_classes=1)

    @framework("pytorch")
    def fckpls_v1(input_shape: Tuple[int, int], params: Optional[Dict[str, Any]] = None) -> FCKPLSModule:
        """FCK-PLS V1 with learnable free kernels (regression)."""
        p = dict(params or {})
        p["version"] = "v1"
        return FCKPLSModule(input_shape, p, num_classes=1)

    @framework("pytorch")
    def fckpls_v2(input_shape: Tuple[int, int], params: Optional[Dict[str, Any]] = None) -> FCKPLSModule:
        """FCK-PLS V2 with alpha/sigma parametric kernels (regression)."""
        p = dict(params or {})
        p["version"] = "v2"
        return FCKPLSModule(input_shape, p, num_classes=1)

    # Note: Classification variants are not provided because PLS is inherently
    # a regression method. For PLS-DA (Discriminant Analysis), the output
    # needs additional thresholding/classification layers which would require
    # a different architecture.

except ImportError:
    # nirs4all not available - skip factory function registration
    pass
