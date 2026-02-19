"""Torch backend for AOM-PLS.

Provides GPU-accelerated operator bank computations using torch.nn.functional.conv1d
for batched adjoint operations, enabling efficient per-component preprocessing
selection on large datasets.

Uses NIPALS deflation (matching the NumPy backend) and direct scoring
(no log transform) for correct operator selection.

Uses float32 by default (sufficient precision for NIRS-scale data).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _sparsemax_torch(z):
    """Sparsemax activation using torch tensors.

    Parameters
    ----------
    z : torch.Tensor of shape (d,)
        Input logits.

    Returns
    -------
    p : torch.Tensor of shape (d,)
        Sparse probability vector.
    """
    import torch

    d = z.shape[0]
    z_sorted, _ = torch.sort(z, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=0)
    k_range = torch.arange(1, d + 1, dtype=z.dtype, device=z.device)
    thresholds = (cumsum - 1.0) / k_range
    support = z_sorted > thresholds
    k_star = torch.max(torch.where(support)[0]) + 1 if torch.any(support) else 1
    tau = (cumsum[k_star - 1] - 1.0) / k_star
    return torch.clamp(z - tau, min=0.0)

class TorchOperatorBank:
    """Manages operator bank as torch conv1d kernels for batched operations.

    Groups all SG-type operators into a single batched conv1d call,
    and handles detrend/identity operators separately.

    Parameters
    ----------
    operators : list of LinearOperator
        Initialized operator bank from sklearn/aom_pls.py.
    device : torch.device
        Target device (cpu or cuda).
    """

    def __init__(self, operators, device):
        import torch

        from nirs4all.operators.models.sklearn.aom_pls import (
            SavitzkyGolayOperator,
        )

        self.device = device
        self.n_operators = len(operators)
        self.operators = operators

        # Classify operators by type for batched processing
        self._sg_indices = []
        self._sg_conv_kernels = []
        self._sg_adj_kernels = []
        self._other_indices = []

        for i, op in enumerate(operators):
            if isinstance(op, SavitzkyGolayOperator):
                self._sg_indices.append(i)
                self._sg_conv_kernels.append(op._conv_kernel)
                self._sg_adj_kernels.append(op._adj_kernel)
            else:
                self._other_indices.append(i)

        # Build batched conv1d weights for SG operators
        if self._sg_indices:
            # Pad all kernels to the same length
            max_len = max(len(k) for k in self._sg_adj_kernels)
            padded_kernels = []
            for k in self._sg_adj_kernels:
                pad_total = max_len - len(k)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                padded = np.pad(k, (pad_left, pad_right), mode='constant')
                padded_kernels.append(padded)
            # Shape: (n_sg, 1, kernel_size) for conv1d
            kernel_array = np.stack(padded_kernels)[:, np.newaxis, :]
            self._adj_weight = torch.tensor(kernel_array, dtype=torch.float32, device=device)
            self._adj_padding = (max_len - 1) // 2

            # Same for forward kernels
            padded_fwd = []
            for k in self._sg_conv_kernels:
                pad_total = max_len - len(k)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                padded = np.pad(k, (pad_left, pad_right), mode='constant')
                padded_fwd.append(padded)
            fwd_array = np.stack(padded_fwd)[:, np.newaxis, :]
            self._fwd_weight = torch.tensor(fwd_array, dtype=torch.float32, device=device)

        # Precompute Frobenius norms
        self.nus = torch.tensor(
            [op.frobenius_norm_sq() for op in operators],
            dtype=torch.float32,
            device=device,
        )

    def apply_adjoint_all(self, c):
        """Apply all operator adjoints to vector c.

        Parameters
        ----------
        c : torch.Tensor of shape (p,)
            Input vector.

        Returns
        -------
        gradients : torch.Tensor of shape (B, p)
            g_{b} = A_b^T c for each operator b.
        """
        import torch
        import torch.nn.functional as F

        p = c.shape[0]
        B = self.n_operators
        gradients = torch.zeros(B, p, dtype=torch.float32, device=self.device)

        # Batch SG adjoint operations via conv1d
        if self._sg_indices:
            # c as (1, 1, p) for conv1d input
            c_3d = c.unsqueeze(0).unsqueeze(0)  # (1, 1, p)
            # Conv1d with multiple output channels (one per SG operator)
            sg_results = F.conv1d(c_3d, self._adj_weight, padding=self._adj_padding)  # (1, n_sg, p)
            sg_results = sg_results.squeeze(0)  # (n_sg, p)
            for local_idx, global_idx in enumerate(self._sg_indices):
                gradients[global_idx] = sg_results[local_idx]

        # Non-SG operators: apply individually (identity, detrend, composed)
        c_np = c.cpu().numpy()
        for global_idx in self._other_indices:
            g_np = self.operators[global_idx].apply_adjoint(c_np)
            gradients[global_idx] = torch.tensor(g_np, dtype=torch.float32, device=self.device)

        return gradients

    def apply_forward(self, op_idx, x):
        """Apply forward operator to a vector.

        Parameters
        ----------
        op_idx : int
            Operator index.
        x : torch.Tensor of shape (p,) or (1, p)
            Input vector.

        Returns
        -------
        result : torch.Tensor of shape matching input
            A_b @ x.
        """
        import torch
        import torch.nn.functional as F

        from nirs4all.operators.models.sklearn.aom_pls import SavitzkyGolayOperator

        op = self.operators[op_idx]
        if isinstance(op, SavitzkyGolayOperator) and self._sg_indices:
            local_idx = self._sg_indices.index(op_idx)
            x_3d = x.reshape(1, 1, -1)
            weight = self._fwd_weight[local_idx:local_idx + 1]  # (1, 1, L)
            result = F.conv1d(x_3d, weight, padding=self._adj_padding)
            return result.reshape(x.shape)

        # Fallback to numpy for other operators
        x_np = x.cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        result_np = op.apply(x_np)
        return torch.tensor(result_np.reshape(x.shape), dtype=torch.float32, device=self.device)

def aompls_fit_torch(
    X: NDArray,
    Y: NDArray,
    operators: list,
    n_components: int,
    tau: float,
    n_orth: int,
    gate: str = "hard",
) -> dict:
    """Fit AOM-PLS model using Torch backend with NIPALS deflation.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Centered X matrix.
    Y : ndarray of shape (n, q)
        Centered Y matrix.
    operators : list of LinearOperator
        Initialized operator bank.
    n_components : int
        Maximum number of components.
    tau : float
        Sparsemax temperature (only used when gate='sparsemax').
    n_orth : int
        Number of OPLS orthogonal components.
    gate : str
        'hard' for argmax, 'sparsemax' for soft mixing.

    Returns
    -------
    artifacts : dict
        Fitted artifacts (converted to numpy).
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n, p = X.shape
    q = Y.shape[1]
    B = len(operators)
    eps = 1e-7

    # OPLS pre-filter (use numpy - small computation)
    P_orth = None
    if n_orth > 0:
        from nirs4all.operators.models.sklearn.aom_pls import _opls_prefilter
        X, P_orth, _ = _opls_prefilter(X, Y[:, 0] if q == 1 else Y, n_orth)

    # Move data to torch
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    # Build batched operator bank
    bank = TorchOperatorBank(operators, device)

    # ---- Global operator selection via R² scoring ----
    c_0 = X_t.T @ Y_t
    if q == 1:
        c_0 = c_0[:, 0]
        y_score = Y_t[:, 0]
    else:
        u0, s0, vt0 = torch.linalg.svd(c_0, full_matrices=False)
        c_0 = u0[:, 0] * s0[0]
        y_score = Y_t @ vt0[0]

    y_norm_sq = torch.dot(y_score, y_score)
    block_grads_0 = bank.apply_adjoint_all(c_0)  # (B, p)
    scores_0 = torch.zeros(B, dtype=torch.float32, device=device)
    for b in range(B):
        g_b = block_grads_0[b]
        g_norm = torch.linalg.norm(g_b)
        if g_norm < eps:
            continue
        w_hat_b = g_b / g_norm
        a_w = bank.apply_forward(b, w_hat_b)
        a_w_norm = torch.linalg.norm(a_w)
        if a_w_norm < eps:
            continue
        w_b = a_w / a_w_norm
        t_b = X_t @ w_b
        cov_yt = torch.dot(y_score, t_b)
        scores_0[b] = cov_yt ** 2 / (y_norm_sq * torch.dot(t_b, t_b) + eps)

    if gate == "hard":
        best_b = int(torch.argmax(scores_0))
        gamma_row = torch.zeros(B, dtype=torch.float32, device=device)
        gamma_row[best_b] = 1.0
        selected_ops = [(best_b, 1.0)]
    else:
        score_max = torch.max(scores_0)
        gamma_row = _sparsemax_torch(scores_0 / (tau * score_max + eps))
        selected_ops = [(b, float(gamma_row[b])) for b in range(B) if gamma_row[b] > eps]

    # ---- NIPALS with selected operator(s) ----
    W = torch.zeros(p, n_components, dtype=torch.float32, device=device)
    T = torch.zeros(n, n_components, dtype=torch.float32, device=device)
    P = torch.zeros(p, n_components, dtype=torch.float32, device=device)
    Q = torch.zeros(q, n_components, dtype=torch.float32, device=device)
    Gamma = torch.zeros(n_components, B, dtype=torch.float32, device=device)

    X_res = X_t.clone()
    Y_res = Y_t.clone()

    n_extracted = 0

    for k in range(n_components):
        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
        else:
            u_svd, s_svd, _ = torch.linalg.svd(c_k, full_matrices=False)
            c_k = u_svd[:, 0] * s_svd[0]

        c_norm = torch.linalg.norm(c_k)
        if c_norm < eps:
            break

        # Compute weight using globally selected operator(s)
        w_k = torch.zeros(p, dtype=torch.float32, device=device)
        for b_idx, weight in selected_ops:
            g_b = bank.apply_adjoint_all(c_k)[b_idx]
            g_norm = torch.linalg.norm(g_b)
            if g_norm < eps:
                continue
            w_hat_b = g_b / g_norm
            a_w = bank.apply_forward(b_idx, w_hat_b)
            a_w_norm = torch.linalg.norm(a_w)
            if a_w_norm < eps:
                continue
            w_k = w_k + weight * (a_w / a_w_norm)

        w_norm = torch.linalg.norm(w_k)
        if w_norm < eps:
            break
        w_k = w_k / w_norm

        Gamma[k] = gamma_row

        t_k = X_res @ w_k
        tt = t_k @ t_k
        if tt < eps:
            break

        p_k = (X_res.T @ t_k) / tt
        q_k = (Y_res.T @ t_k) / tt

        W[:, k] = w_k
        T[:, k] = t_k
        P[:, k] = p_k
        Q[:, k] = q_k

        n_extracted = k + 1

        X_res = X_res - torch.outer(t_k, p_k)
        Y_res = Y_res - torch.outer(t_k, q_k)

    # Compute prefix regression coefficients
    B_coefs = torch.zeros(n_extracted, p, q, dtype=torch.float32, device=device)
    for k in range(n_extracted):
        W_a = W[:, :k + 1]
        P_a = P[:, :k + 1]
        Q_a = Q[:, :k + 1]
        PtW = P_a.T @ W_a
        R_a = W_a @ torch.linalg.pinv(PtW)
        B_coefs[k] = R_a @ Q_a.T

    # Convert back to numpy
    return {
        "n_extracted": n_extracted,
        "W": W[:, :n_extracted].cpu().numpy().astype(np.float64),
        "T": T[:, :n_extracted].cpu().numpy().astype(np.float64),
        "P": P[:, :n_extracted].cpu().numpy().astype(np.float64),
        "Q": Q[:, :n_extracted].cpu().numpy().astype(np.float64),
        "Gamma": Gamma[:n_extracted].cpu().numpy().astype(np.float64),
        "B_coefs": B_coefs[:n_extracted].cpu().numpy().astype(np.float64),
        "P_orth": P_orth,
    }
