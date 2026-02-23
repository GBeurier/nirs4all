import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from nirs4all.operators.models.sklearn.aom_pls import LinearOperator, AOMPLSRegressor, default_operator_bank
from pseudo_linear_aom import PseudoLinearSNVOperator

# =============================================================================
# Differentiable PLS1 Implementation (PyTorch)
# =============================================================================
def torch_pls1(X, y, n_components):
    """Differentiable PLS1 implementation in PyTorch."""
    X_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    X_k = X - X_mean
    y_k = y - y_mean

    W, P, Q = [], [], []
    for _ in range(n_components):
        w = torch.matmul(X_k.T, y_k)
        w_norm = torch.norm(w)
        if w_norm < 1e-8: break
        w = w / w_norm

        t = torch.matmul(X_k, w)
        t_norm_sq = torch.sum(t ** 2) + 1e-8

        p = torch.matmul(X_k.T, t) / t_norm_sq
        q = torch.matmul(y_k.T, t) / t_norm_sq

        X_k = X_k - torch.matmul(t, p.T)
        y_k = y_k - t * q

        W.append(w)
        P.append(p)
        Q.append(q)

    if not W:
        B = torch.zeros((X.shape[1], 1), device=X.device)
    else:
        W = torch.cat(W, dim=1)
        P = torch.cat(P, dim=1)
        Q = torch.cat(Q, dim=1)
        PTW = torch.matmul(P.T, W)
        PTW = PTW + torch.eye(PTW.shape[0], device=PTW.device) * 1e-6
        B = torch.matmul(W, torch.matmul(torch.linalg.inv(PTW), Q.T))

    intercept = y_mean - torch.matmul(X_mean, B)
    return B, intercept

# =============================================================================
# 2. Joint Optimization via Block-Coordinate Descent (BCD)
# =============================================================================
class BCDPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=15, max_iter=100, lr=0.05):
        self.n_components = n_components
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.kernel = nn.Parameter(torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], dtype=torch.float32).view(1, 1, 5))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        optimizer = torch.optim.Adam([self.kernel, self.scale, self.bias], lr=self.lr)

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            k_norm = F.softmax(self.kernel.view(-1), dim=0).view(1, 1, 5)
            X_smooth = F.conv1d(X_t.unsqueeze(1), k_norm, padding=2).squeeze(1)
            X_pre = self.scale * X_smooth + self.bias

            B, intercept = torch_pls1(X_pre, y_t, self.n_components)
            y_pred = torch.matmul(X_pre, B) + intercept

            loss = F.mse_loss(y_pred, y_t)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            k_norm = F.softmax(self.kernel.view(-1), dim=0).view(1, 1, 5)
            X_smooth = F.conv1d(X_t.unsqueeze(1), k_norm, padding=2).squeeze(1)
            X_pre = self.scale * X_smooth + self.bias
            self.B_, self.intercept_ = torch_pls1(X_pre, y_t, self.n_components)
            self.B_ = self.B_.numpy()
            self.intercept_ = self.intercept_.numpy()
            self.k_norm_ = k_norm.numpy()
            self.scale_ = self.scale.item()
            self.bias_ = self.bias.item()

        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        k_norm = torch.tensor(self.k_norm_, dtype=torch.float32)
        X_smooth = F.conv1d(X_t.unsqueeze(1), k_norm, padding=2).squeeze(1)
        X_pre = self.scale_ * X_smooth + self.bias_
        return (X_pre.numpy() @ self.B_ + self.intercept_).flatten()

# =============================================================================
# 3. Multi-Armed Bandit Operator Selection (Dynamic NIPALS)
# =============================================================================
# CHANGELOG:
#   v1: Original - missing numpy import, aggressive pruning (50%), no validation split
#   v2: Added numpy import, validation-based n_components, UCB scoring
#   v3: Holdout-guided per-component selection (premature stopping issue)
#   v4: Two-phase bandit: phase 1 = quick R²-based screening to find top-K ops,
#       phase 2 = full NIPALS + holdout RMSE for top-K ops (like baseline).
#       Includes pseudo-linear SNV + full default bank.
# =============================================================================
class BanditAOMPLSRegressor(BaseEstimator, RegressorMixin):
    """Bandit-accelerated AOM-PLS: efficiently searches a large operator bank.

    Strategy:
    1. Quick screen: extract 1 component per operator, rank by R². Keep top-K.
    2. Full evaluation: run full NIPALS for each top-K operator (like baseline
       AOM-PLS), select by holdout RMSE.

    This achieves same quality as baseline AOM-PLS but scales to larger banks
    by avoiding full NIPALS for unpromising operators.
    """
    def __init__(self, n_components=15, top_k=15, screen_components=3):
        self.n_components = n_components
        self.top_k = top_k
        self.screen_components = screen_components

    def _build_operator_bank(self, X):
        """Full default bank + pseudo-linear SNV."""
        snv_op = PseudoLinearSNVOperator()
        snv_op.fit(X)
        ops = default_operator_bank() + [snv_op]
        p = X.shape[1]
        for op in ops:
            op.initialize(p)
        return ops

    @staticmethod
    def _nipals_extract(X, Y, operator, n_components, eps=1e-12):
        """Standard NIPALS extraction for a single operator (mirrors AOM-PLS)."""
        n, p = X.shape
        q = Y.shape[1]
        W = np.zeros((p, n_components), dtype=np.float64)
        T = np.zeros((n, n_components), dtype=np.float64)
        P = np.zeros((p, n_components), dtype=np.float64)
        Q = np.zeros((q, n_components), dtype=np.float64)
        X_res, Y_res = X.copy(), Y.copy()
        n_ext = 0

        for k in range(n_components):
            c_k = X_res.T @ Y_res
            if q == 1:
                c_k = c_k[:, 0]
            else:
                u, s, _ = np.linalg.svd(c_k, full_matrices=False)
                c_k = u[:, 0] * s[0]
            if np.linalg.norm(c_k) < eps:
                break

            g = operator.apply_adjoint(c_k)
            g_norm = np.linalg.norm(g)
            if g_norm < eps:
                break
            w_hat = g / g_norm
            a_w = operator.apply(w_hat.reshape(1, -1)).ravel()
            a_w_norm = np.linalg.norm(a_w)
            if a_w_norm < eps:
                break
            w_k = a_w / a_w_norm

            t_k = X_res @ w_k
            tt = t_k @ t_k
            if tt < eps:
                break

            p_k = (X_res.T @ t_k) / tt
            q_k = (Y_res.T @ t_k) / tt

            W[:, k], T[:, k], P[:, k], Q[:, k] = w_k, t_k, p_k, q_k
            X_res -= np.outer(t_k, p_k)
            Y_res -= np.outer(t_k, q_k)
            n_ext = k + 1

        # Prefix B coefficients
        B_prefix = []
        for kk in range(1, n_ext + 1):
            PtW = P[:, :kk].T @ W[:, :kk]
            try:
                R = W[:, :kk] @ np.linalg.inv(PtW)
            except np.linalg.LinAlgError:
                R = W[:, :kk] @ np.linalg.pinv(PtW)
            B_prefix.append(R @ Q[:, :kk].T)

        return B_prefix, n_ext

    def fit(self, X, y):
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        X_c = X - self.x_mean_
        Y_c = (y - self.y_mean_).reshape(-1, 1)

        operators = self._build_operator_bank(X)
        eps = 1e-12

        # --- Phase 1: Quick screen (few-component NIPALS per operator, score by cumulative R²) ---
        op_scores = np.full(len(operators), -np.inf)
        y_ss = np.sum(Y_c ** 2)

        for i, op in enumerate(operators):
            B_prefix, n_ext = self._nipals_extract(X_c, Y_c, op, self.screen_components)
            if n_ext > 0:
                y_pred = X_c @ B_prefix[n_ext - 1]
                ss_res = np.sum((Y_c - y_pred) ** 2)
                op_scores[i] = 1.0 - ss_res / (y_ss + eps)

        top_indices = np.argsort(op_scores)[::-1][:self.top_k]
        self.screened_ops_ = [operators[i].name for i in top_indices if op_scores[i] > -np.inf]

        # --- Phase 2: Full NIPALS + holdout RMSE for top-K operators ---
        n = X_c.shape[0]
        n_val = max(1, int(n * 0.2))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        x_mean_tr = np.mean(X[train_idx], axis=0)
        y_mean_tr = np.mean(y[train_idx])
        X_tr_c = X[train_idx] - x_mean_tr
        Y_tr_c = (y[train_idx] - y_mean_tr).reshape(-1, 1)
        X_val_c = X[val_idx] - x_mean_tr
        y_val = y[val_idx].reshape(-1, 1)

        # Re-initialize operators on training data for pseudo-linear ones
        ops_tr = self._build_operator_bank(X[train_idx])

        best_rmse = np.inf
        best_op_idx = top_indices[0]
        best_n_comp = 1

        for i in top_indices:
            if op_scores[i] <= -np.inf:
                continue
            op = ops_tr[i]
            B_prefix, n_ext = self._nipals_extract(X_tr_c, Y_tr_c, op, self.n_components)

            for kk in range(n_ext):
                y_pred = X_val_c @ B_prefix[kk] + y_mean_tr
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_op_idx = i
                    best_n_comp = kk + 1

        # --- Final fit on ALL data with best operator and n_components ---
        best_op = operators[best_op_idx]
        B_prefix_full, n_ext_full = self._nipals_extract(X_c, Y_c, best_op, self.n_components)

        k_use = min(best_n_comp, n_ext_full)
        self.B_coefs_ = B_prefix_full[k_use - 1] if k_use > 0 else np.zeros((X.shape[1], 1))
        self.n_components_ = k_use
        self.selected_op_ = best_op.name
        self.selected_ops_ = [best_op.name]
        return self

    def predict(self, X):
        X_c = X - self.x_mean_
        return (X_c @ self.B_coefs_ + self.y_mean_).flatten()

# =============================================================================
# 4. Differentiable PLS with Implicit Layers
# =============================================================================
class DiffPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=15, epochs=150, lr=0.005):
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        class ResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(channels)
                self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(channels)
            def forward(self, x):
                res = x
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                return F.relu(x + res)

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            ResBlock(16),
            ResBlock(16),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            X_pre = self.net(X_t).squeeze(1) + X_t.squeeze(1)

            B, intercept = torch_pls1(X_pre, y_t, self.n_components)
            y_pred = torch.matmul(X_pre, B) + intercept

            loss = F.mse_loss(y_pred, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            self.net.eval()
            X_pre = self.net(X_t).squeeze(1) + X_t.squeeze(1)
            self.B_, self.intercept_ = torch_pls1(X_pre, y_t, self.n_components)
            self.B_ = self.B_.numpy()
            self.intercept_ = self.intercept_.numpy()
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.net.eval()
        with torch.no_grad():
            X_pre = self.net(X_t).squeeze(1) + X_t.squeeze(1)
        return (X_pre.numpy() @ self.B_ + self.intercept_).flatten()

# =============================================================================
# 7. Latent Space PLS
# =============================================================================
class LatentPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, latent_dim=64, n_components=15, epochs=150, lr=0.005):
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(max(1, self.latent_dim // 32)),
            nn.Flatten()
        )

        dummy_out = self.encoder(X_t)
        flat_size = dummy_out.shape[1]

        self.decoder = nn.Sequential(
            nn.Linear(flat_size, 32 * max(1, X.shape[1] // 4)),
            nn.Unflatten(1, (32, max(1, X.shape[1] // 4))),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
        )

        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            Z = self.encoder(X_t)
            X_rec = self.decoder(Z)
            if X_rec.shape[2] != X_t.shape[2]:
                X_rec = F.interpolate(X_rec, size=X_t.shape[2])
            loss = F.mse_loss(X_rec, X_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            self.encoder.eval()
            Z_np = self.encoder(X_t).numpy()

        self.pls_ = AOMPLSRegressor(n_components=min(self.n_components, Z_np.shape[1]))
        self.pls_.fit(Z_np, y)
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.encoder.eval()
        with torch.no_grad():
            Z_np = self.encoder(X_t).numpy()
        return self.pls_.predict(Z_np).flatten()

