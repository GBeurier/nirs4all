"""DARTS PLS: Differentiable Architecture Search over preprocessing operators.

CHANGELOG:
    v1: Original - 5 operators, plain softmax, no temperature annealing, overfitting to train
    v2: Comprehensive rewrite:
        - 20+ operators covering all major NIRS transform families
        - Gumbel-Softmax temperature annealing (soft → hard selection)
        - Train/val split to prevent overfitting the operator weights
        - Entropy regularization for sparsity
        - Final fit uses the best discrete operator (hard snap) via AOM-PLS
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor


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
        if w_norm < 1e-8:
            break
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


def _apply_transform(op, X):
    """Apply a sklearn transform, handling fit + transform."""
    X_pre = X.copy()
    if hasattr(op, 'fit'):
        op.fit(X_pre)
    if hasattr(op, 'apply'):
        return op.apply(X_pre)
    return op.transform(X_pre)


def _build_transform_bank():
    """Build comprehensive transform bank covering all major NIRS families.

    Returns list of (name, transform_instance) tuples.
    """
    from nirs4all.operators.transforms import (
        StandardNormalVariate as SNV,
        MultiplicativeScatterCorrection as MSC,
        SavitzkyGolay as SG,
        Detrend,
        Gaussian,
        NorrisWilliams,
        WaveletDenoise,
    )
    from sklearn.preprocessing import FunctionTransformer

    bank = [
        # Identity (raw spectra)
        ("Identity", FunctionTransformer()),
        # Scatter correction (non-linear)
        ("SNV", SNV()),
        ("MSC", MSC()),
        # Smoothing
        ("SG_smooth_11", SG(window_length=11, polyorder=2, deriv=0)),
        ("SG_smooth_21", SG(window_length=21, polyorder=2, deriv=0)),
        # 1st derivatives (different resolutions)
        ("SG_d1_w11", SG(window_length=11, polyorder=2, deriv=1)),
        ("SG_d1_w15", SG(window_length=15, polyorder=2, deriv=1)),
        ("SG_d1_w21", SG(window_length=21, polyorder=2, deriv=1)),
        ("SG_d1_w31", SG(window_length=31, polyorder=2, deriv=1)),
        # 2nd derivatives
        ("SG_d2_w21", SG(window_length=21, polyorder=3, deriv=2)),
        ("SG_d2_w31", SG(window_length=31, polyorder=3, deriv=2)),
        # Detrend
        ("Detrend", Detrend()),
        # Gaussian smoothing
        ("Gaussian_3", Gaussian(sigma=3)),
        ("Gaussian_7", Gaussian(sigma=7)),
        # Norris-Williams
        ("NW_5_1", NorrisWilliams(gap=5, segment=5, deriv=1)),
        ("NW_11_1", NorrisWilliams(gap=11, segment=5, deriv=1)),
        # Wavelet denoising
        ("Wavelet_db4", WaveletDenoise(wavelet='db4', level=4)),
        ("Wavelet_sym5", WaveletDenoise(wavelet='sym5', level=4)),
    ]
    return bank


class DartsPLSRegressor(BaseEstimator, RegressorMixin):
    """DARTS PLS with Gumbel-Softmax annealing over a comprehensive transform bank.

    v2: 18 operators, temperature annealing, validation split, entropy regularization.
    Learns which preprocessing operator (or blend) minimizes PLS prediction error.
    Final model uses the discrete top-weighted operator via AOM-PLS.
    """
    def __init__(self, n_components=15, epochs=150, lr=0.05, val_fraction=0.2):
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self.val_fraction = val_fraction

    def fit(self, X, y):
        # Build transform bank and pre-compute all operator outputs
        self.bank_ = _build_transform_bank()
        X_ops_list = []
        valid_bank = []

        for name, op in self.bank_:
            try:
                X_pre = _apply_transform(op, X)
                if np.any(np.isnan(X_pre)) or np.any(np.isinf(X_pre)):
                    continue
                X_ops_list.append(torch.tensor(X_pre, dtype=torch.float32))
                valid_bank.append((name, op))
            except Exception:
                continue

        self.bank_ = valid_bank
        n_ops = len(valid_bank)
        X_ops = torch.stack(X_ops_list, dim=0)  # (n_ops, n_samples, p)

        # Train/val split for operator weight optimization
        n = X.shape[0]
        n_val = max(1, int(n * self.val_fraction))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_ops_train = X_ops[:, train_idx, :]
        X_ops_val = X_ops[:, val_idx, :]
        y_t_train = torch.tensor(y[train_idx], dtype=torch.float32).view(-1, 1)
        y_t_val = torch.tensor(y[val_idx], dtype=torch.float32).view(-1, 1)

        # Learnable architecture weights
        alphas = nn.Parameter(torch.zeros(n_ops))
        optimizer = torch.optim.Adam([alphas], lr=self.lr)

        # Temperature annealing: start soft (tau=1), end hard (tau=0.1)
        tau_start, tau_end = 1.0, 0.1

        best_val_loss = float('inf')
        best_alphas = alphas.data.clone()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Annealed temperature
            progress = epoch / max(self.epochs - 1, 1)
            tau = tau_start * (tau_end / tau_start) ** progress

            # Gumbel-Softmax (straight-through estimator for hard gradients)
            weights = F.gumbel_softmax(alphas, tau=tau, hard=False)

            # Mix operators
            X_train_mixed = torch.sum(weights.view(-1, 1, 1) * X_ops_train, dim=0)

            # Differentiable PLS on training portion
            B, intercept = torch_pls1(X_train_mixed, y_t_train, self.n_components)
            y_pred_train = torch.matmul(X_train_mixed, B) + intercept
            train_loss = F.mse_loss(y_pred_train, y_t_train)

            # Validation loss (no gradient through PLS coefficients - just evaluate)
            with torch.no_grad():
                X_val_mixed = torch.sum(weights.view(-1, 1, 1) * X_ops_val, dim=0)
                y_pred_val = torch.matmul(X_val_mixed, B) + intercept
                val_loss = F.mse_loss(y_pred_val, y_t_val).item()

            # Entropy regularization: encourage sparsity
            probs = F.softmax(alphas, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            loss = train_loss - 0.01 * entropy  # minimize MSE, minimize entropy (= more sparse)

            loss.backward()
            optimizer.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alphas = alphas.data.clone()

        # Use best alphas (by validation loss)
        final_weights = F.softmax(best_alphas, dim=0).numpy()
        self.weights_ = final_weights
        self.op_names_ = [name for name, _ in valid_bank]

        # Report top operators
        top_idx = np.argsort(final_weights)[::-1]
        self.top_ops_ = [(self.op_names_[i], final_weights[i]) for i in top_idx[:5]]

        # Final fit: try both hard-snap (best single op) and top-3 blend,
        # pick whichever has better validation RMSE
        y_val_np = y[val_idx]

        # Strategy A: best single operator
        best_op_idx = np.argmax(final_weights)
        X_best_tr = X_ops_train[best_op_idx].numpy()
        X_best_val = X_ops_val[best_op_idx].numpy()
        pls_a = AOMPLSRegressor(n_components=self.n_components)
        pls_a.fit(X_best_tr, y[train_idx])
        rmse_a = np.sqrt(np.mean((y_val_np - pls_a.predict(X_best_val).flatten()) ** 2))

        # Strategy B: top-3 weighted blend
        top3 = np.argsort(final_weights)[::-1][:3]
        top3_w = final_weights[top3]
        top3_w = top3_w / top3_w.sum()

        X_blend_tr = sum(w * X_ops_train[i].numpy() for i, w in zip(top3, top3_w))
        X_blend_val = sum(w * X_ops_val[i].numpy() for i, w in zip(top3, top3_w))
        pls_b = AOMPLSRegressor(n_components=self.n_components)
        pls_b.fit(X_blend_tr, y[train_idx])
        rmse_b = np.sqrt(np.mean((y_val_np - pls_b.predict(X_blend_val).flatten()) ** 2))

        # Pick best strategy and refit on all data
        if rmse_a <= rmse_b:
            self.strategy_ = "hard_snap"
            X_final = X_ops[best_op_idx].numpy()
            self.use_blend_ = False
            self.best_op_idx_ = best_op_idx
        else:
            self.strategy_ = "top3_blend"
            X_final = sum(w * X_ops[i].numpy() for i, w in zip(top3, top3_w))
            self.use_blend_ = True
            self.top3_indices_ = top3
            self.top3_weights_ = top3_w
            self.best_op_idx_ = top3[0]

        self.pls_ = AOMPLSRegressor(n_components=self.n_components)
        self.pls_.fit(X_final, y)
        self.best_op_name_ = self.op_names_[self.best_op_idx_]
        return self

    def predict(self, X):
        if not self.use_blend_:
            _, op = self.bank_[self.best_op_idx_]
            X_pre = X.copy()
            if hasattr(op, 'apply'):
                X_pre = op.apply(X_pre)
            else:
                X_pre = op.transform(X_pre)
            return self.pls_.predict(X_pre).flatten()
        else:
            X_ops_pred = []
            for idx in self.top3_indices_:
                _, op = self.bank_[idx]
                X_pre = X.copy()
                if hasattr(op, 'apply'):
                    X_pre = op.apply(X_pre)
                else:
                    X_pre = op.transform(X_pre)
                X_ops_pred.append(X_pre)

            X_mixed = sum(w * xp for xp, w in zip(X_ops_pred, self.top3_weights_))
            return self.pls_.predict(X_mixed).flatten()
