"""Smoke tests for the training loop on synthetic data — verify that loss decreases."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nicon_v2.training import (
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    pick_device,
    predict_torch_regressor,
    train_torch_regressor,
)


def _toy_problem(n: int = 64, p: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.3, size=p)
    X = rng.normal(size=(n, p))
    y = X @ w + rng.normal(scale=0.1, size=n)
    X_test = rng.normal(size=(n // 2, p))
    y_test = X_test @ w + rng.normal(scale=0.1, size=n // 2)
    return X, y, X_test, y_test


class TinyNet(nn.Module):
    """A small Conv → flatten → linear net with enough capacity to learn the synthetic linear toy problem."""

    def __init__(self, p: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * p, 1),
        )

    def forward(self, x):
        return self.body(x)


def test_train_loop_reduces_loss_on_synthetic():
    X, y, X_test, y_test = _toy_problem(n=200, p=64)
    x_proc = StandardXProcessor().fit(X)
    y_proc = StandardYProcessor().fit(y)
    Xs = x_proc.transform(X)
    ys = y_proc.transform(y)
    model = TinyNet(p=Xs.shape[1])
    # Validation MSE on randomly-initialised TinyNet is ~1.0 (since y is standardised);
    # the train loop should bring it well below 1.0 if it works.
    cfg = TrainConfig(epochs=80, patience=20, batch_size=16, lr=5e-3, weight_decay=0.0, seed=0, use_amp=False)
    model, info = train_torch_regressor(model, Xs, ys, cfg)
    assert info["best_val_loss"] < 0.95
    pred_scaled = predict_torch_regressor(model, x_proc.transform(X_test))
    pred = y_proc.inverse_transform(pred_scaled)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((y_test - np.mean(y)) ** 2)))
    # Tiny GAP-conv net should beat the constant predictor on this linear toy problem
    assert rmse < rmse_naive * 0.95


def test_pick_device_returns_torch_device():
    dev = pick_device("cpu")
    assert dev.type == "cpu"
    dev = pick_device("auto")
    assert isinstance(dev, torch.device)
