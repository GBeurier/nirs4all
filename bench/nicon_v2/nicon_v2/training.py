"""Training loop and y-processing helpers for nicon_v2.

Phase 0 only needs a single-network train loop with early stopping. Ensembles,
augmentation hooks, conformal calibration etc. land in later phases.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import rmse


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed numpy, torch and python random for full reproducibility on a single process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# y-processing
# ---------------------------------------------------------------------------


@dataclass
class StandardYProcessor:
    """Center-and-scale y for training. Inverse-transform at evaluation.

    Captures only the train mean/std; never recomputed on test.
    """

    mean: float = 0.0
    std: float = 1.0

    def fit(self, y: np.ndarray) -> "StandardYProcessor":
        y = np.asarray(y, dtype=float)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y) + 1e-12)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.std + self.mean


# ---------------------------------------------------------------------------
# x-processing
# ---------------------------------------------------------------------------


@dataclass
class StandardXProcessor:
    """Per-feature mean/std centering for X. Fitted on train only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardXProcessor":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardXProcessor was not fitted")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    device: str = "auto"
    seed: int = 0
    use_amp: bool = True
    log_every_n_epochs: int = 50
    one_cycle: bool = True
    # Phase 1b knobs.
    augmenter: object | None = None     # callable(x, generator) -> x'
    cmixup: object | None = None        # callable(x, y, sigma_y, generator) -> (x, y)
    cmixup_sigma_y: float = 0.0


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Train/val split (random) + tensor loaders. Returns (train_loader, val_loader, X_val, y_val).

    All tensors are kept on the target device for small datasets (NIRS is tiny);
    we don't pin / multi-worker because the per-batch overhead exceeds the gain.
    """
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_tr = torch.from_numpy(X_train[train_idx]).float().to(device)
    y_tr = torch.from_numpy(y_train[train_idx]).float().to(device)
    X_va = torch.from_numpy(X_train[val_idx]).float().to(device)
    y_va = torch.from_numpy(y_train[val_idx]).float().to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_va, y_va)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, X_train[val_idx], y_train[val_idx]


def train_torch_regressor(
    model: nn.Module,
    X_train_scaled: np.ndarray,   # already standardized X
    y_train_scaled: np.ndarray,   # already standardized y
    config: TrainConfig,
    n_input_channels: int = 1,
) -> tuple[nn.Module, dict[str, float]]:
    """Train a PyTorch regression model with MSE + early stopping.

    Returns ``(best_model, info)`` where ``info`` contains ``best_val_loss``,
    ``best_epoch``, ``train_time_s``.
    """
    set_global_seed(config.seed)
    device = pick_device(config.device)
    model = model.to(device)

    n, p = X_train_scaled.shape
    if n_input_channels != 1:
        # Caller responsibility: we always feed (N, C, L). For Phase 0 we use 1 channel.
        raise NotImplementedError("Phase 0 only supports n_input_channels=1")
    X_train_scaled = X_train_scaled.reshape(n, 1, p)

    train_loader, val_loader, _, _ = _make_loaders(
        X_train_scaled, y_train_scaled, config.val_fraction, config.batch_size, device, config.seed,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=config.lr,
            total_steps=max(1, config.epochs * max(1, len(train_loader))),
            pct_start=0.1,
            anneal_strategy="cos",
        )
    else:
        scheduler = None

    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None
    aug_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    t0 = time.time()
    for epoch in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            if config.augmenter is not None:
                xb = config.augmenter(xb, aug_generator)
            if config.cmixup is not None and config.cmixup_sigma_y > 0:
                xb, yb = config.cmixup(xb, yb, config.cmixup_sigma_y, aug_generator)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optim)
                scaler_amp.update()
            else:
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
            if scheduler is not None:
                scheduler.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                pred = model(xb).squeeze(-1)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else math.inf
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    return model, {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "train_time_s": float(train_time),
        "device": str(device),
    }


@torch.no_grad()
def predict_torch_regressor(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Run a forward pass on (N, p), returning a 1-D ndarray of predictions in the network's output scale."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n, p = X.shape
    X_t = torch.from_numpy(X.reshape(n, 1, p)).float().to(device)
    out = []
    for i in range(0, n, batch_size):
        chunk = X_t[i : i + batch_size]
        pred = model(chunk).squeeze(-1).detach().cpu().numpy()
        out.append(pred)
    return np.concatenate(out, axis=0)
