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
    # V6 — knowledge distillation from a training-time teacher.
    teacher_predictions: np.ndarray | None = None    # shape (n,), aligned with X_train_scaled
    distill_lambda: float = 0.0                       # weight on distillation MSE term
    # R13 — Stochastic Weight Averaging.
    use_swa: bool = False                              # enable SWA over the last epochs
    swa_start_frac: float = 0.75                      # SWA averaging starts at swa_start_frac * epochs
    swa_lr: float | None = None                        # constant SWA LR; None → uses current optimiser LR
    # R17 B — heteroscedastic / robust loss.
    # If `loss_type=="studentt"`, the model's last linear must output 2 channels
    # (μ, log σ²). Loss = Student-t NLL with degrees-of-freedom `studentt_df`.
    # If `loss_type=="huber"`, scalar output, Huber loss with delta `huber_delta`.
    # Default `"mse"` keeps the round-12-16 behaviour.
    loss_type: str = "mse"
    studentt_df: float = 5.0
    huber_delta: float = 1.0
    # R17 F — auxiliary multi-task target (e.g. PLS projection coefficients).
    # ``aux_targets`` shape (n, n_components), aligned with X_train_scaled.
    # If model exposes ``last_aux_pred`` (NiconV2A.aux_head is set), the
    # training loss adds ``aux_lambda * MSE(model.last_aux_pred, aux_targets)``.
    aux_targets: np.ndarray | None = None
    aux_lambda: float = 0.0
    # R17 C-boost — feature-stacking variant of PLS-residual hybrid.
    # ``boost_signals`` shape (n, B) (or (n,) for B=1), aligned with
    # X_train_scaled. When set, model.forward(x, boost_signal=batch_z) is
    # called with the per-sample boost signal concatenated to GAP features
    # before the head. Y target stays unchanged (no residual subtraction).
    boost_signals: np.ndarray | None = None


def _compute_supervised_loss(
    pred: torch.Tensor,
    yb: torch.Tensor,
    loss_type: str,
    studentt_df: float,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, point_pred)``.

    For ``loss_type=="mse"``: ``point_pred = pred.squeeze(-1)``, MSE loss.
    For ``loss_type=="huber"``: same shape, Huber loss with `huber_delta`.
    For ``loss_type=="studentt"``: ``pred`` has 2 channels (μ, log σ²);
        loss = Student-t NLL with degrees-of-freedom `studentt_df`;
        ``point_pred = μ`` for downstream val and inference.

    For Student-t NLL with df=ν > 0, location μ, scale s = exp(0.5·log σ²):

        NLL = -log Γ((ν+1)/2) + log Γ(ν/2) + 0.5·log(πν) + log s
              + ((ν+1)/2)·log(1 + (1/ν)·((y-μ)/s)²)

    We minimise the per-batch mean.
    """
    if loss_type == "mse":
        point_pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        loss = nn.functional.mse_loss(point_pred, yb)
        return loss, point_pred
    if loss_type == "huber":
        point_pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        loss = nn.functional.huber_loss(point_pred, yb, delta=huber_delta)
        return loss, point_pred
    if loss_type == "studentt":
        # pred shape: (N, 2) → (μ, log σ²)
        if pred.dim() != 2 or pred.shape[-1] != 2:
            raise RuntimeError(f"studentt loss requires 2-channel output; got {pred.shape}")
        mu = pred[..., 0]
        log_var = pred[..., 1].clamp(min=-10.0, max=10.0)
        log_s = 0.5 * log_var
        s = torch.exp(log_s).clamp(min=1e-6)
        nu = float(studentt_df)
        z = (yb - mu) / s
        # log-Gamma constants are constants in the model parameters; drop for simpler grad
        const = 0.5 * math.log(math.pi * nu)
        nll = log_s + 0.5 * (nu + 1.0) * torch.log1p(z * z / nu) + const
        return nll.mean(), mu
    raise ValueError(f"unknown loss_type {loss_type!r}; expected mse | huber | studentt")


def _model_extra_loss(model: nn.Module) -> torch.Tensor | None:
    """Return a model-side penalty (e.g. AOM operator L2-from-init) or None.

    Looks for a method named ``operator_regularisation_loss`` (preferred) or
    ``regularisation_loss`` and calls it. Both V2A and the operator layers expose it.
    Codex round 6 F1: this was previously *defined* but never *called* during
    training, so the L2-from-init penalty was inactive.
    """
    for attr in ("operator_regularisation_loss", "regularisation_loss"):
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                val = fn()
            except Exception:
                continue
            if isinstance(val, torch.Tensor) and val.requires_grad:
                return val
            if isinstance(val, torch.Tensor) and val.detach().abs().sum().item() > 0.0:
                return val
    return None


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    teacher_predictions: np.ndarray | None = None,
    aux_targets: np.ndarray | None = None,
    boost_signals: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Train/val split (random) + tensor loaders. Returns (train_loader, val_loader, X_val, y_val).

    All tensors are kept on the target device for small datasets (NIRS is tiny);
    we don't pin / multi-worker because the per-batch overhead exceeds the gain.

    If `teacher_predictions` is given (shape (n,), aligned with X_train), each
    train batch yields a third tensor ``z_teacher`` for distillation.
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

    has_teacher = teacher_predictions is not None
    has_aux = aux_targets is not None
    has_boost = boost_signals is not None
    tensors: list[torch.Tensor] = [X_tr, y_tr]
    if has_teacher:
        tensors.append(torch.from_numpy(np.asarray(teacher_predictions)[train_idx]).float().to(device))
    if has_aux:
        aux_arr = np.asarray(aux_targets)[train_idx]
        if aux_arr.ndim == 1:
            aux_arr = aux_arr[:, None]
        tensors.append(torch.from_numpy(aux_arr).float().to(device))
    if has_boost:
        boost_arr = np.asarray(boost_signals)[train_idx]
        if boost_arr.ndim == 1:
            boost_arr = boost_arr[:, None]
        tensors.append(torch.from_numpy(boost_arr).float().to(device))
    train_ds: TensorDataset = TensorDataset(*tensors)
    # Validation set also needs boost signals (for forward at val time).
    if has_boost:
        boost_va = np.asarray(boost_signals)[val_idx]
        if boost_va.ndim == 1:
            boost_va = boost_va[:, None]
        val_ds = TensorDataset(X_va, y_va,
                                torch.from_numpy(boost_va).float().to(device))
    else:
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

    distill_active = (
        config.distill_lambda > 0.0
        and config.teacher_predictions is not None
        and len(config.teacher_predictions) == n
    )
    aux_active = (
        config.aux_lambda > 0.0
        and config.aux_targets is not None
        and len(config.aux_targets) == n
    )
    boost_active = (
        config.boost_signals is not None
        and len(config.boost_signals) == n
    )
    train_loader, val_loader, _, _ = _make_loaders(
        X_train_scaled, y_train_scaled, config.val_fraction, config.batch_size, device, config.seed,
        teacher_predictions=(config.teacher_predictions if distill_active else None),
        aux_targets=(config.aux_targets if aux_active else None),
        boost_signals=(config.boost_signals if boost_active else None),
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

    # R13 — Stochastic Weight Averaging.
    swa_active = bool(config.use_swa)
    swa_start_epoch = int(round(config.swa_start_frac * config.epochs)) if swa_active else config.epochs + 1
    swa_model: torch.optim.swa_utils.AveragedModel | None = None
    if swa_active:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr = float(config.swa_lr) if config.swa_lr is not None else float(config.lr) * 0.1

    loss_fn = nn.MSELoss()  # kept for distill MSE term and val (which uses point_pred)
    loss_type = str(getattr(config, "loss_type", "mse"))
    studentt_df = float(getattr(config, "studentt_df", 5.0))
    huber_delta = float(getattr(config, "huber_delta", 1.0))
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None
    aug_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    t0 = time.time()
    distill_lambda = float(config.distill_lambda) if distill_active else 0.0
    aux_lambda = float(config.aux_lambda) if aux_active else 0.0
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            # Order: (X, y, [teacher_z], [aux_target], [boost_signal])
            xb, yb = batch[0], batch[1]
            cursor = 2
            zb = batch[cursor] if distill_active else None
            if distill_active:
                cursor += 1
            ab = batch[cursor] if aux_active else None
            if aux_active:
                cursor += 1
            bsb = batch[cursor] if boost_active else None
            optim.zero_grad(set_to_none=True)
            if config.augmenter is not None:
                xb = config.augmenter(xb, aug_generator)
            if config.cmixup is not None and config.cmixup_sigma_y > 0:
                # cmixup also blends y; if distillation is active we stop teacher
                # mixing for that batch (mixed sample has no clean teacher target).
                xb, yb = config.cmixup(xb, yb, config.cmixup_sigma_y, aug_generator)
                zb = None
            # Codex round 6 F1 — wire model.operator_regularisation_loss() (or
            # any other extra penalty exposed by the model) into the training loss.
            if use_amp:
                with torch.amp.autocast("cuda"):
                    raw_pred = model(xb, boost_signal=bsb) if bsb is not None else model(xb)
                    loss, point_pred = _compute_supervised_loss(
                        raw_pred, yb, loss_type, studentt_df, huber_delta,
                    )
                    if zb is not None:
                        loss = loss + distill_lambda * loss_fn(point_pred, zb)
                    if ab is not None and getattr(model, "last_aux_pred", None) is not None:
                        loss = loss + aux_lambda * loss_fn(model.last_aux_pred, ab)
                    extra = _model_extra_loss(model)
                    if extra is not None:
                        loss = loss + extra
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optim)
                scaler_amp.update()
            else:
                raw_pred = model(xb, boost_signal=bsb) if bsb is not None else model(xb)
                loss, point_pred = _compute_supervised_loss(
                    raw_pred, yb, loss_type, studentt_df, huber_delta,
                )
                if zb is not None:
                    loss = loss + distill_lambda * loss_fn(point_pred, zb)
                if ab is not None and getattr(model, "last_aux_pred", None) is not None:
                    loss = loss + aux_lambda * loss_fn(model.last_aux_pred, ab)
                extra = _model_extra_loss(model)
                if extra is not None:
                    loss = loss + extra
                loss.backward()
                optim.step()
            # During SWA, freeze the OneCycleLR schedule at swa_lr.
            if scheduler is not None and not (swa_active and epoch >= swa_start_epoch):
                scheduler.step()

        # SWA: hold LR constant at swa_lr and update the averaged weights.
        if swa_active and epoch >= swa_start_epoch:
            for pg in optim.param_groups:
                pg["lr"] = swa_lr
            assert swa_model is not None
            swa_model.update_parameters(model)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch_v in val_loader:
                xb_v, yb_v = batch_v[0], batch_v[1]
                bs_v = batch_v[2] if (boost_active and len(batch_v) >= 3) else None
                raw_pred = model(xb_v, boost_signal=bs_v) if bs_v is not None else model(xb_v)
                # Use point prediction (μ for studentt) as val metric so val curves
                # remain comparable across loss_type configurations.
                if loss_type == "studentt" and raw_pred.dim() == 2 and raw_pred.shape[-1] == 2:
                    point_pred = raw_pred[..., 0]
                else:
                    point_pred = raw_pred.squeeze(-1) if raw_pred.dim() > 1 else raw_pred
                val_losses.append(loss_fn(point_pred, yb_v).item())
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

    # Final selection: prefer SWA weights if their val loss is strictly better
    # than the early-stopping checkpoint; otherwise use the early-stopping
    # checkpoint (this prevents SWA from regressing on small datasets where the
    # average of late-epoch models has drifted past the optimum).
    used_swa = False
    if swa_active and swa_model is not None:
        # SWA must be evaluated through its module attribute; AveragedModel forwards.
        swa_model.eval()
        with torch.no_grad():
            val_losses = []
            for batch_v in val_loader:
                xb_v, yb_v = batch_v[0], batch_v[1]
                bs_v = batch_v[2] if (boost_active and len(batch_v) >= 3) else None
                raw_pred = swa_model(xb_v, boost_signal=bs_v) if bs_v is not None else swa_model(xb_v)
                if loss_type == "studentt" and raw_pred.dim() == 2 and raw_pred.shape[-1] == 2:
                    point_pred = raw_pred[..., 0]
                else:
                    point_pred = raw_pred.squeeze(-1) if raw_pred.dim() > 1 else raw_pred
                val_losses.append(loss_fn(point_pred, yb_v).item())
        swa_val = float(np.mean(val_losses)) if val_losses else math.inf
        if swa_val + 1e-8 < best_val:
            # Copy SWA weights into model.state_dict (drop the AveragedModel "module." prefix).
            swa_sd = {k.replace("module.", "", 1): v.detach().cpu().clone()
                      for k, v in swa_model.module.state_dict().items()}
            model.load_state_dict(swa_sd)
            used_swa = True
            best_val = swa_val
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    return model, {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "train_time_s": float(train_time),
        "device": str(device),
        "used_swa": bool(used_swa),
    }


@torch.no_grad()
def predict_torch_regressor(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 256,
    boost_signals: np.ndarray | None = None,
) -> np.ndarray:
    """Run a forward pass on (N, p), returning a 1-D ndarray of predictions in the network's output scale."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n, p = X.shape
    X_t = torch.from_numpy(X.reshape(n, 1, p)).float().to(device)
    boost_t = None
    if boost_signals is not None:
        bs = np.asarray(boost_signals, dtype=float)
        if bs.ndim == 1:
            bs = bs[:, None]
        boost_t = torch.from_numpy(bs).float().to(device)
    out = []
    for i in range(0, n, batch_size):
        chunk = X_t[i : i + batch_size]
        if boost_t is not None:
            raw = model(chunk, boost_signal=boost_t[i : i + batch_size])
        else:
            raw = model(chunk)
        # Heteroscedastic 2-channel head: keep μ only.
        if raw.dim() == 2 and raw.shape[-1] == 2:
            point = raw[..., 0]
        else:
            point = raw.squeeze(-1) if raw.dim() > 1 else raw
        out.append(point.detach().cpu().numpy())
    return np.concatenate(out, axis=0)
