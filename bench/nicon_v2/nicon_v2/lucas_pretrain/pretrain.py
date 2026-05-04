"""Pretrain V2A backbone on LUCAS-SOC.

Train a NiconV2A model on the LUCAS-SOC calibration set (4200 wavelengths,
SOC concentration target) and save the resulting state_dict to disk. Length-
invariant parameters (Conv kernels, branch SE, branch norms, head Linear)
will load cleanly into a target-domain model at any sequence length; length-
dependent parameters (LowRank Detrend/Whittaker U/V matrices, MSC mean) are
silently re-initialised by ``load_state_dict(strict=False)`` on the target.

Usage::

    PYTHONPATH=bench/nicon_v2:bench/AOM_v0:bench/AOM_v0/Ridge \\
      python -m nicon_v2.lucas_pretrain.pretrain \\
        --output bench/nicon_v2/checkpoints/lucas_v2l_5k.pt \\
        --n-subsample 5000 --epochs 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..models.v2_aom_cnn import build_nicon_v2a
from ..training import (
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    pick_device,
    predict_torch_regressor,
    set_global_seed,
    train_torch_regressor,
)
from .lucas_loader import DEFAULT_SUBSET, load_lucas_soc

DEFAULT_PARAMS = {
    "bank": "extended_lowrank",
    "trainable_ops": True,
    "operator_reg_lambda": 0.0,
    "branch_se": True,
    "lowrank_rank": 32,
    "learnable_rms": True,
    "trunk_channels": (32, 64, 96, 128),
    "trunk_kernels": (7, 5, 3, 3),
}


def _log_y_transform(y: np.ndarray) -> np.ndarray:
    """log1p — SOC has a heavy right tail (range 0-548 g/kg)."""
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def main() -> int:
    parser = argparse.ArgumentParser(description="LUCAS pretraining for V2A backbone")
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--n-subsample", type=int, default=5000,
                        help="random subsample of the calibration split (None = full)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True,
                        help="path to save the pretrained state_dict (e.g. lucas_v2l_5k.pt)")
    parser.add_argument("--log-target", action="store_true",
                        help="apply log1p to SOC before standardising (heavy right tail)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    device = pick_device("auto")
    print(f"[lucas-pretrain] device: {device}", flush=True)

    print(f"[lucas-pretrain] loading LUCAS subset {args.subset!r} (n_subsample={args.n_subsample}) ...", flush=True)
    t0 = time.time()
    X_cal, y_cal, X_val, y_val, wavelengths = load_lucas_soc(
        subset=args.subset, n_subsample=args.n_subsample, seed=args.seed,
    )
    print(f"[lucas-pretrain] loaded X_cal={X_cal.shape} X_val={X_val.shape} wl={wavelengths.shape} "
          f"(load_time={time.time() - t0:.1f}s)", flush=True)

    if args.log_target:
        y_cal = _log_y_transform(y_cal)
        y_val = _log_y_transform(y_val)
        print(f"[lucas-pretrain] log1p target: y_cal range=[{y_cal.min():.3f}, {y_cal.max():.3f}]", flush=True)

    x_proc = StandardXProcessor().fit(X_cal)
    y_proc = StandardYProcessor().fit(y_cal)
    X_cal_s = x_proc.transform(X_cal)
    X_val_s = x_proc.transform(X_val)
    y_cal_s = y_proc.transform(y_cal)
    y_val_s = y_proc.transform(y_val)

    n_features = X_cal_s.shape[1]
    print(f"[lucas-pretrain] building V2A at p={n_features} with params={DEFAULT_PARAMS}", flush=True)
    model = build_nicon_v2a(input_shape=(1, n_features), params=DEFAULT_PARAMS).to(device)
    model.fit_branches(torch.from_numpy(X_cal_s.reshape(-1, 1, n_features)).float().to(device))

    config = TrainConfig(
        seed=args.seed, device=device.type, batch_size=args.batch_size,
        lr=args.lr, epochs=args.epochs, patience=args.patience,
    )
    print(f"[lucas-pretrain] training: epochs={args.epochs} patience={args.patience} batch_size={args.batch_size}",
          flush=True)
    t0 = time.time()
    model, info = train_torch_regressor(model, X_cal_s, y_cal_s, config)
    print(f"[lucas-pretrain] done in {time.time() - t0:.1f}s "
          f"best_epoch={info['best_epoch']} best_val_loss={info['best_val_loss']:.4f}", flush=True)

    pred_val = predict_torch_regressor(model, X_val_s, device=device)
    rmse_scaled = float(np.sqrt(np.mean((pred_val - y_val_s) ** 2)))
    naive_rmse_scaled = float(np.sqrt(np.mean(y_val_s ** 2)))
    print(f"[lucas-pretrain] val RMSE (scaled): {rmse_scaled:.4f}  vs naive {naive_rmse_scaled:.4f} "
          f"(R²≈{1.0 - (rmse_scaled / max(naive_rmse_scaled, 1e-12)) ** 2:.3f})", flush=True)

    torch.save({
        "state_dict": model.state_dict(),
        "params": DEFAULT_PARAMS,
        "input_p": n_features,
        "n_train": int(X_cal_s.shape[0]),
        "subset": args.subset,
        "log_target": bool(args.log_target),
        "best_epoch": int(info["best_epoch"]),
        "val_rmse_scaled": rmse_scaled,
    }, args.output)
    print(f"[lucas-pretrain] wrote checkpoint to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
