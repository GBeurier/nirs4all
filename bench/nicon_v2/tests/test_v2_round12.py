"""Round 12 — V3-AOMTransformer / V6-Distillation / V7-TTA smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nicon_v2.models.v2_aom_cnn import build_nicon_v2a
from nicon_v2.training import (
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    predict_torch_regressor,
    train_torch_regressor,
)


def _toy_spectra(n: int = 80, p: int = 196, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.3, size=p)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X @ w + rng.normal(scale=0.1, size=n)).astype(np.float32)
    return X, y


def _build_model_for(params: dict, p: int):
    return build_nicon_v2a(input_shape=(1, p), params=params)


def test_v3_aom_transformer_trunk_forward_and_train():
    """V3 transformer trunk produces (N,1) output and trains."""
    X, y = _toy_spectra(n=80, p=196)
    x_proc = StandardXProcessor().fit(X)
    y_proc = StandardYProcessor().fit(y)
    Xs = x_proc.transform(X)
    ys = y_proc.transform(y)

    params = {
        "bank": "extended_lowrank", "lowrank_rank": 32, "trainable_ops": True,
        "matrix_trainable_ops": True, "branch_se": True, "learnable_rms": True,
        "trunk_type": "hybrid_transformer",
        "transformer_d_model": 32, "transformer_heads": 2,
        "transformer_layers": 1, "transformer_ff": 64,
    }
    model = _build_model_for(params, p=196)
    model.fit_branches(torch.from_numpy(Xs.reshape(-1, 1, 196)).float())

    cfg = TrainConfig(epochs=10, patience=10, batch_size=16, lr=5e-3,
                      weight_decay=0.0, seed=0, use_amp=False)
    model, info = train_torch_regressor(model, Xs, ys, cfg)
    assert np.isfinite(info["best_val_loss"])
    pred = predict_torch_regressor(model, Xs)
    assert pred.shape == (80,)


def test_v6_distillation_loss_includes_teacher():
    """V6 — when teacher_predictions + distill_lambda are set, the loss function uses them."""
    X, y = _toy_spectra(n=80, p=128)
    x_proc = StandardXProcessor().fit(X)
    y_proc = StandardYProcessor().fit(y)
    Xs = x_proc.transform(X)
    ys = y_proc.transform(y)
    # Teacher = y itself in scaled space (perfect teacher) → distillation should accelerate fit.
    teacher = ys.copy()

    params = {
        "bank": "extended_lowrank", "lowrank_rank": 16, "trainable_ops": True,
        "matrix_trainable_ops": True, "branch_se": True, "learnable_rms": True,
    }
    cfg = TrainConfig(epochs=15, patience=15, batch_size=16, lr=5e-3,
                      weight_decay=0.0, seed=0, use_amp=False,
                      teacher_predictions=teacher, distill_lambda=1.0)
    model = _build_model_for(params, p=128)
    model.fit_branches(torch.from_numpy(Xs.reshape(-1, 1, 128)).float())
    model, info = train_torch_regressor(model, Xs, ys, cfg)
    assert np.isfinite(info["best_val_loss"])


def test_v6_distillation_disabled_when_lambda_zero():
    """V6 — when distill_lambda=0, the trainer ignores teacher_predictions (no extra loss term)."""
    X, y = _toy_spectra(n=64, p=128)
    x_proc = StandardXProcessor().fit(X)
    y_proc = StandardYProcessor().fit(y)
    Xs = x_proc.transform(X)
    ys = y_proc.transform(y)

    params = {
        "bank": "extended_lowrank", "lowrank_rank": 16, "trainable_ops": True,
        "matrix_trainable_ops": True, "branch_se": True, "learnable_rms": True,
    }
    cfg = TrainConfig(epochs=5, patience=5, batch_size=16, lr=5e-3,
                      weight_decay=0.0, seed=0, use_amp=False,
                      teacher_predictions=ys.copy(), distill_lambda=0.0)
    model = _build_model_for(params, p=128)
    model.fit_branches(torch.from_numpy(Xs.reshape(-1, 1, 128)).float())
    # Should NOT raise (the train loader has only 2 tensors per batch).
    model, info = train_torch_regressor(model, Xs, ys, cfg)
    assert np.isfinite(info["best_val_loss"])


def test_v7_tta_wrapper_averages_predictions():
    """V7 — _predict_with_tta returns shape-correct predictions averaged over K augmented copies."""
    import sys
    from pathlib import Path
    bench_root = Path(__file__).resolve().parent.parent
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))
    from benchmarks.run_baseline_benchmark import _predict_with_tta

    X, y = _toy_spectra(n=64, p=196)
    x_proc = StandardXProcessor().fit(X)
    y_proc = StandardYProcessor().fit(y)
    Xs = x_proc.transform(X)
    ys = y_proc.transform(y)

    params = {
        "bank": "extended_lowrank", "lowrank_rank": 16, "trainable_ops": True,
        "matrix_trainable_ops": True, "branch_se": True, "learnable_rms": True,
    }
    model = _build_model_for(params, p=196)
    model.fit_branches(torch.from_numpy(Xs.reshape(-1, 1, 196)).float())
    cfg = TrainConfig(epochs=3, patience=3, batch_size=16, lr=5e-3,
                      weight_decay=0.0, seed=0, use_amp=False)
    model, _ = train_torch_regressor(model, Xs, ys, cfg)

    device = next(model.parameters()).device
    pred = _predict_with_tta(model, Xs, device, tta_k=5, seed=0)
    assert pred.shape == (64,)
    assert np.all(np.isfinite(pred))
