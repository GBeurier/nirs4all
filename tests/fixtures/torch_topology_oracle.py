"""Independent real Torch/Ridge oracle for sequential structural DL tests.

Callers supply train-only encoded features, aligned OOF matrices, and the exact
native-task seed. This module neither discovers folds nor imports production
factories, controllers, DAG adapters, or topology code. CPU execution is explicit;
the oracle does not change Torch's process-wide numerical thread configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def _features(values: Any, *, dtype: Any = np.float32) -> np.ndarray:
    """Copy a finite, nonempty feature matrix without changing caller arrays."""
    matrix = np.array(values, dtype=dtype, order="C", copy=True)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Oracle features must be a nonempty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Oracle features must contain only finite values")
    return matrix


def _targets(values: Any, rows: int, *, dtype: Any = np.float32) -> np.ndarray:
    """Copy mono-target regression labels into an explicit N×1 matrix."""
    labels = np.array(values, dtype=dtype, order="C", copy=True)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.shape != (rows, 1) or not np.isfinite(labels).all():
        raise ValueError("Oracle targets must be finite and have shape N×1")
    return labels


def fit_torch(params: dict[str, Any], X_train: Any, y_train: Any, *, seed: int) -> torch.nn.Module:
    """Fit a fresh CPU float32 MLP using the declared number of epochs.

    Args:
        params: Flat DagMLTorchEstimator declaration: factory_params.hidden_units,
            epochs, batch_size, and lr. Optional optimizer/loss/device/layout/task
            fields must match Adam/MSELoss/cpu/2d/regression. Patience is ignored:
            this oracle has no validation split or early stopping. learning_rate
            must be absent or None; lr is the only active learning-rate field.
        X_train: Already encoded training features, never validation/test rows.
        y_train: Regression labels in shape N or N×1.
        seed: Explicit uint32 native task seed; no fold/variant seed derivation.

    Returns:
        The actual trained torch.nn.Sequential module in evaluation mode.

    The CPU RNG state is restored on normal exit and on failure. Initial module
    weights and DataLoader shuffling share the seeded global CPU RNG, in that
    order. No separate sampler generator or GPU RNG mutation is introduced.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    fixed = {"optimizer": "Adam", "loss": "MSELoss", "device": "cpu", "force_layout": "2d", "task_type": "regression"}
    for name, expected in fixed.items():
        if name in params and params[name] != expected:
            raise ValueError(f"Oracle requires {name}={expected!r}")
    if params.get("learning_rate") is not None:
        raise ValueError("Oracle uses lr; learning_rate must be None or absent")
    hidden = params["factory_params"]["hidden_units"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    for name, value, maximum in (("hidden_units", hidden, 128), ("epochs", epochs, 100), ("batch_size", batch_size, 1024)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"Oracle {name} must be an integer in 1..{maximum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Oracle seed must be an explicit uint32 native task seed")
    lr = float(params["lr"])
    if not np.isfinite(lr) or not 1e-6 <= lr <= 0.1:
        raise ValueError("Oracle lr must lie in 1e-6..0.1")
    features = _features(X_train)
    labels = _targets(y_train, len(features))
    inputs = torch.tensor(features, dtype=torch.float32, device="cpu")
    targets = torch.tensor(labels, dtype=torch.float32, device="cpu")

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        # Seed only CPU: torch.manual_seed would also mutate accelerator RNGs.
        torch.random.default_generator.manual_seed(seed)
        model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(features.shape[1], hidden, dtype=torch.float32, device="cpu"),
            nn.ReLU(),
            nn.Linear(hidden, 1, dtype=torch.float32, device="cpu"),
        )
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss(reduction="mean")
        model.train()
        for _ in range(epochs):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_inputs), batch_targets)
                loss.backward()
                optimizer.step()
        model.eval()
    return model


def predict_torch(model: torch.nn.Module, X: Any) -> np.ndarray:
    """Return actual CPU model predictions as a float32 N×1 array."""
    import torch

    features = _features(X)
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("Oracle model parameters must be CPU float32")
    inputs = torch.tensor(features, dtype=torch.float32, device="cpu")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model(inputs).detach().cpu().numpy().copy()
    finally:
        model.train(was_training)
    if result.shape != (len(features), 1) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("Oracle model predictions must be finite float32 N×1")
    return result


def fit_predict_torch(params: dict[str, Any], X_train: Any, y_train: Any, X_predict: Any, *, seed: int) -> np.ndarray:
    """Fit a fresh real Torch model and predict without reusing another fit."""
    return predict_torch(fit_torch(params, X_train, y_train, seed=seed), X_predict)


def fit_ridge_predict(X_oof: Any, y_train: Any, X_predict: Any, *, alpha: float) -> np.ndarray:
    """Fit sklearn Ridge with intercept/SVD on caller-aligned OOF rows.

    This helper does not generate folds, join IDs, select alpha, or use in-sample
    branch predictions. The caller selects an ordered candidate alpha and passes
    the correctly aligned OOF training matrix and held-out prediction columns.
    """
    from sklearn.linear_model import Ridge

    value = float(alpha)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Oracle Ridge alpha must be finite and nonnegative")
    train = _features(X_oof, dtype=np.float64)
    labels = _targets(y_train, len(train), dtype=np.float64)
    predict = _features(X_predict, dtype=np.float64)
    if predict.shape[1] != train.shape[1]:
        raise ValueError("Oracle Ridge train/predict column counts must match")
    model = Ridge(alpha=value, fit_intercept=True, solver="svd")
    model.fit(train, labels)
    result = np.asarray(model.predict(predict), dtype=np.float64).reshape(-1, 1)
    if result.shape != (len(predict), 1) or not np.isfinite(result).all():
        raise ValueError("Oracle Ridge predictions must be finite N×1")
    return result
