"""Small user-owned joint networks and independent training oracle: tests only."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class JointInputs(nn.Module):
    """Two to four separately trained encoders, fused inside one forward call."""

    def __init__(self, input_shapes: Mapping[str, tuple[int, ...]], hidden_units: int = 3) -> None:
        super().__init__()
        self.names = tuple(input_shapes)
        self.encoders = nn.ModuleDict({
            name: nn.Linear(shape[0], hidden_units, dtype=torch.float32, device="cpu")
            for name, shape in input_shapes.items()
        })
        self.head = nn.Linear(hidden_units * len(self.names), 1, dtype=torch.float32, device="cpu")

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        assert set(inputs) == set(self.names)
        return self.head(torch.cat([torch.tanh(self.encoders[name](inputs[name])) for name in self.names], dim=1))


def joint_factory(*, input_shapes: Mapping[str, tuple[int, ...]], hidden_units: int = 3) -> JointInputs:
    """An ordinary user importable factory, not a product model family."""
    return JointInputs(input_shapes, hidden_units)


class _OracleJoint(nn.Module):
    """Independent declaration; never invokes the SDK adapter or trainer."""

    def __init__(self, names: tuple[str, ...], widths: Mapping[str, int], hidden: int) -> None:
        super().__init__()
        self.names = names
        self.encoders = nn.ModuleDict({name: nn.Linear(widths[name], hidden, device="cpu", dtype=torch.float32) for name in names})
        self.head = nn.Linear(hidden * len(names), 1, device="cpu", dtype=torch.float32)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        encoded = [torch.tanh(self.encoders[name](inputs[name])) for name in self.names]
        return self.head(torch.cat(encoded, dim=1))


def fit_reference(
    features: Mapping[str, np.ndarray], targets: np.ndarray, *, initial_state: Mapping[str, torch.Tensor],
    rng_after_init: torch.Tensor, params: Mapping[str, Any],
) -> nn.Module:
    """Fit fresh Torch Adam/MSE from the witnessed initial weights and RNG.

    The snapshots fix initialization and shuffling only. All gradient updates
    and predictions are independently computed with real Torch operations.
    """
    names = tuple(features)
    hidden = initial_state["head.weight"].shape[1] // len(names)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        reference = _OracleJoint(names, {name: features[name].shape[1] for name in names}, hidden)
        reference.load_state_dict({name: value.detach().clone() for name, value in initial_state.items()})
        torch.set_rng_state(rng_after_init)
        dataset = TensorDataset(
            *[torch.as_tensor(features[name], dtype=torch.float32, device="cpu") for name in names],
            torch.as_tensor(targets, dtype=torch.float32, device="cpu").reshape(-1, 1),
        )
        loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(reference.parameters(), lr=params["lr"])
        loss = nn.MSELoss(reduction="mean")
        for _ in range(params["epochs"]):
            reference.train()
            for *blocks, y in loader:
                optimizer.zero_grad()
                loss(reference(**dict(zip(names, blocks, strict=True))), y).backward()
                optimizer.step()
        reference.eval()
        return reference


def predict_reference(model: nn.Module, features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate the independent model on its separate named tensors."""
    with torch.no_grad():
        return model(**{name: torch.as_tensor(values, dtype=torch.float32, device="cpu") for name, values in features.items()}).numpy()
