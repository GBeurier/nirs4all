"""Actual fixed-shape modalities and an independent Torch oracle; tests only."""
from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class JointNativeShapes(nn.Module):
    """User architecture receives raw axes and explicitly flattens inside forward."""

    def __init__(self, input_shapes: Mapping[str, tuple[int, ...]], hidden_units: int = 3) -> None:
        super().__init__()
        self.shapes = dict(input_shapes)
        self.names = tuple(input_shapes)
        self.seen_shapes: list[dict[str, tuple[int, ...]]] = []
        self.encoders = nn.ModuleDict({name: nn.Linear(prod(shape), hidden_units, dtype=torch.float32, device="cpu")
                                       for name, shape in input_shapes.items()})
        self.head = nn.Linear(hidden_units * len(self.names), 1, dtype=torch.float32, device="cpu")

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        assert set(inputs) == set(self.names)
        observed = {name: tuple(inputs[name].shape[1:]) for name in self.names}
        assert observed == self.shapes, "the host changed raw image/series axes before forward"
        self.seen_shapes.append(observed)
        encodings = [torch.tanh(self.encoders[name](inputs[name].flatten(start_dim=1))) for name in self.names]
        return self.head(torch.cat(encodings, dim=1))


def joint_factory(*, input_shapes: Mapping[str, tuple[int, ...]], hidden_units: int = 3) -> JointNativeShapes:
    return JointNativeShapes(input_shapes, hidden_units)


class _Reference(nn.Module):
    """Independent functional forward, never using product or fixture forward."""

    def __init__(self, features: Mapping[str, np.ndarray], hidden: int) -> None:
        super().__init__()
        self.names = tuple(features)
        self.encoders = nn.ModuleDict({name: nn.Linear(prod(values.shape[1:]), hidden, dtype=torch.float32, device="cpu")
                                       for name, values in features.items()})
        self.head = nn.Linear(hidden * len(self.names), 1, dtype=torch.float32, device="cpu")

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        encodings = []
        for name in self.names:
            raw = inputs[name].reshape(inputs[name].shape[0], -1)
            layer = self.encoders[name]
            encodings.append(F.linear(raw, layer.weight, layer.bias).tanh())
        return F.linear(torch.cat(encodings, dim=1), self.head.weight, self.head.bias)


def fit_reference(
    features: Mapping[str, np.ndarray], targets: np.ndarray, *, initial_state: Mapping[str, torch.Tensor],
    rng_after_init: torch.Tensor, params: Mapping[str, Any],
) -> nn.Module:
    names = tuple(features)
    hidden = initial_state["head.weight"].shape[1] // len(names)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        reference = _Reference(features, hidden)
        reference.load_state_dict({name: tensor.detach().clone() for name, tensor in initial_state.items()})
        torch.set_rng_state(rng_after_init)
        tensors = [torch.as_tensor(features[name], dtype=torch.float32, device="cpu") for name in names]
        loader = DataLoader(TensorDataset(*tensors, torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)),
                            batch_size=params["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(reference.parameters(), lr=params["lr"])
        for _ in range(params["epochs"]):
            reference.train()
            for *blocks, target in loader:
                optimizer.zero_grad()
                prediction = reference(**dict(zip(names, blocks, strict=True)))
                F.mse_loss(prediction, target, reduction="mean").backward()
                optimizer.step()
        return reference.eval()


def predict_reference(model: nn.Module, features: Mapping[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        return model(**{name: torch.as_tensor(value, dtype=torch.float32, device="cpu") for name, value in features.items()}).numpy()
