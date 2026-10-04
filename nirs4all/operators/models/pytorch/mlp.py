"""Small width-agnostic Torch factory for the closed CPU regression profile."""
from __future__ import annotations

from typing import Any


def structural_mlp(input_shape: tuple[int, ...], hidden_units: int = 8) -> Any:
    """Construct a fresh flat-input scalar regressor from existing Torch layers.

    The coordinator supplies the actual selected source width. This factory
    creates no fitted state and retains no samples, folds, or optimizer state.
    """
    import torch

    if len(input_shape) != 1 or type(input_shape[0]) is not int or input_shape[0] < 1:
        raise ValueError("structural_mlp requires a nonempty one-dimensional feature shape")
    if type(hidden_units) is not int or not 1 <= hidden_units <= 128:
        raise ValueError("structural_mlp hidden_units must be an integer in [1,128]")
    if (input_shape[0] + 1) * hidden_units + hidden_units + 1 > 1_000_000:
        raise ValueError("structural_mlp exceeds the signed CPU parameter budget")
    return torch.nn.Sequential(torch.nn.Flatten(start_dim=1), torch.nn.Linear(input_shape[0], hidden_units, device="cpu", dtype=torch.float32),
                               torch.nn.ReLU(), torch.nn.Linear(hidden_units, 1, device="cpu", dtype=torch.float32))
