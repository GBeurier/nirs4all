"""Explicit missing-source prediction policy shared by live and archive hosts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def predict_present_rows(predict: Callable[[Any], Any], block: Any, presence: np.ndarray, output_width: int) -> np.ndarray:
    """Predict observed rows in original target space; absent rows remain zero.

    The callable must include any captured inverse target transform. No encoder
    or model is called when this source is absent for every requested row.
    """
    mask = np.asarray(presence)
    if mask.dtype.kind != "b" or mask.shape != (len(block),):
        raise ValueError("source presence must be a boolean mask aligned with the raw rows")
    if type(output_width) is not int or output_width < 1:
        raise ValueError("source prediction width must be a positive integer")
    values = np.zeros((len(mask), output_width), dtype=float)
    if mask.any():
        observed = block if mask.all() else block.take_rows(mask) if hasattr(block, "take_rows") else block[mask]
        prediction = np.asarray(predict(observed), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        if prediction.shape != (int(mask.sum()), output_width) or not np.isfinite(prediction).all():
            raise ValueError("observed source predictions must be finite and match the declared target width")
        values[mask] = prediction
    return values


def append_source_presence(predictions: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Append one declared availability feature to one source's predictions."""
    values, mask = np.asarray(predictions, dtype=float), np.asarray(presence)
    if values.ndim != 2 or mask.dtype.kind != "b" or mask.shape != (len(values),):
        raise ValueError("source predictions and boolean presence must have aligned rows")
    if not np.isfinite(values).all() or np.any(values[~mask] != 0):
        raise ValueError("missing-source predictions must be zero and observed predictions finite")
    return np.column_stack([values, mask.astype(float)])
