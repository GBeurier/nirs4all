"""Metrics for nicon_v2 benchmark runs.

All metrics operate on the **original (un-scaled) y** scale; the y-processing
inverse transform is the responsibility of the model wrapper.
"""

from __future__ import annotations

import math

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch rmse: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch mae: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_pred - y_true))


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean per-sample Gaussian NLL with σ²-clipped lower bound for stability."""
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if y_true.size == 0:
        return float("nan")
    sigma = np.clip(sigma, 1e-6, None)
    z = (y_true - mu) / sigma
    return float(0.5 * np.mean(z ** 2 + 2.0 * np.log(sigma) + math.log(2.0 * math.pi)))


def coverage_at_alpha(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def width_at_alpha(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.size == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def relative_rmsep(rmsep_value: float, ref: float | None) -> float | None:
    if ref is None or not (ref > 0):
        return None
    return float((rmsep_value - ref) / ref)
