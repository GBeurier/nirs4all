"""Regression and classification metrics used across AOM_v0."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (np.asarray(y_true) - np.asarray(y_pred)).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (np.asarray(y_true) - np.asarray(y_pred)).ravel()
    return float(np.mean(np.abs(diff)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = np.unique(y_true)
    accs = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        accs.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(accs)) if accs else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for cls in classes:
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def log_loss(y_true: np.ndarray, proba: np.ndarray, classes=None, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    if classes is None:
        classes = np.unique(y_true)
    cls_to_idx = {int(c): i for i, c in enumerate(classes)}
    losses = []
    for i, yi in enumerate(y_true):
        idx = cls_to_idx.get(int(yi))
        if idx is None:
            continue
        p = max(min(float(proba[i, idx]), 1.0 - eps), eps)
        losses.append(-np.log(p))
    return float(np.mean(losses)) if losses else float("nan")


def brier_score_binary(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba_pos = np.asarray(proba_pos).ravel()
    return float(np.mean((proba_pos - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (Naeini et al., 2015) on the predicted top class."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    correctness = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = float(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(np.mean(confidence[in_bin]))
        avg_acc = float(np.mean(correctness[in_bin]))
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)
