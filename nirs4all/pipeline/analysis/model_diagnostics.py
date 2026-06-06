"""Model diagnostics over cross-validation predictions.

Pure numerical analyses used by result-inspection UIs:

- :func:`bias_variance_decomposition` — per-sample bias²/variance over
  repeated CV predictions of the same samples.
- :func:`robustness_axes` — normalized multi-axis robustness profile
  (CV stability, train/val gap, absolute score, fold coverage) across a
  set of chains.
- :func:`learning_curve_points` — train/val score aggregation by training-set
  size, and :func:`estimate_train_size` for the fold-based approximation when
  the exact size is unavailable.

These previously lived in the nirs4all-studio HTTP layer (flagged as a
boundary violation in its 2026-06-05 tech-debt closeout, INS-04). They take
plain numbers/sequences — callers assemble the data from the workspace store.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BiasVarianceDecomposition",
    "bias_variance_decomposition",
    "estimate_train_size",
    "learning_curve_points",
    "robustness_axes",
]


# =========================================================================
# Bias-variance decomposition
# =========================================================================


@dataclass
class BiasVarianceDecomposition:
    """Aggregated bias²/variance over samples with repeated predictions.

    Attributes:
        bias_squared: Mean of per-sample ``(mean_pred - y_true)²``.
        variance: Mean of per-sample ``Var(y_pred)``.
        total_error: ``bias_squared + variance``.
        n_samples: Number of samples with >= 2 predictions that contributed.
    """

    bias_squared: float
    variance: float
    total_error: float
    n_samples: int


def bias_variance_decomposition(
    sample_predictions: Mapping[Hashable, Sequence[tuple[float, float]]],
) -> BiasVarianceDecomposition | None:
    """Decompose repeated-prediction error into bias² and variance.

    For every sample observed in two or more folds/chains:
    ``bias² = (mean_pred - y_true)²`` and ``variance = Var(y_pred)``;
    the result aggregates both as means over contributing samples.

    Args:
        sample_predictions: Mapping of sample key to its ``(y_true, y_pred)``
            pairs collected across folds/chains. Non-finite pairs are ignored.

    Returns:
        The aggregated decomposition, or ``None`` when no sample has at least
        two finite predictions.
    """
    biases_sq: list[float] = []
    variances: list[float] = []

    for pairs in sample_predictions.values():
        finite_pairs = [
            (yt, yp)
            for yt, yp in pairs
            if not (math.isnan(yt) or math.isnan(yp) or math.isinf(yt) or math.isinf(yp))
        ]
        if len(finite_pairs) < 2:
            continue
        y_true_val = finite_pairs[0][0]
        preds = [yp for _, yp in finite_pairs]
        mean_pred = float(np.mean(preds))
        biases_sq.append((mean_pred - y_true_val) ** 2)
        variances.append(float(np.var(preds)))

    if not biases_sq:
        return None

    mean_bias_sq = float(np.mean(biases_sq))
    mean_var = float(np.mean(variances))
    return BiasVarianceDecomposition(
        bias_squared=mean_bias_sq,
        variance=mean_var,
        total_error=mean_bias_sq + mean_var,
        n_samples=len(biases_sq),
    )


# =========================================================================
# Robustness profile
# =========================================================================


def robustness_axes(
    chains: Sequence[Mapping[str, Any]],
    *,
    lower_better: bool = False,
) -> list[dict[str, dict[str, float]]]:
    """Compute the normalized robustness axes for a set of chains.

    Each input mapping provides the raw ingredients per chain:
    ``fold_scores`` (per-fold scores for the inspected partition),
    ``train_score``, ``val_score``, ``score`` (the headline score) and
    ``fold_count``. Axes are normalized 0-1 across the input set
    (higher = more robust):

    - ``cv_stability``: ``1 - std(fold_scores)/max_std``
    - ``train_test_gap``: ``1 - |train - val|/max_gap``
    - ``score_absolute``: min-max normalized headline score (flipped when
      *lower_better*)
    - ``fold_count_ratio``: ``fold_count / max_fold_count``

    Returns:
        One dict per input chain (same order), mapping axis name to
        ``{"value": normalized, "raw": raw_value}``.
    """
    if not chains:
        return []

    raw: list[dict[str, float]] = []
    for chain in chains:
        fold_scores = [float(s) for s in (chain.get("fold_scores") or []) if s is not None]
        cv_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        train_score = chain.get("train_score")
        val_score = chain.get("val_score")
        gap = abs(train_score - val_score) if train_score is not None and val_score is not None else 0.0
        score = chain.get("score")
        abs_score = float(score) if score is not None else 0.0
        fold_count = float(chain.get("fold_count") or 0)
        raw.append({"cv_std": cv_std, "gap": gap, "abs_score": abs_score, "fold_count": fold_count})

    max_std = max(d["cv_std"] for d in raw) or 1.0
    max_gap = max(d["gap"] for d in raw) or 1.0
    max_folds = max(d["fold_count"] for d in raw) or 1.0
    scores = [d["abs_score"] for d in raw]
    score_min, score_max = min(scores), max(scores)
    score_range = (score_max - score_min) or 1.0

    profiles: list[dict[str, dict[str, float]]] = []
    for d in raw:
        norm_score = (d["abs_score"] - score_min) / score_range
        if lower_better:
            norm_score = 1.0 - norm_score
        profiles.append({
            "cv_stability": {"value": 1.0 - (d["cv_std"] / max_std if max_std > 0 else 0.0), "raw": d["cv_std"]},
            "train_test_gap": {"value": 1.0 - (d["gap"] / max_gap if max_gap > 0 else 0.0), "raw": d["gap"]},
            "score_absolute": {"value": norm_score, "raw": d["abs_score"]},
            "fold_count_ratio": {"value": d["fold_count"] / max_folds if max_folds > 0 else 0.0, "raw": d["fold_count"]},
        })
    return profiles


# =========================================================================
# Learning curve
# =========================================================================


def estimate_train_size(val_size: int, fold_count: int) -> int:
    """Approximate the training-set size from a validation-fold size.

    Assumes K-fold style splitting: ``total ≈ val_size * K / (K - 1)`` and
    ``train ≈ total - val_size``. Used when the exact per-fold training size
    is not recorded alongside predictions.

    Args:
        val_size: Number of validation samples in one fold.
        fold_count: Number of CV folds (values < 2 are treated as 5).

    Returns:
        The estimated training-set size (0 when *val_size* is not positive).
    """
    if val_size <= 0:
        return 0
    k = fold_count if fold_count and fold_count > 1 else 5
    total_approx = int(val_size * k / max(1, k - 1))
    return total_approx - val_size


def learning_curve_points(
    size_scores: Mapping[int, Sequence[Mapping[str, float | None]]],
) -> list[dict[str, Any]]:
    """Aggregate train/val scores per training-set size.

    Args:
        size_scores: Mapping of training size to entries with optional
            ``train`` and ``val`` scores.

    Returns:
        Points sorted by training size, each with ``train_size``,
        ``train_mean``/``train_std``, ``val_mean``/``val_std`` (``None`` when
        insufficient data; std requires >= 2 values) and ``count``.
    """
    points: list[dict[str, Any]] = []
    for size in sorted(size_scores.keys()):
        entries = size_scores[size]
        train_vals = [v for e in entries if (v := e.get("train")) is not None]
        val_vals = [v for e in entries if (v := e.get("val")) is not None]
        points.append({
            "train_size": size,
            "train_mean": float(np.mean(train_vals)) if train_vals else None,
            "train_std": float(np.std(train_vals)) if len(train_vals) > 1 else None,
            "val_mean": float(np.mean(val_vals)) if val_vals else None,
            "val_std": float(np.std(val_vals)) if len(val_vals) > 1 else None,
            "count": len(entries),
        })
    return points
