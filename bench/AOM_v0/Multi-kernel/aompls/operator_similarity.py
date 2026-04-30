"""Operator similarity tools for the exploration / active-bank pipeline.

Two notions of similarity are supported:

- **Intrinsic**: probe-based response of operator on a deterministic basis
  (Diracs, multi-frequency cosines, low-degree polynomials, Gaussian peaks).
- **Dataset-dependent**: cosine similarity of `A S` vs `B S` for the current
  cross-covariance `S = X^T Y`.

Both are matrix-free: they call `op.apply_cov` / `op.transform` and never
materialize the explicit `p x p` matrix.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .operators import LinearSpectralOperator


def make_probe_basis(p: int, random_state: int = 0) -> np.ndarray:
    """Build a deterministic probe matrix of shape `(K, p)` covering common
    spectroscopic patterns: Diracs, low-degree polynomials, Gaussian peaks,
    and multi-frequency cosines.
    """
    rng = np.random.default_rng(random_state)
    rows: List[np.ndarray] = []
    # Diracs at evenly spaced positions
    for pos in np.linspace(p // 8, p - p // 8, 4, dtype=int):
        z = np.zeros(p)
        z[max(0, min(p - 1, int(pos)))] = 1.0
        rows.append(z)
    # Low-degree polynomials
    t = np.linspace(-1.0, 1.0, p)
    for d in range(4):
        rows.append(t**d)
    # Gaussian peaks at varying widths
    for sigma in (p / 30.0, p / 15.0, p / 8.0):
        for center in (p // 4, p // 2, 3 * p // 4):
            rows.append(np.exp(-((np.arange(p) - center) ** 2) / (2.0 * sigma * sigma)))
    # Multi-frequency cosines
    for freq in (1, 3, 7, 15):
        rows.append(np.cos(freq * np.pi * t))
        rows.append(np.sin(freq * np.pi * t))
    # White noise probe
    rows.append(rng.standard_normal(p))
    return np.stack(rows, axis=0)


def operator_response(op: LinearSpectralOperator, probe: np.ndarray) -> np.ndarray:
    """Apply `op` to each row of `probe` (treats rows as spectra).

    Returns a `(K, p)` matrix where row `i` is `op` applied to `probe[i]`.
    """
    op.fit(np.zeros((1, probe.shape[1])))
    return op.transform(probe)


def response_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Absolute cosine similarity between two arrays (flattened)."""
    av = np.asarray(a).ravel()
    bv = np.asarray(b).ravel()
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < eps or nb < eps:
        return 0.0
    return float(abs(av @ bv) / (na * nb))


def probe_gain(op: LinearSpectralOperator, probe: np.ndarray) -> float:
    """RMS amplification of `op` on the probe basis."""
    out = operator_response(op, probe)
    rms_in = float(np.sqrt(np.mean(probe * probe)))
    rms_out = float(np.sqrt(np.mean(out * out)))
    if rms_in < 1e-12:
        return 1.0
    return rms_out / rms_in


def keep_top_diverse(
    items: Sequence[tuple[float, np.ndarray, object]],
    top_m: int,
    cosine_threshold: float,
) -> List[tuple[float, np.ndarray, object]]:
    """Return up to `top_m` items, sorted by score (low first), pruning any
    item whose response cosine to a kept item exceeds `cosine_threshold`.

    Each item is a tuple `(score, response_vector, payload)`.
    """
    sorted_items = sorted(items, key=lambda t: t[0])
    kept: List[tuple[float, np.ndarray, object]] = []
    for sc, resp, payload in sorted_items:
        if len(kept) >= top_m:
            break
        if any(response_cosine(resp, k_resp) >= cosine_threshold for _, k_resp, _ in kept):
            continue
        kept.append((sc, resp, payload))
    return kept


def prune_by_intrinsic_similarity(
    operators: Sequence[LinearSpectralOperator],
    p: int,
    cosine_threshold: float = 0.995,
    random_state: int = 0,
) -> List[LinearSpectralOperator]:
    """Prune operators that produce nearly identical responses on the probe."""
    probe = make_probe_basis(p, random_state=random_state)
    items = []
    for op in operators:
        resp = operator_response(op, probe)
        items.append((0.0, resp, op))
    diverse = keep_top_diverse(items, top_m=len(items), cosine_threshold=cosine_threshold)
    return [payload for _, _, payload in diverse]
