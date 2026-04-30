"""Synthetic data generators for mkR / MKM / BLUP tests.

Three regimes:

- **R1 (oracle)**: 1 active block of B, high SNR.
- **R2 (mixture)**: 3 active blocks, moderate SNR, mixed magnitudes.
- **R3 (correlated)**: 2 quasi-identical blocks (alignment > 0.95).

Each generator returns (X, y, true_eta, sigma_e2, ops, diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aompls.banks import bank_by_name


@dataclass(frozen=True)
class SyntheticDataset:
    X: np.ndarray
    y: np.ndarray
    true_eta: np.ndarray
    sigma_e2: float
    operator_bank: str
    snr: float
    seed: int


def _build_smooth_X(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth-ish spectra: Gaussian-bump mixture + small noise."""
    centres = rng.uniform(0.1 * p, 0.9 * p, size=(n, 5))
    widths = rng.uniform(0.02 * p, 0.08 * p, size=(n, 5))
    amps = rng.normal(0, 1.0, size=(n, 5))
    grid = np.arange(p, dtype=float)
    X = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(5):
            X[i] += amps[i, k] * np.exp(-((grid - centres[i, k]) ** 2) / (2 * widths[i, k] ** 2))
    X += rng.normal(0, 0.05, size=(n, p))
    return X


def make_R1(
    n: int = 200,
    p: int = 400,
    snr: float = 5.0,
    seed: int = 0,
) -> SyntheticDataset:
    """1 active block, high SNR."""
    rng = np.random.default_rng(seed)
    X = _build_smooth_X(n, p, rng)
    Xc = X - X.mean(axis=0)
    ops = bank_by_name("compact", p=p)
    B = len(ops)
    # Pick an identifiable compact-bank block. The first non-identity
    # smoother is almost collinear with identity on smooth spectra, which
    # makes variance recovery non-identifiable for MKM.
    active_idx = min(5, B - 1)
    op = ops[active_idx]
    op.fit(Xc)
    Z = op.transform(Xc)               # (n, p)
    beta_active = rng.normal(0, 1.0, size=p)
    signal = Z @ beta_active
    signal_var = float(np.var(signal))
    sigma_e2 = signal_var / max(snr, 1e-6)
    e = rng.normal(0, np.sqrt(sigma_e2), size=n)
    y = signal + e
    true_eta = np.zeros(B, dtype=float)
    true_eta[active_idx] = 1.0
    return SyntheticDataset(
        X=X.astype(float),
        y=y.astype(float),
        true_eta=true_eta,
        sigma_e2=float(sigma_e2),
        operator_bank="compact",
        snr=float(snr),
        seed=int(seed),
    )


def make_R2(
    n: int = 200,
    p: int = 400,
    snr: float = 3.0,
    seed: int = 1,
) -> SyntheticDataset:
    """3 active blocks, moderate SNR, mixed magnitudes."""
    rng = np.random.default_rng(seed)
    X = _build_smooth_X(n, p, rng)
    Xc = X - X.mean(axis=0)
    ops = bank_by_name("compact", p=p)
    B = len(ops)
    active = [min(1, B - 1), min(2, B - 1), min(3, B - 1)]
    weights_truth = np.array([0.5, 0.3, 0.2])
    signal = np.zeros(n, dtype=float)
    for w, idx in zip(weights_truth, active, strict=False):
        op = ops[idx]
        op.fit(Xc)
        Z = op.transform(Xc)
        beta = rng.normal(0, 1.0, size=p)
        signal += w * (Z @ beta)
    signal_var = float(np.var(signal))
    sigma_e2 = signal_var / max(snr, 1e-6)
    e = rng.normal(0, np.sqrt(sigma_e2), size=n)
    y = signal + e
    true_eta = np.zeros(B, dtype=float)
    for w, idx in zip(weights_truth, active, strict=False):
        true_eta[idx] += float(w)
    true_eta /= true_eta.sum()
    return SyntheticDataset(
        X=X.astype(float),
        y=y.astype(float),
        true_eta=true_eta,
        sigma_e2=float(sigma_e2),
        operator_bank="compact",
        snr=float(snr),
        seed=int(seed),
    )


def make_R3(
    n: int = 200,
    p: int = 400,
    snr: float = 3.0,
    seed: int = 2,
) -> SyntheticDataset:
    """Two quasi-identical blocks (high alignment) + one independent.

    We mimic this by creating two synthetic operators that differ only by
    a small perturbation; we use the compact bank's identity and savgol_d1
    plus a hand-built tiny perturbation of identity (the 'noisy_identity'
    via custom op is overkill; we approximate by reusing identity twice).

    For simplicity, R3 sets both true and recovered weights to test that
    sum of "near-duplicate" weights matches truth. Implementation note:
    the compact bank typically has identity as block 0 and savgol_d1 as
    block 1 (sufficiently different); to actually create R3 we would
    inject a hand-built duplicate. For v1 we mock R3 by constructing a
    bank with [identity, identity_clone, savgol_d1, ...]. The clone is
    indistinguishable from the original, so K_0 ≈ K_1.

    Returns a dataset where the duplicate pair shares the active variance.
    """
    from aompls.operators import IdentityOperator, SavitzkyGolayOperator

    rng = np.random.default_rng(seed)
    X = _build_smooth_X(n, p, rng)
    Xc = X - X.mean(axis=0)
    op_a = IdentityOperator(p=p)
    op_b = IdentityOperator(p=p)  # duplicate of identity
    op_c = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1, p=p)
    ops = [op_a, op_b, op_c]
    for op in ops:
        op.fit(Xc)
    # Active = identity-pair; we use op_a's transform for the signal.
    Z = op_a.transform(Xc)
    beta = rng.normal(0, 1.0, size=p)
    signal = Z @ beta
    signal_var = float(np.var(signal))
    sigma_e2 = signal_var / max(snr, 1e-6)
    e = rng.normal(0, np.sqrt(sigma_e2), size=n)
    y = signal + e
    # The "truth" splits weight 0.5/0.5 between op_a and op_b (they are
    # indistinguishable); op_c gets 0.
    true_eta = np.array([0.5, 0.5, 0.0], dtype=float)
    return SyntheticDataset(
        X=X.astype(float),
        y=y.astype(float),
        true_eta=true_eta,
        sigma_e2=float(sigma_e2),
        operator_bank="r3_pair",  # sentinel: caller must use ops directly
        snr=float(snr),
        seed=int(seed),
    )
