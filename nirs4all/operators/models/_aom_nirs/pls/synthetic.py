"""Deterministic synthetic data generators for tests and microbenchmarks.

The generators produce reproducible regression and classification toy spectra
with known structural variations: smooth baseline, oscillatory bands, scatter,
and chemical signatures. They are not meant to model real NIRS data faithfully
but to exercise the operator-adaptive PLS engines under controlled conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SyntheticDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    wavelengths: np.ndarray
    name: str = "synthetic"


def make_regression(
    n_train: int = 80,
    n_test: int = 40,
    p: int = 200,
    n_targets: int = 1,
    noise: float = 0.05,
    random_state: int = 0,
) -> SyntheticDataset:
    """Generate a deterministic regression dataset with smooth structured spectra."""
    rng = np.random.default_rng(random_state)
    n = n_train + n_test
    wavelengths = np.linspace(900.0, 1700.0, p)
    # True latent factors
    base1 = np.exp(-((wavelengths - 1100.0) ** 2) / (60.0**2))
    base2 = np.exp(-((wavelengths - 1450.0) ** 2) / (40.0**2))
    base3 = 0.5 * np.sin(wavelengths / 60.0)
    factors = np.stack([base1, base2, base3], axis=0)  # (3, p)
    # Sample concentrations
    Csig = rng.standard_normal((n, 3))
    X = Csig @ factors  # (n, p)
    # Add baseline drift (low-frequency polynomial)
    t = np.linspace(-1.0, 1.0, p)
    drift_coeffs = rng.standard_normal((n, 3))
    drift = drift_coeffs @ np.stack([np.ones_like(t), t, t**2], axis=0)
    X = X + 0.3 * drift
    # Add scatter (multiplicative)
    scatter = 1.0 + 0.05 * rng.standard_normal(n)
    X = X * scatter[:, None]
    # Add small noise
    X = X + noise * rng.standard_normal(X.shape)
    # Targets are linear in the chemistry concentrations
    if n_targets == 1:
        weights = np.array([1.5, -0.7, 0.3])
        y = Csig @ weights + 0.05 * rng.standard_normal(n)
        y = y.reshape(-1, 1)
    else:
        weights = rng.standard_normal((3, n_targets))
        y = Csig @ weights + 0.05 * rng.standard_normal((n, n_targets))
    return SyntheticDataset(
        X_train=X[:n_train],
        X_test=X[n_train:],
        y_train=y[:n_train].ravel() if n_targets == 1 else y[:n_train],
        y_test=y[n_train:].ravel() if n_targets == 1 else y[n_train:],
        wavelengths=wavelengths,
        name=f"synthetic_regression_n{n_train}_p{p}",
    )


def make_classification(
    n_train: int = 90,
    n_test: int = 60,
    p: int = 180,
    n_classes: int = 3,
    noise: float = 0.05,
    random_state: int = 0,
) -> SyntheticDataset:
    """Generate a deterministic classification dataset with separable classes."""
    rng = np.random.default_rng(random_state)
    n = n_train + n_test
    wavelengths = np.linspace(900.0, 1700.0, p)
    centers = np.linspace(1000.0, 1600.0, n_classes)
    sigmas = np.linspace(60.0, 30.0, n_classes)
    factors = np.stack(
        [np.exp(-((wavelengths - c) ** 2) / (s**2)) for c, s in zip(centers, sigmas)],
        axis=0,
    )
    classes = rng.integers(0, n_classes, size=n)
    Csig = np.zeros((n, n_classes))
    for i, cls in enumerate(classes):
        Csig[i, cls] = 1.0 + 0.1 * rng.standard_normal()
        for other in range(n_classes):
            if other != cls:
                Csig[i, other] = 0.05 * rng.standard_normal()
    X = Csig @ factors
    # Add baseline drift
    t = np.linspace(-1.0, 1.0, p)
    drift_coeffs = rng.standard_normal((n, 3))
    drift = drift_coeffs @ np.stack([np.ones_like(t), t, t**2], axis=0)
    X = X + 0.2 * drift
    X = X + noise * rng.standard_normal(X.shape)
    # Permute deterministically so train/test contains all classes
    order = rng.permutation(n)
    X = X[order]
    classes = classes[order]
    return SyntheticDataset(
        X_train=X[:n_train],
        X_test=X[n_train:],
        y_train=classes[:n_train].astype(int),
        y_test=classes[n_train:].astype(int),
        wavelengths=wavelengths,
        name=f"synthetic_classification_n{n_train}_p{p}_c{n_classes}",
    )


def small_pls1_dataset(p: int = 24, n: int = 30, K: int = 3, noise: float = 0.02, random_state: int = 0):
    """Tiny PLS1 dataset: y is a linear combination of the first `K` PLS factors."""
    rng = np.random.default_rng(random_state)
    n_total = n + 10
    base = np.linspace(-1.0, 1.0, p)
    factors = []
    for k in range(K):
        factors.append(np.cos((k + 1) * np.pi * base))
    factors_arr = np.stack(factors, axis=0)
    C = rng.standard_normal((n_total, K))
    X = C @ factors_arr
    X = X + noise * rng.standard_normal(X.shape)
    weights = rng.standard_normal(K)
    y = C @ weights + noise * rng.standard_normal(n_total)
    return X[:n], y[:n], X[n:], y[n:]
