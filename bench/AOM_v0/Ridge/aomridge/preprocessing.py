"""Fold-local feature standardisation for AOM-Ridge.

Provides a thin wrapper that fits per-feature mean/scale on the *training*
fold only and returns the centred / scaled matrix. The estimator uses it
for both the CV path and the final refit so the original-space coefficient
can be back-mapped consistently.

For ``x_scale="feature_std"``:

```text
x_mean = X.mean(axis=0)
x_scale = X.std(axis=0)
X_proc = (X - x_mean) / x_scale
```

For ``x_scale="feature_rms"``:

```text
x_mean = X.mean(axis=0)        # if center=True, else 0
x_scale = sqrt(mean((X - x_mean)^2, axis=0))
X_proc = (X - x_mean) / x_scale
```

Coefficient back-mapping: the dual Ridge yields ``beta_proc`` for the
processed inputs. The original-space coefficient is ``beta = beta_proc /
x_scale`` (component-wise), and the intercept is rebuilt from the original
mean.
"""

from __future__ import annotations

import numpy as np


def fit_feature_scaler(
    X: np.ndarray, mode: str, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_mean, x_scale)`` for the chosen mode.

    ``mode``:

    - ``"none"``: ``x_mean = 0``, ``x_scale = 1``.
    - ``"center"``: ``x_mean = X.mean(0)``, ``x_scale = 1``.
    - ``"feature_std"``: ``x_mean = X.mean(0)``, ``x_scale = X.std(0)``
      (biased std; matches ``StandardScaler`` defaults).
    - ``"feature_rms"``: ``x_mean = X.mean(0)``, ``x_scale = RMS of centered``.
    """
    n, p = X.shape
    if mode == "none":
        return np.zeros(p), np.ones(p)
    if mode == "center":
        return X.mean(axis=0), np.ones(p)
    if mode == "feature_std":
        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0, ddof=0)
        x_scale = np.where(x_scale > eps, x_scale, 1.0)
        return x_mean, x_scale
    if mode == "feature_rms":
        x_mean = X.mean(axis=0)
        Xc = X - x_mean
        x_scale = np.sqrt(np.mean(Xc * Xc, axis=0))
        x_scale = np.where(x_scale > eps, x_scale, 1.0)
        return x_mean, x_scale
    raise ValueError(f"unknown x_scale mode: {mode!r}")


def apply_feature_scaler(
    X: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray
) -> np.ndarray:
    return (X - x_mean) / x_scale
