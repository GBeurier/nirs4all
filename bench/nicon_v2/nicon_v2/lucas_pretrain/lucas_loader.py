"""Loader for LUCAS soil-spectroscopy data.

The LUCAS data lives at::

    /home/delete/NIRS DB/regression/LUCAS/<dataset>/{Xcal,Xval,Ycal,Yval}.csv

X files: 4200 wavelengths (400-2499.5 nm at 0.5 nm), values are absorbance,
delimiter=";". First row is the wavelength header.

Y files: single column "x", target value (e.g. SOC concentration).

This module provides a single ``load_lucas_soc`` function that returns the
calibration + validation arrays as float32 numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LUCAS_ROOT = Path("/home/delete/NIRS DB/regression/LUCAS")
DEFAULT_SUBSET = "LUCAS_SOC_all_26650_NocitaKS"


def _read_x(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=";", header=0, decimal=".")
    wavelengths = np.asarray(df.columns.astype(float), dtype=np.float32)
    X = df.to_numpy(dtype=np.float32)
    return X, wavelengths


def _read_y(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=0)
    # Single column, name is "x" but use the only available column robustly.
    return df.iloc[:, 0].to_numpy(dtype=np.float32)


def load_lucas_soc(
    subset: str = DEFAULT_SUBSET,
    *,
    n_subsample: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_cal, y_cal, X_val, y_val, wavelengths).

    X arrays are (N, p) float32; y arrays are (N,) float32; wavelengths
    is (p,) float32. If ``n_subsample`` is given, randomly sample that
    many rows from the calibration set (validation is kept full).
    """
    root = LUCAS_ROOT / subset
    X_cal, wavelengths = _read_x(root / "Xcal.csv")
    X_val, _ = _read_x(root / "Xval.csv")
    y_cal = _read_y(root / "Ycal.csv")
    y_val = _read_y(root / "Yval.csv")
    if X_cal.shape[0] != y_cal.shape[0]:
        raise RuntimeError(f"X_cal/y_cal length mismatch: {X_cal.shape} vs {y_cal.shape}")
    if X_val.shape[0] != y_val.shape[0]:
        raise RuntimeError(f"X_val/y_val length mismatch: {X_val.shape} vs {y_val.shape}")

    if n_subsample is not None and n_subsample < X_cal.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X_cal.shape[0])[:n_subsample]
        X_cal = X_cal[idx]
        y_cal = y_cal[idx]

    return X_cal, y_cal, X_val, y_val, wavelengths
