"""Data loading utilities."""

import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from CSV files.

    Args:
        data_path: Path to directory containing X and y files.

    Returns:
        Tuple of (X, y) numpy arrays.

    Raises:
        FileNotFoundError: If data files not found.
    """
    patterns = [
        ("X.csv", "Y.csv"),
    ]

    for x_file, y_file in patterns:
        x_path = os.path.join(data_path, x_file)
        y_path = os.path.join(data_path, y_file)

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Loading data from {x_path}...")

            # Read with header detection
            x_df = pd.read_csv(x_path, header=None, sep=";")
            y_df = pd.read_csv(y_path, header=None, sep=";")

            # Check for headers
            try:
                float(x_df.iloc[0, 0])
            except (ValueError, TypeError):
                x_df = pd.read_csv(x_path, header=0, sep=";")

            try:
                float(y_df.iloc[0, 0])
            except (ValueError, TypeError):
                y_df = pd.read_csv(y_path, header=0, sep=";")

            X = x_df.values.astype(np.float64)
            y = y_df.values.astype(np.float64).ravel()

            # Handle shape mismatch
            min_samples = min(X.shape[0], y.shape[0])
            X, y = X[:min_samples], y[:min_samples]

            # Handle NaN
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[valid], y[valid]

            print(f"Loaded X: {X.shape}, y: {y.shape}")
            return X, y

    raise FileNotFoundError(f"Could not find data in {data_path}")
