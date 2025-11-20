"""
Binning utilities for regression target values.

This module provides utilities to bin continuous regression targets
into discrete classes for balanced augmentation.
"""
from typing import Tuple
import numpy as np


class BinningCalculator:
    """Calculate bins for continuous regression targets."""

    @staticmethod
    def bin_continuous_targets(
        y: np.ndarray,
        bins: int = 10,
        strategy: str = "equal_width"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin continuous target values into discrete classes.

        Args:
            y: Continuous target values (1D array)
            bins: Number of bins (1-1000). Default: 10
            strategy: "quantile" (equal probability) or "equal_width" (uniform spacing, default)

        Returns:
            Tuple of (bin_indices, bin_edges)
            - bin_indices: 0-based bin index for each sample
            - bin_edges: Edge values defining bin boundaries

        Raises:
            ValueError: If invalid parameters or y contains NaN
        """
        if y is None or len(y) == 0:
            raise ValueError("y cannot be empty")

        y = np.asarray(y).flatten()

        if np.isnan(y).any():
            raise ValueError("y contains NaN values")

        if bins < 1 or bins > 1000:
            raise ValueError(f"bins must be between 1 and 1000, got {bins}")

        if strategy not in ("quantile", "equal_width"):
            raise ValueError(f"strategy must be 'quantile' or 'equal_width', got {strategy}")

        # Single bin case
        if bins == 1:
            return np.zeros(len(y), dtype=int), np.array([y.min(), y.max()])

        # Get bin edges
        if strategy == "quantile":
            bin_edges = BinningCalculator._quantile_binning(y, bins)
        else:  # equal_width
            bin_edges = BinningCalculator._equal_width_binning(y, bins)

        # Assign samples to bins using digitize (right=True for right-inclusive intervals)
        bin_indices = np.digitize(y, bin_edges, right=True)

        return bin_indices, bin_edges

    @staticmethod
    def _quantile_binning(y: np.ndarray, bins: int) -> np.ndarray:
        """
        Create bin edges using quantiles (equal probability per bin).

        Each bin will have approximately n_samples/bins items.
        """
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(y, quantiles)

        # Ensure edges are strictly increasing (handle duplicates from constant regions)
        bin_edges = np.unique(bin_edges)

        # If we got fewer unique edges than expected, pad with original edges
        if len(bin_edges) < bins + 1:
            # Use equal width as fallback to maintain bin count
            return np.linspace(y.min(), y.max(), bins + 1)

        return bin_edges

    @staticmethod
    def _equal_width_binning(y: np.ndarray, bins: int) -> np.ndarray:
        """
        Create bin edges with uniform width.

        Each bin has width = (y.max() - y.min()) / bins.
        """
        return np.linspace(y.min(), y.max(), bins + 1)

