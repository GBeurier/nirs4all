"""
ScoreNormalizer - Normalize scores for visualization.
"""
import numpy as np


class ScoreNormalizer:
    """Normalize scores for visualization.

    Handles normalization to [0, 1] range with support for
    both higher-is-better and lower-is-better metrics.
    """

    @staticmethod
    def normalize(matrix: np.ndarray, higher_better: bool, per_row: bool = False) -> np.ndarray:
        """Normalize matrix values to [0, 1] range.

        Args:
            matrix: Input matrix to normalize.
            higher_better: Whether higher values are better.
            per_row: If True, normalize each row independently. If False, normalize globally.

        Returns:
            Normalized matrix with values in [0, 1] range.
        """
        normalized = matrix.copy()

        if per_row:
            # Normalize each row independently
            for i in range(normalized.shape[0]):
                row = normalized[i, :]
                valid_mask = ~np.isnan(row)

                if not np.any(valid_mask):
                    continue

                valid_scores = row[valid_mask]
                min_val = np.min(valid_scores)
                max_val = np.max(valid_scores)

                if max_val > min_val:
                    row[valid_mask] = (valid_scores - min_val) / (max_val - min_val)
                    if not higher_better:
                        row[valid_mask] = 1 - row[valid_mask]
                else:
                    row[valid_mask] = 0.5

                normalized[i, :] = row
        else:
            # Global normalization
            valid_mask = ~np.isnan(normalized)

            if not np.any(valid_mask):
                return normalized

            valid_scores = normalized[valid_mask]
            min_val = np.min(valid_scores)
            max_val = np.max(valid_scores)

            if max_val > min_val:
                normalized[valid_mask] = (valid_scores - min_val) / (max_val - min_val)
                if not higher_better:
                    normalized[valid_mask] = 1 - normalized[valid_mask]
            else:
                normalized[valid_mask] = 0.5

        return normalized

    @staticmethod
    def is_higher_better(metric: str) -> bool:
        """Check if metric is higher-is-better.

        Args:
            metric: Metric name.

        Returns:
            True if higher is better, False otherwise.
        """
        return metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']
