"""
DataAggregator - Aggregate scores using different strategies.
"""
import numpy as np
from typing import List


class DataAggregator:
    """Aggregate scores using different strategies.

    Supports multiple aggregation methods with proper handling of
    ranking information (when display and rank metrics differ).
    """

    @staticmethod
    def aggregate(scores: List, method: str, higher_better: bool) -> float:
        """Aggregate scores using specified method.

        Args:
            scores: List of scores (can be floats or tuples of (display_score, rank_score)).
            method: Aggregation method ('best', 'worst', 'mean', 'median').
            higher_better: Whether higher values are better.

        Returns:
            Aggregated score value.
        """
        if not scores:
            return np.nan

        # Check if we have tuples (at least one must be a tuple for tuple handling)
        has_tuples = any(isinstance(s, tuple) for s in scores)

        # Handle tuple scores (display_score, rank_score)
        if has_tuples:
            # Normalize all to tuples (for mixed cases, treat floats as (score, score))
            normalized_scores = []
            for s in scores:
                if isinstance(s, tuple):
                    normalized_scores.append(s)
                else:
                    normalized_scores.append((float(s), float(s)))

            if method in ['best', 'worst']:
                # Select based on rank_score (second element)
                rank_scores = [s[1] for s in normalized_scores]
                if method == 'best':
                    if higher_better:
                        best_idx = np.argmax(rank_scores)
                    else:
                        best_idx = np.argmin(rank_scores)
                else:  # worst
                    if higher_better:
                        best_idx = np.argmin(rank_scores)
                    else:
                        best_idx = np.argmax(rank_scores)
                return normalized_scores[best_idx][0]  # Return display_score
            else:
                # For mean/median, aggregate display_scores
                scores = [s[0] for s in normalized_scores]

        # Aggregate simple scores
        if method == 'best':
            return float(np.max(scores) if higher_better else np.min(scores))
        elif method == 'worst':
            return float(np.min(scores) if higher_better else np.max(scores))
        elif method == 'mean':
            return float(np.mean(scores))
        elif method == 'median':
            return float(np.median(scores))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
