"""
Grouped Splitter Wrapper for universal group support.

This module provides a wrapper that enables any sklearn-compatible splitter
to work with grouped samples by aggregating samples by group, passing
"virtual samples" to the inner splitter, and expanding fold indices back
to the original dataset.
"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class GroupedSplitterWrapper(BaseCrossValidator):
    """
    Wraps any sklearn-compatible splitter to add group-awareness.

    This wrapper aggregates samples by group into "virtual samples",
    passes them to the inner splitter, and expands the fold indices
    back to the original sample space. This ensures that all samples
    from the same group are always in the same fold (train or test),
    preventing data leakage.

    Parameters
    ----------
    splitter : BaseCrossValidator
        Any sklearn-compatible cross-validator (e.g., KFold, ShuffleSplit,
        StratifiedKFold).

    aggregation : str, default="mean"
        Method for aggregating X features within groups:
        - "mean": Use group centroid (average of all samples)
        - "median": Use group median (robust to outliers)
        - "first": Use first sample in each group (fast, no aggregation)

    y_aggregation : str or None, default=None
        Method for aggregating y values within groups. If None, inferred
        from splitter type:
        - "mean": For regression (continuous y)
        - "mode": For classification (categorical y)
        - "first": Use first y value in group

    Examples
    --------
    >>> from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
    >>> import numpy as np
    >>>
    >>> # Basic usage with KFold
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> groups = np.repeat(np.arange(20), 5)  # 20 groups, 5 samples each
    >>>
    >>> wrapper = GroupedSplitterWrapper(KFold(n_splits=5))
    >>> for train_idx, test_idx in wrapper.split(X, y, groups=groups):
    ...     # train_idx and test_idx are original sample indices
    ...     # All samples from the same group are in the same fold
    ...     train_groups = set(groups[train_idx])
    ...     test_groups = set(groups[test_idx])
    ...     assert len(train_groups & test_groups) == 0  # No overlap
    >>>
    >>> # Usage with ShuffleSplit
    >>> wrapper = GroupedSplitterWrapper(ShuffleSplit(n_splits=1, test_size=0.2))
    >>> for train_idx, test_idx in wrapper.split(X, y, groups=groups):
    ...     pass  # Groups are respected
    >>>
    >>> # Usage with StratifiedKFold (stratifies on aggregated y)
    >>> y_class = np.random.randint(0, 3, 100)
    >>> wrapper = GroupedSplitterWrapper(
    ...     StratifiedKFold(n_splits=3),
    ...     y_aggregation="mode"
    ... )
    >>> for train_idx, test_idx in wrapper.split(X, y_class, groups=groups):
    ...     pass  # Groups are respected, stratification on group mode

    Notes
    -----
    The wrapper is transparent when no groups are provided - it simply
    delegates to the inner splitter without any aggregation.

    See Also
    --------
    sklearn.model_selection.GroupKFold : Native group-aware K-fold splitter.
    sklearn.model_selection.GroupShuffleSplit : Native group-aware shuffle split.
    nirs4all.operators.splitters.SPXYGFold : SPXY-based group-aware splitter.
    """

    def __init__(self, splitter, aggregation="mean", y_aggregation=None):
        self.splitter = splitter
        self.aggregation = aggregation
        self.y_aggregation = y_aggregation

        # Validate aggregation parameter
        valid_aggregations = ("mean", "median", "first")
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got '{aggregation}'"
            )

        # Validate y_aggregation parameter if provided
        valid_y_aggregations = ("mean", "mode", "first", None)
        if y_aggregation not in valid_y_aggregations:
            raise ValueError(
                f"y_aggregation must be one of {valid_y_aggregations}, got '{y_aggregation}'"
            )

    def _aggregate(self, X, y, groups):
        """Aggregate samples by group into representative virtual samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,) or None
            Target values.
        groups : ndarray of shape (n_samples,)
            Group labels for each sample.

        Returns
        -------
        X_rep : ndarray of shape (n_groups, n_features)
            Representative features for each group.
        y_rep : ndarray of shape (n_groups,) or None
            Representative targets for each group.
        group_indices : list of lists
            For each group, the list of sample indices belonging to it.
        unique_groups : ndarray
            Unique group labels in order.
        """
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Compute group representatives for X
        X_rep = np.zeros((n_groups, X.shape[1]), dtype=X.dtype)
        group_indices = []

        for i, g in enumerate(unique_groups):
            mask = groups == g
            indices = np.where(mask)[0].tolist()
            group_indices.append(indices)

            # Aggregate X
            if self.aggregation == "mean":
                X_rep[i] = X[mask].mean(axis=0)
            elif self.aggregation == "median":
                X_rep[i] = np.median(X[mask], axis=0)
            elif self.aggregation == "first":
                X_rep[i] = X[mask][0]

        # Compute group representatives for y
        y_rep = None
        if y is not None:
            y = np.asarray(y)
            y_rep = np.zeros(n_groups, dtype=y.dtype)

            y_agg = self.y_aggregation or self._infer_y_aggregation()

            for i, g in enumerate(unique_groups):
                mask = groups == g
                y_group = y[mask]

                if y_agg == "mean":
                    y_rep[i] = y_group.mean()
                elif y_agg == "mode":
                    from scipy import stats
                    mode_result = stats.mode(y_group, keepdims=False)
                    # Extract scalar from mode result to avoid numpy deprecation warning
                    mode_value = mode_result.mode
                    if hasattr(mode_value, 'item'):
                        mode_value = mode_value.item()
                    y_rep[i] = mode_value
                elif y_agg == "first":
                    y_rep[i] = y_group[0]

        return X_rep, y_rep, group_indices, unique_groups

    def _infer_y_aggregation(self):
        """Infer y aggregation method from splitter type.

        Returns
        -------
        str
            The inferred aggregation method ("mode" for stratified, "mean" otherwise).
        """
        splitter_name = self.splitter.__class__.__name__
        if "Stratified" in splitter_name:
            return "mode"  # Classification - use mode for aggregation
        return "mean"  # Default for regression

    def _expand_indices(self, rep_indices, group_indices):
        """Expand representative indices to original sample indices.

        Parameters
        ----------
        rep_indices : array-like
            Indices into the representative (group) array.
        group_indices : list of lists
            For each group, the list of sample indices belonging to it.

        Returns
        -------
        ndarray
            Original sample indices corresponding to the representative indices.
        """
        return np.concatenate([group_indices[i] for i in rep_indices])

    def split(self, X, y=None, groups=None):
        """Generate train/test indices with group-awareness.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples. If None, delegates to the inner
            splitter without any aggregation.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X = np.asarray(X)

        if groups is None:
            # No groups - delegate to original splitter
            yield from self.splitter.split(X, y)
            return

        groups = np.asarray(groups)

        if y is not None:
            y = np.asarray(y)

        # Aggregate by groups
        X_rep, y_rep, group_indices, unique_groups = self._aggregate(X, y, groups)

        # Split on representative samples
        for train_rep, test_rep in self.splitter.split(X_rep, y_rep):
            train_indices = self._expand_indices(train_rep, group_indices)
            test_indices = self._expand_indices(test_rep, group_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Ignored, exists for compatibility.
        y : object
            Ignored, exists for compatibility.
        groups : object
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of folds/iterations from the inner splitter.
        """
        return self.splitter.get_n_splits(X, y, groups)

    def __repr__(self):
        """Return string representation of the wrapper."""
        return (
            f"GroupedSplitterWrapper(splitter={self.splitter!r}, "
            f"aggregation='{self.aggregation}', "
            f"y_aggregation={self.y_aggregation!r})"
        )
