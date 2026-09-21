"""Fixed-width host encodings of variable-length multichannel series."""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted


class SequenceSummary(TransformerMixin, BaseEstimator):
    """Summarize each ragged series independently, without padding or resampling.

    Input is a :class:`nirs4all_io.ragged.RaggedSeriesBatch`. Each row contains
    a variable number of observations and the same number of channels. Output
    columns are ordered by channel, then by ``statistics``; the optional
    ``length`` column comes last. Standard deviation uses ``ddof=0``.

    Observations have equal weight: time coordinates are deliberately ignored.
    Empty series and nonfinite observed values are refused. Missing sources
    must be excluded by the source-presence policy before reaching this encoder.
    Fitting records the channel contract, not training values or lengths. New
    series lengths are therefore accepted by the fitted encoder.

    Args:
        statistics: Ordered, distinct subset of ``mean``, ``std``, ``min`` and
            ``max``. An empty sequence is permitted when ``include_length`` is
            true, producing only the length column.
        include_length: Append the number of observations in each series.
        channel_names: Optional distinct, nonempty names, one per channel.
            Otherwise use ``channel_0``, ``channel_1``, etc. The source schema
            is responsible for preserving channel semantics across datasets.
    """

    def __init__(
        self, statistics: Sequence[str] = ("mean", "std", "min", "max"), *,
        include_length: bool = True, channel_names: Sequence[str] | None = None,
    ) -> None:
        self.statistics = statistics
        self.include_length = include_length
        self.channel_names = channel_names

    @staticmethod
    def _names(value: Any, n_channels: int, label: str) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"{label} must contain one nonempty string per channel.")
        names = tuple(value)
        if len(names) != n_channels or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{label} must contain {n_channels} distinct nonempty strings.")
        return names

    @staticmethod
    def _validate_batch(X: Any, *, fitting: bool) -> Any:
        # Keep the public transform importable with older IO installations;
        # using ragged inputs requires the IO release that owns this type.
        from nirs4all_io.ragged import RaggedSeriesBatch

        if not isinstance(X, RaggedSeriesBatch):
            raise TypeError("SequenceSummary requires a RaggedSeriesBatch; implicit padding or array conversion is not supported.")
        if fitting and not len(X):
            raise ValueError("SequenceSummary requires at least one training series.")
        if X.values.dtype.kind not in "iuf" or not np.isfinite(X.values).all():
            raise ValueError("SequenceSummary requires finite real numeric observations.")
        if np.any(X.lengths == 0):
            raise ValueError("SequenceSummary does not accept empty series; filter absent sources before encoding.")
        return X

    def fit(self, X: Any, y: Any = None) -> Self:
        """Record a channel contract from the supplied training rows only."""
        if not isinstance(self.statistics, (list, tuple)) or any(not isinstance(stat, str) or stat not in {"mean", "std", "min", "max"} for stat in self.statistics):
            raise ValueError("statistics must be an ordered sequence chosen from mean, std, min and max.")
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError("statistics must not contain duplicates.")
        if type(self.include_length) is not bool:
            raise ValueError("include_length must be a boolean.")
        if not self.statistics and not self.include_length:
            raise ValueError("SequenceSummary requires at least one output statistic or include_length=True.")
        batch = self._validate_batch(X, fitting=True)
        if y is not None:
            check_consistent_length(batch, y)
        n_channels = batch.shape[2]
        names = (tuple(f"channel_{index}" for index in range(n_channels)) if self.channel_names is None
                 else self._names(self.channel_names, n_channels, "channel_names"))
        self.n_features_in_ = n_channels
        self.channel_names_ = names
        self.statistics_ = tuple(self.statistics)
        self.include_length_ = self.include_length
        self.n_features_out_ = n_channels * len(self.statistics_) + int(self.include_length_)
        if self.channel_names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return independent per-channel summaries for new series lengths."""
        check_is_fitted(self, ["n_features_in_", "statistics_", "n_features_out_"])
        batch = self._validate_batch(X, fitting=False)
        if batch.shape[2] != self.n_features_in_:
            raise ValueError(f"SequenceSummary expected {self.n_features_in_} channels, got {batch.shape[2]}.")
        output = np.empty((len(batch), self.n_features_out_), dtype=np.float64)
        for index in range(len(batch)):
            values = np.asarray(batch[index], dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                summaries = [getattr(np, statistic)(values, axis=0) for statistic in self.statistics_]
            if summaries:
                output[index, :self.n_features_in_ * len(summaries)] = np.stack(summaries, axis=1).ravel()
            if self.include_length_:
                output[index, -1] = len(values)
        if not np.isfinite(output).all():
            raise ValueError("SequenceSummary statistics overflowed; rescale observations before encoding.")
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Name output columns in their stable channel/statistic order."""
        check_is_fitted(self, ["channel_names_", "statistics_", "include_length_"])
        names = self.channel_names_
        if input_features is not None:
            names = self._names(input_features, self.n_features_in_, "input_features")
            if hasattr(self, "feature_names_in_") and names != self.channel_names_:
                raise ValueError("input_features must match the fitted channel_names.")
        features = [f"{name}_{statistic}" for name in names for statistic in self.statistics_]
        return np.asarray([*features, "length"] if self.include_length_ else features, dtype=object)

    def _more_tags(self) -> dict[str, Any]:
        return {"X_types": ["3darray"]}

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        return tags
