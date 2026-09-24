"""Host numerical operator for a step-local missing-value replacement node."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceMissingValues(TransformerMixin, BaseEstimator):
    """Replace NaNs without learning from validation or inference rows."""

    def __init__(self, fill_value: float = 0) -> None:
        self.fill_value = fill_value

    def fit(self, X: Any, y: Any = None) -> ReplaceMissingValues:  # noqa: ARG002 - stateless transform
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X)
        return np.where(np.isnan(values), self.fill_value, values)
