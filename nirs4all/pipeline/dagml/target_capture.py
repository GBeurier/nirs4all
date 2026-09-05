"""Target-output transforms attached only to captured host artifacts.

Training and native scoring operate in the dataset's numeric target space.
Exported predictors additionally need to restore the labels supplied by the
user. Keeping that decoder outside the runtime ``y_processing`` transform
prevents decoded labels from leaking back into DAG score blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CapturedTargetTransform(TransformerMixin, BaseEstimator):
    """Compose an optional model-target transform with a fitted label decoder."""

    def __init__(self, transformer: Any, decoder: Any) -> None:
        self.transformer = transformer
        self.decoder = decoder

    def fit(self, y: Any, *_args: Any, **_kwargs: Any) -> CapturedTargetTransform:
        if self.transformer is not None:
            self.transformer.fit(y)
        return self

    def fit_transform(self, y: Any, *_args: Any, **_kwargs: Any) -> Any:
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.fit_transform(y)

    def transform(self, y: Any) -> Any:
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.transform(y)

    def inverse_numeric(self, y: Any) -> Any:
        """Restore numeric target space without decoding labels."""
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.inverse_transform(y)

    def decode(self, y: Any) -> Any:
        """Decode numeric targets to the labels supplied by the user."""
        return self.decoder.inverse_transform(y)

    def inverse_transform(self, y: Any) -> Any:
        return self.decode(self.inverse_numeric(y))

    @property
    def classes_(self) -> Any:
        """Expose the captured public class axis when the decoder has one."""
        classes = getattr(self.decoder, "classes_", None)
        if classes is not None:
            return classes
        columns = getattr(self.decoder, "column_transformers", None)
        if isinstance(columns, dict) and len(columns) == 1:
            member = next(iter(columns.values()))
            classes = getattr(member, "classes_", None)
            if classes is not None:
                return classes
        raise AttributeError("captured target decoder has no single class axis")


def captured_target_transform(transformer: Any, decoder: Any) -> Any:
    """Return the public capture transform, avoiding identity/double wrappers."""
    if decoder is None or isinstance(transformer, CapturedTargetTransform):
        return transformer
    return CapturedTargetTransform(transformer, decoder)
