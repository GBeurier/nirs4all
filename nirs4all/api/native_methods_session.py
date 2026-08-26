"""Stateful native Methods training session without a legacy workspace."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .native_result import NativeMethodsRunResult
from .native_training import run_native_methods
from .result import PredictResult


class NativeMethodsSession:
    """Own one portable Methods pipeline across train, predict and Archive V2 save.

    This object retains only the fitted DAG-ML/Methods estimator produced by
    :func:`run_native_methods`. It does not construct a ``PipelineRunner``, a
    legacy workspace, or a cached N4MM handle; each prediction is an
    identity-bound loaded-package replay.
    """

    def __init__(self, pipeline: list[Any], *, name: str = "", random_state: int | None = None) -> None:
        if not isinstance(pipeline, list):
            raise TypeError("NativeMethodsSession requires a list pipeline")
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError("NativeMethodsSession random_state must be an integer or None")
        self._pipeline = pipeline
        self._name = name
        self._random_state = random_state
        self._result: NativeMethodsRunResult | None = None
        self._closed = False

    @property
    def pipeline(self) -> list[Any]:
        """The exact pipeline declaration owned by this session."""

        return self._pipeline

    @property
    def name(self) -> str:
        """The immutable public name for the portable training run."""

        return self._name

    @property
    def random_state(self) -> int | None:
        """The seed used by the session's native training run."""

        return self._random_state

    @property
    def closed(self) -> bool:
        """Whether this session has released its fitted estimator."""

        return self._closed

    @property
    def is_trained(self) -> bool:
        """Whether native training completed and the session is still open."""

        return self._result is not None and not self._closed

    @property
    def result(self) -> NativeMethodsRunResult:
        """The fitted native result, refusing access before training."""

        self._require_open()
        if self._result is None:
            raise ValueError("NativeMethodsSession must be trained before accessing its result")
        return self._result

    def run(self, dataset: Mapping[str, Any]) -> NativeMethodsRunResult:
        """Train the session pipeline through the strict portable Methods lane."""

        self._require_open()
        self._result = run_native_methods(
            self._pipeline,
            dataset,
            name=self._name,
            save_charts=False,
            random_state=self._random_state,
        )
        return self._result

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any,
        groups: Any = None,
        metadata: Any = None,
    ) -> PredictResult:
        """Replay the fitted package for one explicitly identified cohort."""

        self._require_open()
        values = np.asarray(
            self.result.native_estimator.predict_with_identity(
                X,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
            ),
            dtype=float,
        )
        return PredictResult(
            y_pred=values,
            metadata={"engine": "native", "sample_ids": [str(value) for value in sample_ids]},
            model_name="MethodsN4MM",
            preprocessing_steps=[],
        )

    def save(self, path: str | Path) -> Path:
        """Persist the current portable package directly as a Core Archive V2."""

        self._require_open()
        return Path(self.result.export(path))

    def close(self) -> None:
        """Release the fitted in-memory estimator reference."""

        self._result = None
        self._closed = True

    def __enter__(self) -> NativeMethodsSession:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("NativeMethodsSession is closed")


__all__ = ["NativeMethodsSession"]
