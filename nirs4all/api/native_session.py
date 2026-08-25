"""Stateful native Methods session with no legacy workspace dependency."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .native_result import NativeMethodsRunResult
from .native_training import run_native_methods
from .result import PredictResult


class NativeMethodsSession:
    """One portable Methods pipeline through train, predict, export, and close.

    This session holds only the fitted DAG-ML/Methods estimator and never
    creates a ``PipelineRunner`` or legacy workspace.  A persisted session is
    loaded through :func:`nirs4all.load_session` with ``engine='native'`` and
    becomes a :class:`NativeArchiveSession`.
    """

    def __init__(
        self,
        pipeline: list[Any],
        *,
        name: str = "",
        random_state: int | None = None,
    ) -> None:
        if not isinstance(pipeline, list):
            raise TypeError("engine='native' session requires a list pipeline")
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError("engine='native' session random_state must be an integer or None")
        self._pipeline = pipeline
        self._name = name
        self._random_state = random_state
        self._result: NativeMethodsRunResult | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""

        return self._closed

    @property
    def is_trained(self) -> bool:
        """Whether this session owns a fitted native result."""

        return self._result is not None and not self._closed

    @property
    def result(self) -> NativeMethodsRunResult:
        """The fitted native run result, or raise before training."""

        if not self.is_trained:
            raise ValueError("NativeMethodsSession must be trained before accessing its result")
        assert self._result is not None
        return self._result

    def run(self, dataset: Mapping[str, Any]) -> NativeMethodsRunResult:
        """Fit this session's portable Methods pipeline exactly once per call."""

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
        """Replay the fitted native package for one explicitly identified cohort."""

        self._require_open()
        values = np.asarray(
            self.result.native_estimator.predict_with_identity(
                X,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
            )
        )
        return PredictResult(
            y_pred=values,
            metadata={"engine": "native", "sample_ids": list(sample_ids)},
            model_name="MethodsN4MM",
            preprocessing_steps=[],
        )

    def save(self, path: str | Path) -> Path:
        """Persist the fitted package as Core Archive V2 without refitting."""

        self._require_open()
        return self.result.export(path)

    def close(self) -> None:
        """Release this session's in-memory estimator reference."""

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
