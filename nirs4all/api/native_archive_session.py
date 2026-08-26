"""Fail-closed public session for a Core Archive V2 Methods predictor.

This module deliberately does not adapt :class:`~nirs4all.pipeline.PipelineRunner`.
Core validates the archive member, DAG-ML validates its Package V2, and Methods
hydrates the N4MM for each individual PREDICT call.  The session retains no
native model handle and therefore cannot accidentally route a native archive
through the legacy pipeline runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .result import PredictResult


class NativeArchiveSession:
    """Reusable PREDICT-only session for one validated native Archive V2.

    Use :func:`load_native_archive_session` rather than constructing this class
    directly.  Training, retraining and explanation intentionally remain
    unavailable: providing a partial legacy emulation would weaken the native
    execution boundary.
    """

    def __init__(self, archive_path: str | Path) -> None:
        path = Path(archive_path)
        if not path.is_file():
            raise FileNotFoundError(f"native archive not found: {path}")
        if path.suffix.lower() != ".n4a":
            raise ValueError("native archive sessions require an Archive V2 .n4a path")

        # Validate while opening.  This proves the Core/DAG-ML closure before a
        # caller treats the object as a loaded native session; no N4MM is
        # hydrated at this stage.
        from nirs4all.pipeline.dagml.native_archive_replay import validate_methods_archive_v2

        validate_methods_archive_v2(path)
        self._archive_path = path
        self._closed = False

    @property
    def archive_path(self) -> Path:
        """The immutable native archive selected for every replay."""

        return self._archive_path

    @property
    def closed(self) -> bool:
        """Whether this session was explicitly closed."""

        return self._closed

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any,
        groups: Any = None,
        metadata: Any = None,
    ) -> PredictResult:
        """Replay this archive for an explicitly identified feature cohort."""

        if self._closed:
            raise RuntimeError("NativeArchiveSession is closed")
        from nirs4all.pipeline.dagml.native_archive_replay import (
            predict_methods_archive_v2_raw,
        )

        values = predict_methods_archive_v2_raw(
            self._archive_path,
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
        )
        matrix = np.asarray(values, dtype=float)
        return PredictResult(
            y_pred=matrix,
            metadata={
                "engine": "native",
                "archive_path": str(self._archive_path),
                "sample_ids": [str(sample_id) for sample_id in sample_ids],
            },
            model_name="MethodsN4MM",
            preprocessing_steps=[],
        )

    def close(self) -> None:
        """Close this session; native replay resources are invocation-local."""

        self._closed = True

    def __enter__(self) -> NativeArchiveSession:
        if self._closed:
            raise RuntimeError("NativeArchiveSession is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def load_native_archive_session(path: str | Path) -> NativeArchiveSession:
    """Load one Core Archive V2 PREDICT session without a legacy runner."""

    return NativeArchiveSession(path)


__all__ = ["NativeArchiveSession", "load_native_archive_session"]
