"""Native-only :class:`RunResult` projection for portable Methods training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.result import _scores_to_run_result

from .result import RunResult


class NativeMethodsRunResult(RunResult):
    """A ``RunResult`` retaining exactly one fitted portable Methods estimator."""

    def __init__(self, projected: RunResult, estimator: DagMLPipelineEstimator, *, archive_id: str) -> None:
        super().__init__(
            predictions=projected.predictions,
            per_dataset={
                dataset_name: {**info, "engine": "native"}
                for dataset_name, info in projected.per_dataset.items()
            },
        )
        self._dagml_score_set = projected._dagml_score_set  # noqa: SLF001
        self._native_estimator = estimator
        self._native_archive_id = archive_id
        self._native_archive_reference: dict[str, str] | None = None

    @classmethod
    def from_estimator(
        cls,
        estimator: DagMLPipelineEstimator,
        *,
        dataset_name: str,
        model_name: str,
        metric: str = "rmse",
        task_type: str = "regression",
    ) -> NativeMethodsRunResult:
        """Project an already-fitted estimator's canonical score evidence."""

        outcome = _outcome_document(estimator)
        scores = outcome.get("score_set")
        fingerprint = outcome.get("outcome_fingerprint")
        if not isinstance(scores, Mapping):
            raise DagMLNativeCoverageError("native Methods training did not return a canonical ScoreSet")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise DagMLNativeCoverageError(
                "native Methods training did not return an outcome fingerprint for Archive V2 export"
            )
        projected = _scores_to_run_result(
            dict(scores), dataset_name, model_name, metric, task_type
        )
        return cls(projected, estimator, archive_id=f"archive:{fingerprint}")

    @property
    def native_estimator(self) -> DagMLPipelineEstimator:
        """The fitted estimator used for strict identity-bound replay."""

        return self._native_estimator

    @property
    def native_archive_reference(self) -> dict[str, str] | None:
        """Core-issued reference from the most recent successful export."""

        return None if self._native_archive_reference is None else dict(self._native_archive_reference)

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
    ) -> Path:
        """Write a Core Archive V2 from the captured native package only."""

        if format != "n4a":
            raise ValueError("native Methods export supports only format='n4a' (Core Archive V2)")
        if source is not None or chain_id is not None:
            raise NotImplementedError(
                "native Methods export does not accept legacy source=/chain_id= selectors"
            )
        path = Path(output_path)
        if path.suffix.lower() != ".n4a":
            raise ValueError("native Methods export requires a .n4a Archive V2 path")
        self._native_archive_reference = self._native_estimator.export_native_archive(
            path, archive_id=self._native_archive_id
        )
        return path


def _outcome_document(estimator: DagMLPipelineEstimator) -> Mapping[str, Any]:
    outcome = getattr(estimator, "training_outcome_", None)
    to_dict = getattr(outcome, "to_dict", None)
    if callable(to_dict):
        outcome = to_dict()
    if not isinstance(outcome, Mapping):
        raise DagMLNativeCoverageError("native Methods training did not return a structured outcome")
    return outcome


__all__ = ["NativeMethodsRunResult"]
