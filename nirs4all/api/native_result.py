"""Native-only ``RunResult`` projection for the portable Methods lane.

This module is intentionally separate from the legacy workspace result path.
It exposes the canonical DAG-ML score evidence and writes Archive V2 through
the fitted native estimator; neither operation may materialize a
``PipelineRunner`` or a host-sidecar model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.result import _scores_to_run_result


class NativeMethodsRunResult(RunResult):
    """A ``RunResult`` backed only by a fitted portable Methods estimator.

    The compatibility projection consumes the exact native ScoreSet.  It does
    not reconstruct scores, attach a legacy workspace, or retain a Python
    model.  Predictions on new cohorts are intentionally performed through
    :attr:`native_estimator` with explicit sample identities.
    """

    def __init__(self, projected: RunResult, estimator: DagMLPipelineEstimator, *, archive_id: str) -> None:
        if not isinstance(archive_id, str) or not archive_id:
            raise ValueError("native archive_id must be a non-empty string")
        super().__init__(
            predictions=projected.predictions,
            per_dataset={
                dataset_name: {**info, "engine": "native"}
                for dataset_name, info in projected.per_dataset.items()
            },
        )
        self._dagml_score_set = projected._dagml_score_set  # noqa: SLF001
        self._dagml_node_results = projected._dagml_node_results  # noqa: SLF001
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
        """Project a fitted estimator's canonical ScoreSet without re-execution."""

        outcome = _outcome_document(estimator)
        scores = outcome.get("score_set")
        if not isinstance(scores, Mapping):
            raise DagMLNativeCoverageError("native Methods training did not return a canonical ScoreSet")
        fingerprint = outcome.get("outcome_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise DagMLNativeCoverageError("native Methods training did not return an outcome fingerprint for Archive V2 export")
        projected = _scores_to_run_result(
            dict(scores),
            dataset_name,
            model_name,
            metric,
            task_type,
        )
        return cls(projected, estimator, archive_id=f"archive:{fingerprint}")

    @property
    def native_estimator(self) -> DagMLPipelineEstimator:
        """The fitted native estimator for identity-bound PREDICT replay."""

        return self._native_estimator

    @property
    def native_archive_reference(self) -> dict[str, str] | None:
        """The exact Core-issued reference from the last successful export."""

        return None if self._native_archive_reference is None else dict(self._native_archive_reference)

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Write Core Archive V2 directly from the fitted native package.

        Legacy workspace selectors and ``compatibility='legacy-refit'`` are
        rejected: this result must never turn a native training request into a
        second host fit.
        """

        if format != "n4a":
            raise ValueError("native Methods export supports only format='n4a' (Core Archive V2)")
        if source is not None or chain_id is not None:
            raise NotImplementedError("native Methods export does not accept legacy source=/chain_id= selectors")
        if compatibility is not None:
            raise ValueError("native Methods export never accepts legacy-refit compatibility")
        path = Path(output_path)
        if path.suffix.lower() != ".n4a":
            raise ValueError("native Methods export requires a .n4a Archive V2 path")
        self._native_archive_reference = self._native_estimator.export_native_archive(
            path,
            archive_id=self._native_archive_id,
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
