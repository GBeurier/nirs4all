"""Native-only ``RunResult`` projection for the portable Methods lane.

This module is intentionally separate from the legacy workspace result path.
It exposes the canonical DAG-ML score evidence and writes Archive V2 through
the fitted native estimator; neither operation may materialize a
``PipelineRunner`` or a host-sidecar model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsIndex

from nirs4all.api.result import RunResult
from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError
from nirs4all.pipeline.dagml.result import _scores_to_run_result

from .native_retrain_lineage import DIAGNOSTIC_KEY as RETRAIN_LINEAGE_DIAGNOSTIC_KEY
from .native_retrain_lineage import NativeRetrainLineage
from .native_witness import NativeMethodsExecutionClaim, _LiveMethodsWitness

_RESULT_FACTORY_CAPABILITY = object()


class NativeMethodsRunResult(RunResult):
    """A ``RunResult`` backed only by a fitted portable Methods estimator.

    The compatibility projection consumes the exact native ScoreSet.  It does
    not reconstruct scores, attach a legacy workspace, or retain a Python
    model.  Predictions on new cohorts are intentionally performed through
    :attr:`native_estimator` with explicit sample identities.
    """

    def __init__(
        self,
        projected: RunResult,
        estimator: DagMLPipelineEstimator,
        *,
        archive_id: str,
        live_witness: _LiveMethodsWitness,
        _factory_capability: object,
    ) -> None:
        if _factory_capability is not _RESULT_FACTORY_CAPABILITY:
            raise TypeError("native Methods results can only be created by the verified internal factory")
        if not isinstance(archive_id, str) or not archive_id:
            raise ValueError("native archive_id must be a non-empty string")
        if type(live_witness) is not _LiveMethodsWitness:
            raise TypeError("native Methods results require an internal live execution witness")
        super().__init__(
            predictions=projected.predictions,
            per_dataset={dataset_name: {**info, "engine": "native"} for dataset_name, info in projected.per_dataset.items()},
        )
        self._dagml_score_set = projected._dagml_score_set  # noqa: SLF001
        self._dagml_node_results = projected._dagml_node_results  # noqa: SLF001
        self._native_estimator = estimator
        self._native_archive_id = archive_id
        self._native_archive_reference: dict[str, str] | None = None
        self._live_witness = live_witness

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

        witness = _LiveMethodsWitness.from_estimator(estimator)
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
        if witness._claim_for_estimator(estimator).outcome_fingerprint != fingerprint:
            raise DagMLNativeCoverageError("native Methods live witness fingerprint does not match the projected training outcome")
        return cls(
            projected,
            estimator,
            archive_id=f"archive:{fingerprint}",
            live_witness=witness,
            _factory_capability=_RESULT_FACTORY_CAPABILITY,
        )

    @property
    def native_estimator(self) -> DagMLPipelineEstimator:
        """The fitted native estimator for identity-bound PREDICT replay."""

        return self._native_estimator

    @property
    def native_archive_reference(self) -> dict[str, str] | None:
        """The exact Core-issued reference from the last successful export."""

        return None if self._native_archive_reference is None else dict(self._native_archive_reference)

    @property
    def native_execution_claim(self) -> NativeMethodsExecutionClaim:
        """Return the current audit-only claim for attached strict execution.

        This property fails closed once the result has been detached or closed.
        It intentionally exposes neither the raw PyO3 result nor native input
        buffers, callbacks, library paths, or controllers.
        """

        witness = getattr(self, "_live_witness", None)
        if type(witness) is not _LiveMethodsWitness:
            raise DagMLNativeCoverageError("native Methods result has no live execution witness")
        return witness._claim_for_estimator(self._native_estimator)

    @property
    def native_execution_is_live(self) -> bool:
        """Whether the strict process-local execution witness remains attached."""

        witness = getattr(self, "_live_witness", None)
        return type(witness) is _LiveMethodsWitness and witness._is_live_for_estimator(self._native_estimator)

    def detach(self) -> None:
        """Release legacy and strict native runtime resources deterministically."""

        witness = getattr(self, "_live_witness", None)
        if type(witness) is _LiveMethodsWitness:
            witness.detach()
        super().detach()

    def close(self) -> None:
        """Close ordinary result resources and detach the strict native facade."""

        witness = getattr(self, "_live_witness", None)
        if type(witness) is _LiveMethodsWitness:
            witness.detach()
        super().close()

    @property
    def native_methods_hpo_resume_state(self) -> dict[str, Any] | None:
        """Return the attested native HPO checkpoint/evidence, when present.

        The value is the exact DAG-ML durable state (including the opaque
        N4MOPT checkpoint and completed terminal trial ledger), copied only at
        the outer mapping boundary so callers cannot mutate the retained
        training outcome.  Native public resume is intentionally not exposed
        yet: accepting an arbitrary checkpoint here would bypass the package
        binding and provenance validation owned by DAG-ML.
        """

        outcome = _outcome_document(self._native_estimator)
        state = outcome.get("methods_hpo_resume_state")
        if state is None:
            return None
        if not isinstance(state, Mapping):
            raise DagMLNativeCoverageError("native Methods HPO state is not a structured mapping")
        return dict(state)

    def hpo_resume_package_json(self) -> str:
        """Return the complete signed package required to resume native HPO.

        The package — not a free checkpoint or a Python trial list — is the
        portable resume carrier.  Passing its exact JSON back as
        ``tuning={"engine": "methods-hpo", ..., "resume_package_json": ...}``
        lets DAG-ML validate the checkpoint, terminal ledger, plan, fold set,
        identities, influence and SELECT binding before it requests another
        native trial.  Results without a Methods HPO state cannot be used as a
        resume parent.
        """

        if self.native_methods_hpo_resume_state is None:
            raise ValueError("native result has no Methods HPO resume state")
        package = getattr(self._native_estimator, "predictor_package_", None)
        serializer = getattr(package, "json", None)
        if not callable(serializer):
            raise DagMLNativeCoverageError(
                "native Methods HPO result does not retain a strict portable package serializer"
            )
        package_json = serializer()
        if not isinstance(package_json, str) or not package_json:
            raise DagMLNativeCoverageError(
                "native Methods HPO package serializer returned an invalid payload"
            )
        return package_json

    @property
    def native_selected_variant_id(self) -> str | None:
        """Return the scheduler-selected native variant identity, if any."""

        outcome = _outcome_document(self._native_estimator)
        value = outcome.get("selected_variant_id")
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise DagMLNativeCoverageError("native selected variant id is malformed")
        return value

    @property
    def native_conformal_calibration(self) -> dict[str, Any] | None:
        """Return the DAG-ML-attested conformal state, when calibration ran.

        The mapping is the native contract emitted after the exact calibration
        replay was attached.  No Python interval state is reconstructed here;
        Archive V2 and native PREDICT remain the authoritative persistence and
        materialization paths.
        """

        outcome = _outcome_document(self._native_estimator)
        calibration = outcome.get("conformal_calibration")
        if calibration is None:
            return None
        if not isinstance(calibration, Mapping):
            raise DagMLNativeCoverageError("native conformal calibration is not a structured mapping")
        return dict(calibration)

    @property
    def native_retrain_lineage(self) -> dict[str, Any] | None:
        """Return strict, durable parent evidence for a native full retrain."""

        outcome = _outcome_document(self._native_estimator)
        diagnostics = outcome.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise DagMLNativeCoverageError("native Methods outcome diagnostics are not a structured mapping")
        lineage = diagnostics.get(RETRAIN_LINEAGE_DIAGNOSTIC_KEY)
        if lineage is None:
            return None
        if not isinstance(lineage, Mapping):
            raise DagMLNativeCoverageError("native Methods retrain lineage is not a structured mapping")
        try:
            return NativeRetrainLineage.from_mapping(lineage).to_dict()
        except ValueError as error:
            raise DagMLNativeCoverageError("native Methods retrain lineage is malformed") from error

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

    def export_model(
        self,
        output_path: str | Path,
        source: dict[str, Any] | None = None,
        format: str | None = None,
        fold: int | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Refuse legacy model export before it can create any side effect.

        A strict Methods result is persistable only as its native Archive V2.
        The inherited compatibility exporter may refit through legacy paths,
        which would invalidate the execution claim.
        """

        _ = (output_path, source, format, fold, compatibility)
        raise NotImplementedError("native Methods export_model is unavailable; use export(path, format='n4a') for Core Archive V2")

    def __copy__(self) -> NativeMethodsRunResult:
        raise TypeError("a live native Methods result cannot be copied; export its Archive V2 instead")

    def __deepcopy__(self, memo: dict[int, Any]) -> NativeMethodsRunResult:
        _ = memo
        raise TypeError("a live native Methods result cannot be deep-copied; export its Archive V2 instead")

    def __reduce__(self) -> Any:
        raise TypeError("a live native Methods result cannot be serialized; export its Archive V2 instead")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        _ = protocol
        raise TypeError("a live native Methods result cannot be serialized; export its Archive V2 instead")


def _outcome_document(estimator: DagMLPipelineEstimator) -> Mapping[str, Any]:
    outcome = getattr(estimator, "training_outcome_", None)
    to_dict = getattr(outcome, "to_dict", None)
    if callable(to_dict):
        outcome = to_dict()
    if not isinstance(outcome, Mapping):
        raise DagMLNativeCoverageError("native Methods training did not return a structured outcome")
    return outcome


__all__ = ["NativeMethodsRunResult"]
