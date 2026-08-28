"""Live, process-local provenance for the strict Methods execution lane.

The public ``engine=\"native\"`` path already has an intentionally separate
Dag-ML entry point: it supplies native Methods inputs and never installs a
Python operator callback.  This module does *not* turn that local fact into a
portable attestation.  It retains the exact live ``dag_ml.TrainingResult``
facade only long enough to make the local execution boundary observable and to
release its process-local resources deterministically.

The durable evidence remains the signed outcome and portable package owned by
Dag-ML.  The internal live witness relies only on Dag-ML's public Python
lifecycle surface, is deliberately neither serializable nor copyable, and is
invalid after ``detach()``/``close()``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, SupportsIndex

from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError

if TYPE_CHECKING:
    from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator


@dataclass(frozen=True, slots=True)
class NativeMethodsExecutionClaim:
    """Audit-only description of one locally witnessed strict execution.

    This is intentionally a small immutable projection.  It is not a signed
    receipt and cannot be used to establish provenance after the live witness
    has been detached or a result has crossed a process boundary.
    """

    schema_version: int
    execution_entrypoint: str
    execution_mode: str
    outcome_fingerprint: str
    methods_library_mode: str
    portable_artifacts_required: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible audit projection."""

        return asdict(self)


_WITNESS_FACTORY_CAPABILITY = object()


class _LiveMethodsWitness:
    """Non-serializable live holder for a strict Dag-ML Methods result.

    Constructing a witness validates the exact facts nirs4all controls at the
    boundary: the compiled request uses Methods inputs, has an explicit native
    library path, forbids a Python callback, and returned an attached
    ``dag_ml.TrainingResult`` whose outcome fingerprint is the final outcome
    fingerprint.  Dag-ML remains authoritative for graph and controller
    validation.
    """

    __slots__ = (
        "_claim",
        "_estimator",
        "_factory_capability",
        "_lock",
        "_training_result",
    )

    def __init__(
        self,
        estimator: Any,
        training_result: Any,
        claim: NativeMethodsExecutionClaim,
        *,
        _factory_capability: object,
    ) -> None:
        if _factory_capability is not _WITNESS_FACTORY_CAPABILITY:
            raise TypeError("live Methods witnesses can only be created by the verified internal factory")
        self._estimator = estimator
        self._training_result: Any | None = training_result
        self._claim = claim
        self._factory_capability = _factory_capability
        self._lock = RLock()

    @classmethod
    def from_estimator(cls, estimator: DagMLPipelineEstimator) -> _LiveMethodsWitness:
        """Bind a witness to a fully finalized strict Methods estimator.

        Call this only after optional result-mutating operations (currently
        conformal calibration) have completed, so the witness sees the final
        outcome fingerprint.
        """

        execution = getattr(estimator, "native_training_execution_", None)
        if execution is None:
            raise DagMLNativeCoverageError("strict Methods result omitted its native training execution")
        if getattr(execution, "op_callback", object()) is not None:
            raise DagMLNativeCoverageError("strict Methods execution must not retain a Python operator callback")
        methods_inputs = getattr(execution, "methods_inputs", None)
        if not isinstance(methods_inputs, Mapping) or not methods_inputs:
            raise DagMLNativeCoverageError("strict Methods execution omitted non-empty native Methods inputs")
        library_path = getattr(execution, "methods_library_path", None)
        if not isinstance(library_path, str) or not library_path:
            raise DagMLNativeCoverageError("strict Methods execution omitted its explicit libn4m path")
        path = Path(library_path)
        if not path.is_absolute():
            raise DagMLNativeCoverageError("strict Methods execution libn4m path must be absolute")
        try:
            canonical_path = path.resolve(strict=True)
        except OSError as error:
            raise DagMLNativeCoverageError("strict Methods execution libn4m path is unavailable") from error
        if not canonical_path.is_file() or str(canonical_path) != library_path:
            raise DagMLNativeCoverageError("strict Methods execution libn4m path must be canonical and regular")

        dagml_module = getattr(estimator, "dagml_module", "dag_ml")
        if dagml_module != "dag_ml" or getattr(estimator, "native_client", None) is not None:
            raise DagMLNativeCoverageError("a live Methods witness requires the standard dag_ml facade and default Dag-ML client")
        try:
            dag_ml = importlib.import_module(dagml_module)
        except ImportError as error:  # pragma: no cover - dependency failure is environment-specific
            raise DagMLNativeCoverageError("strict Methods witness cannot import the configured Dag-ML facade") from error
        expected_type = getattr(dag_ml, "TrainingResult", None)
        if not isinstance(expected_type, type):
            raise DagMLNativeCoverageError("configured Dag-ML facade does not expose TrainingResult")
        training_result = getattr(estimator, "training_result_", None)
        if type(training_result) is not expected_type:
            raise DagMLNativeCoverageError("strict Methods execution did not retain the exact Dag-ML TrainingResult facade")
        if getattr(training_result, "is_attached", None) is not True:
            raise DagMLNativeCoverageError("strict Methods TrainingResult must be attached when its live witness is created")
        outcome_fingerprint = getattr(training_result, "outcome_fingerprint", None)
        if not isinstance(outcome_fingerprint, str) or not outcome_fingerprint:
            raise DagMLNativeCoverageError("strict Methods TrainingResult omitted its outcome fingerprint")
        if re.fullmatch(r"[0-9a-f]{64}", outcome_fingerprint) is None:
            raise DagMLNativeCoverageError("strict Methods TrainingResult outcome fingerprint must be canonical SHA-256")
        result_outcome = getattr(training_result, "outcome", None)
        result_outcome_to_dict = getattr(result_outcome, "to_dict", None)
        if callable(result_outcome_to_dict):
            result_outcome = result_outcome_to_dict()
        if not isinstance(result_outcome, dict) or result_outcome.get("outcome_fingerprint") != outcome_fingerprint:
            raise DagMLNativeCoverageError("strict Methods TrainingResult outcome does not match its canonical fingerprint")

        outcome = getattr(estimator, "training_outcome_", None)
        to_dict = getattr(outcome, "to_dict", None)
        if callable(to_dict):
            outcome = to_dict()
        if not isinstance(outcome, dict) or outcome.get("outcome_fingerprint") != outcome_fingerprint:
            raise DagMLNativeCoverageError("strict Methods TrainingResult fingerprint does not match the finalized training outcome")

        return cls(
            estimator,
            training_result,
            NativeMethodsExecutionClaim(
                schema_version=1,
                execution_entrypoint="dag_ml.execute_methods_training",
                execution_mode="methods_callback_free",
                outcome_fingerprint=outcome_fingerprint,
                methods_library_mode="explicit_absolute",
                portable_artifacts_required=True,
            ),
            _factory_capability=_WITNESS_FACTORY_CAPABILITY,
        )

    def _claim_for_estimator(self, estimator: Any) -> NativeMethodsExecutionClaim:
        """Return the claim only for the exact estimator this witness owns."""

        with self._lock:
            training_result = self._training_result
            if self._factory_capability is not _WITNESS_FACTORY_CAPABILITY:
                raise DagMLNativeCoverageError("the live Methods witness was not created by the internal factory")
            if estimator is not self._estimator:
                raise DagMLNativeCoverageError("the live Methods witness does not own this estimator")
            if (
                training_result is None
                or getattr(training_result, "is_attached", None) is not True
            ):
                raise DagMLNativeCoverageError("the live Methods witness is no longer attached")
            if getattr(self._estimator, "training_result_", None) is not training_result:
                raise DagMLNativeCoverageError("the live Methods witness no longer owns the estimator TrainingResult")
            if getattr(training_result, "outcome_fingerprint", None) != self._claim.outcome_fingerprint:
                raise DagMLNativeCoverageError("the live Methods witness no longer matches its outcome fingerprint")
            return self._claim

    @property
    def claim(self) -> NativeMethodsExecutionClaim:
        """The immutable, audit-only local execution claim."""

        return self._claim_for_estimator(self._estimator)

    def _is_live_for_estimator(self, estimator: Any) -> bool:
        """Whether this witness is still live for one exact estimator."""

        try:
            self._claim_for_estimator(estimator)
        except DagMLNativeCoverageError:
            return False
        return True

    @property
    def is_live(self) -> bool:
        """Whether the attached Dag-ML result is still held by this witness."""

        return self._is_live_for_estimator(self._estimator)

    def detach(self) -> bool:
        """Detach the exact retained Dag-ML result once and release it.

        Portable outcome/package state is preserved by Dag-ML's own detach
        contract.  Repeated calls are harmless and return ``False``.
        """

        with self._lock:
            training_result = self._training_result
            if training_result is None:
                return False
            if getattr(training_result, "is_attached", None) is False:
                self._training_result = None
                return False
            if getattr(training_result, "is_attached", None) is not True:
                raise DagMLNativeCoverageError("strict Methods TrainingResult attachment state is inconsistent")
            detach = getattr(training_result, "detach", None)
            if not callable(detach):
                raise DagMLNativeCoverageError("strict Methods TrainingResult does not expose detach()")
            detached = detach()
            if not isinstance(detached, bool):
                raise DagMLNativeCoverageError("strict Methods TrainingResult detach() must return bool")
            if detached:
                if getattr(training_result, "is_attached", None) is not False:
                    raise DagMLNativeCoverageError("strict Methods TrainingResult detach() did not release its facade")
                self._training_result = None
                return True
            if getattr(training_result, "is_attached", None) is False:
                self._training_result = None
                return False
            raise DagMLNativeCoverageError("strict Methods TrainingResult detach() reported False while still attached")

    close = detach

    def __copy__(self) -> _LiveMethodsWitness:
        raise TypeError("a live Methods witness cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> _LiveMethodsWitness:
        _ = memo
        raise TypeError("a live Methods witness cannot be deep-copied")

    def __reduce__(self) -> Any:
        raise TypeError("a live Methods witness cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        _ = protocol
        raise TypeError("a live Methods witness cannot be serialized")


__all__ = ["NativeMethodsExecutionClaim"]
