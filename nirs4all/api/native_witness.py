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


@dataclass(frozen=True, slots=True)
class NativeMethodsTerminalExecutionClaim:
    """Audit-only description of one attached strict terminal execution.

    The native receipt remains the authoritative evidence.  This projection
    only identifies the live local boundary that still owns the attached
    ``TrainingResult`` used for lifecycle management.
    """

    schema_version: int
    execution_entrypoint: str
    execution_mode: str
    outcome_fingerprint: str
    terminal_run_id: str
    receipt_fingerprint: str
    methods_library_mode: str
    portable_artifacts_required: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible audit projection."""

        return asdict(self)


_WITNESS_FACTORY_CAPABILITY = object()
_TERMINAL_WITNESS_FACTORY_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class _TerminalNativeResultComponents:
    """Exact native terminal values captured once at the facade boundary.

    ``MethodsTerminalPredictionResult.training_result`` deliberately creates a
    new Python facade on every getter access.  The first facade is only the
    sklearn fitted-state compatibility marker required by the replay adapter;
    the second, distinct facade is retained privately and is the sole
    lifecycle authority for detach/close.  The frozen terminal result and
    receipt remain native authority; this internal transport is not an
    attestation.
    """

    terminal_result: Any
    compatibility_training_result: Any
    lifecycle_training_result: Any
    terminal_receipt: Any
    portable_predictor_package: Any


def _strict_terminal_native_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Load the exact public PyO3 types required by the terminal boundary."""

    try:
        dag_ml = importlib.import_module("dag_ml")
    except ImportError as error:  # pragma: no cover - dependency failure is environment-specific
        raise DagMLNativeCoverageError("strict terminal Methods witness cannot import the standard dag_ml facade") from error
    training_type = getattr(dag_ml, "TrainingResult", None)
    terminal_type = getattr(dag_ml, "MethodsTerminalPredictionResult", None)
    receipt_type = getattr(dag_ml, "MethodsTerminalPredictionReceipt", None)
    if not isinstance(training_type, type) or not isinstance(terminal_type, type) or not isinstance(receipt_type, type):
        raise DagMLNativeCoverageError("configured DAG-ML facade lacks the strict terminal native result types")
    return training_type, terminal_type, receipt_type


def _extract_terminal_native_result_components(terminal_result: Any) -> _TerminalNativeResultComponents:
    """Type-check a raw terminal result before reading its native components."""

    training_type, terminal_type, receipt_type = _strict_terminal_native_types()
    if type(terminal_result) is not terminal_type:
        raise DagMLNativeCoverageError("strict terminal Methods execution did not return the exact frozen DAG-ML terminal result")
    compatibility_training_result = getattr(terminal_result, "training_result", None)
    lifecycle_training_result = getattr(terminal_result, "training_result", None)
    terminal_receipt = getattr(terminal_result, "terminal_receipt", None)
    portable_predictor_package = getattr(terminal_result, "portable_predictor_package", None)
    if type(compatibility_training_result) is not training_type:
        raise DagMLNativeCoverageError("strict terminal Methods result did not expose the exact Dag-ML TrainingResult facade")
    if type(lifecycle_training_result) is not training_type:
        raise DagMLNativeCoverageError("strict terminal Methods result did not expose a private Dag-ML lifecycle facade")
    if lifecycle_training_result is compatibility_training_result:
        raise DagMLNativeCoverageError(
            "strict terminal Methods result must expose a distinct private Dag-ML lifecycle facade"
        )
    if type(terminal_receipt) is not receipt_type:
        raise DagMLNativeCoverageError("strict terminal Methods result did not expose the exact frozen DAG-ML receipt")
    if portable_predictor_package is None:
        raise DagMLNativeCoverageError("strict terminal Methods result omitted its Package V2")
    return _TerminalNativeResultComponents(
        terminal_result=terminal_result,
        compatibility_training_result=compatibility_training_result,
        lifecycle_training_result=lifecycle_training_result,
        terminal_receipt=terminal_receipt,
        portable_predictor_package=portable_predictor_package,
    )


def _require_exact_terminal_receipt_binding(terminal_result: Any, terminal_receipt: Any) -> Any:
    """Return only the frozen receipt still owned by its exact native result.

    This is intentionally type- and identity-based.  It never accepts a
    decoded JSON/dict receipt, and it does not treat a snapshot as a seal.
    """

    _training_type, terminal_type, receipt_type = _strict_terminal_native_types()
    if type(terminal_result) is not terminal_type:
        raise DagMLNativeCoverageError("strict terminal Methods result is not the exact frozen DAG-ML result type")
    if type(terminal_receipt) is not receipt_type:
        raise DagMLNativeCoverageError("strict terminal Methods receipt is not the exact frozen DAG-ML receipt type")
    if getattr(terminal_result, "terminal_receipt", None) is not terminal_receipt:
        raise DagMLNativeCoverageError("strict terminal Methods result no longer owns its frozen receipt")
    return terminal_receipt


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


class _LiveMethodsTerminalWitness:
    """Live lifecycle holder for DAG-ML's frozen terminal result and receipt.

    The terminal result and receipt are retained verbatim.  Only the attached
    ``TrainingResult`` facade is detached on close; the receipt is neither
    copied, decoded nor treated as an Archive V2 persistence payload.
    """

    __slots__ = (
        "_claim",
        "_estimator",
        "_factory_capability",
        "_lock",
        "_terminal_receipt",
        "_terminal_result",
        "_training_result",
    )

    def __init__(
        self,
        estimator: Any,
        training_result: Any,
        terminal_result: Any,
        terminal_receipt: Any,
        claim: NativeMethodsTerminalExecutionClaim,
        *,
        _factory_capability: object,
    ) -> None:
        if _factory_capability is not _TERMINAL_WITNESS_FACTORY_CAPABILITY:
            raise TypeError("live terminal Methods witnesses can only be created by the verified internal factory")
        self._estimator = estimator
        self._training_result: Any | None = training_result
        self._terminal_result = terminal_result
        self._terminal_receipt = terminal_receipt
        self._claim = claim
        self._factory_capability = _factory_capability
        self._lock = RLock()

    @classmethod
    def from_terminal_components(
        cls,
        estimator: DagMLPipelineEstimator,
        components: _TerminalNativeResultComponents,
    ) -> _LiveMethodsTerminalWitness:
        """Bind verified native terminal components to one private lifecycle facade."""

        execution = getattr(estimator, "native_training_execution_", None)
        if execution is None:
            raise DagMLNativeCoverageError("strict terminal Methods result omitted its execution bundle")
        if hasattr(execution, "op_callback"):
            raise DagMLNativeCoverageError("strict terminal Methods execution must not retain a Python operator callback")
        methods_inputs = getattr(execution, "methods_inputs", None)
        if not isinstance(methods_inputs, Mapping) or not methods_inputs:
            raise DagMLNativeCoverageError("strict terminal Methods execution omitted non-empty native Methods inputs")
        library_path = getattr(execution, "methods_library_path", None)
        _require_canonical_methods_library_path(library_path, context="strict terminal Methods execution")

        dagml_module = getattr(estimator, "dagml_module", "dag_ml")
        if dagml_module != "dag_ml" or getattr(estimator, "native_client", None) is not None:
            raise DagMLNativeCoverageError(
                "a live terminal Methods witness requires the standard dag_ml facade and default Dag-ML client"
            )
        if type(components) is not _TerminalNativeResultComponents:
            raise TypeError("live terminal Methods witnesses require verified native terminal components")
        training_type, terminal_type, receipt_type = _strict_terminal_native_types()
        terminal_result = components.terminal_result
        compatibility_training_result = components.compatibility_training_result
        training_result = components.lifecycle_training_result
        terminal_receipt = components.terminal_receipt
        if type(terminal_result) is not terminal_type:
            raise DagMLNativeCoverageError("strict terminal Methods execution did not retain the exact frozen DAG-ML terminal result")
        if type(terminal_receipt) is not receipt_type:
            raise DagMLNativeCoverageError("strict terminal Methods result did not retain the exact frozen DAG-ML receipt")
        _require_exact_terminal_receipt_binding(terminal_result, terminal_receipt)
        if type(compatibility_training_result) is not training_type:
            raise DagMLNativeCoverageError("strict terminal Methods result did not retain its replay compatibility facade")
        if type(training_result) is not training_type or training_result is compatibility_training_result:
            raise DagMLNativeCoverageError("strict terminal Methods result did not retain a distinct private lifecycle facade")
        _require_attached_outcome(training_result, estimator, context="strict terminal Methods")

        terminal_run_id = getattr(terminal_receipt, "terminal_run_id", None)
        receipt_fingerprint = getattr(terminal_receipt, "receipt_fingerprint", None)
        run_id = getattr(execution, "run_id", None)
        expected_terminal_run_id = (
            f"{run_id}:methods-terminal-predict" if isinstance(run_id, str) and run_id else None
        )
        if not isinstance(terminal_run_id, str) or terminal_run_id != expected_terminal_run_id:
            raise DagMLNativeCoverageError("strict terminal Methods receipt is not bound to this terminal RunId")
        if not isinstance(receipt_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", receipt_fingerprint) is None:
            raise DagMLNativeCoverageError("strict terminal Methods receipt omitted a canonical fingerprint")
        outcome_fingerprint = getattr(training_result, "outcome_fingerprint", None)
        if not isinstance(outcome_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", outcome_fingerprint) is None:
            raise DagMLNativeCoverageError("strict terminal Methods TrainingResult omitted its canonical outcome fingerprint")
        return cls(
            estimator,
            training_result,
            terminal_result,
            terminal_receipt,
            NativeMethodsTerminalExecutionClaim(
                schema_version=1,
                execution_entrypoint="dag_ml.execute_methods_cv_refit_terminal_predict",
                execution_mode="methods_cv_refit_terminal_predict_callback_free",
                outcome_fingerprint=outcome_fingerprint,
                terminal_run_id=terminal_run_id,
                receipt_fingerprint=receipt_fingerprint,
                methods_library_mode="explicit_absolute",
                portable_artifacts_required=True,
            ),
            _factory_capability=_TERMINAL_WITNESS_FACTORY_CAPABILITY,
        )

    def _claim_for_estimator(self, estimator: Any) -> NativeMethodsTerminalExecutionClaim:
        """Return a live claim only while the exact TrainingResult is attached."""

        with self._lock:
            training_result = self._training_result
            if self._factory_capability is not _TERMINAL_WITNESS_FACTORY_CAPABILITY:
                raise DagMLNativeCoverageError("the live terminal Methods witness was not created by the internal factory")
            if estimator is not self._estimator:
                raise DagMLNativeCoverageError("the live terminal Methods witness does not own this estimator")
            if training_result is None or getattr(training_result, "is_attached", None) is not True:
                raise DagMLNativeCoverageError("the live terminal Methods witness is no longer attached")
            if getattr(training_result, "outcome_fingerprint", None) != self._claim.outcome_fingerprint:
                raise DagMLNativeCoverageError("the live terminal Methods witness no longer matches its outcome fingerprint")
            _require_attached_outcome(training_result, self._estimator, context="strict terminal Methods")
            receipt = _require_exact_terminal_receipt_binding(self._terminal_result, self._terminal_receipt)
            if getattr(receipt, "terminal_run_id", None) != self._claim.terminal_run_id:
                raise DagMLNativeCoverageError("the live terminal Methods receipt no longer matches its terminal RunId")
            if getattr(receipt, "receipt_fingerprint", None) != self._claim.receipt_fingerprint:
                raise DagMLNativeCoverageError("the live terminal Methods receipt no longer matches its fingerprint")
            return self._claim

    def _terminal_result_for_estimator(self, estimator: Any) -> Any:
        """Return the retained raw result only for this result's presentation path.

        This remains available after lifecycle detach because the raw result is
        the native owner of the historical frozen receipt.  It does not expose
        a new claim and never reconstructs a Python receipt wrapper.
        """

        with self._lock:
            if self._factory_capability is not _TERMINAL_WITNESS_FACTORY_CAPABILITY:
                raise DagMLNativeCoverageError("the live terminal Methods witness was not created by the internal factory")
            if estimator is not self._estimator:
                raise DagMLNativeCoverageError("the live terminal Methods witness does not own this estimator")
            _require_exact_terminal_receipt_binding(self._terminal_result, self._terminal_receipt)
            return self._terminal_result

    def _terminal_receipt_for_estimator(self, estimator: Any) -> Any:
        """Return the exact frozen receipt without making a new lifecycle claim."""

        self._terminal_result_for_estimator(estimator)
        return self._terminal_receipt

    @property
    def claim(self) -> NativeMethodsTerminalExecutionClaim:
        """The immutable audit-only description of the attached terminal call."""

        return self._claim_for_estimator(self._estimator)

    def _is_live_for_estimator(self, estimator: Any) -> bool:
        try:
            self._claim_for_estimator(estimator)
        except DagMLNativeCoverageError:
            return False
        return True

    @property
    def is_live(self) -> bool:
        """Whether the lifecycle TrainingResult facade is still attached."""

        return self._is_live_for_estimator(self._estimator)

    def detach(self) -> bool:
        """Detach only the exact lifecycle facade, retaining native receipt objects."""

        with self._lock:
            training_result = self._training_result
            if training_result is None:
                return False
            if getattr(training_result, "is_attached", None) is False:
                self._training_result = None
                return False
            if getattr(training_result, "is_attached", None) is not True:
                raise DagMLNativeCoverageError("strict terminal Methods TrainingResult attachment state is inconsistent")
            detach = getattr(training_result, "detach", None)
            if not callable(detach):
                raise DagMLNativeCoverageError("strict terminal Methods TrainingResult does not expose detach()")
            detached = detach()
            if not isinstance(detached, bool):
                raise DagMLNativeCoverageError("strict terminal Methods TrainingResult detach() must return bool")
            if detached:
                if getattr(training_result, "is_attached", None) is not False:
                    raise DagMLNativeCoverageError("strict terminal Methods TrainingResult detach() did not release its facade")
                self._training_result = None
                return True
            if getattr(training_result, "is_attached", None) is False:
                self._training_result = None
                return False
            raise DagMLNativeCoverageError("strict terminal Methods TrainingResult detach() reported False while still attached")

    close = detach

    def __copy__(self) -> _LiveMethodsTerminalWitness:
        raise TypeError("a live terminal Methods witness cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> _LiveMethodsTerminalWitness:
        _ = memo
        raise TypeError("a live terminal Methods witness cannot be deep-copied")

    def __reduce__(self) -> Any:
        raise TypeError("a live terminal Methods witness cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        _ = protocol
        raise TypeError("a live terminal Methods witness cannot be serialized")


def _require_canonical_methods_library_path(library_path: Any, *, context: str) -> None:
    """Verify a terminal witness retained its explicit native runtime path."""

    if not isinstance(library_path, str) or not library_path:
        raise DagMLNativeCoverageError(f"{context} omitted its explicit libn4m path")
    path = Path(library_path)
    if not path.is_absolute():
        raise DagMLNativeCoverageError(f"{context} libn4m path must be absolute")
    try:
        canonical_path = path.resolve(strict=True)
    except OSError as error:
        raise DagMLNativeCoverageError(f"{context} libn4m path is unavailable") from error
    if not canonical_path.is_file() or str(canonical_path) != library_path:
        raise DagMLNativeCoverageError(f"{context} libn4m path must be canonical and regular")


def _require_attached_outcome(training_result: Any, estimator: Any, *, context: str) -> None:
    """Validate the native outcome only through the documented TrainingResult facade."""

    if getattr(training_result, "is_attached", None) is not True:
        raise DagMLNativeCoverageError(f"{context} TrainingResult must be attached when its live witness is created")
    outcome_fingerprint = getattr(training_result, "outcome_fingerprint", None)
    if not isinstance(outcome_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", outcome_fingerprint) is None:
        raise DagMLNativeCoverageError(f"{context} TrainingResult omitted its canonical outcome fingerprint")
    result_outcome = getattr(training_result, "outcome", None)
    result_outcome_to_dict = getattr(result_outcome, "to_dict", None)
    if callable(result_outcome_to_dict):
        result_outcome = result_outcome_to_dict()
    if not isinstance(result_outcome, dict) or result_outcome.get("outcome_fingerprint") != outcome_fingerprint:
        raise DagMLNativeCoverageError(f"{context} TrainingResult outcome does not match its canonical fingerprint")
    outcome = getattr(estimator, "training_outcome_", None)
    to_dict = getattr(outcome, "to_dict", None)
    if callable(to_dict):
        outcome = to_dict()
    if not isinstance(outcome, dict) or outcome.get("outcome_fingerprint") != outcome_fingerprint:
        raise DagMLNativeCoverageError(f"{context} TrainingResult fingerprint does not match the finalized training outcome")


__all__ = ["NativeMethodsExecutionClaim", "NativeMethodsTerminalExecutionClaim"]
