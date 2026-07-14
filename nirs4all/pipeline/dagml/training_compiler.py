"""Prepared DAG-ML training-contract compiler for the native estimator.

P3-R0 deliberately does not lower arbitrary nirs4all pipeline syntax yet.  It
closes the next safe seam: an already-built DAG-ML training contract bundle can
be injected into :class:`~nirs4all.pipeline.dagml.estimator.DagMLPipelineEstimator`
through a small compiler object, with fit identity diagnostics added in one
place before the native binding call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .estimator import DagMLPipelineEstimator, DagMLTrainingExecution
from .fit_identity import DagMLFitIdentityFrame
from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request

_COMPAT_IDENTITY_WARNING = (
    "native DAG-ML fit is using deterministic compatibility sample ids derived from X/y content and row position; explicit sample_ids are required for future conformal, leakage, and exchangeability claims"
)


@dataclass(frozen=True)
class DagMLPreparedTrainingContracts:
    """Opaque DAG-ML training contracts already prepared by a trusted builder.

    The fields map one-to-one to ``DagMLTrainingExecution`` and ultimately to
    ``dag_ml.execute_training(...)``.  This type performs only seam-level
    validation: object identity, required text ids, callback callability, and
    diagnostics normalization.  DAG-ML remains the authority for graph,
    relation, envelope, scoring, and influence semantics.
    """

    request: Any
    data_envelopes: Mapping[str, Any]
    relations: Any
    training_influence: Any
    op_callback: Callable[[Any], Any]
    outcome_id: str
    run_id: str
    bundle_id: str
    warnings: Sequence[str] = ()
    diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DagMLTrainingRequestContracts:
    """Executable contract bundle whose request still needs local assembly."""

    request_spec: DagMLTrainingRequestSpec
    data_envelopes: Mapping[str, Any]
    relations: Any
    training_influence: Any
    op_callback: Callable[[Any], Any]
    outcome_id: str
    run_id: str
    bundle_id: str
    warnings: Sequence[str] = ()
    diagnostics: Mapping[str, Any] | None = None

    def to_prepared(self) -> DagMLPreparedTrainingContracts:
        """Assemble the signed request and return prepared execution contracts."""

        return DagMLPreparedTrainingContracts(
            request=assemble_training_request(self.request_spec),
            data_envelopes=self.data_envelopes,
            relations=self.relations,
            training_influence=self.training_influence,
            op_callback=self.op_callback,
            outcome_id=self.outcome_id,
            run_id=self.run_id,
            bundle_id=self.bundle_id,
            warnings=self.warnings,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True)
class PreparedDagMLTrainingCompiler:
    """Compiler adapter for pre-built DAG-ML training contracts."""

    contracts: DagMLPreparedTrainingContracts
    include_fit_identity_diagnostics: bool = True
    additional_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def compile_fit(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution:
        """Return native execution inputs for one estimator ``fit`` call."""

        _ = (estimator, X, y, sample_ids, groups, metadata)
        return compile_prepared_training_contracts(
            self.contracts,
            identity_frame=identity_frame,
            include_fit_identity_diagnostics=self.include_fit_identity_diagnostics,
            additional_diagnostics=self.additional_diagnostics,
        )


@dataclass(frozen=True)
class DagMLTrainingContractFactoryCompiler:
    """Compiler adapter around a callable that builds prepared contracts.

    This is the intended handoff point for P3-R1 lowerers: the lowerer owns
    pipeline syntax, dataset envelopes, relations, influence, and callbacks; the
    shared seam below still validates and annotates the resulting native
    contract before the estimator calls DAG-ML.
    """

    factory: Callable[..., DagMLPreparedTrainingContracts]
    include_fit_identity_diagnostics: bool = True
    additional_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def compile_fit(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution:
        """Build, validate and package contracts for one estimator ``fit`` call."""

        contracts = self.factory(
            estimator,
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            identity_frame=identity_frame,
        )
        return compile_prepared_training_contracts(
            contracts,
            identity_frame=identity_frame,
            include_fit_identity_diagnostics=self.include_fit_identity_diagnostics,
            additional_diagnostics=self.additional_diagnostics,
        )


@dataclass(frozen=True)
class DagMLTrainingRequestCompiler:
    """Compiler adapter for a request-spec plus executable DAG-ML contracts."""

    contracts: DagMLTrainingRequestContracts
    include_fit_identity_diagnostics: bool = True
    additional_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    dagml_module: str | None = None

    def compile_fit(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution:
        """Assemble a signed request and return native execution inputs."""

        _ = (estimator, X, y, sample_ids, groups, metadata)
        prepared = self.contracts.to_prepared()
        if self.dagml_module is not None:
            prepared = replace(prepared, request=_sign_request_if_supported(prepared.request, self.dagml_module))
        return compile_prepared_training_contracts(
            prepared,
            identity_frame=identity_frame,
            include_fit_identity_diagnostics=self.include_fit_identity_diagnostics,
            additional_diagnostics=self.additional_diagnostics,
        )


def compile_prepared_training_contracts(
    contracts: DagMLPreparedTrainingContracts,
    *,
    identity_frame: DagMLFitIdentityFrame,
    include_fit_identity_diagnostics: bool = True,
    additional_diagnostics: Mapping[str, Any] | None = None,
) -> DagMLTrainingExecution:
    """Validate and package prepared DAG-ML training contracts for execution."""

    _validate_prepared_contracts(contracts)
    diagnostics = _merged_diagnostics(
        contracts,
        identity_frame=identity_frame,
        include_fit_identity_diagnostics=include_fit_identity_diagnostics,
        additional_diagnostics=additional_diagnostics,
    )
    warnings = _merged_warnings(contracts, identity_frame)
    return DagMLTrainingExecution(
        request=contracts.request,
        data_envelopes=dict(contracts.data_envelopes),
        relations=contracts.relations,
        training_influence=contracts.training_influence,
        op_callback=contracts.op_callback,
        outcome_id=contracts.outcome_id,
        run_id=contracts.run_id,
        bundle_id=contracts.bundle_id,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _validate_prepared_contracts(contracts: DagMLPreparedTrainingContracts) -> None:
    if not isinstance(contracts, DagMLPreparedTrainingContracts):
        raise TypeError("contracts must be a DagMLPreparedTrainingContracts instance")
    if not isinstance(contracts.data_envelopes, Mapping):
        raise TypeError("data_envelopes must be a mapping keyed by DAG-ML data id")
    for key in contracts.data_envelopes:
        if not isinstance(key, str) or not key:
            raise ValueError("data_envelopes keys must be non-empty strings")
    if not callable(contracts.op_callback):
        raise TypeError("op_callback must be callable")
    _require_non_empty_text("outcome_id", contracts.outcome_id)
    _require_non_empty_text("run_id", contracts.run_id)
    _require_non_empty_text("bundle_id", contracts.bundle_id)
    for warning in contracts.warnings:
        if not isinstance(warning, str) or not warning:
            raise ValueError("warnings must contain only non-empty strings")
    if contracts.diagnostics is not None and not isinstance(contracts.diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping when provided")


def _sign_request_if_supported(request: Any, dagml_module: str) -> Any:
    import importlib

    dag_ml = importlib.import_module(dagml_module)
    signer = getattr(dag_ml, "sign_training_request", None)
    if callable(signer):
        return signer(request).to_dict()
    return request


def _require_non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _merged_diagnostics(
    contracts: DagMLPreparedTrainingContracts,
    *,
    identity_frame: DagMLFitIdentityFrame,
    include_fit_identity_diagnostics: bool,
    additional_diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = dict(contracts.diagnostics or {})
    diagnostics.update(additional_diagnostics or {})
    if include_fit_identity_diagnostics:
        diagnostics.setdefault("nirs4all_fit_identity_fingerprint", identity_frame.fingerprint)
        diagnostics.setdefault("nirs4all_fit_identity_n_samples", identity_frame.n_samples)
        diagnostics.setdefault("nirs4all_fit_identity_explicit_sample_ids", identity_frame.explicit_sample_ids)
    return diagnostics


def _merged_warnings(
    contracts: DagMLPreparedTrainingContracts,
    identity_frame: DagMLFitIdentityFrame,
) -> tuple[str, ...]:
    warnings = list(contracts.warnings)
    if not identity_frame.explicit_sample_ids and _COMPAT_IDENTITY_WARNING not in warnings:
        warnings.append(_COMPAT_IDENTITY_WARNING)
    return tuple(warnings)


__all__ = [
    "DagMLPreparedTrainingContracts",
    "DagMLTrainingContractFactoryCompiler",
    "DagMLTrainingRequestCompiler",
    "DagMLTrainingRequestContracts",
    "PreparedDagMLTrainingCompiler",
    "compile_prepared_training_contracts",
]
