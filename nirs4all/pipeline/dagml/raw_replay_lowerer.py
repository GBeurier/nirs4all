"""Lower an X-only raw cohort into a strict Methods portable replay.

This is deliberately the PREDICT counterpart to :mod:`raw_training_lowerer`.
It does not reconstruct a training graph, reuse the training relation, invent a
target hash, or calculate TCV1 fingerprints in Python.  The frozen Package V2
supplies the schema/plan requirements; the current cohort supplies only its
explicit identities, feature bytes and relations; DAG-ML signs the final replay
request.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .estimator import DagMLPipelineEstimator, DagMLReplayExecution
from .fit_identity import DagMLCalibrationIdentityFrame, DagMLPredictIdentityFrame
from .methods_replay import MethodsN4mmReplayCallbacks


class RawArrayMethodsReplayError(RuntimeError):
    """A raw-array Methods replay package cannot be safely lowered."""


@dataclass(frozen=True)
class _ArrayReplayResolver:
    """Resolve exactly the caller's current feature rows by stable sample id."""

    values_by_sample_id: Mapping[str, np.ndarray]

    def resolve_features(
        self,
        sample_ids: list[str],
        *,
        include_augmented: bool,
    ) -> dict[str, np.ndarray]:
        if include_augmented:
            raise RawArrayMethodsReplayError(
                "portable raw-array Methods replay does not support augmented inputs"
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise RawArrayMethodsReplayError("DAG-ML replay requested duplicate sample identities")
        try:
            rows = [self.values_by_sample_id[sample_id] for sample_id in sample_ids]
        except KeyError as error:
            raise RawArrayMethodsReplayError(
                "DAG-ML replay requested a sample identity absent from the current cohort"
            ) from error
        return {"values": np.ascontiguousarray(np.vstack(rows), dtype=float)}


@dataclass(frozen=True)
class RawArrayMethodsReplayCompiler:
    """Compile a PREDICT cohort for one native Methods Package V2.

    The class is intentionally single-output and raw-Matrix-only.  More general
    data plans need a separate, explicitly attested materializer rather than a
    permissive fallback here.  Ordinary PREDICT cohorts carry no target proof;
    a typed calibration frame may carry an independently measured-target proof
    while still keeping those values out of provider execution inputs.
    """

    package: Any
    outcome_id: str = "outcome:nirs4all.raw_predict"
    run_id: str = "run:nirs4all.raw_predict"
    request_id: str = "replay:nirs4all.raw_predict"
    dagml_module: str = "dag_ml"
    fallback: Any = None
    methods_library_path: str | None = None

    def compile_replay(
        self,
        estimator: DagMLPipelineEstimator | None,
        X: Any,
        *,
        mode: str,
        identity_frame: DagMLPredictIdentityFrame | DagMLCalibrationIdentityFrame,
    ) -> DagMLReplayExecution:
        """Return native replay inputs for the current, target-free cohort."""

        if mode != "predict":
            raise RawArrayMethodsReplayError(
                "raw-array Methods portable replay supports PREDICT only"
            )
        if not identity_frame.explicit_sample_ids:
            raise RawArrayMethodsReplayError(
                "raw-array Methods portable replay requires explicit current sample_ids"
            )
        values = np.ascontiguousarray(np.asarray(X, dtype=float))
        if values.ndim != 2 or values.shape[0] != identity_frame.n_samples:
            raise RawArrayMethodsReplayError(
                "current X must be a two-dimensional matrix aligned with sample_ids"
            )
        if not np.isfinite(values).all():
            raise RawArrayMethodsReplayError("current X contains a non-finite value")

        package = _package_document(self.package)
        _require_native_methods_package(package)
        bundle = _object(package, "execution_bundle")
        requirements = _requirements(bundle)
        binding = _single_output_binding(package)
        relations = _current_relations(identity_frame)
        dag_ml = importlib.import_module(self.dagml_module)
        relation_fingerprint = _relation_fingerprint(dag_ml, relations)
        envelopes = {
            key: {
                "schema_version": 1,
                "schema_fingerprint": requirement["schema_fingerprint"],
                "plan_fingerprint": requirement["plan_fingerprint"],
                "relation_fingerprint": relation_fingerprint,
                "data_content_fingerprint": identity_frame.data_content_fingerprint,
                "target_content_fingerprint": getattr(identity_frame, "target_content_fingerprint", None),
                "coordinator_relations": relations,
            }
            for key, requirement in requirements.items()
        }
        unsigned_request = {
            "schema_version": 1,
            "request_id": self.request_id,
            "source_outcome_fingerprint": _source_outcome_fingerprint(package),
            "phase": "PREDICT",
            "data_envelope_keys": sorted(envelopes),
            "output_binding_ids": [binding["binding_id"]],
            "request_fingerprint": "0" * 64,
        }
        signer = getattr(dag_ml, "sign_training_replay_request", None)
        if not callable(signer):
            raise RawArrayMethodsReplayError(
                "installed dag_ml lacks native replay-request signing; upgrade DAG-ML"
            )
        request = signer(unsigned_request)
        methods_inputs = {
            key: {
                "sample_ids": list(identity_frame.sample_ids),
                "x": values.tolist(),
                "target_names": list(binding["target_names"]),
            }
            for key in requirements
        }
        if self.methods_library_path is None:
            # Compatibility tests and pre-published bindings may still use the
            # old explicit hydration callback.  The public fit path always
            # supplies a library path and therefore never takes this branch.
            resolver = _ArrayReplayResolver(
                dict(zip(identity_frame.sample_ids, values, strict=True))
            )
            callbacks = MethodsN4mmReplayCallbacks(
                resolver,
                target_names_by_node={binding["node_id"]: list(binding["target_names"])},
                fallback=self.fallback,
            )
            return DagMLReplayExecution(
                request=request,
                data_envelopes=envelopes,
                artifact_handles={},
                op_callback=callbacks.op_callback,
                artifact_callback=callbacks.artifact_callback,
                cleanup=callbacks.close,
                outcome_id=self.outcome_id,
                run_id=self.run_id,
            )
        return DagMLReplayExecution(
            request=request,
            data_envelopes=envelopes,
            artifact_handles={},
            op_callback=None,
            outcome_id=self.outcome_id,
            run_id=self.run_id,
            methods_inputs=methods_inputs,
            methods_library_path=self.methods_library_path,
        )


@dataclass(frozen=True)
class RawArrayMethodsPortableRefitReplayCompiler:
    """Compile an identity-bound PREDICT request for one Package V3 child.

    Package V3 is deliberately not coerced into the V2 predictor shape: it
    carries a fresh full-refit outcome and Core validates it through its
    dedicated replay entry point.  This lowerer derives only current-cohort
    envelopes and numeric provider inputs; it never interprets N4MM bytes.
    """

    package: Any
    outcome_id: str = "outcome:nirs4all.raw_refit_predict"
    run_id: str = "run:nirs4all.raw_refit_predict"
    request_id: str = "replay:nirs4all.raw_refit_predict"
    dagml_module: str = "dag_ml"
    methods_library_path: str | None = None

    def compile_replay(
        self,
        _estimator: DagMLPipelineEstimator | None,
        X: Any,
        *,
        mode: str,
        identity_frame: DagMLPredictIdentityFrame,
    ) -> DagMLReplayExecution:
        """Return PREDICT-only native V3 replay inputs for the current cohort."""

        if mode != "predict":
            raise RawArrayMethodsReplayError(
                "raw-array Methods Package V3 replay supports PREDICT only"
            )
        if self.methods_library_path is None:
            raise RawArrayMethodsReplayError(
                "raw-array Methods Package V3 replay requires an explicit libn4m path"
            )
        if not identity_frame.explicit_sample_ids:
            raise RawArrayMethodsReplayError(
                "raw-array Methods Package V3 replay requires explicit current sample_ids"
            )
        values = np.ascontiguousarray(np.asarray(X, dtype=float))
        if values.ndim != 2 or values.shape[0] != identity_frame.n_samples:
            raise RawArrayMethodsReplayError(
                "current X must be a two-dimensional matrix aligned with sample_ids"
            )
        if not np.isfinite(values).all():
            raise RawArrayMethodsReplayError("current X contains a non-finite value")

        package = validate_native_methods_refit_package_v3(self.package)
        outcome = _object(package, "outcome")
        plan = _object(outcome, "effective_plan")
        binding = _single_refit_v3_output_binding(outcome)
        requirements = _refit_v3_requirements(plan)
        relations = _current_relations(identity_frame)
        dag_ml = importlib.import_module(self.dagml_module)
        relation_fingerprint = _relation_fingerprint(dag_ml, relations)
        envelopes = {
            key: {
                "schema_version": 1,
                "schema_fingerprint": requirement["schema_fingerprint"],
                "plan_fingerprint": requirement["plan_fingerprint"],
                "relation_fingerprint": relation_fingerprint,
                "data_content_fingerprint": identity_frame.data_content_fingerprint,
                "target_content_fingerprint": None,
                "coordinator_relations": relations,
            }
            for key, requirement in requirements.items()
        }
        signer = getattr(dag_ml, "sign_training_replay_request", None)
        if not callable(signer):
            raise RawArrayMethodsReplayError(
                "installed dag_ml lacks native replay-request signing; upgrade DAG-ML"
            )
        request = signer(
            {
                "schema_version": 1,
                "request_id": self.request_id,
                "source_outcome_fingerprint": _refit_v3_outcome_fingerprint(outcome),
                "phase": "PREDICT",
                "data_envelope_keys": sorted(envelopes),
                "output_binding_ids": [binding["binding_id"]],
                "request_fingerprint": "0" * 64,
            }
        )
        methods_inputs = {
            key: {
                "sample_ids": list(identity_frame.sample_ids),
                "x": values.tolist(),
                "target_names": list(binding["target_names"]),
            }
            for key in requirements
        }
        return DagMLReplayExecution(
            request=request,
            data_envelopes=envelopes,
            artifact_handles={},
            op_callback=None,
            outcome_id=self.outcome_id,
            run_id=self.run_id,
            methods_inputs=methods_inputs,
            methods_library_path=self.methods_library_path,
        )


def _package_document(package: Any) -> dict[str, Any]:
    if hasattr(package, "to_dict") and callable(package.to_dict):
        package = package.to_dict()
    if isinstance(package, str):
        try:
            package = json.loads(package)
        except json.JSONDecodeError as error:
            raise RawArrayMethodsReplayError("portable predictor package is not JSON") from error
    if not isinstance(package, dict):
        raise RawArrayMethodsReplayError("portable predictor package must be an object")
    return package


def validate_native_methods_package(package: Any) -> dict[str, Any]:
    """Return a package only when it carries replayable Methods evidence.

    Training callers use this at their public boundary so a host-sidecar result
    cannot masquerade as a native raw pipeline and fail only during prediction
    or Archive V2 export.
    """

    document = _package_document(package)
    _require_native_methods_package(document)
    return document


def validate_native_methods_refit_package_v3(package: Any) -> dict[str, Any]:
    """Return a Package V3 only after its native raw-refit shape is present.

    DAG-ML remains the TCV1 authority.  This Python boundary merely validates
    the narrowly required shape for lowering current, explicitly identified
    arrays and refuses V2 or host-sidecar substitution before replay.
    """

    document = _package_document(package)
    if document.get("schema_version") != 3:
        raise RawArrayMethodsReplayError("raw-array Methods refit replay requires Package V3")
    outcome = _object(document, "outcome")
    bundle = _object(outcome, "execution_bundle")
    raw = bundle.get("raw_artifact_payloads")
    records = bundle.get("refit_artifacts")
    if not isinstance(raw, dict) or not raw:
        raise RawArrayMethodsReplayError("Package V3 has no durable raw Methods artifacts")
    if not isinstance(records, list) or len(records) != 1:
        raise RawArrayMethodsReplayError(
            "raw-array Methods Package V3 requires exactly one refit artifact"
        )
    artifact = _artifact_document(records[0]) if isinstance(records[0], Mapping) else {}
    artifact_id = artifact.get("id")
    if (
        artifact.get("kind") != "n4m_model"
        or not isinstance(artifact_id, str)
        or artifact_id not in raw
    ):
        raise RawArrayMethodsReplayError(
            "Package V3 N4MM refit artifact has no matching durable raw payload"
        )
    _single_refit_v3_output_binding(outcome)
    _refit_v3_requirements(_object(outcome, "effective_plan"))
    _refit_v3_outcome_fingerprint(outcome)
    return document


def _require_native_methods_package(package: Mapping[str, Any]) -> None:
    if package.get("schema_version") != 2:
        raise RawArrayMethodsReplayError("raw-array Methods replay requires Package V2")
    bundle = _object(package, "execution_bundle")
    raw = bundle.get("raw_artifact_payloads")
    if not isinstance(raw, dict) or not raw:
        raise RawArrayMethodsReplayError("Package V2 has no durable raw Methods artifacts")
    artifacts = bundle.get("refit_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RawArrayMethodsReplayError("Package V2 has no refit artifact list")
    raw_ids = {
        artifact_id
        for artifact_id, payload in raw.items()
        if isinstance(artifact_id, str) and isinstance(payload, (str, list, bytes, bytearray))
    }
    # Package V2 serializes refit artifacts as records containing ``artifact``.
    # Keep accepting the historical flattened test fixture while all public
    # production paths use the nested record shape.
    methods = [
        record
        for record in artifacts
        if isinstance(record, dict)
        and _artifact_document(record).get("kind") == "n4m_model"
    ]
    if len(methods) != 1:
        raise RawArrayMethodsReplayError(
            "raw-array Methods replay requires exactly one n4m_model refit artifact"
        )
    artifact_id = _artifact_document(methods[0]).get("id", methods[0].get("artifact_id"))
    if not isinstance(artifact_id, str) or artifact_id not in raw_ids:
        raise RawArrayMethodsReplayError(
            "Package V2 N4MM refit artifact has no matching durable raw payload"
        )


def _artifact_document(record: Mapping[str, Any]) -> Mapping[str, Any]:
    artifact = record.get("artifact")
    return artifact if isinstance(artifact, Mapping) else record


def _requirements(bundle: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw = bundle.get("data_requirements")
    if not isinstance(raw, list) or not raw:
        raise RawArrayMethodsReplayError("Package V2 has no data requirements")
    requirements: dict[str, dict[str, str]] = {}
    for requirement in raw:
        if not isinstance(requirement, dict):
            raise RawArrayMethodsReplayError("Package V2 data requirement is not an object")
        node_id = requirement.get("node_id")
        input_name = requirement.get("input_name")
        schema = requirement.get("schema_fingerprint")
        plan = requirement.get("plan_fingerprint")
        if not all(isinstance(value, str) and value for value in (node_id, input_name, schema, plan)):
            raise RawArrayMethodsReplayError("Package V2 data requirement lacks stable fingerprints")
        key = f"{node_id}.{input_name}"
        if key in requirements:
            raise RawArrayMethodsReplayError("Package V2 repeats a data requirement key")
        requirements[key] = {
            "schema_fingerprint": cast(str, schema),
            "plan_fingerprint": cast(str, plan),
        }
    return requirements


def _single_output_binding(package: Mapping[str, Any]) -> dict[str, Any]:
    bindings = package.get("output_bindings")
    if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], dict):
        raise RawArrayMethodsReplayError(
            "raw-array Methods replay requires exactly one Package V2 output binding"
        )
    binding = bindings[0]
    for field in ("binding_id", "node_id"):
        if not isinstance(binding.get(field), str) or not binding[field]:
            raise RawArrayMethodsReplayError(f"Package V2 output binding lacks {field}")
    targets = binding.get("target_names")
    if not isinstance(targets, list) or not targets or not all(isinstance(name, str) and name for name in targets):
        raise RawArrayMethodsReplayError("Package V2 output binding lacks target_names")
    return binding


def _single_refit_v3_output_binding(outcome: Mapping[str, Any]) -> dict[str, Any]:
    bindings = outcome.get("output_bindings")
    if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], dict):
        raise RawArrayMethodsReplayError(
            "Package V3 requires exactly one output binding for raw Methods replay"
        )
    binding = bindings[0]
    for field in ("binding_id", "node_id"):
        if not isinstance(binding.get(field), str) or not binding[field]:
            raise RawArrayMethodsReplayError(f"Package V3 output binding lacks {field}")
    targets = binding.get("target_names")
    if not isinstance(targets, list) or not targets or not all(
        isinstance(name, str) and name for name in targets
    ):
        raise RawArrayMethodsReplayError("Package V3 output binding lacks target_names")
    return binding


def _refit_v3_requirements(plan: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    node_plans = plan.get("node_plans")
    if not isinstance(node_plans, dict) or not node_plans:
        raise RawArrayMethodsReplayError("Package V3 effective plan has no node plans")
    requirements: dict[str, dict[str, str]] = {}
    for node in node_plans.values():
        if not isinstance(node, Mapping):
            raise RawArrayMethodsReplayError("Package V3 node plan is not an object")
        bindings = node.get("data_bindings")
        if not isinstance(bindings, list):
            raise RawArrayMethodsReplayError("Package V3 node plan lacks data bindings")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise RawArrayMethodsReplayError("Package V3 data binding is not an object")
            node_id = binding.get("node_id")
            input_name = binding.get("input_name")
            schema = binding.get("schema_fingerprint")
            plan_fingerprint = binding.get("plan_fingerprint")
            if not all(
                isinstance(value, str) and value
                for value in (node_id, input_name, schema, plan_fingerprint)
            ):
                raise RawArrayMethodsReplayError(
                    "Package V3 data binding lacks stable fingerprints"
                )
            assert isinstance(schema, str)
            assert isinstance(plan_fingerprint, str)
            key = f"{node_id}.{input_name}"
            if key in requirements:
                raise RawArrayMethodsReplayError("Package V3 repeats a data requirement key")
            requirements[key] = {
                "schema_fingerprint": schema,
                "plan_fingerprint": plan_fingerprint,
            }
    if not requirements:
        raise RawArrayMethodsReplayError("Package V3 has no data requirements")
    return requirements


def _source_outcome_fingerprint(package: Mapping[str, Any]) -> str:
    outcome = _object(package, "training_outcome")
    fingerprint = outcome.get("outcome_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RawArrayMethodsReplayError("Package V2 training outcome lacks its fingerprint")
    return fingerprint


def _refit_v3_outcome_fingerprint(outcome: Mapping[str, Any]) -> str:
    fingerprint = outcome.get("outcome_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RawArrayMethodsReplayError("Package V3 outcome lacks its fingerprint")
    return fingerprint


def _current_relations(
    identity_frame: DagMLPredictIdentityFrame | DagMLCalibrationIdentityFrame,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sample_id, group, metadata in zip(
        identity_frame.sample_ids,
        identity_frame.groups,
        identity_frame.metadata_rows,
        strict=True,
    ):
        records.append(
            {
                "observation_id": sample_id,
                "sample_id": sample_id,
                "target_id": None,
                "group_id": group,
                "origin_sample_id": None,
                "source_id": None,
                "is_augmented": False,
                "metadata": metadata,
            }
        )
    return {"records": records}


def _relation_fingerprint(dag_ml: Any, relations: dict[str, Any]) -> str:
    fingerprint = getattr(dag_ml, "sample_relation_set_fingerprint_json", None)
    if not callable(fingerprint):
        raise RawArrayMethodsReplayError(
            "installed dag_ml lacks native relation fingerprinting; upgrade DAG-ML"
        )
    value = fingerprint(json.dumps(relations, sort_keys=True, separators=(",", ":")))
    if not isinstance(value, str) or len(value) != 64:
        raise RawArrayMethodsReplayError("native DAG-ML returned an invalid relation fingerprint")
    return value


def _object(container: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = container.get(name)
    if not isinstance(value, dict):
        raise RawArrayMethodsReplayError(f"Package V2 lacks object `{name}`")
    return value


__all__ = [
    "RawArrayMethodsReplayCompiler",
    "RawArrayMethodsPortableRefitReplayCompiler",
    "RawArrayMethodsReplayError",
    "validate_native_methods_package",
    "validate_native_methods_refit_package_v3",
]
