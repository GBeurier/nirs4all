"""Target-free raw-array replay for portable Methods N4MM packages.

This is deliberately a narrow bridge, not a second scheduler.  DAG-ML owns
the package, request signatures, data-view planning and artifact lifecycle;
``pls4all`` owns N4MM import and numerical prediction.  The host only binds
the caller's exact X rows and identities into the V2 PREDICT cohort.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .envelope import sample_relations
from .estimator import DagMLPipelineEstimator, DagMLReplayExecution
from .fit_identity import DagMLPredictIdentityFrame
from .identity import IdentityMap, SampleIdentity
from .methods_replay import MethodsN4mmReplayCallbacks, MethodsPortableReplayError
from .native_client import DagMLNativeCoverageError


class _RawArrayPredictResolver:
    """Resolve the exact caller-supplied X rows by stable sample id.

    It purposely has no target accessor.  A PREDICT replay must not create a
    target sentinel merely to reuse a fit-time resolver.
    """

    def __init__(self, X: Any, identity: IdentityMap) -> None:
        values = np.asarray(X)
        if values.ndim != 2 or len(values) != len(identity.identities):
            raise ValueError("raw Methods replay requires a 2D X aligned with its PREDICT identity frame")
        self._values = values
        self._row_by_id = {sample.observation_id: sample.sample_int for sample in identity.identities}

    def resolve_features(
        self,
        observation_ids: list[str],
        *,
        include_augmented: bool,
    ) -> dict[str, Any]:
        if include_augmented:
            raise MethodsPortableReplayError("portable Methods PREDICT must not request augmented rows")
        try:
            rows = [self._row_by_id[sample_id] for sample_id in observation_ids]
        except KeyError as error:
            raise MethodsPortableReplayError(f"portable Methods PREDICT requested an unknown sample identity {error.args[0]!r}") from error
        return {"values": self._values[rows]}


@dataclass(frozen=True)
class RawArrayMethodsReplayCompiler:
    """Compile a Methods-only PREDICT replay from a fitted raw-array estimator.

    The fit compiler must have retained its validated training envelopes on the
    estimator.  Only a Package V2 in ``portable_required`` mode, with native
    raw artifacts, is accepted; host sidecars and Python execution callbacks
    are never used as fallbacks.
    """

    request_id: str = "replay:nirs4all.raw_methods_predict"
    outcome_id: str = "outcome:nirs4all.raw_methods_predict"
    run_id: str = "run:nirs4all.raw_methods_predict"
    dagml_module: str = "dag_ml"
    context_type: type[Any] | None = None
    model_type: type[Any] | None = None

    def compile_replay(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        *,
        mode: str,
        identity_frame: DagMLPredictIdentityFrame,
    ) -> DagMLReplayExecution:
        if mode != "predict":
            raise DagMLNativeCoverageError("RawArrayMethodsReplayCompiler supports PREDICT only; probability and explanation replay require an explicit native output contract")
        package = _contract_mapping(estimator.predictor_package_, "portable predictor package")
        _require_native_methods_package(package)
        training_execution = getattr(estimator, "training_execution_", None)
        templates = getattr(training_execution, "data_envelopes", None)
        if not isinstance(templates, Mapping):
            raise DagMLNativeCoverageError("portable Methods replay requires the validated fit-time data envelopes retained by the estimator")

        data_keys = _data_requirement_keys(package)
        output_bindings = _output_bindings(package)
        missing_templates = sorted(set(data_keys).difference(templates))
        if missing_templates:
            raise DagMLNativeCoverageError("portable Methods replay is missing validated fit-time envelopes for " + ", ".join(missing_templates))
        identity = _identity_from_predict_frame(identity_frame)
        relations = sample_relations(
            identity,
            metadata_by_sample=_metadata_by_sample(identity_frame),
            group_by_sample=_groups_by_sample(identity_frame),
        )
        facade = _import_facade(self.dagml_module)
        cohort_request = {
            "role": "inference",
            "relations": relations,
            "target_names": _target_names(output_bindings[0]),
            "data_content_fingerprint": identity_frame.data_content_fingerprint,
        }
        data_envelopes = {
            key: _predict_envelope(
                templates[key],
                facade=facade,
                cohort_request=cohort_request,
                data_content_fingerprint=identity_frame.data_content_fingerprint,
            )
            for key in data_keys
        }
        request = {
            "schema_version": 1,
            "request_id": self.request_id,
            "source_outcome_fingerprint": _source_outcome_fingerprint(package),
            "phase": "predict",
            "data_envelope_keys": data_keys,
            "output_binding_ids": [_binding_id(binding) for binding in output_bindings],
            "request_fingerprint": "",
        }
        signer = getattr(facade, "sign_training_replay_request", None)
        if not callable(signer):
            raise DagMLNativeCoverageError("native DAG-ML facade does not expose sign_training_replay_request(); install the PREDICT cohort binding")
        signed = signer(request)
        request = _contract_mapping(signed, "signed training replay request")
        callbacks = MethodsN4mmReplayCallbacks(
            _RawArrayPredictResolver(X, identity),
            target_names_by_node={_node_id(binding): _target_names(binding) for binding in output_bindings},
            context_type=self.context_type,
            model_type=self.model_type,
        )
        return DagMLReplayExecution(
            request=request,
            data_envelopes=data_envelopes,
            artifact_handles={},
            op_callback=callbacks.op_callback,
            artifact_callback=callbacks.artifact_callback,
            outcome_id=self.outcome_id,
            run_id=self.run_id,
            diagnostics={
                "nirs4all_replay": "raw_array_methods_n4mm_predict_v1",
                "nirs4all_predict_identity_fingerprint": identity_frame.fingerprint,
            },
        )


def _contract_mapping(value: Any, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise DagMLNativeCoverageError(f"{label} must be a JSON object")
    return dict(value)


def _require_native_methods_package(package: Mapping[str, Any]) -> None:
    if package.get("schema_version") != 2:
        raise DagMLNativeCoverageError("portable Methods replay requires PortablePredictorPackage V2")
    if package.get("fitted_artifact_mode") != "portable_required":
        raise DagMLNativeCoverageError("portable Methods replay refuses host-sidecar predictor packages")
    bundle = package.get("execution_bundle")
    raw_payloads = bundle.get("raw_artifact_payloads") if isinstance(bundle, Mapping) else None
    if not isinstance(raw_payloads, Mapping) or not raw_payloads:
        raise DagMLNativeCoverageError("portable Methods replay requires durable raw N4MM artifact payloads")
    bindings = package.get("artifact_bindings")
    if not isinstance(bindings, list) or not bindings or any(not isinstance(binding, Mapping) or binding.get("load_mode") != "native_portable" for binding in bindings):
        raise DagMLNativeCoverageError("portable Methods replay requires native-portable artifact bindings only")


def _data_requirement_keys(package: Mapping[str, Any]) -> list[str]:
    identities = package.get("data_identities")
    if not isinstance(identities, list) or not identities:
        raise DagMLNativeCoverageError("portable Methods replay package has no data identities")
    keys = [item.get("requirement_key") for item in identities if isinstance(item, Mapping)]
    if len(keys) != len(identities) or any(not isinstance(key, str) or not key for key in keys):
        raise DagMLNativeCoverageError("portable Methods replay package has invalid data identity keys")
    text_keys = [str(key) for key in keys]
    if text_keys != sorted(set(text_keys)):
        raise DagMLNativeCoverageError("portable Methods replay package data identity keys are not canonical")
    return text_keys


def _output_bindings(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = package.get("output_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise DagMLNativeCoverageError("portable Methods replay package has no output bindings")
    if len(bindings) != 1:
        raise DagMLNativeCoverageError("raw portable Methods replay supports exactly one selected output binding")
    out = [_contract_mapping(binding, "portable Methods output binding") for binding in bindings]
    binding_ids = [_binding_id(binding) for binding in out]
    if binding_ids != sorted(binding_ids):
        raise DagMLNativeCoverageError("portable Methods replay package output bindings are not canonical")
    for binding in out:
        _binding_id(binding)
        _node_id(binding)
        _target_names(binding)
    return out


def _source_outcome_fingerprint(package: Mapping[str, Any]) -> str:
    outcome = package.get("training_outcome")
    fingerprint = outcome.get("outcome_fingerprint") if isinstance(outcome, Mapping) else None
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise DagMLNativeCoverageError("portable Methods replay package has no source outcome fingerprint")
    return fingerprint


def _binding_id(binding: Mapping[str, Any]) -> str:
    value = binding.get("binding_id")
    if not isinstance(value, str) or not value:
        raise DagMLNativeCoverageError("portable Methods replay output binding lacks a stable binding id")
    return value


def _node_id(binding: Mapping[str, Any]) -> str:
    value = binding.get("node_id")
    if not isinstance(value, str) or not value:
        raise DagMLNativeCoverageError("portable Methods replay output binding lacks a stable node id")
    return value


def _target_names(binding: Mapping[str, Any]) -> list[str]:
    value = binding.get("target_names")
    if not isinstance(value, list) or not value or not all(isinstance(name, str) and name for name in value):
        raise DagMLNativeCoverageError("portable Methods replay output binding has no target schema")
    return list(value)


def _identity_from_predict_frame(frame: DagMLPredictIdentityFrame) -> IdentityMap:
    identities = tuple(SampleIdentity(index, index, sample_id, sample_id, False) for index, sample_id in enumerate(frame.sample_ids))
    return IdentityMap(
        fingerprint=frame.fingerprint,
        identities=identities,
        _to_int={sample.observation_id: sample.sample_int for sample in identities},
        _to_wire={sample.sample_int: sample.observation_id for sample in identities},
    )


def _metadata_by_sample(frame: DagMLPredictIdentityFrame) -> dict[str, dict[int, Any]]:
    columns: dict[str, dict[int, Any]] = {}
    for index, row in enumerate(frame.metadata_rows):
        for name, value in row.items():
            columns.setdefault(name, {})[index] = value
    return columns


def _groups_by_sample(frame: DagMLPredictIdentityFrame) -> dict[int, str]:
    return {index: group for index, group in enumerate(frame.groups) if group is not None}


def _import_facade(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise DagMLNativeCoverageError(f"native DAG-ML facade {module_name!r} is not importable for portable Methods replay") from error


def _predict_envelope(
    template: Any,
    *,
    facade: Any,
    cohort_request: Mapping[str, Any],
    data_content_fingerprint: str,
) -> dict[str, Any]:
    envelope = copy.deepcopy(_contract_mapping(template, "fit-time data envelope"))
    envelope["data_content_fingerprint"] = data_content_fingerprint
    envelope.pop("target_content_fingerprint", None)
    attach = getattr(facade, "attach_predict_cohort_to_envelope", None)
    if not callable(attach):
        raise DagMLNativeCoverageError("native DAG-ML facade does not expose attach_predict_cohort_to_envelope(); install the PREDICT cohort binding")
    return _contract_mapping(attach(envelope, dict(cohort_request)), "PREDICT data envelope")


__all__ = ["RawArrayMethodsReplayCompiler"]
