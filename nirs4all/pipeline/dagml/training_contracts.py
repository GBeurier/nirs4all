"""Native DAG-ML training contract assembly helpers.

These helpers are the P3-R1 substrate for lowering nirs4all syntax to the
public ``dag_ml.execute_training(...)`` surface.  They intentionally assemble
and sign only the contract shell that nirs4all owns; DAG-ML remains the
authority for graph, campaign, controller, projection and runtime validation.
"""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TCV1_PREFIX = b"DAGML-TCV1\0"
TCV1_INT_MIN = -(2**63)
TCV1_INT_MAX = 2**64 - 1


@dataclass(frozen=True)
class DagMLTrainingRequestSpec:
    """Inputs for a signed DAG-ML ``TrainingRequest`` shell."""

    request_id: str
    plan_id: str
    graph: Mapping[str, Any]
    campaign: Mapping[str, Any]
    controller_manifests: Sequence[Mapping[str, Any]]
    data_identities: Sequence[Mapping[str, Any]]
    selection_metric: str = "rmse"
    selection_objective: str = "minimize"
    selection_output_id: str = "output:prediction"
    output_requests: Sequence[Mapping[str, Any]] = field(default_factory=lambda: (_default_regression_output(),))
    seed: int = 12345
    refit: bool = True
    scheduler_workers: int = 1
    cpu_threads: int = 1
    cv_artifacts: str = "metadata_only"
    prediction_caches: str = "retain"
    fitted_artifacts: str = "allow_host_sidecar"
    parameter_patches: Sequence[Mapping[str, Any]] = ()
    patch_policies: Sequence[Mapping[str, Any]] = ()
    influence_requirements: Sequence[Mapping[str, Any]] = ()
    training_losses: Sequence[Mapping[str, Any]] = ()
    selection_required_metric_level: str | None = None
    selection_evaluation_scope: str | None = None


def assemble_training_request(spec: DagMLTrainingRequestSpec) -> dict[str, Any]:
    """Build and TCV1-sign a DAG-ML ``TrainingRequest`` dictionary."""

    request = {
        "schema_version": 1,
        "request_id": spec.request_id,
        "plan_id": spec.plan_id,
        "graph": dict(spec.graph),
        "campaign": dict(spec.campaign),
        "controller_manifests": _sorted_controller_manifests(spec.controller_manifests),
        "data_identities": [dict(identity) for identity in spec.data_identities],
        "parameter_patches": [dict(patch) for patch in spec.parameter_patches],
        "patch_policies": [dict(policy) for policy in spec.patch_policies],
        "influence_requirements": [dict(requirement) for requirement in spec.influence_requirements],
        "options": _training_options(spec),
        "request_fingerprint": "0" * 64,
    }
    if spec.training_losses:
        request["training_losses"] = _sorted_training_losses(spec.training_losses)
    request["request_fingerprint"] = tcv1_fingerprint_without(request, "request_fingerprint")
    return request


def training_data_identity_from_binding(
    binding: Mapping[str, Any],
    *,
    data_content_fingerprint: str,
    target_content_fingerprint: str,
) -> dict[str, Any]:
    """Build and sign one ``TrainingDataIdentity`` from a DAG-ML data binding."""

    identity = {
        "requirement_key": f"{binding['node_id']}.{binding['input_name']}",
        "schema_fingerprint": binding["schema_fingerprint"],
        "plan_fingerprint": binding["plan_fingerprint"],
        "relation_fingerprint": binding["relation_fingerprint"],
        "data_content_fingerprint": data_content_fingerprint,
        "target_content_fingerprint": target_content_fingerprint,
        "identity_fingerprint": "0" * 64,
    }
    identity["identity_fingerprint"] = tcv1_fingerprint_without(identity, "identity_fingerprint")
    return identity


def validate_training_request_with_dagml(request: Mapping[str, Any], dagml_module: Any) -> Any:
    """Return a validated ``dag_ml.TrainingRequest`` object."""

    return dagml_module.TrainingRequest(dict(request))


def tcv1_fingerprint_without(document: Mapping[str, Any], field: str) -> str:
    """TCV1 SHA-256 of an object while omitting its self-fingerprint field."""

    if field not in document:
        raise ValueError(f"missing self-fingerprint field {field!r}")
    return tcv1_sha256({key: value for key, value in document.items() if key != field})


def tcv1_sha256(value: Any) -> str:
    """Return lowercase SHA-256 for DAG-ML Typed Canonical Value v1."""

    return hashlib.sha256(TCV1_PREFIX + _tcv1_encode(value)).hexdigest()


def _training_options(spec: DagMLTrainingRequestSpec) -> dict[str, Any]:
    selection = {
        "id": f"selection:{spec.selection_metric}",
        "metric": {"name": spec.selection_metric, "objective": spec.selection_objective},
        "require_finite": True,
    }
    if spec.selection_required_metric_level is not None:
        selection["required_metric_level"] = spec.selection_required_metric_level
    if spec.selection_evaluation_scope is not None:
        selection["evaluation_scope"] = spec.selection_evaluation_scope
    return {
        "refit": spec.refit,
        "refit_strategy": "refit_one" if spec.refit else None,
        "seed": spec.seed,
        "selection": selection,
        "selection_output_id": spec.selection_output_id,
        "outputs": [dict(output) for output in spec.output_requests],
        "scheduler": {"kind": "sequential", "backend": None, "workers": spec.scheduler_workers},
        "resources": {"cpu_threads": spec.cpu_threads, "gpu_devices": []},
        "artifacts": {
            "cv_artifacts": spec.cv_artifacts,
            "prediction_caches": spec.prediction_caches,
            "fitted_artifacts": spec.fitted_artifacts,
        },
    }


def _default_regression_output() -> dict[str, Any]:
    return {
        "output_id": "output:prediction",
        "node_id": "model:base",
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "target_names": ["target"],
        "target_units": [None],
        "class_labels": [[]],
        "output_order": "target_order",
        "target_space": "raw",
    }


def _sorted_controller_manifests(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((_canonical_controller_manifest(manifest) for manifest in manifests), key=lambda manifest: manifest["controller_id"])


_PHASE_ORDER = {name: index for index, name in enumerate(("COMPILE", "PLAN", "FIT_CV", "SELECT", "REFIT", "PREDICT", "EXPLAIN"))}


def _sorted_training_losses(roles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    canonical_roles = [_canonical_training_loss_role(role) for role in roles]
    return sorted(canonical_roles, key=_training_loss_sort_key)


def _canonical_training_loss_role(role: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(role)
    phases = out.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, str | bytes | bytearray):
        raise TypeError("training loss role phases must be a sequence")
    out["phases"] = sorted(set(phases), key=_PHASE_ORDER.__getitem__)
    return out


def _training_loss_sort_key(role: Mapping[str, Any]) -> tuple[str, tuple[bool, str], tuple[int, ...]]:
    node_id = role.get("node_id")
    output_id = role.get("output_id")
    phases = role["phases"]
    if not isinstance(node_id, str):
        raise TypeError("training loss role node_id must be a string")
    if output_id is not None and not isinstance(output_id, str):
        raise TypeError("training loss role output_id must be a string or None")
    return (
        node_id,
        (output_id is not None, output_id or ""),
        tuple(_PHASE_ORDER[phase] for phase in phases),
    )

_CAPABILITY_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "deterministic",
            "thread_safe",
            "process_safe",
            "needs_python_gil",
            "emits_predictions",
            "consumes_oof_predictions",
            "emits_artifacts",
            "stateful",
            "emits_relation",
            "uses_core_rng",
            "shape_changing",
            "generates_data",
            "generates_model",
            "expands_variants",
            "aggregates_predictions",
            "supports_sample_weights",
            "supports_row_resampling",
            "supports_backend_loss_weights",
            "supports_missing_masks",
            "supports_configurable_loss",
            "supports_custom_loss",
            "supports_differentiable_loss",
            "uses_training_weights",
            "uses_early_stopping",
            "performs_internal_tuning",
            "trains_aggregation",
        )
    )
}


def _canonical_controller_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(manifest)
    if "supported_phases" in out:
        out["supported_phases"] = sorted(set(out["supported_phases"]), key=_PHASE_ORDER.__getitem__)
    if "capabilities" in out:
        out["capabilities"] = sorted(set(out["capabilities"]), key=_CAPABILITY_ORDER.__getitem__)
    if not out.get("operator_selectors"):
        out.pop("operator_selectors", None)
    return out


def _tcv1_encode(value: Any) -> bytes:
    _validate_strict_json(value)
    if value is None:
        return b"N"
    if value is False:
        return b"F"
    if value is True:
        return b"T"
    if isinstance(value, int) and not isinstance(value, bool):
        payload = str(value).encode("ascii")
        return b"I" + _u64(len(payload)) + payload
    if isinstance(value, float):
        return b"D" + struct.pack(">d", 0.0 if value == 0.0 else value)
    if isinstance(value, str):
        payload = unicodedata.normalize("NFC", value).encode("utf-8")
        return b"S" + _u64(len(payload)) + payload
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return b"A" + _u64(len(value)) + b"".join(_tcv1_encode(member) for member in value)
    if isinstance(value, Mapping):
        items = sorted(
            (
                unicodedata.normalize("NFC", str(key)).encode("utf-8"),
                unicodedata.normalize("NFC", str(key)),
                member,
            )
            for key, member in value.items()
        )
        return b"O" + _u64(len(items)) + b"".join(_tcv1_encode(key) + _tcv1_encode(member) for _encoded, key, member in items)
    raise TypeError(f"TCV1 does not support {type(value).__name__}")


def _validate_strict_json(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _strict_text(value)
        return
    if isinstance(value, int):
        if value < TCV1_INT_MIN or value > TCV1_INT_MAX:
            raise ValueError("integer is outside the TCV1 range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TCV1 forbids NaN and infinity")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for member in value:
            _validate_strict_json(member)
        return
    if isinstance(value, Mapping):
        encoded_keys: set[bytes] = set()
        for key, member in value.items():
            normalized, encoded = _strict_text(key)
            if encoded in encoded_keys:
                raise ValueError(f"object has NFC-colliding key {normalized!r}")
            encoded_keys.add(encoded)
            _validate_strict_json(member)
        return
    raise TypeError(f"TCV1 requires JSON-native values, got {type(value).__name__}")


def _strict_text(value: Any) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise TypeError("TCV1 object keys and strings must be text")
    value.encode("utf-8")
    normalized = unicodedata.normalize("NFC", value)
    return normalized, normalized.encode("utf-8")


def _u64(value: int) -> bytes:
    if value < 0 or value > 2**64 - 1:
        raise ValueError("TCV1 length exceeds u64")
    return struct.pack(">Q", value)


__all__ = [
    "DagMLTrainingRequestSpec",
    "assemble_training_request",
    "tcv1_fingerprint_without",
    "tcv1_sha256",
    "training_data_identity_from_binding",
    "validate_training_request_with_dagml",
]
