"""Fail-closed public replay adapter for Core Archive V2/V3.

This module only performs bounded routing and host-input adaptation.  Core
owns archive validation, DAG-ML owns replay contracts and scheduling, and the
Methods runtime owns numerical execution.  No legacy runner or Python model
callback is part of this path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from .identity import validate_data_id

_MAX_MANIFEST_BYTES = 1_048_576
_FINGERPRINT_PREFIX = b"n4a-matrix-f64-le.v1\0"
_CORE_PROFILES = {
    2: "nirs4all.archive_workspace.v2",
    3: "nirs4all.archive_workspace.v3",
}


class CoreArchiveReplayError(RuntimeError):
    """A recognized Core archive cannot be replayed safely."""


def detect_core_archive_version(path: str | Path) -> int | None:
    """Return 2/3 only for an exact Core archive routing manifest.

    The manifest read is a bounded routing hint, not archive validation.  A
    recognized archive is subsequently reopened and fully validated by Core.
    Legacy/non-Core bundles return ``None`` and keep their historical path.
    A manifest that claims the Core family but is inconsistent fails closed.
    """

    candidate = Path(path)
    if not candidate.is_file() or candidate.suffix.lower() != ".n4a":
        return None
    try:
        with zipfile.ZipFile(candidate) as archive:
            try:
                info = archive.getinfo("manifest.json")
            except KeyError:
                return None
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise CoreArchiveReplayError("Core archive manifest exceeds the routing budget")
            with archive.open(info) as manifest_file:
                payload = manifest_file.read(_MAX_MANIFEST_BYTES + 1)
            if len(payload) > _MAX_MANIFEST_BYTES:
                raise CoreArchiveReplayError("Core archive manifest exceeds the routing budget")
    except CoreArchiveReplayError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    profile = manifest.get("profile")
    writer = manifest.get("writer")
    owner = writer.get("product_aggregate_owner") if isinstance(writer, dict) else None
    claims_core = owner == "nirs4all-core" or profile in _CORE_PROFILES.values()
    if not claims_core:
        return None

    version = manifest.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _CORE_PROFILES
        or manifest.get("persistence_kind") != "n4a_archive"
        or profile != _CORE_PROFILES[version]
        or owner != "nirs4all-core"
    ):
        raise CoreArchiveReplayError(
            "archive claims the nirs4all-core family but has an incompatible routing manifest"
        )
    return version


def predict_core_methods_archive_v2(
    archive_path: str | Path,
    data: Any,
    *,
    methods_library_path: str | Path | None,
    outcome_id: str = "outcome:nirs4all.core_archive_predict",
    run_id: str = "run:nirs4all.core_archive_predict",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay one exact, explicitly identified raw cohort through Core V2."""

    X, sample_ids, groups, metadata_rows = _normalize_dataset(data)
    core = _load_core_bridge()
    read_package = getattr(core, "read_portable_predictor_package_v2", None)
    replay = getattr(core, "replay_methods_archive_v2", None)
    if not callable(read_package) or not callable(replay):
        raise CoreArchiveReplayError(
            "installed nirs4all-core is too old for callback-free Archive V2 Methods replay"
        )
    try:
        package_bytes = read_package(str(Path(archive_path)))
    except Exception as error:
        raise CoreArchiveReplayError("nirs4all-core refused Archive V2 validation") from error
    package = _decode_package(package_bytes)
    request, envelopes, methods_inputs = _build_replay_contracts(
        package,
        X,
        sample_ids=sample_ids,
        groups=groups,
        metadata_rows=metadata_rows,
    )
    library_path = _resolve_methods_library_path(methods_library_path)
    try:
        outcome = replay(
            str(Path(archive_path)),
            request,
            envelopes,
            methods_inputs,
            methods_library_path=library_path,
            outcome_id=outcome_id,
            run_id=run_id,
            warnings=(),
            diagnostics={"caller": "nirs4all.predict", "archive_schema_version": 2},
        )
    except Exception as error:
        raise CoreArchiveReplayError("Core/DAG-ML/Methods Archive V2 replay failed") from error
    values = _decode_prediction(outcome, sample_ids)
    return values, {
        "engine": "core-native",
        "archive_path": str(Path(archive_path)),
        "archive_schema_version": 2,
        "sample_ids": list(sample_ids),
        "outcome_id": outcome_id,
        "run_id": run_id,
    }


def _load_core_bridge() -> Any:
    try:
        return importlib.import_module("nirs4all_core")
    except ImportError as error:
        raise CoreArchiveReplayError(
            "Core Archive V2 replay requires a matching nirs4all-core native wheel"
        ) from error


def _decode_package(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise CoreArchiveReplayError("Core Archive V2 reader returned non-byte package data")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoreArchiveReplayError("Core Archive V2 package is not UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise CoreArchiveReplayError("Core Archive V2 does not contain a Package V2 object")
    return document


def _normalize_dataset(
    data: Any,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str | None, ...], tuple[dict[str, Any], ...]]:
    if not isinstance(data, Mapping) or "X" not in data or "sample_ids" not in data:
        raise TypeError(
            "Core Archive V2 predict requires data={'X': matrix, 'sample_ids': explicit_ids}"
        )
    X = np.ascontiguousarray(np.asarray(data["X"], dtype=np.dtype("<f8")))
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Core Archive V2 predict requires a non-empty rank-2 X matrix")
    if not np.isfinite(X).all():
        raise ValueError("Core Archive V2 predict X must contain only finite values")
    raw_ids = data["sample_ids"]
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise TypeError("Core Archive V2 predict sample_ids must be a sequence")
    if len(raw_ids) != X.shape[0]:
        raise ValueError("Core Archive V2 predict sample_ids must align with X rows")
    sample_ids = tuple(validate_data_id(str(value)) for value in raw_ids)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Core Archive V2 predict sample_ids must be unique")
    groups = _normalize_groups(data.get("groups"), X.shape[0])
    metadata_rows = _normalize_metadata(data.get("metadata"), X.shape[0])
    return X, sample_ids, groups, metadata_rows


def _normalize_groups(value: Any, rows: int) -> tuple[str | None, ...]:
    if value is None:
        return (None,) * rows
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError("Core Archive V2 predict groups must align with X rows")
    groups = tuple(None if item is None else str(item) for item in value)
    if any(item == "" for item in groups):
        raise ValueError("Core Archive V2 predict group ids must be non-empty")
    return groups


def _normalize_metadata(value: Any, rows: int) -> tuple[dict[str, Any], ...]:
    if value is None:
        return tuple({} for _ in range(rows))
    if isinstance(value, Mapping):
        normalized: list[dict[str, Any]] = [{} for _ in range(rows)]
        for key, column in value.items():
            if not isinstance(column, Sequence) or isinstance(column, (str, bytes)) or len(column) != rows:
                raise ValueError(f"Core Archive V2 metadata column {key!r} must align with X rows")
            for index, item in enumerate(column):
                normalized[index][str(key)] = _json_scalar(item)
        return tuple(normalized)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError("Core Archive V2 metadata rows must align with X rows")
    normalized = []
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("Core Archive V2 metadata rows must be mappings")
        normalized.append({str(key): _json_scalar(item) for key, item in row.items()})
    return tuple(normalized)


def _json_scalar(value: Any) -> str | bool | int | float | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("Core Archive V2 metadata values must be finite JSON scalars")


def _build_replay_contracts(
    package: Mapping[str, Any],
    X: np.ndarray,
    *,
    sample_ids: tuple[str, ...],
    groups: tuple[str | None, ...],
    metadata_rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _object(package, "execution_bundle")
    requirements = _requirements(bundle)
    binding = _single_binding(package)
    relations = {
        "records": [
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
            for sample_id, group, metadata in zip(sample_ids, groups, metadata_rows, strict=True)
        ]
    }
    dag_ml = _load_dag_ml()
    fingerprint_fn = getattr(dag_ml, "sample_relation_set_fingerprint_json", None)
    signer = getattr(dag_ml, "sign_training_replay_request", None)
    if not callable(fingerprint_fn) or not callable(signer):
        raise CoreArchiveReplayError(
            "installed dag-ml is too old to fingerprint and sign Archive V2 replay contracts"
        )
    relation_fingerprint = fingerprint_fn(_canonical_json(relations))
    if not isinstance(relation_fingerprint, str) or len(relation_fingerprint) != 64:
        raise CoreArchiveReplayError("DAG-ML returned an invalid relation fingerprint")
    data_fingerprint = _feature_fingerprint(X)
    envelopes = {
        key: {
            "schema_version": 1,
            "schema_fingerprint": requirement["schema_fingerprint"],
            "plan_fingerprint": requirement["plan_fingerprint"],
            "relation_fingerprint": relation_fingerprint,
            "data_content_fingerprint": data_fingerprint,
            "target_content_fingerprint": None,
            "coordinator_relations": relations,
        }
        for key, requirement in requirements.items()
    }
    outcome = _object(package, "training_outcome")
    source_fingerprint = outcome.get("outcome_fingerprint")
    if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
        raise CoreArchiveReplayError("Package V2 training outcome lacks its fingerprint")
    request = signer(
        {
            "schema_version": 1,
            "request_id": "replay:nirs4all.core_archive_predict",
            "source_outcome_fingerprint": source_fingerprint,
            "phase": "PREDICT",
            "data_envelope_keys": sorted(envelopes),
            "output_binding_ids": [binding["binding_id"]],
            "request_fingerprint": "0" * 64,
        }
    )
    if hasattr(request, "to_dict") and callable(request.to_dict):
        request = request.to_dict()
    if not isinstance(request, dict):
        raise CoreArchiveReplayError("DAG-ML replay request signer returned a non-object")
    inputs = {
        key: {
            "sample_ids": list(sample_ids),
            "x": X.tolist(),
            "target_names": list(binding["target_names"]),
        }
        for key in requirements
    }
    return request, envelopes, inputs


def _load_dag_ml() -> Any:
    try:
        return importlib.import_module("dag_ml")
    except ImportError as error:
        raise CoreArchiveReplayError(
            "Core Archive V2 replay requires a matching DAG-ML Python facade"
        ) from error


def _requirements(bundle: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw = bundle.get("data_requirements")
    if not isinstance(raw, list) or not raw:
        raise CoreArchiveReplayError("Package V2 has no data requirements")
    requirements: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise CoreArchiveReplayError("Package V2 data requirement is not an object")
        values = tuple(item.get(name) for name in ("node_id", "input_name", "schema_fingerprint", "plan_fingerprint"))
        if not all(isinstance(value, str) and value for value in values):
            raise CoreArchiveReplayError("Package V2 data requirement lacks stable fingerprints")
        node_id, input_name, schema, plan = cast(tuple[str, str, str, str], values)
        key = f"{node_id}.{input_name}"
        if key in requirements:
            raise CoreArchiveReplayError("Package V2 repeats a data requirement key")
        requirements[key] = {"schema_fingerprint": schema, "plan_fingerprint": plan}
    return requirements


def _single_binding(package: Mapping[str, Any]) -> dict[str, Any]:
    raw = package.get("output_bindings")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise CoreArchiveReplayError("Package V2 replay requires exactly one output binding")
    binding = raw[0]
    if not isinstance(binding.get("binding_id"), str) or not binding["binding_id"]:
        raise CoreArchiveReplayError("Package V2 output binding lacks binding_id")
    targets = binding.get("target_names")
    if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item for item in targets):
        raise CoreArchiveReplayError("Package V2 output binding lacks target_names")
    return binding


def _object(container: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = container.get(name)
    if not isinstance(value, dict):
        raise CoreArchiveReplayError(f"Package V2 lacks object `{name}`")
    return value


def _feature_fingerprint(X: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_PREFIX)
    hasher.update(struct.pack("<QQ", *X.shape))
    hasher.update(X.tobytes(order="C"))
    return hasher.hexdigest()


def _resolve_methods_library_path(path: str | Path | None) -> str:
    if path is not None:
        return str(Path(path))
    try:
        n4m = importlib.import_module("n4m")
    except ImportError as error:
        raise CoreArchiveReplayError(
            "Core Archive V2 replay requires nirs4all-methods or an explicit methods_library_path"
        ) from error
    resolver = getattr(n4m, "library_path", None)
    if not callable(resolver):
        raise CoreArchiveReplayError("installed nirs4all-methods does not expose library_path()")
    value = resolver()
    if not isinstance(value, (str, Path)):
        raise CoreArchiveReplayError("nirs4all-methods returned an invalid library path")
    return str(value)


def _decode_prediction(outcome: Any, sample_ids: tuple[str, ...]) -> np.ndarray:
    if not isinstance(outcome, dict):
        raise CoreArchiveReplayError("Core Archive V2 replay returned a non-object outcome")
    outputs = outcome.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise CoreArchiveReplayError("Core Archive V2 replay requires exactly one output")
    blocks = outputs[0].get("predictions")
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], dict):
        raise CoreArchiveReplayError("Core Archive V2 replay requires exactly one prediction block")
    block = blocks[0]
    if block.get("sample_ids") != list(sample_ids):
        raise CoreArchiveReplayError("Core Archive V2 prediction identities do not match the request")
    values = np.asarray(block.get("values"), dtype=float)
    if values.ndim != 2 or values.shape[0] != len(sample_ids) or not np.isfinite(values).all():
        raise CoreArchiveReplayError("Core Archive V2 predictions are not a finite aligned matrix")
    return cast(np.ndarray, values)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "CoreArchiveReplayError",
    "detect_core_archive_version",
    "predict_core_methods_archive_v2",
]
