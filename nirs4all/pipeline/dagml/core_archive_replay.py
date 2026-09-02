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
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .identity import validate_data_id

_MAX_MANIFEST_BYTES = 1_048_576
_MAX_PACKAGE_BYTES = 8_388_608
_MAX_METHODS_LIBRARY_BYTES = 67_108_864
_CORE_PROFILES = {
    2: "nirs4all.archive_workspace.v2",
    3: "nirs4all.archive_workspace.v3",
}


class CoreArchiveReplayError(RuntimeError):
    """A recognized Core archive cannot be replayed safely."""


class CoreArchiveDependencyError(CoreArchiveReplayError):
    """A required native Archive V2 component is absent or incompatible."""

    def __init__(self, dependency: str, message: str, *, mitigation: str) -> None:
        self.dependency = dependency
        self.mitigation = mitigation
        super().__init__(message)


@dataclass(frozen=True)
class CoreArchiveValidation:
    """Core-derived, byte-bound evidence cached by one strict consumer."""

    core: Any
    package: dict[str, Any]
    predictors: tuple[dict[str, Any], ...]
    methods_library_path: str
    methods_library_sha256: str


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
    validated_archive: tuple[Path, CoreArchiveValidation] | None = None,
    expected_archive_fingerprint: str | None = None,
    outcome_id: str = "outcome:nirs4all.core_archive_predict",
    run_id: str = "run:nirs4all.core_archive_predict",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay one exact, explicitly identified raw cohort through Core V2.

    ``validated_archive`` is the immutable package-contract cache owned by a
    loaded :class:`Session`. Core still reopens and validates the archive for
    every replay, and the Methods runtime keeps all native handles scoped to
    that call. When ``expected_archive_fingerprint`` is supplied, the adapter
    performs one bounded-memory full-file hash before reading ``data``; this
    binds the single Session cache entry to its source and is not a general
    replay-result cache.
    """

    candidate = Path(archive_path)
    if (
        expected_archive_fingerprint is not None
        and _archive_fingerprint(candidate) != expected_archive_fingerprint
    ):
        raise CoreArchiveReplayError(
            "Core Archive V2 changed after Session validation; load a new session"
        )
    if validated_archive is None:
        validation = validate_core_methods_archive_v2(
            candidate,
            methods_library_path=methods_library_path,
        )
    else:
        validated_path, validation = validated_archive
        if candidate != validated_path:
            raise CoreArchiveReplayError(
                "Core Archive V2 validation cache does not match the replay path"
            )
        if methods_library_path is not None:
            requested_path, requested_sha256 = _resolve_methods_library_identity(
                methods_library_path
            )
            if (
                requested_path != validation.methods_library_path
                or requested_sha256 != validation.methods_library_sha256
            ):
                raise CoreArchiveReplayError(
                    "Core Archive V2 Session is bound to a different libn4m identity"
                )

    X, sample_ids = _normalize_dataset(data)
    target_names = tuple(_single_binding(validation.package)["target_names"])
    predict = getattr(validation.core, "predict_methods_archive_v2_matrix", None)
    if not callable(predict):
        raise CoreArchiveDependencyError(
            "nirs4all-core",
            "installed nirs4all-core is too old for closed Archive V2 matrix prediction",
            mitigation="install the nirs4all-core version pinned by the release lock",
        )
    try:
        outcome = predict(
            str(candidate),
            list(sample_ids),
            X.tolist(),
            list(target_names),
            methods_library_path=validation.methods_library_path,
            methods_library_sha256=validation.methods_library_sha256,
            request_id="replay:nirs4all.core_archive_predict",
            outcome_id=outcome_id,
            run_id=run_id,
            warnings=(),
            diagnostics={"caller": "nirs4all.predict", "archive_schema_version": 2},
        )
    except Exception as error:
        raise CoreArchiveReplayError("Core/DAG-ML/Methods Archive V2 replay failed") from error
    values = _decode_prediction(outcome, sample_ids, target_names=target_names)
    return values, {
        "engine": "core-native",
        "archive_path": str(candidate),
        "archive_schema_version": 2,
        "sample_ids": list(sample_ids),
        "target_names": list(target_names),
        "native_predictor_descriptors": [
            dict(descriptor) for descriptor in validation.predictors
        ],
        "outcome_id": outcome_id,
        "run_id": run_id,
    }


def validate_core_methods_archive_v2(
    archive_path: str | Path,
    *,
    methods_library_path: str | Path | None = None,
) -> CoreArchiveValidation:
    """Validate one V2 archive and derive its predictors from native bytes."""

    core = _load_core_bridge()
    read_package = getattr(core, "read_portable_predictor_package_v2", None)
    inspect_predictors = getattr(core, "inspect_methods_archive_v2_predictors", None)
    predict = getattr(core, "predict_methods_archive_v2_matrix", None)
    if (
        not callable(read_package)
        or not callable(inspect_predictors)
        or not callable(predict)
    ):
        raise CoreArchiveDependencyError(
            "nirs4all-core",
            "installed nirs4all-core is too old for native predictor inspection and closed Archive V2 prediction",
            mitigation="install the nirs4all-core version pinned by the release lock",
        )
    library_path, library_sha256 = _resolve_methods_library_identity(methods_library_path)
    try:
        package_bytes = read_package(str(Path(archive_path)))
    except Exception as error:
        raise CoreArchiveReplayError("nirs4all-core refused Archive V2 validation") from error
    package = _decode_package(package_bytes)
    try:
        raw_predictors = inspect_predictors(
            str(Path(archive_path)),
            methods_library_path=library_path,
            methods_library_sha256=library_sha256,
        )
    except Exception as error:
        raise CoreArchiveReplayError(
            "Core/DAG-ML/Methods refused Archive V2 predictor inspection"
        ) from error
    predictors = _validate_native_predictor_evidence(package, raw_predictors)
    return CoreArchiveValidation(
        core=core,
        package=package,
        predictors=predictors,
        methods_library_path=library_path,
        methods_library_sha256=library_sha256,
    )


def _load_core_bridge() -> Any:
    try:
        return importlib.import_module("nirs4all_core")
    except ImportError as error:
        raise CoreArchiveDependencyError(
            "nirs4all-core",
            "Core Archive V2 replay requires a matching nirs4all-core native wheel",
            mitigation="install the nirs4all-core native wheel pinned by the release lock",
        ) from error


def _decode_package(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise CoreArchiveReplayError("Core Archive V2 reader returned non-byte package data")
    if len(payload) > _MAX_PACKAGE_BYTES:
        raise CoreArchiveReplayError(
            "Core Archive V2 Package V2 exceeds the 8 MiB Session cache budget"
        )
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoreArchiveReplayError("Core Archive V2 package is not UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise CoreArchiveReplayError("Core Archive V2 does not contain a Package V2 object")
    return document


def _archive_fingerprint(path: Path) -> str:
    """Hash an archive in bounded-memory chunks for Session cache binding."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1_048_576), b""):
                digest.update(chunk)
    except OSError as error:
        raise CoreArchiveReplayError(
            "Core Archive V2 source cannot be fingerprinted"
        ) from error
    return digest.hexdigest()


def _normalize_dataset(
    data: Any,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not isinstance(data, Mapping) or "X" not in data or "sample_ids" not in data:
        raise TypeError(
            "Core Archive V2 predict requires data={'X': matrix, 'sample_ids': explicit_ids}"
        )
    unsupported = sorted(
        key for key in ("groups", "metadata") if data.get(key) is not None
    )
    if unsupported:
        raise ValueError(
            "Core Archive V2 closed matrix prediction does not accept cohort fields: "
            f"{unsupported}"
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
    return X, sample_ids


def _validate_native_predictor_evidence(
    package: Mapping[str, Any],
    raw_predictors: Any,
) -> tuple[dict[str, Any], ...]:
    """Match Core-derived descriptors to the package and admit V1 topology."""

    if not isinstance(raw_predictors, list) or not raw_predictors or not all(
        isinstance(descriptor, dict) for descriptor in raw_predictors
    ):
        raise CoreArchiveReplayError(
            "Core Archive V2 inspection returned an invalid predictor descriptor list"
        )
    predictors = tuple(dict(descriptor) for descriptor in raw_predictors)
    bundle = _object(package, "execution_bundle")
    records = bundle.get("refit_artifacts")
    if not isinstance(records, list):
        raise CoreArchiveReplayError("Package V2 lacks refit_artifacts")
    native_records: list[dict[str, Any]] = []
    embedded: list[dict[str, Any] | None] = []
    for record in records:
        if not isinstance(record, dict):
            raise CoreArchiveReplayError("Package V2 refit artifact is not an object")
        artifact = record.get("artifact")
        if not isinstance(artifact, dict):
            raise CoreArchiveReplayError("Package V2 refit artifact lacks artifact metadata")
        if artifact.get("kind") != "n4m_model":
            continue
        native_records.append(record)
        descriptor = artifact.get("native_predictor_descriptor")
        if descriptor is not None and not isinstance(descriptor, dict):
            raise CoreArchiveReplayError(
                "Package V2 native predictor descriptor is not an object"
            )
        embedded.append(descriptor)

    if len(native_records) != len(predictors):
        raise CoreArchiveReplayError(
            "Core-derived predictors do not exactly cover Package V2 refit artifacts"
        )
    present = [descriptor is not None for descriptor in embedded]
    if any(present) and not all(present):
        raise CoreArchiveReplayError(
            "Package V2 mixes present and historical absent native predictor descriptors"
        )
    if all(present):
        expected = Counter(_canonical_json(descriptor) for descriptor in embedded)
        observed = Counter(_canonical_json(descriptor) for descriptor in predictors)
        if expected != observed:
            raise CoreArchiveReplayError(
                "Package V2 native predictor descriptors do not match Core-derived bytes"
            )

    binding = _single_binding(package)
    final_records = [record for record in native_records if record.get("node_id") == binding.get("node_id")]
    if len(final_records) != 1:
        raise CoreArchiveReplayError(
            "Package V2 output binding must select exactly one native predictor"
        )
    final_record = final_records[0]
    final_index = native_records.index(final_record)
    final_descriptor = predictors[final_index]
    final_owner = final_descriptor.get("owner_controller")
    target_names = binding["target_names"]
    dimensions = final_descriptor.get("dimensions")
    if not isinstance(dimensions, dict) or dimensions.get("n_targets") != len(target_names):
        raise CoreArchiveReplayError(
            "Package V2 final native predictor dimensions do not match output targets"
        )

    owners = [descriptor.get("owner_controller") for descriptor in predictors]
    prediction_inputs = final_record.get("prediction_requirement_keys", [])
    data_inputs = final_record.get("data_requirement_keys", [])
    if not isinstance(prediction_inputs, list) or not isinstance(data_inputs, list):
        raise CoreArchiveReplayError(
            "Package V2 final native predictor requirements are invalid"
        )
    if final_owner == "controller:methods.pls":
        if len(predictors) != 1 or prediction_inputs:
            raise CoreArchiveReplayError(
                "strict Archive V2 prediction accepts only one raw Methods PLS predictor"
            )
    elif final_owner == "controller:methods.ridge":
        if (
            len(predictors) < 2
            or final_index != len(predictors) - 1
            or any(owner != "controller:methods.pls" for owner in owners[:-1])
            or not prediction_inputs
            or data_inputs
        ):
            raise CoreArchiveReplayError(
                "strict Archive V2 prediction accepts Ridge only as the final stacking predictor"
            )
    else:
        raise CoreArchiveReplayError(
            "strict Archive V2 prediction supports raw PLS or final stacking Ridge only"
        )
    return predictors


def _single_binding(package: Mapping[str, Any]) -> dict[str, Any]:
    raw = package.get("output_bindings")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise CoreArchiveReplayError("Package V2 replay requires exactly one output binding")
    binding = raw[0]
    if not isinstance(binding.get("binding_id"), str) or not binding["binding_id"]:
        raise CoreArchiveReplayError("Package V2 output binding lacks binding_id")
    targets = binding.get("target_names")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) and item.strip() for item in targets)
        or len(set(targets)) != len(targets)
    ):
        raise CoreArchiveReplayError("Package V2 output binding lacks target_names")
    return binding


def _object(container: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = container.get(name)
    if not isinstance(value, dict):
        raise CoreArchiveReplayError(f"Package V2 lacks object `{name}`")
    return value


def _resolve_methods_library_path(path: str | Path | None) -> str:
    if path is not None:
        return str(Path(path))
    try:
        n4m = importlib.import_module("n4m")
    except ImportError as error:
        raise CoreArchiveDependencyError(
            "nirs4all-methods",
            "Core Archive V2 replay requires nirs4all-methods or an explicit methods_library_path",
            mitigation="install the nirs4all-methods wheel pinned by the release lock or pass methods_library_path",
        ) from error
    resolver = getattr(n4m, "library_path", None)
    if not callable(resolver):
        raise CoreArchiveDependencyError(
            "nirs4all-methods",
            "installed nirs4all-methods does not expose library_path()",
            mitigation="install the nirs4all-methods wheel pinned by the release lock",
        )
    value = resolver()
    if not isinstance(value, (str, Path)):
        raise CoreArchiveDependencyError(
            "nirs4all-methods",
            "nirs4all-methods returned an invalid library path",
            mitigation="install the nirs4all-methods wheel pinned by the release lock",
        )
    return str(value)


def _resolve_methods_library_identity(
    path: str | Path | None,
) -> tuple[str, str]:
    """Resolve and hash the exact libn4m bytes attested again by Core."""

    candidate = Path(_resolve_methods_library_path(path))
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > _MAX_METHODS_LIBRARY_BYTES:
            raise OSError("libn4m is not a bounded regular file")
        digest = hashlib.sha256()
        with resolved.open("rb") as library:
            for chunk in iter(lambda: library.read(1_048_576), b""):
                digest.update(chunk)
    except OSError as error:
        raise CoreArchiveDependencyError(
            "nirs4all-methods",
            "Core Archive V2 replay cannot attest the selected libn4m file",
            mitigation="install the nirs4all-methods wheel pinned by the release lock or pass its canonical library path",
        ) from error
    return str(resolved), digest.hexdigest()


def _decode_prediction(
    outcome: Any,
    sample_ids: tuple[str, ...],
    *,
    target_names: tuple[str, ...],
) -> np.ndarray:
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
    if (
        values.ndim != 2
        or values.shape != (len(sample_ids), len(target_names))
        or not np.isfinite(values).all()
    ):
        raise CoreArchiveReplayError("Core Archive V2 predictions are not a finite aligned matrix")
    return cast(np.ndarray, values)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "CoreArchiveDependencyError",
    "CoreArchiveReplayError",
    "CoreArchiveValidation",
    "detect_core_archive_version",
    "predict_core_methods_archive_v2",
    "validate_core_methods_archive_v2",
]
