"""Filesystem store for internal conformal calibration results."""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conformal_contracts import CalibratedRunResult, ConformalCalibrationArtifact
from .training_contracts import tcv1_sha256

CONFORMAL_STORE_SCHEMA_ID = "nirs4all.dagml.conformal_store.v1"
CONFORMAL_STORE_VERSION = 1
ARTIFACT_FILENAME = "artifact.json"
RESULT_FILENAME = "calibrated_result.json"
MANIFEST_FILENAME = "manifest.json"
BUNDLE_ROOT = "conformal/"


@dataclass(frozen=True)
class ConformalStoreManifest:
    """Manifest for a filesystem conformal result store."""

    result_fingerprint: str
    artifact_fingerprint: str
    files: Mapping[str, str]
    schema_id: str = CONFORMAL_STORE_SCHEMA_ID
    version: int = CONFORMAL_STORE_VERSION

    def __post_init__(self) -> None:
        if self.schema_id != CONFORMAL_STORE_SCHEMA_ID:
            raise ValueError(f"unsupported conformal store schema_id: {self.schema_id}")
        if self.version != CONFORMAL_STORE_VERSION:
            raise ValueError(f"unsupported conformal store version: {self.version}")
        required_files = {"artifact", "result"}
        if set(self.files) != required_files:
            raise ValueError(f"conformal store files must contain exactly {sorted(required_files)}")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-like manifest form."""

        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "files": {key: self.files[key] for key in sorted(self.files)},
            "fingerprint": self.fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "schema_id": self.schema_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConformalStoreManifest:
        """Parse a conformal store manifest and verify its fingerprint."""

        if not isinstance(payload, Mapping):
            raise TypeError("ConformalStoreManifest payload must be a mapping")
        required = {"artifact_fingerprint", "files", "result_fingerprint", "schema_id", "version"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"ConformalStoreManifest payload is missing keys {missing}")
        if not isinstance(payload["files"], Mapping):
            raise ValueError("ConformalStoreManifest.files must be a mapping")
        manifest = cls(
            result_fingerprint=_non_empty_string(payload["result_fingerprint"], "result_fingerprint"),
            artifact_fingerprint=_non_empty_string(payload["artifact_fingerprint"], "artifact_fingerprint"),
            files={str(key): _non_empty_string(value, f"files.{key}") for key, value in payload["files"].items()},
            schema_id=str(payload["schema_id"]),
            version=int(payload["version"]),
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != manifest.fingerprint:
            raise ValueError("ConformalStoreManifest fingerprint mismatch")
        return manifest

    @classmethod
    def load_json(cls, path: str | Path) -> ConformalStoreManifest:
        """Load and verify a conformal store manifest."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def fingerprint(self) -> str:
        """TCV1 fingerprint of the manifest summary."""

        return tcv1_sha256(
            {
                "artifact_fingerprint": self.artifact_fingerprint,
                "files": {key: self.files[key] for key in sorted(self.files)},
                "result_fingerprint": self.result_fingerprint,
                "schema_id": self.schema_id,
                "version": self.version,
            }
        )


def save_conformal_result_store(
    result: CalibratedRunResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a calibrated result directory with manifest verification data."""

    target = Path(path)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"conformal store target exists and is not a directory: {target}")
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"conformal store target is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    artifact_path = target / ARTIFACT_FILENAME
    result_path = target / RESULT_FILENAME
    result.artifact.save_json(artifact_path)
    result.save_json(result_path)
    manifest = ConformalStoreManifest(
        result_fingerprint=result.fingerprint,
        artifact_fingerprint=result.artifact.fingerprint,
        files={"artifact": ARTIFACT_FILENAME, "result": RESULT_FILENAME},
    )
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_conformal_result_store(path: str | Path) -> CalibratedRunResult:
    """Load and verify a filesystem conformal result store."""

    target = Path(path)
    manifest = ConformalStoreManifest.load_json(target / MANIFEST_FILENAME)
    artifact_path = _resolve_store_file(target, manifest.files["artifact"])
    result_path = _resolve_store_file(target, manifest.files["result"])
    artifact = ConformalCalibrationArtifact.load_json(artifact_path)
    result = CalibratedRunResult.load_json(result_path)
    if artifact.fingerprint != manifest.artifact_fingerprint:
        raise ValueError("conformal store artifact fingerprint mismatch")
    if result.fingerprint != manifest.result_fingerprint:
        raise ValueError("conformal store result fingerprint mismatch")
    if result.artifact.fingerprint != artifact.fingerprint:
        raise ValueError("conformal store result artifact does not match stored artifact")
    return result


def export_conformal_result_bundle(
    result: CalibratedRunResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a calibrated result store as a zipped ``.n4a`` archive."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"conformal bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n4a-conformal-") as tmp:
        store_dir = Path(tmp) / "store"
        save_conformal_result_store(result, store_dir)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
                archive.write(store_dir / filename, arcname=f"{BUNDLE_ROOT}{filename}")
    return target


def attach_conformal_result_to_bundle(
    model_bundle_path: str | Path,
    result: CalibratedRunResult,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a model ``.n4a`` bundle and attach a verified conformal sidecar.

    The source bundle is left unchanged unless the caller explicitly passes the
    same path as ``output_path``; in-place attachment is rejected to avoid
    corrupting ZIP archives during copy. Existing conformal sidecars are refused
    unless ``overwrite=True``.
    """

    source = Path(model_bundle_path)
    if not source.is_file():
        raise FileNotFoundError(f"model bundle not found: {source}")
    if not zipfile.is_zipfile(source):
        raise ValueError(f"model bundle is not a ZIP .n4a archive: {source}")

    target = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}.calibrated{source.suffix}")
    if source.resolve() == target.resolve():
        raise ValueError("in-place conformal attachment is not supported; choose a different output_path")
    if target.exists() and not overwrite:
        raise FileExistsError(f"calibrated bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="n4a-conformal-attach-") as tmp:
        store_dir = Path(tmp) / "store"
        save_conformal_result_store(result, store_dir)
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            existing_conformal = [name for name in src.namelist() if name.startswith(BUNDLE_ROOT)]
            if existing_conformal and not overwrite:
                raise ValueError("model bundle already contains a conformal sidecar; pass overwrite=True to replace it")
            for item in src.infolist():
                if item.filename.startswith(BUNDLE_ROOT):
                    continue
                dst.writestr(item, src.read(item.filename))
            for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
                dst.write(store_dir / filename, arcname=f"{BUNDLE_ROOT}{filename}")
    return target


def load_conformal_result_bundle(path: str | Path) -> CalibratedRunResult:
    """Load and verify a zipped conformal result ``.n4a`` archive."""

    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="n4a-conformal-load-") as tmp:
        extract_dir = Path(tmp) / "store"
        extract_dir.mkdir()
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            members = set(names)
            expected = _expected_conformal_members()
            if members != expected:
                raise ValueError(f"conformal bundle members must be exactly {sorted(expected)}")
            _reject_duplicate_conformal_members(names, expected)
            for member in sorted(expected):
                target = extract_dir / Path(member).name
                target.write_bytes(archive.read(member))
        return load_conformal_result_store(extract_dir)


def load_conformal_result_archive(path: str | Path) -> CalibratedRunResult:
    """Load a conformal sidecar from a conformal-only or full model ``.n4a`` archive."""

    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="n4a-conformal-load-") as tmp:
        extract_dir = Path(tmp) / "store"
        extract_dir.mkdir()
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            members = set(names)
            expected = _expected_conformal_members()
            missing = sorted(expected - members)
            if missing:
                raise ValueError(f"archive does not contain a complete conformal sidecar; missing {missing}")
            _reject_unexpected_conformal_members(names, expected)
            _reject_duplicate_conformal_members(names, expected)
            for member in sorted(expected):
                target = extract_dir / Path(member).name
                target.write_bytes(archive.read(member))
        return load_conformal_result_store(extract_dir)


def _expected_conformal_members() -> set[str]:
    return {f"{BUNDLE_ROOT}{filename}" for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME)}


def _reject_duplicate_conformal_members(names: list[str], expected: set[str]) -> None:
    duplicates = sorted(member for member in expected if names.count(member) > 1)
    if duplicates:
        raise ValueError(f"conformal sidecar contains duplicate members: {duplicates}")


def _reject_unexpected_conformal_members(names: list[str], expected: set[str]) -> None:
    unexpected = sorted(name for name in names if name.startswith(BUNDLE_ROOT) and name not in expected)
    if unexpected:
        raise ValueError(f"archive contains unexpected conformal sidecar members: {unexpected}")


def _resolve_store_file(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"conformal store file escapes store root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"conformal store file not found: {path}")
    return path


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


__all__ = [
    "ARTIFACT_FILENAME",
    "BUNDLE_ROOT",
    "CONFORMAL_STORE_SCHEMA_ID",
    "CONFORMAL_STORE_VERSION",
    "MANIFEST_FILENAME",
    "RESULT_FILENAME",
    "ConformalStoreManifest",
    "attach_conformal_result_to_bundle",
    "export_conformal_result_bundle",
    "load_conformal_result_archive",
    "load_conformal_result_bundle",
    "load_conformal_result_store",
    "save_conformal_result_store",
]
