"""Filesystem store for internal conformal calibration results."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conformal_contracts import CalibratedRunResult, ConformalCalibrationArtifact
from .host_container import JSON_CONTENT_TYPE, HostContainer, HostMember, N4aSource, open_n4a_source, update_host_container, write_host_container
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


def _conformal_members(store_dir: Path) -> list[HostMember]:
    """The three conformal sidecar files in declared order, streamed from ``store_dir``."""
    return [HostMember(f"{BUNDLE_ROOT}{filename}", store_dir / filename, "result", JSON_CONTENT_TYPE) for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME)]


@contextmanager
def _host_container(path: Path, source: N4aSource | None) -> Iterator[HostContainer]:
    """The host container of ``path``: a borrowed ``source`` is never reopened or closed."""
    if source is None:
        if not path.is_file():
            raise FileNotFoundError(f"conformal archive not found: {path}")
        with open_n4a_source(path) as owned:
            yield owned.require_host()
        return
    if source.path.resolve() != path.resolve():
        raise ValueError(f"the supplied source is {source.path}, not {path}")
    yield source.require_host()


def export_conformal_result_bundle(
    result: CalibratedRunResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a calibrated result store as a conformal-only host ``.n4a`` container."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"conformal bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n4a-conformal-") as tmp:
        store_dir = Path(tmp) / "store"
        save_conformal_result_store(result, store_dir)
        write_host_container(target, _conformal_members(store_dir), entrypoint=False)
    return target


def attach_conformal_result_to_bundle(
    model_bundle_path: str | Path,
    result: CalibratedRunResult,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    source: N4aSource | None = None,
) -> Path:
    """Copy a host model ``.n4a`` container and attach a verified conformal sidecar.

    The source container is never modified; retained members are copied and verified by
    Core. In-place attachment is rejected. Existing conformal sidecars are refused unless
    ``overwrite=True``. ``source`` is a borrowed, already opened source of
    ``model_bundle_path``: it is neither reopened nor closed here.
    """

    model = Path(model_bundle_path)
    if not model.is_file():
        raise FileNotFoundError(f"model bundle not found: {model}")

    target = Path(output_path) if output_path is not None else model.with_name(f"{model.stem}.calibrated{model.suffix}")
    if model.resolve() == target.resolve():
        raise ValueError("in-place conformal attachment is not supported; choose a different output_path")
    if target.exists() and not overwrite:
        raise FileExistsError(f"calibrated bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="n4a-conformal-attach-") as tmp, _host_container(model, source) as base:
        if not overwrite and any(name.startswith(BUNDLE_ROOT) for name in base.namelist()):
            raise ValueError("model bundle already contains a conformal sidecar; pass overwrite=True to replace it")
        store_dir = Path(tmp) / "store"
        save_conformal_result_store(result, store_dir)
        update_host_container(base, target, add=_conformal_members(store_dir), remove_prefix=BUNDLE_ROOT)
    return target


def load_conformal_result_bundle(path: str | Path, *, source: N4aSource | None = None) -> CalibratedRunResult:
    """Load and verify a conformal-only host ``.n4a`` container."""

    expected = _expected_conformal_members()
    with _host_container(Path(path), source) as container:
        if set(container.namelist()) != expected:
            raise ValueError(f"conformal bundle members must be exactly {sorted(expected)}")
        return _load_conformal_members(container, expected)


def load_conformal_result_archive(path: str | Path, *, source: N4aSource | None = None) -> CalibratedRunResult:
    """Load a conformal sidecar from a conformal-only or full model host ``.n4a`` container."""

    expected = _expected_conformal_members()
    with _host_container(Path(path), source) as container:
        names = container.namelist()
        missing = sorted(expected - set(names))
        if missing:
            raise ValueError(f"archive does not contain a complete conformal sidecar; missing {missing}")
        _reject_unexpected_conformal_members(names, expected)
        return _load_conformal_members(container, expected)


def _load_conformal_members(container: HostContainer, expected: set[str]) -> CalibratedRunResult:
    """Extract the verified sidecar files privately and load the store from them."""
    with tempfile.TemporaryDirectory(prefix="n4a-conformal-load-") as tmp:
        extract_dir = Path(tmp) / "store"
        for member in sorted(expected):
            container.extract(member, extract_dir / Path(member).name)
        return load_conformal_result_store(extract_dir)


def _expected_conformal_members() -> set[str]:
    return {f"{BUNDLE_ROOT}{filename}" for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME)}


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
