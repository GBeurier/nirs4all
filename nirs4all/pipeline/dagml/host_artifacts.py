"""Bounded-memory sidecars for directory-backed Python host estimators."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from .autogluon_estimator import DagMLAutoGluonEstimator

CHUNK_SIZE = 1024 * 1024


def file_fingerprint(path: Path) -> tuple[str, int]:
    """Hash a file without holding its contents in memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _estimators(value: Any, seen: set[int]) -> Iterator[DagMLAutoGluonEstimator]:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, DagMLAutoGluonEstimator):
        yield value
        return
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    elif hasattr(value, "__dict__"):
        state = vars(value)
        children = [state[key] for key in (
            "estimator", "_model", "model", "steps", "estimators", "members", "member",
            "base", "learner", "base_members", "meta_member", "branches",
        ) if key in state]
    else:
        return
    for child in children:
        yield from _estimators(child, seen)


def _safe_uri(uri: str) -> Path:
    path = PurePosixPath(uri)
    if (not uri or "\\" in uri or path.is_absolute() or any(part in ("", ".", "..") for part in uri.split("/"))
            or ":" in path.parts[0]):
        raise ValueError(f"invalid host sidecar path: {uri!r}")
    return Path(*path.parts)


@contextmanager
def stage_host_artifacts(model: Any, root: Path, prefix: str) -> Iterator[list[dict[str, Any]]]:
    """Copy model directories as independent files while joblib stores references."""
    refs: list[dict[str, Any]] = []
    estimators = list(_estimators(model, set()))
    try:
        for index, estimator in enumerate(estimators):
            predictor = getattr(estimator, "predictor_", None)
            if predictor is None:
                continue
            source = Path(predictor.path)
            if not source.is_dir():
                raise ValueError(f"AutoGluon predictor directory is missing: {source}")
            identifier = f"autogluon_{index}"
            files: list[dict[str, Any]] = []
            for item in sorted(source.rglob("*")):
                if item.is_symlink():
                    raise ValueError("AutoGluon predictor directory contains a symlink")
                if not item.is_file():
                    continue
                relative = item.relative_to(source)
                uri = f"{prefix}/{identifier}/{relative.as_posix()}"
                target = root / _safe_uri(uri)
                target.parent.mkdir(parents=True, exist_ok=True)
                with item.open("rb") as read_stream, target.open("wb") as write_stream:
                    shutil.copyfileobj(read_stream, write_stream, CHUNK_SIZE)
                fingerprint, size = file_fingerprint(target)
                files.append({"uri": uri, "content_fingerprint": fingerprint, "size_bytes": size})
            if not files:
                raise ValueError("AutoGluon predictor directory is empty")
            estimator._external_artifact_id = identifier
            refs.append({"id": identifier, "files": files})
        yield refs
    finally:
        for estimator in estimators:
            estimator.__dict__.pop("_external_artifact_id", None)


def verify_host_artifacts(root: Path, refs: Any) -> dict[str, Path]:
    """Verify every declared sidecar before any model deserialization."""
    if refs is None:
        return {}
    if not isinstance(refs, list):
        raise ValueError("invalid host sidecar manifest")
    directories: dict[str, Path] = {}
    seen_uris: set[str] = set()
    for ref in refs:
        if (not isinstance(ref, dict) or not isinstance(ref.get("id"), str)
                or re.fullmatch(r"autogluon_\d+", ref["id"]) is None or ref["id"] in directories):
            raise ValueError("invalid host sidecar identifier")
        files = ref.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("host sidecar directory has no files")
        for file_ref in files:
            uri = file_ref.get("uri") if isinstance(file_ref, dict) else None
            if not isinstance(uri, str) or uri in seen_uris:
                raise ValueError("invalid or duplicate host sidecar member")
            seen_uris.add(uri)
            path = root / _safe_uri(uri)
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
                raise ValueError("host sidecar path escapes its archive directory")
            fingerprint, size = file_fingerprint(path)
            if fingerprint != file_ref.get("content_fingerprint") or size != file_ref.get("size_bytes"):
                raise ValueError(f"host sidecar integrity mismatch: {uri}")
            parts = PurePosixPath(uri).parts
            if parts[0] != "host_artifacts" or ref["id"] not in parts or parts[-1] == ref["id"]:
                raise ValueError("host sidecar identifier does not match path")
            directory = root.joinpath(*parts[:parts.index(ref["id"]) + 1])
            directories.setdefault(ref["id"], directory)
    return directories


def hydrate_host_artifacts(model: Any, directories: dict[str, Path], *, owner: Any = None) -> None:
    """Attach already verified AutoGluon directories to a loaded model."""
    estimators = list(_estimators(model, set()))
    pending = [estimator for estimator in estimators if hasattr(estimator, "_external_artifact_id")]
    if {estimator._external_artifact_id for estimator in pending} != set(directories):
        raise ValueError("host sidecar manifest does not match model references")
    for estimator in pending:
        estimator.load_external_directory(directories[estimator._external_artifact_id])
        if owner is not None:
            estimator._external_directory_owner = owner
