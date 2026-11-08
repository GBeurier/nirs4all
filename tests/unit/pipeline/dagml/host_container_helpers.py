"""Shared test helpers for Core ``.n4a`` V1 host containers.

Core refuses a container whose bytes no longer match its sealed inventory, so a
tamper test that edits a file on disk never reaches the SDK's semantic checks.
:func:`reseal_host_member` writes a NEW, fully valid container whose chosen member
holds different bytes (inventory digest and declared fingerprint recomputed), so the
semantic validator under test is the one that refuses.

SOURCE ONLY, UNQUALIFIED: written without being executed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from nirs4all.pipeline.dagml.host_container import (
    BLOB_CONTENT_TYPE,
    BUNDLE_CONTENT_TYPE,
    BUNDLE_MANIFEST_MEMBER,
    JSON_CONTENT_TYPE,
    HostMember,
    open_n4a_source,
    write_host_container,
)


def require_core() -> Any:
    """Return ``nirs4all_core``, or skip when it lacks the file-backed container API."""
    core = pytest.importorskip("nirs4all_core")
    if not callable(getattr(core, "open_n4a_file", None)) or not callable(getattr(core, "write_n4a_file", None)):
        pytest.skip("installed nirs4all-core lacks the file-backed .n4a container API")
    return core


def bundle_members(manifest: dict[str, Any] | None = None) -> list[HostMember]:
    """The two members every host model bundle starts with: bundle manifest and pipeline."""
    document = {"bundle_format_version": "1.0"} if manifest is None else manifest
    return [
        HostMember(BUNDLE_MANIFEST_MEMBER, json.dumps(document).encode(), "provenance", BUNDLE_CONTENT_TYPE),
        HostMember("pipeline.json", b"{}", "recipe", JSON_CONTENT_TYPE),
    ]


def write_host_bundle(path: Path, extra: Iterable[HostMember] = ()) -> Path:
    """Write a host model bundle (manifest, pipeline, then ``extra`` in order)."""
    return write_host_container(path, [*bundle_members(), *extra])


def blob(path: str, payload: bytes) -> HostMember:
    """An opaque attachment member."""
    return HostMember(path, payload, "attachment", BLOB_CONTENT_TYPE)


def reseal_host_member(source: Path, target: Path, member: str, payload: bytes | None, *, kind: str = "attachment", content_type: str = BLOB_CONTENT_TYPE) -> Path:
    """Write a valid copy of host container ``source`` at ``target`` with one member changed.

    Args:
        source: Existing host container; never modified.
        target: New container path.
        member: Archive path to replace; an absent path is appended as a new member.
        payload: New bytes, or ``None`` to remove the member.
        kind: Artifact kind of an appended member.
        content_type: Content type of an appended member.

    Returns:
        ``target``. Every other member keeps its bytes, kind, content type and declared position.
    """
    with open_n4a_source(source) as opened:
        container = opened.require_host()
        members: list[HostMember] = []
        found = False
        for name in container.namelist():
            artifact = container.artifact(name)
            data = container.read(name)
            if name == member:
                found = True
                if payload is None:
                    continue
                data = payload
            members.append(HostMember(name, data, artifact["kind"], artifact["content_type"]["id"]))
        if not found and payload is not None:
            members.append(HostMember(member, payload, kind, content_type))
        entrypoint = bool(container.manifest["entrypoints"])
    return write_host_container(target, members, entrypoint=entrypoint)
