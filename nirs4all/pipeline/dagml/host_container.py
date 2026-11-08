"""File-backed Core ``.n4a`` V1 container for Python host payloads.

This module is a thin adapter over Core's owner-neutral container
(``nirs4all_core.open_n4a_file`` / ``nirs4all_core.write_n4a_file``). It contains
no ZIP code, no container parser, no numerics and no fallback to another archive
path. Core owns the layout, integrity checks and the single verified open; the
SDK only declares its own identity (owner, feature, content types) and keeps its
serializers: members are opaque host bytes, joblib models stay trusted pickles,
and nothing is deserialized implicitly.

Core imposes no content-size cap and neither does this module. The optional
``memory_budget_bytes`` forwards the host's optional Core working-memory admission
budget. Core refusals are ``ValueError`` and propagate unchanged.

Members keep their SDK paths. The one SDK manifest lives at
:data:`BUNDLE_MANIFEST_MEMBER`. Every member is one Core artifact whose id is its
path. Core stores physical members sorted by path, so readers must use
:meth:`HostContainer.namelist` (declared order) and never physical order.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

HOST_OWNER = "nirs4all.python-host"
HOST_FEATURE = "nirs4all-python-host@1"
HOST_ENTRYPOINT_ID = "host-predict"
HOST_OPERATION = "host.predict"
BUNDLE_MANIFEST_MEMBER = "host/bundle_manifest.json"
BUNDLE_CONTENT_TYPE = "nirs4all.python-host.bundle"
JOBLIB_CONTENT_TYPE = "nirs4all.python-host.joblib"
JSON_CONTENT_TYPE = "nirs4all.python-host.json"
BLOB_CONTENT_TYPE = "nirs4all.python-host.blob"
FINGERPRINT_PROFILE = "sha256.raw.v1"
NATIVE_FEATURE = "native-methods-package@1"

_CONTENT_TYPE_VERSION = 1
_N4A_SCHEMA = "nirs4all.n4a.container.v1"
_PHYSICAL_PROFILE = "zip-stored.v1"
_STAGED_NAME = "container.n4a"
_PLACEHOLDER_SHA256 = "0" * 64
_MEMORY_PREFIX = "n4a memory refusal:"
_CHUNK_SIZE = 1024 * 1024
_ARTIFACT_KINDS = frozenset(
    {
        "recipe",
        "dataset",
        "model",
        "run",
        "result",
        "score",
        "prediction",
        "residual",
        "training_evidence",
        "full_refit",
        "provenance",
        "attachment",
    }
)
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")


def _core() -> Any:
    """Import Core and require the file-backed container functions."""
    from .core_archive_replay import CoreArchiveDependencyError

    mitigation = "install the nirs4all-core native wheel pinned by the release lock"
    try:
        core = importlib.import_module("nirs4all_core")
    except ImportError as error:
        raise CoreArchiveDependencyError("nirs4all-core", "host containers require a matching nirs4all-core native wheel", mitigation=mitigation) from error
    missing = [name for name in ("open_n4a_file", "write_n4a_file") if not callable(getattr(core, name, None))]
    if missing:
        raise CoreArchiveDependencyError(
            "nirs4all-core",
            "installed nirs4all-core lacks the file-backed .n4a container API: " + ", ".join(missing),
            mitigation=mitigation,
        )
    return core


def _publish(target: Path, fill: Callable[[Path], None]) -> None:
    """Fill a NEW staged path inside an SDK-owned scratch directory, then replace ``target``.

    The scratch directory is created beside ``target`` (same filesystem) and always
    removed. ``fill`` receives a path that does not exist yet: Core refuses an existing
    destination. Any failure leaves a previous ``target`` untouched.
    """
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
        staged = Path(scratch) / _STAGED_NAME
        fill(staged)
        os.replace(staged, target)


@dataclass(frozen=True)
class HostMember:
    """One host payload to store; list order of members is its semantic order.

    Attributes:
        path: Archive path, kept exactly.
        source: ``bytes`` held by the caller, or a regular file streamed by Core.
        kind: Core artifact kind (``recipe``, ``model``, ``provenance``, ``attachment``, ...).
        content_type: Host content type id (version 1).
        fingerprint: ``sha256:<hex>`` of the payload, computed from ``source`` when ``None``.
            It is checked against ``bytes`` sources and Core's final write receipt
            before publication, including streamed ``Path`` sources. Opening also
            compares it with Core's verified inventory (:func:`is_host_manifest`).
    """

    path: str
    source: bytes | Path
    kind: str
    content_type: str
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("a host member needs a nonempty archive path")
        if type(self.source) is not bytes and not isinstance(self.source, Path):
            raise TypeError("a host member source must be bytes or a pathlib.Path")
        if self.kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unknown host member kind: {self.kind!r}")
        if not isinstance(self.content_type, str) or not self.content_type:
            raise ValueError("a host member needs a nonempty content type")
        if self.fingerprint is not None and (not isinstance(self.fingerprint, str) or _FINGERPRINT.fullmatch(self.fingerprint) is None):
            raise ValueError("a host member fingerprint must be 'sha256:' and 64 lowercase hex digits")


def _fingerprint_hex(member: HostMember) -> str:
    source = member.source
    if isinstance(source, Path):
        if member.fingerprint is not None:
            return member.fingerprint.removeprefix("sha256:")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
    actual = hashlib.sha256(source).hexdigest()
    if member.fingerprint is not None and member.fingerprint != f"sha256:{actual}":
        raise ValueError(f"host member fingerprint does not match its bytes: {member.path}")
    return actual


def _checked_members(members: Sequence[HostMember]) -> list[HostMember]:
    items = list(members)
    seen: set[str] = set()
    for member in items:
        if not isinstance(member, HostMember):
            raise TypeError("host container members must be HostMember instances")
        if member.path in seen:
            raise ValueError(f"duplicate host member path: {member.path}")
        seen.add(member.path)
        if isinstance(member.source, Path) and not member.source.is_file():
            raise ValueError(f"host member source is not a regular file: {member.path}")
    return items


def _inventory_entry(path: str) -> dict[str, Any]:
    """New-member inventory placeholder: Core derives the real size and SHA-256."""
    return {"path": path, "byte_length": 0, "sha256": _PLACEHOLDER_SHA256}


def _artifact(member: HostMember) -> dict[str, Any]:
    return {
        "id": member.path,
        "kind": member.kind,
        "owner": HOST_OWNER,
        "content_type": {"id": member.content_type, "version": _CONTENT_TYPE_VERSION},
        "identity": {
            "owner": HOST_OWNER,
            "schema": member.content_type,
            "schema_version": _CONTENT_TYPE_VERSION,
            "fingerprint_profile": FINGERPRINT_PROFILE,
            "fingerprint": _fingerprint_hex(member),
        },
        "storage": {"kind": "embedded", "root_member": member.path, "member_paths": [member.path]},
        "requirements": None,
    }


def _entrypoint(bundle_path: str) -> dict[str, Any]:
    return {
        "id": HOST_ENTRYPOINT_ID,
        "owner": HOST_OWNER,
        "operation": HOST_OPERATION,
        "root_artifact": bundle_path,
        "input_artifacts": [],
        "required_features": [HOST_FEATURE],
        "requirements": None,
    }


def _manifest(members: list[dict[str, Any]], artifacts: list[dict[str, Any]], entrypoints: list[dict[str, Any]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": _N4A_SCHEMA,
        "schema_version": 1,
        "minimum_reader_version": 1,
        "container_id": f"container:nirs4all.host.{uuid.uuid4()}",
        "required_features": [HOST_FEATURE],
        "physical_profile": _PHYSICAL_PROFILE,
        "members": members,
        "artifacts": artifacts,
        "links": [],
        "entrypoints": entrypoints,
    }
    _classify(manifest, verified_inventory=False)
    return manifest


def _write(
    target: Path,
    manifest: dict[str, Any],
    sources: Sequence[HostMember],
    *,
    base: Any,
    memory_budget_bytes: int | None,
) -> Path:
    core = _core()
    manifest_json = json.dumps(manifest)
    pairs = [(member.path, member.source) for member in sources]

    def fill(staged: Path) -> None:
        receipt = core.write_n4a_file(staged, manifest_json, pairs, base=base, memory_budget_bytes=memory_budget_bytes)
        _require_declared_identities(manifest, receipt)

    _publish(target, fill)
    return target


def _require_declared_identities(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    """Refuse before publication a declared fingerprint that is not Core's written member digest.

    ``receipt["members"]`` is Core's own final inventory of the bytes it wrote, so a stale or
    false declaration (for example for a streamed ``Path`` source) is caught here without any
    rehash, reread or reopen. It runs before ``os.replace``: the previous destination is kept.
    """
    written = {entry["path"]: entry["sha256"] for entry in receipt["members"]}
    for artifact in manifest["artifacts"]:
        if artifact["identity"]["fingerprint"] != written.get(artifact["storage"]["root_member"]):
            raise ValueError(f"host artifact fingerprint is not the digest of the member Core wrote: {artifact['id']}")


def is_host_manifest(manifest: Mapping[str, Any]) -> bool:
    """Classify a Core-opened V1 manifest as a host container.

    The manifest must come from a container Core opened: its inventory SHA-256 values
    are then the verified digests, and every host artifact's declared ``sha256.raw.v1``
    fingerprint is compared with the inventory digest of its member. A fingerprint a
    writer merely declared (for example for a streamed ``Path`` source) is rejected here
    when it differs, never presented as verified.

    Returns:
        ``False`` when the manifest neither declares the host feature nor carries a
        host-owned artifact or entrypoint (a canonical or foreign-owner container);
        ``True`` for a valid host shape.

    Raises:
        ValueError: The host feature or host-owned objects are mixed with anything
            else, the host shape is violated, or a fingerprint is not the inventory
            digest of its member. No payload byte is read.
    """
    return _classify(manifest, verified_inventory=True)


def _classify(manifest: Mapping[str, Any], *, verified_inventory: bool) -> bool:
    """Shared classification; a manifest about to be written has placeholder digests, so it skips the fingerprint comparison."""
    features = manifest["required_features"]
    artifacts = manifest["artifacts"]
    entrypoints = manifest["entrypoints"]
    links = manifest["links"]
    host_owned = any(item["owner"] == HOST_OWNER for item in (*artifacts, *entrypoints, *links))
    if HOST_FEATURE not in features and not host_owned:
        return False
    if features != [HOST_FEATURE]:
        raise ValueError(f"a host container must require exactly the {HOST_FEATURE} feature")
    if links:
        raise ValueError("a host container declares no links")
    digests = {entry["path"]: entry["sha256"] for entry in manifest["members"]}
    bundles: list[str] = []
    for artifact in artifacts:
        storage = artifact["storage"]
        if artifact["owner"] != HOST_OWNER:
            raise ValueError("a host container cannot mix artifacts of another owner")
        if storage["kind"] != "embedded" or storage["member_paths"] != [storage["root_member"]] or artifact["id"] != storage["root_member"]:
            raise ValueError("every host artifact must embed exactly one member whose path is its id")
        identity = artifact["identity"]
        if verified_inventory and (identity["fingerprint_profile"] != FINGERPRINT_PROFILE or identity["fingerprint"] != digests[storage["root_member"]]):
            raise ValueError(f"host artifact fingerprint is not the verified inventory digest of its member: {artifact['id']}")
        if artifact["content_type"] == {"id": BUNDLE_CONTENT_TYPE, "version": _CONTENT_TYPE_VERSION}:
            bundles.append(artifact["id"])
    if len(entrypoints) > 1 or len(bundles) > 1:
        raise ValueError("a host container declares at most one bundle and one entrypoint")
    if entrypoints:
        entrypoint = entrypoints[0]
        if (
            entrypoint["id"] != HOST_ENTRYPOINT_ID
            or entrypoint["owner"] != HOST_OWNER
            or entrypoint["operation"] != HOST_OPERATION
            or entrypoint["input_artifacts"]
            or entrypoint["required_features"] != [HOST_FEATURE]
            or entrypoint["requirements"] is not None
            or bundles != [entrypoint["root_artifact"]]
        ):
            raise ValueError("the host entrypoint must be host-predict rooted at the single bundle artifact")
    elif bundles:
        raise ValueError("a host bundle requires the host-predict entrypoint")
    return True


def write_host_container(
    target: Path,
    members: Sequence[HostMember],
    *,
    entrypoint: bool = True,
    memory_budget_bytes: int | None = None,
) -> Path:
    """Write a new host container and publish it atomically at ``target``.

    Args:
        target: Destination; an existing regular file is replaced only on success.
            Its parent directory must already exist.
        members: Payloads in semantic (declared) order.
        entrypoint: Declare ``host-predict`` rooted at the single bundle member. A
            conformal-only container passes ``False`` and carries no bundle member.
        memory_budget_bytes: Optional Core working-memory budget.

    Returns:
        ``target``.

    Raises:
        ValueError: Invalid members or a Core refusal. A previous ``target`` is untouched.
    """
    items = _checked_members(members)
    bundles = [member.path for member in items if member.content_type == BUNDLE_CONTENT_TYPE]
    if entrypoint and len(bundles) != 1:
        raise ValueError("a host container with an entrypoint needs exactly one bundle member")
    manifest = _manifest(
        [_inventory_entry(member.path) for member in items],
        [_artifact(member) for member in items],
        [_entrypoint(bundles[0])] if entrypoint else [],
    )
    return _write(Path(target), manifest, items, base=None, memory_budget_bytes=memory_budget_bytes)


def update_host_container(
    base: HostContainer,
    target: Path,
    *,
    add: Sequence[HostMember],
    remove_prefix: str | None = None,
    memory_budget_bytes: int | None = None,
) -> Path:
    """Write ``base`` plus ``add`` minus ``remove_prefix`` as a new container at ``target``.

    ``base`` is never modified; Core copies and verifies the retained members from its
    open handle. A member of ``add`` whose path exists in ``base`` replaces it in place;
    other new members are appended in order. Members of ``base`` under ``remove_prefix``
    are dropped unless ``add`` replaces them. ``base`` may be the file at ``target``.

    Raises:
        ValueError: ``base`` is not a host container, the result is not a valid host
            shape (for example its bundle was removed), or Core refuses.
    """
    if not base.is_host:
        raise ValueError("only a host container can be updated")
    if remove_prefix is not None and (not isinstance(remove_prefix, str) or not remove_prefix):
        raise ValueError("remove_prefix must be a nonempty string")
    added = _checked_members(add)
    replacements = {member.path: member for member in added}
    artifacts = {artifact["id"]: artifact for artifact in base.manifest["artifacts"]}
    members: list[dict[str, Any]] = []
    declared: list[dict[str, Any]] = []
    for entry in base.manifest["members"]:
        path = entry["path"]
        if path in replacements:
            members.append(_inventory_entry(path))
            declared.append(_artifact(replacements[path]))
        elif remove_prefix is None or not path.startswith(remove_prefix):
            members.append(dict(entry))
            declared.append(copy.deepcopy(artifacts[path]))
    existing = set(base.namelist())
    for member in added:
        if member.path not in existing:
            members.append(_inventory_entry(member.path))
            declared.append(_artifact(member))
    manifest = _manifest(members, declared, copy.deepcopy(base.manifest["entrypoints"]))
    return _write(Path(target), manifest, added, base=base.core_file, memory_budget_bytes=memory_budget_bytes)


class HostContainer:
    """A verified Core V1 file handle read in declared manifest order.

    Obtain it from :func:`open_n4a_source`; it is not constructed directly. The caller
    owns its lifetime and must :meth:`close` it (or use the owning :class:`N4aSource`).
    The file is not locked: Core hashes the bytes each read actually returns, so a file
    changed after the open is refused at the read.

    Attributes:
        path: The container file.
        manifest: The stored manifest decoded once at open; treat as read-only.
        is_host: ``True`` when the manifest is a valid host container.
        core_file: The one Core ``N4aFile`` (for Core file-handle actions).
    """

    def __init__(self, path: Path, core_file: Any, manifest: dict[str, Any], *, is_host: bool) -> None:
        self.path = path
        self.core_file = core_file
        self.manifest = manifest
        self.is_host = is_host
        self._inventory: dict[str, dict[str, Any]] = {entry["path"]: entry for entry in manifest["members"]}
        self._artifacts: dict[str, dict[str, Any]] = {member: artifact for artifact in manifest["artifacts"] if artifact["storage"]["kind"] == "embedded" for member in artifact["storage"]["member_paths"]}
        self._sha256: str | None = None
        self._bundle_manifest: dict[str, Any] | None = None
        self._bundle_loaded = False

    def _require(self, member: str) -> dict[str, Any]:
        try:
            return self._inventory[member]
        except KeyError:
            raise KeyError(member) from None

    @property
    def sha256(self) -> str:
        """``sha256:`` plus Core's container digest, read once from Core's report."""
        if self._sha256 is None:
            self._sha256 = "sha256:" + self.core_file.view()["readability"]["container_sha256"]
        return self._sha256

    @property
    def bundle_manifest(self) -> dict[str, Any] | None:
        """The SDK bundle manifest, or ``None`` when the container has none; treat as read-only."""
        if not self._bundle_loaded:
            if BUNDLE_MANIFEST_MEMBER in self._inventory:
                manifest = self.read_json(BUNDLE_MANIFEST_MEMBER)
                if not isinstance(manifest, dict):
                    raise ValueError("the host bundle manifest must be a JSON object")
                self._bundle_manifest = manifest
            self._bundle_loaded = True
        return self._bundle_manifest

    def namelist(self) -> list[str]:
        """Member paths in the declared manifest order, never the physical order."""
        return [entry["path"] for entry in self.manifest["members"]]

    def artifact(self, member: str) -> dict[str, Any]:
        """The manifest artifact that owns ``member`` (``KeyError`` if absent)."""
        self._require(member)
        return self._artifacts[member]

    def size(self, member: str) -> int:
        """Inventory size in bytes; reads no payload."""
        return int(self._require(member)["byte_length"])

    def digest(self, member: str) -> str:
        """Inventory ``sha256:<hex>`` of a member; reads no payload."""
        digest: str = "sha256:" + self._require(member)["sha256"]
        return digest

    def read(self, member: str) -> bytes:
        """One whole member in RAM, verified by Core to its end."""
        self._require(member)
        data: bytes = self.core_file.member(member)
        return data

    def read_json(self, member: str) -> Any:
        """Decode one member as JSON."""
        return json.loads(self.read(member))

    def extract(self, member: str, destination: Path) -> None:
        """Stream one member into a NEW file, verified by Core during that single read.

        Parents are created and the destination is exclusive: an existing file or symlink
        raises ``FileExistsError`` and is left alone. On any other failure the partial
        destination is unlinked and the error propagates.
        """
        self._require(member)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stream = destination.open("xb")
        try:
            with stream:
                self.core_file.copy_member_to(member, stream)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def copy_to(self, target: Path) -> None:
        """Copy the exact archive bytes to ``target``, checked against the container digest.

        Publishes atomically like the writers: the old ``target`` stays on failure.
        """

        def fill(staged: Path) -> None:
            with staged.open("xb") as stream:
                self.core_file.copy_archive_to(stream)

        _publish(Path(target), fill)

    def view(self) -> dict[str, Any]:
        """Core's readability report; executability is always ``not_evaluated``."""
        view: dict[str, Any] = self.core_file.view()
        return view

    def close(self) -> None:
        """Release the Core handle; idempotent."""
        if not self.core_file.closed:
            self.core_file.close()

    def __enter__(self) -> HostContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_host_joblib(container: HostContainer, member: str, scratch: Path) -> Any:
    """Load one host joblib member after full verified private extraction.

    The caller must have passed every semantic validator (engine and owner policy,
    topology, generated view, attested capture, sidecar verification) BEFORE this call
    and must supply a private ``scratch`` directory. Joblib can execute code: digests
    detect corruption, not malice.

    Raises:
        ValueError: Not a host container, the member is not a host joblib artifact,
            ``scratch`` is not a directory, or Core refuses the member bytes.
    """
    if not container.is_host:
        raise ValueError("joblib members are loaded only from host containers")
    artifact = container.artifact(member)
    if artifact["owner"] != HOST_OWNER or artifact["content_type"] != {"id": JOBLIB_CONTENT_TYPE, "version": _CONTENT_TYPE_VERSION}:
        raise ValueError(f"host member is not a host joblib artifact: {member}")
    scratch = Path(scratch)
    if not scratch.is_dir():
        raise ValueError("the private joblib scratch must be an existing directory")
    import joblib

    private = scratch / f"{uuid.uuid4().hex}.joblib"
    container.extract(member, private)
    try:
        return joblib.load(private)
    finally:
        private.unlink(missing_ok=True)


@dataclass
class N4aSource:
    """The outcome of one Core open of a path; the caller owns and closes it.

    Attributes:
        path: The inspected path.
        kind: ``host`` (valid host container), ``native`` (a Core container that declares
            the native Methods package feature; Core's ``open_native_package`` selected and
            validated its closure), ``canonical`` (any other V1 container, readable but not
            executable here) or ``none``.
        container: The V1 handle for ``host``, ``native`` and ``canonical``, else ``None``.
        package_version: ``2`` or ``3`` for ``native``, taken from Core's native view
            (``package_schema_version``), else ``None``.
        owns_core_file: False for a borrowed Core file; close releases local references only.
        native_package: Core's retained ephemeral ``NativePackage`` for ``native``, else
            ``None``. Pass it to every Core native read, inspect and replay of the same public
            call: they reuse its selected proof, so Core opens and validates the closure once.
    """

    path: Path
    kind: Literal["host", "native", "canonical", "none"]
    container: HostContainer | None
    package_version: int | None = None
    native_package: Any = None
    owns_core_file: bool = True

    def require_host(self) -> HostContainer:
        """The host container, or ``ValueError`` for any other kind."""
        if self.kind != "host" or self.container is None:
            raise ValueError(f"{self.path} is not a host container (kind={self.kind})")
        return self.container

    def native_input(self) -> Any:
        """The retained Core ``NativePackage`` of a native container."""
        if self.kind != "native" or self.native_package is None:
            raise ValueError(f"{self.path} is not a native package container (kind={self.kind})")
        return self.native_package

    def close(self) -> None:
        """Close the V1 handle if any and release the retained native proof; idempotent.

        Core documents no separate proof close: dropping this reference ends the proof's
        lifetime together with the closed original handle. A source borrowed from a public
        call is closed only by its owner.
        """
        if self.container is not None and self.owns_core_file:
            self.container.close()
        if not self.owns_core_file:
            self.container = None
        self.native_package = None

    def __enter__(self) -> N4aSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def declares_native_package(manifest: Mapping[str, Any]) -> bool:
    """Whether a Core-opened manifest declares the native Methods package feature.

    This is a routing declaration only: Core's ``open_native_package`` selects the root and
    entrypoint and validates the closure (an ambiguous or invalid closure is refused there).
    """
    return NATIVE_FEATURE in manifest["required_features"] or any(NATIVE_FEATURE in entrypoint["required_features"] for entrypoint in manifest["entrypoints"])


def open_n4a_source(path: Path, *, memory_budget_bytes: int | None = None, core_file: Any = None) -> N4aSource:
    """Open ``path`` once through Core and classify it; V1 is decided only by Core.

    ``core_file`` borrows a retained Core N4aFile: it is never reopened or closed here,
    even on refusal. Its receipt binds the bytes; ``path`` is caller context only. The
    optional budget applies to new opens; a borrowed file retains its own budget.
    One call makes at most one ``open_n4a_file``; there is no cache, so pass the returned
    source to every consumer of the same public call and close it (context manager).
    A pre-V1 interim ``.n4a`` is refused by Core as an unsupported format and is reported
    here with the re-export message; no reader or converter exists for it. A container that
    declares the native package feature is opened once more by Core's ``open_native_package``
    over the same handle; the SDK selects no root and admits no entrypoint.

    Raises:
        ValueError: Core refused a ``.n4a`` file, Core refused the memory budget, or the
            container mixes the host feature with another owner.
        CoreArchiveDependencyError: Core lacks the file-backed container API.
    """
    candidate = Path(path)
    owns_core_file = core_file is None
    if owns_core_file and not candidate.is_file():
        return N4aSource(candidate, "none", None)
    core = _core()
    if owns_core_file:
        try:
            core_file = core.open_n4a_file(candidate, memory_budget_bytes=memory_budget_bytes)
        except ValueError as error:
            message = str(error)
            if message.startswith(_MEMORY_PREFIX):
                raise
            if candidate.suffix.lower() == ".n4a":
                raise ValueError(f"{message}; {candidate} is not a readable .n4a V1 container: re-export it with the current nirs4all") from error
            return N4aSource(candidate, "none", None)
    try:
        manifest = json.loads(core_file.manifest_bytes())
        is_host = is_host_manifest(manifest)
        native_package = None if is_host or not declares_native_package(manifest) else _open_native_package(core, core_file)
    except BaseException:
        if owns_core_file:
            core_file.close()
        raise
    container = HostContainer(candidate, core_file, manifest, is_host=is_host)
    if is_host:
        return N4aSource(candidate, "host", container, owns_core_file=owns_core_file)
    if native_package is not None:
        return N4aSource(candidate, "native", container, int(native_package.view()["package_schema_version"]), native_package, owns_core_file)
    return N4aSource(candidate, "canonical", container, owns_core_file=owns_core_file)


def _open_native_package(core: Any, core_file: Any) -> Any:
    """Core's single native-closure open over the already verified handle."""
    opener = getattr(core, "open_native_package", None)
    if not callable(opener):
        from .core_archive_replay import CoreArchiveDependencyError

        raise CoreArchiveDependencyError(
            "nirs4all-core",
            "installed nirs4all-core lacks open_native_package for native Package V2/V3 containers",
            mitigation="install the nirs4all-core native wheel pinned by the release lock",
        )
    return opener(core_file)
