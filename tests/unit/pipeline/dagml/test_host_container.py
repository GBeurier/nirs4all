"""Host container foundation over Core's file-backed neutral V1 container.

SOURCE ONLY, UNQUALIFIED: written without being executed. Tests marked ``real Core``
need a ``nirs4all_core`` wheel exposing ``open_n4a_file``/``write_n4a_file`` and skip
otherwise; the others use a recording fake of the Core functions.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nirs4all.pipeline.dagml import host_container as hc
from nirs4all.pipeline.dagml.host_container import (
    BLOB_CONTENT_TYPE,
    BUNDLE_CONTENT_TYPE,
    BUNDLE_MANIFEST_MEMBER,
    HOST_FEATURE,
    JOBLIB_CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    HostContainer,
    HostMember,
    N4aSource,
    is_host_manifest,
    load_host_joblib,
    open_n4a_source,
    update_host_container,
    write_host_container,
)

MODEL_PAYLOAD = b"UNIQUE-MODEL-PAYLOAD-0123456789"


def _listing(directory: Path) -> frozenset[str]:
    """Entry names of ``directory``; the SDK conftest may pre-populate ``tmp_path``."""
    return frozenset(entry.name for entry in directory.iterdir())


def _members() -> list[HostMember]:
    """Declared order differs from the sorted physical order (sub2 before sub10)."""
    return [
        HostMember(BUNDLE_MANIFEST_MEMBER, json.dumps({"source_type": "legacy"}).encode(), "provenance", BUNDLE_CONTENT_TYPE),
        HostMember("pipeline.json", b'{"steps": []}', "recipe", JSON_CONTENT_TYPE),
        HostMember("artifacts/sub2.joblib", MODEL_PAYLOAD, "model", JOBLIB_CONTENT_TYPE),
        HostMember("artifacts/sub10.joblib", b"ten", "model", JOBLIB_CONTENT_TYPE),
    ]


class _WriteSpy:
    """Records Core writes; creates the staged file like Core, optionally failing afterwards."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[SimpleNamespace] = []

    def write_n4a_file(self, destination: Any, manifest_json: str, members: Any, *, base: Any = None, memory_budget_bytes: Any = None) -> dict[str, Any]:
        destination = Path(destination)
        assert not destination.exists(), "Core must receive a NEW path, never a pre-created file"
        self.calls.append(SimpleNamespace(destination=destination, manifest=json.loads(manifest_json), members=list(members), base=base, budget=memory_budget_bytes))
        destination.write_bytes(b"staged-container")
        if self.fail:
            raise ValueError("n4a integrity refusal: simulated")
        return self.receipt(self.calls[-1])

    @staticmethod
    def receipt(call: SimpleNamespace) -> dict[str, Any]:
        """Core's final receipt: written bytes for new members, the base inventory for retained ones."""
        digests = {entry["path"]: entry["sha256"] for entry in call.manifest["members"]}
        for path, source in call.members:
            digests[path] = hashlib.sha256(source if isinstance(source, bytes) else Path(source).read_bytes()).hexdigest()
        return {
            "container_sha256": "0" * 64,
            "manifest_sha256": "1" * 64,
            "byte_length": 0,
            "members": [{"path": entry["path"], "byte_length": 0, "sha256": digests[entry["path"]]} for entry in call.manifest["members"]],
        }


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _WriteSpy:
    recorder = _WriteSpy()
    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(write_n4a_file=recorder.write_n4a_file))
    return recorder


@pytest.fixture
def real_core() -> Any:
    core = pytest.importorskip("nirs4all_core")
    if not callable(getattr(core, "open_n4a_file", None)) or not callable(getattr(core, "write_n4a_file", None)):
        pytest.skip("installed nirs4all-core lacks the file-backed .n4a container API")
    return core


def _captured_manifest(spy: _WriteSpy, tmp_path: Path, members: list[HostMember] | None = None, **kwargs: Any) -> dict[str, Any]:
    """The manifest as Core would store it: the inventory carries the verified digests."""
    write_host_container(tmp_path / "captured.n4a", members or _members(), **kwargs)
    manifest = copy.deepcopy(spy.calls[-1].manifest)
    declared = {artifact["storage"]["root_member"]: artifact["identity"]["fingerprint"] for artifact in manifest["artifacts"]}
    for entry in manifest["members"]:
        entry["sha256"] = declared[entry["path"]]
    return manifest


# --------------------------------------------------------------------------- members


def test_host_member_validates_its_inputs() -> None:
    with pytest.raises(TypeError):
        HostMember("a", "text", "model", JOBLIB_CONTENT_TYPE)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HostMember("a", bytearray(b"x"), "model", JOBLIB_CONTENT_TYPE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        HostMember("a", b"x", "weights", JOBLIB_CONTENT_TYPE)
    with pytest.raises(ValueError, match="path"):
        HostMember("", b"x", "model", JOBLIB_CONTENT_TYPE)
    with pytest.raises(ValueError, match="content type"):
        HostMember("a", b"x", "model", "")
    with pytest.raises(ValueError, match="fingerprint"):
        HostMember("a", b"x", "model", JOBLIB_CONTENT_TYPE, fingerprint="sha256:ABC")


def test_bytes_fingerprint_is_checked_and_a_path_fingerprint_is_only_declared(spy: _WriteSpy, tmp_path: Path) -> None:
    good = "sha256:" + hashlib.sha256(b"payload").hexdigest()
    target = tmp_path / "out.n4a"
    write_host_container(target, [HostMember("p.bin", b"payload", "attachment", BLOB_CONTENT_TYPE, fingerprint=good)], entrypoint=False)
    assert spy.calls[-1].manifest["artifacts"][0]["identity"]["fingerprint"] == good.removeprefix("sha256:")
    other = tmp_path / "other.n4a"
    with pytest.raises(ValueError, match="does not match its bytes"):
        write_host_container(other, [HostMember("p.bin", b"payload", "attachment", BLOB_CONTENT_TYPE, fingerprint="sha256:" + "1" * 64)], entrypoint=False)
    assert not other.exists()
    source = tmp_path / "sidecar.bin"
    source.write_bytes(b"large")
    known = "sha256:" + hashlib.sha256(b"large").hexdigest()
    write_host_container(tmp_path / "p.n4a", [HostMember("s.bin", source, "attachment", BLOB_CONTENT_TYPE, fingerprint=known)], entrypoint=False)
    assert spy.calls[-1].manifest["artifacts"][0]["identity"]["fingerprint"] == known.removeprefix("sha256:")
    assert spy.calls[-1].members == [("s.bin", source)]


# --------------------------------------------------------------------------- writing (fake Core)


def test_write_declares_the_host_identity_in_declared_order(spy: _WriteSpy, tmp_path: Path) -> None:
    manifest = _captured_manifest(spy, tmp_path)
    assert manifest["schema"] == "nirs4all.n4a.container.v1"
    assert manifest["physical_profile"] == "zip-stored.v1"
    assert manifest["required_features"] == [HOST_FEATURE]
    assert manifest["container_id"].startswith("container:nirs4all.host.")
    assert manifest["links"] == []
    assert [entry["path"] for entry in manifest["members"]] == [member.path for member in _members()]
    assert [artifact["id"] for artifact in manifest["artifacts"]] == [member.path for member in _members()]
    entrypoint = manifest["entrypoints"][0]
    assert (entrypoint["id"], entrypoint["operation"], entrypoint["root_artifact"]) == ("host-predict", "host.predict", BUNDLE_MANIFEST_MEMBER)
    assert is_host_manifest(manifest) is True
    assert spy.calls[-1].base is None
    assert spy.calls[-1].members == [(member.path, member.source) for member in _members()]


def test_write_forwards_the_budget_and_returns_the_target(spy: _WriteSpy, tmp_path: Path) -> None:
    target = tmp_path / "out.n4a"
    assert write_host_container(target, _members(), memory_budget_bytes=1 << 20) == target
    assert spy.calls[-1].budget == 1 << 20
    assert target.read_bytes() == b"staged-container"


def test_write_publishes_through_a_new_staged_path_and_cleans_up(spy: _WriteSpy, tmp_path: Path) -> None:
    target = tmp_path / "out.n4a"
    target.write_bytes(b"old")
    before = _listing(tmp_path)
    write_host_container(target, _members())
    staged = spy.calls[-1].destination
    assert staged.parent.parent == tmp_path and staged.parent != tmp_path
    assert not staged.parent.exists()
    assert target.read_bytes() == b"staged-container"
    assert _listing(tmp_path) == before, "no staging entry added or removed besides the replaced destination"


def test_failed_write_keeps_the_old_destination_and_leaves_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    failing = _WriteSpy(fail=True)
    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(write_n4a_file=failing.write_n4a_file))
    target = tmp_path / "out.n4a"
    target.write_bytes(b"old")
    before = _listing(tmp_path)
    with pytest.raises(ValueError, match="simulated"):
        write_host_container(target, _members())
    assert target.read_bytes() == b"old"
    assert _listing(tmp_path) == before, "the failed write left no staging entry"


def test_write_refuses_invalid_member_sets_before_touching_the_destination(spy: _WriteSpy, tmp_path: Path) -> None:
    target = tmp_path / "out.n4a"
    before = _listing(tmp_path)
    bundle = _members()[0]
    other = HostMember("pipeline.json", b"{}", "recipe", JSON_CONTENT_TYPE)
    with pytest.raises(ValueError, match="at least|exactly one bundle"):
        write_host_container(target, [])
    with pytest.raises(ValueError, match="exactly one bundle"):
        write_host_container(target, [other])
    with pytest.raises(ValueError, match="exactly one bundle"):
        write_host_container(target, [bundle, HostMember("host/other.json", b"{}", "provenance", BUNDLE_CONTENT_TYPE)])
    with pytest.raises(ValueError, match="bundle"):
        write_host_container(target, [bundle, other], entrypoint=False)
    with pytest.raises(ValueError, match="duplicate"):
        write_host_container(target, [bundle, other, other])
    with pytest.raises(ValueError, match="regular file"):
        write_host_container(target, [bundle, HostMember("m.bin", tmp_path / "missing.bin", "model", JOBLIB_CONTENT_TYPE)])
    with pytest.raises(TypeError):
        write_host_container(target, [bundle, "pipeline.json"])  # type: ignore[list-item]
    assert spy.calls == []
    assert _listing(tmp_path) == before, "a refused member set creates no destination and no staging entry"
    assert not target.exists()


def test_conformal_only_container_has_no_entrypoint(spy: _WriteSpy, tmp_path: Path) -> None:
    members = [HostMember("conformal/state.json", b"{}", "result", JSON_CONTENT_TYPE)]
    manifest = _captured_manifest(spy, tmp_path, members, entrypoint=False)
    assert manifest["entrypoints"] == []
    assert is_host_manifest(manifest) is True


# --------------------------------------------------------------------------- update (fake Core)


class _FakeFile:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.closed = False

    def manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest).encode()

    def close(self) -> None:
        self.closed = True


def _fake_base(manifest: dict[str, Any], tmp_path: Path) -> HostContainer:
    return HostContainer(tmp_path / "base.n4a", _FakeFile(manifest), manifest, is_host=True)


def test_update_replaces_in_place_drops_by_prefix_and_appends(spy: _WriteSpy, tmp_path: Path) -> None:
    members = [*_members(), HostMember("conformal/old.json", b"{}", "result", JSON_CONTENT_TYPE)]
    base = _fake_base(_captured_manifest(spy, tmp_path, members), tmp_path)
    old_id = base.manifest["container_id"]
    result = update_host_container(
        base,
        tmp_path / "updated.n4a",
        add=[
            HostMember("artifacts/sub2.joblib", b"two-v2", "model", JOBLIB_CONTENT_TYPE),
            HostMember("conformal/new.json", b"{}", "result", JSON_CONTENT_TYPE),
        ],
        remove_prefix="conformal/",
    )
    call = spy.calls[-1]
    assert result == tmp_path / "updated.n4a"
    assert [entry["path"] for entry in call.manifest["members"]] == [BUNDLE_MANIFEST_MEMBER, "pipeline.json", "artifacts/sub2.joblib", "artifacts/sub10.joblib", "conformal/new.json"]
    assert [artifact["id"] for artifact in call.manifest["artifacts"]] == [entry["path"] for entry in call.manifest["members"]]
    assert call.manifest["container_id"] != old_id
    assert call.manifest["entrypoints"] == base.manifest["entrypoints"]
    assert call.members == [("artifacts/sub2.joblib", b"two-v2"), ("conformal/new.json", b"{}")]
    assert call.base is base.core_file
    replaced = next(artifact for artifact in call.manifest["artifacts"] if artifact["id"] == "artifacts/sub2.joblib")
    assert replaced["identity"]["fingerprint"] == hashlib.sha256(b"two-v2").hexdigest()
    retained = next(artifact for artifact in call.manifest["artifacts"] if artifact["id"] == "pipeline.json")
    assert retained == next(artifact for artifact in base.manifest["artifacts"] if artifact["id"] == "pipeline.json")


def test_update_keeps_a_prefixed_member_that_add_replaces(spy: _WriteSpy, tmp_path: Path) -> None:
    members = [*_members(), HostMember("conformal/state.json", b"{}", "result", JSON_CONTENT_TYPE)]
    base = _fake_base(_captured_manifest(spy, tmp_path, members), tmp_path)
    update_host_container(base, tmp_path / "u.n4a", add=[HostMember("conformal/state.json", b"[]", "result", JSON_CONTENT_TYPE)], remove_prefix="conformal/")
    assert spy.calls[-1].manifest["members"][-1]["path"] == "conformal/state.json"


def test_update_refuses_a_result_that_is_not_a_host_shape(spy: _WriteSpy, tmp_path: Path) -> None:
    base = _fake_base(_captured_manifest(spy, tmp_path), tmp_path)
    before = len(spy.calls)
    with pytest.raises(ValueError, match="host-predict|bundle"):
        update_host_container(base, tmp_path / "u.n4a", add=[], remove_prefix="host/")
    with pytest.raises(ValueError, match="bundle"):
        update_host_container(base, tmp_path / "u.n4a", add=[HostMember("host/second.json", b"{}", "provenance", BUNDLE_CONTENT_TYPE)])
    with pytest.raises(ValueError, match="remove_prefix"):
        update_host_container(base, tmp_path / "u.n4a", add=[], remove_prefix="")
    foreign = HostContainer(tmp_path / "f.n4a", _FakeFile({}), {"members": [], "artifacts": []}, is_host=False)
    with pytest.raises(ValueError, match="host container"):
        update_host_container(foreign, tmp_path / "u.n4a", add=[])
    assert len(spy.calls) == before


# --------------------------------------------------------------------------- manifest classification


def test_is_host_manifest_classifies_and_refuses_mixed_shapes(spy: _WriteSpy, tmp_path: Path) -> None:
    host = _captured_manifest(spy, tmp_path)
    assert is_host_manifest(host) is True

    foreign = copy.deepcopy(host)
    foreign["required_features"] = []
    for artifact in foreign["artifacts"]:
        artifact["owner"] = "dag-ml"
        artifact["identity"]["owner"] = "dag-ml"
    foreign["entrypoints"] = []
    assert is_host_manifest(foreign) is False

    def mutated(change: Any) -> dict[str, Any]:
        manifest = copy.deepcopy(host)
        change(manifest)
        return manifest

    broken = {
        "extra feature": lambda m: m["required_features"].append("other@1"),
        "feature missing": lambda m: m.update(required_features=[]),
        "foreign artifact": lambda m: m["artifacts"][1].update(owner="dag-ml"),
        "link": lambda m: m["links"].append({"id": "l", "owner": "x", "relation": "r", "source_artifact": "a", "target_artifact": "b"}),
        "reference artifact": lambda m: m["artifacts"][1].update(storage={"kind": "reference", "root_sha256": None}),
        "multi-member artifact": lambda m: m["artifacts"][1]["storage"].update(member_paths=["pipeline.json", "x"]),
        "id differs from path": lambda m: m["artifacts"][1].update(id="renamed"),
        "second bundle": lambda m: m["artifacts"][1].update(content_type={"id": BUNDLE_CONTENT_TYPE, "version": 1}),
        "wrong operation": lambda m: m["entrypoints"][0].update(operation="other.predict"),
        "foreign entrypoint": lambda m: m["entrypoints"][0].update(owner="dag-ml"),
        "entrypoint inputs": lambda m: m["entrypoints"][0].update(input_artifacts=["pipeline.json"]),
        "entrypoint root": lambda m: m["entrypoints"][0].update(root_artifact="pipeline.json"),
        "two entrypoints": lambda m: m["entrypoints"].append(copy.deepcopy(m["entrypoints"][0])),
        "bundle without entrypoint": lambda m: m.update(entrypoints=[]),
    }
    for change in broken.values():
        with pytest.raises(ValueError):
            is_host_manifest(mutated(change))


def test_a_declared_fingerprint_must_equal_the_verified_inventory_digest(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.bin"
    sidecar.write_bytes(b"real sidecar bytes")
    real = "sha256:" + hashlib.sha256(b"real sidecar bytes").hexdigest()
    members = [*_members(), HostMember("host_artifacts/autogluon_0/part.bin", sidecar, "attachment", BLOB_CONTENT_TYPE, fingerprint=real)]
    manifest = _captured_manifest(spy, tmp_path, members)
    # A container stored elsewhere whose writer declared a different fingerprint than the bytes Core verified.
    artifact = next(item for item in manifest["artifacts"] if item["id"] == "host_artifacts/autogluon_0/part.bin")
    artifact["identity"]["fingerprint"] = "3" * 64
    with pytest.raises(ValueError, match="verified inventory digest.*part.bin"):
        is_host_manifest(manifest)
    wrong_profile = _captured_manifest(spy, tmp_path)
    wrong_profile["artifacts"][1]["identity"]["fingerprint_profile"] = "sha256.canonical.v1"
    with pytest.raises(ValueError, match="verified inventory digest"):
        is_host_manifest(wrong_profile)
    handle = _FakeFile(manifest)
    path = tmp_path / "model.n4a"
    path.write_bytes(b"x")
    _fake_open(monkeypatch, handle)
    with pytest.raises(ValueError, match="verified inventory digest"):
        open_n4a_source(path)
    assert handle.closed


def test_a_stale_path_fingerprint_is_refused_before_publication_and_keeps_the_destination(spy: _WriteSpy, tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.bin"
    sidecar.write_bytes(b"bytes at hash time")
    stale = "sha256:" + hashlib.sha256(b"bytes at hash time").hexdigest()
    sidecar.write_bytes(b"bytes Core wrote after a mutation")
    target = tmp_path / "out.n4a"
    target.write_bytes(b"old")
    before = _listing(tmp_path)
    with pytest.raises(ValueError, match="digest of the member Core wrote.*part.bin"):
        write_host_container(target, [HostMember("part.bin", sidecar, "attachment", BLOB_CONTENT_TYPE, fingerprint=stale)], entrypoint=False)
    assert len(spy.calls) == 1 and spy.calls[0].members == [("part.bin", sidecar)]
    assert target.read_bytes() == b"old"
    assert _listing(tmp_path) == before and {"out.n4a", "sidecar.bin"} <= before, "the refused write left no staging entry"


def test_host_owned_objects_without_the_feature_are_refused_not_classified_canonical(spy: _WriteSpy, tmp_path: Path) -> None:
    manifest = _captured_manifest(spy, tmp_path)
    manifest["required_features"] = []
    with pytest.raises(ValueError, match="feature"):
        is_host_manifest(manifest)


# --------------------------------------------------------------------------- dispatch (fake Core)


def _manifest_for(kind: str, spy: _WriteSpy, tmp_path: Path) -> dict[str, Any]:
    host = _captured_manifest(spy, tmp_path)
    if kind == "host":
        return host
    foreign = copy.deepcopy(host)
    foreign["required_features"] = [] if kind == "canonical" else [HOST_FEATURE]
    foreign["entrypoints"] = []
    for artifact in foreign["artifacts"][1:]:
        artifact["owner"] = "dag-ml"
        artifact["identity"]["owner"] = "dag-ml"
    foreign["artifacts"][0]["owner"] = "dag-ml"
    foreign["artifacts"][0]["identity"]["owner"] = "dag-ml"
    if kind == "mixed":
        foreign["artifacts"][0]["owner"] = hc.HOST_OWNER
        foreign["artifacts"][0]["identity"]["owner"] = hc.HOST_OWNER
    return foreign


def _fake_open(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[Path]:
    calls: list[Path] = []

    def open_n4a_file(path: Any, *, memory_budget_bytes: Any = None) -> Any:
        calls.append(Path(path))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(open_n4a_file=open_n4a_file))
    return calls


def test_a_missing_path_or_directory_is_none_without_any_core_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hc, "_core", lambda: pytest.fail("Core was called"))
    assert open_n4a_source(tmp_path / "missing.n4a") == N4aSource(tmp_path / "missing.n4a", "none", None)
    assert open_n4a_source(tmp_path).kind == "none"


def test_host_container_is_opened_exactly_once_and_closed_with_the_source(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    manifest = _manifest_for("host", spy, tmp_path)
    handle = _FakeFile(manifest)
    path = tmp_path / "model.n4a"
    path.write_bytes(b"x")
    calls = _fake_open(monkeypatch, handle)
    with open_n4a_source(path, memory_budget_bytes=4096) as source:
        assert source.kind == "host" and source.require_host() is source.container
        assert source.container is not None and source.container.is_host and source.container.core_file is handle
        assert source.container.namelist() == [member.path for member in _members()]
        assert not handle.closed
    assert handle.closed and calls == [path]


def test_other_v1_containers_are_canonical_and_never_host(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    handle = _FakeFile(_manifest_for("canonical", spy, tmp_path))
    path = tmp_path / "native.n4a"
    path.write_bytes(b"x")
    _fake_open(monkeypatch, handle)
    source = open_n4a_source(path)
    assert source.kind == "canonical" and source.container is not None and not source.container.is_host
    with pytest.raises(ValueError, match="not a host container"):
        source.require_host()
    source.close()
    assert handle.closed


def test_a_mixed_host_container_is_refused_and_closed(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    handle = _FakeFile(_manifest_for("mixed", spy, tmp_path))
    path = tmp_path / "mixed.n4a"
    path.write_bytes(b"x")
    _fake_open(monkeypatch, handle)
    with pytest.raises(ValueError):
        open_n4a_source(path)
    assert handle.closed


def test_a_core_not_v1_refusal_is_a_reexport_error_never_a_reader_or_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "archive.n4a"
    path.write_bytes(b"x")
    calls = _fake_open(monkeypatch, ValueError("n4a unsupported format: only nirs4all.n4a.container.v1 / zip-stored.v1 reader 1 is supported"))
    with pytest.raises(ValueError, match="unsupported format.*re-export it with the current nirs4all"):
        open_n4a_source(path)
    assert calls == [path]
    other = tmp_path / "archive.bin"
    other.write_bytes(b"x")
    _fake_open(monkeypatch, ValueError("n4a unsupported format: only nirs4all.n4a.container.v1 / zip-stored.v1 reader 1 is supported"))
    assert open_n4a_source(other).kind == "none"


class _FakeNativePackage:
    def __init__(self, version: int) -> None:
        self._version = version

    def view(self) -> dict[str, Any]:
        return {"package_schema_version": self._version, "profile": "native-methods-package"}


def _native_manifest(spy: _WriteSpy, tmp_path: Path, *, global_feature: bool) -> dict[str, Any]:
    manifest = _manifest_for("canonical", spy, tmp_path)
    if global_feature:
        manifest["required_features"] = [hc.NATIVE_FEATURE]
    else:
        manifest["entrypoints"] = [{"id": "predict-external", "owner": "dag-ml", "required_features": [hc.NATIVE_FEATURE]}]
    return manifest


@pytest.mark.parametrize("global_feature", [True, False])
@pytest.mark.parametrize("version", [2, 3])
def test_a_native_package_container_is_classified_by_declared_feature_and_cores_retained_proof(
    monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path, version: int, global_feature: bool,
) -> None:
    handle = _FakeFile(_native_manifest(spy, tmp_path, global_feature=global_feature))
    path = tmp_path / "native.n4a"
    path.write_bytes(b"x")
    opened: list[object] = []

    def open_native_package(file: Any) -> Any:
        opened.append(file)
        return _FakeNativePackage(version)

    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(open_n4a_file=lambda candidate, *, memory_budget_bytes=None: handle, open_native_package=open_native_package))
    source = open_n4a_source(path)
    assert (source.kind, source.package_version) == ("native", version)
    assert opened == [handle] and source.native_input() is source.native_package
    source.close()
    assert handle.closed and source.native_package is None
    with pytest.raises(ValueError, match="not a native package container"):
        source.native_input()


def test_an_ambiguous_or_invalid_native_closure_is_refused_by_core_and_never_classified_canonical(
    monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path,
) -> None:
    handle = _FakeFile(_native_manifest(spy, tmp_path, global_feature=True))
    path = tmp_path / "ambiguous.n4a"
    path.write_bytes(b"x")

    def refuse(file: Any) -> Any:
        raise ValueError("n4a native package refusal: several entrypoints declare predict-external; name one")

    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(open_n4a_file=lambda candidate, *, memory_budget_bytes=None: handle, open_native_package=refuse))
    with pytest.raises(ValueError, match="several entrypoints"):
        open_n4a_source(path)
    assert handle.closed


def test_a_container_without_the_native_feature_never_asks_core_for_a_native_package(
    monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path,
) -> None:
    handle = _FakeFile(_manifest_for("canonical", spy, tmp_path))
    path = tmp_path / "canonical.n4a"
    path.write_bytes(b"x")
    monkeypatch.setattr(hc, "_core", lambda: SimpleNamespace(
        open_n4a_file=lambda candidate, *, memory_budget_bytes=None: handle,
        open_native_package=lambda file: pytest.fail("open_native_package ran on a canonical container"),
    ))
    source = open_n4a_source(path)
    assert (source.kind, source.package_version, source.native_package) == ("canonical", None, None)


def test_other_core_refusals_do_not_use_a_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = tmp_path / "damaged.n4a"
    archive.write_bytes(b"x")
    _fake_open(monkeypatch, ValueError("n4a physical refusal: central directory is damaged"))
    with pytest.raises(ValueError, match="physical refusal.*re-export"):
        open_n4a_source(archive)
    other = tmp_path / "damaged.bin"
    other.write_bytes(b"x")
    assert open_n4a_source(other).kind == "none"


@pytest.mark.parametrize("name", ["big.n4a", "big.bin"])
def test_a_memory_budget_refusal_is_never_hidden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    _fake_open(monkeypatch, ValueError("n4a memory refusal: budget exceeded"))
    with pytest.raises(ValueError) as caught:
        open_n4a_source(path, memory_budget_bytes=4096)
    assert str(caught.value) == "n4a memory refusal: budget exceeded"


def test_a_missing_core_api_is_a_dependency_error_not_a_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nirs4all.pipeline.dagml.core_archive_replay import CoreArchiveDependencyError

    path = tmp_path / "model.n4a"
    path.write_bytes(b"x")
    monkeypatch.setattr(hc.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(CoreArchiveDependencyError, match="open_n4a_file"):
        open_n4a_source(path)

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(hc.importlib, "import_module", missing)
    with pytest.raises(CoreArchiveDependencyError, match="native wheel"):
        write_host_container(tmp_path / "out.n4a", _members())


# --------------------------------------------------------------------------- reading (fake handle)


class _ReadFile(_FakeFile):
    def __init__(self, manifest: dict[str, Any], files: dict[str, bytes], *, fail: bool = False) -> None:
        super().__init__(manifest)
        self.files = files
        self.fail = fail
        self.members_read: list[str] = []

    def member(self, path: str, *, memory_budget_bytes: Any = None) -> bytes:
        self.members_read.append(path)
        return self.files[path]

    def copy_member_to(self, path: str, destination: Any) -> int:
        destination.write(self.files[path][:3])
        if self.fail:
            raise ValueError("n4a integrity refusal: member changed")
        destination.write(self.files[path][3:])
        return len(self.files[path])

    def copy_archive_to(self, destination: Any) -> int:
        destination.write(b"exact archive")
        if self.fail:
            raise ValueError("n4a integrity refusal: archive changed")
        return 13

    def view(self) -> dict[str, Any]:
        return {"readability": {"container_sha256": "ab" * 32}, "executability": "not_evaluated"}


def _reader(spy: _WriteSpy, tmp_path: Path, *, fail: bool = False) -> _ReadFile:
    manifest = _captured_manifest(spy, tmp_path)
    for entry, member in zip(manifest["members"], _members(), strict=True):
        entry["byte_length"] = len(member.source)  # type: ignore[arg-type]
        entry["sha256"] = hashlib.sha256(member.source).hexdigest()  # type: ignore[arg-type]
    return _ReadFile(manifest, {member.path: member.source for member in _members()}, fail=fail)


def test_inventory_reads_use_the_manifest_and_never_the_payload(spy: _WriteSpy, tmp_path: Path) -> None:
    handle = _reader(spy, tmp_path)
    container = HostContainer(tmp_path / "m.n4a", handle, handle.manifest, is_host=True)
    assert container.namelist() == [member.path for member in _members()]
    assert container.size("artifacts/sub10.joblib") == 3
    assert container.digest("artifacts/sub10.joblib") == "sha256:" + hashlib.sha256(b"ten").hexdigest()
    assert container.artifact("pipeline.json")["id"] == "pipeline.json"
    assert container.sha256 == "sha256:" + "ab" * 32
    assert handle.members_read == []
    for call in (container.size, container.digest, container.artifact, container.read, container.read_json):
        with pytest.raises(KeyError):
            call("absent")
    assert handle.members_read == []
    assert container.read("pipeline.json") == b'{"steps": []}'
    assert container.read_json("pipeline.json") == {"steps": []}
    assert container.bundle_manifest == {"source_type": "legacy"}
    assert container.bundle_manifest is container.bundle_manifest
    assert handle.members_read.count(BUNDLE_MANIFEST_MEMBER) == 1


def test_bundle_manifest_is_none_without_the_member(spy: _WriteSpy, tmp_path: Path) -> None:
    members = [HostMember("conformal/state.json", b"{}", "result", JSON_CONTENT_TYPE)]
    manifest = _captured_manifest(spy, tmp_path, members, entrypoint=False)
    container = HostContainer(tmp_path / "c.n4a", _ReadFile(manifest, {"conformal/state.json": b"{}"}), manifest, is_host=True)
    assert container.bundle_manifest is None


def test_extract_is_exclusive_and_discards_unverified_bytes(spy: _WriteSpy, tmp_path: Path) -> None:
    good = _reader(spy, tmp_path)
    container = HostContainer(tmp_path / "m.n4a", good, good.manifest, is_host=True)
    destination = tmp_path / "scratch" / "nested" / "model.joblib"
    container.extract("artifacts/sub2.joblib", destination)
    assert destination.read_bytes() == MODEL_PAYLOAD
    with pytest.raises(FileExistsError):
        container.extract("artifacts/sub2.joblib", destination)
    assert destination.read_bytes() == MODEL_PAYLOAD

    bad = _reader(spy, tmp_path, fail=True)
    broken = HostContainer(tmp_path / "m.n4a", bad, bad.manifest, is_host=True)
    unverified = tmp_path / "scratch" / "unverified.joblib"
    with pytest.raises(ValueError, match="changed"):
        broken.extract("artifacts/sub2.joblib", unverified)
    assert not unverified.exists()
    with pytest.raises(KeyError):
        broken.extract("absent", tmp_path / "scratch" / "absent")
    assert not (tmp_path / "scratch" / "absent").exists()


def test_copy_to_is_atomic_and_keeps_the_old_destination_on_failure(spy: _WriteSpy, tmp_path: Path) -> None:
    good = _reader(spy, tmp_path)
    target = tmp_path / "copy.n4a"
    target.write_bytes(b"old")
    HostContainer(tmp_path / "m.n4a", good, good.manifest, is_host=True).copy_to(target)
    assert target.read_bytes() == b"exact archive"
    bad = _reader(spy, tmp_path, fail=True)
    with pytest.raises(ValueError, match="archive changed"):
        HostContainer(tmp_path / "m.n4a", bad, bad.manifest, is_host=True).copy_to(target)
    assert target.read_bytes() == b"exact archive"
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith("copy") or path.name.startswith(".")) == ["copy.n4a"]


def test_close_is_idempotent_and_the_inventory_survives_it(spy: _WriteSpy, tmp_path: Path) -> None:
    handle = _reader(spy, tmp_path)
    container = HostContainer(tmp_path / "m.n4a", handle, handle.manifest, is_host=True)
    container.close()
    container.close()
    assert handle.closed
    assert container.namelist() and container.size("pipeline.json") == len(b'{"steps": []}')


def test_load_host_joblib_checks_type_and_owner_before_any_extraction_or_load(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("joblib.load ran"))
    handle = _reader(spy, tmp_path)
    container = HostContainer(tmp_path / "m.n4a", handle, handle.manifest, is_host=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="not a host joblib artifact"):
        load_host_joblib(container, "pipeline.json", scratch)
    with pytest.raises(ValueError, match="not a host joblib artifact"):
        load_host_joblib(container, BUNDLE_MANIFEST_MEMBER, scratch)
    with pytest.raises(KeyError):
        load_host_joblib(container, "absent", scratch)
    with pytest.raises(ValueError, match="scratch"):
        load_host_joblib(container, "artifacts/sub2.joblib", tmp_path / "no-such-dir")
    foreign = HostContainer(tmp_path / "f.n4a", handle, handle.manifest, is_host=False)
    with pytest.raises(ValueError, match="only from host containers"):
        load_host_joblib(foreign, "artifacts/sub2.joblib", scratch)
    assert list(scratch.iterdir()) == []


def test_load_host_joblib_never_deserializes_a_member_core_refused(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("joblib.load ran on unverified bytes"))
    handle = _reader(spy, tmp_path, fail=True)
    container = HostContainer(tmp_path / "m.n4a", handle, handle.manifest, is_host=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="changed"):
        load_host_joblib(container, "artifacts/sub2.joblib", scratch)
    assert list(scratch.iterdir()) == []


def test_load_host_joblib_loads_the_verified_private_file_then_removes_it(monkeypatch: pytest.MonkeyPatch, spy: _WriteSpy, tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    handle = _reader(spy, tmp_path)
    container = HostContainer(tmp_path / "m.n4a", handle, handle.manifest, is_host=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen: list[tuple[Path, bytes]] = []

    def load(path: Any, *args: Any, **kwargs: Any) -> str:
        assert not args and not kwargs
        seen.append((Path(path), Path(path).read_bytes()))
        return "model"

    monkeypatch.setattr(joblib, "load", load)
    assert load_host_joblib(container, "artifacts/sub2.joblib", scratch) == "model"
    assert len(seen) == 1 and seen[0][0].parent == scratch and seen[0][1] == MODEL_PAYLOAD
    assert list(scratch.iterdir()) == []


# --------------------------------------------------------------------------- real Core


def _real_write(tmp_path: Path, name: str = "model.n4a", members: list[HostMember] | None = None) -> Path:
    target = tmp_path / name
    write_host_container(target, members or _members())
    return target


def test_real_core_round_trip_keeps_declared_order_and_bytes(real_core: Any, tmp_path: Path) -> None:
    before = _listing(tmp_path)
    path = _real_write(tmp_path)
    raw = path.read_bytes()
    with open_n4a_source(path) as source:
        container = source.require_host()
        assert source.kind == "host"
        assert container.namelist() == [member.path for member in _members()]
        assert container.namelist() != sorted(container.namelist())
        for member in _members():
            assert container.read(member.path) == member.source
            assert container.size(member.path) == len(member.source)  # type: ignore[arg-type]
            assert container.digest(member.path) == "sha256:" + hashlib.sha256(member.source).hexdigest()  # type: ignore[arg-type]
        assert container.sha256 == "sha256:" + hashlib.sha256(raw).hexdigest()
        assert container.bundle_manifest == {"source_type": "legacy"}
        assert container.read_json("pipeline.json") == {"steps": []}
        assert container.view()["executability"] == "not_evaluated"
    after = _listing(tmp_path)
    assert after - before == {"model.n4a"} and before <= after, "only the container was added; no staging entry remains"


def test_real_core_streams_a_path_member_and_verifies_its_fingerprint_source(real_core: Any, tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096
    sidecar = tmp_path / "sidecar.bin"
    sidecar.write_bytes(payload)
    members = [*_members(), HostMember("host_artifacts/autogluon_0/part.bin", sidecar, "attachment", BLOB_CONTENT_TYPE)]
    path = _real_write(tmp_path, members=members)
    with open_n4a_source(path) as source:
        container = source.require_host()
        assert container.read("host_artifacts/autogluon_0/part.bin") == payload
        assert container.artifact("host_artifacts/autogluon_0/part.bin")["identity"]["fingerprint"] == hashlib.sha256(payload).hexdigest()


def test_real_core_update_preserves_order_and_overwrites_in_place(real_core: Any, tmp_path: Path) -> None:
    members = [*_members(), HostMember("conformal/old.json", b"{}", "result", JSON_CONTENT_TYPE)]
    before = _listing(tmp_path)
    path = _real_write(tmp_path, members=members)
    original = path.read_bytes()
    add = [HostMember("artifacts/sub2.joblib", b"two-v2", "model", JOBLIB_CONTENT_TYPE), HostMember("conformal/new.json", b"[]", "result", JSON_CONTENT_TYPE)]
    with open_n4a_source(path) as source:
        base = source.require_host()
        updated = tmp_path / "updated.n4a"
        update_host_container(base, updated, add=add, remove_prefix="conformal/")
        assert path.read_bytes() == original
        update_host_container(base, path, add=add, remove_prefix="conformal/")
    for candidate in (updated, path):
        with open_n4a_source(candidate) as source:
            container = source.require_host()
            assert container.namelist() == [BUNDLE_MANIFEST_MEMBER, "pipeline.json", "artifacts/sub2.joblib", "artifacts/sub10.joblib", "conformal/new.json"]
            assert container.read("artifacts/sub2.joblib") == b"two-v2"
            assert container.read("artifacts/sub10.joblib") == b"ten"
            assert container.read_json("conformal/new.json") == []
    after = _listing(tmp_path)
    assert after - before == {"model.n4a", "updated.n4a"} and before <= after, "only the two containers were added; no staging entry remains"


def test_real_core_conformal_only_container_and_exact_copy(real_core: Any, tmp_path: Path) -> None:
    path = tmp_path / "conformal.n4a"
    write_host_container(path, [HostMember("conformal/state.json", b'{"q": 1}', "result", JSON_CONTENT_TYPE)], entrypoint=False)
    with open_n4a_source(path) as source:
        container = source.require_host()
        assert container.bundle_manifest is None and container.manifest["entrypoints"] == []
        copied = tmp_path / "copy.n4a"
        container.copy_to(copied)
    assert copied.read_bytes() == path.read_bytes()


def test_real_core_refuses_a_member_changed_after_open_and_discards_it(real_core: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = _real_write(tmp_path)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("joblib.load ran on a changed member"))
    with open_n4a_source(path) as source:
        container = source.require_host()
        raw = path.read_bytes()
        offset = raw.index(MODEL_PAYLOAD)
        path.write_bytes(raw[:offset] + b"X" + raw[offset + 1 :])
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        with pytest.raises(ValueError):
            container.extract("artifacts/sub2.joblib", scratch / "model.joblib")
        assert not (scratch / "model.joblib").exists()
        with pytest.raises(ValueError):
            container.read("artifacts/sub2.joblib")
        with pytest.raises(ValueError):
            load_host_joblib(container, "artifacts/sub2.joblib", scratch)
        assert list(scratch.iterdir()) == []


def test_real_core_loads_a_joblib_member_after_full_extraction(real_core: Any, tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    buffer = io.BytesIO()
    joblib.dump({"coef": [1.0, 2.0]}, buffer)
    members = [*_members()[:2], HostMember("artifacts/model.joblib", buffer.getvalue(), "model", JOBLIB_CONTENT_TYPE)]
    path = _real_write(tmp_path, members=members)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with open_n4a_source(path) as source:
        assert load_host_joblib(source.require_host(), "artifacts/model.joblib", scratch) == {"coef": [1.0, 2.0]}
    assert list(scratch.iterdir()) == []


def _foreign_manifest(features: list[str]) -> str:
    return json.dumps(
        {
            "schema": "nirs4all.n4a.container.v1",
            "schema_version": 1,
            "minimum_reader_version": 1,
            "container_id": "container:foreign.test",
            "required_features": features,
            "physical_profile": "zip-stored.v1",
            "members": [{"path": "data/x.bin", "byte_length": 0, "sha256": "0" * 64}],
            "artifacts": [
                {
                    "id": "x",
                    "kind": "attachment",
                    "owner": "host.example",
                    "content_type": {"id": "host.example.blob", "version": 1},
                    "identity": {"owner": "host.example", "schema": "host.example.blob", "schema_version": 1, "fingerprint_profile": "sha256.raw.v1", "fingerprint": "a" * 64},
                    "storage": {"kind": "embedded", "root_member": "data/x.bin", "member_paths": ["data/x.bin"]},
                    "requirements": None,
                }
            ],
            "links": [],
            "entrypoints": [],
        }
    )


def test_real_core_other_v1_owners_are_canonical_and_mixed_host_is_refused(real_core: Any, tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.bin"
    real_core.write_n4a_file(foreign, _foreign_manifest([]), [("data/x.bin", b"x")])
    with open_n4a_source(foreign) as source:
        assert source.kind == "canonical" and source.container is not None and not source.container.is_host
        assert source.container.read("data/x.bin") == b"x"
    mixed = tmp_path / "mixed.n4a"
    real_core.write_n4a_file(mixed, _foreign_manifest([HOST_FEATURE]), [("data/x.bin", b"x")])
    with pytest.raises(ValueError):
        open_n4a_source(mixed)


def test_borrowed_core_inspection_source_one_open_and_lifetime(tmp_path, monkeypatch):
    import nirs4all_core as core  # required integration dependency; never skip

    path = _real_write(tmp_path)
    original = core.open_n4a_file
    seen = []
    def counted(*args, **kwargs):
        seen.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "open_n4a_file", counted)
    with core.open_n4a_file(path) as outer:
        with open_n4a_source(path, core_file=outer) as borrowed:
            assert borrowed.owns_core_file is False
            assert borrowed.require_host().read("pipeline.json") == b'{"steps": []}'
            assert borrowed.container.core_file is outer
        assert borrowed.container is None
        assert borrowed.native_package is None
        assert not outer.closed
        borrowed.close()
        assert not outer.closed
    assert len(seen) == 1
    assert outer.closed
    owned = open_n4a_source(path)
    container = owned.container
    manifest = container.manifest
    file = container.core_file
    owned.close()
    assert file.closed
    assert owned.container is container
    assert owned.container.manifest is manifest
    assert owned.container.path == path
    owned.close()
    assert owned.container is container


def test_borrowed_core_inspection_refusal_never_closes_outer_file(tmp_path):
    import nirs4all_core as core  # required, never skip

    path = tmp_path / "mixed.n4a"
    core.write_n4a_file(path, _foreign_manifest([HOST_FEATURE]), [("data/x.bin", b"x")])
    with core.open_n4a_file(path) as outer:
        with pytest.raises(ValueError):
            open_n4a_source(path, core_file=outer)
        assert not outer.closed
        assert outer.member("data/x.bin") == b"x"


def test_borrowed_core_source_path_is_context_not_reopen(tmp_path):
    import nirs4all_core as core  # required, never skip

    path = _real_write(tmp_path)
    with core.open_n4a_file(path) as outer:
        # A different context path is not opened: evidence binds the retained file.
        context = tmp_path / "not-present.n4a"
        with open_n4a_source(context, core_file=outer) as source:
            assert source.kind == "host"
            assert source.require_host().sha256 == "sha256:" + outer.view()["readability"]["container_sha256"]
        assert not outer.closed
