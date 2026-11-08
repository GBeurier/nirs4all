"""Unit locks for internal conformal filesystem stores."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nirs4all.pipeline.dagml import conformal_store, host_container
from nirs4all.pipeline.dagml.conformal_contracts import calibrate_replayed_predictions, parse_conformal_calibration_spec
from nirs4all.pipeline.dagml.conformal_store import (
    ARTIFACT_FILENAME,
    BUNDLE_ROOT,
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    ConformalStoreManifest,
    attach_conformal_result_to_bundle,
    export_conformal_result_bundle,
    load_conformal_result_archive,
    load_conformal_result_bundle,
    load_conformal_result_store,
    save_conformal_result_store,
)
from nirs4all.pipeline.dagml.host_container import BUNDLE_MANIFEST_MEMBER, JSON_CONTENT_TYPE, HostMember, open_n4a_source, write_host_container
from tests.unit.pipeline.dagml.host_container_helpers import blob, bundle_members, require_core, reseal_host_member, write_host_bundle


@pytest.fixture
def real_core():
    return require_core()


def _staging_leftovers(directory):
    return [path.name for path in directory.iterdir() if path.name.startswith(".") or path.suffix == ".tmp"]


def _sidecar(store, *, only=None):
    """The conformal sidecar members streamed from a saved store directory, in declared order."""
    return [
        HostMember(f"{BUNDLE_ROOT}{filename}", store / filename, "result", JSON_CONTENT_TYPE)
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME)
        if only is None or filename in only
    ]


def test_rejected_attachment_preserves_destination(tmp_path, real_core):
    source = tmp_path / "already-calibrated.n4a"
    target = tmp_path / "output.n4a"
    result = _calibrated_result()
    export_conformal_result_bundle(result, source)
    with pytest.raises(ValueError, match="already contains"):
        attach_conformal_result_to_bundle(source, result, target)
    assert not target.exists()
    assert _staging_leftovers(tmp_path) == []


def test_failed_attachment_and_export_preserve_existing_bundle(tmp_path, monkeypatch, real_core):
    source = write_host_bundle(tmp_path / "model.n4a")
    target = tmp_path / "output.n4a"
    result = _calibrated_result()
    export_conformal_result_bundle(result, target)
    before = target.read_bytes()

    def failing_write(destination, *args, **kwargs):
        Path(destination).write_bytes(b"partial staged container")
        raise OSError("disk write failed")

    failing = SimpleNamespace(open_n4a_file=real_core.open_n4a_file, write_n4a_file=failing_write)
    monkeypatch.setattr(host_container, "_core", lambda: failing)
    for operation in (lambda: attach_conformal_result_to_bundle(source, result, target, overwrite=True),
                      lambda: export_conformal_result_bundle(result, target, overwrite=True)):
        with pytest.raises(OSError, match="disk write failed"):
            operation()
        assert target.read_bytes() == before
        assert _staging_leftovers(tmp_path) == []


def test_attachment_keeps_the_model_container_and_appends_a_verified_sidecar(tmp_path, real_core):
    result = _calibrated_result()
    model = write_host_bundle(tmp_path / "model.n4a", [blob("host_artifacts/autogluon_0/part.bin", b"sidecar bytes")])
    before = model.read_bytes()

    target = attach_conformal_result_to_bundle(model, result)

    assert target == tmp_path / "model.calibrated.n4a"
    assert model.read_bytes() == before, "the source container is never modified"
    with open_n4a_source(model) as base, open_n4a_source(target) as calibrated:
        retained = base.require_host().namelist()
        names = calibrated.require_host().namelist()
        assert names == [*retained, *(f"{BUNDLE_ROOT}{name}" for name in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME))]
        for name in retained:
            assert calibrated.require_host().read(name) == base.require_host().read(name)
        assert calibrated.require_host().manifest["entrypoints"] == base.require_host().manifest["entrypoints"]
    assert load_conformal_result_archive(target).to_dict() == result.to_dict()


def test_attachment_overwrite_replaces_the_sidecar_in_place_and_drops_stale_members(tmp_path, real_core):
    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    stale = HostMember(f"{BUNDLE_ROOT}stale.json", b"{}", "result", JSON_CONTENT_TYPE)
    model = write_host_bundle(tmp_path / "model.n4a", [*_sidecar(store), stale, blob("tail.bin", b"tail")])

    target = attach_conformal_result_to_bundle(model, result, tmp_path / "out.n4a", overwrite=True)

    with open_n4a_source(target) as calibrated:
        assert calibrated.require_host().namelist() == [
            BUNDLE_MANIFEST_MEMBER, "pipeline.json", *(member.path for member in _sidecar(store)), "tail.bin",
        ]
    assert load_conformal_result_archive(target).to_dict() == result.to_dict()


def test_attachment_refuses_non_host_inputs_and_in_place_output(tmp_path, real_core):
    result = _calibrated_result()
    model = write_host_bundle(tmp_path / "model.n4a")
    with pytest.raises(ValueError, match="in-place"):
        attach_conformal_result_to_bundle(model, result, model)
    with pytest.raises(FileNotFoundError):
        attach_conformal_result_to_bundle(tmp_path / "missing.n4a", result)
    import zipfile

    old = tmp_path / "old.n4a"
    with zipfile.ZipFile(old, "w") as archive:
        archive.writestr("model.txt", "model")
    with pytest.raises(ValueError):
        attach_conformal_result_to_bundle(old, result, tmp_path / "from-old.n4a")
    assert not (tmp_path / "from-old.n4a").exists()
    assert _staging_leftovers(tmp_path) == []


def test_loaders_borrow_a_supplied_source_and_a_standalone_call_closes_only_its_own(tmp_path, monkeypatch, real_core):
    result = _calibrated_result()
    model = attach_conformal_result_to_bundle(write_host_bundle(tmp_path / "model.n4a"), result)
    only = export_conformal_result_bundle(result, tmp_path / "only.n4a")

    with open_n4a_source(model) as source, open_n4a_source(only) as only_source:
        with monkeypatch.context() as patch:
            patch.setattr(host_container, "_core", lambda: pytest.fail("a supplied source was reopened"))
            assert load_conformal_result_archive(model, source=source).to_dict() == result.to_dict()
            assert load_conformal_result_bundle(only, source=only_source).to_dict() == result.to_dict()
        assert not source.container.core_file.closed and not only_source.container.core_file.closed
        with pytest.raises(ValueError, match="supplied source"):
            load_conformal_result_archive(only, source=source)

    opened = []

    def recording_open(path, **kwargs):
        handle = real_core.open_n4a_file(path, **kwargs)
        opened.append(handle)
        return handle

    recording = SimpleNamespace(open_n4a_file=recording_open, write_n4a_file=real_core.write_n4a_file)
    monkeypatch.setattr(host_container, "_core", lambda: recording)
    assert load_conformal_result_archive(model).to_dict() == result.to_dict()
    assert len(opened) == 1 and opened[0].closed


def test_attachment_with_a_supplied_source_reads_the_base_through_it(tmp_path, monkeypatch, real_core):
    result = _calibrated_result()
    model = write_host_bundle(tmp_path / "model.n4a")
    with open_n4a_source(model) as source:
        with monkeypatch.context() as patch:
            patch.setattr(conformal_store, "open_n4a_source", lambda *args, **kwargs: pytest.fail("a supplied source was reopened"))
            target = attach_conformal_result_to_bundle(model, result, tmp_path / "out.n4a", source=source)
        assert not source.container.core_file.closed
    assert load_conformal_result_archive(target).to_dict() == result.to_dict()
    other = write_host_bundle(tmp_path / "other.n4a")
    with open_n4a_source(model) as source, pytest.raises(ValueError, match="supplied source"):
        attach_conformal_result_to_bundle(other, result, tmp_path / "mismatch.n4a", source=source)


def test_raw_changes_are_refused_by_core_and_a_resealed_sidecar_reaches_the_semantic_check(tmp_path, real_core):
    result = _calibrated_result()
    exported = export_conformal_result_bundle(result, tmp_path / "result.n4a")

    flipped = tmp_path / "flipped.n4a"
    raw = bytearray(exported.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    flipped.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        load_conformal_result_bundle(flipped)

    payload = json.loads(_read_member(exported, f"{BUNDLE_ROOT}{RESULT_FILENAME}"))
    payload["sample_ids"] = []
    payload.pop("fingerprint")
    resealed = reseal_host_member(exported, tmp_path / "resealed.n4a", f"{BUNDLE_ROOT}{RESULT_FILENAME}", json.dumps(payload, indent=2, sort_keys=True).encode())
    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_bundle(resealed)


def _read_member(path, member):
    with open_n4a_source(path) as source:
        return source.require_host().read(member)


def _calibrated_result():
    spec = parse_conformal_calibration_spec({"coverage": [0.5, 0.8]})
    return calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 2.2, 2.6, 4.3],
        y_pred=[10.0, 20.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        calibration_groups=["a", "a", "b", "b"],
        result_metadata={"phase": "prediction"},
        target_name="moisture",
        predictor_fingerprint="predictor-abc",
    )


def _grouped_calibrated_result():
    spec = parse_conformal_calibration_spec({"coverage": 0.5, "group_by": "group"})
    return calibrate_replayed_predictions(
        y_true_calibration=[1.0, 2.0, 3.0, 4.0],
        y_pred_calibration=[0.9, 1.8, 2.5, 3.4],
        y_pred=[10.0, 20.0, 30.0],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2", "p3"],
        calibration_groups=["instrument-a", "instrument-a", "instrument-b", "instrument-b"],
        prediction_groups=["instrument-a", "instrument-b", "instrument-a"],
        result_metadata={"phase": "grouped-prediction"},
        target_name="moisture",
        predictor_fingerprint="predictor-abc",
    )


def _joint_max_calibrated_result():
    spec = parse_conformal_calibration_spec({"coverage": 0.8, "multi_target": "joint_max"})
    return calibrate_replayed_predictions(
        y_true_calibration=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        y_pred_calibration=[[0.9, 9.8], [2.3, 20.1], [2.4, 30.2], [4.1, 39.6]],
        y_pred=[[10.0, 100.0], [20.0, 200.0]],
        spec=spec,
        calibration_sample_ids=["c1", "c2", "c3", "c4"],
        prediction_sample_ids=["p1", "p2"],
        result_metadata={"phase": "joint-max-prediction"},
        target_name="moisture-fat",
        predictor_fingerprint="predictor-abc",
    )


def test_conformal_result_store_round_trips_with_manifest(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"

    saved = save_conformal_result_store(result, target)
    restored = load_conformal_result_store(target)
    manifest = ConformalStoreManifest.load_json(target / MANIFEST_FILENAME)

    assert saved == target
    assert (target / ARTIFACT_FILENAME).is_file()
    assert (target / RESULT_FILENAME).is_file()
    assert manifest.result_fingerprint == result.fingerprint
    assert manifest.artifact_fingerprint == result.artifact.fingerprint
    assert restored.to_dict() == result.to_dict()


def test_grouped_conformal_result_store_round_trips_vector_qhat(tmp_path) -> None:
    result = _grouped_calibrated_result()
    target = tmp_path / "grouped-calibrated-store"

    save_conformal_result_store(result, target)
    restored = load_conformal_result_store(target)

    assert restored.to_dict() == result.to_dict()
    assert restored.prediction.group_keys == result.prediction.group_keys
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])
    assert sorted(restored.artifact.group_calibrators) == ['["instrument-a"]', '["instrument-b"]']


def test_joint_max_conformal_result_store_round_trips_2d_intervals(tmp_path) -> None:
    result = _joint_max_calibrated_result()
    target = tmp_path / "joint-max-calibrated-store"

    save_conformal_result_store(result, target)
    restored = load_conformal_result_store(target)

    assert restored.to_dict() == result.to_dict()
    assert restored.prediction.y_pred.shape == (2, 2)
    assert restored.conformal_guarantee_status is not None
    assert restored.conformal_guarantee_status["guarantee"] == "split_conformal_joint_max_simultaneous_coverage"
    np.testing.assert_allclose(restored.prediction.interval(0.8).lower, [[9.4, 99.4], [19.4, 199.4]])


def test_grouped_conformal_result_store_rejects_edited_group_qhat(tmp_path) -> None:
    result = _grouped_calibrated_result()
    target = tmp_path / "grouped-calibrated-store"
    save_conformal_result_store(result, target)
    payload = json.loads((target / RESULT_FILENAME).read_text(encoding="utf-8"))
    payload["prediction"]["intervals"][0]["qhat"] = [0.2, 0.2, 0.2]
    payload.pop("fingerprint")
    (target / RESULT_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="interval qhat must match"):
        load_conformal_result_store(target)


def test_conformal_result_store_is_deterministic_with_overwrite(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"

    save_conformal_result_store(result, target)
    first_manifest = (target / MANIFEST_FILENAME).read_text(encoding="utf-8")
    save_conformal_result_store(result, target, overwrite=True)
    second_manifest = (target / MANIFEST_FILENAME).read_text(encoding="utf-8")

    assert first_manifest == second_manifest


def test_conformal_result_store_refuses_non_empty_target_without_overwrite(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    target.mkdir()
    (target / "foreign.txt").write_text("do not clobber", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        save_conformal_result_store(result, target)


def test_conformal_result_store_rejects_manifest_tampering(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    save_conformal_result_store(result, target)
    manifest_path = target / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["result_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_conformal_result_store(target)


def test_conformal_result_store_rejects_result_artifact_mismatch(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    save_conformal_result_store(result, target)
    result_payload = json.loads((target / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["artifact"]["predictor_fingerprint"] = "other-predictor"
    result_payload["artifact"].pop("fingerprint")
    result_payload.pop("fingerprint")
    (target / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_payload = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    # Keep the manifest untouched: loading must notice that the result no longer
    # matches the manifest and the standalone artifact.
    (target / MANIFEST_FILENAME).write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="result fingerprint mismatch|result artifact|artifact_fingerprint"):
        load_conformal_result_store(target)


def test_conformal_result_store_rejects_missing_prediction_sample_ids(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    save_conformal_result_store(result, target)
    result_payload = json.loads((target / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["sample_ids"] = []
    result_payload.pop("fingerprint")
    (target / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_store(target)


def test_conformal_result_store_rejects_non_strict_json_result_metadata(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    save_conformal_result_store(result, target)
    result_payload = json.loads((target / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["metadata"]["bad"] = float("nan")
    result_payload.pop("fingerprint")
    (target / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"CalibratedRunResult.metadata\[bad\].*JSON-compatible"):
        load_conformal_result_store(target)


def test_conformal_result_store_rejects_manifest_file_escape_even_with_valid_manifest_fingerprint(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-store"
    save_conformal_result_store(result, target)
    external = tmp_path / "external-artifact.json"
    external.write_text((target / ARTIFACT_FILENAME).read_text(encoding="utf-8"), encoding="utf-8")
    manifest = ConformalStoreManifest(
        result_fingerprint=result.fingerprint,
        artifact_fingerprint=result.artifact.fingerprint,
        files={"artifact": "../external-artifact.json", "result": RESULT_FILENAME},
    )
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes store root"):
        load_conformal_result_store(target)


def test_conformal_result_bundle_round_trips_as_n4a_archive(tmp_path, real_core) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-result.n4a"

    saved = export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert saved == target
    assert restored.to_dict() == result.to_dict()
    with open_n4a_source(target) as source:
        container = source.require_host()
        assert container.namelist() == [f"{BUNDLE_ROOT}{name}" for name in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME)]
        assert container.manifest["entrypoints"] == [] and container.bundle_manifest is None
        assert {container.artifact(name)["kind"] for name in container.namelist()} == {"result"}
    with pytest.raises(FileExistsError):
        export_conformal_result_bundle(result, target)
    export_conformal_result_bundle(result, target, overwrite=True)


def test_grouped_conformal_result_bundle_round_trips_vector_qhat(tmp_path, real_core) -> None:
    result = _grouped_calibrated_result()
    target = tmp_path / "grouped-calibrated-result.n4a"

    export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert restored.to_dict() == result.to_dict()
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])


def test_joint_max_conformal_result_bundle_round_trips_2d_intervals(tmp_path, real_core) -> None:
    result = _joint_max_calibrated_result()
    target = tmp_path / "joint-max-calibrated-result.n4a"

    export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert restored.to_dict() == result.to_dict()
    np.testing.assert_allclose(restored.prediction.interval(0.8).upper, [[10.6, 100.6], [20.6, 200.6]])


def _tampered_store(tmp_path):
    """A saved store whose result payload has no prediction sample ids (semantic, not integrity, damage)."""
    store = tmp_path / "store"
    save_conformal_result_store(_calibrated_result(), store)
    result_payload = json.loads((store / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["sample_ids"] = []
    result_payload.pop("fingerprint")
    (store / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return store


def test_grouped_full_archive_loader_round_trips_vector_qhat(tmp_path, real_core) -> None:
    result = _grouped_calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    target = write_host_bundle(tmp_path / "model-with-grouped-sidecar.n4a", _sidecar(store))

    restored = load_conformal_result_archive(target)

    assert restored.to_dict() == result.to_dict()
    assert restored.conformal_guarantee_status is not None
    assert restored.conformal_guarantee_status["guarantee"] == "split_conformal_group_marginal_coverage"
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])


def test_conformal_result_bundle_rejects_missing_prediction_sample_ids(tmp_path, real_core) -> None:
    target = tmp_path / "missing-sample-ids.n4a"
    write_host_container(target, _sidecar(_tampered_store(tmp_path)), entrypoint=False)

    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_bundle(target)


def test_conformal_result_bundle_rejects_unexpected_members(tmp_path, real_core) -> None:
    target = tmp_path / "bad.n4a"
    write_host_container(target, [HostMember(f"{BUNDLE_ROOT}{MANIFEST_FILENAME}", b"{}", "result", JSON_CONTENT_TYPE), blob("unexpected.txt", b"bad")], entrypoint=False)

    with pytest.raises(ValueError, match="members"):
        load_conformal_result_bundle(target)
    with pytest.raises(ValueError, match="missing"):
        load_conformal_result_archive(target)


def test_conformal_bundle_loader_rejects_a_model_container_and_loaders_refuse_a_missing_path(tmp_path, real_core) -> None:
    result = _calibrated_result()
    model = attach_conformal_result_to_bundle(write_host_bundle(tmp_path / "model.n4a"), result)
    with pytest.raises(ValueError, match="members must be exactly"):
        load_conformal_result_bundle(model)
    with pytest.raises(FileNotFoundError):
        load_conformal_result_archive(tmp_path / "missing.n4a")


def test_duplicate_conformal_sidecar_members_cannot_be_written(tmp_path, real_core) -> None:
    store = tmp_path / "store"
    save_conformal_result_store(_calibrated_result(), store)
    target = tmp_path / "duplicate.n4a"
    members = [*_sidecar(store), *_sidecar(store, only={ARTIFACT_FILENAME})]

    with pytest.raises(ValueError, match="duplicate"):
        write_host_container(target, members, entrypoint=False)
    with pytest.raises(ValueError, match="duplicate"):
        write_host_container(target, [*bundle_members(), *members])
    assert not target.exists()


def test_full_archive_loader_rejects_missing_prediction_sample_ids(tmp_path, real_core) -> None:
    target = write_host_bundle(tmp_path / "model-with-missing-sample-ids.n4a", _sidecar(_tampered_store(tmp_path)))

    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_archive(target)


def test_full_archive_loader_rejects_unexpected_conformal_sidecar_members(tmp_path, real_core) -> None:
    store = tmp_path / "store"
    save_conformal_result_store(_calibrated_result(), store)
    target = write_host_bundle(tmp_path / "model-with-extra-sidecar-member.n4a", [*_sidecar(store), HostMember(f"{BUNDLE_ROOT}extra.json", b"{}", "result", JSON_CONTENT_TYPE)])

    with pytest.raises(ValueError, match="unexpected conformal sidecar members"):
        load_conformal_result_archive(target)
