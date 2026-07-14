"""Unit locks for internal conformal filesystem stores."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nirs4all.pipeline.dagml.conformal_contracts import calibrate_replayed_predictions, parse_conformal_calibration_spec
from nirs4all.pipeline.dagml.conformal_store import (
    ARTIFACT_FILENAME,
    BUNDLE_ROOT,
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    ConformalStoreManifest,
    export_conformal_result_bundle,
    load_conformal_result_archive,
    load_conformal_result_bundle,
    load_conformal_result_store,
    save_conformal_result_store,
)


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


def test_conformal_result_bundle_round_trips_as_n4a_archive(tmp_path) -> None:
    result = _calibrated_result()
    target = tmp_path / "calibrated-result.n4a"

    saved = export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert saved == target
    assert restored.to_dict() == result.to_dict()


def test_grouped_conformal_result_bundle_round_trips_vector_qhat(tmp_path) -> None:
    result = _grouped_calibrated_result()
    target = tmp_path / "grouped-calibrated-result.n4a"

    export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert restored.to_dict() == result.to_dict()
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])


def test_joint_max_conformal_result_bundle_round_trips_2d_intervals(tmp_path) -> None:
    result = _joint_max_calibrated_result()
    target = tmp_path / "joint-max-calibrated-result.n4a"

    export_conformal_result_bundle(result, target)
    restored = load_conformal_result_bundle(target)

    assert restored.to_dict() == result.to_dict()
    np.testing.assert_allclose(restored.prediction.interval(0.8).upper, [[10.6, 100.6], [20.6, 200.6]])


def test_grouped_full_archive_loader_round_trips_vector_qhat(tmp_path) -> None:
    import zipfile

    result = _grouped_calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    target = tmp_path / "model-with-grouped-sidecar.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version":"1.0"}')
        archive.writestr("pipeline.json", "{}")
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")

    restored = load_conformal_result_archive(target)

    assert restored.to_dict() == result.to_dict()
    assert restored.conformal_guarantee_status is not None
    assert restored.conformal_guarantee_status["guarantee"] == "split_conformal_group_marginal_coverage"
    np.testing.assert_allclose(restored.prediction.interval(0.5).qhat, [0.2, 0.6, 0.2])


def test_conformal_result_bundle_rejects_missing_prediction_sample_ids(tmp_path) -> None:
    import zipfile

    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    result_payload = json.loads((store / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["sample_ids"] = []
    result_payload.pop("fingerprint")
    (store / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target = tmp_path / "missing-sample-ids.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")

    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_bundle(target)


def test_conformal_result_bundle_rejects_unexpected_members(tmp_path) -> None:
    import zipfile

    target = tmp_path / "bad.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(f"{BUNDLE_ROOT}{MANIFEST_FILENAME}", "{}")
        archive.writestr("unexpected.txt", "bad")

    with pytest.raises(ValueError, match="members"):
        load_conformal_result_bundle(target)


def test_conformal_result_bundle_rejects_duplicate_sidecar_members(tmp_path) -> None:
    import zipfile

    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    target = tmp_path / "duplicate.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")
        archive.write(store / ARTIFACT_FILENAME, arcname=f"{BUNDLE_ROOT}{ARTIFACT_FILENAME}")

    with pytest.raises(ValueError, match="duplicate members"):
        load_conformal_result_bundle(target)


def test_full_archive_loader_rejects_duplicate_conformal_sidecar_members(tmp_path) -> None:
    import zipfile

    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    target = tmp_path / "model-with-duplicate-sidecar.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version":"1.0"}')
        archive.writestr("pipeline.json", "{}")
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")
        archive.write(store / RESULT_FILENAME, arcname=f"{BUNDLE_ROOT}{RESULT_FILENAME}")

    with pytest.raises(ValueError, match="duplicate members"):
        load_conformal_result_archive(target)


def test_full_archive_loader_rejects_missing_prediction_sample_ids(tmp_path) -> None:
    import zipfile

    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    result_payload = json.loads((store / RESULT_FILENAME).read_text(encoding="utf-8"))
    result_payload["sample_ids"] = []
    result_payload.pop("fingerprint")
    (store / RESULT_FILENAME).write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target = tmp_path / "model-with-missing-sample-ids.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version":"1.0"}')
        archive.writestr("pipeline.json", "{}")
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")

    with pytest.raises(ValueError, match="sample_ids are required"):
        load_conformal_result_archive(target)


def test_full_archive_loader_rejects_unexpected_conformal_sidecar_members(tmp_path) -> None:
    import zipfile

    result = _calibrated_result()
    store = tmp_path / "store"
    save_conformal_result_store(result, store)
    target = tmp_path / "model-with-extra-sidecar-member.n4a"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", '{"bundle_format_version":"1.0"}')
        archive.writestr("pipeline.json", "{}")
        for filename in (MANIFEST_FILENAME, ARTIFACT_FILENAME, RESULT_FILENAME):
            archive.write(store / filename, arcname=f"{BUNDLE_ROOT}{filename}")
        archive.writestr(f"{BUNDLE_ROOT}extra.json", "{}")

    with pytest.raises(ValueError, match="unexpected conformal sidecar members"):
        load_conformal_result_archive(target)
