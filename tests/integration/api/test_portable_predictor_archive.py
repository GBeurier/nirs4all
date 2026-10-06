"""Public portable archive transport, trust preflight and real N4ME replay."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import nirs4all
from nirs4all.api import portable_archive

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "portable_role_pipeline" / "five_model_capture.json"


def test_public_exports_do_not_require_native_archive_dependencies() -> None:
    for name in portable_archive.__all__:
        assert getattr(nirs4all, name) is getattr(portable_archive, name)


def test_reader_requires_native_semantic_validation_before_returning_package(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    package_bytes = b'{"schema_version":2}'
    members = {"dagml/portable_predictor_package.json": package_bytes}
    manifest = {"schema_version": 2}

    class Package:
        def __init__(self, text: str) -> None:
            events.append("package")
            assert text == package_bytes.decode()

    def read(path: str) -> dict[str, Any]:
        events.append("core")
        assert path == "portable.n4a"
        return {"manifest": manifest, "members": members}

    def validate(actual_manifest: Any, package: Any, actual_members: Any) -> None:
        events.append("dag")
        assert actual_manifest is manifest and actual_members is members
        assert isinstance(package, Package)
        raise ValueError("incomplete RAW closure")

    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(read_archive_v2_payloads=read))
    monkeypatch.setitem(sys.modules, "dag_ml", SimpleNamespace(PortablePredictorPackage=Package, validate_archive_v2_portable_payloads=validate))
    with pytest.raises(ValueError, match="incomplete RAW closure"):
        nirs4all.read_portable_predictor_archive_v2("portable.n4a")
    assert events == ["core", "package", "dag"]


@pytest.mark.parametrize("payloads", [None, {}, {"manifest": {}, "members": []}, {"manifest": {}, "members": {}}])
def test_reader_rejects_invalid_native_transport(monkeypatch: pytest.MonkeyPatch, payloads: Any) -> None:
    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(read_archive_v2_payloads=lambda _: payloads))
    monkeypatch.setitem(sys.modules, "dag_ml", SimpleNamespace(validate_archive_v2_portable_payloads=lambda *_: None))
    with pytest.raises(ValueError, match="Core Archive V2 reader"):
        nirs4all.read_portable_predictor_archive_v2("portable.n4a")


def test_missing_semantic_validator_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(read_archive_v2_payloads=lambda _: pytest.fail("must not read")))
    monkeypatch.setitem(sys.modules, "dag_ml", SimpleNamespace())
    with pytest.raises(ImportError, match="matching Core and DAG-ML"):
        nirs4all.read_portable_predictor_archive_v2("portable.n4a")


def test_explicit_trust_cannot_be_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portable_archive, "read_portable_predictor_archive_v2", lambda _: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="explicit trusted"):
        nirs4all.replay_portable_predictor_archive_v2(
            "portable.n4a", {}, {}, None, lambda _: {}, outcome_id="outcome:test", run_id="run:test", artifact_callback=lambda _: None,
        )


def _live_runtime() -> tuple[Any, Any, Any]:
    required = os.environ.get("NIRS4ALL_REQUIRE_PORTABLE_ARCHIVE_V2") == "1"
    try:
        dag_ml = importlib.import_module("dag_ml")
        core = importlib.import_module("nirs4all_core")
        role_pipeline = importlib.import_module("n4m.roles").RolePipeline
        if not callable(getattr(core, "read_archive_v2_payloads", None)) or not callable(getattr(dag_ml, "validate_archive_v2_portable_payloads", None)):
            raise ImportError("matching native portable archive facades are unavailable")
    except ImportError as error:
        if required:
            pytest.fail(str(error))
        pytest.skip(str(error))
    return dag_ml, core, role_pipeline


@pytest.fixture
def capture() -> dict[str, Any]:
    source = Path(os.environ.get("NIRS4ALL_ROLE_PIPELINE_CAPTURE_JSON", str(_FIXTURE)))
    data = json.loads(source.read_text())
    if "completeArchive" in data:
        archive = data["completeArchive"]
        return {
            "training_request_json": archive["requestJson"],
            "package_json": archive["packageJson"],
            "outcome_json": archive["outcomeJson"],
            "trusted_manifests": [data["manifest"]],
            "data_envelopes": archive["predictEnvelopes"],
            "replay_request_json": archive["replayRequestJson"],
            "heldout_rows": archive["heldoutRows"],
            "expected_prediction": archive["replay"]["outputs"][0]["predictions"][0],
        }
    return data


@pytest.mark.parametrize("producer", ["model:image", "model:metadata", "model:nir", "model:series", "model:meta"])
@pytest.mark.parametrize("fold_id", ["avg", "w_avg"])
def test_capture_oof_reports_match_independent_metrics(capture: dict[str, Any], producer: str, fold_id: str) -> None:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

    outcome = json.loads(capture["outcome_json"])
    [average] = [entry for entry in outcome["oof_averages"] if entry["predictions"]["producer_node"] == producer and entry["predictions"]["fold_id"] == fold_id]
    predictions = average["predictions"]
    truth = average["y_true"]
    assert truth["unit_ids"] == predictions["unit_ids"]
    y_true, y_pred = np.asarray(truth["values"]).ravel(), np.asarray(predictions["values"]).ravel()
    # Rust's round uses half-away-from-zero for the integer class identities.
    true_labels = np.sign(y_true) * np.floor(np.abs(y_true) + 0.5)
    predicted_labels = np.sign(y_pred) * np.floor(np.abs(y_pred) + 0.5)
    reference = {
        "mse": mean_squared_error(y_true, y_pred), "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "f1": f1_score(true_labels, predicted_labels, average="weighted"),
    }
    for score_set in (outcome["score_set"], outcome["execution_bundle"]["scores"]):
        [report] = [entry for entry in score_set["reports"] if entry["producer_node"] == producer and entry["fold_id"] == fold_id]
        assert report["variant_id"] == outcome["selected_variant_id"]
        assert report["row_count"] == len(y_true)
        for name, value in reference.items():
            assert report["metrics"][name] == pytest.approx(value, rel=1e-12, abs=1e-12)
            assert report["metrics"][f"{name}:y"] == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("resign", [False, True], ids=["integrity", "semantic"])
def test_native_outcome_rejects_inconsistent_oof_accuracy(capture: dict[str, Any], resign: bool) -> None:
    from nirs4all.pipeline.dagml.training_contracts import tcv1_fingerprint_without

    dag_ml, _core, _role_pipeline = _live_runtime()
    outcome = json.loads(capture["outcome_json"])
    dag_ml.TrainingOutcome(outcome)
    assert tcv1_fingerprint_without(outcome, "outcome_fingerprint") == outcome["outcome_fingerprint"]
    [report] = [entry for entry in outcome["score_set"]["reports"] if entry["producer_node"] == "model:nir" and entry["fold_id"] == "avg"]
    report["metrics"]["accuracy"] += 0.05
    if resign:
        outcome["outcome_fingerprint"] = tcv1_fingerprint_without(outcome, "outcome_fingerprint")
    message = "OOF average values disagree with selected score report" if resign else "fingerprint does not match original TCV1 JSON"
    with pytest.raises(Exception, match=message):
        dag_ml.TrainingOutcome(outcome)


@pytest.fixture
def archive_path(tmp_path: Path, capture: dict[str, Any]) -> Path:
    dag_ml, _core, _role_pipeline = _live_runtime()
    package = dag_ml.PortablePredictorPackage(capture["package_json"])
    outcome = dag_ml.TrainingOutcome(capture["outcome_json"])
    path = tmp_path / "five-models.n4a"
    reference = nirs4all.write_portable_predictor_archive_v2(path, archive_id="archive:sdk.five-models", outcome=outcome, package=package)
    assert reference["archive_id"] == "archive:sdk.five-models"
    assert reference["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    return path


class _ReadOnlyController:
    """Import captured N4ME states through Methods; never fit or read targets."""

    def __init__(self, role_pipeline: Any, capture: dict[str, Any]) -> None:
        self.role_pipeline = role_pipeline
        self.capture = capture
        self.states: dict[int, tuple[Any, dict[str, Any], dict[str, Any]]] = {}
        self.operations: list[str] = []
        self.next_handle = 0

    def artifact(self, message: dict[str, Any]) -> dict[str, Any] | None:
        operation = message["operation"]
        self.operations.append(operation)
        if operation == "release":
            del self.states[message["handle"]["handle"]]
            return None
        assert operation == "hydrate"
        request = message["request"]
        payload = bytes(message["payload"])
        assert hashlib.sha256(payload).hexdigest() == request["artifact"]["content_fingerprint"]
        saved = json.loads(payload)
        assert saved["schema"] == "dagml.methods.regression.v1"
        assert saved["node_id"] == request["node_id"]
        assert saved["params_fingerprint"] == request["params_fingerprint"]
        model = self.role_pipeline.from_states(saved["steps"], saved["states"], feature_names=saved["feature_names"])
        self.next_handle += 1
        self.states[self.next_handle] = (model, saved, request["artifact"])
        return {"handle": self.next_handle, "kind": "model", "owner_controller": request["controller_id"]}

    def predict(self, task: dict[str, Any]) -> dict[str, Any]:
        assert task["phase"] == "PREDICT"
        self.operations.append("PREDICT")
        node = task["node_plan"]
        [(key, artifact)] = task["artifact_inputs"].items()
        model, saved, expected_artifact = self.states[task["input_handles"][key]["handle"]]
        assert saved["node_id"] == node["node_id"] and saved["params_fingerprint"] == node["params_fingerprint"]
        assert artifact["artifact"]["id"] == expected_artifact["id"]
        if task["prediction_inputs"]:
            entries = [(key, value) for key, value in sorted(task["prediction_inputs"].items()) if key.endswith(":predict")]
            inputs = [value for _, value in entries]
            sample_ids = inputs[0]["sample_ids"]
            assert all(value["sample_ids"] == sample_ids for value in inputs)
            features = np.column_stack([value["values"] for value in inputs])
            feature_names = [f"{key.removesuffix(':predict')}/{column}" for key, value in entries for column in range(value["prediction_width"])]
        else:
            [(key, view)] = task["data_views"].items()
            sample_ids = view["sample_ids"]
            [source] = view["source_ids"]
            features = np.asarray([self.capture["heldout_rows"][source]], dtype=float)
            feature_names = [f"{key.removesuffix(':predict')}/{column}" for column in range(features.shape[1])]
        assert sample_ids == self.capture["expected_prediction"]["sample_ids"]
        if saved["feature_names"] != feature_names:
            raise ValueError("current feature order differs from the saved Methods recipe")
        values = np.asarray(model.predict(features), dtype=float).reshape(-1, 1).tolist()
        return {
            "node_id": node["node_id"], "outputs": {}, "artifacts": [], "artifact_handles": {},
            "predictions": [{"producer_node": node["node_id"], "partition": "final", "fold_id": None, "sample_ids": sample_ids, "values": values, "target_names": ["y"]}],
            "lineage": {
                "record_id": "lineage:sdk.replay:" + node["node_id"], "run_id": task["run_id"], "node_id": node["node_id"], "phase": "PREDICT",
                "controller_id": node["controller_id"], "controller_version": node["controller_version"], "variant_id": task["variant_id"], "fold_id": None,
                "branch_path": task["branch_path"], "input_lineage": [], "artifact_refs": [], "params_fingerprint": node["params_fingerprint"],
                "seed": task["seed"], "unsafe_flags": [], "metrics": {}, "loss_attestations": [], "early_stopping_records": [],
            },
        }


def _replay(path: Path, capture: dict[str, Any], controller: _ReadOnlyController, *, trusted: Any = None, request: Any = None) -> Any:
    return nirs4all.replay_portable_predictor_archive_v2(
        path, capture["replay_request_json"] if request is None else request, capture["data_envelopes"],
        capture["trusted_manifests"] if trusted is None else trusted, controller.predict,
        outcome_id="outcome:sdk.five-models.replay", run_id="run:sdk.five-models.replay", artifact_callback=controller.artifact,
    )


def test_real_five_model_archive_replays_exact_captured_refit_without_fit(
    archive_path: Path, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dag_ml, _core, role_pipeline = _live_runtime()
    package = nirs4all.read_portable_predictor_archive_v2(archive_path).to_dict()
    assert len(package["artifact_bindings"]) == len(package["execution_bundle"]["raw_artifact_payloads"]) == 5
    assert {record["artifact"]["kind"] for record in package["execution_bundle"]["refit_artifacts"]} == {"methods_role_pipeline"}
    for envelope in capture["data_envelopes"].values():
        cohort = envelope["predict_cohort"]
        assert cohort["role"] == "inference"
        assert all(record["target_id"] is None for record in cohort["relations"]["records"])
    original = archive_path.read_bytes()
    monkeypatch.setattr(role_pipeline, "fit", lambda *_args, **_kwargs: pytest.fail("replay must not fit"))
    controller = _ReadOnlyController(role_pipeline, capture)
    outcome = _replay(archive_path, capture, controller).to_dict()
    block = outcome["outputs"][0]["predictions"][0]
    assert block["sample_ids"] == capture["expected_prediction"]["sample_ids"]
    np.testing.assert_allclose(block["values"], capture["expected_prediction"]["values"], rtol=1e-10, atol=1e-12)
    assert controller.operations.count("hydrate") == controller.operations.count("PREDICT") == controller.operations.count("release") == 5
    assert not controller.states
    assert archive_path.read_bytes() == original


def test_untrusted_controller_or_tampered_request_is_refused_before_callbacks(archive_path: Path, capture: dict[str, Any]) -> None:
    _dag_ml, _core, role_pipeline = _live_runtime()
    controller = _ReadOnlyController(role_pipeline, capture)
    trusted = copy.deepcopy(capture["trusted_manifests"])
    trusted[0]["controller_version"] = "999.0.0"
    with pytest.raises(Exception):
        _replay(archive_path, capture, controller, trusted=trusted)
    request = json.loads(capture["replay_request_json"])
    request["request_id"] = "replay:tampered"
    with pytest.raises(Exception):
        _replay(archive_path, capture, controller, request=request)
    assert not controller.operations and not controller.states


@pytest.mark.parametrize("mutation", ["raw_hash", "coverage", "relative_path"])
def test_corrupt_archive_is_refused_before_callbacks(archive_path: Path, capture: dict[str, Any], mutation: str) -> None:
    _dag_ml, _core, role_pipeline = _live_runtime()
    with ZipFile(archive_path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["manifest.json"])
    refs = manifest["payloads"]["methods"]["role_pipelines"]
    if mutation == "raw_hash":
        member = refs[0]["member_path"]
        members[member] = members[member] + b"corruption"
    elif mutation == "coverage":
        refs.pop()
        members["manifest.json"] = json.dumps(manifest).encode()
    else:
        refs[0]["member_path"] = "../outside.json"
        members["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as destination:
        for name, payload in members.items():
            destination.writestr(name, payload)
    controller = _ReadOnlyController(role_pipeline, capture)
    with pytest.raises(Exception):
        _replay(archive_path, capture, controller)
    assert not controller.operations and not controller.states
    assert not (archive_path.parent / "outside.json").exists()


def test_role_archive_does_not_widen_existing_n4mm_prediction_lane(archive_path: Path) -> None:
    replay = importlib.import_module("nirs4all.pipeline.dagml.core_archive_replay")
    with pytest.raises(replay.CoreArchiveReplayError):
        nirs4all.predict(model=archive_path, data={"X": [[1.0, 2.0]], "sample_ids": ["heldout"]})


def test_prediction_error_releases_all_hydrated_states(archive_path: Path, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _dag_ml, _core, role_pipeline = _live_runtime()
    controller = _ReadOnlyController(role_pipeline, capture)
    monkeypatch.setattr(role_pipeline, "predict", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected prediction error")))
    with pytest.raises(Exception):
        _replay(archive_path, capture, controller)
    assert controller.operations.count("hydrate") == controller.operations.count("release") == 5
    assert not controller.states
