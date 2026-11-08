"""One-open SDK routing with recording storage/host seams; no scientific fits."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nirs4all.api.predict import predict
from nirs4all.api.result import PredictResult
from nirs4all.api.retrain import retrain
from nirs4all.api.session import Session, SessionClosedError, load_session
from nirs4all.pipeline.bundle.loader import BundleLoader as RealBundleLoader
from nirs4all.pipeline.dagml import host_container as host
from nirs4all.pipeline.dagml.rt import RtError


class RecordingFile:
    """Storage stand-in returning a verified inventory and observing lifetime."""

    def __init__(self, members: dict[str, bytes], *, bundle: bool) -> None:
        self.members = members
        self.closed = False
        self.close_calls = 0
        self.reads: list[str] = []
        self.manifest_reads = 0
        self.manifest = {
            "required_features": [host.HOST_FEATURE], "links": [],
            "members": [{"path": name, "byte_length": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in members.items()],
            "artifacts": [{
                "id": name, "owner": host.HOST_OWNER, "kind": "recipe",
                "content_type": {"id": host.BUNDLE_CONTENT_TYPE if name == host.BUNDLE_MANIFEST_MEMBER else host.JSON_CONTENT_TYPE, "version": 1},
                "storage": {"kind": "embedded", "root_member": name, "member_paths": [name]},
                "identity": {"fingerprint_profile": host.FINGERPRINT_PROFILE, "fingerprint": hashlib.sha256(payload).hexdigest()},
            } for name, payload in members.items()],
            "entrypoints": [{
                "id": host.HOST_ENTRYPOINT_ID, "owner": host.HOST_OWNER, "operation": host.HOST_OPERATION,
                "root_artifact": host.BUNDLE_MANIFEST_MEMBER, "input_artifacts": [],
                "required_features": [host.HOST_FEATURE], "requirements": None,
            }] if bundle else [],
        }

    def manifest_bytes(self) -> bytes:
        assert not self.closed
        self.manifest_reads += 1
        return json.dumps(self.manifest).encode()

    def member(self, path: str) -> bytes:
        assert not self.closed
        self.reads.append(path)
        return self.members[path]

    def view(self) -> dict[str, Any]:
        assert not self.closed
        return {"readability": {"container_sha256": "a" * 64}}

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Drive the real open helper; every attempted Core open is counted."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    opened: list[RecordingFile] = []
    path = tmp_path / "model.container"
    path.write_bytes(b"recording Core storage")
    state: dict[str, Any] = {"manifest": {"source_type": "dagml_native"}, "extra": {}}

    def open_file(candidate: Path, *, memory_budget_bytes: int | None = None) -> RecordingFile:
        assert Path(candidate) == path and memory_budget_bytes is None
        members = dict(state["extra"])
        if state["manifest"] is not None:
            members = {host.BUNDLE_MANIFEST_MEMBER: json.dumps(state["manifest"]).encode(), **members}
        handle = RecordingFile(members, bundle=state["manifest"] is not None)
        opened.append(handle)
        return handle

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("routing crossed a forbidden fit/write/deserialization seam")

    monkeypatch.setitem(sys.modules, "nirs4all_core", SimpleNamespace(open_n4a_file=open_file, write_n4a_file=forbidden))
    monkeypatch.setattr("joblib.load", forbidden)
    return path, state, opened


@pytest.mark.parametrize("fail", [False, True])
def test_general_predict_opens_once_and_closes_on_success_or_error(storage, monkeypatch, fail):
    path, _, opened = storage

    def replay(candidate, data, *, source, output):
        assert candidate == path and output is None
        assert source.container.core_file is opened[0] and not opened[0].closed
        if fail:
            raise ValueError("replay refusal")
        return PredictResult(y_pred=np.array([2.0]))

    monkeypatch.setattr("nirs4all.pipeline.dagml.general_archive.predict_general_archive", replay)
    if fail:
        with pytest.raises(ValueError, match="replay refusal"):
            predict(path, [[1.0]], engine="dag-ml")
    else:
        assert predict(path, [[1.0]], engine="dag-ml").y_pred.tolist() == [2.0]
    assert len(opened) == 1 and opened[0].close_calls == 1
    assert opened[0].manifest_reads == 1
    assert opened[0].reads.count(host.BUNDLE_MANIFEST_MEMBER) == 1


@pytest.mark.parametrize("operation", ["predict", "session", "retrain"])
def test_native_host_bundle_refuses_before_joblib(storage, operation):
    path, _, opened = storage
    if operation == "predict":
        with pytest.raises(RtError) as caught:
            predict(path, object(), engine="native")
        assert caught.value.unsupported_capability == "host_artifact_requires_dagml"
    elif operation == "session":
        with pytest.raises(RtError) as caught:
            load_session(path, engine="native")
        assert caught.value.unsupported_capability == "host_artifact_requires_dagml"
    else:
        with pytest.raises(RtError, match="parent"):
            retrain(path, object(), engine="native")
    assert len(opened) == 1 and opened[0].close_calls == 1


class BorrowedLoader:
    """A loader seam requiring an explicitly borrowed, still-open source."""

    def __init__(self, path, *, source):
        assert source.path == path and not source.container.core_file.closed
        self.source = source
        self.pipeline_config = {"steps": [], "name": "borrowed"}
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        assert not self.source.container.core_file.closed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_predict_passes_loader_and_releases_source_after_runner(storage, monkeypatch, fail):
    path, state, opened = storage
    state["manifest"] = {"bundle_format_version": "1.0"}
    calls: list[Any] = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        def predict(self, *, prediction_obj, **kwargs):
            assert isinstance(prediction_obj, BorrowedLoader)
            assert prediction_obj.source.container.core_file is opened[0]
            calls.append(prediction_obj)
            if fail:
                raise ValueError("prediction refusal")
            return np.array([3.0]), SimpleNamespace()

        def close(self):
            assert not opened[0].closed
            calls.append("runner closed")

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", BorrowedLoader)
    module = importlib.import_module("nirs4all.api.predict")
    monkeypatch.setattr(module, "PipelineRunner", Runner)
    if fail:
        with pytest.raises(ValueError, match="prediction refusal"):
            predict(path, [[1.0]], engine="legacy")
    else:
        assert predict(path, [[1.0]], engine="legacy").y_pred.tolist() == [3.0]
    assert calls[-1] == "runner closed" and calls[0].close_calls == 1
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_legacy_session_retains_the_same_loader_through_prediction_and_close(storage, monkeypatch):
    path, state, opened = storage
    state["manifest"] = {"bundle_format_version": "1.0"}
    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", BorrowedLoader)
    session = load_session(path, engine="legacy")
    loader = session._bundle_loader
    assert isinstance(loader, BorrowedLoader) and not opened[0].closed
    seen: list[Any] = []

    class Runner:
        def predict(self, *, prediction_obj, **kwargs):
            seen.append(prediction_obj)
            assert prediction_obj is loader and not opened[0].closed
            return np.array([4.0]), SimpleNamespace()

        def close(self):
            assert not opened[0].closed

    session._runner = Runner()
    session.predict([[1.0]], engine="legacy")
    session.predict([[2.0]], engine="legacy")
    assert seen == [loader, loader] and len(opened) == 1
    session.close()
    session.close()
    assert loader.close_calls == 1 and opened[0].close_calls == 1
    with pytest.raises(SessionClosedError):
        session.predict([[3.0]])


@pytest.mark.parametrize("fail", [False, True])
def test_general_session_retains_source_or_closes_failed_load(storage, monkeypatch, fail):
    path, _, opened = storage
    loaded = {"archive_fingerprint": "sha256:" + "a" * 64}

    def load(candidate, *, source):
        assert source.container.core_file is opened[0] and not opened[0].closed
        if fail:
            raise ValueError("model refusal")
        return loaded

    def replay(candidate, data, *, source, loaded_archive, expected_archive_fingerprint):
        assert loaded_archive is loaded and expected_archive_fingerprint == loaded["archive_fingerprint"]
        assert source.container.core_file is opened[0] and not opened[0].closed
        return PredictResult(y_pred=np.array([5.0]))

    monkeypatch.setattr("nirs4all.pipeline.dagml.general_archive.load_general_archive", load)
    monkeypatch.setattr("nirs4all.pipeline.dagml.general_archive.predict_general_archive", replay)
    if fail:
        with pytest.raises(ValueError, match="model refusal"):
            load_session(path)
    else:
        session = load_session(path)
        assert not opened[0].closed
        session.predict([[1.0]])
        session.predict([[2.0]])
        session.close()
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_conformal_only_native_prediction_borrows_the_source(storage, monkeypatch):
    path, state, opened = storage
    state["manifest"] = None
    state["extra"] = {"conformal/result.json": b"{}"}
    calibrated = object()

    def load(candidate, *, source):
        assert candidate == path and source.container.core_file is opened[0] and not opened[0].closed
        return calibrated

    def apply(model, **kwargs):
        assert model is calibrated and not opened[0].closed
        return PredictResult(y_pred=np.array([6.0]))

    monkeypatch.setattr("nirs4all.pipeline.dagml.conformal_store.load_conformal_result_archive", load)
    monkeypatch.setattr(importlib.import_module("nirs4all.api.calibrate"), "predict_calibrated", apply)
    result = predict(path, {"y_pred": [6.0], "sample_ids": ["sample.one"]}, engine="native")
    assert result.y_pred.tolist() == [6.0]
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_full_retrain_shares_source_and_accepts_training_spec_above_old_cap(storage, monkeypatch):
    path, state, opened = storage
    payload = json.dumps({"steps": [{"model": "recorded", "label": "x" * 1_048_576}]}).encode()
    state["extra"] = {"train_pipeline.json": payload}
    module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(module, "require_dagml_retrain_backend", lambda: None)
    result = SimpleNamespace(_dagml_refit_artifacts=[object()], per_dataset={"new": {}})

    def run(*, pipeline, **kwargs):
        assert len(opened) == 1 and not opened[0].closed
        assert pipeline[0]["model"] == "recorded"
        return result

    monkeypatch.setattr(importlib.import_module("nirs4all.api.run"), "run", run)
    assert retrain(path, object(), engine="dag-ml") is result
    assert len(opened) == 1 and opened[0].close_calls == 1
    assert opened[0].reads.count("train_pipeline.json") == 1
    assert result._retrain_lineage["source_training_spec_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result._retrain_lineage["source_bundle_sha256"] == "a" * 64


def test_predictor_and_resolver_borrow_an_existing_loader(storage, monkeypatch):
    from nirs4all.pipeline.predictor import Predictor
    from nirs4all.pipeline.resolver import PredictionResolver, ResolvedPrediction

    path, _, opened = storage
    resolved = ResolvedPrediction()

    class ReplayLoader(BorrowedLoader):
        relation_replay_manifest = None
        metadata = None

        def __init__(self, candidate, *, source):
            super().__init__(candidate, source=source)
            self.bundle_path = candidate

        def get_partitioner_routing(self):
            return None

        def predict(self, data):
            assert not self.source.container.core_file.closed
            return np.array([7.0])

        def to_resolved_prediction(self):
            assert not self.source.container.core_file.closed
            return resolved

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", ReplayLoader)
    with host.open_n4a_source(path) as source:
        loader = ReplayLoader(path, source=source)
        values, _ = Predictor(SimpleNamespace()).predict(loader, {"X": [[1.0]]})
        assert values.tolist() == [7.0]
        assert PredictionResolver(path.parent).resolve(loader) is resolved
        resolved.close()
        assert loader.close_calls == 0 and not opened[0].closed
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_attached_conformal_recursion_reuses_the_same_source(storage, monkeypatch):
    path, state, opened = storage
    state["manifest"] = {"bundle_format_version": "1.0"}
    state["extra"] = {"conformal/result.json": b"{}"}
    calibrated = object()

    def load(candidate, *, source):
        assert candidate == path and source.container.core_file is opened[0] and not opened[0].closed
        return calibrated

    def apply(model, **kwargs):
        assert model is calibrated and not opened[0].closed
        return PredictResult(y_pred=np.array([8.0]), intervals={0.9: object()})

    class Runner:
        def __init__(self, **kwargs):
            pass

        def predict(self, *, prediction_obj, **kwargs):
            assert isinstance(prediction_obj, BorrowedLoader)
            assert prediction_obj.source.container.core_file is opened[0]
            return np.array([8.0]), SimpleNamespace()

        def close(self):
            assert not opened[0].closed

    monkeypatch.setattr("nirs4all.pipeline.bundle.BundleLoader", BorrowedLoader)
    monkeypatch.setattr(importlib.import_module("nirs4all.api.predict"), "PipelineRunner", Runner)
    monkeypatch.setattr("nirs4all.pipeline.dagml.conformal_store.load_conformal_result_archive", load)
    monkeypatch.setattr(importlib.import_module("nirs4all.api.calibrate"), "predict_calibrated", apply)
    result = predict(path, {"X": [[1.0]], "sample_ids": ["sample.one"]}, engine="legacy", coverage=0.9)
    assert result.interval_coverages == (0.9,)
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_unset_engine_dispatches_a_host_bundle_to_the_general_profile(storage, monkeypatch):
    """No engine= and no N4A_ENGINE: the default selects the general DAG profile, never the native refusal."""
    path, _, opened = storage

    def replay(candidate, data, *, source, output):
        assert candidate == path and output is None and not opened[0].closed
        return PredictResult(y_pred=np.array([2.0]))

    monkeypatch.setattr("nirs4all.pipeline.dagml.general_archive.predict_general_archive", replay)
    assert predict(path, [[1.0]]).y_pred.tolist() == [2.0]
    assert len(opened) == 1 and opened[0].close_calls == 1


@pytest.mark.parametrize("selector", ["environment", "parameter"])
def test_explicit_native_selection_of_a_host_bundle_stays_strict(storage, monkeypatch, selector):
    path, _, opened = storage
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.general_archive.predict_general_archive",
        lambda *args, **kwargs: pytest.fail("an explicit native selection reached the general profile"),
    )
    if selector == "environment":
        monkeypatch.setenv("N4A_ENGINE", "native")
    with pytest.raises(RtError) as caught:
        predict(path, [[1.0]], **({"engine": "native"} if selector == "parameter" else {}))
    assert caught.value.unsupported_capability == "host_artifact_requires_dagml"
    assert len(opened) == 1 and opened[0].close_calls == 1


def test_unset_engine_replays_y_pred_on_an_attached_bundle_while_explicit_native_stays_strict(storage, monkeypatch):
    path, state, opened = storage
    state["manifest"] = {"bundle_format_version": "1.0"}
    state["extra"] = {"conformal/result.json": b"{}"}
    calibrated = object()

    def load(candidate, *, source):
        assert candidate == path and source.container.core_file is opened[-1] and not opened[-1].closed
        return calibrated

    def apply(model, **kwargs):
        assert model is calibrated
        return PredictResult(y_pred=np.array([9.0]))

    monkeypatch.setattr("nirs4all.pipeline.dagml.conformal_store.load_conformal_result_archive", load)
    monkeypatch.setattr(importlib.import_module("nirs4all.api.calibrate"), "predict_calibrated", apply)
    data = {"y_pred": [9.0], "sample_ids": ["sample.one"]}
    assert predict(path, data).y_pred.tolist() == [9.0]
    assert len(opened) == 1 and opened[0].close_calls == 1

    with pytest.raises(RtError) as caught:
        predict(path, data, engine="native")
    assert caught.value.unsupported_capability == "host_artifact_requires_dagml"
    assert len(opened) == 2 and all(handle.close_calls == 1 for handle in opened)


def held_loader(source):
    """A real ``BundleLoader`` holding an open source, built without reopening the container."""
    loader = RealBundleLoader.__new__(RealBundleLoader)
    loader.bundle_path = source.path
    loader._source = source
    loader._owns_source = False
    return loader


def test_session_retrain_normalizes_its_loader_once_and_borrows_the_held_source(storage, monkeypatch):
    path, state, opened = storage
    payload = json.dumps({"steps": [{"model": "recorded"}]}).encode()
    state["manifest"] = {"bundle_format_version": "1.0"}  # a single-model bundle that is not a dagml_native export
    state["extra"] = {"train_pipeline.json": payload}
    module = importlib.import_module("nirs4all.api.retrain")
    monkeypatch.setattr(module, "require_dagml_retrain_backend", lambda: None)
    result = SimpleNamespace(_dagml_refit_artifacts=[object()], per_dataset={"new": {}}, best_score=0.5, num_predictions=1)

    def run(*, pipeline, **kwargs):
        assert pipeline == [{"model": "recorded"}] and not opened[0].closed
        return result

    monkeypatch.setattr(importlib.import_module("nirs4all.api.run"), "run", run)
    with host.open_n4a_source(path) as source:
        session = Session()
        session._status = "trained"
        session._bundle_path = path
        session._bundle_loader = held_loader(source)
        assert session.retrain(object(), mode="full") is result
        assert len(opened) == 1 and not opened[0].closed, "the Session's held source is borrowed, not reopened or closed"
        assert opened[0].reads.count("train_pipeline.json") == 1
    assert result._retrain_lineage["source_training_spec_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result._retrain_lineage["source_bundle_sha256"] == "a" * 64
    assert len(opened) == 1 and opened[0].close_calls == 1
