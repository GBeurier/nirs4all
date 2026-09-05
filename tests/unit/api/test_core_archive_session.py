"""Session routing for callback-free Core Archive V2 prediction."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import nirs4all
import nirs4all.api as api_module
import nirs4all.pipeline as pipeline_module
import nirs4all.pipeline.bundle as bundle_module
from nirs4all.api.native_archive_training import NativeMethodsArchiveRunResult
from nirs4all.api.result import PredictResult
from nirs4all.api.session import Session, SessionClosedError, load_session
from nirs4all.pipeline.dagml import core_archive_replay
from nirs4all.pipeline.dagml.rt import RtError


def _archive(path: Path, version: int, *, core: bool = True) -> Path:
    if core:
        manifest = {
            "schema_version": version,
            "profile": f"nirs4all.archive_workspace.v{version}",
            "persistence_kind": "n4a_archive",
            "writer": {"product_aggregate_owner": "nirs4all-core"},
        }
    else:
        manifest = {"bundle_format_version": "1.0"}
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    return path


def _package() -> dict[str, Any]:
    descriptor = {
        "descriptor_type": "dagml.native_predictor_descriptor.v1",
        "schema_version": 1,
        "artifact_sha256": "a" * 64,
        "owner_controller": "controller:methods.pls",
        "format": "N4MM",
        "format_version": 1,
        "writer_abi": {"major": 2, "minor": 4, "patch": 0},
        "storage_algorithm": 0,
        "capabilities": 3,
        "dimensions": {
            "training_samples": 8,
            "n_features": 2,
            "n_targets": 1,
            "n_components": 1,
        },
        "descriptor_fingerprint": "b" * 64,
    }
    return {
        "schema_version": 2,
        "execution_bundle": {
            "data_requirements": [
                {
                    "node_id": "model:methods",
                    "input_name": "x",
                    "schema_fingerprint": "s" * 64,
                    "plan_fingerprint": "p" * 64,
                }
            ],
            "refit_artifacts": [
                {
                    "node_id": "model:methods",
                    "controller_id": "controller:methods.pls",
                    "artifact": {
                        "id": "artifact:methods",
                        "kind": "n4m_model",
                        "controller_id": "controller:methods.pls",
                        "backend": "raw",
                        "content_fingerprint": "a" * 64,
                        "abi_major": 2,
                        "abi_min_minor": 0,
                        "native_predictor_descriptor": descriptor,
                    },
                    "params_fingerprint": "p" * 64,
                    "data_requirement_keys": ["model:methods.x"],
                    "prediction_requirement_keys": [],
                }
            ],
        },
        "training_outcome": {"outcome_fingerprint": "o" * 64},
        "output_bindings": [
            {
                "binding_id": "binding:prediction",
                "node_id": "model:methods",
                "target_names": ["protein"],
            }
        ],
    }


def _descriptor() -> dict[str, Any]:
    descriptor = _package()["execution_bundle"]["refit_artifacts"][0]["artifact"][
        "native_predictor_descriptor"
    ]
    assert isinstance(descriptor, dict)
    return descriptor


def _install_core(
    monkeypatch: pytest.MonkeyPatch,
    core: Any,
) -> None:
    real_import = core_archive_replay.importlib.import_module
    monkeypatch.setattr(
        core_archive_replay.importlib,
        "import_module",
        lambda name: core if name == "nirs4all_core" else real_import(name),
    )
    monkeypatch.setattr(
        core_archive_replay,
        "_resolve_methods_library_identity",
        lambda path: (str(path or "/opt/lib/libn4m.so"), "f" * 64),
    )


def _never_runner(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"PipelineRunner must not be constructed: {args!r} {kwargs!r}")


def _never_bundle(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"BundleLoader must not inspect a Core archive: {args!r} {kwargs!r}")


def test_load_session_preserves_explicit_methods_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _archive(tmp_path / "explicit-runtime.n4a", 2)
    expected = SimpleNamespace(methods_library_path="/opt/lib/libn4m.so.2.5.0")
    observed: list[tuple[Path, str | Path | None]] = []

    def validate(archive_path: Path, *, methods_library_path: str | Path | None = None) -> Any:
        observed.append((archive_path, methods_library_path))
        return expected

    monkeypatch.setattr(core_archive_replay, "validate_core_methods_archive_v2", validate)
    with load_session(path, methods_library_path=expected.methods_library_path) as loaded:
        assert loaded._core_archive_validation == (path, expected)
        assert loaded._runner is None
    assert observed == [(path, expected.methods_library_path)]


def test_load_v2_session_validates_core_and_predicts_without_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    observed: dict[str, Any] = {"reads": 0, "replays": []}

    def read(archive_path: str) -> bytes:
        observed["reads"] += 1
        observed["read_path"] = archive_path
        return json.dumps(_package()).encode()

    def replay(
        archive_path: str,
        sample_ids: list[str],
        x: list[list[float]],
        target_names: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        replay_number = len(observed["replays"]) + 1
        observed["replays"].append(
            {
                "archive_path": archive_path,
                "sample_ids": sample_ids,
                "x": x,
                "target_names": target_names,
                "kwargs": kwargs,
            }
        )
        return {
            "outputs": [
                {
                    "predictions": [
                        {
                            "sample_ids": sample_ids,
                            "values": [[3.5 + replay_number] for _ in sample_ids],
                        }
                    ]
                }
            ]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=read,
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [_descriptor()],
        predict_methods_archive_v2_matrix=replay,
    )
    _install_core(monkeypatch, core)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    session = load_session(path)
    assert session._core_archive_validation is not None
    assert session._core_archive_fingerprint is not None
    cached_package_json = json.dumps(
        session._core_archive_validation[1].package,
        sort_keys=True,
        separators=(",", ":"),
    )
    monkeypatch.setattr(
        core_archive_replay,
        "detect_core_archive_version",
        lambda _: pytest.fail("a loaded Core session must not redispatch to legacy"),
    )
    result = session.predict(
        {"X": [[1.0, 2.0]], "sample_ids": ["sample.one"]},
        methods_library_path="/opt/lib/libn4m.so",
    )
    second = session.predict(
        {
            "X": [[3.0, 4.0], [5.0, 6.0]],
            "sample_ids": ["sample.two", "sample.three"],
        },
        engine="native",
        methods_library_path="/opt/lib/libn4m.so",
    )

    assert isinstance(result, PredictResult)
    np.testing.assert_array_equal(result.y_pred, [[4.5]])
    np.testing.assert_array_equal(second.y_pred, [[5.5], [5.5]])
    assert result.metadata["engine"] == "core-native"
    assert second.metadata["sample_ids"] == ["sample.two", "sample.three"]
    assert session.is_trained
    assert session._runner is None
    assert observed["reads"] == 1
    assert observed["read_path"] == str(path)
    assert len(observed["replays"]) == 2
    assert observed["replays"][0]["kwargs"]["methods_library_path"] == "/opt/lib/libn4m.so"
    assert observed["replays"][0]["sample_ids"] == ["sample.one"]
    assert session._core_archive_validation is not None
    assert json.dumps(
        session._core_archive_validation[1].package,
        sort_keys=True,
        separators=(",", ":"),
    ) == cached_package_json
    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="cannot use the legacy"):
        session.predict(object(), engine="legacy")
    with pytest.raises(RtError) as run_caught:
        session.run(object())
    assert run_caught.value.cause == "unsupported_capability"
    assert run_caught.value.unsupported_capability == "core_archive_v2_prediction_only"
    with pytest.raises(RtError) as caught:
        session.retrain(object(), engine="legacy")
    assert caught.value.cause == "unsupported_capability"
    assert caught.value.unsupported_capability == "core_archive_v2_prediction_only"
    assert session._runner is None
    session.close()
    session.close()
    assert session.status == "closed"
    assert not session.is_trained
    assert session._core_archive_validation is None
    assert session._core_archive_fingerprint is None
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.predict(
            {"X": [[7.0, 8.0]], "sample_ids": ["sample.four"]},
            methods_library_path="/opt/lib/libn4m.so",
        )
    assert len(observed["replays"]) == 2


def test_native_session_run_owns_one_validated_archive_and_releases_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {"reads": 0, "replays": 0, "run_options": []}

    def read(_: str) -> bytes:
        observed["reads"] += 1
        return json.dumps(_package()).encode()

    def replay(
        _: str,
        sample_ids: list[str],
        _x: list[list[float]],
        _target_names: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        observed["replays"] += 1
        return {
            "outputs": [{
                "predictions": [{
                    "sample_ids": sample_ids,
                    "values": [[2.0] for _ in sample_ids],
                }]
            }]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=read,
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [_descriptor()],
        predict_methods_archive_v2_matrix=replay,
    )
    _install_core(monkeypatch, core)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)

    class FakeNativeResult(NativeMethodsArchiveRunResult):
        best_score = 0.25
        num_predictions = 3

        @property
        def native_execution_is_live(self) -> bool:
            return self._live

        @native_execution_is_live.setter
        def native_execution_is_live(self, value: bool) -> None:
            self._live = value

        def __init__(self) -> None:
            self.exports: list[Path] = []
            self.detach_calls = 0
            self.native_execution_is_live = True

        def export(
            self,
            output_path: str | Path,
            format: str = "n4a",
            source: dict[str, Any] | None = None,
            chain_id: str | None = None,
            *,
            compatibility: str | None = None,
        ) -> Path:
            assert format == "n4a"
            assert source is None and chain_id is None and compatibility is None
            destination = Path(output_path)
            self.exports.append(destination)
            _archive(destination, 2)
            validation = core_archive_replay.validate_core_methods_archive_v2(
                destination,
                methods_library_path="/opt/lib/libn4m.so",
            )
            self._native_export_validation = (
                destination,
                validation,
                core_archive_replay._archive_fingerprint(destination),
            )
            return destination

        def close(self) -> None:
            if self.native_execution_is_live:
                self.detach_calls += 1
                self.native_execution_is_live = False

    results: list[FakeNativeResult] = []
    run_module = importlib.import_module("nirs4all.api.run")

    def fake_run(pipeline: list[Any], dataset: Any, **options: Any) -> FakeNativeResult:
        assert pipeline == ["native-step"]
        assert options["engine"] == "native"
        assert options["save_charts"] is False
        observed["run_options"].append(dict(options))
        result = FakeNativeResult()
        results.append(result)
        options["session"]._adopt_native_result(result, dataset)
        return result

    monkeypatch.setattr(run_module, "run", fake_run)
    dataset = {"X": [[0.0]], "y": [1.0], "sample_ids": ["fit.one"]}
    prediction = {"X": [[2.0]], "sample_ids": ["predict.one"]}
    native = Session(pipeline=["native-step"], name="portable")

    first = native.run(
        dataset,
        engine="native",
        methods_library_path="/opt/lib/libn4m.so",
    )
    first_archive = native._core_archive_path
    assert first is results[0]
    assert first.detach_calls == 1
    assert first_archive is not None and first_archive.is_file()
    assert native._runner is None
    assert observed["reads"] == 1
    native.predict(prediction, methods_library_path="/opt/lib/libn4m.so")
    native.predict(prediction, engine="native", methods_library_path="/opt/lib/libn4m.so")
    assert observed["replays"] == 2

    saved = native.save(tmp_path / "saved.n4a")
    assert saved.is_file()
    assert first.detach_calls == 1
    with load_session(saved) as resumed:
        resumed.predict(prediction, methods_library_path="/opt/lib/libn4m.so")
        assert resumed._runner is None
    assert observed["reads"] == 3
    assert observed["replays"] == 3
    assert len(observed["run_options"]) == 1
    assert observed["run_options"][0]["methods_library_path"] == "/opt/lib/libn4m.so"

    second = native.run(dataset, engine="native")
    second_archive = native._core_archive_path
    assert second is results[1]
    assert second.detach_calls == 1
    assert first.detach_calls == 1
    assert not first_archive.exists()
    assert second_archive is not None and second_archive.is_file()
    assert len(native.history) == 2
    assert all(item["engine"] == "native" for item in native.history)
    native.close()
    native.close()
    assert second.detach_calls == 1
    assert not second_archive.exists()
    assert native._core_archive_path is None
    assert native._core_archive_validation is None
    assert native._core_archive_fingerprint is None


@pytest.mark.parametrize("legacy_owner", ["bundle", "runner", "result"])
def test_native_run_refuses_every_legacy_session_owner(legacy_owner: str) -> None:
    session = Session(pipeline=["native-step"])
    if legacy_owner == "bundle":
        session._bundle_path = Path("legacy.n4a")
    elif legacy_owner == "runner":
        session._runner = object()  # type: ignore[assignment]
    else:
        session._last_result = object()  # type: ignore[assignment]

    owner_message = "PipelineRunner" if legacy_owner == "runner" else legacy_owner
    with pytest.raises(RtError, match=rf"already owns a legacy {owner_message}") as caught:
        session.run({}, engine="native")

    assert caught.value.verb == "run"
    assert caught.value.cause == "unsupported_capability"
    assert caught.value.unsupported_capability == "legacy_session_native_engine"


def test_native_session_refuses_runner_and_all_explicit_legacy_top_level_paths() -> None:
    session = Session()
    session._core_archive_path = Path("native.n4a")

    legacy_calls = [
        lambda: session.runner,
        lambda: nirs4all.run(object(), object(), engine="legacy", session=session),  # type: ignore[arg-type]
        lambda: nirs4all.predict(session=session, engine="legacy"),
        lambda: nirs4all.retrain(object(), object(), engine="legacy", session=session),  # type: ignore[arg-type]
        lambda: nirs4all.explain(object(), object(), engine="legacy", session=session),  # type: ignore[arg-type]
    ]
    for legacy_call in legacy_calls:
        with pytest.raises(RtError, match="native Archive V2 Session cannot enter engine='legacy'") as caught:
            legacy_call()
        assert caught.value.cause == "unsupported_capability"
        assert caught.value.unsupported_capability == "native_session_legacy_engine"

    assert session._runner is None


def test_explicit_legacy_remains_available_for_fresh_and_legacy_sessions() -> None:
    fresh = Session()
    existing_runner = object()
    legacy = Session()
    legacy._runner = existing_runner  # type: ignore[assignment]
    legacy._last_result = object()  # type: ignore[assignment]

    fresh._prepare_legacy_access("run")
    legacy._prepare_legacy_access("run")
    assert legacy.runner is existing_runner


def test_session_close_detaches_an_owned_native_result_once() -> None:
    assert nirs4all.SessionClosedError is SessionClosedError
    assert api_module.SessionClosedError is SessionClosedError

    class AttachedTrainingResult:
        is_attached = True

        def __init__(self) -> None:
            self.detach_calls = 0

        def detach(self) -> None:
            self.detach_calls += 1
            self.is_attached = False

    training_result = AttachedTrainingResult()
    native_result = NativeMethodsArchiveRunResult.__new__(NativeMethodsArchiveRunResult)
    native_result._native_training_result = training_result
    session = Session()
    session._last_result = native_result

    session.close()
    session.close()

    assert training_result.detach_calls == 1
    assert session.status == "closed"
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.runner
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.run({})
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.predict({})
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.retrain({})
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.save("closed.n4a")
    with pytest.raises(SessionClosedError, match="Session is closed"):
        session.__enter__()


def test_load_v2_missing_core_fails_without_bundle_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    real_import = core_archive_replay.importlib.import_module

    def missing_core(name: str) -> Any:
        if name == "nirs4all_core":
            raise ImportError("missing test wheel")
        return real_import(name)

    monkeypatch.setattr(core_archive_replay.importlib, "import_module", missing_core)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    with pytest.raises(core_archive_replay.CoreArchiveDependencyError, match="matching nirs4all-core") as caught:
        load_session(path)
    assert caught.value.dependency == "nirs4all-core"
    assert "release lock" in caught.value.mitigation


def test_load_v2_old_core_fails_without_bundle_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    old_core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: json.dumps(_package()).encode()
    )
    real_import = core_archive_replay.importlib.import_module
    monkeypatch.setattr(
        core_archive_replay.importlib,
        "import_module",
        lambda name: old_core if name == "nirs4all_core" else real_import(name),
    )
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    with pytest.raises(core_archive_replay.CoreArchiveDependencyError, match="too old") as caught:
        load_session(path)
    assert caught.value.dependency == "nirs4all-core"


def test_v2_session_closes_reloads_and_rejects_changed_source_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    observed = {"reads": 0, "replays": 0}

    def read(_: str) -> bytes:
        observed["reads"] += 1
        return json.dumps(_package()).encode()

    def replay(
        _: str,
        sample_ids: list[str],
        _x: list[list[float]],
        _target_names: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        observed["replays"] += 1
        return {
            "outputs": [
                {
                    "predictions": [
                        {"sample_ids": sample_ids, "values": [[1.0] for _ in sample_ids]}
                    ]
                }
            ]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=read,
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [_descriptor()],
        predict_methods_archive_v2_matrix=replay,
    )
    _install_core(monkeypatch, core)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    first = load_session(path)
    first.predict(
        {"X": [[1.0, 2.0]], "sample_ids": ["sample.one"]},
        methods_library_path="/opt/lib/libn4m.so",
    )
    first.close()
    assert first._core_archive_validation is None
    assert first._core_archive_fingerprint is None

    resumed = load_session(path)
    resumed.predict(
        {"X": [[3.0, 4.0]], "sample_ids": ["sample.two"]},
        methods_library_path="/opt/lib/libn4m.so",
    )
    assert observed == {"reads": 2, "replays": 2}
    path.write_bytes(path.read_bytes() + b"changed-after-validation")
    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="changed after Session validation"):
        resumed.predict(object(), methods_library_path="/opt/lib/libn4m.so")
    assert observed == {"reads": 2, "replays": 2}
    assert resumed._runner is None
    resumed.close()


def test_v2_session_refuses_oversized_cached_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: b"x"
        * (core_archive_replay._MAX_PACKAGE_BYTES + 1),
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [_descriptor()],
        predict_methods_archive_v2_matrix=lambda *_args, **_kwargs: {},
    )
    _install_core(monkeypatch, core)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="8 MiB Session cache budget"):
        load_session(path)


def test_load_v3_refuses_prediction_session_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "refit.n4a", 3)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    with pytest.raises(NotImplementedError, match="full-refit/retrain.*not a serialized-model"):
        load_session(path)


def test_load_non_core_archive_keeps_legacy_bundle_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "legacy.n4a", 1, core=False)
    observed: list[Path] = []

    class FakeBundleLoader:
        def __init__(self, archive_path: Path) -> None:
            observed.append(archive_path)
            self.pipeline_config = {"steps": ["legacy-step"], "name": "legacy"}

    monkeypatch.setattr(bundle_module, "BundleLoader", FakeBundleLoader)

    session = load_session(path)

    assert observed == [path]
    assert session.name == "legacy"
    assert session.pipeline == ["legacy-step"]
    assert session.is_trained
    assert session._core_archive_path is None


def test_top_level_native_run_refuses_loaded_legacy_session_before_options_or_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "legacy.n4a", 1, core=False)

    class FakeBundleLoader:
        def __init__(self, _archive_path: Path) -> None:
            self.pipeline_config = {"steps": ["legacy-step"], "name": "legacy"}

    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    monkeypatch.setattr(bundle_module, "BundleLoader", FakeBundleLoader)
    monkeypatch.setattr(
        native_module,
        "_require_archive_runtime",
        lambda: pytest.fail("mixed Session authority must fail before native runtime access"),
    )
    session = load_session(path)
    snapshot = dict(vars(session))

    with pytest.raises(RtError, match="already owns a legacy bundle") as caught:
        nirs4all.run(
            session.pipeline,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            engine="native",
            session=session,
        )

    assert caught.value.verb == "run"
    assert caught.value.cause == "unsupported_capability"
    assert caught.value.unsupported_capability == "legacy_session_native_engine"
    assert vars(session) == snapshot
