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

import nirs4all.pipeline as pipeline_module
import nirs4all.pipeline.bundle as bundle_module
from nirs4all.api.result import PredictResult
from nirs4all.api.session import load_session
from nirs4all.pipeline.dagml import core_archive_replay


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
            ]
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


def _never_runner(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"PipelineRunner must not be constructed: {args!r} {kwargs!r}")


def _never_bundle(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"BundleLoader must not inspect a Core archive: {args!r} {kwargs!r}")


def test_load_v2_session_validates_core_and_predicts_without_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    observed: dict[str, Any] = {"reads": 0}

    def read(archive_path: str) -> bytes:
        observed["reads"] += 1
        observed["read_path"] = archive_path
        return json.dumps(_package()).encode()

    def replay(
        archive_path: str,
        request: dict[str, Any],
        data_envelopes: dict[str, Any],
        methods_inputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed.update(
            archive_path=archive_path,
            request=request,
            data_envelopes=data_envelopes,
            methods_inputs=methods_inputs,
            kwargs=kwargs,
        )
        return {
            "outputs": [
                {
                    "predictions": [
                        {"sample_ids": ["sample.one"], "values": [[4.5]]}
                    ]
                }
            ]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=read,
        replay_methods_archive_v2=replay,
    )
    dag_ml = SimpleNamespace(
        sample_relation_set_fingerprint_json=lambda _: "r" * 64,
        sign_training_replay_request=lambda request: {
            **request,
            "request_fingerprint": "f" * 64,
        },
    )
    real_import = core_archive_replay.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "nirs4all_core":
            return core
        if name == "dag_ml":
            return dag_ml
        return real_import(name)

    monkeypatch.setattr(core_archive_replay.importlib, "import_module", fake_import)
    monkeypatch.setattr(pipeline_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(bundle_module, "BundleLoader", _never_bundle)

    session = load_session(path)
    monkeypatch.setattr(
        core_archive_replay,
        "detect_core_archive_version",
        lambda _: pytest.fail("a loaded Core session must not redispatch to legacy"),
    )
    result = session.predict(
        {"X": [[1.0, 2.0]], "sample_ids": ["sample.one"]},
        methods_library_path="/opt/lib/libn4m.so",
    )

    assert isinstance(result, PredictResult)
    np.testing.assert_array_equal(result.y_pred, [[4.5]])
    assert result.metadata["engine"] == "core-native"
    assert session.is_trained
    assert session._runner is None
    assert observed["reads"] == 2
    assert observed["read_path"] == str(path)
    assert observed["kwargs"]["methods_library_path"] == "/opt/lib/libn4m.so"
    assert observed["methods_inputs"]["model:methods.x"]["sample_ids"] == [
        "sample.one"
    ]
    with pytest.raises(NotImplementedError, match="full-refit/retrain"):
        session.retrain({"X": [[2.0]], "sample_ids": ["sample.two"]})
    assert session._runner is None


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

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="matching nirs4all-core"):
        load_session(path)


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

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="too old"):
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
