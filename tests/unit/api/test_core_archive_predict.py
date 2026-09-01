"""Public fail-closed Core Archive V2/V3 prediction routing tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from nirs4all.api.result import PredictResult
from nirs4all.pipeline.dagml import core_archive_replay

predict_module = importlib.import_module("nirs4all.api.predict")


def _archive(path: Path, version: int) -> Path:
    manifest = {
        "schema_version": version,
        "profile": f"nirs4all.archive_workspace.v{version}",
        "persistence_kind": "n4a_archive",
        "writer": {"product_aggregate_owner": "nirs4all-core"},
    }
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


def test_predict_replays_core_v2_without_constructing_pipeline_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    observed: dict[str, Any] = {}

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
                        {
                            "sample_ids": ["sample.one", "sample.two"],
                            "values": [[1.5], [2.5]],
                        }
                    ]
                }
            ]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: json.dumps(_package()).encode(),
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
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)

    result = predict_module.predict(
        model=path,
        data={
            "X": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "sample_ids": ["sample.one", "sample.two"],
            "groups": ["g1", "g2"],
            "metadata": {"batch": [1, 2]},
        },
        methods_library_path="/opt/lib/libn4m.so",
    )

    assert isinstance(result, PredictResult)
    np.testing.assert_array_equal(result.y_pred, [[1.5], [2.5]])
    assert result.metadata["engine"] == "core-native"
    assert "serialized_model_predict" not in result.metadata
    assert observed["archive_path"] == str(path)
    assert observed["request"]["phase"] == "PREDICT"
    assert observed["request"]["request_fingerprint"] == "f" * 64
    assert observed["request"]["data_envelope_keys"] == ["model:methods.x"]
    envelope = observed["data_envelopes"]["model:methods.x"]
    assert envelope["target_content_fingerprint"] is None
    assert envelope["coordinator_relations"]["records"][0]["group_id"] == "g1"
    assert envelope["coordinator_relations"]["records"][1]["metadata"] == {"batch": 2}
    assert observed["methods_inputs"] == {
        "model:methods.x": {
            "sample_ids": ["sample.one", "sample.two"],
            "x": [[1.0, 2.0], [3.0, 4.0]],
            "target_names": ["protein"],
        }
    }
    assert observed["kwargs"]["methods_library_path"] == "/opt/lib/libn4m.so"


def test_recognized_v2_missing_core_wheel_fails_without_legacy_fallback(
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
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(
        predict_module,
        "_predict_from_model",
        lambda *args, **kwargs: pytest.fail("legacy model replay must not run"),
    )

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="matching nirs4all-core"):
        predict_module.predict(
            model=path,
            data={"X": [[1.0]], "sample_ids": ["sample.one"]},
            methods_library_path="/opt/lib/libn4m.so",
        )


def test_recognized_v2_old_core_wheel_fails_without_pipeline_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "portable.n4a", 2)
    old_core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: json.dumps(_package()).encode()
    )
    monkeypatch.setattr(
        core_archive_replay.importlib,
        "import_module",
        lambda name: old_core if name == "nirs4all_core" else importlib.import_module(name),
    )
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="too old"):
        predict_module.predict(
            model=path,
            data={"X": [[1.0]], "sample_ids": ["sample.one"]},
            methods_library_path="/opt/lib/libn4m.so",
        )


def test_core_v3_is_refused_as_full_refit_not_serialized_predict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "refit.n4a", 3)
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)

    with pytest.raises(NotImplementedError, match="full-refit/retrain.*not a serialized-model"):
        predict_module.predict(
            model=path,
            data={"X": [[1.0]], "sample_ids": ["sample.one"]},
        )


def test_non_core_model_keeps_historical_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "legacy.n4a"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"bundle_format_version": "1.0"}))
    expected = PredictResult(y_pred=np.asarray([7.0]), model_name="legacy")
    observed: dict[str, Any] = {}

    def legacy(**kwargs: Any) -> PredictResult:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(predict_module, "_predict_from_model", legacy)

    result = predict_module.predict(model=path, data=np.asarray([[1.0]]))

    assert result is expected
    assert observed["model"] == path
