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
from nirs4all.pipeline.dagml.rt import RtError

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


def _descriptor(
    *,
    owner: str = "controller:methods.pls",
    artifact_sha256: str = "a" * 64,
    n_targets: int = 1,
) -> dict[str, Any]:
    ridge = owner == "controller:methods.ridge"
    return {
        "descriptor_type": "dagml.native_predictor_descriptor.v1",
        "schema_version": 1,
        "artifact_sha256": artifact_sha256,
        "owner_controller": owner,
        "format": "N4MM",
        "format_version": 1,
        "writer_abi": {"major": 2, "minor": 4, "patch": 0},
        "storage_algorithm": 11 if ridge else 0,
        "capabilities": 5 if ridge else 3,
        "dimensions": {
            "training_samples": 8,
            "n_features": 2,
            "n_targets": n_targets,
            "n_components": 0 if ridge else 1,
        },
        "descriptor_fingerprint": ("b" if ridge else "c") * 64,
    }


def _package(
    target_names: list[str] | None = None,
    *,
    descriptor: dict[str, Any] | None | object = ...,
) -> dict[str, Any]:
    targets = target_names or ["protein"]
    native_descriptor = _descriptor(n_targets=len(targets)) if descriptor is ... else descriptor
    artifact = {
        "id": "artifact:methods",
        "kind": "n4m_model",
        "controller_id": "controller:methods.pls",
        "backend": "raw",
        "content_fingerprint": "a" * 64,
        "abi_major": 2,
        "abi_min_minor": 0,
    }
    if native_descriptor is not None:
        artifact["native_predictor_descriptor"] = native_descriptor
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
                    "artifact": artifact,
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
                "target_names": targets,
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
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"n4m-test-library")
    package = _package(["protein", "moisture"])
    descriptor = package["execution_bundle"]["refit_artifacts"][0]["artifact"][
        "native_predictor_descriptor"
    ]
    observed: dict[str, Any] = {}

    def replay(
        archive_path: str,
        sample_ids: list[str],
        x: list[list[float]],
        target_names: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed.update(
            archive_path=archive_path,
            sample_ids=sample_ids,
            x=x,
            target_names=target_names,
            kwargs=kwargs,
        )
        return {
            "outputs": [
                {
                    "predictions": [
                        {
                            "sample_ids": ["sample.one", "sample.two"],
                            "values": [[1.5, 10.5], [2.5, 20.5]],
                        }
                    ]
                }
            ]
        }

    core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: json.dumps(package).encode(),
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [descriptor],
        predict_methods_archive_v2_matrix=replay,
    )
    real_import = core_archive_replay.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "nirs4all_core":
            return core
        return real_import(name)

    monkeypatch.setattr(core_archive_replay.importlib, "import_module", fake_import)
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)

    result = predict_module.predict(
        model=path,
        data={
            "X": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "sample_ids": ["sample.one", "sample.two"],
        },
        methods_library_path=library,
    )

    assert isinstance(result, PredictResult)
    np.testing.assert_array_equal(result.y_pred, [[1.5, 10.5], [2.5, 20.5]])
    assert result.metadata["engine"] == "core-native"
    assert result.metadata["target_names"] == ["protein", "moisture"]
    assert result.metadata["native_predictor_descriptors"] == [descriptor]
    assert "serialized_model_predict" not in result.metadata
    assert observed["archive_path"] == str(path)
    assert observed["sample_ids"] == ["sample.one", "sample.two"]
    assert observed["x"] == [[1.0, 2.0], [3.0, 4.0]]
    assert observed["target_names"] == ["protein", "moisture"]
    assert observed["kwargs"]["methods_library_path"] == str(library.resolve())
    assert len(observed["kwargs"]["methods_library_sha256"]) == 64


def test_historical_descriptor_absence_uses_core_derived_evidence() -> None:
    package = _package(descriptor=None)
    derived = _descriptor()

    predictors = core_archive_replay._validate_native_predictor_evidence(
        package,
        [derived],
    )

    assert predictors == (derived,)


def test_mixed_embedded_descriptor_generation_is_refused() -> None:
    package = _package()
    ridge = _descriptor(
        owner="controller:methods.ridge",
        artifact_sha256="d" * 64,
    )
    package["execution_bundle"]["refit_artifacts"].append(
        {
            "node_id": "model:ridge",
            "controller_id": "controller:methods.ridge",
            "artifact": {
                "id": "artifact:ridge",
                "kind": "n4m_model",
                "controller_id": "controller:methods.ridge",
                "backend": "raw",
                "content_fingerprint": "d" * 64,
                "abi_major": 2,
                "abi_min_minor": 3,
            },
            "params_fingerprint": "q" * 64,
            "data_requirement_keys": [],
            "prediction_requirement_keys": ["model:methods.prediction"],
        }
    )
    package["output_bindings"][0]["node_id"] = "model:ridge"

    with pytest.raises(
        core_archive_replay.CoreArchiveReplayError,
        match="mixes present and historical absent",
    ):
        core_archive_replay._validate_native_predictor_evidence(
            package,
            [_descriptor(), ridge],
        )


def test_accepts_only_ridge_as_final_stacking_predictor() -> None:
    package = _package()
    ridge = _descriptor(
        owner="controller:methods.ridge",
        artifact_sha256="d" * 64,
    )
    ridge_record = {
        "node_id": "model:ridge",
        "controller_id": "controller:methods.ridge",
        "artifact": {
            "id": "artifact:ridge",
            "kind": "n4m_model",
            "controller_id": "controller:methods.ridge",
            "backend": "raw",
            "content_fingerprint": "d" * 64,
            "abi_major": 2,
            "abi_min_minor": 3,
            "native_predictor_descriptor": ridge,
        },
        "params_fingerprint": "q" * 64,
        "data_requirement_keys": [],
        "prediction_requirement_keys": ["model:methods.prediction"],
    }
    package["execution_bundle"]["refit_artifacts"].append(ridge_record)
    package["output_bindings"][0]["node_id"] = "model:ridge"

    assert core_archive_replay._validate_native_predictor_evidence(
        package,
        [_descriptor(), ridge],
    ) == (_descriptor(), ridge)

    package["execution_bundle"]["refit_artifacts"] = [ridge_record]
    with pytest.raises(
        core_archive_replay.CoreArchiveReplayError,
        match="Ridge only as the final stacking predictor",
    ):
        core_archive_replay._validate_native_predictor_evidence(package, [ridge])


def test_core_v2_refuses_prediction_width_that_disagrees_with_binding() -> None:
    outcome = {
        "outputs": [
            {
                "predictions": [
                    {"sample_ids": ["sample.one"], "values": [[1.5]]}
                ]
            }
        ]
    }

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="aligned matrix"):
        core_archive_replay._decode_prediction(
            outcome,
            ("sample.one",),
            target_names=("protein", "moisture"),
        )


def test_core_v2_cross_link_refusal_never_falls_back_to_pipeline_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _archive(tmp_path / "version-mismatch.n4a", 2)
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"n4m-test-library")
    package = _package()
    descriptor = package["execution_bundle"]["refit_artifacts"][0]["artifact"][
        "native_predictor_descriptor"
    ]

    def replay(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("execution bundle content is not cross-linked by outcome reference")

    core = SimpleNamespace(
        read_portable_predictor_package_v2=lambda _: json.dumps(package).encode(),
        inspect_methods_archive_v2_predictors=lambda *_args, **_kwargs: [descriptor],
        predict_methods_archive_v2_matrix=replay,
    )
    real_import = core_archive_replay.importlib.import_module

    def fake_import(name: str) -> Any:
        if name == "nirs4all_core":
            return core
        return real_import(name)

    monkeypatch.setattr(core_archive_replay.importlib, "import_module", fake_import)
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)
    monkeypatch.setattr(
        predict_module,
        "_predict_from_model",
        lambda *args, **kwargs: pytest.fail("legacy model replay must not run"),
    )

    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="replay failed"):
        predict_module.predict(
            model=path,
            data={"X": [[1.0]], "sample_ids": ["sample.one"]},
            methods_library_path=library,
        )


@pytest.mark.parametrize("target_names", [["protein", "protein"], ["protein", " "]])
def test_core_v2_refuses_ambiguous_target_names(target_names: list[str]) -> None:
    with pytest.raises(core_archive_replay.CoreArchiveReplayError, match="target_names"):
        core_archive_replay._single_binding(_package(target_names))


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


def test_non_core_archive_requires_explicit_legacy_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "legacy.n4a"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"bundle_format_version": "1.0"}))
    monkeypatch.setattr(predict_module, "PipelineRunner", _never_runner)

    with pytest.raises(RtError, match="requires conversion") as caught:
        predict_module.predict(model=path, data=np.asarray([[1.0]]))

    assert caught.value.verb == "predict"
    assert caught.value.cause == "unsupported_capability"
