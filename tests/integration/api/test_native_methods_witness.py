"""Installed-wheel lifecycle proof for the strict native Methods witness."""

from __future__ import annotations

import importlib
import json
import os
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml.native_archive_replay import (
    predict_methods_archive_v2_raw_result,
    validate_methods_archive_v2,
)
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError

_REQUIRE_N4M = os.environ.get("NIRS4ALL_REQUIRE_N4M") == "1"

pytestmark = pytest.mark.methods


def _require_installed_runtime() -> tuple[Any, Any]:
    """Import the published runtime only after the test removes library overrides."""

    try:
        import dag_ml
        import n4m
    except Exception as error:  # pragma: no cover - exact loader failures depend on the host wheel
        message = f"installed Methods witness runtime is unavailable: {error}"
        if _REQUIRE_N4M:
            pytest.fail(message, pytrace=True)
        pytest.skip(message)
    if not callable(getattr(dag_ml, "execute_methods_training", None)) or not isinstance(
        getattr(dag_ml, "TrainingResult", None), type
    ):
        message = "installed dag-ml wheel does not expose the strict Methods TrainingResult surface"
        if _REQUIRE_N4M:
            pytest.fail(message, pytrace=True)
        pytest.skip(message)
    return dag_ml, n4m


def test_installed_methods_witness_claim_closes_through_the_public_dagml_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public native route yields a live claim and closes the real wheel facade."""

    monkeypatch.delenv("N4M_LIB_PATH", raising=False)
    assert "N4M_LIB_PATH" not in os.environ
    dag_ml, n4m = _require_installed_runtime()
    assert callable(n4m.library_path)

    import nirs4all

    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]],
        dtype=float,
    )
    targets = np.arange(6.0, dtype=float)
    result = nirs4all.run(
        [KFold(n_splits=3), {"model": PLSRegression(n_components=1)}],
        {"X": features, "y": targets, "sample_ids": [f"fit-{index}" for index in range(len(features))]},
        engine="native",
        save_charts=False,
    )

    assert type(result) is nirs4all.NativeMethodsRunResult
    estimator = result.native_estimator
    training_result = estimator.training_result_
    assert type(training_result) is dag_ml.TrainingResult
    assert training_result.is_attached is True
    claim = result.native_execution_claim
    assert claim.execution_entrypoint == "dag_ml.execute_methods_training"
    assert claim.execution_mode == "methods_callback_free"
    assert claim.methods_library_mode == "explicit_absolute"
    assert claim.portable_artifacts_required is True
    assert claim.outcome_fingerprint == training_result.outcome_fingerprint
    assert claim.outcome_fingerprint == training_result.outcome.to_dict()["outcome_fingerprint"]
    assert result.native_execution_is_live is True

    result.close()

    assert training_result.is_attached is False
    assert result.native_execution_is_live is False
    with pytest.raises(DagMLNativeCoverageError, match="no longer attached"):
        _ = result.native_execution_claim
    assert estimator.detach_native_training_result() is False

    archive_path = result.export(tmp_path / "native-methods-witness.n4a")
    assert archive_path.is_file()
    validate_methods_archive_v2(archive_path)


def test_installed_terminal_predict_is_callback_free_and_archives_without_a_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the published CV→REFIT→terminal-PREDICT facade end to end."""

    monkeypatch.delenv("N4M_LIB_PATH", raising=False)
    dag_ml, n4m = _require_installed_runtime()
    assert version("dag-ml") == "0.3.22"
    assert version("nirs4all-methods") == "1.0.13"
    assert dag_ml.version() == "0.3.22"
    assert tuple(n4m.abi_version()) == (2, 3, 0)
    assert callable(getattr(dag_ml, "execute_methods_cv_refit_terminal_predict", None))
    assert isinstance(getattr(dag_ml, "MethodsTerminalPredictionResult", None), type)
    assert isinstance(getattr(dag_ml, "MethodsTerminalPredictionReceipt", None), type)

    import nirs4all
    from nirs4all.api import native_training

    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]],
        dtype=float,
    )
    targets = np.arange(6.0, dtype=float)
    training = {"X": features, "y": targets, "sample_ids": ["fit-z", "fit-a", "fit-y", "fit-b", "fit-x", "fit-c"]}
    pipeline = [KFold(n_splits=3, shuffle=False), {"model": PLSRegression(n_components=1)}]

    def runtime_reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid strict terminal request reached the native runtime")

    with monkeypatch.context() as preflight:
        preflight.setattr(native_training, "resolve_methods_library_path", runtime_reached)
        preflight.setattr(
            native_training.DagMLNativeClient,
            "execute_methods_cv_refit_terminal_predict",
            runtime_reached,
        )
        with pytest.raises(ValueError, match="feature widths"):
            nirs4all.run(
                pipeline,
                training,
                engine="native",
                save_charts=False,
                terminal_predict={"X": np.ones((2, 3)), "sample_ids": ["bad-a", "bad-b"]},
            )
        with pytest.raises(ValueError, match="target-free"):
            nirs4all.run(
                pipeline,
                training,
                engine="native",
                save_charts=False,
                terminal_predict={
                    "X": np.ones((2, 2)),
                    "sample_ids": ["bad-a", "bad-b"],
                    "external_oof": {"cache": "forbidden"},
                },
            )

    def legacy_runner_reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("strict terminal prediction constructed a legacy PipelineRunner")

    def generic_methods_training_reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("strict terminal prediction selected generic Methods training")

    terminal_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    native_terminal_predict = dag_ml.execute_methods_cv_refit_terminal_predict

    def observe_terminal_predict(*args: object, **kwargs: object) -> object:
        terminal_calls.append((args, kwargs))
        assert len(args) == 7
        assert "op_callback" not in kwargs
        return native_terminal_predict(*args, **kwargs)

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "PipelineRunner", legacy_runner_reached)
    monkeypatch.setattr(
        native_training.DagMLNativeClient,
        "execute_methods_training",
        generic_methods_training_reached,
    )
    monkeypatch.setattr(dag_ml, "execute_methods_training", generic_methods_training_reached)
    monkeypatch.setattr(dag_ml, "execute_methods_cv_refit_terminal_predict", observe_terminal_predict)
    result = nirs4all.run(
        pipeline,
        training,
        engine="native",
        save_charts=False,
        terminal_predict={
            "X": np.asarray([[8.0, 0.0], [-1.0, 4.0]], dtype=float),
            "sample_ids": ["predict-z", "predict-a"],
        },
    )
    try:
        assert type(result) is nirs4all.NativeMethodsRunResult
        assert len(terminal_calls) == 1
        execution = result.native_estimator.native_training_execution_
        assert not hasattr(execution, "op_callback")
        assert execution.run_id.startswith("run:nirs4all.strict_terminal.")

        witness = result._live_witness  # noqa: SLF001 - assert the opaque native authority remains exact.
        raw_terminal_result = witness._terminal_result  # noqa: SLF001 - public result intentionally keeps it private.
        receipt = result.terminal_receipt
        assert type(raw_terminal_result) is dag_ml.MethodsTerminalPredictionResult
        assert type(receipt) is dag_ml.MethodsTerminalPredictionReceipt
        assert raw_terminal_result.terminal_receipt is receipt
        for native_type in (dag_ml.MethodsTerminalPredictionResult, dag_ml.MethodsTerminalPredictionReceipt):
            with pytest.raises(TypeError):
                native_type()
            with pytest.raises(TypeError):
                object.__new__(native_type)
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(receipt, "terminal_run_id", "forged:terminal")

        terminal_prediction = result.terminal_prediction
        assert terminal_prediction["sample_ids"] == ["predict-a", "predict-z"]
        values_by_sample_id = dict(zip(terminal_prediction["sample_ids"], terminal_prediction["values"], strict=True))
        assert values_by_sample_id["predict-a"][0] < values_by_sample_id["predict-z"][0]
        claim = result.native_execution_claim
        assert claim.execution_entrypoint == "dag_ml.execute_methods_cv_refit_terminal_predict"
        assert claim.terminal_run_id == f"{execution.run_id}:methods-terminal-predict"
        assert claim.receipt_fingerprint == receipt.receipt_fingerprint

        result.close()
        result.close()
        assert result.terminal_receipt is receipt
        assert result.native_execution_is_live is False
        with pytest.raises(DagMLNativeCoverageError, match="no longer attached"):
            _ = result.native_execution_claim

        archive_path = result.export(tmp_path / "native-terminal.n4a")
        assert archive_path.is_file()
        validate_methods_archive_v2(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            cache_payload = json.loads(archive.read("dagml/prediction_cache_payload_set.json"))
            outcome_payload = json.loads(archive.read("dagml/training_outcome.json"))
            package_bytes = archive.read("dagml/portable_predictor_package.json")
            archive_json = "\n".join(
                archive.read(name).decode("utf-8") for name in names if name.endswith(".json")
            )
        assert cache_payload == {
            "bundle_id": outcome_payload["execution_bundle"]["bundle_id"],
            "schema_version": 2,
            "caches": [],
        }
        assert outcome_payload["portable_prediction_caches"] is None
        assert all("receipt" not in name.lower() for name in names)
        assert "terminal_run_id" not in archive_json
        assert "receipt_fingerprint" not in archive_json
        assert receipt.terminal_run_id not in archive_json
        assert receipt.receipt_fingerprint not in archive_json

        from nirs4all_core import read_portable_predictor_package_v2

        assert read_portable_predictor_package_v2(str(archive_path)) == package_bytes
        package = dag_ml.PortablePredictorPackage(package_bytes.decode("utf-8"))
        assert package.to_dict()["schema_version"] == 2
        replay = predict_methods_archive_v2_raw_result(
            archive_path,
            np.asarray([[4.0, 3.0], [8.0, 0.0]], dtype=float),
            sample_ids=["fresh-z", "fresh-a"],
        )
        assert replay.sample_ids == ("fresh-z", "fresh-a")
        assert replay.values.shape == (2, 1)
    finally:
        result.close()


def test_installed_native_stacking_keeps_normal_oof_cache_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The terminal discard exception must not replace normal retained OOF caches."""

    monkeypatch.delenv("N4M_LIB_PATH", raising=False)
    _require_installed_runtime()

    import nirs4all

    result = nirs4all.run(
        [
            KFold(n_splits=2, shuffle=False),
            {"branch": [[{"model": PLSRegression(n_components=1)}], [{"model": PLSRegression(n_components=1)}]]},
            {"merge": "predictions"},
            {"model": Ridge(alpha=0.25)},
        ],
        {
            "X": np.asarray(
                [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0], [6.0, 1.0], [7.0, 0.0]],
                dtype=float,
            ),
            "y": np.arange(8.0, dtype=float),
            "sample_ids": [f"fit-{index}" for index in range(8)],
        },
        engine="native",
        save_charts=False,
    )
    try:
        outcome = result.native_estimator.training_outcome_.to_dict()
        retained_caches = outcome["portable_prediction_caches"]
        assert isinstance(retained_caches, dict)
        assert retained_caches["schema_version"] == 2
        assert len(retained_caches["caches"]) == 2
        assert all(cache["format"] == "dag-ml-json-prediction-blocks-v2" for cache in retained_caches["caches"])
        assert all(cache["block_count"] == 2 for cache in retained_caches["caches"])

        archive_path = result.export(tmp_path / "native-stacking-oof.n4a")
        with zipfile.ZipFile(archive_path) as archive:
            archived_caches = json.loads(archive.read("dagml/prediction_cache_payload_set.json"))
        assert archived_caches == retained_caches
    finally:
        result.close()
