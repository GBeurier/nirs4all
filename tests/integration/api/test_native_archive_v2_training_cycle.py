"""Opt-in real run/save/close/load/predict proof for Archive V2/N4MM."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.api.native_archive_training import NativeArchiveTrainingError
from nirs4all.api.session import Session, SessionClosedError
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from nirs4all.pipeline.dagml.core_archive_replay import CoreArchiveReplayError

_REQUIRE_ENV = "NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2"
_LIBRARY_ENV = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"


def _pipeline() -> list[object]:
    return [KFold(n_splits=3), {"model": PLSRegression(n_components=1)}]


def _portable_roadmap_pipeline() -> list[object]:
    return [
        KFold(n_splits=3),
        SNV(),
        SavitzkyGolay(window_length=3, polyorder=1),
        {"model": PLSRegression(n_components=1)},
    ]


def _dataset() -> dict[str, object]:
    features = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
        ],
        dtype=float,
    )
    return {
        "X": features,
        "y": np.arange(6.0, dtype=float),
        "sample_ids": [f"fit.{index}" for index in range(len(features))],
    }


def _multi_target_dataset() -> dict[str, object]:
    dataset = _dataset()
    first = np.asarray(dataset["y"], dtype=float)
    dataset["y"] = np.column_stack((first, 10.0 + 2.0 * first))
    dataset["target_names"] = ["protein", "moisture"]
    return dataset


def test_native_archive_training_refuses_a_missing_runtime_before_pipeline_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("native preflight constructed PipelineRunner")

    real_import = native_module.importlib.import_module

    def unavailable(name: str, package: str | None = None) -> object:
        if name in {"dag_ml", "nirs4all_core"}:
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(native_module.importlib, "import_module", unavailable)
    with pytest.raises(NativeArchiveTrainingError, match="matching dag-ml and nirs4all-core"):
        run_module.run(_pipeline(), _dataset(), engine="native", save_charts=False)


def test_native_archive_training_refuses_unreplayable_preprocessing_without_legacy_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("unsupported native preprocessing constructed PipelineRunner")

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(native_module, "_require_archive_runtime", lambda: (object(), object()))
    monkeypatch.setattr(native_module, "_resolve_methods_library_path", lambda _path: "/opt/lib/libn4m.so")

    with pytest.raises(
        ValueError,
        match="SNV or Savitzky-Golay requires an upstream DAG-ML Methods controller and replay contract",
    ):
        run_module.run(
            _portable_roadmap_pipeline(),
            _dataset(),
            engine="native",
            save_charts=False,
        )


def test_native_archive_training_refuses_closed_session_before_runtime_or_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("closed native Session constructed PipelineRunner")

    session = Session()
    session.close()
    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(
        native_module,
        "_require_archive_runtime",
        lambda: pytest.fail("closed native Session inspected the native runtime"),
    )

    with pytest.raises(SessionClosedError, match="Session is closed"):
        run_module.run(
            _pipeline(),
            _dataset(),
            engine="native",
            session=session,
            save_charts=False,
        )

    assert session._runner is None
    assert session.status == "closed"


def test_native_archive_training_adopts_the_result_without_pipeline_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("native Session adoption constructed PipelineRunner")

    prepared = type(
        "Prepared",
        (),
        {
            "request": {"schema_version": 1},
            "data_envelopes": {"model:methods.x": {}},
            "relations": {},
            "training_influence": {},
            "outcome_id": "outcome:test",
            "run_id": "run:test",
            "bundle_id": "bundle:test",
            "warnings": (),
            "diagnostics": {},
        },
    )()
    contracts = type("Contracts", (), {"to_prepared": lambda self: prepared})()

    class TrainingResult:
        is_attached = True
        outcome = {"score_set": {"scores": []}, "outcome_fingerprint": "f" * 64}

        def export_portable_predictor_package(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"schema_version": 2}

        def detach(self) -> None:
            self.is_attached = False

    training_result = TrainingResult()
    dag_ml = type(
        "DagMl",
        (),
        {
            "sign_training_request": staticmethod(lambda request: request),
            "execute_methods_training": staticmethod(lambda *_args, **_kwargs: training_result),
        },
    )()
    sentinel = object()
    adopted: list[tuple[object, object]] = []
    session = Session()
    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(native_module, "_require_archive_runtime", lambda: (dag_ml, object()))
    monkeypatch.setattr(native_module, "_resolve_methods_library_path", lambda _: "/opt/lib/libn4m.so")
    monkeypatch.setattr(native_module, "lower_raw_array_training_contracts", lambda *_args, **_kwargs: contracts)
    monkeypatch.setattr(native_module, "_scores_to_run_result", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(native_module, "NativeMethodsArchiveRunResult", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        session,
        "_adopt_native_result",
        lambda result, dataset: adopted.append((result, dataset)),
    )

    dataset = _dataset()
    result = run_module.run(
        _pipeline(),
        dataset,
        engine="native",
        session=session,
        save_charts=False,
        methods_library_path="/opt/lib/libn4m.so",
    )

    assert result is sentinel
    assert adopted == [(sentinel, dataset)]
    assert session._runner is None


def test_native_archive_training_requires_unambiguous_multi_target_names() -> None:
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    dataset = _multi_target_dataset()

    missing = dict(dataset)
    missing.pop("target_names")
    with pytest.raises(ValueError, match="multi-target.*explicit target_names"):
        native_module._normalize_training_arrays(missing)

    mismatch = {**dataset, "target_names": ["protein"]}
    with pytest.raises(ValueError, match="length must match y width 2"):
        native_module._normalize_training_arrays(mismatch)

    duplicate = {**dataset, "target_names": ["protein", "protein"]}
    with pytest.raises(ValueError, match="must be unique"):
        native_module._normalize_training_arrays(duplicate)

    _, targets, _, target_names = native_module._normalize_training_arrays(dataset)
    assert targets.shape == (6, 2)
    assert target_names == ("protein", "moisture")


@pytest.mark.methods
@pytest.mark.skipif(
    os.environ.get(_REQUIRE_ENV) != "1",
    reason=f"set {_REQUIRE_ENV}=1 with installed native wheels to run",
)
def test_real_native_run_saves_and_replays_after_process_close(tmp_path: Path) -> None:
    import nirs4all

    library = os.environ.get(_LIBRARY_ENV)
    if not library:
        pytest.fail(f"{_LIBRARY_ENV} must point to the exact libn4m release runtime")
    library_path = Path(library)
    if not library_path.is_file():
        pytest.fail(f"{_LIBRARY_ENV} does not identify a file: {library_path}")

    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("native Archive V2 lifecycle constructed PipelineRunner")

    prediction_features = [[8.0, 0.0], [-1.0, 4.0]]
    prediction_ids = ["predict.z", "predict.a"]
    original_runner = run_module.PipelineRunner
    run_module.PipelineRunner = LegacyPathReached
    try:
        result = nirs4all.run(
            _pipeline(),
            _multi_target_dataset(),
            engine="native",
            save_charts=False,
            methods_library_path=library_path,
        )
        archive_path = result.export(tmp_path / "native-methods-v2.n4a")
        reference = result.native_archive_reference
        assert reference is not None
        assert reference["archive_id"].startswith("archive:")
        assert len(reference["archive_sha256"]) == 64
        result.close()
        result.close()
        assert result.native_execution_is_live is False

        owned_session = Session(
            pipeline=_pipeline(),
            name="native-session",
            methods_library_path=library_path,
        )
        owned_archive: Path | None = None
        try:
            owned_result = owned_session.run(
                _multi_target_dataset(),
                engine="native",
            )
            owned_archive = owned_session._core_archive_path
            assert owned_result.native_execution_is_live is False
            assert owned_archive is not None and owned_archive.is_file()
            assert owned_session._runner is None
            owned_prediction = owned_session.predict(
                {"X": prediction_features, "sample_ids": prediction_ids},
                methods_library_path=library_path,
            )
            session_archive = owned_session.save(tmp_path / "native-session-v2.n4a")
        finally:
            owned_session.close()
            owned_session.close()
        assert owned_archive is not None
        assert not owned_archive.exists()
        with nirs4all.load_session(session_archive) as resumed_session:
            resumed_prediction = resumed_session.predict(
                {"X": prediction_features, "sample_ids": prediction_ids},
                methods_library_path=library_path,
            )
        np.testing.assert_allclose(
            resumed_prediction.y_pred,
            owned_prediction.y_pred,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        run_module.PipelineRunner = original_runner

    revalidation_archive = tmp_path / "native-methods-v2-revalidation.n4a"
    shutil.copyfile(archive_path, revalidation_archive)
    replacement = tmp_path / "native-methods-v2-revalidation-tampered.n4a"
    with nirs4all.load_session(revalidation_archive) as revalidated_session:
        before_tamper = revalidated_session.predict(
            {"X": prediction_features, "sample_ids": prediction_ids},
            methods_library_path=library_path,
        )
        assert before_tamper.metadata["sample_ids"] == prediction_ids
        _tamper_n4mm(revalidation_archive, replacement)
        os.replace(replacement, revalidation_archive)
        with pytest.raises(CoreArchiveReplayError, match="validation|replay|refused"):
            revalidated_session.predict(
                {"X": prediction_features, "sample_ids": prediction_ids},
                methods_library_path=library_path,
            )

    child = textwrap.dedent(
        """
        import importlib
        import json
        import os

        class LegacyPathReached:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("fresh Archive V2 replay constructed PipelineRunner")

        run_module = importlib.import_module("nirs4all.api.run")
        predict_module = importlib.import_module("nirs4all.api.predict")
        run_module.PipelineRunner = LegacyPathReached
        predict_module.PipelineRunner = LegacyPathReached

        import nirs4all

        data = {
            "X": json.loads(os.environ["N4A_PREDICT_X"]),
            "sample_ids": json.loads(os.environ["N4A_PREDICT_IDS"]),
        }
        second_data = {
            "X": list(reversed(data["X"])),
            "sample_ids": list(reversed(data["sample_ids"])),
        }
        with nirs4all.load_session(os.environ["N4A_ARCHIVE"]) as loaded:
            from_session = loaded.predict(
                data,
                methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
            )
            second_from_session = loaded.predict(
                second_data,
                engine="native",
                methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
            )
        loaded.close()
        try:
            loaded.predict(
                data,
                methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
            )
        except RuntimeError as error:
            closed_refusal = str(error)
        else:
            raise AssertionError("closed native session accepted another prediction")
        direct = nirs4all.predict(
            model=os.environ["N4A_ARCHIVE"],
            data=data,
            engine="native",
            methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
        )
        print(json.dumps({
            "session_ids": from_session.metadata["sample_ids"],
            "session_target_names": from_session.metadata["target_names"],
            "session_values": from_session.y_pred.tolist(),
            "second_session_ids": second_from_session.metadata["sample_ids"],
            "second_session_values": second_from_session.y_pred.tolist(),
            "direct_ids": direct.metadata["sample_ids"],
            "direct_target_names": direct.metadata["target_names"],
            "direct_values": direct.y_pred.tolist(),
            "closed_refusal": closed_refusal,
        }))
        """
    )
    child_env = os.environ.copy()
    child_env.update(
        N4A_ARCHIVE=str(archive_path),
        N4A_METHODS_LIBRARY=str(library_path),
        N4A_PREDICT_X=json.dumps(prediction_features),
        N4A_PREDICT_IDS=json.dumps(prediction_ids),
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), child_env.get("PYTHONPATH", "")]
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert observed["session_ids"] == prediction_ids
    assert observed["session_target_names"] == ["protein", "moisture"]
    assert observed["second_session_ids"] == list(reversed(prediction_ids))
    assert observed["direct_ids"] == prediction_ids
    assert observed["direct_target_names"] == ["protein", "moisture"]
    assert observed["session_values"] == observed["direct_values"]
    assert observed["second_session_values"] == list(reversed(observed["direct_values"]))
    assert all(len(row) == 2 for row in observed["direct_values"])
    assert "Session is closed" in observed["closed_refusal"]
    assert all(math.isfinite(value) for row in observed["direct_values"] for value in row)

    tampered = tmp_path / "tampered.n4a"
    _tamper_n4mm(archive_path, tampered)
    with pytest.raises(CoreArchiveReplayError, match="validation|replay|refused"):
        nirs4all.predict(
            model=tampered,
            data={"X": prediction_features, "sample_ids": prediction_ids},
            engine="native",
            methods_library_path=library_path,
        )


def _tamper_n4mm(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        member_path = manifest["payloads"]["methods"]["n4mm"][0]["member_path"]
        with zipfile.ZipFile(destination, "x") as altered:
            for info in archive.infolist():
                payload = archive.read(info.filename)
                if info.filename == member_path:
                    payload = payload[:-1] + bytes([payload[-1] ^ 1])
                altered.writestr(info, payload)
