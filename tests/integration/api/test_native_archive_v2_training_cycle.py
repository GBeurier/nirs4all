"""Opt-in real run/save/close/load/predict proof for Archive V2/N4MM."""

from __future__ import annotations

import importlib
import json
import math
import os
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
from nirs4all.pipeline.dagml.core_archive_replay import CoreArchiveReplayError

_REQUIRE_ENV = "NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2"
_LIBRARY_ENV = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"


def _pipeline() -> list[object]:
    return [KFold(n_splits=3), {"model": PLSRegression(n_components=1)}]


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

    original_runner = run_module.PipelineRunner
    run_module.PipelineRunner = LegacyPathReached
    try:
        result = nirs4all.run(
            _pipeline(),
            _dataset(),
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
        assert result.native_execution_is_live is False
    finally:
        run_module.PipelineRunner = original_runner

    prediction_features = [[8.0, 0.0], [-1.0, 4.0]]
    prediction_ids = ["predict.z", "predict.a"]
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
        with nirs4all.load_session(os.environ["N4A_ARCHIVE"]) as loaded:
            from_session = loaded.predict(
                data,
                methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
            )
        direct = nirs4all.predict(
            model=os.environ["N4A_ARCHIVE"],
            data=data,
            engine="native",
            methods_library_path=os.environ["N4A_METHODS_LIBRARY"],
        )
        print(json.dumps({
            "session_ids": from_session.metadata["sample_ids"],
            "session_values": from_session.y_pred.tolist(),
            "direct_ids": direct.metadata["sample_ids"],
            "direct_values": direct.y_pred.tolist(),
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
    assert observed["direct_ids"] == prediction_ids
    assert observed["session_values"] == observed["direct_values"]
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
