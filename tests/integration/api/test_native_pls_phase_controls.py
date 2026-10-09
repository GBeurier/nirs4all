"""Mandatory opt-in native phase execution, closed RAW inspection and detached replay."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate

PROFILE = "n4m.pls_role_pipeline.v1"
_REQUIRE_ENV = "NIRS4ALL_REQUIRE_NATIVE_PLS_PHASE_CONTROLS"
_LIBRARY_ENV = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"
pytestmark = [
    pytest.mark.methods,
    pytest.mark.skipif(os.environ.get(_REQUIRE_ENV) != "1", reason=f"set {_REQUIRE_ENV}=1 with matching installed native wheels"),
]


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> str:
    selected = os.environ.get(_LIBRARY_ENV)
    assert selected and Path(selected).is_file(), f"{_LIBRARY_ENV} must select the exact Methods library"
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    return str(Path(selected).resolve())


def _dataset() -> dict[str, Any]:
    # Unequal feature scales make PLS's scale flags scientifically observable.
    rng = np.random.default_rng(923)
    latent = rng.normal(size=(36, 8))
    features = latent * np.array([0.02, 30.0, 0.4, 9.0, 0.07, 70.0, 1.3, 4.0])
    targets = 2.8 * latent[:, 0] - 1.7 * latent[:, 2] + 0.8 * latent[:, 6] + 0.025 * rng.normal(size=36)
    return {"X": features, "y": targets, "sample_ids": [f"train.{i:02d}" for i in range(36)], "target_names": ["response"]}


def _pipeline(*, components: int = 1, scale: bool = True, smooth: bool = False,
              train: dict[str, Any] | None = None, refit: dict[str, Any] | None = None,
              trials: int | None = None, space: dict[str, Any] | None = None,
              resume: dict[str, Any] | None = None, fold_seed: int = 17) -> list[Any]:
    terminal: dict[str, Any] = {"model": PLSRegression(n_components=components, scale=scale)}
    if train is not None:
        terminal["train_params"] = train
    if refit is not None:
        terminal["refit_params"] = refit
    if trials is not None:
        terminal["finetune_params"] = {
            "engine": "n4m", "n_trials": trials, "sampler": "random", "pruner": "none",
            "approach": "grouped", "seed": 6, "metric": "rmse", "direction": "minimize",
            "model_params": space if space is not None else {"n_components": ["int", 1, 3], "scale": [False, True]},
        }
        if resume is not None:
            terminal["finetune_params"]["resume_package"] = resume
    transforms = [StandardNormalVariate(), SavitzkyGolay(window_length=5, polyorder=2)] if smooth else []
    return [KFold(n_splits=3, shuffle=True, random_state=fold_seed), *transforms, terminal]


def _run(pipeline: list[Any], dataset: dict[str, Any], library: str) -> Any:
    import nirs4all

    return nirs4all.run(pipeline, dataset, engine="native", native_profile=PROFILE, save_charts=False, verbose=0, methods_library_path=library)


_ORACLE = r'''
import json, pathlib, sys
import numpy as np
import n4m
from n4m._ffi import lib
from n4m.roles import RolePipeline

data = json.loads(sys.stdin.read())
assert pathlib.Path(lib._name).resolve() == pathlib.Path(sys.argv[1]).resolve()
X, y = np.asarray(data["X"], dtype=float), np.asarray(data["y"], dtype=float)
def recipe(n, scale):
    steps = []
    if data["smooth"]:
        steps += [("preprocessing.scatter.snv", {"with_mean": True, "with_std": True, "ddof": 0}),
                  ("preprocessing.derivatives.savitzky_golay", {"window_length": 5, "polyorder": 2, "deriv": 0, "delta": 1.0, "mode": "interp", "cval": 0.0})]
    return steps + [("models.pls.pls_regression", {"n_components": n, "solver": "nipals", "center_x": True, "center_y": True, "scale_x": scale, "scale_y": scale})]
oof = np.empty_like(y)
for train, validation in data["folds"]:
    model = RolePipeline(recipe(data["train"]["n_components"], data["train"]["scale"])).fit(X[train], y[train])
    oof[validation] = np.asarray(model.predict(X[validation])).reshape(-1)
    del model
model = RolePipeline(recipe(data["refit"]["n_components"], data["refit"]["scale"])).fit(X, y)
heldout = model.predict(np.asarray(data["heldout"], dtype=float)).reshape(-1, 1)
print(json.dumps({"oof": oof.tolist(), "heldout": heldout.tolist(), "version": n4m.version(), "library": str(pathlib.Path(lib._name).resolve())}))
'''


def _oracle(dataset: dict[str, Any], library: str, tmp_path: Path, *, train: dict[str, Any],
            refit: dict[str, Any], heldout: np.ndarray, smooth: bool = False) -> dict[str, Any]:
    folds = [(a.tolist(), b.tolist()) for a, b in KFold(3, shuffle=True, random_state=17).split(dataset["X"])]
    data = {"X": dataset["X"].tolist(), "y": dataset["y"].tolist(), "folds": folds,
            "train": train, "refit": refit, "heldout": heldout.tolist(), "smooth": smooth}
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _ORACLE, library], input=json.dumps(data), cwd=tmp_path,
        env={**os.environ, "N4M_LIB_PATH": library}, capture_output=True, text=True, check=True, timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["library"] == library
    # Qualified public Methods wheels retain their exact native version and ABI.
    assert result["version"] in {"1.2.1+abi.2.14.0", "1.2.1+abi.2.15.0", "1.2.1+abi.2.16.0", "1.2.1+abi.2.17.0", "1.3.2+abi.2.17.0", "1.3.4+abi.2.17.0"}
    return result


def _oof(result: Any, dataset: dict[str, Any]) -> np.ndarray:
    # DAG exposes both ordinary (avg) and weighted (w_avg) OOF summaries.
    # The public CV score and this independent KFold oracle use ordinary OOF.
    blocks = [block for block in result._native_outcome["oof_averages"] if block["predictions"]["fold_id"] == "avg"]
    assert len(blocks) == 1
    block = blocks[0]["predictions"]
    assert block["partition"] == "validation" and block["level"] == "sample"
    assert block["target_names"] == ["response"]
    ids = [unit["id"] for unit in block["unit_ids"]]
    assert len(set(ids)) == len(dataset["sample_ids"]) and set(ids) == set(dataset["sample_ids"])
    by_id = dict(zip(ids, block["values"], strict=True))
    targets = blocks[0]["y_true"]
    assert targets["target_names"] == block["target_names"]
    true_by_id = dict(zip([unit["id"] for unit in targets["unit_ids"]], targets["values"], strict=True))
    np.testing.assert_array_equal(np.asarray([true_by_id[identifier] for identifier in dataset["sample_ids"]]).reshape(-1), dataset["y"])
    return np.asarray([by_id[identifier] for identifier in dataset["sample_ids"]]).reshape(-1)


def _inspect(archive: Path, library: str, expected: dict[str, Any], *, steps: int) -> dict[str, Any]:
    import nirs4all

    inspection = nirs4all.inspect_portable_predictor_archive_v2(archive, methods_library_path=library)
    assert inspection["native_profile"] == PROFILE and inspection["training_performed"] is False
    assert Path(inspection["archive_path"]).resolve() == archive.resolve()
    assert len(inspection["models"]) == 1
    model = inspection["models"][0]
    assert model["native_profile"] == PROFILE and model["model_params"] == expected
    assert model["target_names"] == ["response"] and len(model["feature_names"]) == 8
    assert len(model["params_fingerprint"]) == 64 and model["node_id"]
    assert len(model["steps"]) == steps
    params = model["steps"][-1]["params"]
    assert model["steps"][-1]["methodId"] == "models.pls.pls_regression"
    assert params["n_components"] == expected["n_components"]
    assert params["scale_x"] is expected["scale"] and params["scale_y"] is expected["scale"]
    assert params["center_x"] is True and params["center_y"] is True
    assert params["solver"] == "nipals"
    if steps == 3:
        assert [step["methodId"] for step in model["steps"][:2]] == [
            "preprocessing.scatter.snv", "preprocessing.derivatives.savitzky_golay",
        ]
        smooth_params = model["steps"][1]["params"]
        assert smooth_params["window_length"] == 5 and smooth_params["polyorder"] == 2
        assert smooth_params["deriv"] == 0 and smooth_params["mode"] == "interp"
    with zipfile.ZipFile(archive) as container:
        manifest = json.loads(container.read("manifest.json"))
    assert len(manifest["payloads"]["methods"]["role_pipelines"]) == 1
    assert not manifest["payloads"]["methods"].get("n4mm")
    return inspection


_REPLAY = r'''
import hashlib, importlib, json, pathlib, sys
import nirs4all, dag_ml
archive, library = sys.argv[1:3]
data = json.loads(sys.stdin.read())
def forbidden(*args, **kwargs):
    raise AssertionError("detached prediction attempted FIT, HPO or legacy execution")
for name in ("execute_methods_training", "execute_training"):
    if hasattr(dag_ml, name): setattr(dag_ml, name, forbidden)
training = importlib.import_module("nirs4all.api.native_archive_training")
training.run_native_methods_archive = forbidden
run = importlib.import_module("nirs4all.api.run")
run.run = forbidden
if hasattr(run, "PipelineRunner"): run.PipelineRunner = forbidden
nirs4all.run = forbidden
try:
    from n4m.roles import RolePipeline
    RolePipeline.fit = forbidden
except ImportError:
    raise AssertionError("native Methods oracle facade is missing")
root = pathlib.Path(nirs4all.__file__).parent
assert "site-packages" in root.parts, "fresh replay requires the installed SDK wheel"
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in data["source_hashes"]}
assert hashes == data["source_hashes"], "fresh process loaded different SDK production bytes"
inspection = nirs4all.inspect_portable_predictor_archive_v2(archive, methods_library_path=library)
prediction = nirs4all.predict(archive, {"X": data["X"], "sample_ids": data["sample_ids"]}, engine="native", methods_library_path=library, verbose=0)
assert inspection["training_performed"] is False
assert prediction.metadata["training_performed"] is False
assert prediction.metadata["native_profile"] == "n4m.pls_role_pipeline.v1"
print(json.dumps({"predictions": prediction.y_pred.tolist(), "sample_ids": prediction.metadata["sample_ids"], "target_names": prediction.metadata["target_names"], "model_params": inspection["models"][0]["model_params"], "sdk_origin": str(root.resolve()), "source_hashes": hashes}))
'''


def _fresh_predict(archive: Path, library: str, heldout: np.ndarray, tmp_path: Path) -> dict[str, Any]:
    import nirs4all

    root = Path(nirs4all.__file__).parent
    files = [
        "__init__.py", "api/__init__.py", "api/run.py", "api/portable_archive.py", "api/native_archive_training.py",
        "pipeline/dagml/native_pls_phase_controls.py", "pipeline/dagml/native_pls_phase_replay.py",
        "pipeline/dagml/core_archive_replay.py", "pipeline/dagml/raw_training_lowerer.py", "pipeline/dagml/raw_replay_lowerer.py",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    data = {"X": heldout.tolist(), "sample_ids": [f"heldout.new.{i}" for i in range(len(heldout))], "source_hashes": hashes}
    reference = nirs4all.predict(
        archive, {"X": heldout, "sample_ids": data["sample_ids"]}, engine="native", methods_library_path=library, verbose=0,
    )
    assert reference.metadata["training_performed"] is False
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["N4M_LIB_PATH"] = library
    executable = os.environ.get("NIRS4ALL_NATIVE_PLS_INSTALLED_PYTHON", sys.executable)
    assert Path(executable).is_file(), "fresh-process qualification Python must exist"
    completed = subprocess.run(
        [executable, "-I", "-B", "-c", _REPLAY, str(archive), library], input=json.dumps(data), cwd=tmp_path,
        env=environment, check=True, capture_output=True, text=True, timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["sample_ids"] == data["sample_ids"] and result["target_names"] == ["response"]
    np.testing.assert_array_equal(result["predictions"], reference.y_pred)
    return result


@pytest.mark.parametrize("smooth", [False, True])
def test_train_controls_cv_and_refit_controls_saved_raw_and_fresh_prediction(
    library: str, tmp_path: Path, smooth: bool,
) -> None:
    dataset = _dataset()
    heldout = dataset["X"][[1, 11, 25]] * 0.83 + 0.017
    train = {"n_components": 2, "scale": False}
    refit = {"n_components": 3, "scale": True}
    oracle = _oracle(dataset, library, tmp_path, train=train, refit=refit, heldout=heldout, smooth=smooth)
    with _run(_pipeline(train=train, refit=refit, smooth=smooth), dataset, library) as result:
        np.testing.assert_allclose(_oof(result, dataset), oracle["oof"], atol=1.0e-11, rtol=1.0e-11)
        assert result.cv_best_score == pytest.approx(np.sqrt(np.mean((np.asarray(oracle["oof"]) - dataset["y"]) ** 2)), abs=1.0e-11)
        phase_patches = {patch["path"][0]: patch for patch in result._native_outcome["parameter_patches"] if patch["namespace"] == "fit"}
        assert phase_patches["train_params"]["value"] == train
        assert phase_patches["refit_params"]["value"] == refit
        assert result._native_outcome["training_request_fingerprint"]
        assert result._native_package["execution_bundle"]["refit_artifacts"]
        archive = Path(result.export(tmp_path / "phase-winner.n4a"))
        _inspect(archive, library, refit, steps=3 if smooth else 1)
    assert result.native_execution_is_live is False
    replay = _fresh_predict(archive, library, heldout, tmp_path)
    assert replay["model_params"] == refit
    np.testing.assert_allclose(replay["predictions"], oracle["heldout"], atol=1.0e-11, rtol=1.0e-11)


def test_scale_changes_actual_cv_predictions_not_only_saved_metadata(library: str, tmp_path: Path) -> None:
    dataset = _dataset()
    observed = []
    for scale in (False, True):
        params = {"n_components": 1, "scale": scale}
        oracle = _oracle(dataset, library, tmp_path, train=params, refit=params, heldout=dataset["X"][:2])
        with _run(_pipeline(scale=scale), dataset, library) as result:
            values = _oof(result, dataset)
            np.testing.assert_allclose(values, oracle["oof"], atol=1.0e-11, rtol=1.0e-11)
            archive = Path(result.export(tmp_path / f"scale-{scale}.n4a"))
            _inspect(archive, library, params, steps=1)
            observed.append(values)
    assert np.max(np.abs(observed[0] - observed[1])) > 0.1


@pytest.mark.parametrize("space", [{"n_components": ["int", 1, 3]}, {"scale": [False, True]},
                                  {"n_components": ["int", 1, 3], "scale": [False, True]}])
def test_trial_params_are_executed_and_explicit_refit_wins(
    library: str, tmp_path: Path, space: dict[str, Any],
) -> None:
    dataset = _dataset()
    heldout = dataset["X"][[4, 8]] * 1.07
    refit = {"n_components": 3, "scale": False}
    with _run(_pipeline(trials=3, space=space, refit=refit), dataset, library) as result:
        best = result.tuning_best_params
        assert set(best) == {f"model.{name}" for name in space}
        if "scale" in space:
            assert type(best["model.scale"]) is bool
        if "n_components" in space:
            assert type(best["model.n_components"]) is int and 1 <= best["model.n_components"] <= 3
        train = {"n_components": best.get("model.n_components", 1), "scale": best.get("model.scale", True)}
        oracle = _oracle(dataset, library, tmp_path, train=train, refit=refit, heldout=heldout)
        np.testing.assert_allclose(_oof(result, dataset), oracle["oof"], atol=1.0e-11, rtol=1.0e-11)
        assert result.tuning_best_value == pytest.approx(np.sqrt(np.mean((np.asarray(oracle["oof"]) - dataset["y"]) ** 2)), abs=1.0e-11)
        state = result._native_outcome["methods_hpo_resume_state"]
        assert state["trial_history_len"] == 3
        assert all(entry["trial"]["status"] == "completed" for entry in state["terminal_trials"])
        archive = Path(result.export(tmp_path / "trial-refit.n4a"))
        _inspect(archive, library, refit, steps=1)
    replay = _fresh_predict(archive, library, heldout, tmp_path)
    np.testing.assert_allclose(replay["predictions"], oracle["heldout"], atol=1.0e-11, rtol=1.0e-11)


def test_train_nonsearch_key_and_trial_axis_are_both_effective(library: str, tmp_path: Path) -> None:
    dataset = _dataset()
    with _run(_pipeline(train={"n_components": 2}, trials=3, space={"scale": [False, True]}), dataset, library) as result:
        scale = result.tuning_best_params["model.scale"]
        assert type(scale) is bool
        params = {"n_components": 2, "scale": scale}
        oracle = _oracle(dataset, library, tmp_path, train=params, refit=params, heldout=dataset["X"][:2])
        np.testing.assert_allclose(_oof(result, dataset), oracle["oof"], atol=1.0e-11, rtol=1.0e-11)
        _inspect(Path(result.export(tmp_path / "fixed-components.n4a")), library, params, steps=1)


def test_checkpoint_budget_extension_matches_uninterrupted_campaign(library: str, tmp_path: Path) -> None:
    dataset = _dataset()
    refit = {"n_components": 3, "scale": False}
    with _run(_pipeline(trials=2, refit=refit), dataset, library) as interrupted:
        resume = interrupted.tuning_resume_package
        assert resume is not None
    with _run(_pipeline(trials=4, refit=refit, resume=resume), dataset, library) as continued:
        with _run(_pipeline(trials=4, refit=refit), dataset, library) as uninterrupted:
            assert continued.tuning_best_params == uninterrupted.tuning_best_params
            assert continued.tuning_best_value == uninterrupted.tuning_best_value
            np.testing.assert_array_equal(_oof(continued, dataset), _oof(uninterrupted, dataset))
            resumed_trials = continued._native_outcome["methods_hpo_resume_state"]["terminal_trials"]
            full_trials = uninterrupted._native_outcome["methods_hpo_resume_state"]["terminal_trials"]
            assert len(resumed_trials) == len(full_trials) == 4
            assert [(entry["trial"]["parameters"], entry["trial"]["status"]) for entry in resumed_trials] == [
                (entry["trial"]["parameters"], entry["trial"]["status"]) for entry in full_trials
            ]
        _inspect(Path(continued.export(tmp_path / "resumed.n4a")), library, refit, steps=1)


@pytest.mark.parametrize("change", ["train", "refit", "recipe", "data", "folds", "profile"])
def test_checkpoint_cannot_resume_a_changed_scientific_contract(library: str, change: str) -> None:
    import dag_ml

    import nirs4all

    dataset = _dataset()
    original_pipeline = (
        _pipeline(trials=2, space={"n_components": ["int", 1, 3]}) if change == "profile" else
        _pipeline(trials=2, space={"scale": [False, True]}, train={"n_components": 2}, refit={"scale": False})
    )
    with _run(original_pipeline, dataset, library) as original:
        resume = original.tuning_resume_package
        assert resume is not None
    options: dict[str, Any] = {"trials": 3, "space": {"scale": [False, True]}, "train": {"n_components": 2}, "refit": {"scale": False}, "resume": copy.deepcopy(resume)}
    if change == "train":
        options["train"] = {"n_components": 3}
    if change == "refit":
        options["refit"] = {"scale": True}
    if change == "recipe":
        options["smooth"] = True
    if change == "folds":
        options["fold_seed"] = 18
    if change == "data":
        dataset["X"] = dataset["X"].copy()
        dataset["X"][0, 0] += 0.01
    with pytest.raises((ValueError, RuntimeError, dag_ml.DagMlRuntimeError), match="resume|checkpoint"):
        if change == "profile":
            # A historical v1 operation cannot consume a v2 role-pipeline checkpoint.
            old_pipeline = _pipeline(trials=3, space={"n_components": ["int", 1, 3]}, resume=resume)
            nirs4all.run(old_pipeline, dataset, engine="native", save_charts=False, verbose=0, methods_library_path=library)
        else:
            _run(_pipeline(**options), dataset, library)


def test_historical_default_archive_remains_n4mm(library: str, tmp_path: Path) -> None:
    import nirs4all

    with nirs4all.run(_pipeline(), _dataset(), engine="native", save_charts=False, verbose=0, methods_library_path=library) as result:
        archive = result.export(tmp_path / "historical-profile.n4a")
    with zipfile.ZipFile(archive) as container:
        manifest = json.loads(container.read("manifest.json"))
    assert len(manifest["payloads"]["methods"]["n4mm"]) == 1
    assert not manifest["payloads"]["methods"].get("role_pipelines")


def test_detached_predictor_uses_saved_state_after_training_arrays_are_gone(library: str, tmp_path: Path) -> None:
    dataset = _dataset()
    heldout = dataset["X"][[7, 20]] * 0.91
    params = {"n_components": 2, "scale": False}
    oracle = _oracle(dataset, library, tmp_path, train=params, refit=params, heldout=heldout)
    with _run(_pipeline(components=2, scale=False), dataset, library) as result:
        archive = Path(result.export(tmp_path / "independent.n4a"))
    assert result.native_execution_is_live is False
    del dataset
    replay = _fresh_predict(archive, library, heldout, tmp_path)
    np.testing.assert_allclose(replay["predictions"], oracle["heldout"], atol=1.0e-11, rtol=1.0e-11)
