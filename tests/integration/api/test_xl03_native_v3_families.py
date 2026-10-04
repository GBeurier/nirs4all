"""Genuine XL03 native families: trained parent, fresh REFIT, Archive V3 and cold replay.

N4ME is trained by the native DAG witness; the SDK exercises its public
from_package/export/load/predict boundary. RolePipeline additionally exercises
public run/retrain. Captures are outputs of real native execution, never state
fixtures fabricated by Python. An enabled qualification lane must provision
both captures and the installed child; missing prerequisites are failures.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

_REQUIRE = "NIRS4ALL_REQUIRE_XL03_INSTALLED"
_CAPTURE = "NIRS4ALL_XL03_CAPTURE_DIR"
_CHILD = "NIRS4ALL_XL03_INSTALLED_PYTHON"
_LIBRARY = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"
_FAMILIES = ("n4me-pls", "role-pls")
pytestmark = [pytest.mark.methods, pytest.mark.skipif(os.environ.get(_REQUIRE) != "1", reason=f"set {_REQUIRE}=1 with genuine native captures and exact installed wheels")]


def _document(contract: Any) -> dict[str, Any]:
    if isinstance(contract, dict):
        return contract
    serializer = getattr(contract, "json", None)
    assert callable(serializer), "native contract must retain its authoritative JSON serializer"
    return json.loads(serializer())


@pytest.fixture
def runtime() -> tuple[str, Path, str]:
    library, capture, child = (os.environ.get(name) for name in (_LIBRARY, _CAPTURE, _CHILD))
    assert library and Path(library).is_file(), f"{_LIBRARY} must select the exact Methods binary"
    assert capture and Path(capture).is_dir(), f"{_CAPTURE} must contain actual DAG XL03 capture outputs"
    assert child and Path(child).is_file(), f"{_CHILD} must select the fresh installed quartet interpreter"
    return str(Path(library).resolve()), Path(capture), child


def _load_capture(family: str, capture: Path, library: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from nirs4all.api.native_refit_result import NativeMethodsRefitResult

    package_path = capture / f"{family}-package-v3.json"
    oracle_path = capture / f"{family}-oracle-data.json"
    assert package_path.is_file() and oracle_path.is_file(), "native execution captures are mandatory"
    package = json.loads(package_path.read_text())
    parent = json.loads((capture / f"{family}-parent-package-v2.json").read_text())
    assert parent["schema_version"] == 2 and parent["execution_bundle"]["raw_artifact_payloads"]
    assert package["schema_version"] == 3 and package["outcome"]["schema_version"] == 3
    assert package["outcome"]["execution_bundle"]["raw_artifact_payloads"] != parent["execution_bundle"]["raw_artifact_payloads"]
    assert not any(name in package["outcome"] for name in ("score_set", "score_reports", "oof_averages"))
    child = NativeMethodsRefitResult.from_package_json(package_path.read_text(), methods_library_path=library)
    return child, package, json.loads(oracle_path.read_text())


def _archive_bytes(child: Any, archive: Path) -> tuple[dict[str, Any], str]:
    package = _document(child.package)
    assert child.export(archive) == archive
    from nirs4all_core import read_portable_refit_package_v3

    assert json.loads(read_portable_refit_package_v3(archive)) == package
    with zipfile.ZipFile(archive) as container:
        manifest = json.loads(container.read("manifest.json"))
        assert manifest["schema_version"] == 3
        references = manifest["payloads"]["methods"]
        actual_refs = [reference for name in ("n4me", "role_pipelines") for reference in references.get(name, [])]
        artifacts = package["outcome"]["execution_bundle"]["refit_artifacts"]
        raw = package["outcome"]["execution_bundle"]["raw_artifact_payloads"]
        assert len(actual_refs) == len(artifacts) == 1
        ref = actual_refs[0]
        artifact = artifacts[0]["artifact"]
        assert ref["artifact_id"] == artifact["id"] and ref["member_path"] == artifact["uri"]
        payload = container.read(ref["member_path"])
        # The native JSON byte-array owns exactly the same RAW state as Core.
        assert payload == bytes(raw[artifact["id"]])
        assert hashlib.sha256(payload).hexdigest() == ref["raw_sha256"] == artifact["content_fingerprint"]
        entry = next(item for item in manifest["member_inventory"] if item["path"] == ref["member_path"])
        for name in ("semantic_profile", "semantic_fingerprint", "raw_sha256"):
            assert entry[name] == ref[name]
        if artifact["kind"] == "methods_role_pipeline":
            assert artifact["controller_id"] == "controller:methods.native.regression"
            assert artifact["plugin"] == "dagml.methods.native.regression" and artifact["plugin_version"] == "1.0.0"
            assert ref["semantic_profile"] == "dagml_methods_role_pipeline_raw_sha256"
        else:
            assert payload[:4] == b"N4ME" and artifact["abi_major"] == 2 and artifact["abi_min_minor"] == 13
            assert artifact.get("plugin") is None and ref["semantic_profile"] == "n4me_raw_sha256"
    return package, ref["member_path"]


_COLD = r"""
import hashlib, importlib, json, pathlib, sys
import numpy as np
import nirs4all, dag_ml, nirs4all_core, n4m
from nirs4all.pipeline.dagml.native_client import DagMLNativeClient
from n4m._ffi import lib
from n4m.roles import RolePipeline
input = json.loads(sys.stdin.read())
for module in (nirs4all, dag_ml, nirs4all_core, n4m):
    origin = pathlib.Path(module.__file__).resolve()
    assert origin.is_relative_to(pathlib.Path(sys.prefix).resolve()) and 'site-packages' in origin.parts, str(origin)
assert pathlib.Path(lib._name).resolve() == pathlib.Path(sys.argv[2]).resolve()
root = pathlib.Path(nirs4all.__file__).parent
hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in input['sdk_hashes']}
assert hashes == input['sdk_hashes'], 'installed SDK source bytes differ'
def forbidden(*args, **kwargs): raise AssertionError('cold replay attempted FIT/HPO/legacy execution')
for name in ('execute_training','execute_methods_training','execute_methods_portable_full_refit','run_host_hpo_search_in_process'):
    if hasattr(dag_ml,name): setattr(dag_ml,name,forbidden)
for name in ('RolePipeline','Estimator','MultimodalPipeline','MultimodalClassifierPipeline'):
    cls = getattr(n4m,name,None)
    if cls is not None and hasattr(cls,'fit'): cls.fit = forbidden
RolePipeline.fit = forbidden
for name in ('run','retrain'): setattr(nirs4all,name,forbidden)
for module_name in ('nirs4all.api.native_training','nirs4all.api.native_archive_training'):
    module = importlib.import_module(module_name)
    for name in ('fit_native_pipeline','run_native_methods_archive','refit_native_methods'):
        if hasattr(module,name): setattr(module,name,forbidden)
def doc(value): return value if isinstance(value,dict) else json.loads(value.json())
trace=[]
original=DagMLNativeClient.replay_loaded_methods_portable_refit_package_v3
def observe(self,package,request,envelopes,inputs,**kwargs):
    before=doc(request)
    result=original(self,package,request,envelopes,inputs,**kwargs)
    after=doc(request)
    assert before==after and before['phase']=='PREDICT'
    outcome=doc(result)
    assert outcome['lineage'] and all(item['phase']=='PREDICT' for item in outcome['lineage'])
    assert not any('y' in value for value in inputs.values())
    trace.append({'request':before,'lineage':outcome['lineage'],'outputs':outcome['outputs']})
    return result
DagMLNativeClient.replay_loaded_methods_portable_refit_package_v3=observe
loaded=nirs4all.load_session(sys.argv[1],engine='native',methods_library_path=sys.argv[2])
assert loaded.archive_schema_version==3
result=nirs4all.predict(data={'X':np.asarray(input['X']),'sample_ids':input['ids']},session=loaded,engine='native',methods_library_path=sys.argv[2],verbose=0)
assert result.metadata['sample_ids']==input['ids'] and len(trace)==1
assert trace[0]['outputs'][0]['predictions'][0]['sample_ids']==input['ids']
print(json.dumps({'values':result.y_pred.tolist(),'trace':trace,'sdk_hashes':hashes,'origins':{m.__name__:str(pathlib.Path(m.__file__).resolve()) for m in (nirs4all,dag_ml,nirs4all_core,n4m)}}))
"""


def _cold_predict(archive: Path, X: np.ndarray, ids: list[str], runtime: tuple[str, Path, str], tmp_path: Path) -> dict[str, Any]:
    import nirs4all

    library, _capture, child = runtime
    root = Path(nirs4all.__file__).parent
    names = ["api/native_refit_result.py", "api/native_training.py", "pipeline/dagml/raw_replay_lowerer.py", "pipeline/dagml/native_client.py"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    environment.update(N4M_LIB_PATH=library, N4M_LIBRARY_PATH=library)
    completed = subprocess.run([child, "-I", "-B", "-c", _COLD, str(archive), library], input=json.dumps({"X": X.tolist(), "ids": ids, "sdk_hashes": hashes}), cwd=tmp_path, env=environment, text=True, capture_output=True, check=True, timeout=90)
    observed = json.loads(completed.stdout)
    assert observed["sdk_hashes"] == hashes
    return observed


@pytest.mark.parametrize("family", _FAMILIES)
def test_xl03_true_native_refit_capture_exports_exact_raw_and_replays_installed_without_fit(family: str, runtime: tuple[str, Path, str], tmp_path: Path) -> None:
    from nirs4all.api.native_refit_result import NativeMethodsRefitResult

    library, capture, _child = runtime
    child, package, data = _load_capture(family, capture, library)
    snapshot = child.package_json()
    archive = tmp_path / f"{family}.n4a"
    _archive_bytes(child, archive)
    loaded = NativeMethodsRefitResult.load_archive(archive, methods_library_path=library)
    assert _document(loaded.package) == package
    X = np.asarray(data["predict_X"])
    ids = data["predict_ids"]
    oracle = PLSRegression(n_components=data["n_components"], scale=data["scale"]).fit(np.asarray(data["X"]), np.asarray(data["y"])).predict(X).reshape(-1, 1)
    observed = _cold_predict(archive, X, ids, runtime, tmp_path)
    np.testing.assert_allclose(observed["values"], oracle, rtol=1e-10, atol=1e-10)
    assert child.package_json() == snapshot
    request = observed["trace"][0]["request"]
    assert request["source_outcome_fingerprint"] == package["outcome"]["outcome_fingerprint"]


@pytest.mark.parametrize("mode", ("phase_controls", "hpo"))
@pytest.mark.parametrize("root_seed", (0, 41))
def test_xl03_public_role_pipeline_parent_run_fresh_full_refit_and_cold_archive_oracle(mode: str, root_seed: int, runtime: tuple[str, Path, str], tmp_path: Path) -> None:
    import nirs4all
    from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate

    library, _capture, _child = runtime
    rng = np.random.default_rng(3203)
    X = rng.normal(size=(30, 8))
    y = 2.0 * X[:, 0] - 0.7 * X[:, 4]
    terminal: dict[str, Any] = {"model": PLSRegression(n_components=1, scale=True)}
    if mode == "phase_controls":
        terminal["train_params"] = {"n_components": 1, "scale": False}
        terminal["refit_params"] = {"n_components": 2, "scale": True}
    else:
        terminal["finetune_params"] = {
            "engine": "n4m", "n_trials": 2, "sampler": "random", "pruner": "none",
            "approach": "grouped", "seed": 6, "metric": "rmse", "direction": "minimize",
            "model_params": {"n_components": ["int", 1, 3, 1]},
        }
    pipeline = [KFold(3), StandardNormalVariate(), SavitzkyGolay(window_length=5, polyorder=2), terminal]
    with nirs4all.run(pipeline, {"X": X, "y": y, "sample_ids": [f"parent.{i}" for i in range(30)]}, engine="native", native_profile="n4m.pls_role_pipeline.v1", methods_library_path=library, random_state=root_seed, save_charts=False, verbose=0) as parent:
        parent_package = _document(parent._native_package_contract)
        assert parent_package["effective_plan"]["campaign"]["root_seed"] == root_seed
        parent_snapshot = json.dumps(parent_package, sort_keys=True)
        if mode == "hpo":
            trials = parent._native_outcome["methods_hpo_resume_state"]["terminal_trials"]
            assert len(trials) == 2 and all(entry["trial"]["status"] == "completed" for entry in trials)
        target_X, target_y = X * 0.8 + 0.25, 1.5 * y - 0.4
        child = nirs4all.retrain(parent, {"X": target_X, "y": target_y, "sample_ids": [f"target.{i}" for i in range(30)]}, engine="native", mode="full", verbose=0)
        assert json.dumps(_document(parent._native_package_contract), sort_keys=True) == parent_snapshot
    package, _raw_path = _archive_bytes(child, tmp_path / "public-role.n4a")
    assert package["outcome"]["effective_plan"]["campaign"]["root_seed"] == root_seed
    model_nodes = [node for node in package["outcome"]["effective_plan"]["node_plans"].values() if node["kind"] == "model"]
    assert len(model_nodes) == 1 and model_nodes[0]["controller_id"] == "controller:methods.native.regression"
    prediction_X = target_X[[12, 3, 21]] + 0.1
    def transform(values: np.ndarray) -> np.ndarray:
        snv = (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, keepdims=True)
        return savgol_filter(snv, 5, 2, axis=1, mode="interp")
    params = model_nodes[0]["params"]
    controls = params.get("phase_controls", {}).get("refit_params", {})
    components = controls.get("n_components", params["n_components"])
    if mode == "phase_controls":
        assert components == 2 and controls["scale"] is True
    else:
        assert components in (1, 2, 3)
    oracle = PLSRegression(n_components=components, scale=True).fit(transform(target_X), target_y).predict(transform(prediction_X)).reshape(-1, 1)
    observed = _cold_predict(tmp_path / "public-role.n4a", prediction_X, ["prediction.c", "prediction.a", "prediction.b"], runtime, tmp_path)
    np.testing.assert_allclose(observed["values"], oracle, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("mutation", ("raw_state", "missing_member", "cross_reference"))
def test_xl03_actual_native_archive_tampering_refuses_before_replay(family: str, mutation: str, runtime: tuple[str, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.api.native_refit_result import NativeMethodsRefitResult
    from nirs4all.pipeline.dagml.native_client import DagMLNativeClient

    library, capture, _child = runtime
    child, _package, _data = _load_capture(family, capture, library)
    archive = tmp_path / "original.n4a"
    _package, raw_path = _archive_bytes(child, archive)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as container:
        members = {name: container.read(name) for name in container.namelist()}
    if mutation == "raw_state":
        raw = bytearray(members[raw_path])
        raw[-1] ^= 1
        members[raw_path] = bytes(raw)
    elif mutation == "missing_member":
        del members[raw_path]
    else:
        manifest = json.loads(members["manifest.json"])
        name = "n4me" if family == "n4me-pls" else "role_pipelines"
        manifest["payloads"]["methods"][name][0]["raw_sha256"] = "0" * 64
        members["manifest.json"] = json.dumps(manifest).encode()
    forged = tmp_path / "forged.n4a"
    with zipfile.ZipFile(forged, "w") as container:
        for name, payload in members.items():
            container.writestr(name, payload)
    calls: list[bool] = []
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        raise AssertionError("invalid archive reached native replay")
    monkeypatch.setattr(DagMLNativeClient, "replay_loaded_methods_portable_refit_package_v3", forbidden)
    with pytest.raises(ValueError, match="validation refused"):
        NativeMethodsRefitResult.load_archive(forged, methods_library_path=library)
    assert not calls and hashlib.sha256(archive.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("family", _FAMILIES)
def test_xl03_genuine_package_and_archive_collision_preserve_existing_bytes_and_failed_new_write_is_removed(family: str, runtime: tuple[str, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library, capture, _child = runtime
    child, _package, _data = _load_capture(family, capture, library)
    package_path = child.save_package(tmp_path / "package.json")
    before = package_path.read_bytes()
    with pytest.raises(FileExistsError):
        child.save_package(package_path)
    assert package_path.read_bytes() == before
    archive = tmp_path / "archive.n4a"
    _archive_bytes(child, archive)
    before_archive = archive.read_bytes()
    with pytest.raises(ValueError):
        child.export(archive)
    assert archive.read_bytes() == before_archive
    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync refusal")
    module = importlib.import_module("nirs4all.api.native_refit_result")
    monkeypatch.setattr(module.os, "fsync", fail_fsync)
    failed = tmp_path / "failed-new-package.json"
    with pytest.raises(OSError, match="injected fsync refusal"):
        child.save_package(failed)
    assert not failed.exists() and package_path.read_bytes() == before
