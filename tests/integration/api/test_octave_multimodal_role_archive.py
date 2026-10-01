"""Real Octave HPO through public SDK APIs and installed archive replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

import nirs4all

pytestmark = [pytest.mark.methods]
_REQUIRED = "NIRS4ALL_REQUIRE_OCTAVE_ROLE_CAMPAIGN"
_SDK_FILES = ("__init__.py", "api/__init__.py", "api/dagml_training.py", "api/portable_archive.py")


def _load_campaign(dag_root: Path) -> Any:
    path = dag_root / "scripts/qualify_multimodal_methods_hpo_octave.py"
    spec = importlib.util.spec_from_file_location("octave_role_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime() -> dict[str, Any]:
    values = {
        "capture": os.environ.get("NIRS4ALL_OCTAVE_ROLE_NODE_CAPTURE"),
        "octave": os.environ.get("DAG_ML_OCTAVE"),
        "installed_python": os.environ.get("NIRS4ALL_OCTAVE_ROLE_INSTALLED_PYTHON"),
        "mex": os.environ.get("DAG_ML_OCTAVE_MEX_PATH"),
    }
    dag_root = Path(os.environ.get("NIRS4ALL_DAG_ML_ROOT", str(Path(__file__).resolve().parents[4] / "dag-ml")))
    problems = [f"missing {name}" for name, value in values.items() if not value]
    for name, value in values.items():
        if value and not Path(value).exists():
            problems.append(f"{name} does not exist: {value}")
    script = dag_root / "scripts/qualify_multimodal_methods_hpo_octave.py"
    if not script.is_file():
        problems.append(f"Octave campaign helper does not exist: {script}")
    if problems:
        message = "; ".join(problems)
        if os.environ.get(_REQUIRED) == "1":
            pytest.fail("Mandatory real Octave SDK qualification: " + message)
        pytest.skip("Set real Octave/MEX, fresh Node capture and installed Python paths: " + message)
    return {**{name: Path(value).resolve() for name, value in values.items()},
            "dag_root": dag_root.resolve(), "module": _load_campaign(dag_root)}


@pytest.fixture(scope="module")
def campaign(runtime: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    work = tmp_path_factory.mktemp("sdk-octave-role")
    capture = json.loads(runtime["capture"].read_text())
    calls: list[str] = []
    # Observe real delegates; no scientific call or result is replaced.
    original_hpo, original_training = nirs4all.run_host_hpo_search, nirs4all.execute_training

    def hpo(*args: Any, **kwargs: Any) -> Any:
        calls.append("HPO")
        return original_hpo(*args, **kwargs)

    def train(*args: Any, **kwargs: Any) -> Any:
        calls.append("TRAIN")
        return original_training(*args, **kwargs)

    data = runtime["module"].train_from_python_api(
        runtime["capture"], runtime["octave"], work / "training", hpo_search=hpo, execute_training=train,
    )
    assert calls == ["HPO", "TRAIN"], "Both scientific operations must use the public SDK facade"
    archive = work / "five-octave-models.n4a"
    reference = nirs4all.write_portable_predictor_archive_v2(
        archive, archive_id="archive:sdk.octave.four-sources", outcome=data["outcome"], package=data["package"],
    )
    assert reference["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert nirs4all.read_portable_predictor_archive_v2(archive).to_dict() == data["package"].to_dict()
    return {**data, "runtime": runtime, "capture": capture, "work": work, "archive": archive, "reference": reference}


_FRESH_REPLAY = r'''
import hashlib, importlib.util, json, os, pathlib, sys
from collections import Counter
import nirs4all
import dag_ml
import dag_ml._dag_ml as native
context = json.loads(sys.stdin.read())
sdk = pathlib.Path(nirs4all.__file__).resolve().parent
assert not sdk.is_relative_to(pathlib.Path(context["sdk_checkout"]).resolve())
hashes = {name: hashlib.sha256((sdk / name).read_bytes()).hexdigest() for name in context["source_hashes"]}
assert hashes == context["source_hashes"], "Installed SDK differs from the qualified checkout"
assert hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest() == context["dag_native_sha256"]
def forbidden(*args, **kwargs): raise AssertionError("Fresh replay attempted FIT or HPO")
nirs4all.execute_training = dag_ml.execute_training = forbidden
nirs4all.run_host_hpo_search = dag_ml.run_host_hpo_search_in_process = forbidden
spec = importlib.util.spec_from_file_location("octave_replay", pathlib.Path(context["dag_root"]) / "scripts/qualify_multimodal_methods_hpo_octave.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
work = pathlib.Path.cwd()
prepared = module.prepare_octave(pathlib.Path(context["octave"]), work, "fresh-installed", context["heldout"], context["operators"], None)
assert prepared["manifest"] == context["trusted_manifest"]
with module.OctaveWorker(prepared) as worker:
    outcome = nirs4all.replay_portable_predictor_archive_v2(
        sys.argv[1], context["replay_request"], context["replay_envelopes"], [context["trusted_manifest"]], worker.operator,
        outcome_id="outcome:sdk.octave.fresh", run_id="run:sdk.octave.fresh", artifact_callback=worker.artifact).to_dict()
    callbacks = Counter(item["operation"] for item in worker.calls)
    assert callbacks["hydrate"] == callbacks["PREDICT"] == callbacks["release"] == 5
    assert not worker.hydrated and not callbacks["FIT_CV"] and not callbacks["REFIT"] and not callbacks["export"]
events = [json.loads(line) for line in pathlib.Path(prepared["audit_path"]).read_text().splitlines()]
lifecycle = Counter(event["operation"] for event in events)
assert lifecycle["hydrate"] == lifecycle["PREDICT"] == lifecycle["release"] == lifecycle["dispose"] == 5
assert lifecycle["fit"] == 0
print(json.dumps({"outcome": outcome, "source_hashes": hashes, "sdk": str(sdk), "callbacks": dict(callbacks), "lifecycle": dict(lifecycle)}))
'''


def test_public_octave_hpo_five_raw_models_and_installed_replay_without_fit(campaign: dict[str, Any], tmp_path: Path) -> None:
    import dag_ml._dag_ml as dag_native

    capture, search = campaign["capture"], campaign["search"]
    assert len(search["trials"]) == 3
    assert search["selected_trial_index"] == capture["resumed"]["selected_trial_index"]
    for actual, expected in zip(search["trials"], capture["resumed"]["trials"], strict=True):
        assert actual["params"] == expected["params"]
        assert actual["score"] == pytest.approx(expected["score"], rel=1e-8, abs=1e-8)
        campaign["runtime"]["module"].compare_scores(actual["scores"], expected["scores"], 1e-8)
    package = campaign["package"].to_dict()
    bundle = package["execution_bundle"]
    records, raw = bundle["refit_artifacts"], bundle["raw_artifact_payloads"]
    assert len(records) == len(raw) == len(package["artifact_bindings"]) == 5
    assert {record["node_id"] for record in records} == {"model:" + name for name in (*capture["sourceRows"], "meta")}
    expected_package = json.loads(capture["completeArchive"]["packageJson"])
    expected_saved = {entry["node_id"]: json.loads(bytes(expected_package["execution_bundle"]["raw_artifact_payloads"][entry["artifact"]["id"]]))
                      for entry in expected_package["execution_bundle"]["refit_artifacts"]}
    for record in records:
        artifact = record["artifact"]
        assert artifact["kind"] == "methods_role_pipeline" and artifact["plugin"] == "dagml.methods.octave.regression"
        payload = bytes(raw[artifact["id"]])
        assert hashlib.sha256(payload).hexdigest() == artifact["content_fingerprint"]
        saved = json.loads(payload)
        assert set(saved) == {"schema", "node_id", "params_fingerprint", "target_names", "steps", "feature_names", "states"}
        assert saved["schema"] == "dagml.methods.regression.v1" and saved["node_id"] == record["node_id"]
        assert saved["params_fingerprint"] == record["params_fingerprint"]
        assert saved["target_names"] == expected_saved[record["node_id"]]["target_names"]
        assert saved["steps"] == expected_saved[record["node_id"]]["steps"] == [{"class": "n4m:models.regularized.ridge", "params": {"alpha": 0.05}}]
        assert saved["feature_names"] == expected_saved[record["node_id"]]["feature_names"]
        assert len(saved["states"]) == 1 and bytes(saved["states"][0]).startswith(b"N4ME")
    with ZipFile(campaign["archive"]) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 11
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["payloads"]["methods"]["n4mm"] == []
        assert len(manifest["payloads"]["methods"]["role_pipelines"]) == 5
    source_root = Path(nirs4all.__file__).resolve().parent
    context = {
        "dag_root": str(campaign["runtime"]["dag_root"]), "octave": str(campaign["runtime"]["octave"]),
        "sdk_checkout": str(source_root), "heldout": campaign["heldout"], "operators": campaign["operators"],
        "replay_request": campaign["replay_request"].json(), "replay_envelopes": campaign["replay_envelopes"],
        "trusted_manifest": campaign["runtime"]["module"].OCTAVE_MANIFEST,
        "source_hashes": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in _SDK_FILES},
        "dag_native_sha256": hashlib.sha256(Path(dag_native.__file__).read_bytes()).hexdigest(),
    }
    # Only current heldout rows and signed contracts reach the consumer.
    assert "sourceRows" not in context and "targets" not in context
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [str(campaign["runtime"]["installed_python"]), "-I", "-B", "-c", _FRESH_REPLAY, str(campaign["archive"])],
        input=json.dumps(context), cwd=tmp_path, env=environment, capture_output=True, text=True, check=False, timeout=240,
    )
    (tmp_path / "fresh-installed.stdout.log").write_text(completed.stdout)
    (tmp_path / "fresh-installed.stderr.log").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    proof = json.loads(completed.stdout)
    actual = proof["outcome"]["outputs"][0]["predictions"][0]
    expected = capture["completeArchive"]["replay"]["outputs"][0]["predictions"][0]
    assert actual["sample_ids"] == expected["sample_ids"] and actual["target_names"] == expected["target_names"]
    np.testing.assert_allclose(actual["values"], expected["values"], rtol=1e-8, atol=1e-8)
    assert proof["source_hashes"] == context["source_hashes"] and proof["lifecycle"].get("fit", 0) == 0


@pytest.mark.parametrize("mutation", ["trust", "unsigned_phase", "raw_bytes", "missing_raw", "aliased_raw"])
def test_public_octave_archive_refuses_invalid_authority_before_callbacks(campaign: dict[str, Any], tmp_path: Path, mutation: str) -> None:
    import dag_ml

    module = campaign["runtime"]["module"]
    archive = campaign["archive"]
    trust = [copy.deepcopy(module.OCTAVE_MANIFEST)]
    request: Any = campaign["replay_request"]
    if mutation == "trust":
        trust[0]["controller_version"] = "999.0.0"
    elif mutation == "unsigned_phase":
        request = request.to_dict()
        request["phase"] = "REFIT"
    else:
        archive = tmp_path / "invalid.n4a"
        with ZipFile(campaign["archive"]) as source, ZipFile(archive, "w") as destination:
            manifest = json.loads(source.read("manifest.json"))
            member_path = manifest["payloads"]["methods"]["role_pipelines"][0]["member_path"]
            for member in source.infolist():
                payload, name = source.read(member.filename), member.filename
                if name == member_path:
                    if mutation == "missing_raw":
                        continue
                    if mutation == "aliased_raw":
                        name = "artifacts/alias.json"
                    else:
                        payload = bytes([payload[0] ^ 1]) + payload[1:]
                destination.writestr(name, payload)
    calls: list[str] = []

    def forbidden(message: Any) -> Any:
        calls.append("callback")
        pytest.fail("Invalid archive or authority reached a controller")

    with pytest.raises((ValueError, RuntimeError, dag_ml.DagMlError)):
        nirs4all.replay_portable_predictor_archive_v2(
            archive, request, campaign["replay_envelopes"], trust, forbidden,
            outcome_id="outcome:sdk.octave.invalid", run_id="run:sdk.octave.invalid", artifact_callback=forbidden,
        )
    assert calls == []


def test_public_octave_replay_failure_releases_every_native_state(campaign: dict[str, Any], tmp_path: Path) -> None:
    import dag_ml

    module = campaign["runtime"]["module"]
    prepared = module.prepare_octave(campaign["runtime"]["octave"], tmp_path, "failed-replay", campaign["heldout"], campaign["operators"], None)

    def failed_predict(task: dict[str, Any]) -> Any:
        assert task["phase"] == "PREDICT"
        raise RuntimeError("injected SDK replay prediction failure")

    with module.OctaveWorker(prepared) as worker:
        with pytest.raises((RuntimeError, dag_ml.DagMlError), match="injected SDK replay"):
            nirs4all.replay_portable_predictor_archive_v2(
                campaign["archive"], campaign["replay_request"], campaign["replay_envelopes"], [module.OCTAVE_MANIFEST], failed_predict,
                outcome_id="outcome:sdk.octave.failure", run_id="run:sdk.octave.failure", artifact_callback=worker.artifact,
            )
        operations = Counter(item["operation"] for item in worker.calls)
        assert operations["hydrate"] == operations["release"] == 5 and not worker.hydrated
    lifecycle = Counter(json.loads(line)["operation"] for line in Path(prepared["audit_path"]).read_text().splitlines())
    assert lifecycle["hydrate"] == lifecycle["release"] == lifecycle["dispose"] == 5
    assert lifecycle["fit"] == lifecycle["PREDICT"] == 0
