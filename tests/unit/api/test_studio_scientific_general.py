"""The v2 host contract uses real general DAG results and bounded JSON."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.api import studio_scientific_general as general
from nirs4all.pipeline.config.component_serialization import serialize_component


def _request(tmp_path, classification=False):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 8))
    y = X[:, 0] * 2 - X[:, 1]
    if classification:
        y = (y > np.median(y)).astype(int)
    splitter = StratifiedKFold(3) if classification else KFold(3)
    model = LogisticRegression() if classification else Ridge()
    return {
        "schema": general.STUDIO_GENERAL_JOB_SCHEMA,
        "operation": "run", "job_id": "job-general",
        "pipeline": serialize_component([StandardScaler(), splitter, model]),
        "dataset": {"X": X.tolist(), "y": y.tolist()},
        "options": {"workspace_path": str(tmp_path), "name": "general", "project": "studio"},
    }


@pytest.mark.parametrize("classification", [False, True])
def test_general_contract_executes_canonical_pipeline_and_real_workspace(tmp_path, monkeypatch, classification):
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    # Source checkout qualification only; installed-prefix attestation is
    # independently exercised by the installed host integration gate.
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy host"))
    response = general.studio_scientific_job_v2(_request(tmp_path, classification))
    json.dumps(response, allow_nan=False)
    assert response["schema"] == general.STUDIO_GENERAL_RESULT_SCHEMA
    assert response["engine"] == "dag-ml"
    assert response["result"]["native_score_sets_available"]
    assert response["result"]["prediction_count"] > 0
    assert np.isfinite(response["result"]["validation_score"])
    with WorkspaceStore(tmp_path) as store:
        for run_id in response["result"]["run_ids"]:
            assert store.get_run(run_id)["status"] == "completed"
        assert store.query_predictions().height == response["result"]["prediction_count"]


@pytest.mark.parametrize("change,code", [
    ({"engine": "legacy"}, "engine_forbidden"),
    ({"allow_fallback": True}, "engine_forbidden"),
    ({"extra": "unknown"}, "invalid_shape"),
    ({"pipeline": [{"function": "os.system", "params": {"command": "must-not-execute"}}]}, "operator_package_forbidden"),
    ({"pipeline": [{"class": "user_plugin.Model"}]}, "operator_package_forbidden"),
    ({"pipeline": [{"class": ""}]}, "invalid_operator"),
    ({"options": {"workspace_path": "relative"}}, "workspace_required"),
    ({"options": {"workspace_path": "/workspace", "engine": "legacy"}}, "unknown_option"),
])
def test_invalid_general_request_fails_before_runtime_or_operator_instantiation(tmp_path, monkeypatch, change, code):
    request = {**_request(tmp_path), **change}
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: pytest.fail("runtime touched before validation"))
    monkeypatch.setattr(general, "deserialize_component", lambda *args: pytest.fail("operator instantiated before validation"))
    with pytest.raises(general.StudioScientificJobError) as error:
        general.studio_scientific_job_v2(request)
    assert error.value.code == code


def test_general_operator_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("operator failure")

    monkeypatch.setattr(general, "_run_strict_product", fail)
    with pytest.raises(RuntimeError, match="operator failure"):
        general.studio_scientific_job_v2(_request(tmp_path))
    assert len(calls) == 1
    assert calls[0]["engine"] == "dag-ml"
    assert calls[0]["allow_fallback"] is False
