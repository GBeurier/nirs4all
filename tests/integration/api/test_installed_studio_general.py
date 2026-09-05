"""Installed general Studio host: real isolated process and real workspace."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2") != "1", reason="requires the installed native V1 train")
@pytest.mark.parametrize("classification", [False, True])
def test_installed_studio_general_host_keeps_rust_boundary_and_durable_results(tmp_path, classification):
    script = r'''
import json
import sys
from pathlib import Path
import numpy as np
import nirs4all
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

request = json.loads(sys.stdin.read())
response = nirs4all.studio_scientific_job_v2(request)
assert response["engine"] == "dag-ml"
assert response["result"]["native_score_sets_available"]
assert np.isfinite(response["result"]["validation_score"])
with WorkspaceStore(Path(request["options"]["workspace_path"])) as store:
    assert all(store.get_run(run_id)["status"] == "completed" for run_id in response["result"]["run_ids"])
    assert store.query_predictions().height == response["result"]["prediction_count"]
print(json.dumps(response, allow_nan=False))
'''
    import numpy as np

    X = np.random.default_rng(64).normal(size=(30, 8))
    y = X[:, 0] - 2 * X[:, 1]
    if classification:
        y = (y > np.median(y)).astype(int)
    request = {
        "schema": "nirs4all.studio-scientific-job.v2", "operation": "run", "job_id": "installed-host",
        "pipeline": [
            "sklearn.preprocessing.StandardScaler",
            {"class": "sklearn.model_selection.StratifiedKFold" if classification else "sklearn.model_selection.KFold", "params": {"n_splits": 3}},
            "sklearn.linear_model.LogisticRegression" if classification else "sklearn.linear_model.Ridge",
        ],
        "dataset": {"X": X.tolist(), "y": y.tolist()},
        "options": {"workspace_path": str(tmp_path), "project": "installed-studio", "verbose": 0},
    }
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("N4M_", "N4A_")) and key not in {"PYTHONPATH", "LD_LIBRARY_PATH", "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"}
    }
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script], input=json.dumps(request), text=True, capture_output=True, env=env, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    response = json.loads(completed.stdout)
    assert response["schema"] == "nirs4all.studio-scientific-job-result.v2"
    assert response["job_id"] == request["job_id"]
    assert response["result"]["run_ids"]
