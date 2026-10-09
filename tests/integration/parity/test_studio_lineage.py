"""Studio identities attach and recover from exact owner evidence only."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.api.studio_lineage import reconcile_studio_job_lineage, record_studio_run_provenance, recover_studio_job_lineage, validate_studio_provenance
from nirs4all.api.studio_scientific import StudioScientificJobError
from nirs4all.pipeline.config.component_serialization import serialize_component
from nirs4all.pipeline.dagml.dataset import _materialize_dataset
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

pytestmark = pytest.mark.parity


def test_exact_job_provenance_and_old_record_recovery_never_fit(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 6))
    data = {"X": x, "y": x[:, 0] + x[:, 1]}
    pipeline = [KFold(3), {"model": PLSRegression(2)}]
    start = datetime.now(UTC).replace(microsecond=0)
    result = nirs4all.run(pipeline, data, engine="dag-ml", name="lineage", verbose=0,
                         save_charts=False, workspace_path=tmp_path)
    try:
        run_id = next(iter(result.per_dataset.values()))["run_id"]
    finally:
        result.close()
    end = datetime.now(UTC) + timedelta(seconds=1)
    digest = _materialize_dataset(data).content_hash()
    provenance = validate_studio_provenance({"job_id": "job-1", "dataset_ids_by_hash": {digest: "dataset-1"}}, "job-1")
    assert record_studio_run_provenance(tmp_path, [run_id], provenance) == {run_id: "dataset-1"}
    with WorkspaceStore(tmp_path) as store:
        record = store.get_run(run_id)
        assert record["config"]["studio_provenance"] == provenance
        assert record["datasets"][0]["linked_dataset_id"] == "dataset-1"
    monkeypatch.setattr(PLSRegression, "fit", lambda *args, **kwargs: pytest.fail("metadata recovery must never fit"))
    request = {"pipeline": serialize_component(pipeline), "datasets": [{"dataset_id": "dataset-1", "config": data}],
               "run_name": "lineage", "started_at": start.isoformat(), "completed_at": end.isoformat()}
    recovered = recover_studio_job_lineage(tmp_path, **request)
    assert recovered == {"run_ids": [run_id], "dataset_run_ids": {run_id: "dataset-1"}}
    assert reconcile_studio_job_lineage(tmp_path, "job-1", **request) == recovered
    assert recover_studio_job_lineage(tmp_path, **{**request, "run_name": "another"})["run_ids"] == []
    assert recover_studio_job_lineage(tmp_path, **{**request, "pipeline": serialize_component([KFold(3), {"model": PLSRegression(3)}])})["run_ids"] == []
    changed = {"X": x + 1, "y": data["y"]}
    assert recover_studio_job_lineage(tmp_path, **{**request, "datasets": [{"dataset_id": "dataset-1", "config": changed}]})["run_ids"] == []
    with pytest.raises(StudioScientificJobError, match="already belongs"):
        record_studio_run_provenance(tmp_path, [run_id], {**provenance, "job_id": "job-2"})


def test_recovery_refuses_duplicate_matching_runs(tmp_path):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(24, 6))
    data = {"X": x, "y": x[:, 0]}
    pipeline = [KFold(3), {"model": PLSRegression(2)}]
    start = datetime.now(UTC).replace(microsecond=0)
    for _ in range(2):
        result = nirs4all.run(pipeline, data, engine="dag-ml", name="duplicate", verbose=0,
                             save_charts=False, workspace_path=tmp_path)
        result.close()
    result = recover_studio_job_lineage(tmp_path, pipeline=serialize_component(pipeline),
                                       datasets=[{"dataset_id": "dataset-1", "config": data}], run_name="duplicate",
                                       started_at=start.isoformat(), completed_at=(datetime.now(UTC) + timedelta(seconds=1)).isoformat())
    assert result["run_ids"] == []


def test_scientific_general_contract_attaches_authorized_provenance(tmp_path, monkeypatch):
    from nirs4all.api import studio_scientific_general as general

    rng = np.random.default_rng(18)
    x = rng.normal(size=(24, 6))
    data = {"X": x, "y": x[:, 0] + x[:, 1]}
    digest = _materialize_dataset(data).content_hash()
    monkeypatch.setattr(general, "_ambient_runtime_preflight", lambda: None)
    response = general.studio_scientific_job_v2({
        "schema": general.STUDIO_GENERAL_JOB_SCHEMA, "operation": "run", "job_id": "job-provenance",
        "pipeline": serialize_component([KFold(3), {"model": PLSRegression(2)}]),
        "dataset": {"X": x.tolist(), "y": data["y"].tolist()},
        "options": {"workspace_path": str(tmp_path), "save_charts": False,
                    "studio_provenance": {"job_id": "job-provenance", "dataset_ids_by_hash": {digest: "dataset-provenance"}}},
    })
    run_id = response["result"]["run_ids"][0]
    assert response["result"]["dataset_run_ids"] == {run_id: "dataset-provenance"}
    with WorkspaceStore(tmp_path) as store:
        assert store.get_run(run_id)["datasets"][0]["linked_dataset_id"] == "dataset-provenance"


@pytest.mark.parametrize("provenance", [
    {"job_id": "another", "dataset_ids_by_hash": {"0" * 64: "id"}},
    {"job_id": "job", "dataset_ids_by_hash": {}},
    {"job_id": "job", "dataset_ids_by_hash": {"not-hash": "id"}},
    {"job_id": "job", "dataset_ids_by_hash": {"0" * 64: "../../path"}},
])
def test_invalid_host_provenance_is_rejected(provenance):
    with pytest.raises(StudioScientificJobError):
        validate_studio_provenance(provenance, "job")
