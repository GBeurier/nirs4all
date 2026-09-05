"""Batch composition retains independently executed DAG results and ownership."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.public_batch import DagMLBatchResult, run_dagml_public


@pytest.mark.parametrize("pipeline_count,dataset_count", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_batch_runs_real_dag_and_retains_children(pipeline_count, dataset_count, tmp_path, monkeypatch):
    import nirs4all
    from nirs4all.pipeline.runner import PipelineRunner

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy batch execution")

    monkeypatch.setattr(PipelineRunner, "run", forbidden)
    rng = np.random.default_rng(314159)
    X = rng.normal(size=(36, 12))
    y = 2 * X[:, 0] - X[:, 1]
    pipelines = [[StandardScaler(), KFold(3, shuffle=True, random_state=42), {"model": Ridge(index + 1)}] for index in range(pipeline_count)]
    datasets = [(X + index * 0.1, y + index * 0.2) for index in range(dataset_count)]
    result = nirs4all.run(
        pipelines if pipeline_count > 1 else pipelines[0], datasets if dataset_count > 1 else datasets[0],
        verbose=0, random_state=42, name="batch", workspace_path=tmp_path,
    )
    assert isinstance(result, DagMLBatchResult)
    assert len(result.runs) == pipeline_count * dataset_count
    assert result.num_predictions == sum(child.num_predictions for child in result.runs)
    assert result.execution_engine == "dag-ml"
    assert result._dagml_score_set is None  # No invented combined native identity.
    assert all(child._dagml_score_set is not None and child._dagml_refit_artifacts for child in result.runs)
    assert all(np.isfinite(child.cv_best_score) for child in result.runs)
    for child in result.runs:
        assert result._source_run(child.best) is child
    assert sum(len(info["runs"]) for info in result.per_dataset.values()) == len(result.runs)
    before = [child.num_predictions for child in result.runs]
    assert result.num_predictions == sum(before)  # Aggregation did not mutate child buffers.
    import joblib

    exported = result.export_model(tmp_path / "batch-model.joblib")
    selected = result._source_run(None).export_model(tmp_path / "selected-model.joblib")
    np.testing.assert_allclose(joblib.load(exported).predict(X[:5]), joblib.load(selected).predict(X[:5]), rtol=0, atol=0)
    result.close()
    result.close()


def test_singleton_list_unwraps_without_constructing_a_batch(monkeypatch):
    import nirs4all.pipeline.dagml.run_backend as backend

    marker = object()
    calls = []

    def execute(pipeline, dataset, **options):
        calls.append((pipeline, dataset, options))
        return marker

    monkeypatch.setattr(backend, "run_via_dagml", execute)
    pipeline = [KFold(3), {"model": Ridge()}]
    dataset = (np.ones((9, 2)), np.arange(9))
    assert run_dagml_public([pipeline], [dataset], name="single") is marker
    assert len(calls) == 1
    assert calls[0][0] is pipeline and calls[0][1] is dataset


def test_runtime_error_closes_prior_children_without_retry(monkeypatch):
    import nirs4all.pipeline.dagml.run_backend as backend

    calls = []

    class Completed:
        def close(self):
            calls.append("close")

    def execute(*args, **kwargs):
        calls.append("execute")
        if calls.count("execute") == 2:
            raise RuntimeError("scientific failure")
        return Completed()

    monkeypatch.setattr(backend, "run_via_dagml", execute)
    with pytest.raises(RuntimeError, match="scientific failure"):
        run_dagml_public([KFold(3), {"model": Ridge()}], [(np.ones((9, 2)), np.arange(9))] * 3)
    assert calls == ["execute", "execute", "close"]


def test_caller_scratch_is_disjoint_for_every_child(monkeypatch, tmp_path):
    import nirs4all.pipeline.dagml.run_backend as backend
    from nirs4all.api.result import RunResult
    from nirs4all.data.predictions import Predictions

    scratch = []

    def execute(*args, **kwargs):
        scratch.append(kwargs["workdir"])
        return RunResult(Predictions(), {})

    monkeypatch.setattr(backend, "run_via_dagml", execute)
    dataset = (np.ones((9, 2)), np.arange(9))
    for _ in range(2):
        run_dagml_public([KFold(3), {"model": Ridge()}], [dataset, dataset], workdir=tmp_path)
    assert len(set(scratch)) == 4
    assert all(path.is_relative_to(tmp_path) and path != tmp_path for path in scratch)
