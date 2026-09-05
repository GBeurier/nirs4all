"""General Python API workflows execute on DAG-ML without implicit legacy."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import nirs4all


@pytest.mark.parametrize("input_form", ["tuple", "mapping", "spectro"])
@pytest.mark.parametrize("scale_target", [False, True])
def test_default_general_run_accepts_historical_inputs(input_form, scale_target, monkeypatch, tmp_path):
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy execution"))
    rng = np.random.default_rng(37)
    X = rng.normal(size=(24, 6))
    y = X @ np.arange(1.0, 7.0) + rng.normal(scale=0.05, size=24)
    if input_form == "spectro":
        from nirs4all.data import SpectroDataset

        dataset = SpectroDataset("general-api")
        dataset.add_samples(X[:18], {"partition": "train"})
        dataset.add_targets(y[:18])
        dataset.add_samples(X[18:], {"partition": "test"})
        dataset.add_targets(y[18:])
    else:
        dataset = (X, y) if input_form == "tuple" else {"X": X, "y": y}
    pipeline = [MinMaxScaler()]
    if scale_target:
        pipeline.append({"y_processing": MinMaxScaler()})
    pipeline.extend([KFold(3), {"model": PLSRegression(2)}])
    with nirs4all.run(pipeline, dataset, verbose=0, workspace_path=tmp_path) as result:
        assert result.execution_engine == "dag-ml"
        assert result.num_predictions > 0
        assert np.isfinite(result.cv_best_score)


@pytest.mark.parametrize("save_artifacts", [False, True])
def test_general_session_preserves_workspace_predictions_and_captured_export(tmp_path, monkeypatch, save_artifacts):
    import joblib

    from nirs4all.data import Predictions, SpectroDataset
    from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy execution"))
    rng = np.random.default_rng(71)
    X = rng.normal(size=(30, 6))
    y = X @ np.arange(1.0, 7.0)
    data = SpectroDataset("session-data")
    data.add_samples(X[:24], {"partition": "train"})
    data.add_targets(y[:24])
    data.add_samples(X[24:], {"partition": "test"})
    data.add_targets(y[24:])
    pipeline = [MinMaxScaler(), KFold(3), {"model": PLSRegression(2)}]
    with nirs4all.session(pipeline, workspace_path=tmp_path, verbose=0, save_artifacts=save_artifacts) as session:
        first = session.run(data, project="acceptance")
        second = session.run(data, project="acceptance")
        assert session.execution_engine == "dag-ml"
        assert session._runner is None
        assert len(session.history) == 2
        assert session.workspace_path == tmp_path
        assert first.artifacts_path == second.artifacts_path == tmp_path
        run_ids = [next(iter(result.per_dataset.values()))["run_id"] for result in (first, second)]
        assert len(set(run_ids)) == 2
        with WorkspaceStore(tmp_path) as store:
            for run_id in run_ids:
                stored = store.get_run(run_id)
                assert stored["status"] == "completed"
                assert stored["project_id"] == store.get_project_by_name("acceptance")["project_id"]
            assert store.query_predictions().height == first.num_predictions + second.num_predictions
        persisted = Predictions(tmp_path, load_arrays=True)
        assert persisted.num_predictions == first.num_predictions + second.num_predictions
        persisted.close()
        if save_artifacts:
            assert first._dagml_results_dir.is_dir()
            # Export is replay-only. Any hidden refit after this point fails.
            monkeypatch.setattr(PLSRegression, "fit", lambda *args, **kwargs: pytest.fail("export retrained the model"))
            model_path = first.export_model(tmp_path / "captured.joblib")
            model = joblib.load(model_path)
            scored = first.predictions.filter_predictions(partition="test", fold_id="final", load_arrays=True)
            # SpectroDataset stores features as float32. Replay the exact
            # materialized input rather than the pre-materialization float64.
            replay_x = data.x({"partition": "test"}, layout="2d")
            np.testing.assert_allclose(np.asarray(model.predict(replay_x)), np.asarray(scored[0]["y_pred"]), atol=1e-10)
        else:
            assert first._dagml_results_dir is None
            assert not (tmp_path / "native_results").exists()


def test_general_memory_only_request_does_not_open_workspace(monkeypatch):
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.storage.workspace_store.WorkspaceStore.__init__", lambda *args, **kwargs: pytest.fail("unexpected durable workspace"))
    X = np.random.default_rng(2).normal(size=(18, 6))
    with nirs4all.run([KFold(3), PLSRegression(2)], (X, X[:, 0]), verbose=0, save_artifacts=False) as result:
        assert result.execution_engine == "dag-ml"
        assert result.artifacts_path is None
        assert result._dagml_results_dir is None


@pytest.mark.parametrize("verbose", [0, 1, 2, 3])
@pytest.mark.parametrize("naming,label", [("nirs", "RMSECV"), ("ml", "CV_Score")])
def test_general_logging_honors_verbosity_and_metric_naming(verbose, naming, label, capsys):
    X = np.random.default_rng(8).normal(size=(18, 6))
    with nirs4all.run([KFold(3), PLSRegression(2)], (X, X[:, 0]), verbose=verbose, report_naming=naming, save_artifacts=False) as result:
        assert np.isfinite(result.cv_best_score)
    output = capsys.readouterr().out
    assert (f"DAG-ML completed: {label}=" in output) == (verbose > 0)
