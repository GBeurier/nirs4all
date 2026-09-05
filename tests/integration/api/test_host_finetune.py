"""Host optimizer proposals with true DAG evaluation and isolated outer folds."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _data():
    rng = np.random.default_rng(234)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    return X, (X @ np.arange(1.0, 7.0) + rng.normal(scale=0.25, size=40)).astype(np.float32)


def _run(X, y, tmp_path, *, sampler="grid", model_params=None, n_trials=2):
    import nirs4all

    return nirs4all.run(
        [StandardScaler(), ShuffleSplit(2, test_size=0.25, random_state=42),
         {"model": PLSRegression(), "finetune_params": {
             "approach": "single", "sampler": sampler, "verbose": 0, "seed": 42,
             "n_trials": n_trials, "model_params": model_params or {"n_components": [1, 2]},
         }}],
        (X, y), workspace_path=tmp_path, save_artifacts=False, save_charts=False, verbose=0,
    )


@pytest.mark.parametrize("sampler,space", [("grid", {"n_components": [1, 2]}), ("tpe", {"n_components": ("int", 1, 3)})])
def test_host_search_real_scores_selection_and_export(tmp_path, monkeypatch, sampler, space):
    import nirs4all
    from nirs4all.optimization.optuna import OptunaManager

    monkeypatch.setattr(OptunaManager, "finetune", lambda *args, **kwargs: pytest.fail("legacy optimizer execution"))
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy scheduler"))
    X, y = _data()
    result = _run(X, y, tmp_path, sampler=sampler, model_params=space)
    artifact = result._dagml_refit_artifacts[0]
    fitted = artifact["estimator"]
    evidence = fitted._nirs4all_host_hpo
    assert evidence["profile"] == "host_optimizer_search_v1" and evidence["portable"] is False
    assert evidence["scope"]["phase"] == "REFIT"
    assert len(evidence["trials"]) == 2
    assert evidence["evaluation"] == {"role": "inner_parameter_selection", "outer_validation_used": False, "test_used": False}
    train, val = next(ShuffleSplit(1, test_size=0.2, random_state=42).split(X))
    for trial in evidence["trials"]:
        expected = make_pipeline(StandardScaler(), PLSRegression(**trial["params"])).fit(X[train], y[train].astype(float))
        score = np.sqrt(np.mean((expected.predict(X[val]).ravel() - y[val]) ** 2))
        assert trial["score"] == pytest.approx(score, rel=1e-7, abs=1e-7)
        assert trial["scores"]["reports"]
    winner = min(evidence["trials"], key=lambda trial: (trial["score"], trial["trial_index"]))
    assert evidence["selected_params"] == winner["params"]
    assert fitted.steps[-1][1].get_params()["n_components"] == winner["params"]["n_components"]
    expected = fitted.predict(X[:7]).ravel()
    monkeypatch.setattr(PLSRegression, "fit", lambda *args, **kwargs: pytest.fail("export/replay fitted"))
    archive = tmp_path / "tuned.n4a"
    result.export(archive)
    np.testing.assert_array_equal(nirs4all.predict(archive, X[:7]).y_pred, expected)
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    recorded = load_general_archive(archive)["manifest"]["host_hpo"]
    assert recorded["portable"] is False
    assert [search["scope"]["phase"] for search in recorded["searches"]] == ["FIT_CV", "FIT_CV", "REFIT"]
    result.close()


def test_outer_validation_poison_cannot_change_tuning_or_prediction(tmp_path, monkeypatch):
    from nirs4all.pipeline.dagml import host_finetune

    X, y = _data()
    _, val = next(ShuffleSplit(2, test_size=0.25, random_state=42).split(X))
    original = host_finetune.run_scoped_finetune
    recorded = []

    def capture(*args, **kwargs):
        answer = original(*args, **kwargs)
        if kwargs["scope"]["fold_id"] == "fold0":
            recorded.append(answer)
        return answer

    monkeypatch.setattr(host_finetune, "run_scoped_finetune", capture)
    first = _run(X, y, tmp_path / "first")
    poisoned = y.copy()
    poisoned[val] += 10000
    second = _run(X, poisoned, tmp_path / "second")
    assert len(recorded) == 2
    assert recorded[0][0] == recorded[1][0]
    assert [trial["score"] for trial in recorded[0][1]["trials"]] == [trial["score"] for trial in recorded[1][1]["trials"]]
    rows = [result.predictions.filter_predictions(partition="val", fold_id="0", load_arrays=True)[0] for result in (first, second)]
    np.testing.assert_array_equal(rows[0]["y_pred"], rows[1]["y_pred"])
    first.close()
    second.close()


def test_grid_exhaustion_stops_before_budget_without_duplicate_trials(tmp_path):
    X, y = _data()
    result = _run(X, y, tmp_path, n_trials=5)
    evidence = result._dagml_refit_artifacts[0]["estimator"]._nirs4all_host_hpo
    assert len(evidence["trials"]) == 2
    assert {trial["params"]["n_components"] for trial in evidence["trials"]} == {1, 2}
    result.close()


def test_failed_inner_fit_propagates_once_without_retry(tmp_path, monkeypatch):
    X, y = _data()
    calls = []

    def fail_fit(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("deliberate inner candidate failure")

    monkeypatch.setattr(PLSRegression, "fit", fail_fit)
    with pytest.raises(Exception, match="deliberate inner candidate failure"):
        _run(X, y, tmp_path, n_trials=5)
    assert calls == [1]
