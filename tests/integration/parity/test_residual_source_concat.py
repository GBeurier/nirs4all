"""ResidualModel after a shared by-source preprocessing and concat merge."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.models.residual import ResidualModel
from nirs4all.operators.transforms import StandardNormalVariate

from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("source_style", ["shared", "distinct"])
def test_residual_after_by_source_concat_refit_and_replay(tmp_path, monkeypatch, mechanism: str, source_style: str) -> None:
    """Both residual stages apply source-local X transforms within each fold."""
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    source = dataset_path("multi")
    source_steps = (
        [StandardNormalVariate()]
        if source_style == "shared" else {
            "source_0": [StandardNormalVariate()],
            "source_1": [StandardScaler()],
            "source_2": [MinMaxScaler()],
        }
    )
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": {"by_source": True, "steps": source_steps}},
        {"merge": "concat"},
        {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate=False)},
    ]
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(native.cv_best_score)
    assert np.isfinite(native.best_rmse)
    archive = native.export(tmp_path / "residual_by_source.n4a")
    dataset = DatasetConfigs(source).get_dataset_at(0)
    features = dataset.x({"partition": "test"}, layout="2d")
    targets = np.asarray(dataset.y({"partition": "test"})).ravel()
    replay = nirs4all.predict(archive, features)
    assert np.sqrt(np.mean((targets - np.asarray(replay.y_pred).ravel()) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_named_by_source_residual_learner_search_replays(tmp_path, monkeypatch, mechanism: str) -> None:
    """A named, source-local residual keeps a real learner search through archive replay."""
    import optuna

    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    studies = []
    original_optimize = optuna.study.Study.optimize

    def capture_optimize(study, *args, **kwargs):
        outcome = original_optimize(study, *args, **kwargs)
        studies.append(study)
        return outcome

    monkeypatch.setattr(optuna.study.Study, "optimize", capture_optimize)
    source = dataset_path("multi")
    search = {"n_trials": 2, "sampler": "grid", "approach": "single",
              "model_params": {"alpha": [0.01, 1.0]}}
    operator = ResidualModel(
        base=PLSRegression(n_components=2), learner=Ridge(), gate=False,
        name="named_by_source_search", finetune_space=search,
    )
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"branch": {"by_source": True, "steps": [StandardNormalVariate()]}},
        {"merge": "concat"},
        {"model": operator},
    ]
    legacy = nirs4all.run(
        pipeline, source, engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert np.isfinite(legacy.cv_best_score)
        rows = legacy.predictions.filter_predictions(model_name=operator.name, load_arrays=True)
        assert rows and all(np.isfinite(row["val_score"]) for row in rows if row["partition"] == "val")
        assert any(
            len(study.trials) == 2
            and {trial.params.get("alpha") for trial in study.trials} == {0.01, 1.0}
            and all(np.isfinite(trial.value) for trial in study.trials)
            for study in studies
        )
    finally:
        legacy.close()
        monkeypatch.setattr(optuna.study.Study, "optimize", original_optimize)

    native = nirs4all.run(
        pipeline, source, engine="dag-ml", refit=True, allow_fallback=False,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        learner = next(artifact["estimator"] for artifact in native._dagml_refit_artifacts
                       if artifact["controller_id"] == "controller:nirs4all.residual_learner")
        trials = learner._nirs4all_host_hpo["trials"]
        assert {trial["params"]["alpha"] for trial in trials} == {0.01, 1.0}
        assert learner._nirs4all_host_hpo["evaluation"]["outer_validation_used"] is False
        dataset = DatasetConfigs(source).get_dataset_at(0)
        features = dataset.x({"partition": "test"}, layout="2d")
        targets = np.asarray(dataset.y({"partition": "test"})).ravel()
        replay = nirs4all.predict(native.export(tmp_path / "named_by_source_search.n4a"), features)
        actual = np.asarray(replay.y_pred).ravel()
        assert actual.shape == targets.shape
        assert np.sqrt(np.mean((targets - actual) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()
