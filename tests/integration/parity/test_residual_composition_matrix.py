"""ResidualModel variant selection composes with target and feature prefixes."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models.residual import ResidualModel

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("prefix", ["target_processing", "feature_merge"])
def test_residual_choices_after_prefix_refit_and_replay(tmp_path, monkeypatch, mechanism: str, prefix: str) -> None:
    """A 2×2 base/learner choice keeps its native winner through archive replay."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(920)
    features = rng.normal(size=(34, 8))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=34)

    def dataset() -> SpectroDataset:
        result = SpectroDataset("residual_composition_matrix")
        result.add_samples(features[:30], {"partition": "train"}, headers=[str(index) for index in range(8)])
        result.add_samples(features[30:], {"partition": "test"})
        result.add_targets(targets.reshape(-1, 1))
        return result

    def pipeline() -> list:
        prefix_steps = (
            [{"y_processing": StandardScaler()}]
            if prefix == "target_processing"
            else [{"branch": [[StandardScaler()], [MinMaxScaler()]]}, {"merge": "features"}]
        )
        return [
            KFold(2, shuffle=True, random_state=1),
            *prefix_steps,
            {"residual": {
                "base": {"_or_": [PLSRegression(n_components=2), Ridge(alpha=1.0)]},
                "learner": {"_or_": [Ridge(alpha=0.5), ElasticNet(alpha=0.02, max_iter=1000)]},
                "gate": "auto",
            }},
        ]

    legacy = nirs4all.run(pipeline(), dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy", save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy_names = {
        row["config_name"] for row in legacy.predictions.filter_predictions(load_arrays=False)
        if row.get("config_name")
    }
    assert len(legacy_names) == 4
    legacy.close()

    native = nirs4all.run(pipeline(), dataset(), engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert len(native.runs) == 4
        assert {run.cv_best["config_name"] for run in native.runs} == legacy_names
        assert np.isfinite(native.best_rmse)
        archive = native.export(tmp_path / "residual_composition.n4a")
        predicted = np.asarray(nirs4all.predict(archive, features[30:]).y_pred).ravel()
        assert np.sqrt(np.mean((targets[30:] - predicted) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_after_target_and_prediction_feature_join_keeps_pls_fit_scope(tmp_path, monkeypatch, mechanism: str) -> None:
    """Both engines fit and replay a residual model after prediction features."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(922)
    features = rng.normal(size=(34, 8))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=34)

    def dataset() -> SpectroDataset:
        result = SpectroDataset("residual_prediction_join_small")
        result.add_samples(features[:30], {"partition": "train"}, headers=[str(index) for index in range(8)])
        result.add_samples(features[30:], {"partition": "test"})
        result.add_targets(targets.reshape(-1, 1))
        return result

    def pipeline() -> list:
        return [
            KFold(2, shuffle=True, random_state=1),
            {"y_processing": StandardScaler()},
            {"branch": [[{"model": Ridge(alpha=1.0)}], [{"model": Ridge(alpha=2.0)}]]},
            {"merge": "predictions"},
            {"model": ResidualModel(base=PLSRegression(n_components=2), learner=Ridge(), gate="auto")},
        ]

    legacy = nirs4all.run(pipeline(), dataset(), engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy_join", save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(pipeline(), dataset(), engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native_join", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        archive = native.export(tmp_path / "residual_prediction_join_small.n4a")
        predicted = np.asarray(nirs4all.predict(archive, features[30:]).y_pred).ravel()
        assert np.sqrt(np.mean((targets[30:] - predicted) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_residual_learner_search_and_training_controls_after_target_prefix(tmp_path, monkeypatch, mechanism: str) -> None:
    """Learner search and fit options keep the transformed target in native refit."""
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    source = dataset_path("regression")
    search = {"n_trials": 2, "sampler": "grid", "approach": "single", "model_params": {"alpha": [0.01, 1.0]}}

    def pipeline() -> list:
        return [
            KFold(2, shuffle=True, random_state=1),
            {"y_processing": StandardScaler()},
            {"model": ResidualModel(
                base=PLSRegression(n_components=2), learner=Ridge(), gate="auto",
                train_params={"fit_intercept": False}, finetune_space=search,
            )},
        ]

    legacy = nirs4all.run(pipeline(), source, engine="legacy", refit=False,
                          workspace_path=tmp_path / "legacy_search", save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    legacy.close()

    native = nirs4all.run(pipeline(), source, engine="dag-ml", allow_fallback=False, refit=True,
                          workspace_path=tmp_path / "native_search", save_artifacts=False, save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        learner = next(artifact["estimator"] for artifact in native._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.residual_learner")
        assert learner.fit_intercept is False
        assert {trial["params"]["alpha"] for trial in learner._nirs4all_host_hpo["trials"]} == {0.01, 1.0}
        assert learner._nirs4all_host_hpo["evaluation"]["outer_validation_used"] is False
        dataset = DatasetConfigs(source).get_dataset_at(0)
        features = dataset.x({"partition": "test"}, layout="2d")
        targets = np.asarray(dataset.y({"partition": "test"})).ravel()
        archive = native.export(tmp_path / "residual_search_target.n4a")
        predicted = np.asarray(nirs4all.predict(archive, features).y_pred).ravel()
        assert np.sqrt(np.mean((targets - predicted) ** 2)) == pytest.approx(native.best_rmse, abs=1e-5)
    finally:
        native.close()
