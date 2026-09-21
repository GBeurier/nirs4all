"""Repeated Python training must never bind a chain to another run's fitted state."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("second_features", [5, 9])
def test_repeated_named_runs_preserve_both_models_and_export_provenance(tmp_path, second_features):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    rng = np.random.default_rng(909)
    fitted = []
    for index, features in enumerate((5, second_features)):
        X = rng.normal(size=(40, features)) + index * 12
        y = X[:, 0] * (index + 1) + rng.normal(scale=.05, size=40)
        expected = make_pipeline(StandardScaler(), Ridge()).fit(X[:32], y[:32]).predict(X[32:])
        with nirs4all.run(
            [StandardScaler(), KFold(n_splits=2), Ridge()], (X, y, {"train": 32}),
            name="Repeated pipeline", engine="legacy", workspace_path=tmp_path / "workspace",
            verbose=0, save_charts=False,
        ) as result:
            row = next(row for row in result.predictions.filter_predictions(fold_id="final", load_arrays=True)
                       if row["partition"] == "test")
            np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected, atol=1e-5)
            archive = result.export(tmp_path / f"run-{index}.n4a")
            fitted.append((row["chain_id"], X[32:], expected, archive))

    # Reopen after both runs: verify both old and new chains, not just the last
    # in-memory model or an archive exported before the collision could occur.
    with WorkspaceStore.open_readonly(tmp_path / "workspace") as store:
        identities = []
        for index, (chain_id, X, expected, archive) in enumerate(fitted):
            chain = store.get_chain(chain_id)
            identities.append(set(chain["shared_artifacts"]["1"]) | set(chain["fold_artifacts"].values()))
            new_export = store.export_chain(chain_id, tmp_path / f"reopened-{index}.n4a")
            for model in (archive, new_export):
                predicted = nirs4all.predict(model=model, data=X, engine="legacy").y_pred
                np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, atol=1e-5)
            predicted = nirs4all.predict(chain_id=chain_id, workspace_path=tmp_path / "workspace", data=X, engine="legacy").y_pred
            np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, atol=1e-5)
        assert identities[0].isdisjoint(identities[1])


def test_existing_artifact_identity_refuses_changed_content(tmp_path):
    from nirs4all.pipeline.storage import WorkspaceStore

    with WorkspaceStore(tmp_path) as store:
        fields = {"artifact_id": "pipeline$abcdef:all", "path": "a/model.joblib", "content_hash": "a" * 64,
                  "operator_class": "Ridge", "artifact_type": "model", "format": "joblib", "size_bytes": 100}
        assert store.register_existing_artifact(**fields) == fields["artifact_id"]
        assert store.register_existing_artifact(**fields) == fields["artifact_id"]
        with pytest.raises(ValueError, match="Artifact identity collision"):
            store.register_existing_artifact(**{**fields, "content_hash": "b" * 64})
        # Failure must preserve the original durable identity for existing chains.
        assert store.get_artifact_path(fields["artifact_id"]).as_posix().endswith("/a/model.joblib")


def test_store_export_replays_target_scaling(tmp_path):
    from sklearn.compose import TransformedTargetRegressor

    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    rng = np.random.default_rng(910)
    X = rng.normal(size=(40, 5))
    y = 100 + 30 * X[:, 0] + rng.normal(scale=.05, size=40)
    expected = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge()), transformer=StandardScaler(),
    ).fit(X[:32], y[:32]).predict(X[32:])
    with nirs4all.run(
        [StandardScaler(), {"y_processing": StandardScaler()}, KFold(n_splits=2), Ridge()],
        (X, y, {"train": 32}), engine="legacy", workspace_path=tmp_path / "workspace",
        verbose=0, save_charts=False,
    ) as result:
        row = next(row for row in result.predictions.filter_predictions(fold_id="final", load_arrays=True)
                   if row["partition"] == "test")
        np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), expected, rtol=1e-6, atol=1e-5)
        chain_id = row["chain_id"]
    with WorkspaceStore.open_readonly(tmp_path / "workspace") as store:
        archive = store.export_chain(chain_id, tmp_path / "scaled.n4a")
    predicted = nirs4all.predict(model=archive, data=X[32:], engine="legacy").y_pred
    np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, rtol=1e-6, atol=1e-5)
    predicted = nirs4all.predict(
        chain_id=chain_id, workspace_path=tmp_path / "workspace", data=X[32:], engine="legacy",
    ).y_pred
    np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, rtol=1e-6, atol=1e-5)


def test_train_only_sequential_models_have_independent_replayable_chains(tmp_path):
    """A model without folds must not become preprocessing of the next model."""
    from sklearn.cross_decomposition import PLSRegression

    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    rng = np.random.default_rng(222)
    X = rng.normal(size=(30, 5))
    y = 2 * X[:, 0] - X[:, 1]
    models = [PLSRegression(n_components=2), Ridge(alpha=1)]
    expected = {
        type(model).__name__: make_pipeline(StandardScaler(), model).fit(X, y).predict(X[:6]).ravel()
        for model in models
    }
    workspace = tmp_path / "train-only"
    with nirs4all.run(
        [StandardScaler(), *models], (X, y), engine="legacy", workspace_path=workspace,
        verbose=0, save_charts=False, plots_visible=False,
    ) as result:
        assert result.num_predictions > 0
        assert set(result.get_models()) == set(expected)
        assert result.best == {}  # No validation score is invented for training-only execution.
        rows = result.predictions.filter_predictions(partition="train", load_arrays=True)
        assert len({row["chain_id"] for row in rows}) == 2
        identities = [(row["model_name"], row["chain_id"]) for row in rows]

    with WorkspaceStore.open_readonly(workspace) as store:
        for name, chain_id in identities:
            chain = store.get_chain(chain_id)
            assert chain["model_name"] == chain["model_class"] == name
            assert chain["fold_strategy"] == "shared"
            assert chain["cv_fold_count"] == 0
            assert chain["cv_val_score"] is None
            assert chain["final_test_score"] is None
            assert chain["cv_train_score"] is not None
            import json
            scores = json.loads(chain["cv_scores"])
            assert scores["train"]["rmse"] == pytest.approx(chain["cv_train_score"], abs=1e-6)
            assert not scores.get("val") and not scores.get("test")
            # Both direct replay and exported replay must use this model's fitted state.
            np.testing.assert_allclose(store.replay_chain(chain_id, X[:6]).ravel(), expected[name], atol=1e-6)
            archive = store.export_chain(chain_id, tmp_path / f"{name}.n4a")
            actual = nirs4all.predict(model=archive, data=X[:6], engine="legacy", verbose=0).y_pred
            np.testing.assert_allclose(np.asarray(actual).ravel(), expected[name], atol=1e-6)
