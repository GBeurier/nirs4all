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
