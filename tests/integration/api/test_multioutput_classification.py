"""Independent target labels survive voting, storage and both replay routes.

Classification metrics pool target entries into one vector; regression metrics
average per-target values. These checks assert pooled balanced accuracy, not
multilabel exact match.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from nirs4all.data.ensemble_utils import EnsembleUtils


def test_hard_voting_keeps_independent_outputs_and_weights():
    first = np.array([[0, 1], [1, 0], [1, 1]])
    second = np.array([[1, 1], [1, 1], [0, 0]])
    third = np.array([[1, 0], [0, 0], [1, 0]])
    np.testing.assert_array_equal(EnsembleUtils.compute_hard_voting([first, second, third]),
                                  [[1, 1], [1, 0], [1, 0]])
    np.testing.assert_array_equal(EnsembleUtils.compute_hard_voting([first, second, third],
                                  np.array([.8, .1, .1])), first)
    assert EnsembleUtils.compute_hard_voting([first[:, 0], second[:, 0]]).shape == (3, 1)
    with pytest.raises(ValueError, match="matching sample and output"):
        EnsembleUtils.compute_hard_voting([first, second[:, 0]])


@pytest.mark.parametrize("estimator_class", [ClassifierChain, MultiOutputClassifier])
@pytest.mark.parametrize("with_test", [False, True])
def test_multioutput_classification_preserves_target_axes(tmp_path, estimator_class, with_test):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(719).normal(size=(64, 6)).astype("float32")
    y = np.column_stack([X[:, 0] > 0, X[:, 1] > 0]).astype(int)
    dataset = {"train_x": X[:48], "train_y": y[:48], "test_x": X[48:], "test_y": y[48:]} if with_test else (X, y)
    workspace = tmp_path / "workspace"
    nirs4all.run([KFold(2), estimator_class(LogisticRegression())], dataset,
                 workspace_path=workspace, engine="legacy", verbose=0)
    with WorkspaceStore(workspace) as store:
        rows = store.query_predictions().to_dicts()
        assert rows
        for row in rows:
            saved = store.get_prediction(row["prediction_id"], load_arrays=True)
            if saved["y_pred"] is None:
                continue
            truth, actual = np.asarray(saved["y_true"]), np.asarray(saved["y_pred"])
            assert truth.shape == actual.shape
            assert actual.ndim == 2 and actual.shape[1] == 2
            assert set(np.unique(actual)) <= {0, 1}
            assert row["metric"] == "balanced_accuracy"
            assert row[f'{row["partition"]}_score'] == pytest.approx(balanced_accuracy_score(truth.ravel(), actual.ravel()))
        for chain_id in {row["chain_id"] for row in rows}:
            chain = store.get_chain(chain_id)
            assert chain is not None
            fold_models = [store.load_artifact(artifact_id) for artifact_id in chain["fold_artifacts"].values()]
            predictions = [model.predict(X[:8]) for model in fold_models]
            expected = EnsembleUtils.compute_hard_voting(predictions)
            direct = store.replay_chain(chain_id, X[:8])
            assert direct.shape == (8, 2)
            np.testing.assert_array_equal(direct, expected)
            archive = store.export_chain(chain_id, tmp_path / f"{chain_id}.n4a")
            replay = nirs4all.predict(archive, X[:8], engine="legacy", verbose=0)
            assert np.asarray(replay.y_pred).shape == (8, 2)
            np.testing.assert_array_equal(replay.y_pred, expected)


def test_multioutput_regression_preserves_axes_and_macro_rmse(tmp_path):
    from sklearn.linear_model import MultiTaskElasticNet

    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(221).normal(size=(48, 6)).astype("float32")
    y = np.column_stack([3 * X[:, 0] - X[:, 2], 2 * X[:, 1] + X[:, 3]])
    workspace = tmp_path / "workspace"
    nirs4all.run([KFold(2), MultiTaskElasticNet(alpha=.01)], (X, y),
                 workspace_path=workspace, engine="legacy", verbose=0)
    with WorkspaceStore(workspace) as store:
        rows = store.query_predictions().to_dicts()
        for row in rows:
            saved = store.get_prediction(row["prediction_id"], load_arrays=True)
            if saved["y_pred"] is None:
                continue
            truth, actual = np.asarray(saved["y_true"]), np.asarray(saved["y_pred"])
            assert truth.shape == actual.shape and actual.ndim == 2 and actual.shape[1] == 2
            per_target_rmse = np.sqrt(np.mean((truth - actual) ** 2, axis=0))
            assert row[f'{row["partition"]}_score'] == pytest.approx(np.mean(per_target_rmse))
        for chain_id in {row["chain_id"] for row in rows}:
            direct = store.replay_chain(chain_id, X[:8])
            assert direct.shape == (8, 2)
            archive = store.export_chain(chain_id, tmp_path / f"{chain_id}.n4a")
            replay = nirs4all.predict(archive, X[:8], engine="legacy", verbose=0)
            assert np.asarray(replay.y_pred).shape == (8, 2)
            np.testing.assert_allclose(replay.y_pred, direct, atol=1e-6)
