"""Sparse sklearn transformer outputs enter dense storage only within a budget."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

from nirs4all.utils.transform_output import normalize_transform_output


@pytest.mark.parametrize("operator", [
    OneHotEncoder(handle_unknown="ignore"), KNeighborsTransformer(n_neighbors=3),
    RadiusNeighborsTransformer(radius=2), RandomTreesEmbedding(n_estimators=5, max_depth=3, random_state=4),
    SparseRandomProjection(n_components=3, random_state=4), TfidfTransformer(),
])
def test_real_sparse_transformer_pipeline_preserves_numeric_features(tmp_path, operator):
    import nirs4all

    X = np.random.default_rng(45).integers(1, 5, size=(32, 4)).astype(np.float32)
    y = X[:, 0] - X[:, 1] + X[:, 2] * .1
    with nirs4all.run([operator, Ridge()], (X, y), engine="legacy", workspace_path=tmp_path,
                     verbose=0, save_charts=False, refit=False) as result:
        predictions = result.predictions.filter_predictions(partition="train", load_arrays=True)
        assert predictions and len(predictions[0]["y_pred"]) == len(X)
        assert np.isfinite(np.asarray(predictions[0]["y_pred"])).all()


def test_sparse_conversion_matches_values_and_preserves_dense_identity():
    dense = np.array([[0., 1.], [2., 0.]], dtype=np.float32)
    converted = normalize_transform_output(csr_matrix(dense), "test")
    np.testing.assert_array_equal(converted, dense)
    assert converted.dtype == dense.dtype
    assert normalize_transform_output(dense, "test") is dense


def test_large_sparse_output_is_rejected_before_allocation(monkeypatch):
    sparse = csr_matrix((100_000, 100_000), dtype=np.float64)
    monkeypatch.setattr(sparse, "toarray", lambda: pytest.fail("oversized conversion must not allocate"))
    with pytest.raises(ValueError, match="80000000000 bytes.*sparse conversion limit"):
        normalize_transform_output(sparse, "OneHotEncoder")


def test_sparse_training_and_replay_share_the_same_dense_boundary(tmp_path):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(46).integers(1, 5, size=(32, 4)).astype(np.float32)
    y = X[:, 0] - X[:, 1] + np.linspace(0, .1, len(X), dtype=np.float32)
    with nirs4all.run([OneHotEncoder(), StandardScaler(), Ridge()], (X, y), engine="legacy",
                     workspace_path=tmp_path, verbose=0, save_charts=False, refit=False) as result:
        row = result.predictions.filter_predictions(partition="train", load_arrays=True)[0]
        chain_id = row["chain_id"]
    expected = make_pipeline(OneHotEncoder(sparse_output=False), StandardScaler(), Ridge()).fit(X, y).predict(X)
    with WorkspaceStore.open_readonly(tmp_path) as store:
        actual = store.replay_chain(chain_id, X)
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        archive = store.export_chain(chain_id, tmp_path / "sparse.n4a")
    predicted = nirs4all.predict(model=archive, data=X, engine="legacy").y_pred
    np.testing.assert_allclose(np.asarray(predicted).ravel(), expected, atol=1e-5)
