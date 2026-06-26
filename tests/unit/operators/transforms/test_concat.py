"""Unit tests for ``FeatureConcat`` — the JSON-spec'd horizontal feature concatenation transformer.

``FeatureConcat`` is the single-source replace-mode lowering of nirs4all's ``concat_transform`` onto
the dag-ml engine (backlog #27): it ``np.hstack``-es several sub-transformers' outputs (each fit on the
training rows) into one wider 2D feature matrix — ``sklearn.pipeline.FeatureUnion`` semantics, but
parameterised by a JSON-serializable ``operations`` spec so the whole concat round-trips as one DSL node.
"""

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.operators.transforms.concat import FeatureConcat


@pytest.fixture
def X():
    return np.random.RandomState(0).rand(20, 8)


def _spec(fqn, **params):
    return {"class": fqn, "params": params}


def test_single_ops_hstack_like_feature_union(X):
    """Two single transformers concatenate column-wise, matching the equivalent ``FeatureUnion``."""
    fc = FeatureConcat(operations=[
        _spec("sklearn.preprocessing._data.StandardScaler"),
        _spec("sklearn.preprocessing._data.MinMaxScaler"),
    ])
    out = fc.fit_transform(X)
    assert out.shape == (20, 16)  # 8 + 8

    union = FeatureUnion([("a", StandardScaler()), ("b", MinMaxScaler())])
    assert np.allclose(out, union.fit_transform(X))


def test_column_count_changes(X):
    """A dimensionality-reducer concat produces a narrower-than-input matrix (column count changes)."""
    fc = FeatureConcat(operations=[
        _spec("sklearn.decomposition._pca.PCA", n_components=3),
        _spec("sklearn.decomposition._truncated_svd.TruncatedSVD", n_components=2, algorithm="arpack", random_state=0),
    ])
    assert fc.fit_transform(X).shape == (20, 5)  # 3 + 2


def test_chain_entry_applies_sequentially(X):
    """A chain entry (`[A, B]`) applies ``B(A(X))`` before concatenation, matching a manual sequence."""
    fc = FeatureConcat(operations=[
        _spec("sklearn.preprocessing._data.MinMaxScaler"),
        [_spec("sklearn.preprocessing._data.StandardScaler"), _spec("sklearn.preprocessing._data.MinMaxScaler")],
    ])
    out = fc.fit_transform(X)
    assert out.shape == (20, 16)

    single = MinMaxScaler().fit_transform(X)
    chained = MinMaxScaler().fit_transform(StandardScaler().fit_transform(X))
    assert np.allclose(out, np.hstack([single, chained]))


def test_fit_then_transform_new_rows(X):
    """``transform`` re-applies the fitted sub-transformers to unseen rows (the fold-val/test path)."""
    fc = FeatureConcat(operations=[
        _spec("sklearn.preprocessing._data.StandardScaler"),
        _spec("sklearn.preprocessing._data.MinMaxScaler"),
    ]).fit(X)
    new = np.random.RandomState(1).rand(5, 8)
    out = fc.transform(new)
    assert out.shape == (5, 16)
    # The StandardScaler half uses the TRAIN mean/scale, not the new rows' — a true apply, not a refit.
    train_scaler = StandardScaler().fit(X)
    assert np.allclose(out[:, :8], train_scaler.transform(new))


def test_all_stateless_chain_does_not_raise_not_fitted(X):
    """A chain of nirs4all STATELESS transformers transforms cleanly (no spurious ``NotFittedError``).

    A plain ``make_pipeline`` of all-stateless steps would raise ``NotFittedError`` from
    ``Pipeline.transform``'s ``check_is_fitted``; the ``_Chain`` wrapper avoids that.
    """
    fc = FeatureConcat(operations=[
        [
            _spec("nirs4all.operators.transforms.scalers.StandardNormalVariate"),
            _spec("nirs4all.operators.transforms.scalers.StandardNormalVariate"),
        ],
    ]).fit(X)
    assert fc.transform(X).shape == (20, 8)


def test_empty_operations_rejected(X):
    """An empty ``operations`` spec has nothing to concatenate and is rejected at fit."""
    with pytest.raises(ValueError, match="non-empty"):
        FeatureConcat(operations=[]).fit(X)
    with pytest.raises(ValueError, match="non-empty"):
        FeatureConcat().fit(X)
