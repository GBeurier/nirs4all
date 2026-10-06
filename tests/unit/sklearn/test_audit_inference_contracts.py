"""Fitted DAG adapters must replay preprocessing for every inference method."""

import gc
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.sklearn import NIRSPipeline, NIRSPipelineClassifier


@pytest.mark.parametrize("adapter", [NIRSPipeline, NIRSPipelineClassifier])
def test_unfitted_adapter_clone_and_invalid_parameters(adapter):
    original = adapter(fold=2)
    copied = clone(original)
    assert copied.get_params() == {"fold": 2}
    assert not copied.is_fitted_
    with pytest.raises(ValueError, match="Invalid parameters"):
        original.set_params(bogus=1)


def test_dag_classifier_replays_scaler_for_probabilities_and_cleans_export(tmp_path):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(90, 5)) * 100 + 50
    y = (X[:, 0] + X[:, 1] > 100).astype(int)
    result = nirs4all.run([StandardScaler(), KFold(3), {"model": LogisticRegression()}], (X, y),
                         engine="dag-ml", workspace_path=tmp_path / "workspace", verbose=0)
    adapter = NIRSPipelineClassifier.from_result(result)
    transformed = adapter.transform(X)
    assert not np.allclose(transformed, X)
    np.testing.assert_allclose(adapter.predict_proba(X), adapter.model_.predict_proba(transformed))
    np.testing.assert_array_equal(adapter.predict(X), adapter.classes_[adapter.predict_proba(X).argmax(axis=1)])
    directory = Path(adapter._source_path).parent
    assert directory.exists()
    del adapter
    gc.collect()
    assert not directory.exists()
