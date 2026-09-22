import json

import numpy as np
import pytest
from sklearn.decomposition import SparseCoder
from sklearn.linear_model import Ridge

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.bool_])
def test_numeric_array_roundtrip_retains_shape_and_dtype(dtype):
    original = np.array([[0, 1], [1, 0]], dtype=dtype)
    payload = json.loads(json.dumps(serialize_component(original)))
    restored = deserialize_component(payload, strict_imports=True)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)


def test_old_list_configs_still_deserialize_with_array_type_hint():
    restored = deserialize_component([[1., 0.], [0., 1.]], infer_type=np.ndarray)
    assert isinstance(restored, np.ndarray)
    np.testing.assert_array_equal(restored, np.eye(2))


def test_sparse_coder_instance_survives_pipeline_serialization_and_replay(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(90).normal(size=(32, 4))
    y = X[:, 0] + X[:, 1] * .2
    coder = SparseCoder(dictionary=np.eye(4), transform_n_nonzero_coefs=2)
    restored = deserialize_component(json.loads(json.dumps(serialize_component(coder))), strict_imports=True)
    expected = coder.transform(X)
    np.testing.assert_allclose(restored.transform(X), expected)
    model = Ridge().fit(expected, y)
    with nirs4all.run([coder, Ridge()], (X, y), engine="legacy", workspace_path=tmp_path,
                     verbose=0, save_charts=False, refit=False) as result:
        row = result.predictions.filter_predictions(partition="train", load_arrays=True)[0]
        np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), model.predict(expected), atol=1e-5)
    with WorkspaceStore.open_readonly(tmp_path) as store:
        np.testing.assert_allclose(store.replay_chain(row["chain_id"], X), model.predict(expected), atol=1e-5)
