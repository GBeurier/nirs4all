"""Host materialization of a DAG-ML-attested prediction feature matrix."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.pipeline.dagml import node_runner


def test_prediction_feature_handle_preserves_native_column_and_sample_order() -> None:
    task = {
        "run_id": "run:join",
        "phase": "FIT_CV",
        "fold_id": "fold:0",
        "node_plan": {
            "node_id": "merge:prediction.features",
            "kind": "prediction_join",
            "controller_id": "controller:nirs4all.prediction_feature_join",
            "controller_version": "test",
            "params_fingerprint": "a" * 64,
        },
        "prediction_feature_matrix": {
            "sample_ids": ["s2", "s1"],
            "columns": ["model:b.pred__y", "model:a.pred__y"],
            "values": [[20.0, 2.0], [10.0, 1.0]],
        },
        "prediction_feature_off_fold_matrix": {
            "sample_ids": ["s3"],
            "columns": ["model:b.pred__y", "model:a.pred__y"],
            "values": [[30.0, 3.0]],
        },
    }
    store: dict[int, object] = {}
    result = node_runner._run_prediction_feature_join_node(task, store)
    assert set(result["outputs"]) == {"out", "x_out"}
    handle = result["outputs"]["x_out"]["handle"]
    chain = node_runner._fitted_input_chain(
        {"input_handles": {"data:x": {"kind": "data", "handle": handle}}}, store,
    )
    assert isinstance(chain, node_runner._PredictionFeatureChain)
    np.testing.assert_array_equal(
        chain.transform_ids(np.zeros((3, 1)), ["s3", "s1", "s2"]),
        [[30.0, 3.0], [10.0, 1.0], [20.0, 2.0]],
    )
    with pytest.raises(ValueError, match="no attested row"):
        chain.transform_ids(np.zeros((1, 1)), ["missing"])
    with pytest.raises(ValueError, match="repeated sample"):
        node_runner._PredictionFeatureChain(
            task["prediction_feature_matrix"],
            {"sample_ids": ["s1"], "columns": task["prediction_feature_matrix"]["columns"], "values": [[1.0, 1.0]]},
        )
