"""Canonical branches and extracted meta estimators retain declared controls."""

import json

import dag_ml
import pytest
from sklearn.linear_model import Ridge

from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.run_paths import _canonical_branch_step, _run_by_source_stacking_branch, _stacking_model_metadata
from nirs4all.pipeline.dagml_bridge import controller_manifests


def test_canonical_branch_keeps_fingerprinted_training_and_refit_metadata():
    step = {"model": Ridge(), "train_params": {"alpha": 3}, "refit_params": {"alpha": 7}}
    canonical = _canonical_branch_step(step, "controlled-model")
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers({"id": "metadata", "steps": [canonical]}, controller_manifests()).graph.to_dict()
    metadata = graph["nodes"][0]["metadata"]
    assert metadata["nirs4all_train_params"] == {"alpha": 3}
    assert metadata["nirs4all_refit_params"] == {"alpha": 7}
    assert json.loads(json.dumps(canonical))["metadata"] == metadata


def test_meta_estimator_extraction_does_not_discard_control_declarations():
    pipeline = [{"branch": [[{"model": Ridge()}], [{"model": Ridge(2)}]]}, {"merge": "predictions"},
                {"model": Ridge(0.1), "train_params": {"alpha": 3}, "refit_params": {"alpha": 7}}]
    assert _stacking_model_metadata(pipeline) == {
        "nirs4all_train_params": {"alpha": 3}, "nirs4all_refit_params": {"alpha": 7},
    }


def test_source_prediction_stacking_rejects_inconsistent_source_layout(tmp_path):
    from types import SimpleNamespace

    with pytest.raises(DagMlUnsupported, match="aligned per-source feature counts"):
        _run_by_source_stacking_branch(
            [{"model": Ridge(), "train_params": {"alpha": 3}}], [], None, 2, SimpleNamespace(num_features=[2]),
            "", "", "", tmp_path, "rmse", "regression",
        )
