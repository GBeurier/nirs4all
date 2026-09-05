"""The no-split projection never authors CV/OOF evidence."""

import copy

import numpy as np
import pytest

from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml.full_train import _project_full_train
from nirs4all.pipeline.dagml.identity import mint_identity


def _outcome(has_test):
    dataset = SpectroDataset("full-train")
    dataset.add_samples(np.arange(6).reshape(3, 2), {"partition": "train"})
    dataset.add_targets(np.arange(3, dtype=float))
    identity = mint_identity(dataset)
    ids = [identity.to_wire(index) for index in range(3)]
    partitions = [("final", ids[:2], [[0.0], [1.0]])]
    if has_test:
        partitions.append(("test", ids[2:], [[2.0]]))
    reports, predictions, targets = [], [], []
    for partition, samples, values in partitions:
        reports.append({"producer_node": "model", "partition": partition, "fold_id": None,
                        "level": "sample", "metrics": {"rmse": 0.0, "r2": 1.0}})
        predictions.append({"producer_node": "model", "partition": partition, "fold_id": None,
                            "sample_ids": samples, "values": values, "target_names": ["y"]})
        targets.append({"level": "sample", "unit_ids": [{"level": "sample", "id": sample} for sample in samples],
                        "values": values, "target_names": ["y"]})
    return {"scores": {"reports": reports}, "node_results": [{"predictions": predictions, "regression_targets": targets}]}, identity


def _project(outcome, identity):
    return _project_full_train(outcome, identity, dataset_name="full-train", model_id="model",
                               model_name="Ridge", metric="rmse", task_type="regression", config_name="config", artifacts=[])


@pytest.mark.parametrize("has_test", [False, True])
def test_projection_exposes_only_real_scores_and_marks_test_alias(has_test):
    outcome, identity = _outcome(has_test)
    before = copy.deepcopy(outcome)
    result = _project(outcome, identity)
    assert result.execution_engine == "dag-ml"
    assert result.num_predictions == (3 if has_test else 1)
    assert result.cv_best == {}
    assert np.isnan(result.cv_best_score)
    assert result.best
    assert result._dagml_score_set is outcome["scores"]
    assert outcome == before
    rows = result.predictions.filter_predictions(load_arrays=True)
    assert {row["fold_id"] for row in rows} == {"final"}
    for row in rows:
        scope = row["result_metadata"]["evaluation"]
        assert scope["cross_validation"] is False
        assert scope["independent_model_selection_holdout"] is False
        assert scope["test_used_for_validation"] is has_test
        assert scope["training_scope"] == "resubstitution"
        assert row["train_score"] == 0.0
    if has_test:
        val, test = (next(row for row in rows if row["partition"] == part) for part in ("val", "test"))
        np.testing.assert_array_equal(val["y_pred"], test["y_pred"])
        assert val["sample_indices"] == test["sample_indices"]
        assert val["result_metadata"]["native_partition"] == "test"
        assert result.best_score == 0.0
    else:
        assert rows[0]["val_score"] is None
        assert rows[0]["test_score"] is None
        assert np.isnan(result.best_score)


@pytest.mark.parametrize("mutation", ["validation", "fold", "duplicate", "target_identity"])
def test_projection_refuses_ambiguous_or_invented_evidence(mutation):
    outcome, identity = _outcome(True)
    if mutation == "validation":
        outcome["scores"]["reports"][0]["partition"] = "validation"
    elif mutation == "fold":
        outcome["scores"]["reports"][0]["fold_id"] = "fold0"
    elif mutation == "duplicate":
        outcome["scores"]["reports"].append(copy.deepcopy(outcome["scores"]["reports"][0]))
    else:
        outcome["node_results"][0]["regression_targets"][0]["unit_ids"].reverse()
    with pytest.raises(ValueError):
        _project(outcome, identity)
