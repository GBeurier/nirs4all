"""Parity tests for the dag-ml node runner (migration phase 2b-ii.1).

Proves the host-controller execution core produces NUMERICALLY CORRECT real predictions
through the dag-ml NodeTask/NodeResult contract: a model node fits the real operator on
real SpectroDataset rows (via the resolver) and its FIT_CV validation predictions match a
direct sklearn fit/predict; REFIT→PREDICT round-trips a persisted model. Scoped to a
model-on-raw-features graph (cross-node feature chaining is the deferred A3 gap).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.node_runner import run_node
from nirs4all.pipeline.dagml.resolver import MaterializationResolver

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not installed (nirs4all[dagml])")


@pytest.fixture(scope="module")
def slice_fixture():
    """A model-on-raw-features plan + resolver + node lookup over the regression dataset."""
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    identity = mint_identity(dataset)
    resolver = MaterializationResolver(dataset, identity)
    pipeline = [ShuffleSplit(n_splits=3, random_state=42), {"model": PLSRegression(n_components=5)}]
    plan = build_dagml_plan(pipeline, plan_id="p", dsl_id="model_only").to_dict()
    nodes = {node["id"]: node for node in plan["graph_plan"]["graph"]["nodes"]}
    model_id = next(node_id for node_id, node in nodes.items() if node["kind"] == "model")
    node_plan = plan["node_plans"][model_id]
    train_ints = dataset.index_column("sample", {"partition": "train"})
    test_ints = dataset.index_column("sample", {"partition": "test"})
    return {
        "dataset": dataset, "identity": identity, "resolver": resolver,
        "node_lookup": lambda node_id: nodes[node_id], "node_plan": node_plan,
        "train_ints": train_ints, "test_ints": test_ints,
    }


def test_fit_cv_predictions_match_direct_sklearn(slice_fixture) -> None:
    f = slice_fixture
    train_ints, val_ints = f["train_ints"][:90], f["train_ints"][90:120]
    to_wire = f["identity"].to_wire
    task = {
        "phase": "FIT_CV", "fold_id": "fold0", "run_id": "r", "variant_id": None,
        "node_plan": f["node_plan"],
        "data_views": {
            "data:x": {"partition": "fold_train", "sample_ids": [to_wire(i) for i in train_ints]},
            "data:x:validation": {"partition": "fold_validation", "sample_ids": [to_wire(i) for i in val_ints]},
        },
    }
    result = run_node(task, f["resolver"], f["node_lookup"], {})
    block = result["predictions"][0]
    assert block["partition"] == "validation" and block["fold_id"] == "fold0"
    assert block["sample_ids"] == [to_wire(i) for i in val_ints]

    ds = f["dataset"]
    expected_model = PLSRegression(n_components=5)
    expected_model.fit(np.asarray(ds.x({"sample": train_ints}, layout="2d")), np.asarray(ds.y({"sample": train_ints})))
    x_val = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"))[0] for i in val_ints])
    expected = np.asarray(expected_model.predict(x_val), dtype=float).reshape(len(val_ints), -1)
    assert np.allclose(np.asarray(block["values"], dtype=float), expected, atol=1e-9)


def test_refit_then_predict_round_trips_the_model(slice_fixture) -> None:
    f = slice_fixture
    to_wire = f["identity"].to_wire
    store: dict = {}

    refit = run_node(
        {"phase": "REFIT", "run_id": "r", "variant_id": None, "node_plan": f["node_plan"],
         "data_views": {"data:x": {"partition": "full_train", "sample_ids": [to_wire(i) for i in f["train_ints"]]}}},
        f["resolver"], f["node_lookup"], store,
    )
    assert refit["predictions"][0]["partition"] == "final"
    artifact_id = next(iter(refit["artifact_handles"]))
    assert artifact_id in store  # the fitted model was persisted

    predict = run_node(
        {"phase": "PREDICT", "run_id": "r", "variant_id": None, "node_plan": f["node_plan"],
         "replay_artifact_id": artifact_id,
         "data_views": {"data:x": {"partition": "predict", "sample_ids": [to_wire(i) for i in f["test_ints"]]}}},
        f["resolver"], f["node_lookup"], store,
    )
    block = predict["predictions"][0]
    assert block["partition"] == "final"
    assert block["sample_ids"] == [to_wire(i) for i in f["test_ints"]]

    ds = f["dataset"]
    x_test = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"))[0] for i in f["test_ints"]])
    expected = np.asarray(store[artifact_id].predict(x_test), dtype=float).reshape(len(f["test_ints"]), -1)
    assert np.allclose(np.asarray(block["values"], dtype=float), expected, atol=1e-9)
