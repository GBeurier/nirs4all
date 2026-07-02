"""Parity tests for the dag-ml node runner (migration phase 2b-ii.1).

Proves the host-controller execution core produces NUMERICALLY CORRECT real predictions
through the dag-ml NodeTask/NodeResult contract: a model node fits the real operator on
real SpectroDataset rows (via the resolver) and its FIT_CV validation predictions match a
direct sklearn fit/predict; REFIT→PREDICT round-trips a persisted model. Scoped to a
model-on-raw-features graph (cross-node feature chaining is the deferred A3 gap).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.node_runner import _resolve_finetune_best_params, run_node
from nirs4all.pipeline.dagml.process_adapter import describe, run_jsonl_loop
from nirs4all.pipeline.dagml.resolver import MaterializationResolver

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not importable (core dependency; broken install?)")


def test_finetune_best_params_are_cached_per_node_variant(monkeypatch) -> None:
    """The native callback runs Optuna once and reuses the same best params for fold/refit fits."""
    from sklearn.ensemble import RandomForestRegressor

    from nirs4all.optimization import optuna as optuna_mod

    calls: list[dict] = []

    class FakeOptunaManager:
        def finetune(self, *args, **kwargs):
            calls.append(kwargs)
            return optuna_mod.FinetuneResult(best_params={"n_estimators": 23}, best_value=1.0, n_trials=1)

    class FakeResolver:
        _dataset = object()

        def partition_wire_ids(self, partition):
            assert partition == "train"
            return ["s0", "s1", "s2", "s3"]

        def resolve_features(self, ids, include_augmented=True):
            assert ids == ["s0", "s1", "s2", "s3"]
            return {"values": np.arange(12, dtype=np.float32).reshape(4, 3)}

        def resolve_targets(self, ids):
            assert ids == ["s0", "s1", "s2", "s3"]
            return {"values": np.array([0.0, 1.0, 0.0, 1.0])}

    monkeypatch.setattr(optuna_mod, "OptunaManager", FakeOptunaManager)
    graph_node = {
        "metadata": {
            "nirs4all_finetune_params": {
                "approach": "single",
                "n_trials": 1,
                "model_params": {"n_estimators": ["int", 10, 40]},
            }
        }
    }
    store: dict = {}

    first = _resolve_finetune_best_params(
        graph_node=graph_node,
        node_id="model0",
        variant_label="base",
        model=RandomForestRegressor(random_state=42),
        upstream=[],
        resolver=FakeResolver(),
        model_store=store,
    )
    second = _resolve_finetune_best_params(
        graph_node=graph_node,
        node_id="model0",
        variant_label="base",
        model=RandomForestRegressor(random_state=42),
        upstream=[],
        resolver=FakeResolver(),
        model_store=store,
    )
    assert first == second == {"n_estimators": 23}
    assert len(calls) == 1


def _require_methods_snv_available() -> None:
    from nirs4all.operators.methods import n4m_ops

    status = n4m_ops.methods_binding_status()
    library_path = status.get("library_path")
    blockers = []
    if not status.get("snv_available"):
        blockers.append("SNV binding")
    if not status.get("abi_version"):
        blockers.append("n4m ABI")
    if not library_path or not Path(str(library_path)).exists():
        blockers.append("loadable libn4m")
    if not blockers:
        return
    message = status["message"] or f"nirs4all-methods SNV route unavailable: {', '.join(blockers)}"
    mitigation = status.get("mitigation")
    if mitigation:
        message = f"{message} {mitigation}".strip()
    if os.environ.get("NIRS4ALL_REQUIRE_N4M") == "1":
        pytest.fail(message, pytrace=True)
    pytest.skip(message)


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
    handle = refit["artifact_handles"][next(iter(refit["artifact_handles"]))]["handle"]
    assert isinstance(handle, int) and handle in store  # the fitted model was persisted under its u64 handle

    # PREDICT recomputes the same deterministic handle from node+variant (persistent worker).
    predict = run_node(
        {"phase": "PREDICT", "run_id": "r", "variant_id": None, "node_plan": f["node_plan"],
         "data_views": {"data:x": {"partition": "predict", "sample_ids": [to_wire(i) for i in f["test_ints"]]}}},
        f["resolver"], f["node_lookup"], store,
    )
    block = predict["predictions"][0]
    assert block["partition"] == "final"
    assert block["sample_ids"] == [to_wire(i) for i in f["test_ints"]]

    ds = f["dataset"]
    x_test = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"))[0] for i in f["test_ints"]])
    expected = np.asarray(store[handle]["estimator"].predict(x_test), dtype=float).reshape(len(f["test_ints"]), -1)
    assert np.allclose(np.asarray(block["values"], dtype=float), expected, atol=1e-9)


def test_jsonl_loop_handshake_and_task(slice_fixture) -> None:
    """The JSONL frame loop acks init/close and returns a result frame with real predictions."""
    f = slice_fixture
    to_wire = f["identity"].to_wire
    train, val = f["train_ints"][:90], f["train_ints"][90:120]
    task = {
        "phase": "FIT_CV", "fold_id": "fold0", "run_id": "r", "variant_id": None, "node_plan": f["node_plan"],
        "data_views": {
            "data:x": {"partition": "fold_train", "sample_ids": [to_wire(i) for i in train]},
            "data:x:validation": {"partition": "fold_validation", "sample_ids": [to_wire(i) for i in val]},
        },
    }
    store: dict = {}
    infile = io.StringIO("\n".join(json.dumps(frame) for frame in (
        {"type": "init"}, {"type": "task", "task": task}, {"type": "close"},
    )) + "\n")
    outfile = io.StringIO()
    run_jsonl_loop(infile, outfile, lambda t: run_node(t, f["resolver"], f["node_lookup"], store))

    frames = [json.loads(line) for line in outfile.getvalue().splitlines() if line.strip()]
    assert frames[0] == {"type": "ack", "schema_version": 1, "status": "initialized"}
    assert frames[1]["type"] == "result"
    block = frames[1]["result"]["predictions"][0]
    assert block["partition"] == "validation" and block["sample_ids"] == [to_wire(i) for i in val]
    assert frames[2] == {"type": "ack", "schema_version": 1, "status": "closed"}

    description = describe()
    assert "jsonl" in description["supported_modes"]
    # worker_env is required by dag-ml-cli for --persistent mode (verified end-to-end).
    assert {"node_task_json_v1", "worker_env", "persistent_workers"} <= set(description["capabilities"])


def test_fit_cv_applies_upstream_snv_chain(slice_fixture) -> None:
    """A model node with an upstream SNV transform fits the real Pipeline(SNV, PLS) on fold-train (A3)."""
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    f = slice_fixture
    plan = build_dagml_plan([StandardNormalVariate(), {"model": PLSRegression(n_components=5)}], plan_id="p", dsl_id="snv_pls").to_dict()
    graph = plan["graph_plan"]["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    model_id = next(node_id for node_id, node in nodes.items() if node["kind"] == "model")
    train, val = f["train_ints"][:90], f["train_ints"][90:120]
    to_wire = f["identity"].to_wire
    task = {
        "phase": "FIT_CV", "fold_id": "fold0", "run_id": "r", "variant_id": None, "node_plan": plan["node_plans"][model_id],
        "data_views": {
            "data:x": {"partition": "fold_train", "sample_ids": [to_wire(i) for i in train]},
            "data:x:validation": {"partition": "fold_validation", "sample_ids": [to_wire(i) for i in val]},
        },
    }
    got = np.asarray(run_node(task, f["resolver"], nodes.__getitem__, {}, graph["edges"])["predictions"][0]["values"], dtype=float)

    ds = f["dataset"]
    # Compare on the dataset's NATIVE storage dtype (float32): the resolver/node_runner preserve it
    # (no .tolist()/float64 widening), so the oracle fits/predicts X at float32 too — matching dtype
    # AND fold order makes this an exact (FP-level) parity check, not a float32-vs-float64 comparison.
    expected_pipe = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    expected_pipe.fit(np.asarray(ds.x({"sample": train}, layout="2d")), np.asarray(ds.y({"sample": train}), dtype=float))
    x_val = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"))[0] for i in val])
    expected = np.asarray(expected_pipe.predict(x_val), dtype=float).reshape(len(val), -1)
    assert np.allclose(got, expected, atol=1e-6)  # same dtype + same fold order -> exact up to FP

    # Without edges the chain is NOT applied (raw features) — proves the SNV step is load-bearing.
    raw = np.asarray(run_node(task, f["resolver"], nodes.__getitem__, {}, None)["predictions"][0]["values"], dtype=float)
    assert not np.allclose(raw, expected, atol=1e-2)


def test_fit_cv_methods_snv_opt_in_matches_python_oracle(slice_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in native SNV route preserves Python-reference SNV->PLS predictions."""
    from sklearn.pipeline import make_pipeline

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml import node_runner as node_runner_module
    from nirs4all.pipeline.dagml.operator_routing import route_graph_node as real_route_graph_node
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    _require_methods_snv_available()
    monkeypatch.setenv("N4A_DAGML_METHODS_SNV", "1")
    routed_transforms = []

    def recording_route_graph_node(node: dict, *, variant_overrides: dict | None = None) -> object:
        routed = real_route_graph_node(node, variant_overrides=variant_overrides)
        if node.get("kind") == "transform":
            routed_transforms.append(f"{type(routed).__module__}.{type(routed).__name__}")
        return routed

    monkeypatch.setattr(node_runner_module, "route_graph_node", recording_route_graph_node)

    f = slice_fixture
    plan = build_dagml_plan([StandardNormalVariate(), {"model": PLSRegression(n_components=5)}], plan_id="p", dsl_id="methods_snv_pls").to_dict()
    graph = plan["graph_plan"]["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    model_id = next(node_id for node_id, node in nodes.items() if node["kind"] == "model")
    train, val = f["train_ints"][:90], f["train_ints"][90:120]
    to_wire = f["identity"].to_wire
    task = {
        "phase": "FIT_CV", "fold_id": "fold0", "run_id": "r", "variant_id": None, "node_plan": plan["node_plans"][model_id],
        "data_views": {
            "data:x": {"partition": "fold_train", "sample_ids": [to_wire(i) for i in train]},
            "data:x:validation": {"partition": "fold_validation", "sample_ids": [to_wire(i) for i in val]},
        },
    }

    result = run_node(task, f["resolver"], nodes.__getitem__, {}, graph["edges"])
    block = result["predictions"][0]
    assert block["partition"] == "validation"
    assert block["sample_ids"] == [to_wire(i) for i in val]
    assert routed_transforms == ["nirs4all.operators.methods.n4m_ops.MethodsSNV"]
    got = np.asarray(block["values"], dtype=float)

    ds = f["dataset"]
    expected_pipe = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    x_train = np.asarray(ds.x({"sample": train}, layout="2d"), dtype=np.float64)
    expected_pipe.fit(x_train, np.asarray(ds.y({"sample": train}), dtype=float))
    x_val = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"), dtype=np.float64)[0] for i in val])
    expected = np.asarray(expected_pipe.predict(x_val), dtype=float).reshape(len(val), -1)
    assert np.allclose(got, expected, atol=1e-6)


def test_fit_cv_applies_snv_and_y_processing_chain(slice_fixture) -> None:
    """SNV + y_processing + PLS: model node applies the X-chain AND target scaling+inverse (A3.2).

    Uses a NON-affine y transform (PowerTransformer) so the effect is observable — affine
    scalers (MinMaxScaler/StandardScaler) are mathematically no-ops for a linear model's
    inverse-transformed predictions (a useful parity fact, covered by the e2e test).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PowerTransformer

    from nirs4all.operators.transforms.scalers import StandardNormalVariate
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    f = slice_fixture
    pipeline = [StandardNormalVariate(), {"y_processing": PowerTransformer()}, {"model": PLSRegression(n_components=5)}]
    plan = build_dagml_plan(pipeline, plan_id="p", dsl_id="vslice").to_dict()
    graph = plan["graph_plan"]["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    model_id = next(node_id for node_id, node in nodes.items() if node["kind"] == "model")
    y_transform_node = next(node for node in graph["nodes"] if node["kind"] == "y_transform")
    train, val = f["train_ints"][:90], f["train_ints"][90:120]
    to_wire = f["identity"].to_wire
    task = {
        "phase": "FIT_CV", "fold_id": "fold0", "run_id": "r", "variant_id": None, "node_plan": plan["node_plans"][model_id],
        "data_views": {
            "data:x": {"partition": "fold_train", "sample_ids": [to_wire(i) for i in train]},
            "data:x:validation": {"partition": "fold_validation", "sample_ids": [to_wire(i) for i in val]},
        },
    }
    got = np.asarray(run_node(task, f["resolver"], nodes.__getitem__, {}, graph["edges"], y_transform_node)["predictions"][0]["values"], dtype=float)

    # Manual nirs4all-equivalent: fit y transform on train y, fit SNV->PLS on transformed y, inverse predictions.
    ds = f["dataset"]
    y_train = np.asarray(ds.y({"sample": train}), dtype=float).reshape(-1, 1)
    ytf = PowerTransformer().fit(y_train)
    pipe = make_pipeline(StandardNormalVariate(), PLSRegression(n_components=5))
    # X on the dataset's NATIVE storage dtype (float32) — the engine no longer widens to float64, so the
    # oracle must match it for an exact parity check (y stays float64, as the engine feeds y).
    pipe.fit(np.asarray(ds.x({"sample": train}, layout="2d")), ytf.transform(y_train).ravel())
    x_val = np.stack([np.asarray(ds.x({"sample": [i]}, layout="2d"))[0] for i in val])
    expected = ytf.inverse_transform(np.asarray(pipe.predict(x_val), dtype=float).reshape(len(val), -1))
    assert np.allclose(got, expected, atol=1e-6)
    # Without the y transform the (non-affine) target scaling is missing — predictions differ.
    no_ytf = np.asarray(run_node(task, f["resolver"], nodes.__getitem__, {}, graph["edges"], None)["predictions"][0]["values"], dtype=float)
    assert not np.allclose(no_ytf, expected, atol=1e-2)
