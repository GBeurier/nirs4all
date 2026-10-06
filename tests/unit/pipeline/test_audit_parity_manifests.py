"""Strict native plans attest the ports used by stacking and residual fusion."""

import dag_ml
import pytest

from nirs4all.pipeline import dagml_bridge


@pytest.mark.parametrize("fallback", [False, True], ids=["derived", "fallback"])
@pytest.mark.parametrize("composition", ["original_meta", "residual", "prediction_features", "prediction_fusion"])
def test_native_stacking_ports_build_strict_plan(monkeypatch, fallback, composition):
    if fallback:
        monkeypatch.setattr(dagml_bridge, "_derive_controller_manifests_from_dagml", lambda specs: None)
    model = {"kind": "model", "id": "base", "operator": {"ref": "sklearn.linear_model._ridge.Ridge", "name": "Ridge"}, "params": {}}
    if composition.startswith("prediction_"):
        fusion = composition == "prediction_fusion"
        final = {"kind": "merge", "id": "join", "output_as": "predictions" if fusion else "features", "include_original_data": fusion,
                 "metadata": {"controller_id": "controller:nirs4all.prediction_feature_join"}}
    else:
        residual = composition == "residual"
        final = {"kind": "merge_model", "id": "meta",
                 "operator": {"ref": "nirs4all.residual_learner" if residual else "nirs4all.meta_model", "name": "Ridge"},
                 "params": {}, "include_original_data": True,
                 "metadata": {"controller_id": "controller:nirs4all.residual_learner" if residual else "controller:nirs4all.meta_model"}}
        if residual:
            final["metadata"]["residual_target_execution"] = "nested_oof_v1"
    dsl = {"id": "port-witness", "steps": [model, final]}
    manifests = dagml_bridge.controller_manifests(dsl)
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    plan = dag_ml.build_execution_plan("plan:port-witness", artifact.graph, artifact.campaign_template, manifests).to_dict()
    nodes = {node["id"]: node for node in plan["graph_plan"]["graph"]["nodes"]}
    if composition == "original_meta":
        assert {port["name"] for port in nodes["meta"]["ports"]["inputs"]} == {"oof", "x_original"}
        assert plan["node_plans"]["meta"]["controller_id"] == "controller:nirs4all.meta_model"
    else:
        join_id = "meta.residual_fusion" if composition == "residual" else "join"
        output = nodes[join_id]["ports"]["outputs"][0]
        prediction = composition in {"residual", "prediction_fusion"}
        assert output["kind"] == ("prediction" if prediction else "data")
        assert output["name"] == ("prediction" if prediction else "x_out")
        assert plan["node_plans"][join_id]["controller_id"] == "controller:nirs4all.prediction_feature_join"
        if composition == "prediction_fusion":
            assert "x_original" in {port["name"] for port in nodes[join_id]["ports"]["inputs"]}
