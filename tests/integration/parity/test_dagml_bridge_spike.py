"""dag-ml bridge spike: the nirs4all gate-zero pipeline lowers to a dag-ml graph.

Compile-only proof of the nirs4all → dag-ml DSL frontend (migration gap #1). The
whole module is skipped when dag-ml is not importable (a core dependency since the
ADR-17 cutover — only a broken/partial install would lack it), so it degrades to a
skip rather than erroring. Execution via host controllers is a later phase; here we
only assert the DSL lowers to a valid graph.
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml_bridge import (
    build_dagml_plan,
    compile_with_dagml,
    controller_manifests,
    pipeline_to_dsl,
)

from ._registry import get

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not importable (core dependency; broken install?)")


def test_vertical_slice_compiles_to_dagml_graph() -> None:
    """baseline_vertical_slice lowers to transform + y_transform + model nodes."""
    case = get("baseline_vertical_slice")
    artifact = compile_with_dagml(case.pipeline, dsl_id=case.name)

    graph = artifact.graph.to_dict()
    kinds = [node["kind"] for node in graph["nodes"]]
    assert "transform" in kinds, kinds
    assert "y_transform" in kinds, kinds
    assert "model" in kinds, kinds
    # The ShuffleSplit splitter is consumed into the campaign plan (a controller
    # call producing a FoldSet), never a graph node — dag-ml's invariant.
    assert "split" not in kinds and "splitter" not in kinds, kinds
    assert artifact.campaign_template is not None


def test_dsl_shape_for_vertical_slice() -> None:
    """The serialized DSL has the expected compat step objects (no dag-ml needed)."""
    case = get("baseline_vertical_slice")
    dsl = pipeline_to_dsl(case.pipeline, dsl_id=case.name)
    steps = dsl["pipeline"]
    assert dsl["id"] == case.name
    assert any("class" in s and "model" not in s and "y_processing" not in s for s in steps)
    assert any("y_processing" in s for s in steps)
    # The model is serialized by fully-qualified class name (like transforms), so any sklearn-style
    # estimator resolves by import rather than a hardcoded short-name table.
    assert any(isinstance(s.get("model"), str) and s["model"].endswith("PLSRegression") for s in steps)


def test_unsupported_step_is_flagged() -> None:
    """Constructs the spike does not cover fail loudly, naming the keyword."""
    with pytest.raises(NotImplementedError):
        pipeline_to_dsl([{"branch": ["a", "b"]}])


def test_vertical_slice_controller_manifests_validate() -> None:
    """The node-kind manifests validate individually and as a list."""
    import dag_ml

    manifests = controller_manifests()
    # Two model-kind manifests: the base model controller (generic catch-all) + the stacking meta-model
    # controller (bound by metadata.controller_id, declaring consumes_oof_predictions); the others are
    # one each. Distinguished by controller_id (the kind alone is no longer unique for model).
    assert sorted(m["operator_kind"] for m in manifests) == ["model", "model", "prediction_join", "transform", "y_transform"]
    assert sorted(m["controller_id"] for m in manifests) == [
        "controller:nirs4all.merge_concat", "controller:nirs4all.meta_model", "controller:nirs4all.model",
        "controller:nirs4all.transform", "controller:nirs4all.y_transform",
    ]
    for manifest in manifests:
        dag_ml.ControllerManifest(manifest)  # raises on an invalid manifest
    dag_ml.ControllerManifests(manifests)  # raises on an invalid list / duplicate controller_id
    # A prediction/artifact output port forces the matching emits_* capability (both model controllers).
    for model in (m for m in manifests if m["operator_kind"] == "model"):
        assert {"emits_predictions", "emits_artifacts"} <= set(model["capabilities"])
    # The meta-model controller declares it consumes OOF (so the base→meta requires_oof edge is permitted).
    meta = next(m for m in manifests if m["controller_id"] == "controller:nirs4all.meta_model")
    assert "consumes_oof_predictions" in meta["capabilities"]


def test_vertical_slice_builds_execution_plan() -> None:
    """baseline_vertical_slice lowers → compiles-with-controllers → builds a structured plan."""
    case = get("baseline_vertical_slice")
    plan = build_dagml_plan(case.pipeline, plan_id=f"plan:{case.name}", dsl_id=case.name)

    d = plan.to_dict()  # .json() is a method; introspect the parsed dict
    for key in (
        "id", "graph_plan", "campaign", "node_plans", "controller_manifests",
        "variants", "fold_set", "graph_fingerprint", "campaign_fingerprint", "controller_fingerprint",
    ):
        assert key in d, key
    assert d["id"] == f"plan:{case.name}"

    # The graph carries the three node kinds; the splitter is NOT a node.
    node_kinds = [n["kind"] for n in d["graph_plan"]["graph"]["nodes"]]
    assert "transform" in node_kinds and "y_transform" in node_kinds and "model" in node_kinds
    assert "split" not in node_kinds and "splitter" not in node_kinds

    # The ShuffleSplit rode into the campaign as a split_invocation, never an inner CV.
    assert d["campaign"].get("split_invocation") is not None
    assert d["campaign"].get("inner_cv") is None

    # Our manifests resolved onto the nodes by alias selector (binding actually fired):
    # the plan's controller-manifest map and each node_plan's controller_id agree.
    assert set(d["controller_manifests"]) == {
        "controller:nirs4all.transform",
        "controller:nirs4all.y_transform",
        "controller:nirs4all.model",
    }
    bound = {p["kind"]: p["controller_id"] for p in d["node_plans"].values()}
    assert bound == {
        "transform": "controller:nirs4all.transform",
        "y_transform": "controller:nirs4all.y_transform",
        "model": "controller:nirs4all.model",
    }

    # Fingerprints are non-empty determinism anchors.
    for fp in ("graph_fingerprint", "campaign_fingerprint", "controller_fingerprint"):
        assert isinstance(d[fp], str) and d[fp]


def test_bare_scaler_x_step_binds_as_transform_not_y_transform() -> None:
    """A bare sklearn scaler used as an X step is a transform, never a y_transform.

    Regression guard: generic scalers (MinMaxScaler/StandardScaler/RobustScaler)
    are X-transforms or y-transforms purely by DSL position (bare step vs the
    ``{"y_processing": ...}`` wrapper), not by class name. Binding by node kind
    (empty operator_selectors) must keep a bare X-scaler out of the y_transform
    role, otherwise it is silently disconnected from the model path.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import ShuffleSplit
    from sklearn.preprocessing import MinMaxScaler

    plan = build_dagml_plan([MinMaxScaler(), ShuffleSplit(n_splits=3), {"model": PLSRegression(n_components=2)}])
    d = plan.to_dict()
    assert "y_transform" not in [n["kind"] for n in d["graph_plan"]["graph"]["nodes"]]
    bound = {p["kind"]: p["controller_id"] for p in d["node_plans"].values()}
    assert bound == {
        "transform": "controller:nirs4all.transform",
        "model": "controller:nirs4all.model",
    }
