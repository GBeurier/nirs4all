"""dag-ml bridge spike: the nirs4all gate-zero pipeline lowers to a dag-ml graph.

Compile-only proof of the nirs4all → dag-ml DSL frontend (migration gap #1). The
whole module is skipped when dag-ml is not installed (``nirs4all[dagml]``), so it
never breaks environments without the optional engine. Execution via host
controllers is a later phase; here we only assert the DSL lowers to a valid graph.
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.dagml_bridge import compile_with_dagml, pipeline_to_dsl

from ._registry import get

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not installed (nirs4all[dagml])")


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
    assert any(s.get("model") == "PLSRegression" for s in steps)


def test_unsupported_step_is_flagged() -> None:
    """Constructs the spike does not cover fail loudly, naming the keyword."""
    with pytest.raises(NotImplementedError):
        pipeline_to_dsl([{"branch": ["a", "b"]}])
