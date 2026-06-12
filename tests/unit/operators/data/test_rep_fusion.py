"""Tests for the rep_fusion operator configuration."""

from __future__ import annotations

import numpy as np

from nirs4all.data.raw_multisource import RawMultiSourceDataset, RepresentationPlan
from nirs4all.data.relations import RepetitionSpec
from nirs4all.operators.data.rep_fusion import RepFusionConfig


def _dataset() -> RawMultiSourceDataset:
    return RawMultiSourceDataset.from_sources(
        RepetitionSpec(sample_id="sid", link_by="sid"),
        {"A": np.array([[1.0], [3.0]]), "B": np.array([[10.0], [20.0]])},
        {"A": ["S1", "S2"], "B": ["S1", "S2"]},
    )


def test_rep_fusion_config_parses_string():
    config = RepFusionConfig.from_step_value("per_source_aggregate")
    assert config.representation == "per_source_aggregate"
    np.testing.assert_allclose(config.materialize(_dataset()).X, [[1.0, 10.0], [3.0, 20.0]])


def test_rep_fusion_config_parses_nested_plan_manifest():
    plan = RepresentationPlan("stack_padded_masked", max_total_rows=2, missing_source_policy="nan")
    config = RepFusionConfig.from_step_value({"representation_plan": plan.to_dict()})
    assert config.plan == plan
    assert config.to_dict()["representation_plan"]["fingerprint"] == plan.fingerprint()
