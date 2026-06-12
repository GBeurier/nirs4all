"""Tests for the rep_fusion controller."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.controllers.data.rep_fusion import RepFusionController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.raw_multisource import RawMultiSourceDataset
from nirs4all.data.relations import RepetitionSpec
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


def _step(value):
    return ParsedStep(
        operator=None,
        keyword="rep_fusion",
        step_type=StepType.WORKFLOW,
        original_step={"rep_fusion": value},
        metadata={},
    )


def _dataset() -> RawMultiSourceDataset:
    return RawMultiSourceDataset.from_sources(
        RepetitionSpec(sample_id="sid", link_by="sid"),
        {"A": np.array([[1.0], [3.0]]), "B": np.array([[10.0], [20.0]])},
        {"A": ["S1", "S2"], "B": ["S1", "S2"]},
    )


def test_rep_fusion_controller_matches_and_supports_prediction_mode():
    assert RepFusionController.matches({"rep_fusion": "per_source_aggregate"}, None, "rep_fusion")
    assert RepFusionController.supports_prediction_mode() is True


def test_rep_fusion_controller_materializes_raw_multisource_dataset():
    controller = RepFusionController()
    context, output = controller.execute(
        _step("per_source_aggregate"),
        _dataset(),
        ExecutionContext(),
        RuntimeContext(),
        mode="predict",
    )
    assert isinstance(context, ExecutionContext)
    assert output.metadata["representation"] == "per_source_aggregate"
    assert output.metadata["shape"] == [2, 2]
    assert output.metadata["materialization_manifest"]["representation_plan"]["representation"] == "per_source_aggregate"


def test_rep_fusion_controller_rejects_legacy_dataset_until_staged():
    controller = RepFusionController()
    with pytest.raises(ValueError, match="RawMultiSourceDataset"):
        controller.execute(_step("per_source_aggregate"), SpectroDataset(), ExecutionContext(), RuntimeContext())
