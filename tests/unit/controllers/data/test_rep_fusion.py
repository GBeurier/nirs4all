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


def _dataset_with_missing_source() -> RawMultiSourceDataset:
    return RawMultiSourceDataset.from_sources(
        RepetitionSpec(sample_id="sid", link_by="sid", missing_source_policy="drop_incomplete"),
        {"A": np.array([[1.0], [3.0]]), "B": np.array([[10.0]])},
        {"A": ["S1", "S2"], "B": ["S1"]},
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
    override = context.custom["dataset_override"]
    assert isinstance(override, SpectroDataset)
    np.testing.assert_allclose(override.x({"partition": "train"}), np.array([[1.0, 10.0], [3.0, 20.0]]))
    assert context.custom["relation_materialization_manifest"]["fingerprint"] == output.metadata["materialization_manifest"]["fingerprint"]
    assert output.metadata["representation"] == "per_source_aggregate"
    assert output.metadata["shape"] == [2, 2]
    assert output.metadata["materialization_manifest"]["representation_plan"]["representation"] == "per_source_aggregate"
    assert output.metadata["dataset_override"] is True


def test_rep_fusion_controller_materializes_masked_stack_with_mask_features():
    controller = RepFusionController()
    context, output = controller.execute(
        _step({"representation": "stack_padded_masked", "missing_source_policy": "nan"}),
        _dataset_with_missing_source(),
        ExecutionContext(),
        RuntimeContext(),
        mode="predict",
    )

    override = context.custom["dataset_override"]
    assert isinstance(override, SpectroDataset)
    np.testing.assert_allclose(
        override.x({"partition": "train"}),
        np.array([[1.0, 10.0, 1.0, 1.0], [3.0, 0.0, 1.0, 0.0]]),
    )
    assert output.metadata["materialization_manifest"]["has_feature_mask"] is True
    assert output.metadata["materialization_manifest"]["model_shape"] == [2, 4]


def test_rep_fusion_controller_rejects_legacy_dataset_until_staged():
    controller = RepFusionController()
    with pytest.raises(ValueError, match="RawMultiSourceDataset"):
        controller.execute(_step("per_source_aggregate"), SpectroDataset(), ExecutionContext(), RuntimeContext())


def test_rep_fusion_controller_rejects_branch_contexts():
    controller = RepFusionController()
    context = ExecutionContext(custom={"branch_contexts": [{"branch_id": 0}]})

    with pytest.raises(ValueError, match="before branch"):
        controller.execute(_step("per_source_aggregate"), _dataset(), context, RuntimeContext())
