"""
Tests for the Execution Trace module (Phase 2).

Tests the ExecutionTrace, ExecutionStep, and StepArtifacts classes
for recording and replaying pipeline execution traces.
"""

import pytest
from datetime import datetime

from nirs4all.pipeline.trace import (
    ExecutionTrace,
    ExecutionStep,
    StepArtifacts,
    StepExecutionMode,
)


class TestStepArtifacts:
    """Tests for StepArtifacts dataclass."""

    def test_create_empty(self):
        """Test creating empty StepArtifacts."""
        artifacts = StepArtifacts()

        assert artifacts.artifact_ids == []
        assert artifacts.primary_artifact_id is None
        assert artifacts.fold_artifact_ids == {}
        assert artifacts.metadata == {}

    def test_add_artifact(self):
        """Test adding artifacts."""
        artifacts = StepArtifacts()

        artifacts.add_artifact("0001:1:all")
        assert "0001:1:all" in artifacts.artifact_ids
        assert artifacts.primary_artifact_id is None

        artifacts.add_artifact("0001:1:primary", is_primary=True)
        assert "0001:1:primary" in artifacts.artifact_ids
        assert artifacts.primary_artifact_id == "0001:1:primary"

    def test_add_fold_artifact(self):
        """Test adding fold-specific artifacts."""
        artifacts = StepArtifacts()

        artifacts.add_fold_artifact(0, "0001:4:0")
        artifacts.add_fold_artifact(1, "0001:4:1")

        assert artifacts.fold_artifact_ids == {0: "0001:4:0", 1: "0001:4:1"}
        assert "0001:4:0" in artifacts.artifact_ids
        assert "0001:4:1" in artifacts.artifact_ids

    def test_to_dict(self):
        """Test serialization to dictionary."""
        artifacts = StepArtifacts(
            artifact_ids=["0001:1:all", "0001:4:0"],
            primary_artifact_id="0001:4:0",
            fold_artifact_ids={0: "0001:4:0"},
            metadata={"type": "model"}
        )

        d = artifacts.to_dict()

        assert d["artifact_ids"] == ["0001:1:all", "0001:4:0"]
        assert d["primary_artifact_id"] == "0001:4:0"
        assert d["fold_artifact_ids"] == {0: "0001:4:0"}
        assert d["metadata"] == {"type": "model"}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "artifact_ids": ["0001:1:all"],
            "primary_artifact_id": "0001:1:all",
            "fold_artifact_ids": {"0": "0001:4:0", "1": "0001:4:1"},
            "metadata": {"type": "transformer"}
        }

        artifacts = StepArtifacts.from_dict(data)

        assert artifacts.artifact_ids == ["0001:1:all"]
        assert artifacts.primary_artifact_id == "0001:1:all"
        # Keys should be converted to int
        assert artifacts.fold_artifact_ids == {0: "0001:4:0", 1: "0001:4:1"}

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = StepArtifacts(
            artifact_ids=["a", "b", "c"],
            primary_artifact_id="b",
            fold_artifact_ids={0: "a", 1: "c"},
            metadata={"key": "value"}
        )

        restored = StepArtifacts.from_dict(original.to_dict())

        assert restored.artifact_ids == original.artifact_ids
        assert restored.primary_artifact_id == original.primary_artifact_id
        assert restored.fold_artifact_ids == original.fold_artifact_ids
        assert restored.metadata == original.metadata


class TestExecutionStep:
    """Tests for ExecutionStep dataclass."""

    def test_create_basic(self):
        """Test creating a basic execution step."""
        step = ExecutionStep(
            step_index=1,
            operator_type="transform",
            operator_class="MinMaxScaler"
        )

        assert step.step_index == 1
        assert step.operator_type == "transform"
        assert step.operator_class == "MinMaxScaler"
        assert step.execution_mode == StepExecutionMode.TRAIN

    def test_has_artifacts(self):
        """Test checking for artifacts."""
        step = ExecutionStep(step_index=1)
        assert step.has_artifacts() is False

        step.artifacts.add_artifact("0001:1:all")
        assert step.has_artifacts() is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        step = ExecutionStep(
            step_index=4,
            operator_type="model",
            operator_class="PLSRegression",
            operator_config={"n_components": 10},
            execution_mode=StepExecutionMode.TRAIN,
            branch_path=[0, 1],
            branch_name="branch_snv",
            duration_ms=123.45
        )
        step.artifacts.add_artifact("0001:4:all", is_primary=True)

        d = step.to_dict()

        assert d["step_index"] == 4
        assert d["operator_type"] == "model"
        assert d["operator_class"] == "PLSRegression"
        assert d["operator_config"] == {"n_components": 10}
        assert d["execution_mode"] == "train"
        assert d["branch_path"] == [0, 1]
        assert d["duration_ms"] == 123.45

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "step_index": 2,
            "operator_type": "transform",
            "operator_class": "StandardNormalVariate",
            "execution_mode": "predict",
            "artifacts": {
                "artifact_ids": ["0001:2:all"],
                "primary_artifact_id": None,
                "fold_artifact_ids": {},
                "metadata": {}
            }
        }

        step = ExecutionStep.from_dict(data)

        assert step.step_index == 2
        assert step.operator_type == "transform"
        assert step.operator_class == "StandardNormalVariate"
        assert step.execution_mode == StepExecutionMode.PREDICT
        assert "0001:2:all" in step.artifacts.artifact_ids


class TestExecutionTrace:
    """Tests for ExecutionTrace dataclass."""

    def test_create_empty(self):
        """Test creating empty trace."""
        trace = ExecutionTrace()

        assert len(trace.trace_id) == 12  # Default UUID[:12]
        assert trace.pipeline_uid == ""
        assert trace.steps == []
        assert trace.model_step_index is None
        assert trace.fold_weights is None

    def test_create_with_uid(self):
        """Test creating trace with pipeline UID."""
        trace = ExecutionTrace(
            trace_id="test123",
            pipeline_uid="0001_pls_abc123"
        )

        assert trace.trace_id == "test123"
        assert trace.pipeline_uid == "0001_pls_abc123"

    def test_add_step(self):
        """Test adding steps to trace."""
        trace = ExecutionTrace()

        step1 = ExecutionStep(step_index=1, operator_type="transform")
        step2 = ExecutionStep(step_index=2, operator_type="model")

        trace.add_step(step1)
        trace.add_step(step2)

        assert len(trace.steps) == 2
        assert trace.steps[0].step_index == 1
        assert trace.steps[1].step_index == 2

    def test_get_step(self):
        """Test getting step by index."""
        trace = ExecutionTrace()
        trace.add_step(ExecutionStep(step_index=1, operator_type="transform"))
        trace.add_step(ExecutionStep(step_index=4, operator_type="model"))

        step = trace.get_step(4)
        assert step is not None
        assert step.operator_type == "model"

        missing = trace.get_step(999)
        assert missing is None

    def test_get_steps_before(self):
        """Test getting steps before a given index."""
        trace = ExecutionTrace()
        trace.add_step(ExecutionStep(step_index=1))
        trace.add_step(ExecutionStep(step_index=2))
        trace.add_step(ExecutionStep(step_index=3))
        trace.add_step(ExecutionStep(step_index=4))

        before = trace.get_steps_before(3)

        assert len(before) == 2
        assert all(s.step_index < 3 for s in before)

    def test_get_steps_up_to_model(self):
        """Test getting steps up to model."""
        trace = ExecutionTrace(model_step_index=3)
        trace.add_step(ExecutionStep(step_index=1))
        trace.add_step(ExecutionStep(step_index=2))
        trace.add_step(ExecutionStep(step_index=3))
        trace.add_step(ExecutionStep(step_index=4))

        up_to = trace.get_steps_up_to_model()

        assert len(up_to) == 3
        assert all(s.step_index <= 3 for s in up_to)

    def test_get_artifact_ids(self):
        """Test getting all artifact IDs."""
        trace = ExecutionTrace()

        step1 = ExecutionStep(step_index=1)
        step1.artifacts.add_artifact("0001:1:all")

        step2 = ExecutionStep(step_index=2)
        step2.artifacts.add_artifact("0001:2:0")
        step2.artifacts.add_artifact("0001:2:1")

        trace.add_step(step1)
        trace.add_step(step2)

        artifact_ids = trace.get_artifact_ids()

        assert len(artifact_ids) == 3
        assert "0001:1:all" in artifact_ids
        assert "0001:2:0" in artifact_ids
        assert "0001:2:1" in artifact_ids

    def test_set_model_step(self):
        """Test setting model step with fold weights."""
        trace = ExecutionTrace()

        trace.set_model_step(4, fold_weights={0: 0.52, 1: 0.48})

        assert trace.model_step_index == 4
        assert trace.fold_weights == {0: 0.52, 1: 0.48}

    def test_get_model_artifact_id(self):
        """Test getting primary model artifact ID."""
        trace = ExecutionTrace(model_step_index=4)

        step = ExecutionStep(step_index=4, operator_type="model")
        step.artifacts.add_artifact("0001:4:all", is_primary=True)
        trace.add_step(step)

        model_id = trace.get_model_artifact_id()
        assert model_id == "0001:4:all"

    def test_get_fold_artifact_ids(self):
        """Test getting per-fold model artifact IDs."""
        trace = ExecutionTrace(model_step_index=4)

        step = ExecutionStep(step_index=4, operator_type="model")
        step.artifacts.add_fold_artifact(0, "0001:4:0")
        step.artifacts.add_fold_artifact(1, "0001:4:1")
        trace.add_step(step)

        fold_ids = trace.get_fold_artifact_ids()
        assert fold_ids == {0: "0001:4:0", 1: "0001:4:1"}

    def test_finalize(self):
        """Test finalizing trace with summary info."""
        trace = ExecutionTrace()

        trace.finalize(
            preprocessing_chain="SNV>SG>MinMax",
            metadata={"dataset": "wheat", "samples": 100}
        )

        assert trace.preprocessing_chain == "SNV>SG>MinMax"
        assert trace.metadata["dataset"] == "wheat"
        assert trace.metadata["samples"] == 100

    def test_to_dict(self):
        """Test serialization to dictionary."""
        trace = ExecutionTrace(
            trace_id="test123",
            pipeline_uid="0001_pls",
            model_step_index=4,
            fold_weights={0: 0.5, 1: 0.5},
            preprocessing_chain="SNV"
        )
        trace.add_step(ExecutionStep(step_index=1, operator_type="transform"))

        d = trace.to_dict()

        assert d["trace_id"] == "test123"
        assert d["pipeline_uid"] == "0001_pls"
        assert d["model_step_index"] == 4
        assert d["fold_weights"] == {0: 0.5, 1: 0.5}
        assert d["preprocessing_chain"] == "SNV"
        assert len(d["steps"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "trace_id": "abc123",
            "pipeline_uid": "test_pipeline",
            "model_step_index": 3,
            "fold_weights": {"0": 0.6, "1": 0.4},
            "steps": [
                {
                    "step_index": 1,
                    "operator_type": "transform",
                    "operator_class": "SNV",
                    "execution_mode": "train",
                    "artifacts": {"artifact_ids": [], "fold_artifact_ids": {}}
                }
            ],
            "preprocessing_chain": "SNV",
            "metadata": {}
        }

        trace = ExecutionTrace.from_dict(data)

        assert trace.trace_id == "abc123"
        assert trace.pipeline_uid == "test_pipeline"
        assert trace.model_step_index == 3
        assert trace.fold_weights == {0: 0.6, 1: 0.4}
        assert len(trace.steps) == 1
        assert trace.steps[0].operator_class == "SNV"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ExecutionTrace(
            trace_id="roundtrip_test",
            pipeline_uid="test_pipeline_uid",
            model_step_index=5,
            fold_weights={0: 0.33, 1: 0.33, 2: 0.34},
            preprocessing_chain="A>B>C",
            metadata={"key": "value"}
        )
        original.add_step(ExecutionStep(step_index=1, operator_type="a"))
        original.add_step(ExecutionStep(step_index=2, operator_type="b"))

        restored = ExecutionTrace.from_dict(original.to_dict())

        assert restored.trace_id == original.trace_id
        assert restored.pipeline_uid == original.pipeline_uid
        assert restored.model_step_index == original.model_step_index
        assert restored.fold_weights == original.fold_weights
        assert restored.preprocessing_chain == original.preprocessing_chain
        assert len(restored.steps) == len(original.steps)

    def test_repr(self):
        """Test string representation."""
        trace = ExecutionTrace(trace_id="test", model_step_index=4)
        trace.add_step(ExecutionStep(step_index=1))
        trace.steps[0].artifacts.add_artifact("art1")

        repr_str = repr(trace)

        assert "ExecutionTrace" in repr_str
        assert "test" in repr_str
        assert "steps=1" in repr_str


class TestStepExecutionMode:
    """Tests for StepExecutionMode enum."""

    def test_values(self):
        """Test enum values."""
        assert StepExecutionMode.TRAIN.value == "train"
        assert StepExecutionMode.PREDICT.value == "predict"
        assert StepExecutionMode.SKIP.value == "skip"

    def test_str(self):
        """Test string conversion."""
        assert str(StepExecutionMode.TRAIN) == "train"
        assert str(StepExecutionMode.PREDICT) == "predict"

    def test_from_string(self):
        """Test creating from string."""
        assert StepExecutionMode("train") == StepExecutionMode.TRAIN
        assert StepExecutionMode("predict") == StepExecutionMode.PREDICT
        assert StepExecutionMode("skip") == StepExecutionMode.SKIP
