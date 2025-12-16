"""
Tests for the Retrainer module (Phase 7).

Tests the Retrainer class and related utilities for retraining,
transfer learning, and fine-tuning pipelines.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from nirs4all.pipeline.retrainer import (
    Retrainer,
    RetrainMode,
    RetrainConfig,
    StepMode,
    RetrainArtifactProvider,
    ExtractedPipeline,
)
from nirs4all.pipeline.trace import ExecutionTrace, ExecutionStep


class TestRetrainMode:
    """Tests for RetrainMode enum."""

    def test_values(self):
        """Test enum values."""
        assert RetrainMode.FULL.value == "full"
        assert RetrainMode.TRANSFER.value == "transfer"
        assert RetrainMode.FINETUNE.value == "finetune"

    def test_str(self):
        """Test string conversion."""
        assert str(RetrainMode.FULL) == "full"
        assert str(RetrainMode.TRANSFER) == "transfer"

    def test_from_string(self):
        """Test creating from string."""
        assert RetrainMode("full") == RetrainMode.FULL
        assert RetrainMode("transfer") == RetrainMode.TRANSFER
        assert RetrainMode("finetune") == RetrainMode.FINETUNE


class TestStepMode:
    """Tests for StepMode dataclass."""

    def test_create_basic(self):
        """Test creating basic step mode."""
        sm = StepMode(step_index=1, mode='train')

        assert sm.step_index == 1
        assert sm.mode == 'train'
        assert sm.artifact_id is None
        assert sm.kwargs == {}

    def test_create_with_artifact(self):
        """Test creating step mode with artifact."""
        sm = StepMode(
            step_index=2,
            mode='predict',
            artifact_id='0001:2:all'
        )

        assert sm.step_index == 2
        assert sm.mode == 'predict'
        assert sm.artifact_id == '0001:2:all'

    def test_is_train(self):
        """Test is_train method."""
        assert StepMode(step_index=1, mode='train').is_train() is True
        assert StepMode(step_index=1, mode='predict').is_train() is False
        assert StepMode(step_index=1, mode='skip').is_train() is False

    def test_is_predict(self):
        """Test is_predict method."""
        assert StepMode(step_index=1, mode='predict').is_predict() is True
        assert StepMode(step_index=1, mode='train').is_predict() is False


class TestRetrainConfig:
    """Tests for RetrainConfig dataclass."""

    def test_create_default(self):
        """Test creating default config."""
        config = RetrainConfig()

        assert config.mode == RetrainMode.FULL
        assert config.step_modes == []
        assert config.new_model is None
        assert config.epochs is None

    def test_create_with_params(self):
        """Test creating config with parameters."""
        config = RetrainConfig(
            mode=RetrainMode.TRANSFER,
            epochs=10,
            learning_rate=0.001
        )

        assert config.mode == RetrainMode.TRANSFER
        assert config.epochs == 10
        assert config.learning_rate == 0.001

    def test_get_step_mode(self):
        """Test getting step mode override."""
        step_modes = [
            StepMode(step_index=1, mode='predict'),
            StepMode(step_index=3, mode='train'),
        ]
        config = RetrainConfig(step_modes=step_modes)

        assert config.get_step_mode(1).mode == 'predict'
        assert config.get_step_mode(3).mode == 'train'
        assert config.get_step_mode(2) is None

    def test_should_train_step_full(self):
        """Test should_train_step for full mode."""
        config = RetrainConfig(mode=RetrainMode.FULL)

        # In full mode, all steps should train
        assert config.should_train_step(1) is True
        assert config.should_train_step(2) is True
        assert config.should_train_step(3, is_model=True) is True

    def test_should_train_step_transfer(self):
        """Test should_train_step for transfer mode."""
        config = RetrainConfig(mode=RetrainMode.TRANSFER)

        # In transfer mode, only model steps train
        assert config.should_train_step(1) is False
        assert config.should_train_step(2) is False
        assert config.should_train_step(3, is_model=True) is True

    def test_should_train_step_finetune(self):
        """Test should_train_step for finetune mode."""
        config = RetrainConfig(mode=RetrainMode.FINETUNE)

        # In finetune mode, only model steps train
        assert config.should_train_step(1) is False
        assert config.should_train_step(3, is_model=True) is True

    def test_should_train_step_with_override(self):
        """Test should_train_step respects overrides."""
        step_modes = [
            StepMode(step_index=2, mode='train'),  # Force train
        ]
        config = RetrainConfig(mode=RetrainMode.TRANSFER, step_modes=step_modes)

        # Override should take precedence
        assert config.should_train_step(1) is False
        assert config.should_train_step(2) is True  # Override!


class TestRetrainArtifactProvider:
    """Tests for RetrainArtifactProvider."""

    def test_full_mode_no_artifacts(self):
        """Test that full mode doesn't provide artifacts."""
        mock_base = MagicMock()
        mock_base.get_artifact.return_value = "artifact_object"
        mock_base.has_artifacts_for_step.return_value = True

        config = RetrainConfig(mode=RetrainMode.FULL)
        provider = RetrainArtifactProvider(mock_base, config)

        # Full mode should not provide artifacts (train from scratch)
        assert provider.get_artifact(1) is None
        assert provider.has_artifacts_for_step(1) is False

    def test_transfer_mode_provides_preprocessing(self):
        """Test that transfer mode provides preprocessing artifacts."""
        mock_base = MagicMock()
        mock_base.get_artifact.return_value = "artifact_object"
        mock_base.has_artifacts_for_step.return_value = True

        # Create trace to identify step types
        trace = ExecutionTrace()
        trace.add_step(ExecutionStep(step_index=1, operator_type="transform"))
        trace.add_step(ExecutionStep(step_index=4, operator_type="model"))

        config = RetrainConfig(mode=RetrainMode.TRANSFER)
        provider = RetrainArtifactProvider(mock_base, config, trace)

        # Should provide preprocessing but not model
        assert provider.get_artifact(1) == "artifact_object"  # preprocessing
        assert provider.get_artifact(4) is None  # model (train new)

    def test_finetune_mode_provides_model(self):
        """Test that finetune mode provides model artifacts."""
        mock_base = MagicMock()
        mock_base.get_artifact.return_value = "model_object"
        mock_base.has_artifacts_for_step.return_value = True

        trace = ExecutionTrace()
        trace.add_step(ExecutionStep(step_index=1, operator_type="transform"))
        trace.add_step(ExecutionStep(step_index=4, operator_type="model"))

        config = RetrainConfig(mode=RetrainMode.FINETUNE)
        provider = RetrainArtifactProvider(mock_base, config, trace)

        # Should provide model for continuation
        assert provider.get_artifact(1) is None  # preprocessing (don't provide)
        assert provider.get_artifact(4) == "model_object"  # model (for finetuning)

    def test_step_mode_override(self):
        """Test step mode overrides."""
        mock_base = MagicMock()
        mock_base.get_artifact.return_value = "artifact"
        mock_base.has_artifacts_for_step.return_value = True

        step_modes = [
            StepMode(step_index=2, mode='predict'),  # Force use artifact
        ]
        config = RetrainConfig(mode=RetrainMode.FULL, step_modes=step_modes)
        provider = RetrainArtifactProvider(mock_base, config)

        # Override should provide artifact
        assert provider.get_artifact(2) == "artifact"


class TestExtractedPipeline:
    """Tests for ExtractedPipeline dataclass."""

    def test_create_basic(self):
        """Test creating basic extracted pipeline."""
        steps = [{"transform": "A"}, {"model": "B"}]
        extracted = ExtractedPipeline(
            steps=steps,
            model_step_index=2,
            preprocessing_chain="A"
        )

        assert len(extracted) == 2
        assert extracted.model_step_index == 2
        assert extracted.preprocessing_chain == "A"

    def test_get_step(self):
        """Test getting step by index."""
        steps = [{"a": 1}, {"b": 2}, {"c": 3}]
        extracted = ExtractedPipeline(steps=steps)

        assert extracted.get_step(0) == {"a": 1}
        assert extracted.get_step(2) == {"c": 3}

    def test_set_step(self):
        """Test setting step by index."""
        steps = [{"a": 1}, {"b": 2}]
        extracted = ExtractedPipeline(steps=steps)

        extracted.set_step(1, {"new": 99})
        assert extracted.get_step(1) == {"new": 99}

    def test_get_model_step(self):
        """Test getting model step."""
        steps = [{"t": 1}, {"t": 2}, {"model": "PLS"}]
        extracted = ExtractedPipeline(steps=steps, model_step_index=3)

        assert extracted.get_model_step() == {"model": "PLS"}

    def test_get_model_step_none(self):
        """Test getting model step when not set."""
        extracted = ExtractedPipeline(steps=[{"a": 1}])
        assert extracted.get_model_step() is None

    def test_set_model(self):
        """Test replacing model in pipeline."""
        steps = [{"t": 1}, {"model": "OldModel"}]
        extracted = ExtractedPipeline(steps=steps, model_step_index=2)

        extracted.set_model("NewModel")

        assert extracted.steps[1]["model"] == "NewModel"

    def test_set_model_no_index(self):
        """Test set_model raises when no model step identified."""
        extracted = ExtractedPipeline(steps=[{"a": 1}])

        with pytest.raises(ValueError, match="No model step"):
            extracted.set_model("NewModel")

    def test_repr(self):
        """Test string representation."""
        extracted = ExtractedPipeline(
            steps=[1, 2, 3],
            model_step_index=3,
            preprocessing_chain="A>B"
        )

        repr_str = repr(extracted)

        assert "ExtractedPipeline" in repr_str
        assert "steps=3" in repr_str
        assert "model_step=3" in repr_str
        assert "A>B" in repr_str


class TestRetrainerUnit:
    """Unit tests for Retrainer class."""

    def test_create_retrainer(self):
        """Test creating retrainer instance."""
        mock_runner = MagicMock()
        mock_runner.workspace_path = Path("/tmp/workspace")
        mock_runner.runs_dir = Path("/tmp/workspace/runs")

        retrainer = Retrainer(mock_runner)

        assert retrainer.runner == mock_runner
        assert retrainer.resolver is not None
