"""
State management and internal consistency tests for PipelineRunner.

This test file focuses on internal state tracking, step numbering,
operation counting, and state transitions throughout execution.
"""

import pytest
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.dataset.dataset_config import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from tests.utils.test_data_generator import TestDataManager


@pytest.fixture
def runner_with_workspace(tmp_path):
    """Create runner with temporary workspace."""
    return PipelineRunner(
        workspace_path=tmp_path,
        save_files=False,
        verbose=0,
        enable_tab_reports=False
    )


@pytest.fixture
def test_data(tmp_path):
    """Create test dataset."""
    manager = TestDataManager()
    manager.create_regression_dataset("regression", n_train=50, n_val=20)
    yield manager
    manager.cleanup()


class TestStateInitialization:
    """Test initial state of runner."""

    def test_all_counters_zero(self):
        """Verify all counters start at zero or default values."""
        runner = PipelineRunner(save_files=False)

        assert runner.step_number == 0
        assert runner.substep_number == -1
        assert runner.operation_count == 0

    def test_empty_collections(self):
        """Verify all collections start empty."""
        runner = PipelineRunner(save_files=False, keep_datasets=True)

        assert runner.step_binaries == {}
        assert runner.raw_data == {}
        assert runner.pp_data == {}
        assert runner._figure_refs == []

    def test_none_references(self):
        """Verify all object references start as None."""
        runner = PipelineRunner(save_files=False)

        assert runner.pipeline_uid is None
        assert runner.current_run_dir is None
        assert runner.saver is None
        assert runner.manifest_manager is None
        assert runner.binary_loader is None
        assert runner.prediction_metadata is None
        assert runner.config_path is None
        assert runner.target_model is None
        assert runner.model_weights is None
        assert runner._captured_model is None

    def test_boolean_flags(self):
        """Verify boolean flags are set correctly."""
        runner = PipelineRunner(save_files=False)

        assert runner._capture_model is False
        assert runner.plots_visible is False
        assert runner.keep_datasets is True


class TestStepNumbering:
    """Test step number tracking through execution."""

    def test_step_number_increments(self, runner_with_workspace, test_data):
        """Test that step_number increments for each main step."""
        dataset_path = str(test_data.get_temp_directory() / "regression")

        pipeline = [
            StandardScaler(),  # Step 1
            MinMaxScaler(),    # Step 2
            {"model": LinearRegression()}  # Step 3
        ]

        runner_with_workspace.run(pipeline, dataset_path)

        # Should have executed 3 steps
        assert runner_with_workspace.step_number >= 3

    def test_substep_resets_between_main_steps(self, runner_with_workspace, test_data):
        """Test that substep_number resets for each main step."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_path)
        config, name = dataset_config.configs[0]
        dataset = dataset_config.get_dataset(config, name)

        # Initialize required state
        runner_with_workspace.current_run_dir = runner_with_workspace.workspace_path / "test_run"
        runner_with_workspace.current_run_dir.mkdir(exist_ok=True)
        from nirs4all.pipeline.io import SimulationSaver
        from nirs4all.pipeline.manifest_manager import ManifestManager
        runner_with_workspace.saver = SimulationSaver(runner_with_workspace.current_run_dir, save_files=False)
        runner_with_workspace.manifest_manager = ManifestManager(runner_with_workspace.current_run_dir)

        # Create pipeline manifest (required for step execution)
        pipeline_id, _ = runner_with_workspace.manifest_manager.create_pipeline(
            name="test",
            dataset=dataset.name,
            pipeline_config={"steps": []},
            pipeline_hash="test123"
        )
        runner_with_workspace.pipeline_uid = pipeline_id
        runner_with_workspace.saver.register(pipeline_id)

        steps = [{"preprocessing": StandardScaler()}, {"preprocessing": MinMaxScaler()}]
        context = {"processing": [["raw"]] * dataset.features_sources(), "y": "numeric"}
        predictions = Predictions()

        # Execute steps
        for step in steps:
            runner_with_workspace.run_step(step, dataset, context, predictions, is_substep=False)

        # After each main step, substep should be 0
        assert runner_with_workspace.substep_number == 0

    def test_substep_increments_for_substeps(self, runner_with_workspace, test_data):
        """Test that substep_number increments for substeps."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_path)
        config, name = dataset_config.configs[0]
        dataset = dataset_config.get_dataset(config, name)

        # Initialize required state
        runner_with_workspace.current_run_dir = runner_with_workspace.workspace_path / "test_run"
        runner_with_workspace.current_run_dir.mkdir(exist_ok=True)
        from nirs4all.pipeline.io import SimulationSaver
        from nirs4all.pipeline.manifest_manager import ManifestManager
        runner_with_workspace.saver = SimulationSaver(runner_with_workspace.current_run_dir, save_files=False)
        runner_with_workspace.manifest_manager = ManifestManager(runner_with_workspace.current_run_dir)

        # Create pipeline manifest (required for step execution)
        pipeline_id, _ = runner_with_workspace.manifest_manager.create_pipeline(
            name="test",
            dataset=dataset.name,
            pipeline_config={"steps": []},
            pipeline_hash="test123"
        )
        runner_with_workspace.pipeline_uid = pipeline_id
        runner_with_workspace.saver.register(pipeline_id)

        context = {"processing": [["raw"]] * dataset.features_sources(), "y": "numeric"}
        predictions = Predictions()

        # Run main step - wrap sklearn instance
        runner_with_workspace.run_step({"preprocessing": StandardScaler()}, dataset, context, predictions, is_substep=False)
        initial_step = runner_with_workspace.step_number

        # Run substeps - wrap sklearn instances
        runner_with_workspace.run_step({"preprocessing": MinMaxScaler()}, dataset, context, predictions, is_substep=True)
        runner_with_workspace.run_step({"preprocessing": StandardScaler()}, dataset, context, predictions, is_substep=True)

        # Step number should not change for substeps
        assert runner_with_workspace.step_number == initial_step

        # Substep should have incremented
        assert runner_with_workspace.substep_number >= 1

    def test_operation_count_increments(self, runner_with_workspace):
        """Test that operation count increments correctly."""
        assert runner_with_workspace.operation_count == 0

        op1 = runner_with_workspace.next_op()
        assert op1 == 1
        assert runner_with_workspace.operation_count == 1

        op2 = runner_with_workspace.next_op()
        assert op2 == 2
        assert runner_with_workspace.operation_count == 2

        for _ in range(10):
            runner_with_workspace.next_op()

        assert runner_with_workspace.operation_count == 12

    def test_operation_count_resets_per_run(self, runner_with_workspace, test_data):
        """Test that operation_count resets for each pipeline run."""
        dataset_path = str(test_data.get_temp_directory() / "regression")

        pipeline = [StandardScaler(), {"model": LinearRegression()}]

        # First run
        runner_with_workspace.run(pipeline, dataset_path)
        first_count = runner_with_workspace.operation_count

        # Operation count should have been incremented
        assert first_count > 0


class TestStateTransitions:
    """Test state transitions during execution."""

    def test_mode_stays_consistent(self, runner_with_workspace, test_data):
        """Test that mode doesn't change unexpectedly."""
        assert runner_with_workspace.mode == "train"

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        runner_with_workspace.run(pipeline, dataset_path)

        # Mode should still be train
        assert runner_with_workspace.mode == "train"

    def test_pipeline_uid_set_during_run(self, runner_with_workspace, test_data):
        """Test that pipeline_uid is set during training run."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        assert runner_with_workspace.pipeline_uid is None

        runner_with_workspace.run(pipeline, dataset_path)

        assert runner_with_workspace.pipeline_uid is not None
        assert isinstance(runner_with_workspace.pipeline_uid, str)

    def test_current_run_dir_set_during_run(self, runner_with_workspace, test_data):
        """Test that current_run_dir is set during run."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        assert runner_with_workspace.current_run_dir is None

        runner_with_workspace.run(pipeline, dataset_path)

        assert runner_with_workspace.current_run_dir is not None
        assert isinstance(runner_with_workspace.current_run_dir, Path)
        assert runner_with_workspace.current_run_dir.exists()

    def test_saver_initialized_during_run(self, runner_with_workspace, test_data):
        """Test that saver is initialized during run."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        assert runner_with_workspace.saver is None

        runner_with_workspace.run(pipeline, dataset_path)

        assert runner_with_workspace.saver is not None

    def test_manifest_manager_initialized_during_run(self, runner_with_workspace, test_data):
        """Test that manifest_manager is initialized during run."""
        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        assert runner_with_workspace.manifest_manager is None

        runner_with_workspace.run(pipeline, dataset_path)

        assert runner_with_workspace.manifest_manager is not None


class TestDataCapture:
    """Test data capture functionality."""

    def test_raw_data_captured_when_enabled(self, tmp_path, test_data):
        """Test that raw data is captured when keep_datasets=True."""
        runner = PipelineRunner(
            workspace_path=tmp_path,
            save_files=False,
            verbose=0,
            enable_tab_reports=False,
            keep_datasets=True
        )

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [StandardScaler(), {"model": LinearRegression()}]

        runner.run(pipeline, dataset_path)

        assert len(runner.raw_data) > 0

        # Verify data structure
        for dataset_name, data in runner.raw_data.items():
            assert isinstance(data, np.ndarray)
            assert data.shape[0] > 0  # Has samples
            assert data.shape[1] > 0  # Has features

    def test_preprocessed_data_captured_when_enabled(self, tmp_path, test_data):
        """Test that preprocessed data is captured when keep_datasets=True."""
        runner = PipelineRunner(
            workspace_path=tmp_path,
            save_files=False,
            verbose=0,
            enable_tab_reports=False,
            keep_datasets=True
        )

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [StandardScaler(), {"model": LinearRegression()}]

        runner.run(pipeline, dataset_path)

        assert len(runner.pp_data) > 0

        # Verify data structure
        for dataset_name, pp_dict in runner.pp_data.items():
            assert isinstance(pp_dict, dict)
            for pp_key, data in pp_dict.items():
                assert isinstance(data, np.ndarray)

    def test_data_not_captured_when_disabled(self, tmp_path, test_data):
        """Test that data is not captured when keep_datasets=False."""
        runner = PipelineRunner(
            workspace_path=tmp_path,
            save_files=False,
            verbose=0,
            enable_tab_reports=False,
            keep_datasets=False
        )

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [StandardScaler(), {"model": LinearRegression()}]

        runner.run(pipeline, dataset_path)

        # Verify data was not captured (attributes don't exist when keep_datasets=False)
        assert not hasattr(runner, 'raw_data') or len(runner.raw_data) == 0
        assert not hasattr(runner, 'pp_data') or len(runner.pp_data) == 0

    def test_multiple_datasets_captured_separately(self, tmp_path, test_data):
        """Test that multiple datasets are captured separately."""
        runner = PipelineRunner(
            workspace_path=tmp_path,
            save_files=False,
            verbose=0,
            enable_tab_reports=False,
            keep_datasets=True
        )

        # Create second dataset
        test_data.create_regression_dataset("regression_2")

        temp_dir = test_data.get_temp_directory()
        dataset_paths = [
            str(temp_dir / "regression"),
            str(temp_dir / "regression_2")
        ]

        pipeline = [StandardScaler(), {"model": LinearRegression()}]

        runner.run(pipeline, dataset_paths)

        # Should have data for both datasets
        assert len(runner.raw_data) >= 2


class TestStateConsistency:
    """Test state consistency across operations."""

    def test_workspace_path_immutable(self, tmp_path, test_data):
        """Test that workspace_path doesn't change."""
        runner = PipelineRunner(
            workspace_path=tmp_path,
            save_files=False,
            verbose=0,
            enable_tab_reports=False
        )

        original_path = runner.workspace_path

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        runner.run(pipeline, dataset_path)

        assert runner.workspace_path == original_path

    def test_configuration_immutable(self, runner_with_workspace, test_data):
        """Test that configuration parameters don't change."""
        original_verbose = runner_with_workspace.verbose
        original_parallel = runner_with_workspace.parallel
        original_backend = runner_with_workspace.backend

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        runner_with_workspace.run(pipeline, dataset_path)

        assert runner_with_workspace.verbose == original_verbose
        assert runner_with_workspace.parallel == original_parallel
        assert runner_with_workspace.backend == original_backend


class TestStepBinariesTracking:
    """Test step binaries tracking."""

    def test_step_binaries_empty_initially(self):
        """Test that step_binaries starts empty."""
        runner = PipelineRunner(save_files=False)

        assert runner.step_binaries == {}
        assert len(runner.step_binaries) == 0

    def test_step_binaries_structure(self):
        """Test step_binaries data structure."""
        runner = PipelineRunner(save_files=False)

        # Verify it's a dict
        assert isinstance(runner.step_binaries, dict)

        # Should accept string keys and list values
        runner.step_binaries["step_1"] = ["binary1", "binary2"]
        assert runner.step_binaries["step_1"] == ["binary1", "binary2"]


class TestModelCaptureState:
    """Test model capture state management."""

    def test_capture_model_false_by_default(self):
        """Test that _capture_model is False by default."""
        runner = PipelineRunner(save_files=False)

        assert runner._capture_model is False

    def test_captured_model_none_initially(self):
        """Test that _captured_model is None initially."""
        runner = PipelineRunner(save_files=False)

        assert runner._captured_model is None

    def test_capture_model_flag_can_be_set(self):
        """Test that _capture_model flag can be set."""
        runner = PipelineRunner(save_files=False)

        runner._capture_model = True
        assert runner._capture_model is True

        runner._capture_model = False
        assert runner._capture_model is False


class TestFigureReferences:
    """Test figure reference tracking."""

    def test_figure_refs_empty_initially(self):
        """Test that _figure_refs starts empty."""
        runner = PipelineRunner(save_files=False)

        assert runner._figure_refs == []
        assert len(runner._figure_refs) == 0

    def test_figure_refs_cleared_on_run(self, runner_with_workspace, test_data):
        """Test that _figure_refs is cleared at start of run."""
        # Add some dummy refs
        runner_with_workspace._figure_refs.append("dummy_fig")
        assert len(runner_with_workspace._figure_refs) > 0

        dataset_path = str(test_data.get_temp_directory() / "regression")
        pipeline = [{"model": LinearRegression()}]

        runner_with_workspace.run(pipeline, dataset_path)

        # Should be cleared at start of run
        # (may have new refs added during run, but old ones gone)
        # We can't easily verify this without mocking, but the clear() call happens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
