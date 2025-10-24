"""
Test Phase 1: Foundation Layer

Tests workspace structure creation, sequential numbering, and basic operations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from nirs4all.workspace import WorkspaceManager
from nirs4all.workspace.run_manager import RunManager
from nirs4all.pipeline.manifest_manager import ManifestManager
from nirs4all.pipeline.io import SimulationSaver


class TestPhase1Foundation:
    """Test Phase 1 foundation layer implementation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_workspace_initialization(self, temp_workspace):
        """Test workspace directory structure creation."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        # Check main directories
        assert (temp_workspace / "runs").exists()
        assert (temp_workspace / "exports").exists()
        assert (temp_workspace / "library").exists()
        assert (temp_workspace / "catalog").exists()

        # Check library subdirectories
        assert (temp_workspace / "library" / "templates").exists()
        assert (temp_workspace / "library" / "trained" / "filtered").exists()
        assert (temp_workspace / "library" / "trained" / "pipeline").exists()
        assert (temp_workspace / "library" / "trained" / "fullrun").exists()

        # Check catalog subdirectories
        assert (temp_workspace / "catalog" / "reports").exists()
        assert (temp_workspace / "catalog" / "archives" / "filtered").exists()
        assert (temp_workspace / "catalog" / "archives" / "pipeline").exists()
        assert (temp_workspace / "catalog" / "archives" / "best_predictions").exists()

        # Check exports subdirectories
        assert (temp_workspace / "exports" / "best_predictions").exists()
        assert (temp_workspace / "exports" / "session_reports").exists()

    def test_run_creation_no_custom_name(self, temp_workspace):
        """Test run creation without custom name."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        run_mgr = ws.create_run("wheat_sample1")

        # Check run directory name format: YYYY-MM-DD_dataset
        assert run_mgr.run_dir.name.endswith("_wheat_sample1")
        assert "-" in run_mgr.run_dir.name[:10]  # Date part with dashes

    def test_run_creation_with_custom_name(self, temp_workspace):
        """Test run creation with custom name."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        run_mgr = ws.create_run("wheat_sample1", run_name="baseline_test")

        # Check run directory name format: YYYY-MM-DD_dataset_customname
        assert "wheat_sample1_baseline_test" in run_mgr.run_dir.name

    def test_sequential_numbering(self, temp_workspace):
        """Test sequential pipeline numbering."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        run_mgr = ws.create_run("wheat_sample1")
        run_mgr.initialize({"description": "Test run"})

        # Create multiple pipelines
        pipeline1 = run_mgr.create_pipeline_dir("a1b2c3")
        pipeline2 = run_mgr.create_pipeline_dir("d4e5f6")
        pipeline3 = run_mgr.create_pipeline_dir("g7h8i9", pipeline_name="custom")

        # Check sequential numbering
        assert "0001_a1b2c3" in str(pipeline1)
        assert "0002_d4e5f6" in str(pipeline2)
        assert "0003_custom_g7h8i9" in str(pipeline3)

        # Verify directories exist
        assert pipeline1.exists()
        assert pipeline2.exists()
        assert pipeline3.exists()

    def test_binaries_folder_creation(self, temp_workspace):
        """Test _binaries folder creation with underscore prefix."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        run_mgr = ws.create_run("wheat_sample1")
        run_mgr.initialize({"description": "Test run"})

        # Check _binaries folder exists
        assert run_mgr.binaries_dir.exists()
        assert run_mgr.binaries_dir.name == "_binaries"

    def test_manifest_manager_sequential_numbering(self, temp_workspace):
        """Test ManifestManager get_next_pipeline_number method."""
        run_dir = temp_workspace / "test_run"
        run_dir.mkdir(parents=True)

        manifest = ManifestManager(str(temp_workspace))

        # First pipeline number
        num1 = manifest.get_next_pipeline_number(run_dir)
        assert num1 == 1

        # Create pipeline directory
        (run_dir / "0001_abc123").mkdir()

        # Next number
        num2 = manifest.get_next_pipeline_number(run_dir)
        assert num2 == 2

        # Create another
        (run_dir / "0002_def456").mkdir()

        num3 = manifest.get_next_pipeline_number(run_dir)
        assert num3 == 3

        # _binaries should not be counted
        (run_dir / "_binaries").mkdir()
        num4 = manifest.get_next_pipeline_number(run_dir)
        assert num4 == 3  # Still 3, not 4

    def test_simulation_saver_workspace_registration(self, temp_workspace):
        """Test SimulationSaver register_workspace method."""
        # Create fresh workspace for this test
        import tempfile
        test_ws = Path(tempfile.mkdtemp())

        try:
            saver = SimulationSaver()

            # Register without custom names
            pipeline_dir1 = saver.register_workspace(
                test_ws,
                "wheat_sample1",
                "abc123"
            )

            assert pipeline_dir1.exists()
            assert "0001_abc123" in str(pipeline_dir1)
            assert (pipeline_dir1.parent / "_binaries").exists()

            # Register with pipeline name
            saver2 = SimulationSaver()
            pipeline_dir2 = saver2.register_workspace(
                test_ws,
                "wheat_sample1",
                "def456",
                pipeline_name="baseline"
            )

            assert "0002_baseline_def456" in str(pipeline_dir2)

            # Register with run name
            saver3 = SimulationSaver()
            pipeline_dir3 = saver3.register_workspace(
                test_ws,
                "corn_samples",
                "ghi789",
                run_name="experiment1"
            )

            assert "corn_samples_experiment1" in str(pipeline_dir3.parent)
            assert "0001_ghi789" in str(pipeline_dir3)
        finally:
            shutil.rmtree(test_ws)

    def test_list_runs(self, temp_workspace):
        """Test listing runs."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        # Create multiple runs
        run1 = ws.create_run("wheat_sample1")
        run1.initialize({"description": "Test 1"})

        run2 = ws.create_run("wheat_sample2", run_name="custom")
        run2.initialize({"description": "Test 2"})

        # List runs
        runs = ws.list_runs()
        assert len(runs) >= 2

        # Check run data
        assert any("wheat_sample1" in r["name"] for r in runs)
        assert any("wheat_sample2_custom" in r["name"] for r in runs)

    def test_run_summary_update(self, temp_workspace):
        """Test run summary creation and update."""
        ws = WorkspaceManager(temp_workspace)
        ws.initialize_workspace()

        run_mgr = ws.create_run("wheat_sample1")
        run_mgr.initialize({"description": "Test run"})

        # Update summary
        run_mgr.update_summary({
            "status": "completed",
            "total_pipelines": 5,
            "successful_pipelines": 4,
            "failed_pipelines": 1
        })

        # Check summary file exists
        assert run_mgr.run_summary_file.exists()

        # Read and verify
        import json
        with open(run_mgr.run_summary_file) as f:
            summary = json.load(f)

        assert summary["dataset_name"] == "wheat_sample1"
        assert summary["status"] == "completed"
        assert summary["total_pipelines"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
