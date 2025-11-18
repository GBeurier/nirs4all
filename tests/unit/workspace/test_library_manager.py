"""
Test Phase 3: LibraryManager

Tests library management for templates and trained pipelines.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from nirs4all.workspace import LibraryManager


class TestPhase3LibraryManager:
    """Test Phase 3 LibraryManager implementation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_pipeline_dir(self, temp_workspace):
        """Create sample pipeline directory with files."""
        pipeline_dir = temp_workspace / "sample_pipeline"
        pipeline_dir.mkdir(parents=True)

        # Create pipeline.json
        pipeline_config = {
            "preprocessing": [{"name": "StandardScaler"}],
            "model": {"name": "PLSRegression", "n_components": 5},
            "artifacts": [
                {"type": "scaler", "path": "../_binaries/scaler_abc123.pkl"}
            ]
        }
        with open(pipeline_dir / "pipeline.json", 'w') as f:
            json.dump(pipeline_config, f)

        # Create metrics.json
        metrics = {
            "test_score": 0.45,
            "train_score": 0.32,
            "val_score": 0.41
        }
        with open(pipeline_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Create predictions.csv
        (pipeline_dir / "predictions.csv").write_text('y_true,y_pred\n12.5,12.3')

        return pipeline_dir

    @pytest.fixture
    def sample_run_dir(self, temp_workspace, sample_pipeline_dir):
        """Create sample run directory with binaries."""
        run_dir = temp_workspace / "run_dir"

        # Create _binaries folder
        binaries_dir = run_dir / "_binaries"
        binaries_dir.mkdir(parents=True)

        # Create sample binary file
        (binaries_dir / "scaler_abc123.pkl").write_text('binary data')

        # Copy pipeline into run
        pipeline_dest = run_dir / "0001_test"
        shutil.copytree(sample_pipeline_dir, pipeline_dest)

        return run_dir

    def test_library_initialization(self, temp_workspace):
        """Test library manager initializes directory structure."""
        library_dir = temp_workspace / "library"
        library = LibraryManager(library_dir)

        # Verify directories created
        assert library.templates_dir.exists()
        assert library.filtered_dir.exists()
        assert library.pipeline_dir.exists()
        assert library.fullrun_dir.exists()
        assert library.trained_dir.exists()

    def test_save_template(self, temp_workspace):
        """Test saving pipeline template (config only)."""
        library = LibraryManager(temp_workspace / "library")

        pipeline_config = {
            "preprocessing": [{"name": "StandardScaler"}],
            "model": {"name": "PLSRegression", "n_components": 5}
        }

        template_path = library.save_template(
            pipeline_config,
            "baseline_pls",
            "Baseline PLS configuration"
        )

        # Verify template saved
        assert template_path.exists()
        assert template_path.name == "baseline_pls.json"

        # Verify content
        with open(template_path) as f:
            template = json.load(f)
        assert template["name"] == "baseline_pls"
        assert template["type"] == "template"
        assert template["config"] == pipeline_config
        assert "created_at" in template

    def test_save_filtered(self, temp_workspace, sample_pipeline_dir):
        """Test saving filtered pipeline (config + metrics only)."""
        library = LibraryManager(temp_workspace / "library")

        filtered_path = library.save_filtered(
            sample_pipeline_dir,
            "experiment_001",
            "First experiment"
        )

        # Verify files copied
        assert filtered_path.exists()
        assert (filtered_path / "pipeline.json").exists()
        assert (filtered_path / "metrics.json").exists()
        assert not (filtered_path / "predictions.csv").exists()  # Not copied

        # Verify metadata
        assert (filtered_path / "library_metadata.json").exists()
        with open(filtered_path / "library_metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "experiment_001"
        assert metadata["type"] == "filtered"

    def test_save_pipeline_full(self, temp_workspace, sample_run_dir):
        """Test saving full pipeline with binaries."""
        library = LibraryManager(temp_workspace / "library")

        pipeline_dir = sample_run_dir / "0001_test"

        pipeline_path = library.save_pipeline_full(
            sample_run_dir,
            pipeline_dir,
            "production_model",
            "Production ready model"
        )

        # Verify all files copied
        assert pipeline_path.exists()
        assert (pipeline_path / "pipeline.json").exists()
        assert (pipeline_path / "metrics.json").exists()
        assert (pipeline_path / "predictions.csv").exists()

        # Verify binaries copied
        assert (pipeline_path / "_binaries").exists()
        assert (pipeline_path / "_binaries" / "scaler_abc123.pkl").exists()

        # Verify metadata
        with open(pipeline_path / "library_metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "production_model"
        assert metadata["type"] == "pipeline"

    def test_save_fullrun(self, temp_workspace, sample_run_dir):
        """Test saving entire run directory."""
        library = LibraryManager(temp_workspace / "library")

        fullrun_path = library.save_fullrun(
            sample_run_dir,
            "wheat_baseline_run",
            "Complete baseline experiment"
        )

        # Verify entire structure copied
        assert fullrun_path.exists()
        assert (fullrun_path / "_binaries").exists()
        assert (fullrun_path / "0001_test").exists()
        assert (fullrun_path / "0001_test" / "pipeline.json").exists()

        # Verify metadata
        with open(fullrun_path / "library_metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "wheat_baseline_run"
        assert metadata["type"] == "fullrun"

    def test_list_templates(self, temp_workspace):
        """Test listing saved templates."""
        library = LibraryManager(temp_workspace / "library")

        # Save multiple templates
        config1 = {"model": "PLS"}
        config2 = {"model": "RandomForest"}

        library.save_template(config1, "template1")
        library.save_template(config2, "template2")

        # List templates
        templates = library.list_templates()

        assert len(templates) >= 2
        names = [t["name"] for t in templates]
        assert "template1" in names
        assert "template2" in names

    def test_load_template(self, temp_workspace):
        """Test loading template by name."""
        library = LibraryManager(temp_workspace / "library")

        config = {"model": "PLS", "n_components": 5}
        library.save_template(config, "test_template", "Test description")

        # Load template
        loaded = library.load_template("test_template")

        assert loaded["name"] == "test_template"
        assert loaded["config"] == config
        assert loaded["description"] == "Test description"

    def test_list_filtered(self, temp_workspace, sample_pipeline_dir):
        """Test listing filtered pipelines."""
        library = LibraryManager(temp_workspace / "library")

        library.save_filtered(sample_pipeline_dir, "exp1")
        library.save_filtered(sample_pipeline_dir, "exp2")

        filtered = library.list_filtered()

        assert len(filtered) >= 2
        names = [f["name"] for f in filtered]
        assert "exp1" in names
        assert "exp2" in names

    def test_list_pipelines(self, temp_workspace, sample_run_dir):
        """Test listing full pipelines."""
        library = LibraryManager(temp_workspace / "library")

        pipeline_dir = sample_run_dir / "0001_test"
        library.save_pipeline_full(sample_run_dir, pipeline_dir, "model1")
        library.save_pipeline_full(sample_run_dir, pipeline_dir, "model2")

        pipelines = library.list_pipelines()

        assert len(pipelines) >= 2
        names = [p["name"] for p in pipelines]
        assert "model1" in names
        assert "model2" in names

    def test_list_fullruns(self, temp_workspace, sample_run_dir):
        """Test listing full runs."""
        library = LibraryManager(temp_workspace / "library")

        library.save_fullrun(sample_run_dir, "run1")
        library.save_fullrun(sample_run_dir, "run2")

        fullruns = library.list_fullruns()

        assert len(fullruns) >= 2
        names = [r["name"] for r in fullruns]
        assert "run1" in names
        assert "run2" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
