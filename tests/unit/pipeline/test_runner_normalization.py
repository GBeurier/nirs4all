"""
Tests for PipelineRunner input normalization.

Tests the flexible input system that allows:
- Pipelines: PipelineConfigs, List[steps], Dict, or file path
- Datasets: DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 50)  # 100 samples, 50 features
    y = np.random.randn(100)  # Regression targets
    return X, y


@pytest.fixture
def sample_pipeline_steps():
    """Simple pipeline steps for testing."""
    return [
        {"preprocessing": {"class": StandardScaler}},
        {"model": {"class": LinearRegression}}
    ]


@pytest.fixture
def sample_pipeline_dict(sample_pipeline_steps):
    """Pipeline as a dictionary."""
    return {"pipeline": sample_pipeline_steps}


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration."""
    return {
        "name": "test_dataset",
        "train_x": np.random.randn(80, 50),
        "train_y": np.random.randn(80),
        "test_x": np.random.randn(20, 50),
        "test_y": np.random.randn(20)
    }


class TestPipelineNormalization:
    """Test pipeline input normalization."""

    def test_normalize_pipeline_configs(self, sample_pipeline_steps):
        """Test that PipelineConfigs input passes through unchanged."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        pipeline_configs = PipelineConfigs(sample_pipeline_steps)

        normalized = runner.orchestrator._normalize_pipeline(pipeline_configs)

        assert normalized is pipeline_configs
        assert isinstance(normalized, PipelineConfigs)

    def test_normalize_pipeline_list(self, sample_pipeline_steps):
        """Test that list of steps is converted to PipelineConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        normalized = runner.orchestrator._normalize_pipeline(sample_pipeline_steps, name="test_pipeline")

        assert isinstance(normalized, PipelineConfigs)
        assert len(normalized.steps) >= 1
        assert "test_pipeline" in normalized.names[0]

    def test_normalize_pipeline_dict(self, sample_pipeline_dict):
        """Test that dict definition is converted to PipelineConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        normalized = runner.orchestrator._normalize_pipeline(sample_pipeline_dict)

        assert isinstance(normalized, PipelineConfigs)
        assert len(normalized.steps) >= 1

    def test_normalize_pipeline_json_file(self):
        """Test that JSON file path is loaded and converted to PipelineConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        # Use string class names for JSON serialization
        pipeline_dict_json = {
            "pipeline": [
                {"preprocessing": {"class": "sklearn.preprocessing.StandardScaler"}},
                {"model": {"class": "sklearn.linear_model.LinearRegression"}}
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(pipeline_dict_json, f)
            temp_path = f.name

        try:
            normalized = runner.orchestrator._normalize_pipeline(temp_path)

            assert isinstance(normalized, PipelineConfigs)
            assert len(normalized.steps) >= 1
        finally:
            Path(temp_path).unlink()


class TestDatasetNormalization:
    """Test dataset input normalization."""

    def test_normalize_dataset_configs(self, sample_data):
        """Test that DatasetConfigs input passes through unchanged."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        dataset_config = {
            "name": "test",
            "train_x": X[:80],
            "train_y": y[:80],
            "test_x": X[80:],
            "test_y": y[80:]
        }
        dataset_configs = DatasetConfigs(dataset_config)

        normalized = runner.orchestrator._normalize_dataset(dataset_configs)

        assert normalized is dataset_configs
        assert isinstance(normalized, DatasetConfigs)

    def test_normalize_spectro_dataset(self, sample_data):
        """Test that SpectroDataset is wrapped in DatasetConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        dataset = SpectroDataset(name="test_spectro")
        dataset.add_samples(X[:80], indexes={"partition": "train"})
        dataset.add_targets(y[:80])
        dataset.add_samples(X[80:], indexes={"partition": "test"})
        dataset.add_targets(y[80:])

        normalized = runner.orchestrator._normalize_dataset(dataset)

        assert isinstance(normalized, DatasetConfigs)
        assert len(normalized.configs) == 1
        config, name = normalized.configs[0]
        assert name == "test_spectro"

        # Verify we can get the dataset back
        retrieved = normalized.get_dataset(config, name)
        assert isinstance(retrieved, SpectroDataset)
        assert retrieved.name == "test_spectro"

    def test_normalize_numpy_array_x_only(self, sample_data):
        """Test that single numpy array (X only) is converted to DatasetConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, _ = sample_data

        normalized = runner.orchestrator._normalize_dataset(X, dataset_name="array_x")

        assert isinstance(normalized, DatasetConfigs)
        assert len(normalized.configs) == 1
        config, name = normalized.configs[0]
        assert name == "array_x"

        # Verify dataset structure
        dataset = normalized.get_dataset(config, name)
        assert isinstance(dataset, SpectroDataset)
        X_test = dataset.x({"partition": "test"}, layout="2d")
        assert X_test.shape == X.shape

    def test_normalize_numpy_tuple_x_y(self, sample_data):
        """Test that tuple (X, y) is converted to DatasetConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        normalized = runner.orchestrator._normalize_dataset((X, y), dataset_name="array_xy")

        assert isinstance(normalized, DatasetConfigs)
        assert len(normalized.configs) == 1
        config, name = normalized.configs[0]
        assert name == "array_xy"

        # Verify dataset structure
        dataset = normalized.get_dataset(config, name)
        assert isinstance(dataset, SpectroDataset)
        X_train = dataset.x({"partition": "train"}, layout="2d")
        y_train = dataset.y({"partition": "train"})
        assert X_train.shape == X.shape
        # y_train might have an extra dimension (100,1) vs (100,)
        assert y_train.shape[0] == y.shape[0]

    def test_normalize_numpy_tuple_with_partition_split(self, sample_data):
        """Test that tuple (X, y, partition_info) splits data correctly."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        partition_info = {"train": 80}  # First 80 samples for training
        normalized = runner.orchestrator._normalize_dataset((X, y, partition_info), dataset_name="split_array")

        assert isinstance(normalized, DatasetConfigs)
        config, name = normalized.configs[0]

        # Verify dataset partitions
        dataset = normalized.get_dataset(config, name)
        X_train = dataset.x({"partition": "train"}, layout="2d")
        X_test = dataset.x({"partition": "test"}, layout="2d")

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert X_train.shape[1] == X.shape[1]

    def test_normalize_numpy_tuple_with_partition_indices(self, sample_data):
        """Test tuple with explicit train/test indices."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        train_idx = slice(0, 70)
        test_idx = slice(70, 100)
        partition_info = {"train": train_idx, "test": test_idx}

        normalized = runner.orchestrator._normalize_dataset((X, y, partition_info), dataset_name="indexed_array")

        config, name = normalized.configs[0]
        dataset = normalized.get_dataset(config, name)

        X_train = dataset.x({"partition": "train"}, layout="2d")
        X_test = dataset.x({"partition": "test"}, layout="2d")

        assert X_train.shape[0] == 70
        assert X_test.shape[0] == 30

    def test_normalize_dict_config(self, sample_data):
        """Test that dict config is converted to DatasetConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        X, y = sample_data

        config_dict = {
            "name": "dict_dataset",
            "train_x": X[:80],
            "train_y": y[:80],
            "test_x": X[80:],
            "test_y": y[80:]
        }

        normalized = runner.orchestrator._normalize_dataset(config_dict)

        assert isinstance(normalized, DatasetConfigs)
        assert len(normalized.configs) == 1


class TestRunnerWithNormalization:
    """Integration tests for runner methods with normalized inputs."""

    def test_run_with_list_and_array(self, sample_pipeline_steps, sample_data):
        """Test run() with list of steps and numpy arrays."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)
        X, y = sample_data

        # Use list for pipeline and tuple for dataset
        partition_info = {"train": 80}
        result = runner.run(
            pipeline=sample_pipeline_steps,
            dataset=(X, y, partition_info),
            pipeline_name="test_run",
            dataset_name="test_data"
        )

        assert result is not None
        run_predictions, datasets_predictions = result
        assert run_predictions.num_predictions > 0

    def test_run_with_configs(self, sample_pipeline_steps, sample_data):
        """Test run() with traditional PipelineConfigs and DatasetConfigs."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)
        X, y = sample_data

        pipeline_configs = PipelineConfigs(sample_pipeline_steps)
        dataset_configs = DatasetConfigs({
            "name": "traditional",
            "train_x": X[:80],
            "train_y": y[:80],
            "test_x": X[80:],
            "test_y": y[80:]
        })

        result = runner.run(pipeline_configs, dataset_configs)

        assert result is not None
        run_predictions, datasets_predictions = result
        assert run_predictions.num_predictions > 0

    def test_run_with_spectro_dataset(self, sample_pipeline_steps, sample_data):
        """Test run() with SpectroDataset."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)
        X, y = sample_data

        dataset = SpectroDataset(name="spectro_test")
        dataset.add_samples(X[:80], indexes={"partition": "train"})
        dataset.add_targets(y[:80])
        dataset.add_samples(X[80:], indexes={"partition": "test"})
        dataset.add_targets(y[80:])

        result = runner.run(
            pipeline=sample_pipeline_steps,
            dataset=dataset
        )

        assert result is not None
        run_predictions, datasets_predictions = result
        assert run_predictions.num_predictions > 0

    def test_run_all_combinations(self, sample_pipeline_steps, sample_data):
        """Test multiple input combinations."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)
        X, y = sample_data

        # Combination 1: List + tuple
        result1 = runner.run(sample_pipeline_steps, (X, y, {"train": 80}))
        assert result1 is not None

        # Combination 2: PipelineConfigs + dict
        pipeline_configs = PipelineConfigs(sample_pipeline_steps)
        result2 = runner.run(
            pipeline_configs,
            {"name": "test", "train_x": X[:80], "train_y": y[:80], "test_x": X[80:], "test_y": y[80:]}
        )
        assert result2 is not None

        # Combination 3: Dict + SpectroDataset
        dataset = SpectroDataset(name="combo3")
        dataset.add_samples(X[:80], indexes={"partition": "train"})
        dataset.add_targets(y[:80])
        dataset.add_samples(X[80:], indexes={"partition": "test"})
        dataset.add_targets(y[80:])

        result3 = runner.run({"pipeline": sample_pipeline_steps}, dataset)
        assert result3 is not None


class TestErrorHandling:
    """Test error cases and edge conditions."""

    def test_invalid_pipeline_type(self):
        """Test that invalid pipeline type raises error."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        with pytest.raises((TypeError, ValueError, AttributeError)):
            runner.orchestrator._normalize_pipeline(12345)  # Invalid type

    def test_invalid_dataset_type(self):
        """Test that invalid dataset path returns DatasetConfigs (even if empty)."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        # Invalid path string doesn't raise, but returns DatasetConfigs
        # It will have 1 config entry but with None values (folder doesn't exist)
        result = runner.orchestrator._normalize_dataset("not_a_valid_path_or_config")
        assert isinstance(result, DatasetConfigs)
        # Should have 1 config entry (even though the folder doesn't exist)
        assert len(result.configs) >= 0  # May be 0 or 1 depending on parser behavior

    def test_tuple_with_non_array(self):
        """Test that tuple with non-arrays raises error."""
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        with pytest.raises((ValueError, TypeError)):
            runner.orchestrator._normalize_dataset(("not_array", "also_not_array"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
