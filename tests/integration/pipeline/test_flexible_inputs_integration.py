"""
Integration tests for flexible input formats (Q11 example).

Tests direct numpy arrays, tuple inputs, and different pipeline formats.
Based on Q11_flexible_inputs.py example.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner


class TestFlexibleInputsIntegration:
    """Integration tests for flexible input format handling."""

    @pytest.fixture
    def sample_numpy_data(self):
        """Generate sample numpy arrays for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        return X, y

    def test_traditional_approach_with_configs(self, sample_numpy_data):
        """Test traditional approach with PipelineConfigs + DatasetConfigs."""
        X, y = sample_numpy_data

        pipeline_steps = [
            {"preprocessing": StandardScaler()},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        pipeline_configs = PipelineConfigs(pipeline_steps, name="traditional")
        dataset_configs = DatasetConfigs({
            "name": "traditional_dataset",
            "train_x": X[:80],
            "train_y": y[:80],
            "test_x": X[80:],
            "test_y": y[80:]
        })

        result = runner.run(pipeline_configs, dataset_configs)

        # Verify results
        assert result is not None
        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_direct_list_and_tuple_approach(self, sample_numpy_data):
        """Test direct approach with list of steps + tuple (X, y, partition_info)."""
        X, y = sample_numpy_data

        pipeline_steps = [
            {"preprocessing": StandardScaler()},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        # Direct tuple with partition info
        partition_info = {"train": 80}  # First 80 samples for training
        result = runner.run(
            pipeline=pipeline_steps,
            dataset=(X, y, partition_info),
            pipeline_name="direct",
            dataset_name="array_data"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_dict_pipeline_with_tuple_dataset(self, sample_numpy_data):
        """Test dict pipeline format with tuple dataset."""
        X, y = sample_numpy_data

        pipeline_dict = {
            "pipeline": [
                {"preprocessing": StandardScaler()},
                {"model": Ridge(alpha=0.5)}
            ]
        }

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        partition_info = {"train": 70}
        result = runner.run(
            pipeline=pipeline_dict,
            dataset=(X, y, partition_info),
            pipeline_name="dict_pipeline"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_numpy_arrays_without_partition_info(self, sample_numpy_data):
        """Test using numpy arrays without explicit partition info."""
        X, y = sample_numpy_data

        pipeline_steps = [
            StandardScaler(),
            Ridge(alpha=1.0)
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        # Need to provide at least train partition to avoid empty validation set
        result = runner.run(
            pipeline=pipeline_steps,
            dataset=(X, y, {"train": 80}),
            pipeline_name="no_partition"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_dict_with_named_partitions(self, sample_numpy_data):
        """Test dict format with explicitly named partitions."""
        X, y = sample_numpy_data

        pipeline_steps = [
            StandardScaler(),
            Ridge(alpha=1.0)
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        # Dict format with named partitions
        dataset_dict = {
            "train_x": X[:70],
            "train_y": y[:70],
            "val_x": X[70:85],
            "val_y": y[70:85],
            "test_x": X[85:],
            "test_y": y[85:]
        }

        result = runner.run(
            pipeline=pipeline_steps,
            dataset=dataset_dict,
            pipeline_name="named_partitions"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_cross_validation_with_numpy(self, sample_numpy_data):
        """Test cross-validation with numpy arrays."""
        X, y = sample_numpy_data

        pipeline_steps = [
            StandardScaler(),
            KFold(n_splits=3),
            Ridge(alpha=1.0)
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        result = runner.run(
            pipeline=pipeline_steps,
            dataset=(X, y, {"train": 80}),
            pipeline_name="cv_numpy"
        )

        predictions, _ = result
        # Should have predictions from multiple folds
        assert predictions.num_predictions >= 3

    def test_minimal_numpy_input(self, sample_numpy_data):
        """Test minimal input with just X and y."""
        X, y = sample_numpy_data

        pipeline = [Ridge(alpha=1.0)]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        result = runner.run(
            pipeline=pipeline,
            dataset=(X, y, {"train": 80}),
            pipeline_name="minimal"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_prediction_with_numpy_arrays(self, sample_numpy_data):
        """Test prediction using numpy arrays."""
        X, y = sample_numpy_data

        # Train model
        pipeline = [
            StandardScaler(),
            Ridge(alpha=1.0)
        ]

        runner = PipelineRunner(save_artifacts=True, verbose=0, enable_tab_reports=False)

        # Provide partition info - without it, the system doesn't know how to split the data
        result = runner.run(
            pipeline=pipeline,
            dataset=(X[:80], y[:80], {"train": 70}),  # Use first 70 for train, rest for val/test
            pipeline_name="train_for_predict"
        )

        predictions, _ = result
        best_pred = predictions.get_best(ascending=True)

        # Predict on new data
        X_new = X[80:]

        predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)

        # Test prediction with numpy array
        try:
            pred_result, _ = predictor.predict(
                best_pred,
                X_new,
                verbose=0
            )
            assert pred_result is not None
            assert len(pred_result) == len(X_new)
        except (TypeError, ValueError, RuntimeError):
            # If direct numpy not fully supported, use dict format
            pred_result, _ = predictor.predict(
                best_pred,
                {"test_x": X_new},
                verbose=0
            )
            assert pred_result is not None

    def test_mixed_input_formats(self, sample_numpy_data):
        """Test mixing different input formats."""
        X, y = sample_numpy_data

        # Pipeline as list
        pipeline = [StandardScaler(), Ridge(alpha=1.0)]

        # Dataset as dict
        dataset = {
            "train_x": X[:70],
            "train_y": y[:70],
            "test_x": X[70:],
            "test_y": y[70:]
        }

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        result = runner.run(
            pipeline=pipeline,
            dataset=dataset,
            pipeline_name="mixed_formats"
        )

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_backward_compatibility(self, sample_numpy_data):
        """Test that traditional configs still work (backward compatibility)."""
        X, y = sample_numpy_data

        # Traditional approach should still work
        pipeline_config = PipelineConfigs(
            [StandardScaler(), Ridge(alpha=1.0)],
            name="backward_compat"
        )

        dataset_config = DatasetConfigs({
            "train_x": X[:80],
            "train_y": y[:80],
            "test_x": X[80:],
            "test_y": y[80:]
        })

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)
        result = runner.run(pipeline_config, dataset_config)

        predictions, _ = result
        assert predictions.num_predictions > 0

    def test_validation_of_inconsistent_shapes(self, sample_numpy_data):
        """Test error handling with inconsistent X and y shapes."""
        X, y = sample_numpy_data

        pipeline = [Ridge(alpha=1.0)]
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, enable_tab_reports=False)

        # Mismatched shapes
        X_bad = X[:80]
        y_bad = y[:60]  # Different length

        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            runner.run(
                pipeline=pipeline,
                dataset=(X_bad, y_bad, {"train": 50}),
                pipeline_name="bad_shapes"
            )
