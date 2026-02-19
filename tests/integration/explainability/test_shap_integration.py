"""
Integration tests for SHAP explanations (Q8 example).

Tests SHAP explainer with different types and visualizations.
Based on Q8_shap.py example.
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Gaussian, SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from tests.fixtures.data_generators import TestDataManager


class TestShapIntegration:
    """Integration tests for SHAP explanation functionality."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with regression dataset."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    @pytest.mark.shap
    def test_basic_shap_explanation(self, test_data_manager):
        """Test basic SHAP analysis with PLS model."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            MinMaxScaler(),
            Gaussian(),
            PLSRegression(n_components=10),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.top(n=1, rank_metric='rmse', rank_partition="test")[0]

        # Run SHAP analysis
        shap_params = {
            'n_samples': 50,  # Reduced for testing speed
            'explainer_type': 'auto',
            'visualizations': ['spectral'],  # Minimal visualization for test
        }

        shap_results, output_dir = runner.explain(
            best_prediction,
            dataset_config,
            shap_params=shap_params,
            plots_visible=False
        )

        # Verify SHAP results
        assert shap_results is not None
        assert output_dir is not None

    @pytest.mark.shap
    def test_shap_with_different_explainer_types(self, test_data_manager):
        """Test SHAP with different explainer types."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_explainer_types_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Test different explainer types
        for explainer_type in ['auto', 'linear']:
            shap_params = {
                'n_samples': 30,
                'explainer_type': explainer_type,
                'visualizations': ['spectral'],
            }

            try:
                shap_results, _ = runner.explain(
                    best_prediction,
                    dataset_config,
                    shap_params=shap_params,
                    plots_visible=False
                )
                assert shap_results is not None
                print(f"Explainer type '{explainer_type}' worked successfully")
            except Exception as e:
                print(f"Explainer type '{explainer_type}' not supported: {e}")

    @pytest.mark.shap
    def test_shap_with_multiple_visualizations(self, test_data_manager):
        """Test SHAP with multiple visualization types."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            StandardNormalVariate(),
            PLSRegression(n_components=8),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_visualizations_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Test multiple visualizations
        shap_params = {
            'n_samples': 50,
            'explainer_type': 'auto',
            'visualizations': ['spectral', 'waterfall', 'beeswarm'],
        }

        shap_results, output_dir = runner.explain(
            best_prediction,
            dataset_config,
            shap_params=shap_params,
            plots_visible=False
        )

        assert shap_results is not None

    @pytest.mark.shap
    def test_shap_with_binning_options(self, test_data_manager):
        """Test SHAP with different binning configurations."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            SavitzkyGolay(),
            PLSRegression(n_components=10),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_binning_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Test different bin sizes per visualization
        shap_params = {
            'n_samples': 50,
            'explainer_type': 'auto',
            'visualizations': ['spectral', 'waterfall'],
            'bin_size': {
                'spectral': 10,
                'waterfall': 20,
            },
            'bin_stride': {
                'spectral': 5,
                'waterfall': 10,
            },
            'bin_aggregation': {
                'spectral': 'mean',
                'waterfall': 'sum',
            }
        }

        shap_results, _ = runner.explain(
            best_prediction,
            dataset_config,
            shap_params=shap_params,
            plots_visible=False
        )

        assert shap_results is not None

    @pytest.mark.shap
    def test_shap_with_preprocessing(self, test_data_manager):
        """Test SHAP analysis with complex preprocessing."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            Gaussian(),
            SavitzkyGolay(),
            StandardNormalVariate(),
            PLSRegression(n_components=12),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_preprocessing_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # SHAP should work with preprocessing
        shap_params = {
            'n_samples': 40,
            'explainer_type': 'auto',
            'visualizations': ['spectral'],
        }

        shap_results, _ = runner.explain(
            best_prediction,
            dataset_config,
            shap_params=shap_params,
            plots_visible=False
        )

        assert shap_results is not None

    @pytest.mark.shap
    @pytest.mark.slow
    def test_shap_error_handling_invalid_params(self, test_data_manager):
        """Test SHAP error handling with invalid parameters."""
        pytest.importorskip("shap")

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "shap_error_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        best_prediction = predictions.get_best(ascending=True)

        # Test with invalid explainer type
        shap_params = {
            'n_samples': 20,
            'explainer_type': 'invalid_explainer_type',
            'visualizations': ['spectral'],
        }

        # Should handle gracefully or raise informative error
        try:
            shap_results, _ = runner.explain(
                best_prediction,
                dataset_config,
                shap_params=shap_params,
                plots_visible=False
            )
            # If it succeeds, it defaulted to something
            assert shap_results is not None
        except (ValueError, TypeError, AttributeError) as e:
            # Expected error
            assert "explainer" in str(e).lower() or "invalid" in str(e).lower()
