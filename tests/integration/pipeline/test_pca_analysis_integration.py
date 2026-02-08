"""
Integration tests for PCA preprocessing analysis (Q9 example).

Tests PCA-based preprocessing evaluation and cross-dataset metrics.
Based on Q9_acp_spread.py example.
"""

import pytest
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.analysis.transfer import PreprocPCAEvaluator

from tests.fixtures.data_generators import TestDataManager


class TestPCAAnalysisIntegration:
    """Integration tests for PCA preprocessing analysis."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multiple datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression", n_train=48, n_val=16)
        manager.create_classification_dataset("classification", n_train=54, n_val=18)
        yield manager
        manager.cleanup()

    def test_pca_preprocessing_evaluation(self, test_data_manager):
        """Test Q9 style PCA preprocessing evaluation."""
        temp_dir = test_data_manager.get_temp_directory()
        data_paths = [
            str(temp_dir / "regression"),
            str(temp_dir / "classification"),
        ]

        preprocessing_options = [
            Detrend, FirstDerivative, SecondDerivative, Gaussian,
            StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
        ]

        pipeline = [
            MinMaxScaler(),
            {"_or_": preprocessing_options[:3], "pick": (1, 2), "count": 2},
        ]

        pipeline_config = PipelineConfigs(pipeline, "PCA_eval_test")
        dataset_config = DatasetConfigs(data_paths)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, keep_datasets=True, plots_visible=False)
        predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

        # Get datasets for PCA analysis
        datasets_raw = runner.raw_data
        datasets_pp = runner.pp_data

        # Verify datasets were kept
        assert datasets_raw is not None
        assert datasets_pp is not None
        assert len(datasets_raw) > 0

        # Create and fit PCA evaluator
        evaluator = PreprocPCAEvaluator(r_components=4, knn=5)
        evaluator.fit(datasets_raw, datasets_pp)

        # Verify results dataframe exists
        assert evaluator.df_ is not None
        assert len(evaluator.df_) > 0

        # Check expected columns
        expected_cols = ['dataset', 'preproc', 'evr_pre']
        for col in expected_cols:
            assert col in evaluator.df_.columns

    def test_pca_evaluator_metrics(self, test_data_manager):
        """Test PCA evaluator computes key metrics."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"_or_": [Gaussian, StandardNormalVariate, Haar], "pick": 1, "count": 2},
        ]

        pipeline_config = PipelineConfigs(pipeline, "PCA_metrics_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, keep_datasets=True, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        datasets_raw = runner.raw_data
        datasets_pp = runner.pp_data

        evaluator = PreprocPCAEvaluator(r_components=4, knn=3)
        evaluator.fit(datasets_raw, datasets_pp)

        # Check metrics exist and are finite
        assert 'evr_pre' in evaluator.df_.columns
        assert evaluator.df_['evr_pre'].notna().all()
        assert (evaluator.df_['evr_pre'] >= 0).all()
        assert (evaluator.df_['evr_pre'] <= 1).all()

    def test_pca_cross_dataset_analysis(self, test_data_manager):
        """Test cross-dataset PCA analysis."""
        temp_dir = test_data_manager.get_temp_directory()
        data_paths = [
            str(temp_dir / "regression"),
            str(temp_dir / "classification"),
        ]

        pipeline = [
            MinMaxScaler(),
            {"_or_": [Gaussian, StandardNormalVariate], "pick": 1, "count": 2},
        ]

        pipeline_config = PipelineConfigs(pipeline, "cross_dataset_test")
        dataset_config = DatasetConfigs(data_paths)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, keep_datasets=True, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        datasets_raw = runner.raw_data
        datasets_pp = runner.pp_data

        evaluator = PreprocPCAEvaluator(r_components=4, knn=5)
        evaluator.fit(datasets_raw, datasets_pp)

        # Check cross-dataset dataframe
        if not evaluator.cross_dataset_df_.empty:
            assert 'dataset_1' in evaluator.cross_dataset_df_.columns
            assert 'dataset_2' in evaluator.cross_dataset_df_.columns
            assert 'preproc' in evaluator.cross_dataset_df_.columns

    def test_pca_with_minimal_components(self, test_data_manager):
        """Test PCA evaluator with minimal components."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            {"_or_": [Gaussian, Detrend], "pick": 1, "count": 2},
        ]

        pipeline_config = PipelineConfigs(pipeline, "minimal_pca_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, keep_datasets=True, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        datasets_raw = runner.raw_data
        datasets_pp = runner.pp_data

        # Use minimal components
        evaluator = PreprocPCAEvaluator(r_components=3, knn=5)
        evaluator.fit(datasets_raw, datasets_pp)

        assert evaluator.df_ is not None
        assert len(evaluator.df_) > 0

    def test_pca_without_keep_datasets_flag(self, test_data_manager):
        """Test that PCA analysis requires keep_datasets=True."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")

        pipeline = [
            MinMaxScaler(),
            Gaussian(),
        ]

        pipeline_config = PipelineConfigs(pipeline, "no_keep_test")
        dataset_config = DatasetConfigs(dataset_folder)

        # Run without keep_datasets=True
        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0, keep_datasets=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # raw_data and pp_data should not be accessible or should be None/empty
        assert not hasattr(runner, 'raw_data') or runner.raw_data is None or len(runner.raw_data) == 0
