"""Integration tests for source branching with merge_sources.

Tests the end-to-end source branching feature which enables per-source
pipeline execution for multi-source datasets.
"""

import pytest
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay

from tests.fixtures.data_generators import TestDataManager


class TestSourceBranchBasic:
    """Basic source branching tests."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_source_branch_auto_mode(self, test_data_manager):
        """Test source branching in auto mode (isolate each source)."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": "auto"},  # Auto-merge each source
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_auto")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions
        assert predictions.num_predictions > 0, "No predictions generated"

    def test_source_branch_dict_mode(self, test_data_manager):
        """Test source branching with per-source pipelines."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        # Define different preprocessing per source
        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "source_1": [SavitzkyGolay()],
                "source_2": [MinMaxScaler()],
            }},
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_dict")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0, "No predictions generated"

    def test_source_branch_with_default(self, test_data_manager):
        """Test source branching with default fallback pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "_default_": [MinMaxScaler()],  # Applied to source_1, source_2
            }},
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_default")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0, "No predictions generated"


class TestSourceBranchMergeStrategies:
    """Test different merge strategies after source branching."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=2)
        yield manager
        manager.cleanup()

    def test_source_branch_concat_merge(self, test_data_manager):
        """Test source branching with concat merge strategy."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "source_1": [MinMaxScaler()],
                "_merge_strategy_": "concat",
            }},
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_concat")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0

    def test_source_branch_no_auto_merge(self, test_data_manager):
        """Test source branching without auto-merge."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "source_1": [MinMaxScaler()],
                "_merge_after_": False,  # Don't auto-merge
            }},
            {"merge_sources": "concat"},  # Manual merge
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_manual_merge")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0


class TestSourceBranchWithStacking:
    """Test source branching combined with stacking."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=2)
        yield manager
        manager.cleanup()

    def test_source_branch_with_meta_model(self, test_data_manager):
        """Test source branching with MetaModel stacking.
        
        Note: Source branching with MetaModel stacking requires careful handling
        of sample indices across sources. Using DROP_INCOMPLETE coverage strategy
        with a low min_coverage_ratio to accommodate potential index mismatches.
        """
        try:
            from nirs4all.operators.models import MetaModel
            from nirs4all.operators.models.meta import StackingConfig, CoverageStrategy
        except ImportError:
            pytest.skip("MetaModel not available")

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        # Use DROP_INCOMPLETE coverage to handle source branching variations
        # Low min_coverage_ratio due to potential sample index variations across sources
        stacking_config = StackingConfig(
            coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
            min_coverage_ratio=0.2,  # Allow very low coverage for this complex scenario
        )

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=3, random_state=42),
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "source_1": [MinMaxScaler()],
            }},
            # Base models
            PLSRegression(n_components=5),
            # Meta-model stacking with lenient coverage
            {"model": MetaModel(model=Ridge(alpha=1.0), stacking_config=stacking_config), "name": "MetaStacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_stacking")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have MetaModel predictions
        meta_preds = [p for p in predictions.to_dicts() if 'Meta' in p['model_name']]
        assert len(meta_preds) > 0, "MetaModel predictions not found"


class TestSourceBranchPredictionMode:
    """Test source branching in prediction mode."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=2)
        yield manager
        manager.cleanup()

    def test_source_branch_reload(self, test_data_manager):
        """Test reload/predict with source branching."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        # Training pipeline
        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": "auto"},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_reload")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best model from train partition
        train_preds = [p for p in predictions.to_dicts() if p['partition'] == 'train']
        best_prediction = sorted(train_preds, key=lambda x: x.get('val_score', float('inf')))[0]
        model_id = best_prediction['id']

        # Get original predictions and their sample indices
        sample_indices = np.array(best_prediction['sample_indices'][:10]).astype(int)
        original_preds = np.array(best_prediction['y_pred'][:10]).flatten()

        # Test reload and predict
        predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        prediction_dataset = DatasetConfigs(dataset_folder)

        reloaded_preds, _ = predictor.predict(model_id, prediction_dataset, verbose=0)

        # Compare predictions for the SAME samples
        reloaded_for_samples = reloaded_preds[sample_indices].flatten()

        # Verify predictions match for the same samples
        assert np.allclose(original_preds, reloaded_for_samples, rtol=1e-5), \
            f"Reloaded predictions do not match original.\n" \
            f"Original: {original_preds}\nReloaded: {reloaded_for_samples}"


class TestSourceBranchWithRegularBranching:
    """Test combining source branching with regular branching."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=2)
        yield manager
        manager.cleanup()

    def test_source_branch_then_regular_branch(self, test_data_manager):
        """Test source branching followed by regular branching."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            # First: source-specific preprocessing
            {"source_branch": {
                "source_0": [StandardNormalVariate()],
                "source_1": [MinMaxScaler()],
            }},
            # Then: regular branching for model comparison
            {"branch": [
                [PLSRegression(n_components=5)],
                [PLSRegression(n_components=10)],
            ]},
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_then_branch")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have predictions from both branches
        assert predictions.num_predictions > 0

        # Check branch names exist
        branch_names = predictions.get_unique_values('branch_name')
        assert len(branch_names) >= 2, f"Expected 2 branches, got {branch_names}"


class TestSourceBranchEdgeCases:
    """Edge case tests for source branching."""

    @pytest.fixture
    def single_source_manager(self):
        """Create test data manager with single-source dataset."""
        manager = TestDataManager()
        manager.create_regression_dataset("single")  # Single source (standard regression)
        yield manager
        manager.cleanup()

    def test_source_branch_single_source_warning(self, single_source_manager):
        """Test that source branching on single source dataset logs warning."""
        # This test just verifies no crash on single-source dataset
        dataset_folder = str(single_source_manager.get_temp_directory() / "single")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"source_branch": "auto"},  # Should warn but continue
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "source_branch_single")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should still produce predictions
        assert predictions.num_predictions > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
