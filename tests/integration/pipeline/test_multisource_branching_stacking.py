"""
Integration tests for multisource + branching + stacking + reload.

These tests verify that complex pipeline features work correctly
when combined with multi-source datasets (multiple X arrays).

Based on investigation of Roadmap.md concern:
> [multisource] with reload, branches and stack
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import FirstDerivative, SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from tests.fixtures.data_generators import TestDataManager


class TestMultisourceBranching:
    """Test branching with multi-source datasets."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_basic_branching_multisource(self, test_data_manager):
        """Test basic branching with multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [SavitzkyGolay()],
            ]},
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_branching_test")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions for each branch
        assert predictions.num_predictions > 0, "No predictions generated"

        # Check branch names
        branch_names = predictions.get_unique_values('branch_name')
        assert len(branch_names) >= 2, f"Expected 2 branches, got {branch_names}"

        # Verify each branch has valid predictions
        for pred in predictions.to_dicts():
            # Check for valid val_score or test_score (these are the top-level score fields)
            has_valid_score = np.isfinite(pred.get('val_score', np.nan)) or np.isfinite(pred.get('test_score', np.nan))
            assert has_valid_score, f"Invalid score for {pred.get('branch_name')}"

    def test_named_branching_multisource(self, test_data_manager):
        """Test named branches with multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "snv": [StandardNormalVariate()],
                "savgol": [SavitzkyGolay()],
                "deriv": [FirstDerivative()],
            }},
            PLSRegression(n_components=5),
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_named_branching")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify all named branches exist
        branch_names = predictions.get_unique_values('branch_name')
        assert "snv" in branch_names, f"Missing 'snv' branch: {branch_names}"
        assert "savgol" in branch_names, f"Missing 'savgol' branch: {branch_names}"
        assert "deriv" in branch_names, f"Missing 'deriv' branch: {branch_names}"

    def test_branching_multisource_reload(self, test_data_manager):
        """Test reload/predict with branching on multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        # Training pipeline with branches
        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [SavitzkyGolay()],
            ]},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_branching_reload")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get best model from train partition to have sample indices
        train_preds = [p for p in predictions.to_dicts() if p['partition'] == 'train']
        best_prediction = sorted(train_preds, key=lambda x: x.get('val_score', float('inf')))[0]
        model_id = best_prediction['id']

        # Get original predictions and their sample indices
        sample_indices = np.array(best_prediction['sample_indices'][:10]).astype(int)
        original_preds = np.array(best_prediction['y_pred'][:10]).flatten()

        print(f"Best model: {best_prediction['model_name']} (branch: {best_prediction.get('branch_name')})")
        print(f"Model ID: {model_id}")

        # Test reload and predict
        predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        prediction_dataset = DatasetConfigs(dataset_folder)

        reloaded_preds, _ = predictor.predict(model_id, prediction_dataset, verbose=0)

        # Compare predictions for the SAME samples (reloaded is in sample order)
        reloaded_for_samples = reloaded_preds[sample_indices].flatten()

        print(f"Original: {original_preds}")
        print(f"Reloaded: {reloaded_for_samples}")

        # Verify predictions match for the same samples
        assert np.allclose(original_preds, reloaded_for_samples, rtol=1e-5), \
            f"Reloaded predictions do not match original.\n" \
            f"Original: {original_preds}\nReloaded: {reloaded_for_samples}"

class TestMultisourceStacking:
    """Test stacking with multi-source datasets."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_sklearn_stacking_multisource(self, test_data_manager):
        """Test sklearn StackingRegressor with multi-source data."""
        from sklearn.ensemble import StackingRegressor

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        # Create stacking regressor
        base_estimators = [
            ('pls', PLSRegression(n_components=5)),
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=2
        )

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"model": stacking, "name": "Stacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_sklearn_stacking")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0, "No stacking predictions"

        # Verify stacking model worked
        stacking_preds = [p for p in predictions.to_dicts() if 'Stacking' in p['model_name']]
        assert len(stacking_preds) > 0, "Stacking model not found in predictions"

    def test_sklearn_stacking_multisource_reload(self, test_data_manager):
        """Test reload of sklearn stacking model with multi-source data."""
        from sklearn.ensemble import StackingRegressor

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        base_estimators = [
            ('pls', PLSRegression(n_components=5)),
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=2
        )

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"model": stacking, "name": "Stacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_stacking_reload")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Get stacking model (train partition to get predictions with sample_indices)
        stacking_preds = [p for p in predictions.to_dicts()
                         if 'Stacking' in p['model_name'] and p['partition'] == 'train']
        stacking_pred = stacking_preds[0]
        model_id = stacking_pred['id']

        # Get original predictions and their sample indices
        sample_indices = np.array(stacking_pred['sample_indices'][:10]).astype(int)
        original_preds = np.array(stacking_pred['y_pred'][:10]).flatten()

        # Reload and predict
        predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        reloaded_preds, _ = predictor.predict(model_id, DatasetConfigs(dataset_folder), verbose=0)

        # Compare predictions for the SAME samples (reloaded is in sample order)
        reloaded_for_samples = reloaded_preds[sample_indices].flatten()

        assert np.allclose(original_preds, reloaded_for_samples, rtol=1e-5), \
            f"Stacking reload predictions do not match.\n" \
            f"Original: {original_preds}\nReloaded: {reloaded_for_samples}"

class TestMultisourceBranchingStacking:
    """Test combined branching + stacking with multi-source datasets."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_branching_with_stacking_multisource(self, test_data_manager):
        """Test branches with stacking ensemble on multi-source data."""
        from sklearn.ensemble import StackingRegressor

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        base_estimators = [
            ('pls', PLSRegression(n_components=5)),
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=2
        )

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "snv": [StandardNormalVariate()],
                "savgol": [SavitzkyGolay()],
            }},
            {"model": stacking, "name": "Stacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_branch_stacking")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have stacking predictions for each branch
        all_preds = predictions.to_dicts()
        branch_names = {p.get('branch_name') for p in all_preds}

        assert "snv" in branch_names, f"Missing snv branch: {branch_names}"
        assert "savgol" in branch_names, f"Missing savgol branch: {branch_names}"

        # Each branch should have stacking model
        for branch in ["snv", "savgol"]:
            branch_preds = [p for p in all_preds if p.get('branch_name') == branch]
            stacking_in_branch = [p for p in branch_preds if 'Stacking' in p['model_name']]
            assert len(stacking_in_branch) > 0, f"No stacking in {branch} branch"

    def test_in_branch_models_multisource(self, test_data_manager):
        """Test models inside branches with multi-source data."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, random_state=42),
            {"branch": {
                "snv_pls": [StandardNormalVariate(), PLSRegression(n_components=5)],
                "savgol_pls": [SavitzkyGolay(), PLSRegression(n_components=5)],
                "deriv_rf": [FirstDerivative(), RandomForestRegressor(n_estimators=10, random_state=42)],
            }},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_in_branch_models")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        branch_names = predictions.get_unique_values('branch_name')
        assert "snv_pls" in branch_names
        assert "savgol_pls" in branch_names
        assert "deriv_rf" in branch_names

        # Each branch should have predictions
        for branch in ["snv_pls", "savgol_pls", "deriv_rf"]:
            branch_preds = predictions.filter_predictions(branch_name=branch)
            assert len(branch_preds) > 0, f"No predictions for {branch}"

class TestMultisourceMetaModel:
    """Test MetaModel stacking with multi-source datasets."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with multi-source dataset."""
        manager = TestDataManager()
        manager.create_multi_source_dataset("multi", n_sources=3)
        yield manager
        manager.cleanup()

    def test_metamodel_multisource(self, test_data_manager):
        """Test MetaModel stacking on multi-source data."""
        try:
            from nirs4all.operators.models import MetaModel
        except ImportError:
            pytest.skip("MetaModel not available")

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=5),
            RandomForestRegressor(n_estimators=10, random_state=42),
            {"model": MetaModel(model=Ridge(alpha=1.0)), "name": "MetaStacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_metamodel")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Should have MetaModel predictions
        meta_preds = [p for p in predictions.to_dicts() if 'Meta' in p['model_name']]
        assert len(meta_preds) > 0, "MetaModel predictions not found"

    def test_metamodel_with_branches_multisource(self, test_data_manager):
        """Test MetaModel with branches on multi-source data."""
        try:
            from nirs4all.operators.models import MetaModel
        except ImportError:
            pytest.skip("MetaModel not available")

        dataset_folder = str(test_data_manager.get_temp_directory() / "multi")

        pipeline = [
            MinMaxScaler(),
            {"y_processing": MinMaxScaler()},
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": {
                "snv": [StandardNormalVariate()],
                "savgol": [SavitzkyGolay()],
            }},
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0)), "name": "MetaStacking"},
        ]

        pipeline_config = PipelineConfigs(pipeline, "multisource_metamodel_branches")
        dataset_config = DatasetConfigs(dataset_folder)

        runner = PipelineRunner(save_artifacts=True, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # MetaModel should work per branch
        branch_names = predictions.get_unique_values('branch_name')
        assert len(branch_names) >= 2, f"Expected branches, got {branch_names}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
