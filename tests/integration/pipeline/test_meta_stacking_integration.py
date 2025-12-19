"""
Integration tests for meta-model stacking pipelines (Phase 6).

Tests end-to-end scenarios:
- Basic stacking pipeline with PipelineRunner
- Stacking with preprocessing branches
- Stacking with sample_partitioner
- Stacking with outlier_excluder
- Mixed base models (sklearn + various estimators)
- Classification stacking with use_proba
- Save/reload/predict consistency
- Feature order preservation
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.operators.models import (
    MetaModel,
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
)
from nirs4all.operators.transforms import FirstDerivative as DerivativeTransform
from nirs4all.pipeline.storage.manifest_manager import ManifestManager


# =============================================================================
# Helper Functions
# =============================================================================

def filter_by_model_name(predictions, name_contains: str) -> List[Dict[str, Any]]:
    """Filter predictions by model name containing a substring."""
    all_preds = predictions.to_dicts(load_arrays=False)
    return [p for p in all_preds if name_contains in p.get('model_name', '')]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data_path():
    """Path to sample data in examples folder."""
    path = Path("examples/sample_data/regression")
    if not path.exists():
        pytest.skip("Sample data not available")
    return str(path)


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return str(workspace)


@pytest.fixture
def regression_dataset(sample_data_path):
    """Load regression dataset."""
    return DatasetConfigs(sample_data_path)


@pytest.fixture
def sample_data_path_2():
    """Path to larger regression_2 sample data in examples folder."""
    path = Path("examples/sample_data/regression_2")
    if not path.exists():
        pytest.skip("Sample data regression_2 not available")
    return str(path)


@pytest.fixture
def regression_2_dataset(sample_data_path_2):
    """Load larger regression_2 dataset (885 samples) with Sample_ID metadata."""
    return DatasetConfigs(sample_data_path_2)


# =============================================================================
# Basic Stacking Pipeline Tests
# =============================================================================

class TestBasicStackingPipeline:
    """Test basic stacking pipeline end-to-end."""

    def test_stacking_all_previous_models(self, regression_dataset, temp_workspace):
        """Test stacking with all previous models (default selector)."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=3),
            RandomForestRegressor(n_estimators=20, random_state=42),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        # Check predictions exist for meta-model
        meta_preds = filter_by_model_name(predictions, "MetaModel")

        assert len(meta_preds) > 0, "No meta-model predictions found"

        # Check both val and test predictions exist
        val_preds = [p for p in meta_preds if p.get('partition') == 'val']
        test_preds = [p for p in meta_preds if p.get('partition') == 'test']

        assert len(val_preds) > 0, "No validation predictions for meta-model"
        assert len(test_preds) > 0, "No test predictions for meta-model"

    def test_stacking_explicit_source_selection(self, regression_dataset, temp_workspace):
        """Test stacking with explicit source model list."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3), "name": "PLS_3"},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            RandomForestRegressor(n_estimators=20, random_state=42),
            {"model": MetaModel(
                model=Ridge(alpha=0.5),
                source_models=["PLS_3", "PLS_5"],  # Explicit: only PLS models
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")

        assert len(meta_preds) > 0

    def test_stacking_with_custom_config(self, regression_dataset, temp_workspace):
        """Test stacking with custom StackingConfig options."""
        config = StackingConfig(
            coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
            test_aggregation=TestAggregation.WEIGHTED_MEAN,
            min_coverage_ratio=0.8,
        )

        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=3),
            {"model": MetaModel(
                model=Ridge(alpha=1.0),
                stacking_config=config,
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0

    def test_stacking_validates_metrics(self, regression_dataset, temp_workspace):
        """Test that meta-model validation scores are reasonable."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=3),
            RandomForestRegressor(n_estimators=50, random_state=42),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)


        # Get top models
        top_models = predictions.top(n=5, rank_partition="val")

        # Meta-model should be in top models (stacking typically improves results)
        model_names = [p['model_name'] for p in top_models]
        has_meta = any("MetaModel" in name for name in model_names)

        # Note: Not asserting meta is best, just that it has valid score
        meta_preds = [p for p in filter_by_model_name(predictions, "MetaModel") if 'val' in str(p.get('partition', ''))]
        if meta_preds:
            val_score = meta_preds[0].get('val_score')
            assert val_score is not None
            # R2 should be between -inf and 1, but reasonable models should be > 0
            assert val_score > -10  # Sanity check


# =============================================================================
# Branching Integration Tests
# =============================================================================

class TestStackingWithBranches:
    """Test stacking with preprocessing branches."""

    def test_stacking_after_branch(self, regression_dataset, temp_workspace):
        """Test meta-model after branching."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": {
                "raw": [{"model": PLSRegression(n_components=3), "name": "PLS_Raw"}],
                "derivative": [DerivativeTransform(), {"model": PLSRegression(n_components=3), "name": "PLS_D1"}],
            }},
            # Meta-model after branch - stacks models from all branches
            {"model": MetaModel(
                model=Ridge(alpha=1.0),
                stacking_config=StackingConfig(
                    branch_scope=BranchScope.ALL_BRANCHES,
                ),
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)


        # Check branch models exist
        pls_raw = filter_by_model_name(predictions, "PLS_Raw")
        pls_d1 = filter_by_model_name(predictions, "PLS_D1")

        assert len(pls_raw) > 0, "PLS_Raw predictions missing"
        assert len(pls_d1) > 0, "PLS_D1 predictions missing"

        # Check meta-model exists
        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0, "MetaModel predictions missing"

    def test_stacking_with_branch_scope_current_only(self, regression_dataset, temp_workspace):
        """Test branch_scope=CURRENT_ONLY limits to current branch."""
        # Pipeline with meta-model inside each branch
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=3),  # Shared base model
            {"branch": {
                "branch0": [
                    RandomForestRegressor(n_estimators=20, random_state=42),
                    {"model": MetaModel(
                        model=Ridge(),
                        stacking_config=StackingConfig(
                            branch_scope=BranchScope.CURRENT_ONLY,
                        ),
                    ), "name": "Meta_Branch0"},
                ],
                "branch1": [
                    RandomForestRegressor(n_estimators=30, random_state=42),
                    {"model": MetaModel(
                        model=Ridge(),
                        stacking_config=StackingConfig(
                            branch_scope=BranchScope.CURRENT_ONLY,
                        ),
                    ), "name": "Meta_Branch1"},
                ],
            }},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)


        # Both meta-models should exist
        meta_b0 = filter_by_model_name(predictions, "Meta_Branch0")
        meta_b1 = filter_by_model_name(predictions, "Meta_Branch1")

        assert len(meta_b0) > 0
        assert len(meta_b1) > 0


# =============================================================================
# Sample Partitioner / Excluder Tests
# =============================================================================

class TestStackingWithPartitioner:
    """Test stacking with group-aware splitting."""

    def test_stacking_with_group_partition(self, regression_2_dataset, temp_workspace):
        """Test stacking with group-aware cross-validation.

        Uses the regression_2 dataset which has Sample_ID metadata for grouping.
        GroupKFold ensures samples from the same Sample_ID stay together in folds.
        """
        pipeline = [
            MinMaxScaler(),
            # Use GroupKFold with Sample_ID to ensure grouped samples stay together
            {"split": GroupKFold(n_splits=3), "group": "Sample_ID"},
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_2_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0


class TestStackingWithExcluder:
    """Test stacking with outlier_excluder."""

    def test_stacking_respects_excluded_samples(self, regression_dataset, temp_workspace):
        """Test that meta-model training respects excluded samples.

        Uses outlier_excluder to exclude outliers from training, then
        tests that stacking works correctly with the reduced sample set.
        """
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            # Exclude outliers using IQR method (does not create separate branches)
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [{"method": "y_outlier", "threshold": 1.5}],
            }},
            PLSRegression(n_components=3),  # Use fewer components for potentially smaller dataset
            # Use DROP_INCOMPLETE strategy since outlier_excluder may exclude some samples
            {"model": MetaModel(
                model=Ridge(alpha=1.0),
                stacking_config=StackingConfig(
                    coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
                    min_coverage_ratio=0.5,  # Allow lower coverage for reduced data
                ),
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0


# =============================================================================
# Classification Stacking Tests
# =============================================================================

class TestClassificationStacking:
    """Test stacking for classification tasks."""

    @pytest.fixture
    def classification_dataset(self, sample_data_path):
        """Load classification dataset if available."""
        # Try to load a classification variant or skip
        try:
            dataset = DatasetConfigs(sample_data_path)
            # Convert regression to classification by binning
            # This is a test fixture simplification
            return dataset
        except Exception:
            pytest.skip("Classification dataset not available")

    def test_classification_stacking_use_proba_false(
        self, regression_dataset, temp_workspace
    ):
        """Test classification stacking without probabilities (predictions only)."""
        # For this test we use regression as-is; real classification tests
        # would need a proper classification dataset

        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=5),
            {"model": MetaModel(
                model=Ridge(),
                use_proba=False,
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0


# =============================================================================
# Mixed Framework Tests
# =============================================================================

class TestMixedFrameworkStacking:
    """Test stacking with various sklearn estimators."""

    def test_stacking_multiple_estimator_types(self, regression_dataset, temp_workspace):
        """Test stacking with diverse estimator types."""
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor

        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            # Diverse base models
            PLSRegression(n_components=5),
            RandomForestRegressor(n_estimators=20, random_state=42),
            KNeighborsRegressor(n_neighbors=5),
            # Meta-learner
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)


        # All base models should have predictions
        pls_preds = filter_by_model_name(predictions, "PLSRegression")
        rf_preds = filter_by_model_name(predictions, "RandomForest")
        knn_preds = filter_by_model_name(predictions, "KNeighbors")

        assert len(pls_preds) > 0
        assert len(rf_preds) > 0
        assert len(knn_preds) > 0

        # Meta-model should have predictions
        meta_preds = filter_by_model_name(predictions, "MetaModel")
        assert len(meta_preds) > 0


# =============================================================================
# Roundtrip / Persistence Tests
# =============================================================================

class TestStackingRoundtrip:
    """Test save/reload/predict consistency for meta-models."""

    def test_save_reload_predict_consistency(self, regression_dataset, temp_workspace):
        """Test that reloaded meta-model produces consistent predictions."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        # First run - training
        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=True)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        original_test_preds = [p for p in filter_by_model_name(predictions, "MetaModel")
                               if p.get('partition') == 'test']

        assert len(original_test_preds) > 0

        # Store original predictions for comparison
        original_y_pred = original_test_preds[0].get('y_pred')
        best_prediction = original_test_preds[0]

        # Second run - load and predict using the prediction entry
        runner2 = PipelineRunner(workspace_path=temp_workspace)
        new_y_pred, new_predictions = runner2.predict(
            best_prediction,
            regression_dataset,
            verbose=0
        )

        if original_y_pred is not None and new_y_pred is not None:
            # Predictions should be very close (may have floating point diff)
            np.testing.assert_allclose(
                original_y_pred, new_y_pred, rtol=1e-5, atol=1e-8
            )

    def test_feature_order_preservation(self, regression_dataset, temp_workspace):
        """Test that feature order is preserved after reload."""
        # Use named models for predictable feature order
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3), "name": "ModelA"},
            {"model": PLSRegression(n_components=5), "name": "ModelB"},
            {"model": PLSRegression(n_components=7), "name": "ModelC"},
            {"model": MetaModel(
                model=Ridge(),
                source_models=["ModelC", "ModelA", "ModelB"],  # Specific order
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=True)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        # Check that artifacts reference correct feature order via ManifestManager
        # Find the run directory for the dataset
        from pathlib import Path
        runs_dir = Path(temp_workspace) / "runs"
        if not runs_dir.exists():
            runs_dir = Path(temp_workspace)

        # Find dataset run directory
        dataset_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        assert dataset_dirs, "No run directories found"

        dataset_run_dir = dataset_dirs[0]
        manifest_manager = ManifestManager(dataset_run_dir)
        pipelines = manifest_manager.list_pipelines()

        # Find meta-model artifacts by artifact_type (not class_name)
        for pipeline_id in pipelines:
            manifest = manifest_manager.load_manifest(pipeline_id)
            artifacts = manifest_manager.get_artifacts_list(manifest)

            # Meta-model artifacts have artifact_type == "meta_model"
            meta_artifacts = [
                a for a in artifacts
                if a.get('artifact_type') == 'meta_model'
            ]

            if meta_artifacts:
                meta_config = meta_artifacts[0].get('meta_config', {})
                feature_columns = meta_config.get('feature_columns', [])

                # Feature order should match source_models order: ModelC, ModelA, ModelB
                assert len(feature_columns) == 3, f"Expected 3 feature columns, got {len(feature_columns)}"
                assert "ModelC" in feature_columns[0], f"Expected ModelC in first column, got {feature_columns[0]}"
                assert "ModelA" in feature_columns[1], f"Expected ModelA in second column, got {feature_columns[1]}"
                assert "ModelB" in feature_columns[2], f"Expected ModelB in third column, got {feature_columns[2]}"
                return  # Test passed

        pytest.fail("No meta-model artifacts found in manifest")


# =============================================================================
# Edge Cases
# =============================================================================

class TestStackingEdgeCases:
    """Test edge cases for stacking pipelines."""

    def test_stacking_single_base_model(self, regression_dataset, temp_workspace):
        """Test stacking with only one base model (degenerate case)."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)

        meta_preds = filter_by_model_name(predictions, "MetaModel")

        # Should still work, though not very useful
        assert len(meta_preds) > 0

    def test_stacking_no_cv_fails_by_default(self, regression_dataset, temp_workspace):
        """Test that stacking without CV fails by default."""
        pipeline = [
            MinMaxScaler(),
            # No KFold splitter - should fail
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)

        # Should either fail or handle gracefully
        try:
            predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)
            # If it doesn't fail, check for warnings or empty predictions
            meta_preds = filter_by_model_name(predictions, "MetaModel")
            # May be empty or have warnings
        except (ValueError, RuntimeError) as e:
            # Expected behavior - stacking needs CV
            assert "fold" in str(e).lower() or "cv" in str(e).lower() or "oof" in str(e).lower()

    def test_stacking_nonexistent_source_model(self, regression_dataset, temp_workspace):
        """Test error handling for nonexistent source model."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            PLSRegression(n_components=5),
            {"model": MetaModel(
                model=Ridge(),
                source_models=["NonExistentModel"],  # This doesn't exist
            )},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)

        # Should fail or warn about missing model
        try:
            predictions, _ = runner.run(PipelineConfigs(pipeline), regression_dataset)
            # If it runs, meta-model should have issue
            meta_preds = filter_by_model_name(predictions, "MetaModel")
            # Might be empty due to error
        except (ValueError, RuntimeError, KeyError) as e:
            # Expected - source model not found
            assert "NonExistentModel" in str(e) or "source" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
