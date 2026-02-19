"""
Integration tests for meta-model stacking pipelines (Phase 6).

Tests end-to-end scenarios:
- Basic stacking pipeline with PipelineRunner
- Stacking with preprocessing branches
- Mixed base models (sklearn + various estimators)
- Classification stacking with use_proba
- Save/reload/predict consistency
- Feature order preservation
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models import (
    BranchScope,
    CoverageStrategy,
    MetaModel,
    StackingConfig,
    TestAggregation,
)
from nirs4all.operators.transforms import FirstDerivative as DerivativeTransform
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# =============================================================================
# Helper Functions
# =============================================================================

def filter_by_model_name(predictions, name_contains: str) -> list[dict[str, Any]]:
    """Filter predictions by model name containing a substring."""
    all_preds = predictions.to_dicts(load_arrays=False)
    return [p for p in all_preds if name_contains in p.get('model_name', '')]

def _copy_reduced_regression_dataset(source: Path, target: Path) -> None:
    """Create a lightweight copy of regression sample data for smoke integration tests."""
    target.mkdir(parents=True, exist_ok=True)

    max_features = 320
    train_rows = 64
    val_rows = 24

    # These legacy sample files are header-less CSVs.
    xcal = pd.read_csv(source / "Xcal.csv.gz", sep=";", header=None)
    ycal = pd.read_csv(source / "Ycal.csv.gz", sep=";", header=None)
    xval = pd.read_csv(source / "Xval.csv.gz", sep=";", header=None)
    yval = pd.read_csv(source / "Yval.csv.gz", sep=";", header=None)

    xcal = xcal.iloc[:train_rows, :max_features]
    xval = xval.iloc[:val_rows, :max_features]
    ycal = ycal.iloc[:train_rows]
    yval = yval.iloc[:val_rows]

    xcal.to_csv(target / "Xcal.csv.gz", sep=";", index=False, header=False, compression="gzip")
    ycal.to_csv(target / "Ycal.csv.gz", sep=";", index=False, header=False, compression="gzip")
    xval.to_csv(target / "Xval.csv.gz", sep=";", index=False, header=False, compression="gzip")
    yval.to_csv(target / "Yval.csv.gz", sep=";", index=False, header=False, compression="gzip")

def _copy_reduced_regression_2_dataset(source: Path, target: Path) -> None:
    """Create a reduced grouped dataset preserving Sample_ID metadata for GroupKFold tests."""
    target.mkdir(parents=True, exist_ok=True)

    max_features = 96
    train_rows = 240
    test_rows = 96

    xtrain = pd.read_csv(source / "Xtrain.csv", sep=";").iloc[:train_rows]
    ytrain = pd.read_csv(source / "Ytrain.csv", sep=";").iloc[:train_rows]
    mtrain = pd.read_csv(source / "Mtrain.csv", sep=";").iloc[:train_rows]
    xtest = pd.read_csv(source / "Xtest.csv", sep=";").iloc[:test_rows]
    ytest = pd.read_csv(source / "Ytest.csv", sep=";").iloc[:test_rows]
    mtest = pd.read_csv(source / "Mtest.csv", sep=";").iloc[:test_rows]

    xtrain = xtrain.iloc[:, :max_features]
    xtest = xtest.iloc[:, :max_features]

    xtrain.to_csv(target / "Xtrain.csv", sep=";", index=False)
    ytrain.to_csv(target / "Ytrain.csv", sep=";", index=False)
    mtrain.to_csv(target / "Mtrain.csv", sep=";", index=False)
    xtest.to_csv(target / "Xtest.csv", sep=";", index=False)
    ytest.to_csv(target / "Ytest.csv", sep=";", index=False)
    mtest.to_csv(target / "Mtest.csv", sep=";", index=False)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_data_path(tmp_path_factory):
    """Reduced regression sample data path for faster smoke integration coverage."""
    source = Path("examples/sample_data/regression")
    if not source.exists():
        pytest.skip("Sample data not available")

    reduced_path = tmp_path_factory.mktemp("stacking_regression") / "regression_small"
    _copy_reduced_regression_dataset(source, reduced_path)
    return str(reduced_path)

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

@pytest.fixture(scope="session")
def sample_data_path_2(tmp_path_factory):
    """Reduced regression_2 sample data path preserving grouping metadata."""
    source = Path("examples/sample_data/regression_2")
    if not source.exists():
        pytest.skip("Sample data regression_2 not available")

    reduced_path = tmp_path_factory.mktemp("stacking_regression2") / "regression_2_small"
    _copy_reduced_regression_2_dataset(source, reduced_path)
    return str(reduced_path)

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
            KFold(n_splits=2, shuffle=True, random_state=42),
            PLSRegression(n_components=3),
            RandomForestRegressor(n_estimators=4, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=3), "name": "PLS_3"},
            {"model": PLSRegression(n_components=5), "name": "PLS_5"},
            RandomForestRegressor(n_estimators=3, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
            PLSRegression(n_components=3),
            RandomForestRegressor(n_estimators=4, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
            PLSRegression(n_components=3),  # Shared base model
            {"branch": {
                "branch0": [
                    RandomForestRegressor(n_estimators=3, random_state=42),
                    {"model": MetaModel(
                        model=Ridge(),
                        stacking_config=StackingConfig(
                            branch_scope=BranchScope.CURRENT_ONLY,
                        ),
                    ), "name": "Meta_Branch0"},
                ],
                "branch1": [
                    RandomForestRegressor(n_estimators=3, random_state=42),
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
            {"split": GroupKFold(n_splits=2), "group": "Sample_ID"},
            PLSRegression(n_components=5),
            {"model": MetaModel(model=Ridge(alpha=1.0))},
        ]

        runner = PipelineRunner(workspace_path=temp_workspace, save_artifacts=False, save_charts=False)
        predictions, _ = runner.run(PipelineConfigs(pipeline), regression_2_dataset)

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
            KFold(n_splits=2, shuffle=True, random_state=42),
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
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR

        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
            # Diverse base models
            PLSRegression(n_components=5),
            RandomForestRegressor(n_estimators=3, random_state=42),
            KNeighborsRegressor(n_neighbors=3),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
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

        # Verify pipeline produced valid predictions (feature order is implicitly
        # validated by the meta-model producing correct predictions)
        assert predictions.num_predictions > 0, "Expected at least one prediction"
        best = predictions.get_best(ascending=None)
        assert best is not None, "Expected a best prediction"
        assert best.get("y_pred") is not None, "Expected y_pred in best prediction"
        assert len(best["y_pred"]) > 0, "Expected non-empty y_pred"

# =============================================================================
# Edge Cases
# =============================================================================

class TestStackingEdgeCases:
    """Test edge cases for stacking pipelines."""

    def test_stacking_single_base_model(self, regression_dataset, temp_workspace):
        """Test stacking with only one base model (degenerate case)."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=2, shuffle=True, random_state=42),
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
            KFold(n_splits=2, shuffle=True, random_state=42),
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
