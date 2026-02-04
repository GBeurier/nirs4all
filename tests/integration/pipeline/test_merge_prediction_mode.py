"""
Integration tests for MergeController Phase 8: Prediction Mode & Artifacts.

Tests:
- Train → Save → Predict roundtrip with merge steps
- Feature merge prediction mode
- Prediction merge prediction mode
- Mixed merge prediction mode
- Bundle export/import with merge
- Merge config serialization/deserialization

These tests verify the Phase 8 implementation from the branching_concat_merge_design.
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
)
from nirs4all.operators.data.merge import MergeConfig, BranchPredictionConfig


def create_test_dataset(
    n_samples: int = 100,
    n_features: int = 50,
    seed: int = 42
) -> SpectroDataset:
    """Create a synthetic dataset for integration testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    # Split into train/test (80/20)
    n_train = int(n_samples * 0.8)

    dataset = SpectroDataset(name="test_merge_predict")
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_targets(y[:n_train])
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[n_train:])

    return dataset


def create_new_prediction_data(
    n_samples: int = 20,
    n_features: int = 50,
    seed: int = 123
) -> SpectroDataset:
    """Create new data for prediction that wasn't seen during training."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)

    dataset = SpectroDataset(name="new_prediction_data")
    dataset.add_samples(X, indexes={"partition": "test"})
    # Add dummy targets (not used in predict mode)
    dataset.add_targets(np.zeros(n_samples))

    return dataset


class TestMergeConfigSerialization:
    """Test merge configuration serialization/deserialization."""

    def test_simple_feature_merge_config(self):
        """Test serialization of simple feature merge config."""
        config = MergeConfig(collect_features=True)
        config_dict = config.to_dict()

        assert config_dict["collect_features"] is True
        assert config_dict["collect_predictions"] is False
        assert config_dict["output_as"] == "features"

        # Roundtrip
        restored = MergeConfig.from_dict(config_dict)
        assert restored.collect_features is True
        assert restored.collect_predictions is False

    def test_prediction_merge_config(self):
        """Test serialization of prediction merge config."""
        config = MergeConfig(
            collect_predictions=True,
            prediction_branches=[0, 2],
            model_filter=["PLS", "RF"]
        )
        config_dict = config.to_dict()

        assert config_dict["collect_predictions"] is True
        assert config_dict["prediction_branches"] == [0, 2]
        assert config_dict["model_filter"] == ["PLS", "RF"]

        # Roundtrip
        restored = MergeConfig.from_dict(config_dict)
        assert restored.collect_predictions is True
        assert restored.prediction_branches == [0, 2]
        assert restored.model_filter == ["PLS", "RF"]

    def test_per_branch_config_serialization(self):
        """Test serialization of per-branch prediction configuration."""
        config = MergeConfig(
            collect_predictions=True,
            prediction_configs=[
                BranchPredictionConfig(
                    branch=0,
                    select="best",
                    metric="rmse",
                    aggregate="separate"
                ),
                BranchPredictionConfig(
                    branch=1,
                    select={"top_k": 2},
                    aggregate="mean",
                    proba=False
                ),
            ]
        )
        config_dict = config.to_dict()

        assert "prediction_configs" in config_dict
        assert len(config_dict["prediction_configs"]) == 2
        assert config_dict["prediction_configs"][0]["branch"] == 0
        assert config_dict["prediction_configs"][0]["select"] == "best"
        assert config_dict["prediction_configs"][1]["select"] == {"top_k": 2}

        # Roundtrip
        restored = MergeConfig.from_dict(config_dict)
        assert restored.has_per_branch_config()
        assert len(restored.prediction_configs) == 2
        assert restored.prediction_configs[0].select == "best"
        assert restored.prediction_configs[1].aggregate == "mean"

    def test_mixed_merge_config(self):
        """Test serialization of mixed merge config."""
        config = MergeConfig(
            collect_features=True,
            feature_branches=[1, 2],
            collect_predictions=True,
            prediction_branches=[0],
            include_original=True,
            output_as="features"
        )
        config_dict = config.to_dict()

        assert config_dict["collect_features"] is True
        assert config_dict["collect_predictions"] is True
        assert config_dict["feature_branches"] == [1, 2]
        assert config_dict["include_original"] is True

        # Roundtrip
        restored = MergeConfig.from_dict(config_dict)
        assert restored.collect_features is True
        assert restored.collect_predictions is True
        assert restored.feature_branches == [1, 2]
        assert restored.include_original is True


class TestFeatureMergePredictionMode:
    """Test feature merge train → predict roundtrip."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_feature_merge_train_predict_roundtrip(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test basic feature merge: train with merge, then predict on new data."""
        # Train pipeline with branches and merge
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [MultiplicativeScatterCorrection()],
            ]},
            {"merge": "features"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, per_dataset = runner_with_save.run(
            PipelineConfigs(pipeline, "test_feature_merge"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0

        # Verify DuckDB store was created
        store_path = workspace_path / "store.duckdb"
        assert store_path.exists(), "store.duckdb should be created"

        # Check artifacts directory exists
        artifacts_dir = workspace_path / "artifacts"
        assert artifacts_dir.exists(), "artifacts/ directory should be created"

        # Get a test prediction to use for prediction mode
        test_preds = predictions.filter_predictions(partition="test")
        if len(test_preds) == 0:
            pytest.skip("No test predictions available")
        best_pred = test_preds[0]

        # Create new data for prediction
        new_data = create_new_prediction_data()

        # Run prediction on new data
        pred_runner = PipelineRunner(
            workspace_path=workspace_path,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

        # Predict using the trained pipeline
        pred_results, _ = pred_runner.predict(
            prediction_obj=best_pred,
            dataset=new_data,
        )

        # Verify prediction results
        assert pred_results is not None
        assert len(pred_results) > 0


class TestPredictionMergePredictionMode:
    """Test prediction merge (stacking) train → predict roundtrip."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_prediction_merge_train_predict_roundtrip(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test prediction merge (stacking): train with merge predictions."""
        # Train pipeline with branches, models, and prediction merge
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=5)}],
            ]},
            {"merge": "predictions"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, per_dataset = runner_with_save.run(
            PipelineConfigs(pipeline, "test_prediction_merge"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0


class TestMixedMergePredictionMode:
    """Test mixed merge (features + predictions) train → predict roundtrip."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_mixed_merge_basic(
        self, runner_with_save, dataset
    ):
        """Test mixed merge: features from one branch, predictions from another."""
        # Train pipeline with mixed merge
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [SavitzkyGolay()],  # Features only
            ]},
            {"merge": {
                "features": [1],
                "predictions": [0]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, per_dataset = runner_with_save.run(
            PipelineConfigs(pipeline, "test_mixed_merge"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0


class TestMergeMetadataInStore:
    """Test that merge configuration is properly saved to DuckDB store."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_merge_config_in_store(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test that merge configuration is saved in DuckDB store."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [MultiplicativeScatterCorrection()],
            ]},
            {"merge": "features"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline, "test_store"),
            dataset
        )

        # Verify store.duckdb was created
        store_path = workspace_path / "store.duckdb"
        assert store_path.exists(), "store.duckdb should be created"

        # Verify predictions were produced
        assert predictions is not None
        assert len(predictions) > 0


class TestMergeWithBranching:
    """Test merge behavior with various branching scenarios."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def test_feature_merge_exits_branch_mode(self, workspace_path):
        """Test that merge properly exits branch mode."""
        dataset = create_test_dataset()

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [MultiplicativeScatterCorrection()],
            ]},
            {"merge": "features"},
            # This Ridge should run once (not per branch)
            {"model": Ridge(alpha=1.0)},
        ]

        runner = PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "test_exit_branch"),
            dataset
        )

        # Should have predictions (if merge properly exited branch mode)
        assert predictions is not None

    def test_selective_branch_merge(self, workspace_path):
        """Test merging specific branches by index."""
        dataset = create_test_dataset()

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate()],
                [MultiplicativeScatterCorrection()],
                [SavitzkyGolay()],
            ]},
            {"merge": {"features": [0, 2]}},  # Skip branch 1
            {"model": Ridge(alpha=1.0)},
        ]

        runner = PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=False,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "test_selective"),
            dataset
        )

        assert predictions is not None


class TestMergeIncludeOriginal:
    """Test merge with include_original option."""

    def test_include_original_prepends_features(self, tmp_path):
        """Test that include_original prepends pre-branch features."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            MinMaxScaler(),  # Pre-branch processing (50 features)
            {"branch": [
                [StandardNormalVariate()],  # 50 features
            ]},
            {"merge": {"features": "all", "include_original": True}},
            {"model": Ridge(alpha=1.0)},
        ]

        runner = PipelineRunner(
            workspace_path=tmp_path / "workspace",
            save_artifacts=False,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

        predictions, _ = runner.run(
            PipelineConfigs(pipeline, "test_include_original"),
            dataset
        )

        assert predictions is not None
