"""
Integration tests for MergeController Phase 6: Mixed Merge + Asymmetric Handling.

Tests:
- Mixed features + predictions merge (features from some branches, predictions from others)
- Asymmetric branch handling (models in some branches, not others)
- Different feature dimensions per branch
- Different model counts per branch
- Improved error messages with resolution suggestions

These tests verify the Phase 6 implementation from the branching_concat_merge_design.
"""

import pytest
import numpy as np
import warnings
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
    FirstDerivative,
)

# Mark all tests in this module as sklearn-only (no deep learning dependencies)
pytestmark = pytest.mark.sklearn


def _make_runner(tmp_path) -> PipelineRunner:
    """Create an xdist-safe PipelineRunner with isolated workspace."""
    return PipelineRunner(
        workspace_path=tmp_path / "workspace",
        verbose=0,
        save_artifacts=False,
        log_file=False,
    )


def create_test_dataset(n_samples: int = 100, n_features: int = 50, seed: int = 42) -> SpectroDataset:
    """Create a synthetic dataset for integration testing.

    Uses a local RNG (np.random.default_rng) instead of np.random.seed to
    avoid global random state pollution across parallel xdist workers.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = np.sum(X[:, :5], axis=1) + rng.standard_normal(n_samples) * 0.1

    # Split into train/test (80/20)
    n_train = int(n_samples * 0.8)

    dataset = SpectroDataset(name="test_merge_mixed")
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_targets(y[:n_train])
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[n_train:])

    return dataset


class TestMixedFeaturesPredictions:
    """Test mixed merge: features from some branches, predictions from others."""

    def test_mixed_predictions_branch0_features_branch1(self, tmp_path):
        """Test merging predictions from branch 0, features from branch 1."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],  # Branch 0: has model
                [MultiplicativeScatterCorrection(), StandardScaler()],                  # Branch 1: features only
            ]},
            {"merge": {
                "predictions": [0],  # OOF predictions from branch 0
                "features": [1]      # Features from branch 1
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_pred0_feat1"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_mixed_features_branch0_predictions_branch1(self, tmp_path):
        """Test merging features from branch 0, predictions from branch 1."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), PCA(n_components=10)],                        # Branch 0: features only
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=5)}],  # Branch 1: has model
            ]},
            {"merge": {
                "features": [0],      # Features from branch 0
                "predictions": [1]    # OOF predictions from branch 1
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_feat0_pred1"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_mixed_three_branches(self, tmp_path):
        """Test mixed merge with 3 branches: predictions from 0, features from 1 and 2."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],   # Branch 0: model
                [MultiplicativeScatterCorrection(), PCA(n_components=10)],             # Branch 1: features
                [SavitzkyGolay(), StandardScaler()],                                    # Branch 2: features
            ]},
            {"merge": {
                "predictions": [0],
                "features": [1, 2]
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_3branch"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_mixed_multiple_predictions_single_features(self, tmp_path):
        """Test mixed merge with predictions from multiple branches, features from one."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],   # Branch 0: model
                [MultiplicativeScatterCorrection(), {"model": Ridge(alpha=0.5)}],       # Branch 1: model
                [SavitzkyGolay(), PCA(n_components=20)],                               # Branch 2: features
            ]},
            {"merge": {
                "predictions": [0, 1],  # Predictions from branches 0 and 1
                "features": [2]         # Features from branch 2
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_multi_pred"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestMixedWithPerBranchConfig:
    """Test mixed merge combined with per-branch prediction configuration."""

    def test_mixed_with_best_selection(self, tmp_path):
        """Test mixed merge with 'best' model selection for predictions."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],  # Branch 0: 3 models
                [MultiplicativeScatterCorrection(), PCA(n_components=15)],  # Branch 1: features
            ]},
            {"merge": {
                "predictions": [{"branch": 0, "select": "best", "metric": "rmse"}],
                "features": [1]
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_best_selection"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_mixed_with_aggregation(self, tmp_path):
        """Test mixed merge with mean aggregation for predictions."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                ],  # Branch 0: 2 models
                [MultiplicativeScatterCorrection(), StandardScaler()],  # Branch 1: features
            ]},
            {"merge": {
                "predictions": [{"branch": 0, "select": "all", "aggregate": "mean"}],
                "features": [1]
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_aggregation"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestAsymmetricBranches:
    """Test asymmetric branch scenarios (models in some, not others)."""

    def test_asymmetric_error_without_mixed_merge(self, tmp_path):
        """Test that requesting predictions from branch without models raises informative error."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],  # Branch 0: has model
                [MultiplicativeScatterCorrection(), StandardScaler()],                  # Branch 1: NO model
            ]},
            {"merge": "predictions"},  # Will fail - branch 1 has no model
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)

        # Should raise an error about missing predictions with resolution suggestion
        with pytest.raises(RuntimeError) as exc_info:
            runner.run(
                PipelineConfigs(pipeline, "test_asymmetric_error"),
                dataset
            )

        # Check that error message contains helpful information
        error_msg = str(exc_info.value)
        assert "MERGE-E010" in error_msg or "MERGE-E011" in error_msg

    def test_asymmetric_resolved_with_mixed_merge(self, tmp_path):
        """Test that asymmetric branches work correctly with mixed merge."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],  # Branch 0: has model
                [MultiplicativeScatterCorrection(), StandardScaler()],                  # Branch 1: NO model
            ]},
            {"merge": {
                "predictions": [0],  # Only request predictions from branch 0
                "features": [1]      # Get features from branch 1
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_asymmetric_resolved"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestDifferentFeatureDimensions:
    """Test branches with different output feature dimensions.

    In 2D layout (default), features are flattened and concatenated horizontally.
    Different feature dimensions across branches is expected and normal - each
    branch can have different preprocessing (e.g., different PCA components).
    No on_shape_mismatch configuration is needed.
    """

    def test_different_pca_dimensions(self, tmp_path):
        """Test merging features from branches with different PCA dimensions.

        Branch 0 produces 10 features, Branch 1 produces 30 features.
        Final merged output has 40 features (10 + 30).
        """
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), PCA(n_components=10)],   # Branch 0: 10 features
                [MultiplicativeScatterCorrection(), PCA(n_components=30)],  # Branch 1: 30 features
            ]},
            {"merge": "features"},  # Total: 40 features
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_diff_pca_dims"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_different_dims_without_shape_mismatch_config(self, tmp_path):
        """Test that different dimensions work without any on_shape_mismatch config.

        In 2D layout (default), on_shape_mismatch has no effect - features are
        simply concatenated. This test verifies that explicit config is not needed.
        """
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {"branch": [
                [StandardNormalVariate(), PCA(n_components=5)],    # Branch 0: 5 features
                [MultiplicativeScatterCorrection(), PCA(n_components=25)],  # Branch 1: 25 features
            ]},
            {"merge": "features"},  # No on_shape_mismatch needed for 2D concat
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_allow_shape_mismatch"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestDifferentModelCounts:
    """Test branches with different numbers of models."""

    def test_asymmetric_model_counts(self, tmp_path):
        """Test merging predictions from branches with different model counts."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],  # Branch 0: 3 models
                [
                    MultiplicativeScatterCorrection(),
                    {"model": Ridge(alpha=1.0)},
                ],  # Branch 1: 1 model
            ]},
            {"merge": "predictions"},  # Total: 4 prediction features
            {"model": LinearRegression()}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_asymmetric_model_counts"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_asymmetric_model_counts_with_aggregation(self, tmp_path):
        """Test different aggregation strategies for branches with different model counts."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=3)},
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=7)},
                ],  # Branch 0: 3 models
                [
                    MultiplicativeScatterCorrection(),
                    {"model": Ridge(alpha=1.0)},
                ],  # Branch 1: 1 model
            ]},
            {"merge": {
                "predictions": [
                    {"branch": 0, "aggregate": "mean"},      # Mean of 3 → 1 feature
                    {"branch": 1, "aggregate": "separate"},  # Keep as 1 feature
                ]
            }},  # Total: 2 features
            {"model": LinearRegression()}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_asymmetric_counts_agg"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestMixedWithIncludeOriginal:
    """Test mixed merge with include_original flag."""

    def test_mixed_with_original_features(self, tmp_path):
        """Test mixed merge including original pre-branch features."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), PCA(n_components=10)],
            ]},
            {"merge": {
                "predictions": [0],
                "features": [1],
                "include_original": True
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_with_original"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestComplexAsymmetricScenarios:
    """Test complex real-world asymmetric scenarios."""

    def test_spectral_plus_features_pipeline(self, tmp_path):
        """Test scenario: spectral models + feature extraction.

        Scenario:
        - Branch 0: Spectral preprocessing → PLS models for prediction
        - Branch 1: Feature extraction → provide features for stacking
        """
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                # Spectral branch: preprocessing + models
                [
                    StandardNormalVariate(),
                    FirstDerivative(delta=1.0, edge_order=2),
                    {"model": PLSRegression(n_components=5)},
                    {"model": PLSRegression(n_components=10)},
                ],
                # Feature extraction branch: no models
                [
                    MultiplicativeScatterCorrection(),
                    PCA(n_components=15),
                ],
            ]},
            {"merge": {
                "predictions": [{"branch": 0, "select": "best", "metric": "rmse"}],
                "features": [1]
            }},
            {"model": Ridge(alpha=0.5)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_spectral_plus_features"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_ensemble_path_vs_feature_path(self, tmp_path):
        """Test scenario: ensemble models vs feature engineering.

        Scenario:
        - Branch 0: Multiple different models for ensemble
        - Branch 1: Multiple different scalers for feature engineering (no models)
        """
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                # Ensemble branch: multiple diverse models
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                    {"model": Ridge(alpha=0.1)},
                ],
                # Feature engineering branch: no models
                [
                    MultiplicativeScatterCorrection(),
                    MinMaxScaler(),
                ],
            ]},
            {"merge": {
                "predictions": [{"branch": 0, "aggregate": "mean"}],  # Average ensemble
                "features": [1]  # Use engineered features
            }},
            {"model": LinearRegression()}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_ensemble_vs_features"),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0


class TestMixedMergeOnMissing:
    """Test on_missing handling with mixed merge."""

    def test_mixed_on_missing_skip(self, tmp_path):
        """Test that on_missing='skip' works with mixed merge."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), StandardScaler()],
            ]},
            {"merge": {
                "predictions": [0, 1],  # Branch 1 has no models - should skip
                "features": [],
                "on_missing": "skip"
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = _make_runner(tmp_path)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_mixed_on_missing_skip"),
            dataset
        )

        # Should complete (branch 1 predictions skipped)
        assert predictions is not None

    def test_mixed_on_missing_warn(self, tmp_path):
        """Test that on_missing='warn' logs warnings with mixed merge."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), StandardScaler()],
            ]},
            {"merge": {
                "predictions": [0, 1],  # Branch 1 has no models - should warn
                "on_missing": "warn"
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(
            workspace_path=tmp_path / "workspace",
            verbose=1,
            save_artifacts=False,
            log_file=False,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            predictions, per_dataset = runner.run(
                PipelineConfigs(pipeline, "test_mixed_on_missing_warn"),
                dataset
            )

        # Should complete with warning
        assert predictions is not None
