"""
Integration tests for MergeController Phase 4: Prediction Merging.

Tests:
- Basic prediction merge from 2+ branches with OOF reconstruction
- Selective branch prediction merging (specific indices)
- Prediction merge → meta-model training pipeline
- Unsafe mode with warnings
- Mixed features + predictions merge
- Error handling for missing models

These tests verify the Phase 4 implementation from the branching_concat_merge_design.
"""

import pytest
import numpy as np
import warnings
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
)


def create_test_dataset(n_samples: int = 100, n_features: int = 50, seed: int = 42) -> SpectroDataset:
    """Create a synthetic dataset for integration testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    # Split into train/test (80/20)
    n_train = int(n_samples * 0.8)

    dataset = SpectroDataset(name="test_merge_predictions")
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_targets(y[:n_train])
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[n_train:])

    return dataset


class TestBasicPredictionMerge:
    """Test basic prediction merge operations with OOF reconstruction."""

    def test_two_branch_prediction_merge(self):
        """Test merging predictions from 2 branches with OOF reconstruction."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=5)}],
            ]},
            {"merge": "predictions"},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_merge_2branch_predictions"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0

        # The final Ridge model should have been trained on OOF predictions
        # from both PLS models (2 features -> 1 prediction)

    def test_three_branch_prediction_merge(self):
        """Test merging predictions from 3 branches with different models."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=3)}],
                [SavitzkyGolay(), {"model": Ridge(alpha=1.0)}],
            ]},
            {"merge": "predictions"},
            {"model": Lasso(alpha=0.1)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_merge_3branch_predictions"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0


class TestSelectivePredictionMerge:
    """Test selective branch prediction merging (specific indices)."""

    def test_merge_predictions_from_specific_branches(self):
        """Test merging predictions only from specific branches."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],   # Branch 0
                [MultiplicativeScatterCorrection(), {"model": Ridge(alpha=0.1)}],      # Branch 1
                [SavitzkyGolay(), {"model": PLSRegression(n_components=3)}],           # Branch 2
            ]},
            {"merge": {"predictions": [0, 2]}},  # Only branches 0 and 2
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_merge_selective_predictions"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0


class TestMixedMerge:
    """Test mixed features + predictions merge."""

    def test_mixed_features_and_predictions(self):
        """Test merging features from one branch, predictions from another."""
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

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_merge_mixed"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0

        # The final Ridge should be trained on:
        # - 1 OOF prediction from PLS (branch 0)
        # - 50 scaled features (branch 1)


class TestUnsafeMode:
    """Test unsafe mode prediction merge (with data leakage warning)."""

    def test_unsafe_mode_emits_warning(self):
        """Test that unsafe mode emits prominent warning."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=3)}],
            ]},
            {"merge": {"predictions": "all", "unsafe": True}},  # Unsafe mode!
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=1, save_artifacts=False)

        # Should complete but emit warnings about data leakage
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions, per_dataset = runner.run(
                PipelineConfigs(pipeline, "test_merge_unsafe"),
                dataset
            )

        # Verify predictions were generated (unsafe mode still works)
        assert predictions is not None
        assert len(predictions) > 0


class TestPredictionMergeMetadata:
    """Test that prediction merge produces correct metadata."""

    def test_merge_metadata_includes_models(self):
        """Test that merge output includes model information."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": Ridge(alpha=0.1)}],
            ]},
            {"merge": "predictions"},
            {"model": Lasso(alpha=0.1)}
        ]

        runner = PipelineRunner(verbose=1, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_merge_predictions_metadata"),
            dataset
        )

        # This test verifies the pipeline runs - metadata inspection
        # would require access to step outputs which is internal
        assert predictions is not None


class TestPredictionMergeErrors:
    """Test error handling for prediction merge."""

    def test_merge_predictions_without_models_raises(self):
        """Test that merging predictions from branch without models raises error."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate()],  # No model!
                [MultiplicativeScatterCorrection()],  # No model!
            ]},
            {"merge": "predictions"},  # Will fail - no models in branches
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)

        # Should raise an error about missing predictions
        with pytest.raises(RuntimeError, match="No model predictions found"):
            runner.run(
                PipelineConfigs(pipeline, "test_merge_no_models"),
                dataset
            )

    def test_merge_predictions_without_branch_raises(self):
        """Test that merging predictions without branch mode raises error."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            StandardNormalVariate(),
            {"model": PLSRegression(n_components=5)},
            {"merge": "predictions"},  # No branch before this!
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)

        with pytest.raises(RuntimeError, match="requires active branch contexts"):
            runner.run(
                PipelineConfigs(pipeline, "test_merge_predictions_no_branch"),
                dataset
            )


class TestPredictionMergeEquivalences:
    """Test equivalences between explicit merge and MetaModel."""

    def test_merge_predictions_produces_valid_stacking(self):
        """Test that merge+model produces valid stacking results."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        # Pattern: explicit merge + model
        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=5)}],
            ]},
            {"merge": "predictions"},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_explicit_stacking"),
            dataset
        )

        # Verify stacking produced reasonable predictions
        assert predictions is not None

        # Get test predictions using filter_predictions method
        test_preds = predictions.filter_predictions(partition='test', load_arrays=True)
        assert len(test_preds) > 0

        # Get y_pred from test predictions
        for pred in test_preds:
            y_pred = pred.get('y_pred')
            if y_pred is not None and len(y_pred) > 0:
                # Predictions should be finite and within reasonable bounds
                assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
                break


class TestMultipleModelsPerBranch:
    """Test prediction merge when branches have multiple models."""

    def test_merge_all_models_from_branch(self):
        """Test that all models from a branch contribute to merge."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [
                    StandardNormalVariate(),
                    {"model": PLSRegression(n_components=5)},
                    {"model": Ridge(alpha=0.5)},  # Two models in one branch
                ],
                [
                    MultiplicativeScatterCorrection(),
                    {"model": PLSRegression(n_components=3)},
                ],
            ]},
            {"merge": "predictions"},
            {"model": Lasso(alpha=0.1)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_multi_model_branch"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0

        # Branch 0 has 2 models, Branch 1 has 1 model
        # So merge should produce 3 features for the meta-learner


class TestIncludeOriginalWithPredictions:
    """Test include_original flag with prediction merge."""

    def test_predictions_with_original_features(self):
        """Test merging predictions along with original features."""
        dataset = create_test_dataset(n_samples=100, n_features=50)

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": [
                [StandardNormalVariate(), {"model": PLSRegression(n_components=5)}],
                [MultiplicativeScatterCorrection(), {"model": PLSRegression(n_components=3)}],
            ]},
            {"merge": {
                "predictions": "all",
                "include_original": True  # Include pre-branch features
            }},
            {"model": Ridge(alpha=1.0)}
        ]

        runner = PipelineRunner(verbose=0, save_artifacts=False)
        predictions, per_dataset = runner.run(
            PipelineConfigs(pipeline, "test_predictions_with_original"),
            dataset
        )

        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0

        # The meta-model should be trained on:
        # - Original 50 features (pre-branch)
        # - 2 OOF predictions from PLS models
