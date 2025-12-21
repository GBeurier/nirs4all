"""Tests for SpectroDataset merged feature operations.

This module tests:
- add_merged_features: Adding merged features from branch operations
- get_merged_features: Retrieving merged features by processing name

These methods support Phase 2 of the MergeController implementation.

Note: add_merged_features REPLACES all existing processings with the
merged output. This is intentional as merge operations exit branch mode
and create a new unified feature set for subsequent steps.
"""

import pytest
import numpy as np
from nirs4all.data.dataset import SpectroDataset


class TestAddMergedFeatures:
    """Test suite for add_merged_features method."""

    def test_add_merged_features_basic(self):
        """Test adding merged features to a dataset."""
        dataset = SpectroDataset("test")
        # Initial data
        initial_data = np.random.rand(10, 100)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Add merged features (replaces initial processing)
        merged_features = np.random.rand(10, 50)
        dataset.add_merged_features(merged_features, "merged_branch")

        # Verify processing was replaced
        processings = dataset.features_processings(0)
        assert processings == ["merged_branch"]

    def test_add_merged_features_custom_name(self):
        """Test adding merged features with custom processing name."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(5, 20)
        dataset.add_samples(initial_data, {"partition": "test"})

        merged = np.random.rand(5, 30)
        dataset.add_merged_features(merged, "snv_msc_combined")

        processings = dataset.features_processings(0)
        assert processings == ["snv_msc_combined"]

    def test_add_merged_features_wrong_dimensions(self):
        """Test that non-2D array raises ValueError."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(5, 20)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Try 3D array
        bad_features = np.random.rand(5, 2, 10)
        with pytest.raises(ValueError, match="must be 2D"):
            dataset.add_merged_features(bad_features)

    def test_add_merged_features_sample_mismatch(self):
        """Test that sample count mismatch raises ValueError."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(10, 20)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Wrong number of samples
        bad_features = np.random.rand(5, 30)  # 5 instead of 10
        with pytest.raises(ValueError, match="Sample count mismatch"):
            dataset.add_merged_features(bad_features)

    def test_add_merged_features_different_feature_count(self):
        """Test adding merged features with different feature count than original."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(8, 100)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Different feature count replaces original
        merged = np.random.rand(8, 50)
        dataset.add_merged_features(merged, "merged")

        processings = dataset.features_processings(0)
        assert processings == ["merged"]

        # Verify new feature count
        X = dataset.x({}, layout="2d")
        assert X.shape == (8, 50)

    def test_add_merged_features_replaces_all(self):
        """Test that merge replaces all existing processings."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(6, 40)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Add first merge
        merged1 = np.random.rand(6, 20)
        dataset.add_merged_features(merged1, "merge_step_1")

        # Verify only merge_step_1 exists
        processings = dataset.features_processings(0)
        assert processings == ["merge_step_1"]

        # Add second merge (replaces first)
        merged2 = np.random.rand(6, 30)
        dataset.add_merged_features(merged2, "merge_step_2")

        # Verify only merge_step_2 exists now
        processings = dataset.features_processings(0)
        assert processings == ["merge_step_2"]


class TestGetMergedFeatures:
    """Test suite for get_merged_features method."""

    def test_get_merged_features_basic(self):
        """Test retrieving merged features."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(10, 100)
        dataset.add_samples(initial_data, {"partition": "train"})

        merged = np.random.rand(10, 50)
        dataset.add_merged_features(merged, "my_merge")

        # Retrieve merged features
        retrieved = dataset.get_merged_features("my_merge")
        assert retrieved.shape == (10, 50)

    def test_get_merged_features_not_found(self):
        """Test that missing processing name raises ValueError."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(5, 20)
        dataset.add_samples(initial_data, {"partition": "train"})

        with pytest.raises(ValueError, match="not found"):
            dataset.get_merged_features("nonexistent")

    def test_get_merged_features_default_name(self):
        """Test retrieving with default 'merged' name."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(8, 30)
        dataset.add_samples(initial_data, {"partition": "train"})

        merged = np.random.rand(8, 25)
        dataset.add_merged_features(merged)  # Uses default name "merged"

        retrieved = dataset.get_merged_features()  # Uses default name
        assert retrieved.shape == (8, 25)

    def test_get_merged_features_with_selector(self):
        """Test retrieving merged features with sample filter."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(10, 50)
        dataset.add_samples(initial_data[:5], {"partition": "train"})
        dataset.add_samples(initial_data[5:], {"partition": "val"})

        merged = np.random.rand(10, 20)
        dataset.add_merged_features(merged, "merged")

        # Get only train samples
        retrieved = dataset.get_merged_features("merged", selector={"partition": "train"})
        assert retrieved.shape[0] == 5  # Only train samples

    def test_get_merged_features_values_preserved(self):
        """Test that feature values are correctly preserved."""
        dataset = SpectroDataset("test")
        initial_data = np.random.rand(6, 40)
        dataset.add_samples(initial_data, {"partition": "train"})

        # Use specific values to check preservation
        merged = np.arange(6 * 15).reshape(6, 15).astype(float)
        dataset.add_merged_features(merged, "test_merge")

        retrieved = dataset.get_merged_features("test_merge")
        np.testing.assert_array_almost_equal(retrieved, merged)


class TestMergedFeaturesIntegration:
    """Integration tests for merged features workflow."""

    def test_merge_workflow_simulation(self):
        """Simulate a typical merge workflow from branches."""
        dataset = SpectroDataset("test")

        # Original spectral data
        n_samples = 20
        n_wavelengths = 100
        original_data = np.random.rand(n_samples, n_wavelengths)
        dataset.add_samples(original_data, {"partition": "train"})

        # Simulate branch outputs (as would come from merge controller)
        branch_0_features = np.random.rand(n_samples, 50)  # SNV processed
        branch_1_features = np.random.rand(n_samples, 50)  # MSC processed

        # Merge: horizontal concatenation (as merge controller would do)
        merged = np.concatenate([branch_0_features, branch_1_features], axis=1)
        dataset.add_merged_features(merged, "merged_snv_msc")

        # Retrieve and verify
        X_merged = dataset.get_merged_features("merged_snv_msc")
        assert X_merged.shape == (n_samples, 100)  # 50 + 50

    def test_merge_with_predictions_simulation(self):
        """Simulate merge with prediction stacking scenario."""
        dataset = SpectroDataset("test")

        # Original data
        n_samples = 15
        original_data = np.random.rand(n_samples, 80)
        dataset.add_samples(original_data, {"partition": "train"})

        # Simulate OOF predictions from 3 models
        pls_preds = np.random.rand(n_samples, 1)
        rf_preds = np.random.rand(n_samples, 1)
        xgb_preds = np.random.rand(n_samples, 1)

        # Merge predictions
        stacked_preds = np.concatenate([pls_preds, rf_preds, xgb_preds], axis=1)
        dataset.add_merged_features(stacked_preds, "stacked_predictions")

        # Verify stacked predictions
        X_stacked = dataset.get_merged_features("stacked_predictions")
        assert X_stacked.shape == (n_samples, 3)

    def test_mixed_merge_simulation(self):
        """Simulate mixed merge: features from one branch, predictions from another."""
        dataset = SpectroDataset("test")

        n_samples = 12
        original_data = np.random.rand(n_samples, 60)
        dataset.add_samples(original_data, {"partition": "train"})

        # Branch 0: PLS predictions (1 feature)
        pls_oof_predictions = np.random.rand(n_samples, 1)

        # Branch 1: PCA features (30 features)
        pca_features = np.random.rand(n_samples, 30)

        # Mixed merge: [predictions | features]
        mixed = np.concatenate([pls_oof_predictions, pca_features], axis=1)
        dataset.add_merged_features(mixed, "mixed_preds_features")

        X_mixed = dataset.get_merged_features("mixed_preds_features")
        assert X_mixed.shape == (n_samples, 31)  # 1 + 30

    def test_merge_after_preprocessing_chain(self):
        """Test merge after multiple preprocessing steps."""
        dataset = SpectroDataset("test")

        n_samples = 10
        original_data = np.random.rand(n_samples, 50)
        dataset.add_samples(original_data, {"partition": "train"})

        # Simulate preprocessing in each branch
        # Branch 0: SNV -> SavGol -> PCA(20) -> 20 features
        branch_0_result = np.random.rand(n_samples, 20)
        # Branch 1: MSC -> Detrend -> PCA(30) -> 30 features
        branch_1_result = np.random.rand(n_samples, 30)

        # Merge
        merged = np.concatenate([branch_0_result, branch_1_result], axis=1)
        dataset.add_merged_features(merged, "full_pipeline_merge")

        X = dataset.get_merged_features("full_pipeline_merge")
        assert X.shape == (n_samples, 50)  # 20 + 30
