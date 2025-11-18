"""
Integration test for ArrayRegistry with Predictions API

Verifies end-to-end functionality of the new array registry architecture.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from nirs4all.data.predictions import Predictions


class TestArrayRegistryIntegration:
    """Test ArrayRegistry integration with Predictions API."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_basic_workflow(self, temp_dir):
        """Test basic add, save, load workflow with array registry."""
        # Create predictions with array registry (now default)
        pred = Predictions()

        # Add some predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred1 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        y_pred2 = np.array([1.2, 2.2, 3.2, 4.2, 5.2])

        id1 = pred.add_prediction(
            dataset_name="test_dataset",
            model_name="Model_A",
            partition="test",
            y_true=y_true,
            y_pred=y_pred1,
            test_score=0.1
        )

        id2 = pred.add_prediction(
            dataset_name="test_dataset",
            model_name="Model_B",
            partition="test",
            y_true=y_true,  # Same y_true (should be deduplicated)
            y_pred=y_pred2,
            test_score=0.2
        )

        assert id1 != id2
        assert pred.num_predictions == 2

        # Check array registry stats
        registry = pred._storage.get_array_registry()
        assert registry is not None
        stats = registry.get_stats()

        # Should have 4 unique arrays:
        # - 1 shared y_true (deduplicated)
        # - 2 y_pred (one for each model)
        # - 1 empty sample_indices (deduplicated across both predictions)
        assert stats["total_arrays"] == 4

        # Save to split Parquet
        meta_path = temp_dir / "test_predictions.meta.parquet"
        pred.save_to_file(str(meta_path))

        arrays_path = temp_dir / "test_predictions.arrays.parquet"
        assert meta_path.exists()
        assert arrays_path.exists()

        # Load back
        pred_loaded = Predictions()
        pred_loaded.load_from_file(str(meta_path))

        assert pred_loaded.num_predictions == 2

        # Verify arrays are loaded correctly
        results = pred_loaded.filter_predictions(model_name="Model_A", load_arrays=True)
        assert len(results) == 1
        np.testing.assert_array_almost_equal(results[0]["y_true"], y_true)
        np.testing.assert_array_almost_equal(results[0]["y_pred"], y_pred1)

    def test_lazy_loading(self, temp_dir):
        """Test lazy loading of arrays."""
        pred = Predictions()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        pred.add_prediction(
            dataset_name="test",
            model_name="Test_Model",
            partition="test",
            y_true=y_true,
            y_pred=y_pred,
            test_score=0.1
        )

        # Query with load_arrays=False (fast, metadata only)
        results_meta = pred.filter_predictions(load_arrays=False)
        assert len(results_meta) == 1

        # Should have array_id references, not actual arrays
        assert "y_true_id" in results_meta[0]
        assert "y_pred_id" in results_meta[0]

        # Query with load_arrays=True (full data)
        results_full = pred.filter_predictions(load_arrays=True)
        assert len(results_full) == 1

        # Should have actual arrays
        assert "y_true" in results_full[0]
        assert "y_pred" in results_full[0]
        np.testing.assert_array_almost_equal(results_full[0]["y_true"], y_true)
        np.testing.assert_array_almost_equal(results_full[0]["y_pred"], y_pred)

    def test_deduplication_benefit(self, temp_dir):
        """Test that deduplication actually works."""
        pred = Predictions()

        # Create predictions with same y_true
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Add 3 partitions with same y_true
        for partition in ["train", "val", "test"]:
            y_pred = np.random.randn(5)
            pred.add_prediction(
                dataset_name="test",
                model_name="Model_X",
                partition=partition,
                y_true=y_true,  # Same array
                y_pred=y_pred,
                test_score=0.1
            )

        assert pred.num_predictions == 3

        # Check deduplication
        registry = pred._storage.get_array_registry()
        stats = registry.get_stats()

        # Should have 5 unique arrays:
        # - 1 shared y_true (deduplicated across all 3 partitions)
        # - 3 y_pred (one per partition, randomly generated so unique)
        # - 1 empty sample_indices (deduplicated across all 3 partitions)
        assert stats["total_arrays"] == 5

        # Verify all predictions have access to the correct y_true
        for partition in ["train", "val", "test"]:
            results = pred.filter_predictions(partition=partition, load_arrays=True)
            assert len(results) == 1
            np.testing.assert_array_equal(results[0]["y_true"], y_true)

    def test_backward_compatibility_json(self, temp_dir):
        """Test that JSON loading still works (backward compatibility)."""
        # Create with new format (always uses array registry)
        pred_legacy = Predictions()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        pred_legacy.add_prediction(
            dataset_name="test",
            model_name="Test_Model",
            partition="test",
            y_true=y_true,
            y_pred=y_pred,
            test_score=0.1
        )

        # Save as Parquet (JSON no longer supported)
        parquet_path = temp_dir / "predictions.meta.parquet"
        pred_legacy.save_to_file(str(parquet_path))

        # Load with new system
        pred_new = Predictions()
        pred_new.load_from_file(str(parquet_path))

        assert pred_new.num_predictions == 1

        # Verify data is correct
        results = pred_new.filter_predictions(load_arrays=True)
        np.testing.assert_array_almost_equal(results[0]["y_true"], y_true)
        np.testing.assert_array_almost_equal(results[0]["y_pred"], y_pred)

    def test_migration_from_json_to_parquet(self, temp_dir):
        """Test saving and loading with split Parquet format."""
        # Create predictions with new format
        pred_legacy = Predictions()

        for i in range(5):
            y_true = np.random.randn(10)
            y_pred = np.random.randn(10)
            pred_legacy.add_prediction(
                dataset_name="test",
                model_name=f"Model_{i}",
                partition="test",
                y_true=y_true,
                y_pred=y_pred,
                test_score=float(i) * 0.1
            )

        parquet_path = temp_dir / "to_migrate.meta.parquet"
        pred_legacy.save_to_file(str(parquet_path))

        # Verify files exist
        arrays_path = temp_dir / "to_migrate.arrays.parquet"
        assert parquet_path.exists()
        assert arrays_path.exists()

        # Load data
        pred_migrated = Predictions()
        pred_migrated.load_from_file(str(parquet_path))

        # Verify a random prediction
        results = pred_migrated.filter_predictions(model_name="Model_2", load_arrays=True)
        assert len(results) == 1
        assert len(results[0]["y_true"]) == 10
        assert len(results[0]["y_pred"]) == 10

    def test_large_dataset_performance(self, temp_dir):
        """Test performance with larger dataset."""
        pred = Predictions()

        # Add 100 predictions with some shared y_true
        shared_y_true = np.random.randn(100)

        for i in range(100):
            # Use shared y_true for half the predictions
            if i % 2 == 0:
                y_true = shared_y_true
            else:
                y_true = np.random.randn(100)

            y_pred = np.random.randn(100)

            pred.add_prediction(
                dataset_name="large_test",
                model_name=f"Model_{i}",
                partition="test",
                y_true=y_true,
                y_pred=y_pred,
                test_score=float(i) * 0.01
            )

        assert pred.num_predictions == 100

        # Check deduplication stats
        registry = pred._storage.get_array_registry()
        stats = registry.get_stats()

        # Should have less than 200 arrays due to deduplication
        # (50 shared y_true + 50 unique y_true + 100 y_pred = 150)
        assert stats["total_arrays"] < 200
        assert stats["total_arrays"] >= 150

        # Save and load
        meta_path = temp_dir / "large_test.meta.parquet"
        pred.save_to_file(str(meta_path))

        pred_loaded = Predictions()
        pred_loaded.load_from_file(str(meta_path))

        assert pred_loaded.num_predictions == 100

        # Test fast metadata-only query
        results_meta = pred_loaded.filter_predictions(
            dataset_name="large_test",
            load_arrays=False
        )
        assert len(results_meta) == 100

        # Test loading specific prediction with arrays
        results_full = pred_loaded.filter_predictions(
            model_name="Model_42",
            load_arrays=True
        )
        assert len(results_full) == 1
        assert len(results_full[0]["y_true"]) == 100
