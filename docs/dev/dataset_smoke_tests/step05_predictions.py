#!/usr/bin/env python3
"""
Phase 5 Smoke Test - Predictions Block

Tests the PredictionBlock functionality including:
- Adding predictions with metadata
- Filtering predictions
- Inverse transforming predictions
- Integration with SpectroDataset
"""

import numpy as np
import polars as pl
import os
from nirs4all.dataset import SpectroDataset

def test_predictions_block():
    """Test PredictionBlock functionality."""
    print("=== Phase 5 Smoke Test - Predictions Block ===")
    print()

    try:
        # Create dataset with features and targets
        ds = SpectroDataset()
        ds.add_features([np.random.rand(3, 4).astype("float32")])
        ds.targets.add_targets(pl.DataFrame({
            "sample": [0, 1, 2],
            "targets": [[0.], [1.], [0.]],
            "processing": ["raw"] * 3
        }))

        print("✓ Created dataset with 3 samples, 4 features")

        # Add predictions
        preds = np.array([[0.2], [0.8], [0.1]], dtype="float32")
        meta = {
            "model": "logreg",
            "fold": 0,
            "repeat": 0,
            "partition": "val",
            "processing": "regression",
            "seed": 42
        }
        ds.predictions.add_prediction(preds, meta)

        print(f"✓ Added predictions: {ds.predictions}")
        print(f"✓ Predictions table shape: {ds.predictions.table.shape}")
        print(f"✓ Predictions table columns: {ds.predictions.table.columns}")

        # Test filtering predictions
        filtered_preds = ds.predictions.prediction({"model": "logreg"})
        print(f"✓ Filtered predictions shape: {filtered_preds.shape}")

        partition_preds = ds.predictions.prediction({"partition": "val"})
        print(f"✓ Partition predictions shape: {partition_preds.shape}")

        # Test inverse transformation
        def scale_transform(arr):
            return arr * 2.0

        def shift_transform(arr):
            return arr + 0.1

        transformers = [scale_transform, shift_transform]
        transformed_preds = ds.predictions.inverse_transform_prediction(
            transformers, {"model": "logreg"}
        )
        print(f"✓ Transformed predictions shape: {transformed_preds.shape}")
        print(f"✓ Original first prediction: {filtered_preds[0][0]:.3f}")
        print(f"✓ Transformed first prediction: {transformed_preds[0][0]:.3f}")

        # Add another set of predictions
        preds2 = np.array([[0.3], [0.7], [0.2]], dtype="float32")
        meta2 = {
            "model": "svm",
            "fold": 1,
            "repeat": 0,
            "partition": "train",
            "processing": "regression",
            "seed": 42
        }
        ds.predictions.add_prediction(preds2, meta2)

        print(f"✓ Added second set of predictions: {ds.predictions}")

        # Test filtering by different criteria
        svm_preds = ds.predictions.prediction({"model": "svm"})
        train_preds = ds.predictions.prediction({"partition": "train"})

        print(f"✓ SVM predictions shape: {svm_preds.shape}")
        print(f"✓ Train predictions shape: {train_preds.shape}")

        # Test save/load persistence (Phase 6)
        print()
        print("=== Testing Phase 6 - Persistence ===")

        import tempfile
        import shutil
        import os

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "spectro_test")

            # Save dataset
            ds.save(save_path)
            print(f"✓ Saved dataset to: {save_path}")

            # Load dataset
            ds2 = ds.load(save_path)
            print(f"✓ Loaded dataset: {ds2}")

            # Compare features (zero-copy verification)
            x1, = ds.x({}, layout="2d")
            x2, = ds2.x({}, layout="2d")
            assert np.array_equal(x1, x2), "Features don't match after save/load"
            print("✓ Features match after save/load")

            # Compare targets
            y1 = ds.y({})
            y2 = ds2.y({})
            assert np.array_equal(y1, y2), "Targets don't match after save/load"
            print("✓ Targets match after save/load")

            # Compare predictions
            p1 = ds.predictions.prediction({})
            p2 = ds2.predictions.prediction({})
            assert np.array_equal(p1, p2), "Predictions don't match after save/load"
            print("✓ Predictions match after save/load")

            print("✓ Save/Load round-trip OK")

        print()
        print("🎉 Phase 5 - PASSED")
        print("PredictionBlock working correctly!")

    except Exception as e:
        print(f"✗ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("❌ Phase 5 - FAILED")
        return False

    return True

if __name__ == "__main__":
    test_predictions_block()
