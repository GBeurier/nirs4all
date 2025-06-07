#!/usr/bin/env python3
"""
Phase 7 Smoke Test - Integration & Summary

Tests the complete pipeline and summary functionality.
Creates a comprehensive dataset and displays a full summary.
"""

import numpy as np
import polars as pl
from nirs4all.dataset import SpectroDataset
from sklearn.model_selection import KFold

def test_integration_summary():
    """Test complete integration and summary functionality."""
    print("=== Phase 7 Smoke Test - Integration & Summary ===")
    print()

    try:
        # Create a comprehensive dataset
        ds = SpectroDataset()

        # Add multiple feature sources
        features1 = np.random.rand(10, 5).astype("float32")  # NIR spectra
        features2 = np.random.rand(10, 3).astype("float32")  # Chemical properties
        ds.add_features([features1, features2])

        # Add targets with multiple processing versions
        raw_targets = pl.DataFrame({
            "sample": list(range(10)),
            "targets": [[float(i)] for i in range(10)],
            "processing": ["raw"] * 10
        })
        ds.add_targets(raw_targets)

        # Add processed targets
        processed_vals = np.array([[float(i * 2)] for i in range(5)], dtype="float32")
        idx = ds.targets.table.filter(pl.col("sample") < 5)
        ds.update_y(processed_vals, idx, processing_id="normalized")

        # Add metadata
        metadata = pl.DataFrame({
            "sample": list(range(10)),
            "instrument": ["NIR1"] * 5 + ["NIR2"] * 5,
            "operator": ["Alice"] * 3 + ["Bob"] * 4 + ["Charlie"] * 3,
            "batch": ["A"] * 4 + ["B"] * 6,
            "date": ["2023-01-01"] * 10
        })
        ds.add_meta(metadata)

        # Set up cross-validation folds
        kf = KFold(n_splits=3, shuffle=False)
        ds.folds.set_folds(kf.split(np.arange(10)))

        # Add predictions from multiple models
        preds1 = np.array([[0.1 * i] for i in range(5)], dtype="float32")
        meta1 = {
            "model": "linear_regression",
            "fold": 0,
            "repeat": 0,
            "partition": "val",
            "processing": "normalized",
            "seed": 42
        }
        ds.predictions.add_prediction(preds1, meta1)

        preds2 = np.array([[0.2 * i] for i in range(5)], dtype="float32")
        meta2 = {
            "model": "random_forest",
            "fold": 0,
            "repeat": 0,
            "partition": "val",
            "processing": "normalized",
            "seed": 42
        }
        ds.predictions.add_prediction(preds2, meta2)

        preds3 = np.array([[0.15 * i] for i in range(8)], dtype="float32")
        meta3 = {
            "model": "linear_regression",
            "fold": 1,
            "repeat": 0,
            "partition": "train",
            "processing": "normalized",
            "seed": 42
        }
        ds.predictions.add_prediction(preds3, meta3)

        print("✓ Created comprehensive dataset with:")
        print("  - Multiple feature sources")
        print("  - Multiple target processing versions")
        print("  - Rich metadata")
        print("  - Cross-validation folds")
        print("  - Multiple model predictions")
        print()
          # Test basic functionality
        x = ds.x({"sample": [0, 1, 2]}, layout="2d")  # Filter by sample indices
        y = ds.y({"processing": "normalized"})
        meta = ds.meta({"batch": "A"})
        preds = ds.predictions.prediction({"model": "linear_regression"})

        print(f"✓ Feature filtering: {len(x)} sources, first source shape: {x[0].shape}")
        print(f"✓ Target filtering: {y.shape}")
        print(f"✓ Metadata filtering: {meta.shape}")
        print(f"✓ Prediction filtering: {preds.shape}")
        print()

        # Test fold generation
        fold_count = 0
        for x_tr, y_tr, x_val, y_val in ds.folds.get_data(ds, layout="2d"):
            fold_count += 1
            if fold_count == 1:  # Just show first fold
                print(f"✓ Fold data generation: x_train={x_tr[0].shape}, y_train={y_tr.shape}")
                break

        print()
        print("=== COMPREHENSIVE DATASET SUMMARY ===")
        print()

        # Display the full summary
        ds.print_summary()

        print()
        print("🎉 Phase 7 - PASSED")
        print("Complete integration and summary working correctly!")

    except Exception as e:
        print(f"✗ Phase 7 test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("❌ Phase 7 - FAILED")
        return False

    return True

if __name__ == "__main__":
    test_integration_summary()
