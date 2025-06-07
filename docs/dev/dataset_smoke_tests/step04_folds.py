#!/usr/bin/env python3
"""
Phase 4 Smoke Test - Folds Manager

Tests the FoldsManager functionality including:
- Setting folds from sklearn KFold
- Getting data generators with train/val splits
- Integration with SpectroDataset
"""

import numpy as np
import polars as pl
from nirs4all.dataset import SpectroDataset
from sklearn.model_selection import KFold

def test_folds_manager():
    """Test FoldsManager functionality."""
    print("=== Phase 4 Smoke Test - Folds Manager ===")
    print()

    try:
        # Create dataset with features and targets
        ds = SpectroDataset()
        ds.add_features([np.random.rand(12, 2).astype("float32")])

        raw = pl.DataFrame({
            "sample": list(range(12)),
            "targets": [[i] for i in range(12)],
            "processing": ["raw"] * 12
        })
        ds.targets.add_targets(raw)

        print(f"✓ Created dataset with 12 samples, 2 features")

        # Set up KFold splits
        kf = KFold(n_splits=3, shuffle=False)
        ds.folds.set_folds(kf.split(np.arange(12)))

        print(f"✓ Set folds: {ds.folds}")

        # Test the data generator
        fold_count = 0
        for fold in ds.folds.get_data(ds, layout="2d"):
            x_tr, y_tr, x_val, y_val = fold
            print(f"✓ Fold {fold_count}: x_train={x_tr[0].shape}, x_val={x_val[0].shape}")
            print(f"  y_train={y_tr.shape}, y_val={y_val.shape}")
            fold_count += 1

        print(f"✓ Generated {fold_count} folds successfully")

        # Test different layouts
        for fold in ds.folds.get_data(ds, layout="3d"):
            x_tr, y_tr, x_val, y_val = fold
            print(f"✓ 3D layout: x_train={x_tr[0].shape}, x_val={x_val[0].shape}")
            break  # Just test first fold

        print()
        print("🎉 Phase 4 - PASSED")
        print("FoldsManager working correctly!")

    except Exception as e:
        print(f"✗ Phase 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("❌ Phase 4 - FAILED")
        return False

    return True

if __name__ == "__main__":
    test_folds_manager()
