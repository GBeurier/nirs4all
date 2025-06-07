"""
Smoke Test Phase 3 - Targets & Metadata

This test verifies that the SpectroDataset can handle targets and metadata correctly.
"""

import numpy as np
import polars as pl
from nirs4all.dataset import SpectroDataset


def test_targets_basic():
    """Test basic target functionality."""
    try:
        ds = SpectroDataset()

        # Add some features first
        ds.add_features([np.zeros((5, 3), dtype=np.float32)])

        # Create target data
        raw_df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "targets": [[0.1], [0.2], [0.3], [0.4], [0.5]],
            "processing": ["raw"] * 5
        })

        # Add targets
        ds.add_targets(raw_df)

        print(f"✓ Added targets: {ds.targets}")

        # Test getting targets
        targets = ds.y({}, processed=True)
        print(f"✓ Retrieved targets shape: {targets.shape}")
        assert targets.shape == (5, 1), f"Expected (5, 1), got {targets.shape}"

        # Test filtering
        filtered_targets = ds.y({"processing": "raw"})
        print(f"✓ Filtered targets shape: {filtered_targets.shape}")
        assert filtered_targets.shape == (5, 1), "Raw targets should match all targets"

        return True

    except Exception as e:
        print(f"✗ Basic targets test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_update():
    """Test target update functionality."""
    try:
        ds = SpectroDataset()

        # Add features
        ds.add_features([np.zeros((5, 3), dtype=np.float32)])

        # Add initial targets
        raw_df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "targets": [[0.1], [0.2], [0.3], [0.4], [0.5]],
            "processing": ["raw"] * 5
        })
        ds.add_targets(raw_df)

        # Create a processed version for first 3 samples
        idx = ds.targets.table.filter(pl.col("sample") < 3)
        vals = np.array([[10.], [20.], [30.]], dtype=np.float32)
        ds.update_y(vals, idx, processing_id="regression")

        print(f"✓ Updated targets table: {len(ds.targets.table)} rows")

        # Test getting different processing versions
        raw_targets = ds.y({"processing": "raw"})
        regression_targets = ds.y({"processing": "regression"})

        print(f"✓ Raw targets: {raw_targets.shape}")
        print(f"✓ Regression targets: {regression_targets.shape}")

        assert len(raw_targets) == 5, "Should have 5 raw targets"
        assert len(regression_targets) == 3, "Should have 3 regression targets"

        return True

    except Exception as e:
        print(f"✗ Target update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata():
    """Test metadata functionality."""
    try:
        ds = SpectroDataset()

        # Create metadata
        meta_df = pl.DataFrame({
            "sample": [0, 1, 2, 3, 4],
            "instrument": ["NIR1", "NIR2", "NIR1", "NIR2", "NIR1"],
            "temperature": [25.0, 26.0, 24.0, 25.5, 26.5],
            "operator": ["Alice", "Bob", "Alice", "Charlie", "Bob"]
        })

        # Add metadata
        ds.add_meta(meta_df)

        print(f"✓ Added metadata: {ds.metadata}")

        # Test getting all metadata
        all_meta = ds.meta({})
        print(f"✓ All metadata shape: {all_meta.shape}")
        assert all_meta.shape == (5, 4), "Should have 5 rows and 4 columns"

        # Test filtering metadata
        nir1_meta = ds.meta({"instrument": "NIR1"})
        print(f"✓ NIR1 metadata shape: {nir1_meta.shape}")
        assert len(nir1_meta) == 3, "Should have 3 NIR1 samples"

        alice_meta = ds.meta({"operator": "Alice"})
        print(f"✓ Alice metadata shape: {alice_meta.shape}")
        assert len(alice_meta) == 2, "Should have 2 Alice samples"

        return True

    except Exception as e:
        print(f"✗ Metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated():
    """Test integrated functionality with features, targets, and metadata."""
    try:
        ds = SpectroDataset()

        # Add features
        features = np.random.rand(3, 10).astype(np.float32)
        ds.add_features([features])

        # Add targets
        targets_df = pl.DataFrame({
            "sample": [0, 1, 2],
            "targets": [[1.0], [2.0], [3.0]],
            "processing": ["raw"] * 3
        })
        ds.add_targets(targets_df)

        # Add metadata
        meta_df = pl.DataFrame({
            "sample": [0, 1, 2],
            "batch": ["A", "A", "B"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"]
        })
        ds.add_meta(meta_df)

        # Test integrated queries
        batch_a_features = ds.x({"partition": "train"})  # All should be train by default
        batch_a_targets = ds.y({"processing": "raw"})
        batch_a_meta = ds.meta({"batch": "A"})

        print(f"✓ Features shape: {batch_a_features[0].shape}")
        print(f"✓ Targets shape: {batch_a_targets.shape}")
        print(f"✓ Batch A metadata: {len(batch_a_meta)} rows")

        assert batch_a_features[0].shape == (3, 10), "Features should be (3, 10)"
        assert batch_a_targets.shape == (3, 1), "Targets should be (3, 1)"
        assert len(batch_a_meta) == 2, "Should have 2 batch A samples"

        return True

    except Exception as e:
        print(f"✗ Integrated test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Phase 3 Smoke Test - Targets & Metadata ===\n")

    success1 = test_targets_basic()
    print()
    success2 = test_target_update()
    print()
    success3 = test_metadata()
    print()
    success4 = test_integrated()

    if success1 and success2 and success3 and success4:
        print("\n🎉 Phase 3 - PASSED")
        print("Targets and metadata working correctly!")
    else:
        print("\n❌ Phase 3 - FAILED")
        exit(1)
