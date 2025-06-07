"""
Smoke Test Phase 2 - Zero-Copy Slicing

This test verifies that the SpectroDataset can slice features with zero-copy semantics.
"""

import numpy as np
from nirs4all.dataset import SpectroDataset


def test_slice_features():
    """Test slicing features with different filters and layouts."""
    try:
        ds = SpectroDataset()

        # Create test data - one source with 20 rows
        test_data = np.arange(20 * 2).reshape(20, 2).astype(np.float32)
        ds.add_features([test_data])
          # Mark first 15 as train, rest as test
        ds.features.index_df = ds.features.index_df.with_columns(
            partition=pl.when(pl.col("row") < 15).then(pl.lit("train")).otherwise(pl.lit("test"))
        )

        # Test filtering for train partition
        x_train, = ds.x({"partition": "train"}, layout="2d")

        print(f"✓ Train data shape: {x_train.shape}")
        assert x_train.shape == (15, 2), f"Expected (15, 2), got {x_train.shape}"

        # Test filtering for test partition
        x_test, = ds.x({"partition": "test"}, layout="2d")

        print(f"✓ Test data shape: {x_test.shape}")
        assert x_test.shape == (5, 2), f"Expected (5, 2), got {x_test.shape}"

        # Test zero-copy semantics
        shares_memory = np.may_share_memory(x_train, ds.features.sources[0].array)
        print(f"✓ Zero-copy maintained: {shares_memory}")
        assert shares_memory, "Arrays should share memory for zero-copy"

        # Test get_indexed_features
        (x_indexed,), idx_df = ds.get_indexed_features({"partition": "train"}, layout="2d")

        print(f"✓ Indexed features shape: {x_indexed.shape}")
        print(f"✓ Index DataFrame rows: {len(idx_df)}")
        assert x_indexed.shape == (15, 2), "Indexed features should match filtered data"
        assert len(idx_df) == 15, "Index should have 15 rows for train partition"

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_layouts():
    """Test different layout options."""
    try:
        ds = SpectroDataset()

        # Create small test data for layout testing
        test_data = np.arange(6 * 3).reshape(6, 3).astype(np.float32)
        ds.add_features([test_data])

        # Test 2D layout
        x_2d, = ds.x({}, layout="2d")
        print(f"✓ 2D layout shape: {x_2d.shape}")
        assert x_2d.shape == (6, 3), "2D layout should preserve original shape"

        # Test 3D layout
        x_3d, = ds.x({}, layout="3d")
        print(f"✓ 3D layout shape: {x_3d.shape}")
        assert x_3d.shape == (6, 1, 3), "3D layout should add variant dimension"

        # Test 3D transpose layout
        x_3d_t, = ds.x({}, layout="3d_transpose")
        print(f"✓ 3D transpose layout shape: {x_3d_t.shape}")
        assert x_3d_t.shape == (6, 3, 1), "3D transpose should swap last two axes"

        return True

    except Exception as e:
        print(f"✗ Layout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_source_concat():
    """Test source concatenation."""
    try:
        ds = SpectroDataset()

        # Create two sources
        source1 = np.random.rand(5, 10).astype(np.float32)
        source2 = np.random.rand(5, 15).astype(np.float32)
        ds.add_features([source1, source2])

        # Test without concatenation
        x1, x2 = ds.x({}, layout="2d", src_concat=False)
        print(f"✓ Source 1 shape: {x1.shape}, Source 2 shape: {x2.shape}")
        assert x1.shape == (5, 10), "Source 1 should have original shape"
        assert x2.shape == (5, 15), "Source 2 should have original shape"

        # Test with concatenation
        x_concat, = ds.x({}, layout="2d", src_concat=True)
        print(f"✓ Concatenated shape: {x_concat.shape}")
        assert x_concat.shape == (5, 25), "Concatenated should combine all features"

        return True

    except Exception as e:
        print(f"✗ Concatenation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Add polars import for the test
    import polars as pl

    print("=== Phase 2 Smoke Test - Zero-Copy Slicing ===\n")

    success1 = test_slice_features()
    print()
    success2 = test_different_layouts()
    print()
    success3 = test_multi_source_concat()

    if success1 and success2 and success3:
        print("\n🎉 Phase 2 - PASSED")
        print("train shape: (15,2)")
    else:
        print("\n❌ Phase 2 - FAILED")
        exit(1)
