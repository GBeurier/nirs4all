"""
Smoke Test Phase 1 - Add Features

This test verifies that the SpectroDataset can add feature sources correctly.
"""

import numpy as np
from nirs4all.dataset import SpectroDataset


def test_add_features():
    """Test adding features to SpectroDataset."""
    try:
        ds = SpectroDataset()

        # Create test data - two different spectroscopy sources
        raman = np.random.rand(10, 500).astype(np.float32)
        nirs = np.random.rand(10, 2500).astype(np.float32)

        # Add features
        ds.add_features([raman, nirs])

        # Verify results
        n_sources = len(ds.features.sources)
        n_rows = ds.features.n_samples()

        print(f"✓ Added {n_sources} sources successfully")
        print(f"✓ Number of samples: {n_rows}")
        print(f"✓ Feature block: {ds.features}")

        # Verify index was created
        if ds.features.index_df is not None:
            print(f"✓ Index created with {len(ds.features.index_df)} rows")
            print(f"✓ Index columns: {ds.features.index_df.columns}")
        else:
            print("✗ Index was not created")
            return False

        # Verify array properties
        assert n_sources == 2, f"Expected 2 sources, got {n_sources}"
        assert n_rows == 10, f"Expected 10 rows, got {n_rows}"

        # Verify shapes
        assert ds.features.sources[0].array.shape == (10, 500), "Raman source has wrong shape"
        assert ds.features.sources[1].array.shape == (10, 2500), "NIRS source has wrong shape"
        print("✓ All assertions passed")
        return True

    except (ValueError, TypeError, AssertionError, AttributeError) as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_copy():
    """Test that feature sources maintain zero-copy semantics."""
    try:
        ds = SpectroDataset()

        # Create test array
        original = np.random.rand(5, 10).astype(np.float32)
        ds.add_features([original])

        # Verify zero-copy (arrays share memory)
        stored = ds.features.sources[0].array
        shares_memory = np.may_share_memory(original, stored)

        if shares_memory:
            print("✓ Zero-copy semantics maintained")
        else:
            print("✗ Arrays do not share memory")
            return False

        return True

    except Exception as e:
        print(f"✗ Zero-copy test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Phase 1 Smoke Test - Add Features ===\n")

    success1 = test_add_features()
    print()
    success2 = test_zero_copy()

    if success1 and success2:
        print("\n🎉 Phase 1 - PASSED")
        print("Sources: 2, Rows: 10")
    else:
        print("\n❌ Phase 1 - FAILED")
        exit(1)
