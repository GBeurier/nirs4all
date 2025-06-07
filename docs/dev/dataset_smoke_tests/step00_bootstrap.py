"""
Smoke Test Phase 0 - Bootstrap

This test verifies that the SpectroDataset package can be imported successfully.
"""

def test_bootstrap():
    """Test that the SpectroDataset package can be imported."""
    try:
        from nirs4all.dataset import SpectroDataset
        print("✓ SpectroDataset package imported OK")

        # Test basic instantiation
        ds = SpectroDataset()
        print(f"✓ SpectroDataset instantiated: {ds}")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except (AttributeError, TypeError) as e:
        print(f"✗ Instantiation error: {e}")
        return False


if __name__ == "__main__":
    success = test_bootstrap()
    if success:
        print("\n🎉 Phase 0 Bootstrap - PASSED")
    else:
        print("\n❌ Phase 0 Bootstrap - FAILED")
        exit(1)
