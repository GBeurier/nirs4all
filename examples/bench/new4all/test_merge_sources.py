#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from MergeSourcesOperation import MergeSourcesOperation
from PipelineContext import PipelineContext

def test_merge_sources():
    """Test that merge sources operation replaces original sources"""
    print("=== Testing Merge Sources Operation ===")

    # Create test dataset with multiple sources
    dataset = SpectraDataset(task_type='classification')

    # Create data for 3 sources with different feature sizes
    source1 = np.random.random((10, 5))  # 10 samples, 5 features
    source2 = np.random.random((10, 3))  # 10 samples, 3 features
    source3 = np.random.random((10, 7))  # 10 samples, 7 features

    print(f"Original sources:")
    print(f"  Source 1 shape: {source1.shape}")
    print(f"  Source 2 shape: {source2.shape}")
    print(f"  Source 3 shape: {source3.shape}")

    # Add data to dataset
    dataset.add_data([source1, source2, source3], partition="train")

    print(f"\nDataset before merge:")
    print(f"  Number of sources: {len(dataset.features.sources)}")
    for i, source in enumerate(dataset.features.sources):
        print(f"  Source {i} shape: {source.shape}")

    # Test concatenation merge
    context = PipelineContext()
    merge_op = MergeSourcesOperation(merge_strategy="concatenate")
    merge_op.execute(dataset, context)

    print(f"\nDataset after concatenation merge:")
    print(f"  Number of sources: {len(dataset.features.sources)}")
    for i, source in enumerate(dataset.features.sources):
        print(f"  Source {i} shape: {source.shape}")

    # Verify that we now have only one source
    assert len(dataset.features.sources) == 1, f"Expected 1 source after merge, got {len(dataset.features.sources)}"

    # Verify the merged source has the correct shape
    expected_features = 5 + 3 + 7  # Sum of all original feature counts
    actual_shape = dataset.features.sources[0].shape
    expected_shape = (10, expected_features)

    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"

    print(f"✓ Merge operation successful!")
    print(f"  Original sources (5+3+7=15 features) merged into single source with {actual_shape[1]} features")

    # Test that we can still get features normally
    features = dataset.get_features(np.arange(10), concatenate=True)
    print(f"  Retrieved features shape: {features.shape}")

    assert features.shape == expected_shape, f"get_features returned wrong shape: {features.shape}"

    print(f"✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_merge_sources()
    if success:
        print("\n🎉 Merge Sources Operation is working correctly!")
        print("   - Original sources are replaced with merged result")
        print("   - No old sources remain accessible")
        print("   - Features can be retrieved normally from merged source")
    else:
        print("\n❌ Merge Sources Operation failed!")
    exit(0 if success else 1)
