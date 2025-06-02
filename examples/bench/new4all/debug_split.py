#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from SplitOperation import SplitStrategy

def debug_split_issue():
    """Debug the split operation to understand the indexing issue"""
    np.random.seed(42)

    # Create simple dataset
    n_samples = 10
    n_features = 5
    X = np.random.random((n_samples, n_features))

    print("Original data shape:", X.shape)
    print("Original data:")
    print(X)

    # Create dataset
    dataset = SpectraDataset(task_type='classification')
    sample_ids = dataset.add_data([X], partition="train")

    print("\nDataset indices before split:")
    print(dataset.indices)

    print("\nDataset features shape:", dataset.features.sources[0].shape)

    # Apply split
    split_op = SplitStrategy.train_val_test(train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, stratified=False)    # Manually apply split to see what happens
    split_indices = split_op.generate_split_indices(len(dataset))
    print("\nSplit indices:")
    for partition, indices in split_indices.items():
        print(f"  {partition}: {indices}")

    # Apply to dataset
    split_op.apply_splits_to_dataset(dataset, split_indices)

    print("\nDataset indices after split:")
    print(dataset.indices)

    print("\nDataset features shape after split:", dataset.features.sources[0].shape)

    # Test accessing different partitions
    for partition in ['train', 'val', 'test']:
        view = dataset.select(partition=partition)
        print(f"\n{partition} partition:")
        print(f"  Number of samples: {len(view)}")
        print(f"  Row indices: {view.row_indices}")
        print(f"  Sample IDs: {view.sample_ids}")

        if len(view) > 0:
            try:
                features = view.get_features()
                print(f"  Features shape: {features.shape}")
                print(f"  First sample features: {features[0]}")
            except Exception as e:
                print(f"  Error getting features: {e}")

if __name__ == "__main__":
    debug_split_issue()
