#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from SplitOperation import SplitStrategy
from Pipeline import Pipeline
from OperationFactory import OperationFactory

def debug_large_split_issue():
    """Debug the split operation with a larger dataset like in the main test"""
    np.random.seed(42)

    # Create dataset similar to the main test
    n_samples = 200
    n_wavelengths = 100

    # Generate synthetic NIR-like spectra
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Base spectra with some peaks
    base_spectrum = np.exp(-((wavelengths - 1400) / 200) ** 2) + \
                   0.5 * np.exp(-((wavelengths - 1900) / 150) ** 2) + \
                   0.3 * np.exp(-((wavelengths - 2100) / 100) ** 2)

    # Add sample variations
    X_original = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        # Add noise and variations
        noise = np.random.normal(0, 0.02, n_wavelengths)
        intensity_variation = np.random.normal(1.0, 0.1)
        baseline_shift = np.random.normal(0, 0.05)

        X_original[i] = intensity_variation * base_spectrum + baseline_shift + noise

    # Create second source (preprocessed version)
    X_derivative = np.gradient(X_original, axis=1)

    print("Original data shapes:")
    print(f"  X_original: {X_original.shape}")
    print(f"  X_derivative: {X_derivative.shape}")

    # Create dataset
    dataset = SpectraDataset(task_type='classification')
    sample_ids = dataset.add_data([X_original, X_derivative], partition="train")

    print(f"\nDataset info before split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features sources: {len(dataset.features.sources)}")
    print(f"  Source 0 shape: {dataset.features.sources[0].shape}")
    print(f"  Source 1 shape: {dataset.features.sources[1].shape}")

    # Apply split
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, stratified=False)
    split_indices = split_op.generate_split_indices(len(dataset))

    print(f"\nSplit indices:")
    for partition, indices in split_indices.items():
        print(f"  {partition}: {len(indices)} samples, indices range: {indices.min()}-{indices.max()}")

    # Apply to dataset
    split_op.apply_splits_to_dataset(dataset, split_indices)

    print(f"\nDataset info after split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features sources: {len(dataset.features.sources)}")
    print(f"  Source 0 shape: {dataset.features.sources[0].shape}")
    print(f"  Source 1 shape: {dataset.features.sources[1].shape}")

    # Test each partition
    for partition in ['train', 'val', 'test']:
        view = dataset.select(partition=partition)
        print(f"\n{partition} partition:")
        print(f"  Number of samples: {len(view)}")
        print(f"  Row indices range: {view.row_indices.min()}-{view.row_indices.max()}")
        print(f"  Sample IDs range: {min(view.sample_ids)}-{max(view.sample_ids)}")

        if len(view) > 0:
            try:
                features = view.get_features()
                print(f"  Features shape: {features.shape}")
            except Exception as e:
                print(f"  Error getting features: {e}")
                print(f"  Row indices causing issue: {view.row_indices}")

if __name__ == "__main__":
    debug_large_split_issue()
