#!/usr/bin/env python3

import numpy as np
from SpectraDataset import SpectraDataset
from SpectraFeatures import SpectraFeatures

def create_test_dataset():
    """Create a simple test dataset and print its indices"""
    np.random.seed(42)

    # Create simple data
    n_samples = 10
    n_features = 50
    X = np.random.random((n_samples, n_features))

    # Create dataset
    dataset = SpectraDataset(task_type='classification')

    # Add data
    sample_ids = dataset.add_data([X], partition="train")

    print("Dataset indices columns:", dataset.indices.columns)
    print("Dataset indices shape:", dataset.indices.shape)
    print("First few rows of indices:")
    print(dataset.indices.head())

    return dataset

if __name__ == "__main__":
    dataset = create_test_dataset()
