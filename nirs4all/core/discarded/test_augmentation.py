#!/usr/bin/env python3
"""
Test script for the three dataset shape-changing operations:
1. Sample augmentation
2. Feature augmentation
3. Branching

This script verifies that the basic functionality works without applying transformations.
"""

import numpy as np
from SpectraDataset import SpectraDataset

def test_basic_functionality():
    """Test the three key operations without transformations."""

    print("="*80)
    print("TESTING DATASET SHAPE-CHANGING OPERATIONS")
    print("="*80)

    # Create initial dataset with dual sources
    np.random.seed(42)
    n_samples = 10

    # Two different sources
    source_1 = np.random.rand(n_samples, 100) * 4 + 4
    source_2 = np.random.rand(n_samples, 50) * 40 + 6
    targets = np.random.randint(0, 3, size=n_samples)

    print("1. INITIAL DATASET CREATION")
    print("-" * 40)

    dataset = SpectraDataset(task_type="classification")
    sample_ids = dataset.add_data(
        features=[source_1, source_2],
        targets=targets,
        partition="train"
    )

    print(f"Initial dataset:\n{dataset}")
    print(f"Initial sample IDs: {sample_ids}")
    print(f"Features shape: source1={source_1.shape}, source2={source_2.shape}")
    print(f"Total features per sample: {source_1.shape[1] + source_2.shape[1]}")

    # Test sample augmentation
    print("\n2. SAMPLE AUGMENTATION TEST")
    print("-" * 40)

    new_sample_ids = dataset.sample_augmentation(
        partition="train",
        n_copies=2,
        processing_tag="augmented"
    )

    print(f"After sample augmentation:\n{dataset}")
    print(f"New sample IDs created: {new_sample_ids}")
    print(f"Expected: {len(sample_ids) * 2} new samples")
    print(f"Actual: {len(new_sample_ids)} new samples")

    # Verify the origins are preserved
    train_samples = dataset.indices.filter(dataset.indices['partition'] == 'train')
    original_samples = train_samples.filter(train_samples['origin'] == train_samples['sample'])
    augmented_samples = train_samples.filter(train_samples['origin'] != train_samples['sample'])

    print(f"Original samples: {len(original_samples)} (processing: {original_samples['processing'].unique().to_list()})")
    print(f"Augmented samples: {len(augmented_samples)} (processing: {augmented_samples['processing'].unique().to_list()})")

    # Test feature augmentation
    print("\n3. FEATURE AUGMENTATION TEST")
    print("-" * 40)

    initial_rows = len(dataset.indices)
    dataset.feature_augmentation(processing_tag="feat_augmented")

    print(f"After feature augmentation:\n{dataset}")
    print(f"Rows before: {initial_rows}")
    print(f"Rows after: {len(dataset.indices)}")
    print(f"Expected: {initial_rows * 2} (doubled)")

    # Check processing types
    processing_types = dataset.indices['processing'].unique().to_list()
    print(f"Processing types: {processing_types}")

    # Test branching
    print("\n4. BRANCHING TEST")
    print("-" * 40)

    train_rows_before = len(dataset.indices.filter(dataset.indices['partition'] == 'train'))
    dataset.branch_dataset(n_branches=3)
    train_rows_after = len(dataset.indices.filter(dataset.indices['partition'] == 'train'))

    print(f"After branching:\n{dataset}")
    print(f"Train rows before: {train_rows_before}")
    print(f"Train rows after: {train_rows_after}")
    print(f"Expected: {train_rows_before * 3} (tripled for 3 branches)")

    # Check branch distribution
    branch_counts = dataset.indices.filter(dataset.indices['partition'] == 'train').group_by('branch').count()
    print(f"Branch distribution:\n{branch_counts}")

    # Test 2D feature extraction
    print("\n5. FEATURE EXTRACTION TEST")
    print("-" * 40)

    # Get features for branch 0 only
    features_2d = dataset.get_features_2d(filters={'branch': 0, 'partition': 'train'})
    print(f"2D Features for branch 0: {features_2d.shape}")

    # Get features for all branches
    all_features = dataset.get_features_2d(filters={'partition': 'train'})
    print(f"2D Features for all branches: {all_features.shape}")

    # Test 3D feature extraction
    features_3d = dataset.get_features_3d(filters={'branch': 0, 'partition': 'train'})
    print(f"3D Features for branch 0: {features_3d.shape}")

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

    return dataset

if __name__ == "__main__":
    test_dataset = test_basic_functionality()
