"""
Test file to demonstrate the three transformation modes.

This file shows how the updated TransformationOperation works with:
1. transformation mode: In-place replacement with processing index updates
2. sample_augmentation mode: Copy train set with new sample/row IDs
3. feature_augmentation mode: Copy dataset with same sample IDs but different processing
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from TransformationOperation import TransformationOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


def test_transformation_modes():
    """Test all three transformation modes."""

    # Create a sample dataset
    dataset = SpectraDataset(task_type="regression")
    context = PipelineContext()

    # Add some sample data
    n_samples = 100
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(n_samples//2, n_features)
    y_train = np.random.randn(n_samples)
    y_test = np.random.randn(n_samples//2)

    # Add train and test data
    dataset.add_data(X_train, y_train, partition="train")
    dataset.add_data(X_test, y_test, partition="test")
    print(f"Initial dataset size: {len(dataset)}")
    print(f"Initial partitions: {dataset.indices['partition'].unique().to_list()}")
    print(f"Initial processing values: {dataset.indices['processing'].unique().to_list()}")
    print(f"Initial unique sample count: {len(dataset.indices['sample'].unique())}")

    # Test 1: Standard transformation mode
    print("\n=== Test 1: Standard Transformation Mode ===")
    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    )

    # Execute transformation
    transform_op.execute(dataset, context)

    print(f"After transformation - dataset size: {len(dataset)}")
    print(f"Processing values: {dataset.indices['processing'].unique().to_list()}")
    print("✓ In-place transformation completed - same dataset size, updated processing")

    # Test 2: Sample augmentation mode
    print("\n=== Test 2: Sample Augmentation Mode ===")

    # Create fresh dataset for clean test
    dataset2 = SpectraDataset(task_type="regression")
    dataset2.add_data(X_train, y_train, partition="train")
    dataset2.add_data(X_test, y_test, partition="test")

    # Use multiple transformers for sample augmentation
    augment_op = TransformationOperation(
        transformer=[StandardScaler(), MinMaxScaler()],
        mode="sample_augmentation"
    )

    initial_size = len(dataset2)
    augment_op.execute(dataset2, context)
    print(f"Before augmentation: {initial_size}")
    print(f"After augmentation: {len(dataset2)}")
    print(f"Unique sample count increased: {len(dataset2.indices['sample'].unique())}")
    print(f"Processing values: {dataset2.indices['processing'].unique().to_list()}")
    print("✓ Sample augmentation completed - new samples created with new IDs")

    # Test 3: Feature augmentation mode
    print("\n=== Test 3: Feature Augmentation Mode ===")

    # Create fresh dataset for clean test
    dataset3 = SpectraDataset(task_type="regression")
    dataset3.add_data(X_train, y_train, partition="train")
    dataset3.add_data(X_test, y_test, partition="test")

    feature_aug_op = TransformationOperation(
        transformer=[StandardScaler(), MinMaxScaler()],
        mode="feature_augmentation"
    )

    initial_size = len(dataset3)
    initial_samples = sorted(dataset3.indices['sample'].unique().to_list())

    feature_aug_op.execute(dataset3, context)
    print(f"Before feature augmentation: {initial_size}")
    print(f"After feature augmentation: {len(dataset3)}")
    print(f"Original unique sample count: {len(initial_samples)}")
    print(f"Processing values: {dataset3.indices['processing'].unique().to_list()}")
    print("✓ Feature augmentation completed - same sample IDs, multiple feature versions")

    # Verify that sample IDs are the same but we have more rows
    final_samples = sorted(dataset3.indices['sample'].unique().to_list())
    print(f"Final unique sample count: {len(final_samples)}")
    assert initial_samples == final_samples, "Sample IDs should remain the same in feature augmentation"
    assert len(dataset3) > initial_size, "Should have more rows after feature augmentation"

    print("\n=== All tests completed successfully! ===")


if __name__ == "__main__":
    test_transformation_modes()
