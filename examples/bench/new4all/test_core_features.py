import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from SpectraDataset import SpectraDataset
from TargetManager import TargetManager


def test_core_dataset_features():
    """Test core SpectraDataset and TargetManager integration."""

    print("=== Testing Core Dataset Features ===\n")

    # 1. Test SpectraDataset with different target types
    print("1. Testing SpectraDataset with string classification targets...")
      # Create sample data
    features = [np.random.randn(100) for _ in range(10)]
    features = np.array(features)  # Convert to 2D array (10 samples, 100 features each)
    targets = np.array(['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat'])
      # Create dataset with auto task detection
    dataset = SpectraDataset(task_type="auto")
    sample_ids = dataset.add_data(features, targets=targets)

    print(f"   Created dataset with {len(sample_ids)} samples")
    print(f"   Task type: {dataset.task_type}")
    print(f"   Number of classes: {dataset.n_classes}")
    print(f"   Classes: {dataset.classes_}")

    # Test different target representations
    print("\n2. Testing target representations...")

    # Get original targets
    original = dataset.get_targets(sample_ids[:5], "original")
    print(f"   Original targets: {original}")

    # Get classification targets (encoded)
    classification = dataset.get_targets(sample_ids[:5], "classification")
    print(f"   Classification targets: {classification}")

    # Get regression targets (numeric)
    regression = dataset.get_targets(sample_ids[:5], "regression")
    print(f"   Regression targets: {regression}")

    # 3. Test target transformations
    print("\n3. Testing target transformations...")

    from sklearn.preprocessing import StandardScaler
    transformers = [StandardScaler()]

    # Fit and transform targets
    transformed = dataset.fit_transform_targets(
        sample_ids, transformers, "regression", "scaler"
    )
    print(f"   Transformed targets: {transformed[:5]}")

    # Test inverse transform
    predictions = np.array([0.5, -0.3, 1.2, -0.8, 0.9])
    inverse = dataset.inverse_transform_predictions(
        predictions, "regression", "scaler", to_original=True
    )
    print(f"   Inverse transformed: {inverse}")

    # 4. Test regression data
    print("\n4. Testing with regression data...")
    regression_targets = np.random.randn(8) * 10 + 50  # Random values around 50
    regression_features = [np.random.randn(100) for _ in range(8)]
    regression_features = np.array(regression_features)  # Convert to 2D array (8 samples, 100 features each)

    reg_dataset = SpectraDataset(task_type="auto")
    reg_sample_ids = reg_dataset.add_data(regression_features, targets=regression_targets)

    print(f"   Task type: {reg_dataset.task_type}")
    print(f"   Regression values: {regression_targets}")

    # Get different representations
    reg_orig = reg_dataset.get_targets(reg_sample_ids, "original")
    reg_class = reg_dataset.get_targets(reg_sample_ids, "classification")

    print(f"   Original: {reg_orig}")
    print(f"   Binned classes: {reg_class}")
    print(f"   Number of bins: {reg_dataset.n_classes}")
      # 5. Test dataset views and filtering
    print("\n5. Testing dataset views...")
      # Add more data with different partitions
    features2 = [np.random.randn(100) for _ in range(5)]
    features2 = np.array(features2)  # Convert to 2D array (5 samples, 100 features each)
    targets2 = np.array(['bird', 'cat', 'dog', 'bird', 'cat'])
    test_ids = dataset.add_data(features2, targets=targets2, partition="test")

    # Create views using partition
    train_view = dataset.select(partition="train")
    test_view = dataset.select(partition="test")

    print(f"   Train samples: {len(train_view)}")
    print(f"   Test samples: {len(test_view)}")

    # Test view functionality
    train_targets = train_view.get_targets("classification")
    print(f"   Train targets: {train_targets}")

    test_features = test_view.get_features()
    print(f"   Test features shape: {test_features.shape}")

    # 6. Test target info
    print("\n6. Testing target information...")

    target_info = dataset.get_target_info()
    print(f"   Target info: {target_info}")

    print("\n=== Core Dataset Features Test Complete ===")


if __name__ == "__main__":
    test_core_dataset_features()
