import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler
from SpectraDataset import SpectraDataset
from TargetManager import TargetManager

def test_simple():
    """Simple test of core functionality."""

    print("=== Testing Basic TargetManager and SpectraDataset ===\n")

    # Test 1: Basic SpectraDataset creation
    print("1. Creating SpectraDataset...")

    # Create proper 2D feature data
    features = [np.random.randn(10, 100)]  # 10 samples, 100 features
    targets = np.array(['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat'])

    dataset = SpectraDataset(task_type="auto")
    sample_ids = dataset.add_data(features, targets=targets)

    print(f"   Created dataset with {len(sample_ids)} samples")
    print(f"   Task type: {dataset.task_type}")
    print(f"   Classes: {dataset.classes_}")

    # Test 2: Target representations
    print("\n2. Testing target representations...")

    original = dataset.get_targets(sample_ids[:5], "original")
    classification = dataset.get_targets(sample_ids[:5], "classification")
    regression = dataset.get_targets(sample_ids[:5], "regression")

    print(f"   Original: {original}")
    print(f"   Classification: {classification}")
    print(f"   Regression: {regression}")

    # Test 3: Features access
    print("\n3. Testing features access...")

    all_features = dataset.get_features(np.arange(len(sample_ids)))
    print(f"   Features shape: {all_features.shape}")

    # Test 4: Views
    print("\n4. Testing views...")

    # Add test data
    test_features = [np.random.randn(3, 100)]
    test_targets = np.array(['cat', 'dog', 'bird'])
    test_ids = dataset.add_data(test_features, targets=test_targets, partition="test")

    train_view = dataset.select(partition="train")
    test_view = dataset.select(partition="test")

    print(f"   Train samples: {len(train_view)}")
    print(f"   Test samples: {len(test_view)}")

    train_features = train_view.get_features()
    test_targets = test_view.get_targets()

    print(f"   Train features shape: {train_features.shape}")
    print(f"   Test targets: {test_targets}")

    print("\n=== Basic Test Complete ===")

if __name__ == "__main__":
    test_simple()
