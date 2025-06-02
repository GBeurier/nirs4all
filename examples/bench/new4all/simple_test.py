"""
Simple test script for target management functionality.
Tests the TargetManager directly without complex imports.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the TargetManager directly
exec(open('TargetManager.py').read())

def test_target_manager_basic():
    """Test basic TargetManager functionality."""
    print("=== Testing TargetManager Directly ===\n")

    # Test 1: Classification with string labels
    print("1. String classification:")
    tm_class = TargetManager(task_type="classification")
    sample_ids = [1, 2, 3, 4, 5]
    targets = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
    tm_class.add_targets(sample_ids, targets)

    print(f"   Task type: {tm_class.task_type}")
    print(f"   Classes: {tm_class.classes_}")
    print(f"   Original: {tm_class.get_targets(sample_ids, 'original')}")
    print(f"   Regression: {tm_class.get_targets(sample_ids, 'regression')}")
    print(f"   Classification: {tm_class.get_targets(sample_ids, 'classification')}")

    # Test target transformation
    transformers = [StandardScaler()]
    transformed = tm_class.fit_transform_targets(sample_ids, transformers, "regression", "test_key")
    print(f"   Transformed: {transformed}")

    # Test inverse transform
    fake_predictions = np.array([0.5, -1.2, 0.3, 1.1, -0.8])
    inverse_pred = tm_class.inverse_transform_predictions(fake_predictions, "regression", "test_key", True)
    print(f"   Inverse predictions: {inverse_pred}")
    print()

    # Test 2: Regression with binning
    print("2. Regression with binning:")
    tm_reg = TargetManager(task_type="regression")
    reg_targets = np.array([1.5, 2.8, 4.1, 6.3, 8.7, 10.2, 12.5, 15.1])
    reg_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    tm_reg.add_targets(reg_ids, reg_targets)

    print(f"   Regression values: {tm_reg.get_targets(reg_ids, 'regression')}")
    print(f"   Binned classes: {tm_reg.get_targets(reg_ids, 'classification')}")
    print(f"   Number of bins: {tm_reg.n_classes_}")
    print()

    # Test 3: Auto-detection
    print("3. Auto-detection:")
    tm_auto = TargetManager(task_type="auto")
    binary_targets = np.array([0, 1, 1, 0, 1, 0])
    tm_auto.add_targets([20, 21, 22, 23, 24, 25], binary_targets)
    print(f"   Detected type: {tm_auto.task_type}")
    print(f"   Is binary: {tm_auto.is_binary}")
    print()

    print("=== TargetManager Basic Test Complete ===")

if __name__ == "__main__":
    test_target_manager_basic()
