#!/usr/bin/env python3
"""
Test script to validate the fix for the core test notebook
"""

import sys
import os
import numpy as np

# Add the core directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset

def test_fixed_approach():
    """Test the fixed approach using separate datasets for each output."""
    print("Testing fixed approach...")

    dataset_reg_1_1 = SpectraDataset(task_type="regression")  # Single-output regression
    dataset_reg_2_1 = SpectraDataset(task_type="regression")  # Single-output regression (first output)
    dataset_reg_2_2 = SpectraDataset(task_type="regression")  # Single-output regression (second output)
    dataset_cla_2_1 = SpectraDataset(task_type="classification")  # Single-output classification (first output)
    dataset_cla_2_2 = SpectraDataset(task_type="classification")  # Single-output classification (second output)
    dataset_bin_1_1 = SpectraDataset(task_type="binary")  # Binary classification

    f1_source = np.random.rand(100, 1000)
    f2_source = np.random.rand(100, 500)

    # Single-output targets
    reg_target_1 = np.random.rand(100,)  # 1D array for single-output regression
    reg_target_2_first = np.random.rand(100,)  # First output of multi-output regression
    reg_target_2_second = np.random.rand(100,)  # Second output of multi-output regression
    cla_target_2_first = np.random.randint(0, 5, size=(100,))  # First output of multi-output classification
    cla_target_2_second = np.random.randint(0, 5, size=(100,))  # Second output of multi-output classification
    bin_target_1 = np.random.randint(0, 2, size=(100,))  # 1D array for binary classification

    # Add data to datasets
    try:
        dataset_reg_1_1.add_data([f1_source], reg_target_1)
        print("✓ Single-output regression dataset created successfully")

        dataset_reg_2_1.add_data([f1_source, f2_source], reg_target_2_first)
        print("✓ Multi-source single-output regression dataset (first) created successfully")

        dataset_reg_2_2.add_data([f1_source, f2_source], reg_target_2_second)
        print("✓ Multi-source single-output regression dataset (second) created successfully")

        dataset_cla_2_1.add_data([f1_source, f2_source], cla_target_2_first)
        print("✓ Multi-source single-output classification dataset (first) created successfully")

        dataset_cla_2_2.add_data([f1_source, f2_source], cla_target_2_second)
        print("✓ Multi-source single-output classification dataset (second) created successfully")

        dataset_bin_1_1.add_data([f1_source], bin_target_1)
        print("✓ Binary classification dataset created successfully")

        print("\n--- Dataset Information ---")
        print("Dataset for regression 1-1:", dataset_reg_1_1)
        print("Dataset for regression 2-1 (first output):", dataset_reg_2_1)
        print("Dataset for regression 2-2 (second output):", dataset_reg_2_2)
        print("Dataset for classification 2-1 (first output):", dataset_cla_2_1)
        print("Dataset for classification 2-2 (second output):", dataset_cla_2_2)
        print("Dataset for binary classification 1-1:", dataset_bin_1_1)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_approach()
    if success:
        print("\n🎉 All tests passed! The fix works correctly.")
    else:
        print("\n💥 Tests failed. Check the error messages above.")
