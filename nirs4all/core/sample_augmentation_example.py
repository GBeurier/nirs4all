#!/usr/bin/env python3
"""
Example of using the new sample augmentation methods for class balancing.

This example demonstrates:
1. Creating a dataset with imbalanced classes
2. Using get_sample_indices_by_class() to identify class distribution
3. Using sample_augmentation_by_indices() to balance classes
"""

import sys
import os
import numpy as np

# Add the examples directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bench', 'core'))

from SpectraDataset import SpectraDataset
from SpectraTargets import SpectraTargets


def create_imbalanced_dataset():
    """Create a dataset with imbalanced classes for demonstration."""
    print("=== Creating Imbalanced Dataset ===")

    # Create dataset
    dataset = SpectraDataset(task_type="classification")

    # Generate synthetic spectral data
    np.random.seed(42)

    # Majority class: 50 samples
    majority_features = np.random.randn(50, 100) + 1.0  # Slight offset
    majority_targets = np.array(['class_A'] * 50)    # Minority class: 10 samples
    minority_features = np.random.randn(10, 100) - 1.0  # Different offset
    minority_targets = np.array(['class_B'] * 10)

    # Combine all features and targets to add them together
    all_features = np.vstack([majority_features, minority_features])
    all_targets = np.concatenate([majority_targets, minority_targets])

    # Add all data to dataset at once to avoid label encoder issues
    all_ids = dataset.add_data(
        features=all_features,
        targets=all_targets,
        partition="train",
        processing="raw"
    )

    majority_ids = all_ids[:50]  # First 50 are majority class
    minority_ids = all_ids[50:]  # Last 10 are minority class

    print(f"Dataset created:")
    print(f"  Majority class (class_A): {len(majority_ids)} samples")
    print(f"  Minority class (class_B): {len(minority_ids)} samples")
    print(f"  Total samples: {len(dataset)}")

    return dataset


def demonstrate_class_balancing(dataset):
    """Demonstrate class balancing using the new methods."""
    print("\n=== Class Balancing Example ===")

    # 1. Get sample indices by class
    class_samples = dataset.get_sample_indices_by_class("train")
    print("Class distribution before balancing:")
    for class_val, sample_ids in class_samples.items():
        print(f"  {class_val}: {len(sample_ids)} samples {sample_ids[:5]}...")

    # 2. Determine how many samples to augment
    class_sizes = {k: len(v) for k, v in class_samples.items()}
    max_class_size = max(class_sizes.values())
    print(f"\nTarget size for all classes: {max_class_size}")

    # 3. Create augmentation plan
    indices_to_augment = []
    for class_val, sample_ids in class_samples.items():
        current_size = len(sample_ids)
        if current_size < max_class_size:
            n_needed = max_class_size - current_size
            print(f"  {class_val}: Need {n_needed} more samples")

            # Sample with replacement to get the needed samples
            augment_samples = np.random.choice(sample_ids, n_needed, replace=True)
            indices_to_augment.extend(augment_samples.tolist())

    print(f"\nAugmenting {len(indices_to_augment)} samples: {indices_to_augment}")

    # 4. Perform augmentation
    new_sample_ids = dataset.sample_augmentation_by_indices(
        sample_indices_to_augment=indices_to_augment,
        processing_tag="balanced"
    )

    print(f"Created {len(new_sample_ids)} new samples")

    # 5. Verify balancing
    class_samples_after = dataset.get_sample_indices_by_class("train")
    print("\nClass distribution after balancing:")
    for class_val, sample_ids in class_samples_after.items():
        print(f"  {class_val}: {len(sample_ids)} samples")

    print(f"\nTotal samples after balancing: {len(dataset)}")


def demonstrate_specific_augmentation(dataset):
    """Demonstrate augmenting specific samples multiple times."""
    print("\n=== Specific Sample Augmentation Example ===")

    # Get some specific samples to augment
    class_samples = dataset.get_sample_indices_by_class("train")
    print(f"Available classes: {list(class_samples.keys())}")    # Get minority class samples (class with fewer samples)
    class_sizes = {k: len(v) for k, v in class_samples.items()}
    minority_class = min(class_sizes.keys(), key=lambda x: class_sizes[x])
    minority_samples = class_samples[minority_class][:3]  # First 3 minority samples

    # Create multiple copies of each sample
    specific_augmentation = []
    for sample_id in minority_samples:
        specific_augmentation.extend([sample_id] * 3)  # 3 copies each

    print("Augmenting specific samples 3 times each:")
    print(f"  Minority class: {minority_class}")
    print(f"  Original samples: {minority_samples}")
    print(f"  Augmentation list: {specific_augmentation}")

    # Perform specific augmentation
    new_ids = dataset.sample_augmentation_by_indices(
        sample_indices_to_augment=specific_augmentation,
        processing_tag="specific_augment"
    )

    print(f"Created {len(new_ids)} new samples with specific augmentation")
    print(f"Final dataset size: {len(dataset)}")


def main():
    """Main example function."""
    print("Sample Augmentation by Indices Example")
    print("=" * 50)

    # Create imbalanced dataset
    dataset = create_imbalanced_dataset()

    # Demonstrate class balancing
    demonstrate_class_balancing(dataset)

    # Demonstrate specific augmentation
    demonstrate_specific_augmentation(dataset)

    print("\n=== Summary ===")
    print(f"Final dataset contains {len(dataset)} samples")

    # Show processing types
    processing_types = dataset.indices["processing"].unique().to_list()
    print(f"Processing types: {processing_types}")

    for proc_type in processing_types:
        count = len(dataset.indices.filter(dataset.indices["processing"] == proc_type))
        print(f"  {proc_type}: {count} samples")


if __name__ == "__main__":
    main()
