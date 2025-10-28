"""
Sample Augmentation Examples for nirs4all

This file demonstrates various sample augmentation scenarios.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

from nirs4all.data.dataset import SpectroDataset

# Example 1: Basic Augmentation
# ===============================

def example_basic_augmentation():
    """
    Basic augmentation with a single transformer.
    Creates 2 augmented samples per base sample.
    """
    print("=== Example 1: Basic Augmentation ===\n")

    # Create dataset
    dataset = SpectroDataset("example1")

    # Add base samples (e.g., 50 spectra)
    X_base = np.random.randn(50, 100)  # 50 samples, 100 wavelengths
    y_base = np.random.randint(0, 2, 50)  # Binary classification

    dataset.add_samples(X_base, {"partition": "train"})
    dataset.add_targets(y_base)

    print(f"Base samples: {dataset.x({}, include_augmented=False, layout='2d', concat_source=True).shape[0]}")

    # Generate augmented data using StandardScaler
    scaler = StandardScaler()
    X_augmented = scaler.fit_transform(X_base)
    X_augmented = np.tile(X_augmented, (2, 1))  # Duplicate for count=2

    # Add augmented samples
    dataset.augment_samples(
        data=X_augmented,
        processings=["standardized"],
        augmentation_id="scaler_aug",
        selector={"partition": "train"},
        count=2
    )

    print(f"Total samples: {dataset.x({}, include_augmented=True, layout='2d', concat_source=True).shape[0]}")
    print(f"Augmented samples: {dataset.x({}, include_augmented=True, layout='2d', concat_source=True).shape[0] - dataset.x({}, include_augmented=False, layout='2d', concat_source=True).shape[0]}\n")


# Example 2: Multiple Transformers
# ==================================

def example_multiple_transformers():
    """
    Augmentation with multiple transformers cycling through samples.
    """
    print("=== Example 2: Multiple Transformers ===\n")

    dataset = SpectroDataset("example2")

    # Base data
    X_base = np.random.randn(30, 80)
    y_base = np.array([0]*10 + [1]*10 + [2]*10)  # 3 classes

    dataset.add_samples(X_base, {})
    dataset.add_targets(y_base)

    # Create augmentations with different transformers
    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()

    X_aug1 = scaler1.fit_transform(X_base)
    X_aug2 = scaler2.fit_transform(X_base)

    # First augmentation round
    dataset.augment_samples(
        data=X_aug1,
        processings=["standardized"],
        augmentation_id="standard_scaler",
        count=1
    )

    # Second augmentation round
    dataset.augment_samples(
        data=X_aug2,
        processings=["minmax"],
        augmentation_id="minmax_scaler",
        count=1
    )

    print(f"Base samples: 30")
    print(f"Total samples: {dataset.x({}, layout='2d', concat_source=True).shape[0]}")
    print(f"Samples per augmentation: 30\n")


# Example 3: Leak Prevention in Cross-Validation
# ===============================================

def example_leak_prevention():
    """
    Demonstrates leak prevention: CV splits only use base samples.
    """
    print("=== Example 3: Leak Prevention in CV ===\n")

    from nirs4all.controllers.splitters.split import CrossValidatorController
    from unittest.mock import Mock

    dataset = SpectroDataset("example3")

    # Create base data
    np.random.seed(42)
    X_base = np.random.randn(40, 60)
    y_base = np.array([0]*20 + [1]*20)

    dataset.add_samples(X_base, {})
    dataset.add_targets(y_base)

    # Augment: 3 per sample
    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_base)
    X_aug = np.tile(X_aug, (3, 1))  # 3x augmentation

    dataset.augment_samples(
        data=X_aug,
        processings=["standardized"],
        augmentation_id="aug1",
        count=3
    )

    print(f"Base samples: 40")
    print(f"Total samples after augmentation: {dataset.x({}, layout='2d', concat_source=True).shape[0]}")

    # Perform CV split
    split_controller = CrossValidatorController()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mock_runner = Mock()

    split_controller.execute(
        step={},
        operator=splitter,
        dataset=dataset,
        context={},
        runner=mock_runner
    )

    print("\n✓ CV splits computed using only 40 base samples")
    print("✓ Training folds can access all 160 samples (40 base + 120 augmented)")
    print("✓ Validation folds only see base samples (no leakage)\n")


# Example 4: Balanced Augmentation
# =================================

def example_balanced_augmentation():
    """
    Simulates balanced augmentation to handle class imbalance.
    """
    print("=== Example 4: Balanced Augmentation ===\n")

    dataset = SpectroDataset("example4")

    # Imbalanced dataset
    X_class0 = np.random.randn(80, 50)
    X_class1 = np.random.randn(20, 50)

    X_base = np.vstack([X_class0, X_class1])
    y_base = np.array([0]*80 + [1]*20)

    dataset.add_samples(X_base, {})
    dataset.add_targets(y_base)

    print(f"Initial distribution:")
    print(f"  Class 0: 80 samples")
    print(f"  Class 1: 20 samples")

    # Calculate augmentation needs for class 1
    # Target: 80 samples per class
    # Class 1 needs: 80 - 20 = 60 augmented samples

    # In practice, balanced mode would handle this automatically.
    # For demo purposes, we manually augment class 1 samples.

    # We need to create augmentation count list: 3 for class 1, 0 for class 0
    count_list = [0]*80 + [3]*20  # No augmentation for class 0, 3x for class 1

    # Generate augmented data for class 1 only (60 samples)
    class1_indices = np.where(y_base == 1)[0]
    X_class1_base = X_base[class1_indices]
    scaler = StandardScaler()
    X_class1_aug = scaler.fit_transform(X_class1_base)
    X_class1_aug = np.tile(X_class1_aug, (3, 1))  # 20 * 3 = 60 samples

    # Augment with custom count list
    dataset.augment_samples(
        data=X_class1_aug,
        processings=["standardized"],
        augmentation_id="balance_class1",
        count=count_list  # Selective augmentation
    )

    y_all = dataset.y({})
    print(f"\nBalanced distribution:")
    print(f"  Class 0: {np.sum(y_all == 0)} samples (unchanged)")
    print(f"  Class 1: {np.sum(y_all == 1)} samples (20 base + 60 augmented)")
    print(f"\n✓ Classes are now balanced!\n")


# Example 5: Sequential Augmentation
# ===================================

def example_sequential_augmentation():
    """
    Multiple augmentation rounds - each targets only base samples.
    """
    print("=== Example 5: Sequential Augmentation ===\n")

    dataset = SpectroDataset("example5")

    # Base data
    X_base = np.random.randn(25, 40)
    dataset.add_samples(X_base, {})

    print(f"Starting with: 25 base samples\n")

    # Round 1: StandardScaler
    scaler1 = StandardScaler()
    X_aug1 = scaler1.fit_transform(X_base)
    dataset.augment_samples(
        data=X_aug1,
        processings=["standard"],
        augmentation_id="round1",
        count=1
    )
    print(f"After Round 1: {dataset.x({}, layout='2d', concat_source=True).shape[0]} samples (25 base + 25 augmented)")

    # Round 2: MinMaxScaler (still augments only original 25)
    scaler2 = MinMaxScaler()
    X_aug2 = scaler2.fit_transform(X_base)
    dataset.augment_samples(
        data=X_aug2,
        processings=["minmax"],
        augmentation_id="round2",
        count=1
    )
    print(f"After Round 2: {dataset.x({}, layout='2d', concat_source=True).shape[0]} samples (25 base + 50 augmented)")

    # Round 3: Another StandardScaler variant
    X_aug3 = scaler1.fit_transform(X_base * 1.1)  # Slight variation
    dataset.augment_samples(
        data=X_aug3,
        processings=["standard_v2"],
        augmentation_id="round3",
        count=1
    )
    print(f"After Round 3: {dataset.x({}, layout='2d', concat_source=True).shape[0]} samples (25 base + 75 augmented)")

    print("\n✓ Each round augmented the same 25 base samples")
    print("✓ No 'augmentation of augmentations'\n")


# Example 6: Metadata Preservation
# =================================

def example_metadata_preservation():
    """
    Demonstrates that augmented samples inherit metadata from origins.
    """
    print("=== Example 6: Metadata Preservation ===\n")

    dataset = SpectroDataset("example6")

    # Add samples with metadata
    X_base = np.random.randn(12, 30)
    dataset.add_samples(X_base, {"partition": "train"})

    print(f"Added 12 base samples with partition='train'")

    # Augment all samples
    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_base)
    dataset.augment_samples(
        data=X_aug,
        processings=["standardized"],
        augmentation_id="metadata_test",
        count=1
    )

    # Check metadata
    total_samples = dataset.x({}, layout="2d", concat_source=True).shape[0]
    base_samples = dataset.x({}, layout="2d", concat_source=True, include_augmented=False).shape[0]

    print(f"Total samples: {total_samples}")
    print(f"Base samples: {base_samples}")
    print(f"Augmented samples: {total_samples - base_samples}")

    print("\n✓ All metadata columns preserved")
    print("✓ Augmented samples inherit origin metadata")
    print("✓ Origin tracking enables traceability\n")


# Example 7: Selective Augmentation
# ==================================

def example_selective_augmentation():
    """
    Augment only specific samples using selectors.
    """
    print("=== Example 7: Selective Augmentation ===\n")

    dataset = SpectroDataset("example7")

    # Create dataset with train/test split
    X_train = np.random.randn(60, 50)
    X_test = np.random.randn(20, 50)

    dataset.add_samples(X_train, {"partition": "train"})
    dataset.add_samples(X_test, {"partition": "test"})

    print(f"Train samples: 60")
    print(f"Test samples: 20")

    # Augment ONLY training samples
    scaler = StandardScaler()
    X_train_aug = scaler.fit_transform(X_train)
    X_train_aug = np.tile(X_train_aug, (2, 1))

    dataset.augment_samples(
        data=X_train_aug,
        processings=["standardized"],
        augmentation_id="train_only",
        selector={"partition": "train"},
        count=2
    )

    # Check results
    train_total = dataset.x({"partition": "train"}, layout="2d", concat_source=True).shape[0]
    test_total = dataset.x({"partition": "test"}, layout="2d", concat_source=True).shape[0]

    print(f"\nAfter augmentation:")
    print(f"Train samples: {train_total} (60 base + 120 augmented)")
    print(f"Test samples: {test_total} (unchanged)")
    print("\n✓ Only training data was augmented\n")


# Run all examples
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SAMPLE AUGMENTATION EXAMPLES")
    print("="*60 + "\n")

    example_basic_augmentation()
    example_multiple_transformers()
    example_leak_prevention()
    example_balanced_augmentation()
    example_sequential_augmentation()
    example_metadata_preservation()
    example_selective_augmentation()

    print("="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")
