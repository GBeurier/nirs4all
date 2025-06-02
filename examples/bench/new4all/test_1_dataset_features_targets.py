#!/usr/bin/env python3
"""
Test 1: Dataset, Features and Targets

This test validates:
- Multi-source feature handling
- Multi-target support (classification, regression, mixed)
- Index management (sample, partition, group, branch, processing)
- Selection and filtering operations
- Target transformations and representations
"""

import sys
import os
import numpy as np

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from TargetManager import TargetManager
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_multi_source_multi_target_dataset():
    """Create a comprehensive test dataset with multiple sources and target types."""
    print("=== Creating Multi-Source Multi-Target Dataset ===")

    np.random.seed(42)
    n_samples = 150
      # Source 1: NIR spectroscopy (longer wavelength range)
    nir_features = 200
    X_nir = np.random.randn(n_samples, nir_features) * 0.5 + 2.0
    X_nir += np.sin(np.linspace(0, 4 * np.pi, nir_features)) * 0.3  # Add spectral pattern

    # Source 2: Raman spectroscopy (shorter wavelength range)
    raman_features = 120
    X_raman = np.random.randn(n_samples, raman_features) * 0.3 + 1.5
    X_raman += np.cos(np.linspace(0, 2 * np.pi, raman_features)) * 0.2  # Add spectral pattern

    # Source 3: FTIR spectroscopy (different range)
    ftir_features = 80
    X_ftir = np.random.randn(n_samples, ftir_features) * 0.4 + 1.8

    # Create multiple target types

    # Target 1: Classification (protein content level)
    protein_scores = X_nir[:, 50:60].mean(axis=1) + X_raman[:, 30:40].mean(axis=1)
    protein_levels = np.array(['low' if s < 2.0 else 'medium' if s < 3.0 else 'high'
                              for s in protein_scores])

    # Target 2: Regression (moisture content)
    moisture_content = (X_nir[:, 100:110].mean(axis=1) * 5.0 +
                        X_ftir[:, 20:30].mean(axis=1) * 3.0 +
                        np.random.normal(0, 0.5, n_samples) + 15.0)

    # Target 3: Binary classification (quality)
    quality_scores = (X_nir[:, 150:160].mean(axis=1) + X_raman[:, 80:90].mean(axis=1)) / 2
    quality = quality_scores > np.median(quality_scores)

    # Create dataset with automatic task detection
    dataset = SpectraDataset(task_type="auto")

    # Add data in different partitions with different indices
    train_size = 100
    val_size = 30
    test_size = 20

    # Add training data
    train_ids = dataset.add_data(
        features=[X_nir[:train_size], X_raman[:train_size], X_ftir[:train_size]],
        targets=protein_levels[:train_size],
        partition="train",
        group=1,
        branch=0,
        processing="raw"
    )

    # Add validation data
    val_ids = dataset.add_data(
        features=[X_nir[train_size:train_size+val_size],
                 X_raman[train_size:train_size+val_size],
                 X_ftir[train_size:train_size+val_size]],
        targets=protein_levels[train_size:train_size+val_size],
        partition="val",
        group=1,
        branch=0,
        processing="raw"
    )

    # Add test data
    test_ids = dataset.add_data(
        features=[X_nir[train_size+val_size:],
                 X_raman[train_size+val_size:],
                 X_ftir[train_size+val_size:]],
        targets=protein_levels[train_size+val_size:],
        partition="test",
        group=1,
        branch=0,
        processing="raw"
    )

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sources: {len(dataset.features.sources)}")
    print(f"Source shapes: {[s.shape for s in dataset.features.sources]}")
    print(f"Partitions: {sorted(dataset.indices['partition'].unique().to_list())}")
    print(f"Task type: {dataset.task_type}")
    print(f"Classes: {dataset.classes_}")
    print(f"Target distribution: {dict(zip(*np.unique(protein_levels, return_counts=True)))}")

    return dataset, {
        'moisture': moisture_content,
        'quality': quality,
        'protein_scores': protein_scores
    }


def test_multi_source_features():
    """Test multi-source feature handling."""
    print("\n=== Testing Multi-Source Features ===")

    dataset, extra_targets = create_multi_source_multi_target_dataset()

    # Test 1: Feature access modes
    print("\n1. Testing feature access modes...")

    train_view = dataset.select(partition="train")
    print(f"Train view: {len(train_view)} samples")

    # Test separate sources (default for transformations)
    features_separate = train_view.get_features(concatenate=False)
    print(f"Separate sources: {type(features_separate)}, {len(features_separate)} sources")
    for i, source in enumerate(features_separate):
        print(f"  Source {i}: {source.shape}")

    # Test concatenated features (for ML models)
    features_concat = train_view.get_features(concatenate=True)
    print(f"Concatenated: {features_concat.shape}")

    # Verify concatenation
    expected_width = sum(s.shape[1] for s in features_separate)
    assert features_concat.shape[1] == expected_width, f"Concatenation error: {expected_width} != {features_concat.shape[1]}"
    print("✓ Concatenation verified")

    # Test 2: Source-specific operations
    print("\n2. Testing source-specific operations...")

    # Test source means
    source_means = [np.mean(source, axis=0) for source in features_separate]
    print(f"Source means shapes: {[m.shape for m in source_means]}")

    # Test source statistics
    for i, source in enumerate(features_separate):
        print(f"Source {i} - min: {source.min():.3f}, max: {source.max():.3f}, mean: {source.mean():.3f}")


def test_index_management():
    """Test index management and filtering."""
    print("\n=== Testing Index Management ===")

    dataset, _ = create_multi_source_multi_target_dataset()

    # Test 1: Basic index information
    print("\n1. Index overview...")
    print(f"Total samples: {len(dataset)}")
    print(f"Unique samples: {len(dataset.indices['sample'].unique())}")
    print(f"Partitions: {dataset.indices['partition'].unique().to_list()}")
    print(f"Groups: {dataset.indices['group'].unique().to_list()}")
    print(f"Branches: {dataset.indices['branch'].unique().to_list()}")
    print(f"Processing: {dataset.indices['processing'].unique().to_list()}")

    # Test 2: Add data with different indices
    print("\n2. Adding data with different indices...")

    # Add some preprocessed data (same samples, different processing)
    train_view = dataset.select(partition="train")
    train_features = train_view.get_features(concatenate=False)

    # Simulate preprocessing: standardization
    preprocessed_features = []
    for source in train_features:
        std_source = (source - source.mean(axis=0)) / (source.std(axis=0) + 1e-8)
        preprocessed_features.append(std_source)    # Get original sample IDs for train partition
    original_sample_ids = train_view.sample_ids

    # Add preprocessed data with same sample IDs but different processing
    dataset.add_data(
        features=preprocessed_features,
        targets=None,  # No targets needed for same samples
        partition="train",
        group=1,
        branch=1,  # Different branch
        processing="standardized",
        origin=original_sample_ids  # Link to original samples
    )

    print(f"After adding preprocessed data: {len(dataset)} total rows")
    print(f"Unique samples: {len(dataset.indices['sample'].unique())}")
    print(f"Processing types: {dataset.indices['processing'].unique().to_list()}")
    print(f"Branches: {dataset.indices['branch'].unique().to_list()}")

    # Test 3: Complex filtering
    print("\n3. Testing complex filtering...")    # Filter by multiple criteria
    filtered_view = dataset.select(
        partition="train",
        processing="standardized",
        branch=1
    )
    print(f"Train + standardized + branch 1: {len(filtered_view)} samples")

    # Test that the origin links are correct
    original_samples = set(dataset.select(partition="train", processing="raw").sample_ids)
    filtered_view_indices = filtered_view._get_selection()
    origin_links = set(filtered_view_indices['origin'].to_list())

    # The origins should match the original sample IDs
    assert original_samples == origin_links, "Origin links should match original sample IDs"
    print("✓ Origin links verified")


def test_target_management():
    """Test comprehensive target management."""
    print("\n=== Testing Target Management ===")

    dataset, extra_targets = create_multi_source_multi_target_dataset()

    # Test 1: Multiple target representations
    print("\n1. Testing target representations...")

    sample_ids = dataset.select(partition="train").sample_ids[:10]

    # Original targets
    original_targets = dataset.get_targets(sample_ids, "original")
    print(f"Original targets: {original_targets}")

    # Classification targets (encoded)
    class_targets = dataset.get_targets(sample_ids, "classification")
    print(f"Classification targets: {class_targets}")

    # Regression targets (numeric)
    reg_targets = dataset.get_targets(sample_ids, "regression")
    print(f"Regression targets: {reg_targets}")

    # Test 2: Adding different target types
    print("\n2. Adding multiple target types...")

    # Add moisture content as regression targets
    all_sample_ids = dataset.indices['sample'].unique().to_list()

    # Create a secondary target manager for regression
    reg_dataset = SpectraDataset(task_type="regression")

    # Get original features for demonstration
    original_view = dataset.select(processing="raw")
    original_features = original_view.get_features(concatenate=False)
    original_sample_ids = original_view.sample_ids

    # Add data with regression targets
    moisture_values = extra_targets['moisture'][:len(original_sample_ids)]
    reg_sample_ids = reg_dataset.add_data(
        features=original_features,
        targets=moisture_values,
        partition="all"
    )

    print(f"Regression dataset task type: {reg_dataset.task_type}")
    print(f"Moisture range: {moisture_values.min():.2f} - {moisture_values.max():.2f}")    # Test 3: Target transformations
    print("\n3. Testing target transformations...")

    # Apply transformation to regression targets
    transformers = [StandardScaler()]
    transformed_targets = reg_dataset.fit_transform_targets(
        sample_ids=reg_sample_ids[:50],
        transformers=transformers,
        representation="regression",
        transformer_key="moisture_scaler"
    )

    print(f"Original range: {reg_dataset.get_targets(reg_sample_ids[:5], 'regression')}")
    print(f"Transformed range: {transformed_targets[:5]}")    # Test inverse transformation
    test_predictions = np.array([0.5, -0.3, 1.2, -0.8, 0.9])
    inverse_predictions = reg_dataset.inverse_transform_predictions(
        predictions=test_predictions,
        representation="regression",
        transformer_key="moisture_scaler",
        to_original=True
    )
    print(f"Inverse transformed predictions: {inverse_predictions}")


def test_dataset_views_and_selection():
    """Test dataset views and selection capabilities."""
    print("\n=== Testing Dataset Views and Selection ===")

    dataset, _ = create_multi_source_multi_target_dataset()

    # Add some augmented data for more complex testing
    train_view = dataset.select(partition="train")
    train_features = train_view.get_features(concatenate=False)
    train_targets = train_view.get_targets("original")

    # Add augmented samples (new sample IDs)
    augmented_ids = dataset.add_data(
        features=train_features,
        targets=train_targets,
        partition="train_aug",
        group=2,
        branch=0,
        processing="augmented"
    )

    print(f"Added {len(augmented_ids)} augmented samples")
    print(f"Total dataset size: {len(dataset)}")

    # Test 1: Partition-based selection
    print("\n1. Testing partition-based selection...")

    partitions = dataset.indices['partition'].unique().to_list()
    for partition in partitions:
        view = dataset.select(partition=partition)
        print(f"Partition '{partition}': {len(view)} samples")

    # Test 2: Group-based selection
    print("\n2. Testing group-based selection...")

    groups = dataset.indices['group'].unique().to_list()
    for group in groups:
        view = dataset.select(group=group)
        print(f"Group {group}: {len(view)} samples")

    # Test 3: Processing-based selection
    print("\n3. Testing processing-based selection...")

    processing_types = dataset.indices['processing'].unique().to_list()
    for proc in processing_types:
        view = dataset.select(processing=proc)
        print(f"Processing '{proc}': {len(view)} samples")

    # Test 4: Combined selection
    print("\n4. Testing combined selection...")

    # Complex selection
    complex_view = dataset.select(group=1, processing="raw")
    print(f"Group 1 + raw processing: {len(complex_view)} samples")    # Verify selection consistency
    manual_filter = dataset.indices.filter(
        (dataset.indices['group'] == 1) & (dataset.indices['processing'] == "raw")
    )
    assert len(complex_view) == len(manual_filter), "Selection should match manual filter"
    print("✓ Selection consistency verified")

    # Test 5: View operations
    print("\n5. Testing view operations...")

    # Get features from view
    view_features = complex_view.get_features(concatenate=True)
    print(f"View features shape: {view_features.shape}")

    # Get targets from view
    view_targets = complex_view.get_targets("classification")
    print(f"View targets: {len(view_targets)} targets")

    # Verify view integrity
    assert view_features.shape[0] == len(view_targets), "Features and targets should have same length"
    print("✓ View integrity verified")


def main():
    """Run all dataset, features, and targets tests."""
    print("=" * 60)
    print("TEST 1: DATASET, FEATURES AND TARGETS")
    print("=" * 60)

    try:
        # Core tests
        test_multi_source_features()
        test_index_management()
        test_target_management()
        test_dataset_views_and_selection()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
