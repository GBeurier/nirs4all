"""
Test focused on transformation operation to verify source handling
"""
import sys
import os
import numpy as np

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from Pipeline import Pipeline
from OperationFactory import OperationFactory
from SplitOperation import SplitStrategy
from sklearn.preprocessing import StandardScaler


def create_multi_source_dataset():
    """Create a dataset with multiple sources to test the transformation"""
    np.random.seed(42)

    n_samples = 100
    n_wavelengths_nir = 50
    n_wavelengths_raman = 30

    # Create NIR spectra (source 1)
    X_nir = np.random.randn(n_samples, n_wavelengths_nir) + 2.0

    # Create Raman spectra (source 2)
    X_raman = np.random.randn(n_samples, n_wavelengths_raman) + 1.0

    # Create targets
    y = np.random.choice(['low', 'medium', 'high'], n_samples)

    # Create dataset with multiple sources
    dataset = SpectraDataset(task_type='classification')
    sample_ids = dataset.add_data([X_nir, X_raman], targets=y, partition="train")

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Number of sources: {len(dataset.features.sources)}")
    print(f"Source 1 (NIR) shape: {dataset.features.sources[0].shape}")
    print(f"Source 2 (Raman) shape: {dataset.features.sources[1].shape}")

    return dataset


def test_transformation_with_sources():
    """Test transformation operation with multiple sources"""
    print("=== Testing Transformation with Multiple Sources ===")

    # Create multi-source dataset
    dataset = create_multi_source_dataset()

    # Split the data first
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, stratified=True)
    split_pipeline = Pipeline("Split for Transform")
    split_pipeline.add_operation(split_op)
    split_pipeline.execute(dataset)

    print(f"\nAfter split:")
    train_view = dataset.select(partition="train")
    val_view = dataset.select(partition="val")
    test_view = dataset.select(partition="test")
    print(f"  train: {len(train_view)} samples")
    print(f"  val: {len(val_view)} samples")
    print(f"  test: {len(test_view)} samples")

    # Test getting features without concatenation (default for transformations)
    train_features = train_view.get_features(concatenate=False)
    print(f"\nTrain features (separate sources): {type(train_features)}")
    if isinstance(train_features, list):
        for i, source in enumerate(train_features):
            print(f"  Source {i}: {source.shape}")

    # Test getting features with concatenation (for models)
    train_features_concat = train_view.get_features(concatenate=True)
    print(f"\nTrain features (concatenated): {train_features_concat.shape}")

    # Apply transformation
    factory = OperationFactory()
    transform_config = {
        'type': 'transformation',
        'transformer': {'type': 'StandardScaler'},
        'fit_partition': 'train',
        'transform_partitions': ['train', 'val', 'test'],
        'preserve_original': False  # This should replace sources with transformed versions
    }

    transform_op = factory.create_operation(transform_config)
    transform_pipeline = Pipeline("Transform Test")
    transform_pipeline.add_operation(transform_op)

    print(f"\nBefore transformation:")
    print(f"  Number of sources: {len(dataset.features.sources)}")
    for i, source in enumerate(dataset.features.sources):
        print(f"  Source {i} shape: {source.shape}")
        sample_data = source[:5, :5]  # Show first 5x5 sample
        print(f"  Source {i} sample data:\n{sample_data}")

    transform_pipeline.execute(dataset)

    print(f"\nAfter transformation:")
    print(f"  Number of sources: {len(dataset.features.sources)}")
    for i, source in enumerate(dataset.features.sources):
        print(f"  Source {i} shape: {source.shape}")
        sample_data = source[:5, :5]  # Show first 5x5 sample
        print(f"  Source {i} sample data:\n{sample_data}")

    # Test that each source was transformed independently
    train_features_after = train_view.get_features(concatenate=False)
    print(f"\nTrain features after transformation (separate sources): {type(train_features_after)}")
    if isinstance(train_features_after, list):
        for i, source in enumerate(train_features_after):
            print(f"  Source {i}: {source.shape}")
            # Check that mean is close to 0 and std close to 1 (StandardScaler effect)
            print(f"  Source {i} mean: {np.mean(source, axis=0)[:5]}")  # First 5 features
            print(f"  Source {i} std: {np.std(source, axis=0)[:5]}")   # First 5 features

    print("\n✓ Transformation with multiple sources completed successfully!")
    return dataset


if __name__ == "__main__":
    test_transformation_with_sources()
