"""
Example: Using Metadata with SpectroDataset

This example demonstrates how to:
1. Load a dataset with metadata
2. Access and filter metadata
3. Convert metadata to numeric format
4. Use metadata in custom operations
"""

import numpy as np
import pandas as pd
from pathlib import Path

from nirs4all.dataset.dataset_config import DatasetConfigs


def example_load_dataset_with_metadata():
    """Load a dataset with metadata from folder or config."""

    # Method 1: Load from folder (auto-detects M_train.csv, M_test.csv, etc.)
    dataset_folder = "path/to/your/dataset"
    configs = DatasetConfigs(dataset_folder)
    dataset = configs.get_dataset_at(0)

    # Method 2: Load from explicit config
    config = {
        'train_x': 'path/to/X_train.csv',
        'train_y': 'path/to/Y_train.csv',
        'train_group': 'path/to/M_train.csv',  # Metadata for training set
        'test_x': 'path/to/X_test.csv',
        'test_y': 'path/to/Y_test.csv',
        'test_group': 'path/to/M_test.csv',   # Metadata for test set
    }
    configs = DatasetConfigs(config)
    dataset = configs.get_dataset_at(0)

    print(f"Dataset loaded: {dataset.name}")
    print(f"Samples: {dataset.num_samples}")
    print(f"Metadata columns: {dataset.metadata_columns}")

    return dataset


def example_access_metadata(dataset):
    """Access and filter metadata."""

    print("\n" + "="*60)
    print("ACCESSING METADATA")
    print("="*60)

    # Get all metadata as DataFrame
    all_metadata = dataset.metadata()
    print(f"\nAll metadata shape: {all_metadata.shape}")
    print(all_metadata.head())

    # Get metadata for specific partition
    train_metadata = dataset.metadata(selector={"partition": "train"})
    print(f"\nTrain metadata shape: {train_metadata.shape}")
    print(train_metadata.head())

    test_metadata = dataset.metadata(selector={"partition": "test"})
    print(f"\nTest metadata shape: {test_metadata.shape}")

    # Get specific columns only
    batch_location = dataset.metadata(columns=['batch', 'location'])
    print(f"\nBatch and location only:")
    print(batch_location.head())

    # Get a single column as numpy array
    batch_numbers = dataset.metadata_column('batch')
    print(f"\nBatch numbers: {batch_numbers}")

    # Get single column for specific partition
    train_batches = dataset.metadata_column('batch', selector={"partition": "train"})
    print(f"Train batches: {train_batches}")


def example_numeric_encoding(dataset):
    """Convert categorical metadata to numeric format."""

    print("\n" + "="*60)
    print("NUMERIC ENCODING")
    print("="*60)

    # Label encoding (for categorical data)
    location_encoded, encoding_info = dataset.metadata_numeric(
        'location',
        method='label'
    )
    print(f"\nLabel encoded locations shape: {location_encoded.shape}")
    print(f"Encoding info: {encoding_info}")
    print(f"Encoded values: {location_encoded[:10]}")

    # One-hot encoding
    location_onehot, onehot_info = dataset.metadata_numeric(
        'location',
        method='onehot'
    )
    print(f"\nOne-hot encoded locations shape: {location_onehot.shape}")
    print(f"Classes: {onehot_info['classes']}")
    print(f"First 5 rows:")
    print(location_onehot[:5])

    # Encoding is cached - subsequent calls return same encoding
    location_encoded2, _ = dataset.metadata_numeric('location', method='label')
    assert np.array_equal(location_encoded, location_encoded2)
    print("\n✓ Encoding consistency verified (cached)")


def example_filter_by_metadata():
    """Use metadata to filter or group samples."""

    print("\n" + "="*60)
    print("FILTERING BY METADATA")
    print("="*60)

    # Create example dataset with metadata
    from nirs4all.dataset.dataset import SpectroDataset

    dataset = SpectroDataset(name="example_with_metadata")

    # Add samples
    X_train = np.random.rand(20, 10)
    y_train = np.random.rand(20)
    dataset.add_samples(X_train, {"partition": "train"})
    dataset.add_targets(y_train)

    # Add metadata
    metadata_df = pd.DataFrame({
        'batch': [1]*10 + [2]*10,
        'instrument': ['A']*5 + ['B']*5 + ['A']*5 + ['B']*5,
        'quality': np.random.rand(20)
    })
    dataset.add_metadata(metadata_df)

    # Get all training samples
    X_all_train = dataset.x({"partition": "train"})
    print(f"All training samples: {X_all_train.shape}")

    # Get metadata for training samples
    train_meta = dataset.metadata(selector={"partition": "train"})
    print(f"\nTraining metadata:")
    print(train_meta)

    # Manual filtering based on metadata
    # (Note: Direct metadata-based filtering in selector not yet implemented)
    batch_col = dataset.metadata_column('batch', selector={"partition": "train"})
    batch_1_mask = batch_col == 1
    print(f"\nBatch 1 samples: {np.sum(batch_1_mask)}")
    print(f"Batch 2 samples: {np.sum(~batch_1_mask)}")


def example_metadata_operations(dataset):
    """Demonstrate metadata modification operations."""

    print("\n" + "="*60)
    print("METADATA OPERATIONS")
    print("="*60)

    # Update existing metadata
    # Update first 5 train samples' location to 'Updated'
    dataset.update_metadata(
        column='location',
        values=['Updated']*5,
        selector={"partition": "train"}
    )
    print("✓ Updated first 5 training samples' location")

    # Add new metadata column
    quality_scores = np.random.rand(dataset.num_samples)
    dataset.add_metadata_column('quality_score', quality_scores)
    print(f"✓ Added 'quality_score' column")
    print(f"Metadata columns now: {dataset.metadata_columns}")


def example_use_metadata_in_pipeline():
    """Use metadata as additional features in a pipeline."""

    print("\n" + "="*60)
    print("USING METADATA IN PIPELINES")
    print("="*60)

    from nirs4all.dataset.dataset import SpectroDataset
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    # Create example dataset
    dataset = SpectroDataset(name="example_with_metadata")

    # Add samples (simulated spectra)
    X_train = np.random.rand(100, 50)
    y_train = np.random.rand(100) * 10 + 5  # Target values
    dataset.add_samples(X_train, {"partition": "train"})
    dataset.add_targets(y_train)

    # Add metadata
    metadata_df = pd.DataFrame({
        'instrument_id': np.random.choice(['A', 'B', 'C'], 100),
        'temperature': np.random.uniform(20, 25, 100),
        'humidity': np.random.uniform(40, 60, 100)
    })
    dataset.add_metadata(metadata_df)

    # Method 1: Use spectral data only
    X_spectra = dataset.x({"partition": "train"})
    y = dataset.y({"partition": "train"})

    model1 = RandomForestRegressor(n_estimators=50, random_state=42)
    scores1 = cross_val_score(model1, X_spectra, y, cv=5, scoring='r2')
    print(f"\nModel with spectra only - R²: {scores1.mean():.3f} ± {scores1.std():.3f}")

    # Method 2: Combine spectral data with numeric metadata
    # Convert categorical metadata to numeric
    instrument_encoded, _ = dataset.metadata_numeric('instrument_id', method='onehot')
    temperature = dataset.metadata_column('temperature', selector={"partition": "train"}).reshape(-1, 1)
    humidity = dataset.metadata_column('humidity', selector={"partition": "train"}).reshape(-1, 1)

    # Combine features
    X_combined = np.hstack([X_spectra, instrument_encoded, temperature, humidity])

    model2 = RandomForestRegressor(n_estimators=50, random_state=42)
    scores2 = cross_val_score(model2, X_combined, y, cv=5, scoring='r2')
    print(f"Model with spectra + metadata - R²: {scores2.mean():.3f} ± {scores2.std():.3f}")

    print(f"\nFeature counts:")
    print(f"  Spectral features: {X_spectra.shape[1]}")
    print(f"  Metadata features: {X_combined.shape[1] - X_spectra.shape[1]}")
    print(f"  Total features: {X_combined.shape[1]}")


def main():
    """Run all examples."""

    print("="*60)
    print("METADATA USAGE EXAMPLES")
    print("="*60)

    # Note: These examples assume you have actual data files
    # For demonstration, we'll use the programmatic creation examples

    # Example 1: Filtering by metadata
    example_filter_by_metadata()

    # Example 2: Using metadata in pipelines
    example_use_metadata_in_pipeline()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)

    # If you have actual dataset files, you can use:
    # dataset = example_load_dataset_with_metadata()
    # example_access_metadata(dataset)
    # example_numeric_encoding(dataset)
    # example_metadata_operations(dataset)


if __name__ == "__main__":
    main()
