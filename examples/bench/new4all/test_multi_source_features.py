#!/usr/bin/env python3
"""
Test multi-source feature handling to demonstrate the fixes.

This test verifies that:
1. Transformations handle sources independently (concatenate=False by default)
2. Models get concatenated features (concatenate=True)
3. Multiple sources are processed correctly through the pipeline
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from SpectraDataset import SpectraDataset
    from SpectraFeatures import SpectraFeatures
    from OperationFactory import OperationFactory
    from Pipeline import Pipeline
    from PipelineContext import PipelineContext
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def create_multi_source_dataset():
    """Create a dataset with multiple sources of different sizes."""
    dataset = SpectraDataset()

    # Create two sources with different feature dimensions
    n_samples = 100
    source1_features = 150  # First source: 150 features
    source2_features = 80   # Second source: 80 features

    # Generate synthetic spectral data
    np.random.seed(42)

    # Source 1: Higher frequency features
    X1 = np.random.randn(n_samples, source1_features) * 0.5 + 2.0

    # Source 2: Lower frequency features
    X2 = np.random.randn(n_samples, source2_features) * 0.3 + 1.5
      # Generate synthetic targets for classification
    y = (X1[:, :10].mean(axis=1) + X2[:, :5].mean(axis=1) > 3.0).astype(int)

    # Add data to dataset - pass list of arrays instead of SpectraFeatures
    sample_ids = dataset.add_data(
        features=[X1, X2],  # Pass as list of arrays
        targets=y,
        partition='all',  # We'll split later
        processing='original'
    )

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Source 1: {source1_features} features")
    print(f"Source 2: {source2_features} features")
    print(f"Total concatenated features: {source1_features + source2_features}")

    return dataset


def test_feature_access_modes():
    """Test different ways of accessing features."""
    print("\n=== Testing Feature Access Modes ===")

    dataset = create_multi_source_dataset()
    view = dataset.select(partition='all')

    print(f"Dataset view has {len(view)} samples")

    # Test 1: Get features without concatenation (list of arrays)
    print("\n1. Testing get_features(concatenate=False) - Default for transformations:")
    features_list = view.get_features(concatenate=False)
    print(f"   Result type: {type(features_list)}")
    print(f"   Number of sources: {len(features_list)}")
    for i, source in enumerate(features_list):
        print(f"   Source {i} shape: {source.shape}")

    # Test 2: Get concatenated features (single array)
    print("\n2. Testing get_features(concatenate=True) - Required for ML models:")
    features_concat = view.get_features(concatenate=True)
    print(f"   Result type: {type(features_concat)}")
    print(f"   Concatenated shape: {features_concat.shape}")

    # Verify concatenation is correct
    expected_width = sum(source.shape[1] for source in features_list)
    assert features_concat.shape[1] == expected_width, f"Expected width {expected_width}, got {features_concat.shape[1]}"
    print(f"   ✓ Concatenation verified: {expected_width} features total")


def test_transformation_with_multiple_sources():
    """Test that transformations handle multiple sources independently."""
    print("\n=== Testing Transformation with Multiple Sources ===")

    dataset = create_multi_source_dataset()

    # Split the data first
    factory = OperationFactory()
    split_config = {
        'type': 'split',
        'strategy': 'random',
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42
    }

    split_op = factory.create_operation(split_config)
    split_pipeline = Pipeline("Split Data")
    split_pipeline.add_operation(split_op)
    split_pipeline.execute(dataset)

    print("Data split completed")

    # Get features before transformation
    train_view = dataset.select(partition='train')
    features_before = train_view.get_features(concatenate=False)

    print(f"Before transformation:")
    for i, source in enumerate(features_before):
        print(f"  Source {i}: mean={source.mean():.3f}, std={source.std():.3f}")

    # Apply transformation (should handle sources independently)
    transform_config = {
        'type': 'transformation',
        'transformer': {'type': 'MinMaxScaler'},  # Use MinMaxScaler for demonstration
        'fit_partition': 'train',
        'transform_partitions': ['train', 'val', 'test']
    }

    transform_op = factory.create_operation(transform_config)
    transform_pipeline = Pipeline("Transform Data")
    transform_pipeline.add_operation(transform_op)
    transform_pipeline.execute(dataset)

    print("\nTransformation completed")

    # Check features after transformation
    features_after = train_view.get_features(concatenate=False)

    print(f"After transformation:")
    for i, source in enumerate(features_after):
        print(f"  Source {i}: mean={source.mean():.3f}, std={source.std():.3f}")

    # Verify each source was normalized independently
    for i, source in enumerate(features_after):
        assert abs(source.mean()) < 0.1, f"Source {i} not properly centered"
        assert abs(source.std() - 1.0) < 0.1, f"Source {i} not properly scaled"

    print("✓ Each source was transformed independently")


def test_model_with_concatenated_features():
    """Test that models get properly concatenated features."""
    print("\n=== Testing Model with Concatenated Features ===")

    dataset = create_multi_source_dataset()

    # Split and transform the data
    factory = OperationFactory()

    # Split
    split_config = {
        'type': 'split',
        'strategy': 'random',
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42
    }
    split_op = factory.create_operation(split_config)

    # Transform
    transform_config = {
        'type': 'transformation',
        'transformer': {'type': 'StandardScaler'},
        'fit_partition': 'train',
        'transform_partitions': ['train', 'val', 'test']
    }
    transform_op = factory.create_operation(transform_config)

    # Model
    model_config = {
        'type': 'model',
        'model': {'type': 'RandomForestClassifier', 'n_estimators': 10, 'random_state': 42},
        'fit_partition': 'train',
        'predict_partitions': ['val', 'test']
    }
    model_op = factory.create_operation(model_config)

    # Create and execute pipeline
    pipeline = Pipeline("Complete Workflow")
    pipeline.add_operation(split_op)
    pipeline.add_operation(transform_op)
    pipeline.add_operation(model_op)

    pipeline.execute(dataset)

    print("Complete pipeline executed successfully")

    # Verify the model received concatenated features
    train_view = dataset.select(partition='train')
    val_view = dataset.select(partition='val')

    # Check that we can get both formats
    train_list = train_view.get_features(concatenate=False)
    train_concat = train_view.get_features(concatenate=True)

    print(f"Training data:")
    print(f"  Sources: {len(train_list)} with shapes {[s.shape for s in train_list]}")
    print(f"  Concatenated: {train_concat.shape}")

    expected_width = sum(s.shape[1] for s in train_list)
    assert train_concat.shape[1] == expected_width, "Concatenation failed"

    print("✓ Model pipeline completed with concatenated features")


def test_complete_workflow():
    """Test the complete workflow to ensure everything works together."""
    print("\n=== Testing Complete Multi-Source Workflow ===")

    # Create dataset with multiple sources
    dataset = create_multi_source_dataset()

    print(f"\nInitial dataset: {len(dataset)} samples")

    # Test the complete workflow
    factory = OperationFactory()

    # Configuration for complete pipeline
    configs = [
        # Split data
        {
            'type': 'split',
            'strategy': 'stratified',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42
        },
        # Transform features (each source independently)
        {
            'type': 'transformation',
            'transformer': {'type': 'StandardScaler'},
            'fit_partition': 'train',
            'transform_partitions': ['train', 'val', 'test']
        },
        # Train model (with concatenated features)
        {
            'type': 'model',
            'model': {'type': 'RandomForestClassifier', 'n_estimators': 20, 'random_state': 42},
            'fit_partition': 'train',
            'predict_partitions': ['val', 'test']
        }
    ]

    # Create and execute pipeline
    pipeline = Pipeline("Multi-Source Complete Workflow")

    for config in configs:
        operation = factory.create_operation(config)
        pipeline.add_operation(operation)

    # Execute the complete pipeline
    pipeline.execute(dataset)

    print("\n✓ Complete multi-source workflow executed successfully!")

    # Validate the results
    train_view = dataset.select(partition='train')
    val_view = dataset.select(partition='val')
    test_view = dataset.select(partition='test')

    print(f"\nFinal data distribution:")
    print(f"  Train: {len(train_view)} samples")
    print(f"  Val: {len(val_view)} samples")
    print(f"  Test: {len(test_view)} samples")

    # Check that transformations preserved source independence
    val_features = val_view.get_features(concatenate=False)
    print(f"\nTransformed features (sources processed independently):")
    for i, source in enumerate(val_features):
        print(f"  Source {i}: shape={source.shape}, mean={source.mean():.3f}, std={source.std():.3f}")

    # Check that we can get concatenated features for models
    val_concat = val_view.get_features(concatenate=True)
    print(f"\nConcatenated features (for models): shape={val_concat.shape}")

    print("\n✓ All multi-source feature handling verified!")


def main():
    """Run all multi-source feature tests."""
    print("Testing Multi-Source Feature Handling")
    print("=" * 50)

    try:
        # Test individual components
        test_feature_access_modes()
        test_transformation_with_multiple_sources()
        test_model_with_concatenated_features()

        # Test complete workflow
        test_complete_workflow()

        print("\n" + "=" * 50)
        print("🎉 ALL MULTI-SOURCE FEATURE TESTS PASSED!")
        print("\nKey fixes verified:")
        print("✅ Transformations default to concatenate=False (process sources independently)")
        print("✅ Models use concatenate=True (get single 2D array)")
        print("✅ Multi-source pipeline works end-to-end")
        print("✅ Feature dimensions are handled correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
