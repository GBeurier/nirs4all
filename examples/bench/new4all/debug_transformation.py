#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from SplitOperation import SplitStrategy
from TransformationOperation import TransformationOperation
from OperationFactory import OperationFactory
from Pipeline import Pipeline

def debug_transformation_issue():
    """Debug the transformation issue after split"""
    np.random.seed(42)

    # Create a smaller dataset for easier debugging
    n_samples = 20
    n_features = 10

    # Create data
    X_original = np.random.random((n_samples, n_features))
    X_derivative = np.random.random((n_samples, n_features*2)) #np.gradient(X_original, axis=1)

    print("Original data shapes:")
    print(f"  X_original: {X_original.shape}")
    print(f"  X_derivative: {X_derivative.shape}")

    # Create dataset
    dataset = SpectraDataset(task_type='classification')
    sample_ids = dataset.add_data([X_original, X_derivative], partition="train")

    print(f"\nDataset info before split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features sources: {len(dataset.features.sources)}")
    print(f"  Source 0 shape: {dataset.features.sources[0].shape}")
    print(f"  Source 1 shape: {dataset.features.sources[1].shape}")

    # Apply split
    print("\n=== APPLYING SPLIT ===")
    split_op = SplitStrategy.train_val_test(train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, stratified=False)
    split_pipeline = Pipeline("Split Test")
    split_pipeline.add_operation(split_op)
    split_pipeline.execute(dataset)

    print(f"\nDataset info after split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features sources: {len(dataset.features.sources)}")
    print(f"  Source 0 shape: {dataset.features.sources[0].shape}")
    print(f"  Source 1 shape: {dataset.features.sources[1].shape}")

    # Check partitions before transformation
    print(f"\nPartitions after split:")
    for partition in ['train', 'val', 'test']:
        view = dataset.select(partition=partition)
        if len(view) > 0:
            print(f"  {partition}: {len(view)} samples, row indices: {view.row_indices}")
            features = view.get_features()
            if isinstance(features, list):
                print(f"    Features shape: {[f.shape for f in features]}")
            else:
                print(f"    Features shape: {features.shape}")

    # Apply transformation with preserve_original=False
    print(f"\n=== APPLYING TRANSFORMATION (preserve_original=False) ===")
    factory = OperationFactory()
    transform_config = {
        'type': 'transformation',
        'transformer': {'type': 'StandardScaler'},
        'fit_partition': 'train',
        'transform_partitions': ['train', 'val', 'test'],
        'preserve_original': False  # This should modify the sources
    }
    transform_op = factory.create_operation(transform_config)

    transform_pipeline = Pipeline("Transform Test")
    transform_pipeline.add_operation(transform_op)

    try:
        transform_pipeline.execute(dataset)
        print("✓ Transformation completed successfully")
        print(f"\nDataset info after transformation:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Features sources: {len(dataset.features.sources)}")
        for i, source in enumerate(dataset.features.sources):
            print(f"  Source {i} shape: {source.shape}")

        # Check partitions after transformation
        print(f"\nPartitions after transformation:")
        for partition in ['train', 'val', 'test']:
            view = dataset.select(partition=partition)
            if len(view) > 0:
                print(f"  {partition}: {len(view)} samples, row indices: {view.row_indices}")
                try:
                    features = view.get_features()
                    if isinstance(features, list):
                        print(f"    Features shape: {[f.shape for f in features]}")
                    else:
                        print(f"    Features shape: {features.shape}")
                except Exception as e:
                    print(f"    ❌ Error getting features: {e}")

    except Exception as e:
        print(f"❌ Transformation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transformation_issue()
