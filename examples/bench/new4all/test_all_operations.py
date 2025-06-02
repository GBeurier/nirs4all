#!/usr/bin/env python3
"""
Comprehensive test of all fixed operations with multi-source features
"""

import numpy as np
from SpectraDataset import SpectraDataset
from Pipeline import Pipeline
from PipelineContext import PipelineContext
from SplitOperation import SplitOperation
from TransformationOperation import TransformationOperation
from ModelOperation import ModelOperation
from ClusteringOperation import ClusteringOperation
from MergeSourcesOperation import MergeSourcesOperation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def create_test_dataset():
    """Create a simple test dataset with multi-source data"""
    dataset = SpectraDataset()

    # Add multi-source data for testing
    nir_data = np.random.randn(50, 30)  # 50 samples, 30 features
    raman_data = np.random.randn(50, 20)  # 50 samples, 20 features
    targets = np.random.choice([0, 1], size=50)

    dataset.add_data([nir_data, raman_data], targets=targets, partition="train")

    return dataset


def create_multi_source_dataset():
    """Create test dataset with multiple sources for comprehensive testing"""
    print("=== Creating Multi-Source Test Dataset ===")

    # Create dataset
    dataset = SpectraDataset()

    # Add multi-source data (NIR + Raman)
    nir_data = np.random.randn(100, 50)  # NIR spectroscopy (100 samples, 50 wavelengths)
    raman_data = np.random.randn(100, 30)  # Raman spectroscopy (100 samples, 30 wavelengths)
      # Create targets (binary classification)
    targets = np.random.choice([0, 1], size=100)

    # Add data to dataset
    sample_ids = dataset.add_data(
        [nir_data, raman_data],
        targets=targets
    )

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Number of sources: {len(dataset.features.sources)}")
    print(f"Source 0 (NIR) shape: {dataset.features.sources[0].shape}")
    print(f"Source 1 (Raman) shape: {dataset.features.sources[1].shape}")

    # Show target distribution
    unique, counts = np.unique(targets, return_counts=True)
    print(f"Target distribution: {counts}")

    return dataset


def test_complete_pipeline():
    """Test complete multi-source pipeline"""
    try:
        dataset = create_multi_source_dataset()
        context = PipelineContext()

        # Create pipeline
        pipeline = Pipeline()

        # 1. Split data
        pipeline.add_operation(SplitOperation(
            strategy="stratified",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        ))

        # 2. Apply transformation (each source independently)
        pipeline.add_operation(TransformationOperation(
            transformer=StandardScaler(),
            fit_partition="train"
        ))


        # 4. Clustering analysis
        pipeline.add_operation(ClusteringOperation(
            n_clusters=3,
            store_centroids=True,
            evaluate_clustering=True
        ))

        # 5. Merge sources for traditional ML models
        pipeline.add_operation(MergeSourcesOperation(
            merge_strategy="concatenate"
        ))

        # 6. Train a model
        pipeline.add_operation(ModelOperation(
            model=LogisticRegression(random_state=42, max_iter=1000),
            train_on="train",
            predict_on=["val", "test"]
        ))

        # Execute pipeline
        try:
            pipeline.execute(dataset)
            print("✓ Complete pipeline executed successfully!")

            # Show results
            print("\nFinal dataset statistics:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Number of sources: {len(dataset.features.sources)}")
              # Check if transformations were applied
            train_view = dataset.select(partition="train")
            train_features = train_view.get_features(concatenate=False)
            for i, src in enumerate(train_features):
                src_mean = np.mean(src, axis=0)
                src_std = np.std(src, axis=0)
                # Format numpy arrays properly
                mean_str = ", ".join([f"{x:.3f}" for x in src_mean[:3]])
                std_str = ", ".join([f"{x:.3f}" for x in src_std[:3]])
                print(f"  Source {i} - Mean: [{mean_str}]..., Std: [{std_str}]...")

            return True

        except Exception as e:
            print(f"✗ Pipeline execution failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Complete pipeline test failed: {e}")
        return False


def test_individual_operations():
    """Test each operation individually."""
    results = {}

    print("\n=== Testing Individual Operations ===")

    # Test 1: Split Operation
    try:
        dataset = create_test_dataset()
        context = PipelineContext()
        split_op = SplitOperation(strategy="stratified", train_size=0.8, val_size=0.2)
        split_op.execute(dataset, context)
        results["SplitOperation"] = "✓ Pass"
        print("✓ SplitOperation: Pass")
    except Exception as e:
        results["SplitOperation"] = f"✗ Fail: {e}"
        print(f"✗ SplitOperation: {e}")
      # Test 2: Transformation Operation
    try:
        dataset = create_test_dataset()
        context = PipelineContext()
        # Don't add extra data - use the dataset as-is
        transform_op = TransformationOperation(
            transformer=StandardScaler()
        )
        transform_op.execute(dataset, context)
        results["TransformationOperation"] = "✓ Pass"
        print("✓ TransformationOperation: Pass")
    except Exception as e:
        results["TransformationOperation"] = f"✗ Fail: {e}"
        print(f"✗ TransformationOperation: {e}")

    # Test 3: Clustering Operation
    try:
        dataset = create_test_dataset()
        context = PipelineContext()
        cluster_op = ClusteringOperation(n_clusters=3, store_centroids=True)
        cluster_op.execute(dataset, context)
        results["ClusteringOperation"] = "✓ Pass"
        print("✓ ClusteringOperation: Pass")
    except Exception as e:
        results["ClusteringOperation"] = f"✗ Fail: {e}"
        print(f"✗ ClusteringOperation: {e}")

    # Test 4: Model Operation
    try:
        dataset = create_test_dataset()
        context = PipelineContext()
        # Split first
        split_op = SplitOperation(strategy="stratified", train_size=0.8, val_size=0.2)
        split_op.execute(dataset, context)

        model_op = ModelOperation(
            model=LogisticRegression(random_state=42, max_iter=1000),
            train_on="train",
            predict_on=["val"]
        )
        model_op.execute(dataset, context)
        results["ModelOperation"] = "✓ Pass"
        print("✓ ModelOperation: Pass")
    except Exception as e:
        results["ModelOperation"] = f"✗ Fail: {e}"
        print(f"✗ ModelOperation: {e}")


    # Test 6: Merge Sources Operation
    try:
        dataset = create_test_dataset()
        context = PipelineContext()
        merge_op = MergeSourcesOperation(merge_strategy="concatenate")
        merge_op.execute(dataset, context)
        results["MergeSourcesOperation"] = "✓ Pass"
        print("✓ MergeSourcesOperation: Pass")
    except Exception as e:
        results["MergeSourcesOperation"] = f"✗ Fail: {e}"
        print(f"✗ MergeSourcesOperation: {e}")

    print("\n=== Individual Operation Results ===")
    for op_name, result in results.items():
        print(f"{op_name}: {result}")

    return results


def main():
    """Main test function."""
    print("Starting comprehensive operation tests...\n")

    # Test individual operations
    individual_results = test_individual_operations()

    # Test complete pipeline
    pipeline_success = test_complete_pipeline()

    # Summary
    print("\n" + "=" * 50)
    print("FINAL TEST SUMMARY")
    print("=" * 50)

    passed_individual = sum(1 for result in individual_results.values() if result.startswith("✓"))
    total_individual = len(individual_results)

    print(f"Individual Operations: {passed_individual}/{total_individual} passed")
    for op_name, result in individual_results.items():
        print(f"  {op_name}: {result}")

    print(f"Complete Pipeline: {'✓ Pass' if pipeline_success else '✗ Fail'}")

    return passed_individual == total_individual and pipeline_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
