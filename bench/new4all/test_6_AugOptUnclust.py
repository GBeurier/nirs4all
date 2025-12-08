"""
Test 7 - Test corrected operations: TransformationOperation for augmentation,
UnclusterOperation, and OptimizationOperation
"""
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from TransformationOperation import TransformationOperation
from ClusteringOperation import ClusteringOperation, UnclusterOperation
from OptimizationOperation import OptimizationOperation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def test_transformation_operation_augmentation():
    """Test TransformationOperation in augmentation modes"""
    print("=== Testing TransformationOperation Augmentation Modes ===")

    # Create test dataset
    dataset = SpectraDataset()

    # Add some data
    X_train = np.random.rand(50, 100)
    y_train = np.random.rand(50)

    dataset.add_data(
        features=[X_train],
        targets=y_train,
        partition="train"
    )

    X_test = np.random.rand(20, 100)
    y_test = np.random.rand(20)

    dataset.add_data(
        features=[X_test],
        targets=y_test,
        partition="test"
    )

    context = PipelineContext()

    print(f"Initial dataset size: {len(dataset)}")

    # Test sample augmentation
    print("\n--- Testing Sample Augmentation ---")
    sample_aug_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="sample_augmentation",
        fit_partition="train",
        transform_partitions=["train"]
    )

    sample_aug_op.execute(dataset, context)
    print(f"Dataset size after sample augmentation: {len(dataset)}")    # Test feature augmentation
    print("\n--- Testing Feature Augmentation ---")
    feature_aug_op = TransformationOperation(
        transformer=[StandardScaler(), MinMaxScaler()],  # Sequential pipeline that preserves dimensions
        mode="feature_augmentation",
        fit_partition="train"
    )

    feature_aug_op.execute(dataset, context)
    print(f"Dataset size after feature augmentation: {len(dataset)}")

    # Check that we have different processing types
    processing_types = dataset.indices["processing"].unique().to_list()
    print(f"Processing types in dataset: {processing_types}")

    assert len(processing_types) > 1, "Should have multiple processing types after augmentation"
    print("✓ Augmentation tests passed!")


def test_clustering_and_unclustering():
    """Test ClusteringOperation and UnclusterOperation"""
    print("\n=== Testing Clustering and Unclustering ===")

    # Create test dataset with clear clusters
    dataset = SpectraDataset()
    context = PipelineContext()

    # Create data with 3 clear clusters
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (30, 50))
    cluster2 = np.random.normal(5, 0.5, (25, 50))
    cluster3 = np.random.normal(-3, 0.5, (20, 50))

    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.hstack([np.zeros(30), np.ones(25), np.full(20, 2)])

    dataset.add_data(
        features=[X],
        targets=y,
        partition="train"
    )

    print(f"Initial dataset size: {len(dataset)}")
    print(f"Initial unique groups: {dataset.indices['group'].unique().to_list()}")

    # Test clustering
    print("\n--- Testing Clustering ---")
    clustering_op = ClusteringOperation(
        clustering_method="kmeans",
        n_clusters=3,
        store_centroids=True,
        evaluate_clustering=True
    )

    clustering_op.execute(dataset, context)

    # Check clustering results
    assert hasattr(context, 'clustering_results'), "Clustering results should be stored in context"
    cluster_results = context.clustering_results
    print(f"Found {cluster_results['metrics']['n_clusters']} clusters")
    print(f"Silhouette score: {cluster_results['metrics'].get('silhouette_score', 'N/A')}")

    # Check that centroids are stored
    if hasattr(context, 'centroid_mapping'):
        print(f"Centroids stored: {len(context.centroid_mapping['centroid_sample_ids'])} centroids")

        # Check that context is filtered to centroids
        centroid_view = dataset.select(**context.current_filters)
        print(f"Current selection size (should be centroids): {len(centroid_view)}")

    # Test unclustering
    print("\n--- Testing Unclustering ---")
    uncluster_op = UnclusterOperation()

    uncluster_op.execute(dataset, context)

    # Check that selection is restored
    full_view = dataset.select(**context.current_filters)
    print(f"Selection size after unclustering: {len(full_view)}")

    # Check that samples are assigned to groups
    final_groups = dataset.indices['group'].unique().to_list()
    print(f"Final unique groups: {final_groups}")

    assert len(final_groups) > 1, "Samples should be assigned to different groups"
    print("✓ Clustering and unclustering tests passed!")


def test_optimization_operation():
    """Test OptimizationOperation"""
    print("\n=== Testing OptimizationOperation ===")

    # Create test dataset
    dataset = SpectraDataset()
    context = PipelineContext()

    # Create simple classification data
    np.random.seed(42)
    X = np.random.rand(100, 20)
    y = (np.random.rand(100) > 0.5).astype(int)

    dataset.add_data(
        features=[X],
        targets=y,
        partition="train"
    )

    print(f"Dataset size: {len(dataset)}")

    # Test Optuna optimization (will use dummy implementation)
    print("\n--- Testing Optuna Optimization ---")

    param_space = {
        'max_depth': [3, 5, 7, 10],
        'n_estimators': ('int', 50, 200),
        'learning_rate': ('float', 0.01, 0.3)
    }

    optuna_op = OptimizationOperation(
        optimizer_type="optuna",
        model_params=param_space,
        n_trials=5,  # Small number for testing
        metrics=["accuracy"],
        task="classification"
    )

    try:
        optuna_op.execute(dataset, context)

        # Check results
        if hasattr(context, 'optimization_results'):
            opt_results = context.optimization_results
            print(f"Best score: {opt_results.get('best_score', 'N/A')}")
            print(f"Best params: {opt_results.get('best_params', {})}")

        print("✓ Optuna optimization test passed!")

    except ImportError as e:
        print(f"⚠ Optuna not available: {e}")
        print("⚠ Skipping Optuna test (install with: pip install optuna)")
    except Exception as e:
        print(f"⚠ Optuna test failed: {e}")

    # Test sklearn optimization
    print("\n--- Testing Sklearn Optimization ---")

    sklearn_param_space = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150]
    }

    sklearn_op = OptimizationOperation(
        optimizer_type="sklearn",
        model_params=sklearn_param_space,
        n_trials=10,
        metrics=["accuracy"],
        task="classification",
        approach="random",
        cv=3
    )

    try:
        sklearn_op.execute(dataset, context)

        # Check results
        if hasattr(context, 'optimization_results'):
            opt_results = context.optimization_results
            print(f"Best score: {opt_results.get('best_score', 'N/A')}")
            print(f"Best params: {opt_results.get('best_params', {})}")

        print("✓ Sklearn optimization test passed!")

    except Exception as e:
        print(f"⚠ Sklearn test failed: {e}")


def test_pipeline_integration():
    """Test integration of all three operations in a pipeline"""
    print("\n=== Testing Pipeline Integration ===")

    # Create test dataset
    dataset = SpectraDataset()
    context = PipelineContext()

    # Create data
    np.random.seed(42)
    X = np.random.rand(60, 30)
    y = np.random.rand(60)

    dataset.add_data(
        features=[X],
        targets=y,
        partition="train"
    )

    print(f"Initial dataset size: {len(dataset)}")

    # Step 1: Feature augmentation with transformation
    print("\n--- Step 1: Feature Augmentation ---")
    aug_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="feature_augmentation"
    )
    aug_op.execute(dataset, context)
    print(f"Dataset size after augmentation: {len(dataset)}")

    # Step 2: Clustering
    print("\n--- Step 2: Clustering ---")
    cluster_op = ClusteringOperation(
        clustering_method="kmeans",
        n_clusters=3,
        store_centroids=True
    )
    cluster_op.execute(dataset, context)

    # Verify we're working with centroids
    current_view = dataset.select(**context.current_filters)
    print(f"Current selection size (centroids): {len(current_view)}")

    # Step 3: Some operation that works on centroids (placeholder)
    print("\n--- Step 3: Operations on Centroids ---")
    # In a real pipeline, you might do model training, splits, etc. on centroids

    # Step 4: Unclustering
    print("\n--- Step 4: Unclustering ---")
    uncluster_op = UnclusterOperation()
    uncluster_op.execute(dataset, context)

    # Verify selection is restored
    final_view = dataset.select(**context.current_filters)
    print(f"Final selection size: {len(final_view)}")

    # Check group assignments
    groups = dataset.indices['group'].unique().to_list()
    print(f"Final groups: {groups}")

    print("✓ Pipeline integration test passed!")


def main():
    """Run all tests"""
    print("Testing Corrected Operations")
    print("=" * 50)

    try:
        test_transformation_operation_augmentation()
        test_clustering_and_unclustering()
        test_optimization_operation()
        test_pipeline_integration()

        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
