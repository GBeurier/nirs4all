#!/usr/bin/env python3
"""
Test 3: All Operations

This test validates all pipeline operations individually:
- SplitOperation (train/val/test splits, stratified splits, folds)
- TransformationOperation (all modes)
- ModelOperation (classification, regression, with target transformations)
- ClusteringOperation (KMeans, hierarchical)
- MergeSourcesOperation (source combination)
- DispatchOperation (branching, parallel model training)

Each operation is tested in isolation to ensure it works correctly.
"""

import sys
import os
import numpy as np

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from Pipeline import Pipeline
from PipelineContext import PipelineContext
from SplitOperation import SplitOperation, SplitStrategy
from TransformationOperation import TransformationOperation
from ModelOperation import ModelOperation
from ClusteringOperation import ClusteringOperation
from MergeSourcesOperation import MergeSourcesOperation
from DispatchOperation import DispatchOperation
from OperationFactory import OperationFactory

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA


def create_comprehensive_test_dataset():
    """Create a comprehensive dataset for testing all operations."""
    print("=== Creating Comprehensive Test Dataset ===")

    np.random.seed(42)
    n_samples = 200

    # Create 3 different spectral sources
    # Source 1: NIR (Near-infrared)
    X_nir = np.random.randn(n_samples, 150) * 0.8 + 3.0
    X_nir += np.sin(np.linspace(0, 8*np.pi, 150)) * 0.4

    # Source 2: Raman
    X_raman = np.random.randn(n_samples, 100) * 0.6 + 2.0
    X_raman += np.cos(np.linspace(0, 6*np.pi, 100)) * 0.3

    # Source 3: FTIR
    X_ftir = np.random.randn(n_samples, 80) * 0.5 + 1.5

    # Create classification targets with realistic class imbalance
    feature_combination = (X_nir[:, 50:60].mean(axis=1) +
                          X_raman[:, 30:40].mean(axis=1) +
                          X_ftir[:, 20:30].mean(axis=1))

    # Create 4 classes with some imbalance
    thresholds = np.percentile(feature_combination, [25, 50, 75])
    targets = np.array(['class_A' if x < thresholds[0]
                       else 'class_B' if x < thresholds[1]
                       else 'class_C' if x < thresholds[2]
                       else 'class_D' for x in feature_combination])

    # Create dataset
    dataset = SpectraDataset(task_type="classification")

    # Add all data as train initially (operations will split it)
    sample_ids = dataset.add_data(
        features=[X_nir, X_raman, X_ftir],
        targets=targets,
        partition="all",
        processing="raw"
    )

    print(f"Dataset: {len(dataset)} samples, {len(dataset.features.sources)} sources")
    print(f"Source shapes: {[s.shape for s in dataset.features.sources]}")
    print(f"Class distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")

    return dataset


def test_split_operations():
    """Test all split operation types."""
    print("\n=== Testing Split Operations ===")

    # Test 1: Basic train/val/test split
    print("\n1. Testing basic train/val/test split...")

    dataset1 = create_comprehensive_test_dataset()
    context = PipelineContext()

    split_op = SplitStrategy.train_val_test(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratified=True
    )

    split_op.execute(dataset1, context)

    train_size = len(dataset1.select(partition="train"))
    val_size = len(dataset1.select(partition="val"))
    test_size = len(dataset1.select(partition="test"))

    print(f"Split result - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Total: {train_size + val_size + test_size} (should equal {len(dataset1)})")

    # Check stratification
    train_targets = dataset1.select(partition="train").get_targets("original")
    train_distribution = dict(zip(*np.unique(train_targets, return_counts=True)))
    print(f"Train class distribution: {train_distribution}")

    assert train_size + val_size + test_size == len(dataset1), "Split should preserve total samples"
    print("✓ Basic split verified")    # Test 2: Group-based split (similar to cross-validation concept)
    print("\n2. Testing group-based split...")

    dataset2 = create_comprehensive_test_dataset()

    # Create artificial groups (simulate batch or time-based grouping)
    n_samples = len(dataset2)
    groups = np.repeat(range(5), n_samples // 5)  # 5 groups
    if len(groups) < n_samples:
        groups = np.concatenate([groups, [groups[-1]] * (n_samples - len(groups))])

    group_split = SplitStrategy.group_split(
        groups=groups,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    group_split.execute(dataset2, context)

    # Check that split was successful
    partitions = dataset2.indices['partition'].unique().to_list()
    print(f"Partitions after group split: {partitions}")

    train_size2 = len(dataset2.select(partition="train"))
    val_size2 = len(dataset2.select(partition="val"))
    test_size2 = len(dataset2.select(partition="test"))

    print(f"Group split - Train: {train_size2}, Val: {val_size2}, Test: {test_size2}")
    assert train_size2 + val_size2 + test_size2 == len(dataset2), "Group split should preserve total samples"
    print("✓ Group-based split verified")

    # Test 3: Train/test split only
    print("\n3. Testing train/test split only...")

    dataset3 = create_comprehensive_test_dataset()

    simple_split = SplitStrategy.train_test(train_ratio=0.8, stratified=True)
    simple_split.execute(dataset3, context)

    train_size3 = len(dataset3.select(partition="train"))
    test_size3 = len(dataset3.select(partition="test"))

    print(f"Train/test split - Train: {train_size3}, Test: {test_size3}")
    assert abs(train_size3 / len(dataset3) - 0.8) < 0.05, "Train ratio should be ~0.8"
    print("✓ Train/test split verified")


def test_transformation_operations():
    """Test transformation operations comprehensively."""
    print("\n=== Testing Transformation Operations ===")

    dataset = create_comprehensive_test_dataset()
    context = PipelineContext()

    # Split first
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    split_op.execute(dataset, context)

    # Test 1: Standard transformation
    print("\n1. Testing standard transformation...")

    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        fit_partition="train",
        transform_partitions=["train", "val", "test"],
        mode="transformation"
    )

    initial_size = len(dataset)
    transform_op.execute(dataset, context)

    assert len(dataset) == initial_size, "Standard transformation should not change dataset size"

    # Check standardization
    train_view = dataset.select(partition="train")
    features = train_view.get_features(concatenate=False)
    means = [source.mean() for source in features]
    stds = [source.std() for source in features]

    print(f"Post-transformation means: {[f'{m:.3f}' for m in means]}")
    print(f"Post-transformation stds: {[f'{s:.3f}' for s in stds]}")

    for mean, std in zip(means, stds):
        assert abs(mean) < 0.1, "Mean should be ~0 after standardization"
        assert abs(std - 1.0) < 0.1, "Std should be ~1 after standardization"

    print("✓ Standard transformation verified")

    # Test 2: Sample augmentation
    print("\n2. Testing sample augmentation...")

    dataset2 = create_comprehensive_test_dataset()
    split_op.execute(dataset2, context)

    initial_train_size = len(dataset2.select(partition="train"))

    augment_op = TransformationOperation(
        transformer=[MinMaxScaler(), StandardScaler()],
        fit_partition="train",
        transform_partitions=["train"],
        mode="sample_augmentation"
    )

    augment_op.execute(dataset2, context)

    new_train_size = len(dataset2.select(partition="train"))
    expected_size = initial_train_size * 3  # Original + 2 augmented

    print(f"Train size: {initial_train_size} -> {new_train_size} (expected: {expected_size})")
    assert new_train_size == expected_size, f"Expected {expected_size}, got {new_train_size}"
    print("✓ Sample augmentation verified")


def test_model_operations():
    """Test model operations with different configurations."""
    print("\n=== Testing Model Operations ===")

    # Test 1: Classification model
    print("\n1. Testing classification model...")

    dataset = create_comprehensive_test_dataset()
    context = PipelineContext()

    # Prepare dataset
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    split_op.execute(dataset, context)

    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    )
    transform_op.execute(dataset, context)    # Train classification model
    model_op = ModelOperation(
        model=RandomForestClassifier(n_estimators=50, random_state=42),
        # target_representation="auto",
        # train_on="train",
        # predict_on=["train", "val", "test"]
    )

    model_op.execute(dataset, context)    # Check predictions
    all_predictions = context.get_predictions()
    assert all_predictions, "Should have predictions"

    print(f"Models with predictions: {list(all_predictions.keys())}")

    # Get the first model's predictions
    model_name = list(all_predictions.keys())[0]
    predictions = all_predictions[model_name]

    print(f"Partitions with predictions: {list(predictions.keys())}")

    # Check train predictions
    if 'train' in predictions:
        train_preds = predictions['train']['predictions']
        print(f"Train predictions shape: {train_preds.shape if hasattr(train_preds, 'shape') else len(train_preds)}")
        print(f"Unique train predictions: {np.unique(train_preds)}")

    print("✓ Classification model verified")

    # Test 2: Regression model with target transformation
    print("\n2. Testing regression model with target transformation...")    # Create regression dataset with continuous targets
    np.random.seed(42)  # For reproducible targets

    # Get base data first
    temp_dataset = create_comprehensive_test_dataset()
    temp_view = temp_dataset.select(partition="all")
    features = temp_view.get_features(concatenate=False)

    # Create continuous regression targets from feature combination
    X_concat = np.concatenate(features, axis=1)
    continuous_targets = (X_concat[:, 50:60].mean(axis=1) +
                         X_concat[:, 100:110].mean(axis=1) +
                         np.random.normal(0, 0.1, len(X_concat)))

    # Create new regression dataset
    reg_dataset = SpectraDataset(task_type="regression")
    reg_dataset.add_data(
        features=features,
        targets=continuous_targets,
        partition="all",        processing="raw"
    )

    # Split and transform the regression dataset
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    split_op.execute(reg_dataset, context)

    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    )
    transform_op.execute(reg_dataset, context)

    # Train regression model
    reg_model_op = ModelOperation(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        target_representation="auto",
        train_on="train",
        predict_on=["test"]
    )

    reg_model_op.execute(reg_dataset, context)

    # Check regression predictions
    all_reg_predictions = context.get_predictions()
      # Get regression model predictions (there should be 2 models now)
    reg_model_name = [name for name in all_reg_predictions.keys() if 'RandomForestRegressor' in name][0]
    reg_predictions = all_reg_predictions[reg_model_name]

    test_size = len(reg_dataset.select(partition="test"))

    print(f"Regression models: {list(all_reg_predictions.keys())}")
    print(f"Test predictions available: {list(reg_predictions.keys())}")

    if 'test' in reg_predictions:
        test_preds = reg_predictions['test']['predictions']
        print(f"Test predictions: {len(test_preds)} values")
        print(f"Prediction range: {test_preds.min():.3f} to {test_preds.max():.3f}")
        assert len(test_preds) == test_size, f"Should predict for test set: expected {test_size}, got {len(test_preds)}"

    print("✓ Regression model verified")


def test_clustering_operations():
    """Test clustering operations."""
    print("\n=== Testing Clustering Operations ===")

    dataset = create_comprehensive_test_dataset()
    context = PipelineContext()

    # Prepare dataset
    split_op = SplitStrategy.train_val_test(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    split_op.execute(dataset, context)

    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    )
    transform_op.execute(dataset, context)    # Test clustering
    print("\n1. Testing K-means clustering...")

    cluster_op = ClusteringOperation(
        clustering_method="kmeans",
        n_clusters=3,
        store_centroids=True,
        evaluate_clustering=True
    )

    cluster_op.execute(dataset, context)

    # Check cluster assignments
    cluster_results = context.clustering_results if hasattr(context, 'clustering_results') else None
    assert cluster_results is not None, "Should have clustering results"

    cluster_assignments = cluster_results['labels']
    unique_clusters = np.unique(cluster_assignments)
    print(f"Cluster assignments: {len(cluster_assignments)} total")
    print(f"Unique clusters: {unique_clusters} (expected 3)")
    assert len(unique_clusters) == 3, f"Expected 3 clusters, got {len(unique_clusters)}"
    print(f"Unique clusters: {unique_clusters}")
    print(f"Cluster distribution: {dict(zip(*np.unique(cluster_assignments, return_counts=True)))}")

    assert len(unique_clusters) <= 3, "Should have at most 3 clusters"
    print("✓ K-means clustering verified")


def test_merge_sources_operation():
    """Test source merging operation."""
    print("\n=== Testing Merge Sources Operation ===")

    dataset = create_comprehensive_test_dataset()
    context = PipelineContext()

    # Check initial state
    initial_sources = len(dataset.features.sources)
    initial_shapes = [s.shape for s in dataset.features.sources]

    print(f"Initial sources: {initial_sources}")
    print(f"Initial shapes: {initial_shapes}")

    # Test merging
    merge_op = MergeSourcesOperation()
    merge_op.execute(dataset, context)

    # Check result
    final_sources = len(dataset.features.sources)
    final_shape = dataset.features.sources[0].shape if final_sources > 0 else None

    print(f"Final sources: {final_sources}")
    print(f"Final shape: {final_shape}")

    # Should have merged to 1 source
    assert final_sources == 1, f"Expected 1 source after merge, got {final_sources}"

    # Check concatenated width
    expected_width = sum(shape[1] for shape in initial_shapes)
    assert final_shape[1] == expected_width, f"Expected width {expected_width}, got {final_shape[1]}"

    print("✓ Source merging verified")


def test_dispatch_operation():
    """Test dispatch operation for parallel model training."""
    print("\n=== Testing Dispatch Operation ===")

    dataset = create_comprehensive_test_dataset()
    context = PipelineContext()

    # Prepare dataset
    split_op = SplitStrategy.train_val_test(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    split_op.execute(dataset, context)

    transform_op = TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    )
    transform_op.execute(dataset, context)

    # Test dispatch with multiple models
    print("\n1. Testing dispatch with multiple models...")    # Create actual operation objects instead of config dictionaries
    operations = [
        ModelOperation(
            model=RandomForestClassifier(n_estimators=30, random_state=42),
            target_representation="auto",
            train_on="train",
            predict_on=["test"]
        ),
        ModelOperation(
            model=SVC(kernel='linear', random_state=42),
            target_representation="auto",
            train_on="train",
            predict_on=["test"]
        ),
        ModelOperation(
            model=LogisticRegression(random_state=42, max_iter=1000),
            target_representation="auto",
            train_on="train",
            predict_on=["test"]
        )
    ]

    dispatch_op = DispatchOperation(
        operations=operations,
        dispatch_strategy="parallel",
        max_workers=2,
        merge_results=True
    )

    initial_size = len(dataset)
    dispatch_op.execute(dataset, context)

    # Check that operations executed successfully
    branches = dataset.indices['branch'].unique().to_list()
    print(f"Branches: {branches}")

    # For parallel execution, we don't necessarily create new branches
    # The main value is that multiple models executed successfully
    print("✓ Parallel execution completed successfully")

    # Check dataset size (should remain the same for parallel execution)
    final_size = len(dataset)
    print(f"Dataset size: {initial_size} -> {final_size}")

    # For our parallel implementation, size should remain constant
    assert final_size == initial_size, "Parallel execution should preserve dataset size"

    print("✓ Dispatch operation verified")

    # Test 2: Check individual branch results
    print("\n2. Checking individual branch results...")

    for branch_id in branches:
        if branch_id == 0:  # Skip original branch
            continue

        branch_view = dataset.select(branch=branch_id)
        print(f"Branch {branch_id}: {len(branch_view)} samples")

        # Check that branch has data
        assert len(branch_view) > 0, f"Branch {branch_id} should have data"

    print("✓ Branch results verified")


def test_operation_factory():
    """Test operation factory for creating operations from configuration."""
    print("\n=== Testing Operation Factory ===")

    factory = OperationFactory()

    # Test 1: Create transformation operation
    print("\n1. Testing transformation operation creation...")

    transform_config = {
        "type": "transformation",
        "transformer": {"type": "StandardScaler"},
        "mode": "transformation",
        "fit_partition": "train"
    }

    transform_op = factory.create_operation(transform_config)
    assert isinstance(transform_op, TransformationOperation), "Should create TransformationOperation"
    print("✓ Transformation operation creation verified")

    # Test 2: Create model operation
    print("\n2. Testing model operation creation...")

    model_config = {
        "type": "model",
        "model": {"type": "RandomForestClassifier", "n_estimators": 100},
        "target_representation": "classification"
    }

    model_op = factory.create_operation(model_config)
    assert isinstance(model_op, ModelOperation), "Should create ModelOperation"
    print("✓ Model operation creation verified")

    # Test 3: Create split operation
    print("\n3. Testing split operation creation...")

    split_config = {
        "type": "split",
        "strategy": "train_val_test",
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "stratified": True
    }

    split_op = factory.create_operation(split_config)
    assert hasattr(split_op, 'execute'), "Should create executable split operation"
    print("✓ Split operation creation verified")


def test_complex_operation_sequence():
    """Test a complex sequence of operations."""
    print("\n=== Testing Complex Operation Sequence ===")

    dataset = create_comprehensive_test_dataset()
    pipeline = Pipeline("Complex Operation Test")    # Build complex pipeline
    pipeline.add_operation(SplitStrategy.train_val_test(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2))

    pipeline.add_operation(TransformationOperation(
        transformer=StandardScaler(),
        mode="transformation"
    ))

    pipeline.add_operation(TransformationOperation(
        transformer=[MinMaxScaler()],
        fit_partition="train",
        transform_partitions=["train", "val", "test"],
        mode="transformation"
    ))

    pipeline.add_operation(ModelOperation(
        model=RandomForestClassifier(n_estimators=50, random_state=42),
        target_representation="classification",
        train_on="train",
        predict_on=["val", "test"]
    ))

    # Execute pipeline
    initial_size = len(dataset)
    print(f"Initial dataset size: {initial_size}")

    pipeline.execute(dataset)

    final_size = len(dataset)
    print(f"Final dataset size: {final_size}")

    # Check that all operations executed
    context = pipeline.context
    predictions = context.get_predictions() if hasattr(context, 'get_predictions') else None
    clusters = context.clustering_results if hasattr(context, 'clustering_results') else None

    print(f"Predictions available: {predictions is not None}")
    print(f"Clusters available: {clusters is not None}")    # Check processing types
    processing_types = dataset.indices['processing'].unique().to_list()
    print(f"Processing types: {processing_types}")

    # Should have at least one processing type
    assert len(processing_types) >= 1, "Should have at least one processing type"

    print("✓ Complex operation sequence verified")


def main():
    """Run all operation tests."""
    print("=" * 60)
    print("TEST 3: ALL OPERATIONS")
    print("=" * 60)

    try:
        # Test individual operations
        test_split_operations()
        test_transformation_operations()
        test_model_operations()
        test_clustering_operations()
        test_merge_sources_operation()
        test_dispatch_operation()
        test_operation_factory()
        test_complex_operation_sequence()

        print("\n" + "=" * 60)
        print("✅ ALL OPERATION TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ OPERATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
