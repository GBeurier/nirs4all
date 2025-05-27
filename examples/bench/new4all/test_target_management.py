import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from SpectraDataset import SpectraDataset
from TargetManager import TargetManager
from TransformOperation import TransformOperation
from ClusterOperation import ClusterOperation
from ModelOperation import ModelOperation
from PipelineRunner import PipelineRunner


def test_target_management():
    """Test the new target management system with different scenarios."""
    
    print("=== Testing Target Management System ===\n")
    
    # Test 1: Classification with string labels
    print("1. Classification with string labels:")
    dataset_class = SpectraDataset(task_type="classification")
    
    # Add dummy data with string labels
    X_train = np.random.randn(50, 100)
    y_train = np.random.choice(['apple', 'banana', 'cherry'], 50)
    
    sample_ids = dataset_class.add_data(X_train, y_train, partition="train")
    print(f"   Task type: {dataset_class.task_type}")
    print(f"   Classes: {dataset_class.classes_}")
    print(f"   Number of classes: {dataset_class.n_classes}")
    
    # Get targets in different representations
    original_targets = dataset_class.get_targets(sample_ids[:5], "original")
    regression_targets = dataset_class.get_targets(sample_ids[:5], "regression") 
    classification_targets = dataset_class.get_targets(sample_ids[:5], "classification")
    
    print(f"   Original targets (first 5): {original_targets}")
    print(f"   Regression targets (first 5): {regression_targets}")
    print(f"   Classification targets (first 5): {classification_targets}")
    print()
    
    # Test 2: Target transformation with y_pipeline
    print("2. Target transformation with y_pipeline:")
    transformers = [StandardScaler(), MinMaxScaler()]
    transformed_targets = dataset_class.fit_transform_targets(
        sample_ids, transformers, "regression", "scaler_pipeline")
    
    print(f"   Transformed targets (first 5): {transformed_targets[:5]}")
    
    # Test inverse transformation
    fake_predictions = np.array([0.5, 1.2, -0.3, 0.8, 0.1])
    inverse_predictions = dataset_class.inverse_transform_predictions(
        fake_predictions, "regression", "scaler_pipeline", to_original=True)
    
    print(f"   Fake predictions: {fake_predictions}")
    print(f"   Inverse transformed to original: {inverse_predictions}")
    print()
    
    # Test 3: Regression with automatic binning for classification
    print("3. Regression with automatic binning:")
    dataset_reg = SpectraDataset(task_type="regression")
    
    X_reg = np.random.randn(30, 100)
    y_reg = np.random.randn(30) * 10 + 50  # Continuous values around 50
    
    sample_ids_reg = dataset_reg.add_data(X_reg, y_reg, partition="train")
    print(f"   Task type: {dataset_reg.task_type}")
    print(f"   Regression targets (first 5): {dataset_reg.get_targets(sample_ids_reg[:5], 'regression')}")
    print(f"   Binned classification (first 5): {dataset_reg.get_targets(sample_ids_reg[:5], 'classification')}")
    print(f"   Number of bins: {dataset_reg.n_classes}")
    print()
    
    # Test 4: Auto-detection
    print("4. Auto task type detection:")
    dataset_auto = SpectraDataset(task_type="auto")
    
    # Binary classification data
    X_binary = np.random.randn(40, 100)
    y_binary = np.random.choice([0, 1], 40)
    
    dataset_auto.add_data(X_binary, y_binary, partition="train")
    print(f"   Auto-detected task type: {dataset_auto.task_type}")
    print(f"   Is binary: {dataset_auto.target_manager.is_binary}")
    print()
    
    # Test 5: Target info
    print("5. Target information:")
    info = dataset_class.get_target_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print()
    
    print("=== Target Management System Test Complete ===")


def example_pipeline_with_target_management():
    """Example showing how to use the pipeline with proper target management."""
    
    print("=== Pipeline Example with Target Management ===\n")
    
    # Create dataset with mixed string/numeric targets
    dataset = SpectraDataset(task_type="auto")
    
    # Training data with string labels
    X_train = np.random.randn(100, 1000)
    y_train = np.random.choice(['setosa', 'versicolor', 'virginica'], 100)
    
    # Test data
    X_test = np.random.randn(30, 1000)
    y_test = np.random.choice(['setosa', 'versicolor', 'virginica'], 30)
    
    train_ids = dataset.add_data(X_train, y_train, partition="train")
    test_ids = dataset.add_data(X_test, y_test, partition="test")
    
    print(f"Dataset created with task type: {dataset.task_type}")
    print(f"Classes: {list(dataset.classes_)}")
    print(f"Training samples: {len(train_ids)}, Test samples: {len(test_ids)}")
    
    # Apply y_pipeline transformations
    y_transformers = [StandardScaler()]
    transformed_y_train = dataset.fit_transform_targets(
        train_ids, y_transformers, "regression", "y_pipeline")
    
    print(f"Y targets transformed for training")
    
    # Create pipeline
    pipeline = [
        TransformOperation(StandardScaler()),
        ClusterOperation(KMeans(n_clusters=3)),
        ModelOperation(RandomForestClassifier(n_estimators=50))
    ]
    
    # Run pipeline (this would need to be updated to use the new target system)
    runner = PipelineRunner()
    # Note: The runner would need to be updated to handle the new target management
    print("Pipeline defined and ready to run")
    
    # Simulate predictions and inverse transform
    fake_predictions = np.random.randn(len(test_ids))
    original_predictions = dataset.inverse_transform_predictions(
        fake_predictions, "regression", "y_pipeline", to_original=True)
    
    print(f"Example predictions converted back to original format: {original_predictions[:5]}")
    
    print("\n=== Pipeline Example Complete ===")


if __name__ == "__main__":
    test_target_management()
    print("\n" + "="*60 + "\n")
    example_pipeline_with_target_management()
