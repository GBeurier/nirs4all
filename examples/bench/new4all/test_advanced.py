import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from SpectraDataset import SpectraDataset
from TargetManager import TargetManager

def test_advanced_features():
    """Test advanced target management features."""

    print("=== Testing Advanced Target Management ===\n")

    # Test 1: String classification with transformations
    print("1. String classification with target transformations...")

    features = [np.random.randn(50, 200)]
    targets = np.array(['high', 'medium', 'low'] * 16 + ['high', 'medium'])

    dataset = SpectraDataset(task_type="auto")
    sample_ids = dataset.add_data(features, targets=targets)

    print(f"   Dataset: {len(sample_ids)} samples, task: {dataset.task_type}")
    print(f"   Classes: {dataset.classes_}")

    # Fit target transformations
    transformers = [StandardScaler()]
    transformed_targets = dataset.fit_transform_targets(
        sample_ids, transformers, "regression", "standard_scaler"
    )

    print(f"   Original regression targets: {dataset.get_targets(sample_ids[:5], 'regression')}")
    print(f"   Transformed targets: {transformed_targets[:5]}")

    # Test inverse transform
    predictions = np.array([0.5, -0.3, 1.2, -0.8, 0.9])
    inverse_pred = dataset.inverse_transform_predictions(
        predictions, "regression", "standard_scaler", to_original=True
    )
    print(f"   Inverse predictions: {inverse_pred}")

    # Test 2: Regression with binning
    print("\n2. Regression with automatic binning...")

    reg_features = [np.random.randn(30, 150)]
    reg_targets = np.random.normal(25.0, 5.0, 30)  # Temperature-like data

    reg_dataset = SpectraDataset(task_type="auto")
    reg_ids = reg_dataset.add_data(reg_features, targets=reg_targets)

    print(f"   Task type: {reg_dataset.task_type}")
    print(f"   Regression values: {reg_targets[:5]}")

    # Get binned representation
    binned = reg_dataset.get_targets(reg_ids, "classification")
    print(f"   Binned classes: {binned[:10]}")
    print(f"   Number of bins: {reg_dataset.n_classes}")

    # Test 3: Mixed data workflow
    print("\n3. Mixed data workflow...")

    # Create train/test split
    train_features = [np.random.randn(40, 100)]
    train_targets = np.array(['positive', 'negative'] * 20)

    test_features = [np.random.randn(10, 100)]
    test_targets = np.array(['positive', 'negative'] * 5)

    workflow_dataset = SpectraDataset(task_type="binary")
    train_ids = workflow_dataset.add_data(train_features, targets=train_targets, partition="train")
    test_ids = workflow_dataset.add_data(test_features, targets=test_targets, partition="test")

    print(f"   Binary classification: {workflow_dataset.is_binary}")
    print(f"   Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Create views
    train_view = workflow_dataset.select(partition="train")
    test_view = workflow_dataset.select(partition="test")

    # Get data for training
    X_train = train_view.get_features()
    y_train = train_view.get_targets("classification")

    X_test = test_view.get_features()
    y_test = test_view.get_targets("classification")

    print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Simple ML workflow
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"   Model accuracy: {accuracy:.3f}")

    # Convert predictions back to original format
    original_predictions = workflow_dataset.inverse_transform_predictions(
        predictions, "classification", to_original=True
    )
    print(f"   Original predictions: {original_predictions}")

    # Test 4: Target information
    print("\n4. Target information...")

    for name, ds in [("Classification", dataset), ("Regression", reg_dataset), ("Binary", workflow_dataset)]:
        info = ds.get_target_info()
        print(f"   {name}: {info}")

    print("\n=== Advanced Features Test Complete ===")

if __name__ == "__main__":
    test_advanced_features()
