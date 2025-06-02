import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from SpectraDataset import SpectraDataset
from TargetManager import TargetManager
from ModelOperation import ModelOperation
from PipelineContext import PipelineContext


def test_advanced_pipeline_features():
    """Test advanced pipeline features with target management."""

    print("=== Testing Advanced Pipeline Features ===\n")

    # 1. Create a comprehensive dataset
    print("1. Creating multi-partition dataset...")

    # Classification dataset with different partitions
    np.random.seed(42)  # For reproducibility

    # Create features for different partitions
    train_features = [np.random.randn(100, 200)]  # 100 samples, 200 features
    val_features = [np.random.randn(30, 200)]     # 30 samples
    test_features = [np.random.randn(50, 200)]    # 50 samples

    # Create realistic class targets with some imbalance
    train_targets = np.random.choice(['healthy', 'disease_a', 'disease_b'],
                                   size=100, p=[0.6, 0.25, 0.15])
    val_targets = np.random.choice(['healthy', 'disease_a', 'disease_b'],
                                 size=30, p=[0.6, 0.25, 0.15])
    test_targets = np.random.choice(['healthy', 'disease_a', 'disease_b'],
                                  size=50, p=[0.6, 0.25, 0.15])

    # Create dataset
    dataset = SpectraDataset(task_type="classification")

    # Add data with different partitions
    train_ids = dataset.add_data(train_features, targets=train_targets, partition="train")
    val_ids = dataset.add_data(val_features, targets=val_targets, partition="validation")
    test_ids = dataset.add_data(test_features, targets=test_targets, partition="test")

    print(f"   Train samples: {len(train_ids)}")
    print(f"   Validation samples: {len(val_ids)}")
    print(f"   Test samples: {len(test_ids)}")
    print(f"   Classes: {dataset.classes_}")
    print(f"   Task type: {dataset.task_type}")

    # 2. Test advanced target transformations
    print("\n2. Testing target transformation pipelines...")

    # Create a complex target transformation pipeline
    target_pipeline = [
        StandardScaler(),  # Normalize target values
        MinMaxScaler()     # Scale to [0,1] range
    ]

    # Apply transformations to training targets
    context = PipelineContext()
    transformed_targets = dataset.fit_transform_targets(
        train_ids, target_pipeline, "regression", "complex_pipeline"
    )

    print(f"   Original regression targets (first 10): {dataset.get_targets(train_ids[:10], 'regression')}")
    print(f"   Transformed targets (first 10): {transformed_targets[:10]}")

    # Test inverse transformation
    sample_predictions = np.array([0.2, 0.8, 0.1, 0.9, 0.5])
    inverse_preds = dataset.inverse_transform_predictions(
        sample_predictions, "regression", "complex_pipeline", to_original=True
    )
    print(f"   Sample predictions: {sample_predictions}")
    print(f"   Inverse transformed: {inverse_preds}")

    # 3. Test advanced model operations
    print("\n3. Testing advanced model operations...")

    # Test classification model with target transformations
    clf_model = ModelOperation(
        model=RandomForestClassifier(n_estimators=10, random_state=42),
        train_on="train",
        predict_on=["validation", "test"],
        target_representation="classification",
        target_transformers=None,  # Classification doesn't need target transforms
        transformer_key="classification_model"
    )

    # Execute model operation
    clf_model.execute(dataset, context)

    # Get predictions
    predictions = context.get_predictions()
    print(f"   Model predictions stored: {list(predictions.keys())}")

    if clf_model.get_name() in predictions:
        model_preds = predictions[clf_model.get_name()]

        for partition in model_preds:
            if partition == 'validation':
                pred_data = model_preds[partition]
                true_targets = pred_data['true_targets']
                pred_targets = pred_data['predictions']

                accuracy = accuracy_score(true_targets, pred_targets)
                print(f"   Validation accuracy: {accuracy:.3f}")
                print(f"   Validation predictions (first 10): {pred_targets[:10]}")
                print(f"   Validation true targets (first 10): {true_targets[:10]}")

    # 4. Test regression model with target transformations
    print("\n4. Testing regression model with target transformations...")

    # Create regression dataset
    reg_dataset = SpectraDataset(task_type="regression")

    # Create regression targets (continuous values)
    reg_train_targets = np.random.normal(50, 15, 80)  # Mean=50, std=15
    reg_test_targets = np.random.normal(50, 15, 40)

    reg_train_features = [np.random.randn(80, 200)]
    reg_test_features = [np.random.randn(40, 200)]

    reg_train_ids = reg_dataset.add_data(reg_train_features, targets=reg_train_targets, partition="train")
    reg_test_ids = reg_dataset.add_data(reg_test_features, targets=reg_test_targets, partition="test")

    print(f"   Regression train samples: {len(reg_train_ids)}")
    print(f"   Regression test samples: {len(reg_test_ids)}")
    print(f"   Target range: [{reg_train_targets.min():.1f}, {reg_train_targets.max():.1f}]")

    # Test regression model with target scaling
    reg_model = ModelOperation(
        model=RandomForestRegressor(n_estimators=10, random_state=42),
        train_on="train",
        predict_on=["test"],
        target_representation="regression",
        target_transformers=[StandardScaler()],  # Scale targets for better training
        transformer_key="regression_scaler"
    )

    reg_context = PipelineContext()
    reg_model.execute(reg_dataset, reg_context)

    # Evaluate regression results
    reg_predictions = reg_context.get_predictions()
    if reg_model.get_name() in reg_predictions:
        reg_pred_data = reg_predictions[reg_model.get_name()]['test']
        true_reg = reg_pred_data['true_targets']
        pred_reg = reg_pred_data['predictions']

        mse = mean_squared_error(true_reg, pred_reg)
        print(f"   Regression MSE: {mse:.3f}")
        print(f"   True targets (first 5): {true_reg[:5]}")
        print(f"   Predicted targets (first 5): {pred_reg[:5]}")
      # 5. Test dataset views with advanced filtering
    print("\n5. Testing advanced dataset views...")

    # Create views for different data subsets
    train_view = dataset.select(partition="train")
    val_view = dataset.select(partition="validation")

    print(f"   Full train view: {len(train_view)} samples")
    print(f"   Validation view: {len(val_view)} samples")

    # Test target consistency across views
    view_targets = train_view.get_targets("classification")
    view_original = train_view.get_targets("original")

    print(f"   View classification targets (first 10): {view_targets[:10]}")
    print(f"   View original targets (first 10): {view_original[:10]}")

    # Test view splitting by partition
    all_views = dataset.select().split_by("partition")
    print(f"   Split by partition: {list(all_views.keys())}")
    for partition, view in all_views.items():
        print(f"     {partition}: {len(view)} samples")

    # 6. Test target info and metadata
    print("\n6. Testing target information...")

    clf_info = dataset.get_target_info()
    reg_info = reg_dataset.get_target_info()

    print(f"   Classification target info: {clf_info}")
    print(f"   Regression target info: {reg_info}")

    print("\n=== Advanced Pipeline Features Test Complete ===")


if __name__ == "__main__":
    test_advanced_pipeline_features()
