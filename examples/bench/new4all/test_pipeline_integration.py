import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.cross_decomposition import PLSRegression

from SpectraDataset import SpectraDataset
from ModelOperation import ModelOperation
from PipelineContext import PipelineContext


def test_pipeline_integration():
    """Test pipeline integration with enhanced target management."""

    print("=== Testing Pipeline Integration with Target Management ===\n")

    # Test 1: Classification with target transformations
    print("1. Classification pipeline with target transformations...")

    # Create classification dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 150

    features = [np.random.randn(n_samples, n_features)]

    # Create realistic class distribution
    class_probs = np.random.dirichlet([2, 3, 1], n_samples)  # Biased toward middle class
    classes = ['low', 'medium', 'high']
    targets = np.array([classes[np.argmax(prob)] for prob in class_probs])

    dataset = SpectraDataset(task_type="classification")

    # Split into train/test
    train_size = int(0.7 * n_samples)
    train_ids = dataset.add_data([features[0][:train_size]], targets=targets[:train_size], partition="train")
    test_ids = dataset.add_data([features[0][train_size:]], targets=targets[train_size:], partition="test")

    print(f"   Created dataset: {len(train_ids)} train, {len(test_ids)} test")
    print(f"   Task type: {dataset.task_type}")
    print(f"   Classes: {dataset.classes_}")

    # Create pipeline context
    context = PipelineContext()

    # Test model without target transformations
    print("\n   Testing model without target transformations...")
    model_simple = ModelOperation(
        model=RandomForestClassifier(n_estimators=50, random_state=42),
        target_representation="classification"
    )

    model_simple.execute(dataset, context)
    simple_predictions = context.get_predictions()

    # Test model with target transformations
    print("   Testing model with target transformations...")
    target_transformers = [StandardScaler()]

    model_transformed = ModelOperation(
        model=PLSRegression(n_components=10),
        target_representation="regression",
        target_transformers=target_transformers,
        transformer_key="logistic_scaler"
    )

    model_transformed.execute(dataset, context)
    transformed_predictions = context.get_predictions()

    # Evaluate results
    print("\n   Results comparison:")

    test_view = dataset.select(partition="test")
    true_targets = test_view.get_targets("classification")

    for model_name, preds in [("Simple", simple_predictions), ("Transformed", transformed_predictions)]:
        if model_name in [p for p in preds.keys() if "Model" in p]:
            model_key = [k for k in preds.keys() if model_name.lower() in k.lower()][0]
            test_preds = preds[model_key]["test"]["predictions"]

            if isinstance(test_preds[0], str):
                # Convert string predictions to numeric for comparison
                pred_encoded = [list(dataset.classes_).index(pred) for pred in test_preds]
            else:
                pred_encoded = test_preds

            accuracy = accuracy_score(true_targets, pred_encoded)
            print(f"     {model_name} model accuracy: {accuracy:.3f}")

    # Test 2: Regression pipeline
    print("\n2. Regression pipeline with target scaling...")

    # Create regression dataset
    reg_features = [np.random.randn(100, 120)]
    # Simulate temperature data with some structure
    base_temp = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    noise = np.random.normal(0, 2, 100)
    reg_targets = base_temp + noise

    reg_dataset = SpectraDataset(task_type="regression")
    reg_train_size = 70

    reg_train_ids = reg_dataset.add_data(
        [reg_features[0][:reg_train_size]],
        targets=reg_targets[:reg_train_size],
        partition="train"
    )
    reg_test_ids = reg_dataset.add_data(
        [reg_features[0][reg_train_size:]],
        targets=reg_targets[reg_train_size:],
        partition="test"
    )

    print(f"   Regression dataset: {len(reg_train_ids)} train, {len(reg_test_ids)} test")
    print(f"   Target range: {reg_targets.min():.2f} to {reg_targets.max():.2f}")

    # Test regression with scaling
    reg_context = PipelineContext()

    reg_model = ModelOperation(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        target_representation="regression",
        target_transformers=[MinMaxScaler()],
        transformer_key="minmax_scaler"
    )

    reg_model.execute(reg_dataset, reg_context)
    reg_predictions = reg_context.get_predictions()

    # Evaluate regression
    reg_test_view = reg_dataset.select(partition="test")
    true_reg_targets = reg_test_view.get_targets("regression")

    model_key = list(reg_predictions.keys())[0]
    pred_values = reg_predictions[model_key]["test"]["predictions"]

    mse = mean_squared_error(true_reg_targets, pred_values)
    rmse = np.sqrt(mse)

    print(f"   Regression RMSE: {rmse:.3f}")
    print(f"   True targets sample: {true_reg_targets[:5]}")
    print(f"   Predictions sample: {pred_values[:5]}")

    # Test 3: Multi-representation workflow
    print("\n3. Multi-representation workflow...")

    # Use the classification dataset for both classification and regression views
    multi_context = PipelineContext()

    # Train classification model
    class_model = ModelOperation(
        model=RandomForestClassifier(n_estimators=30, random_state=42),
        target_representation="classification"
    )
    class_model.execute(dataset, multi_context)

    # Train regression model on same data (treating classes as numeric)
    reg_on_class_model = ModelOperation(
        model=RandomForestRegressor(n_estimators=30, random_state=42),
        target_representation="regression"
    )
    reg_on_class_model.execute(dataset, multi_context)

    multi_predictions = multi_context.get_predictions()
    print(f"   Trained {len(multi_predictions)} models on same data")

    # Compare predictions
    test_view = dataset.select(partition="test")
    original_targets = test_view.get_targets("original")

    print(f"   Original targets: {original_targets[:5]}")

    for model_name, model_preds in multi_predictions.items():
        if "test" in model_preds:
            preds = model_preds["test"]["predictions"]
            print(f"   {model_name} predictions: {preds[:5]}")

    print("\n=== Pipeline Integration Test Complete ===")

if __name__ == "__main__":
    test_pipeline_integration()
