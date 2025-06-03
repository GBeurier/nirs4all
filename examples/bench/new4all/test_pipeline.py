"""
Test script for the new fold-based spectral dataset architecture
"""
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import our new components
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext
from SimpleSplitOperation import SimpleSplitOperation
from FoldSplitOperation import FoldSplitOperation
from ModelOperation import ModelOperation
from FoldModelOperation import FoldModelOperation


def create_test_dataset(n_samples=200, n_features=100, n_targets=1):
    """Create a synthetic dataset for testing."""

    # Generate synthetic spectral data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # For now, use single target to avoid TargetManager issues
    if n_targets == 1:
        y = np.random.randn(n_samples)  # 1D array
    else:
        y = np.random.randn(n_samples, n_targets)

    # Create SpectraDataset
    dataset = SpectraDataset()

    # Add data
    sample_ids = dataset.add_data(
        features=X,
        targets=y,
        partition="unassigned"
    )

    # Add metadata if needed - for now we'll keep it simple
    # The dataset should have basic structure for testing

    return dataset


def test_simple_split():
    """Test simple train/test split functionality."""
    print("\\n=== Testing Simple Split ===")

    dataset = create_test_dataset()
    context = PipelineContext()

    # Create and execute simple split
    split_op = SimpleSplitOperation(test_size=0.2, random_state=42)
    split_op.execute(dataset, context)

    # Check partitions
    partitions = dataset.indices["partition"].value_counts()
    print(f"Partition counts: {partitions}")

    return dataset


def test_fold_split():
    """Test fold-based split functionality."""
    print("\\n=== Testing Fold Split ===")

    dataset = create_test_dataset()
    context = PipelineContext()

    # Test KFold (for regression data)
    fold_op = FoldSplitOperation(
        fold_strategy="KFold",
        n_splits=5,
        random_state=42
    )
    fold_op.execute(dataset, context)

    # Check folds
    print(f"Number of folds created: {len(dataset.folds)}")
    for i, fold in enumerate(dataset.folds):
        train_size = len(fold["train_indices"])
        val_size = len(fold["val_indices"])
        print(f"Fold {i}: train={train_size}, val={val_size}")

    return dataset


def test_model_operation():
    """Test basic model operation with predictions storage."""
    print("\\n=== Testing Model Operation ===")

    dataset = create_test_dataset()
    context = PipelineContext()

    # Create simple split first
    split_op = SimpleSplitOperation(test_size=0.2, random_state=42)
    split_op.execute(dataset, context)

    # Create and execute model operation
    model_op = ModelOperation(
        model=RandomForestRegressor(n_estimators=10, random_state=42),
        model_name="test_rf"
    )
    model_op.execute(dataset, context)

    # Check predictions
    predictions = dataset.get_predictions()
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions columns: {predictions.columns}")

    # Get test predictions
    test_preds = dataset.get_predictions(partition="test")
    print(f"Test predictions shape: {test_preds.shape}")

    return dataset


def test_fold_model_operation():
    """Test fold-based model operation with cross-validation."""
    print("\\n=== Testing Fold Model Operation ===")

    dataset = create_test_dataset()
    context = PipelineContext()

    # Create fold split first
    fold_op = FoldSplitOperation(
        fold_strategy="KFold",
        n_splits=3,  # Use 3 folds for faster testing
        random_state=42
    )
    fold_op.execute(dataset, context)

    # Create and execute fold model operation
    fold_model_op = FoldModelOperation(
        model=RandomForestRegressor(n_estimators=10, random_state=42),
        model_name="cv_rf",
        aggregation_method="mean"
    )
    fold_model_op.execute(dataset, context)

    # Check fold predictions
    print(f"Number of fold predictions: {len(dataset.get_fold_predictions())}")

    # Check aggregated predictions
    agg_preds = dataset.get_predictions(prediction_type="aggregated")
    print(f"Aggregated predictions shape: {agg_preds.shape}")

    # Check out-of-fold predictions for stacking
    oof_preds = dataset.get_reconstructed_train_predictions()
    print(f"Out-of-fold predictions shape: {oof_preds.shape}")

    return dataset


def test_complete_pipeline():
    """Test complete pipeline with multiple models and stacking."""
    print("\\n=== Testing Complete Pipeline ===")

    dataset = create_test_dataset()
    context = PipelineContext()

    # 1. Create fold split
    fold_op = FoldSplitOperation(
        fold_strategy="KFold",
        n_splits=3,
        random_state=42
    )
    fold_op.execute(dataset, context)

    # 2. Train multiple base models
    models = [
        ("rf", RandomForestRegressor(n_estimators=10, random_state=42)),
        ("rf2", RandomForestRegressor(n_estimators=15, max_depth=5, random_state=43))
    ]

    for model_name, model in models:
        fold_model_op = FoldModelOperation(
            model=model,
            model_name=model_name,
            aggregation_method="mean"
        )
        fold_model_op.execute(dataset, context)

    # 3. Check results
    all_predictions = dataset.get_predictions()
    print(f"Total predictions stored: {all_predictions.shape[0]}")

    # Check unique models
    unique_models = all_predictions["model_name"].unique()
    print(f"Models trained: {unique_models}")

    # Get out-of-fold predictions for each model (for stacking)
    for model_name in unique_models:
        oof_preds = dataset.get_reconstructed_train_predictions(model_name=model_name)
        print(f"OOF predictions for {model_name}: {oof_preds.shape}")

    # Get results summary
    summary = dataset.get_results_summary()
    print(f"Results summary:\\n{summary}")

    return dataset


if __name__ == "__main__":
    print("Testing new fold-based spectral dataset architecture...")

    try:
        # Run all tests
        test_simple_split()
        test_fold_split()
        test_model_operation()
        test_fold_model_operation()
        test_complete_pipeline()

        print("\\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
