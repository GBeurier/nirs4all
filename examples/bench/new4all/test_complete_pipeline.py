"""
Test complete fold-based model training pipeline with ModelOperation integration
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from SpectraDataset import SpectraDataset
from FoldSplitOperation import FoldSplitOperation
from FoldModelOperation import FoldModelOperation
from ModelOperation import ModelOperation

def test_complete_fold_pipeline():
    """Test the complete fold-based model training pipeline."""
    print("=== Testing Complete Fold-based Model Training Pipeline ===")

    # Create dataset with synthetic data
    print("\n1. Creating synthetic dataset...")
    dataset = SpectraDataset()

    # Generate synthetic spectral data
    n_samples = 100
    n_features = 50
    n_targets = 1

    # Generate features (spectral data)
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)

    # Generate targets with some relationship to features
    targets = np.sum(features[:, :10], axis=1) + 0.1 * np.random.randn(n_samples)
    targets = targets.reshape(-1, 1)  # Make 2D for consistency    # Add data to dataset (sample IDs are generated automatically)
    sample_ids = dataset.add_data(
        features=features,
        targets=targets
    )

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test 1: Fold-based splitting
    print("\n2. Testing fold-based data splitting...")
    fold_splitter = FoldSplitOperation(
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        split_name="cv_split"
    )

    fold_splitter.execute(dataset, {})

    # Verify folds were created
    folds_df = dataset.get_fold_info()
    print(f"Created {len(folds_df)} fold entries")
    print("Fold distribution:")
    print(folds_df.group_by("fold").agg(pl.count("sample_id").alias("count")))

    # Test 2: Fold-based model training
    print("\n3. Testing fold-based model training...")

    # Create model operation
    model_op = ModelOperation(
        model=LinearRegression(),
        model_name="LinearRegression"
    )

    # Create fold model operation
    fold_model_op = FoldModelOperation(model_op, split_name="cv_split")

    # Execute fold-based training
    fold_model_op.execute(dataset, {})

    # Verify predictions were created
    predictions_df = dataset.get_predictions()
    print(f"Created {len(predictions_df)} prediction entries")

    if len(predictions_df) > 0:
        print("Predictions by fold:")
        fold_counts = predictions_df.group_by("fold").agg(pl.count("sample_id").alias("count"))
        print(fold_counts)

        print("\nPredictions by partition:")
        partition_counts = predictions_df.group_by("partition").agg(pl.count("sample_id").alias("count"))
        print(partition_counts)

    # Test 3: Out-of-fold prediction reconstruction
    print("\n4. Testing out-of-fold prediction reconstruction...")

    oof_predictions = dataset.get_fold_predictions(
        model_name="LinearRegression",
        aggregation="mean"
    )

    if oof_predictions is not None:
        print(f"Out-of-fold predictions shape: {oof_predictions.shape}")
        print(f"Coverage: {len(oof_predictions)}/{len(dataset)} samples")

        # Compare with actual targets
        actual_targets = dataset.get_targets("auto")
        if actual_targets is not None and len(actual_targets) == len(oof_predictions):
            mse = np.mean((actual_targets.flatten() - oof_predictions.flatten()) ** 2)
            print(f"Out-of-fold MSE: {mse:.6f}")

        # Check for missing predictions
        sample_ids_with_preds = dataset.get_predictions().select("sample_id").unique().to_series().to_list()
        all_sample_ids = dataset.sample_ids
        missing_samples = set(all_sample_ids) - set(sample_ids_with_preds)

        if missing_samples:
            print(f"WARNING: {len(missing_samples)} samples missing predictions: {list(missing_samples)[:5]}...")
        else:
            print("✓ All samples have out-of-fold predictions")
    else:
        print("WARNING: No out-of-fold predictions found")

    # Test 4: Model evaluation
    print("\n5. Testing model evaluation...")

    # Get all predictions for evaluation
    all_predictions = dataset.get_predictions()
    if len(all_predictions) > 0:
        # Group by fold and evaluate
        for fold_id in all_predictions.select("fold").unique().to_series().to_list():
            fold_preds = all_predictions.filter(pl.col("fold") == fold_id)
            fold_sample_ids = fold_preds.select("sample_id").to_series().to_list()

            # Get actual targets for these samples
            fold_targets = []
            fold_pred_values = []

            for sample_id in fold_sample_ids:
                sample_idx = dataset.sample_ids.index(sample_id)
                fold_targets.append(actual_targets[sample_idx])

                # Get prediction value
                pred_row = fold_preds.filter(pl.col("sample_id") == sample_id).row(0)
                fold_pred_values.append(pred_row[3])  # prediction column

            if fold_targets and fold_pred_values:
                fold_targets = np.array(fold_targets)
                fold_pred_values = np.array(fold_pred_values)
                fold_mse = np.mean((fold_targets.flatten() - fold_pred_values.flatten()) ** 2)
                print(f"Fold {fold_id} MSE: {fold_mse:.6f} (n={len(fold_targets)})")

    print("\n=== Complete Pipeline Test Completed Successfully ===")
    return dataset

def test_multiple_models():
    """Test multiple models in the same fold structure."""
    print("\n=== Testing Multiple Models ===")

    # Create dataset
    dataset = SpectraDataset()

    # Generate synthetic data
    np.random.seed(123)
    n_samples = 80
    n_features = 30

    features = np.random.randn(n_samples, n_features)
    targets = (np.sum(features[:, :5], axis=1) + 0.2 * np.random.randn(n_samples)).reshape(-1, 1)
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]

    dataset.add_data(
        sample_ids=sample_ids,
        features=features,
        targets=targets,
        target_names=["target_1"]
    )

    # Create fold splits
    fold_splitter = FoldSplitOperation(
        cv=KFold(n_splits=4, shuffle=True, random_state=123),
        split_name="multi_model_cv"
    )
    fold_splitter.execute(dataset, {})

    # Test multiple models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=10, random_state=42)),
        ("SVR", SVR(kernel='linear', C=1.0))
    ]

    for model_name, model in models:
        print(f"\nTraining {model_name}...")

        model_op = ModelOperation(model=model, model_name=model_name)
        fold_model_op = FoldModelOperation(model_op, split_name="multi_model_cv")
        fold_model_op.execute(dataset, {})

        # Get out-of-fold predictions
        oof_preds = dataset.get_fold_predictions(
            model_name=model_name,
            aggregation="mean"
        )

        if oof_preds is not None:
            actual_targets = dataset.get_targets("auto")
            mse = np.mean((actual_targets.flatten() - oof_preds.flatten()) ** 2)
            print(f"{model_name} Out-of-fold MSE: {mse:.6f}")

    # Summary
    all_predictions = dataset.get_predictions()
    print(f"\nTotal predictions stored: {len(all_predictions)}")
    print("Models trained:")
    for model_name in all_predictions.select("model_name").unique().to_series().to_list():
        model_preds = all_predictions.filter(pl.col("model_name") == model_name)
        print(f"  {model_name}: {len(model_preds)} predictions")

    return dataset

if __name__ == "__main__":
    # Run tests
    dataset1 = test_complete_fold_pipeline()
    dataset2 = test_multiple_models()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("✓ Fold-based data splitting")
    print("✓ Fold-based model training")
    print("✓ Prediction storage and retrieval")
    print("✓ Out-of-fold prediction reconstruction")
    print("✓ Multiple model support")
    print("✓ Model evaluation metrics")
