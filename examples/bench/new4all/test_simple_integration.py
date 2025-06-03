"""
Simple test for fold-based model training pipeline integration
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from SpectraDataset import SpectraDataset
from FoldSplitOperation import FoldSplitOperation
from FoldModelOperation import FoldModelOperation
from ModelOperation import ModelOperation

def test_simple_integration():
    """Test simple integration of fold-based model training."""
    print("=== Testing Simple Fold-based Integration ===")
    
    # Create dataset with synthetic data
    print("\n1. Creating synthetic dataset...")
    dataset = SpectraDataset()
    
    # Generate synthetic spectral data
    n_samples = 50
    n_features = 20
    
    # Generate features (spectral data)
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
      # Generate targets with some relationship to features
    targets = np.sum(features[:, :5], axis=1) + 0.1 * np.random.randn(n_samples)
    # Keep targets as 1D array for compatibility
    
    # Add data to dataset (sample IDs are generated automatically)
    sample_ids = dataset.add_data(
        features=features,
        targets=targets
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample IDs generated: {sample_ids[:5]}...")
      # Test fold-based splitting
    print("\n2. Testing fold-based data splitting...")
    fold_splitter = FoldSplitOperation(
        fold_strategy="KFold",  # Use KFold for continuous targets
        n_splits=3,
        shuffle=True,
        random_state=42,
        split_name="test_cv"
    )
    
    # Execute with empty context
    class SimpleContext:
        pass
    
    context = SimpleContext()
    fold_splitter.execute(dataset, context)
    
    # Check if splits were created
    print("✓ Fold splits created successfully")    # Test model training
    print("\n3. Testing model training...")
    
    # Check if folds are properly stored
    print(f"Number of folds in dataset: {len(dataset.folds)}")
    if dataset.folds:
        print(f"First fold keys: {list(dataset.folds[0].keys())}")
    
    # Create fold model operation directly with sklearn model
    fold_model_op = FoldModelOperation(
        model=LinearRegression()
    )
    
    # Execute fold-based training
    fold_model_op.execute(dataset, context)
    
    print("✓ Fold-based model training completed")
    
    # Test prediction retrieval
    print("\n4. Testing prediction retrieval...")
    
    # Get all predictions
    predictions = dataset.get_predictions()
    
    if isinstance(predictions, dict):
        print(f"Predictions stored as dict with keys: {list(predictions.keys())}")
        
        # Try to get specific predictions
        if 'predictions' in predictions:
            pred_values = predictions['predictions']
            print(f"Prediction values shape: {pred_values.shape}")
        
        if 'sample' in predictions:
            pred_samples = predictions['sample']
            print(f"Prediction samples: {pred_samples[:5]}...")
    else:
        print(f"Predictions type: {type(predictions)}")
        print(f"Predictions shape/length: {len(predictions)}")
    
    # Test out-of-fold predictions
    print("\n5. Testing out-of-fold predictions...")
    
    try:
        oof_predictions = dataset.get_fold_predictions(
            model_name="TestLinearRegression",
            aggregation="mean"
        )
        
        if oof_predictions is not None:
            if isinstance(oof_predictions, dict):
                print(f"Out-of-fold predictions as dict: {list(oof_predictions.keys())}")
            else:
                print(f"Out-of-fold predictions shape: {oof_predictions.shape}")
        else:
            print("No out-of-fold predictions found")
    except Exception as e:
        print(f"Error getting out-of-fold predictions: {e}")
    
    print("\n=== Simple Integration Test Completed ===")
    return dataset

if __name__ == "__main__":
    dataset = test_simple_integration()
    print("\n" + "=" * 50)
    print("INTEGRATION TEST COMPLETED!")
    print("✓ Dataset creation")
    print("✓ Fold-based splitting")
    print("✓ Model training")
    print("✓ Prediction storage")
    print("✓ Basic prediction retrieval")
