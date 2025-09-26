"""
Test script to demonstrate the new fold-aware model training functionality.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Mock classes to simulate the nirs4all structure
class MockDataset:
    def __init__(self):
        self.num_folds = 3
        self.folds = [(np.array([0, 1, 2, 3]), np.array([4, 5])),
                      (np.array([0, 1, 4, 5]), np.array([2, 3])),
                      (np.array([2, 3, 4, 5]), np.array([0, 1]))]

        # Generate sample data
        self._X = np.random.rand(6, 10)  # 6 samples, 10 features
        self._y = np.random.rand(6)

    def x(self, context, layout="2d", concat_source=True):
        partition = context.get("partition", "train")
        if partition == "train":
            return self._X  # Return all training data
        else:
            return self._X[:2]  # Return test data

    def y(self, context):
        partition = context.get("partition", "train")
        if partition == "train":
            return self._y  # Return all training labels
        else:
            return self._y[:2]  # Return test labels

class MockSklearnController:
    def get_preferred_layout(self):
        return "2d"

    def _prepare_train_test_data(self, dataset, context):
        """
        Simulate the new _prepare_train_test_data method behavior
        """
        layout = self.get_preferred_layout()

        # Check if dataset has folds
        if hasattr(dataset, 'num_folds') and dataset.num_folds > 0:
            # Prepare fold-based train/validation splits
            folds_data = []

            # Get all training data first
            train_context = context.copy()
            train_context["partition"] = "train"
            X_all_train = dataset.x(train_context, layout, concat_source=True)
            y_all_train = dataset.y(train_context)

            # For each fold, create train/validation splits
            for fold_idx, (train_indices, val_indices) in enumerate(dataset.folds):
                X_train_fold = X_all_train[train_indices]
                y_train_fold = y_all_train[train_indices]
                X_val_fold = X_all_train[val_indices]
                y_val_fold = y_all_train[val_indices]

                folds_data.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
                print(f"📊 Fold {fold_idx+1}: Train {X_train_fold.shape}, Val {X_val_fold.shape}")

            return folds_data
        else:
            # No folds: use standard train/test split
            train_context = context.copy()
            train_context["partition"] = "train"
            X_train = dataset.x(train_context, layout, concat_source=True)
            y_train = dataset.y(train_context)

            test_context = context.copy()
            test_context["partition"] = "test"
            X_test = dataset.x(test_context, layout, concat_source=True)
            y_test = dataset.y(test_context)

            print(f"📊 No folds - Train: X{X_train.shape}, y{y_train.shape} | Test: X{X_test.shape}, y{y_test.shape}")
            return X_train, y_train, X_test, y_test

def test_fold_handling():
    """Test that the new fold handling works correctly"""
    print("🧪 Testing new fold-aware model training...")

    # Create mock dataset with folds
    dataset = MockDataset()
    controller = MockSklearnController()
    context = {"processing": [["raw"]]}

    print(f"Dataset has {dataset.num_folds} folds")

    # Test the new _prepare_train_test_data method
    data_splits = controller._prepare_train_test_data(dataset, context)

    # Check if we got folds (list) or single split (tuple)
    if isinstance(data_splits, list):
        print(f"✅ Cross-validation mode: Got {len(data_splits)} folds")

        for i, (X_train, y_train, X_val, y_val) in enumerate(data_splits):
            print(f"  Fold {i+1}: Train samples={len(X_train)}, Val samples={len(X_val)}")

            # Train a model on this fold
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            val_score = model.score(X_val, y_val)
            print(f"    Model R² score: {val_score:.4f}")
    else:
        print("❌ Expected folds but got single split")

    # Test with dataset without folds
    print("\n🧪 Testing without folds...")
    dataset.num_folds = 0
    dataset.folds = []

    data_splits = controller._prepare_train_test_data(dataset, context)
    if not isinstance(data_splits, list):
        X_train, y_train, X_test, y_test = data_splits
        print(f"✅ Single training mode: Train {X_train.shape}, Test {X_test.shape}")
    else:
        print("❌ Expected single split but got folds")

if __name__ == "__main__":
    test_fold_handling()
    print("\n🎉 Fold handling tests completed!")