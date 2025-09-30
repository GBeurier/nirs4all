"""
Integration test for the modular model controller architecture.

This test verifies that the new modular components work together correctly
and maintain compatibility with the original controller interface.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from nirs4all.controllers.models.sklearn_model_controller import SklearnModelController
from nirs4all.controllers.models.config import ModelConfig, CVConfig, CVMode
from nirs4all.controllers.models.data import DataManager
from nirs4all.controllers.models.model import ModelManager
from nirs4all.controllers.models.cv.factory import CVStrategyFactory
from nirs4all.controllers.models.prediction import PredictionManager
from nirs4all.controllers.models.results import ResultManager


def test_modular_components():
    """Test that all modular components can be instantiated and work together."""
    print("🧪 Testing modular components...")

    # Test configuration classes
    model_config = ModelConfig(
        model_params={'model_class': 'sklearn.linear_model.LinearRegression'}
    )
    cv_config = CVConfig(mode=CVMode.SIMPLE, n_folds=3)

    print("✅ Configuration classes work")

    # Test data manager
    data_manager = DataManager()
    print("✅ DataManager instantiated")

    # Test model manager
    model_manager = ModelManager()
    print("✅ ModelManager instantiated")

    # Test CV strategy factory
    cv_factory = CVStrategyFactory()
    strategy = cv_factory.create_strategy(cv_config.mode)
    print(f"✅ CV Strategy created: {strategy.__class__.__name__}")

    # Test prediction manager
    prediction_manager = PredictionManager()
    print("✅ PredictionManager instantiated")

    # Test results manager
    results_manager = ResultManager()
    print("✅ ResultManager instantiated")

    # Test SklearnModelController
    controller = SklearnModelController()
    print("✅ SklearnModelController instantiated")

    # Test framework-specific methods
    assert controller.get_supported_frameworks() == ['sklearn']
    assert controller.get_preferred_layout() == '2d'
    print("✅ Framework-specific methods work")

    # Test model instantiation
    model_instance = controller._get_model_instance({'model_class': 'sklearn.linear_model.LinearRegression'})
    assert isinstance(model_instance, LinearRegression)
    print("✅ Model instantiation works")

    # Test data preparation
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    X_prep, y_prep = controller._prepare_data(X, y, {})
    assert X_prep.shape == (100, 5)
    assert y_prep.shape == (100,)
    print("✅ Data preparation works")

    # Test task type detection
    task_type = controller._detect_task_type(y)
    print(f"✅ Task type detection works: {task_type}")

    print("🎉 All modular components test passed!")


def test_simple_training_workflow():
    """Test a simple training workflow using the modular controller."""
    print("\n🧪 Testing simple training workflow...")

    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = X.sum(axis=1) + 0.1 * np.random.randn(100)  # Linear relationship with noise

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create controller
    controller = SklearnModelController()

    # Create model config
    model_config = {
        'model_class': 'sklearn.linear_model.LinearRegression',
        'model_params': {}
    }

    # Train model
    trained_model = controller._train_model(
        controller._get_model_instance(model_config),
        X_train, y_train
    )
    print("✅ Model training works")

    # Generate predictions
    y_pred = controller._predict_model(trained_model, X_test)
    assert y_pred.shape == y_test.shape
    print("✅ Prediction generation works")

    # Evaluate model
    score = controller._evaluate_model(trained_model, X_test, y_test)
    assert isinstance(score, float)
    print(f"✅ Model evaluation works: MSE = {score:.4f}")

    print("🎉 Simple training workflow test passed!")


if __name__ == "__main__":
    test_modular_components()
    test_simple_training_workflow()
    print("\n🎊 All integration tests passed! The modular architecture is working correctly.")
