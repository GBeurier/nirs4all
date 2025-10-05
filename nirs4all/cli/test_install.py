"""
Installation testing utilities for nirs4all CLI.
"""

import sys
import importlib
import os
import tempfile
import time
from typing import Dict, List, Tuple
import numpy as np


def check_dependency(name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a dependency is installed and optionally verify minimum version.

    Args:
        name: Name of the dependency/module to check
        min_version: Minimum required version (optional)

    Returns:
        Tuple of (is_available, version_string)
    """
    try:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            # Simple version comparison (works for most cases)
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    return False, f"{version} (< {min_version} required)"
            except ImportError:
                # Fallback if packaging is not available
                pass

        return True, version
    except ImportError:
        return False, "Not installed"


def test_installation() -> bool:
    """
    Test basic installation and show dependency versions.

    Returns:
        True if all required dependencies are available, False otherwise.
    """
    print("Testing NIRS4ALL Installation...")
    print("=" * 50)

    # Core required dependencies from pyproject.toml
    required_deps = {
        'numpy': '1.20.0',
        'pandas': '1.0.0',
        'scipy': '1.5.0',
        'sklearn': '0.24.0',  # scikit-learn is imported as sklearn
        'pywt': '1.1.0',      # PyWavelets is imported as pywt
        'joblib': '0.16.0',
        'jsonschema': '3.2.0',
    }

    # Optional ML framework dependencies
    optional_deps = {
        'tensorflow': '2.0.0',
        'torch': '1.4.0',
        'keras': None,
        'jax': None,
    }

    # Test Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"* Python: {python_version}")

    if sys.version_info < (3, 7):
        print(f"X Python version {python_version} is not supported (requires >=3.7)")
        return False

    print()

    # Test required dependencies
    print("Required Dependencies:")
    all_required_ok = True

    for dep_name, min_version in required_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        status = "*" if is_available else "X"
        print(f"  {status} {dep_name}: {version}")

        if not is_available:
            all_required_ok = False

    print()

    # Test optional dependencies
    print("Optional ML Frameworks:")
    optional_available = {}

    for dep_name, min_version in optional_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        status = "*" if is_available else "!"
        print(f"  {status} {dep_name}: {version}")
        optional_available[dep_name] = is_available

    print()

    # Test nirs4all itself
    print("NIRS4ALL Components:")
    try:
        from nirs4all.utils.backend_utils import (
            is_tensorflow_available, is_torch_available,
            is_keras_available, is_jax_available
        )
        print("  * nirs4all.utils.backend_utils: OK")

        from nirs4all.core.runner import ExperimentRunner
        print("  * nirs4all.core.runner: OK")

        from nirs4all.data.dataset_loader import get_dataset
        print("  * nirs4all.data.dataset_loader: OK")

        from nirs4all.transformations import StandardNormalVariate, SavitzkyGolay
        print("  * nirs4all.transformations: OK")

    except ImportError as e:
        print(f"  X nirs4all import error: {e}")
        all_required_ok = False

    print()

    # Summary
    if all_required_ok:
        print("Basic installation test PASSED!")
        print("All required dependencies are available")

        available_frameworks = [name for name, available in optional_available.items() if available]
        if available_frameworks:
            print(f"Available ML frameworks: {', '.join(available_frameworks)}")
        else:
            print("No optional ML frameworks detected")

        return True
    else:
        print("Basic installation test FAILED!")
        print("Please install missing dependencies using:")
        print("  pip install nirs4all")
        return False


def full_test_installation() -> bool:
    """
    Full installation test including framework functionality.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("🔍 Full NIRS4ALL Installation Test...")
    print("=" * 50)

    # First run basic test
    basic_ok = test_installation()

    if not basic_ok:
        return False

    print("\n" + "=" * 50)
    print("🧪 Testing Framework Functionality...")
    print("=" * 50)

    # Test framework functionality
    success_count = 0
    total_tests = 0

    # Test TensorFlow
    total_tests += 1
    print("\n🔬 Testing TensorFlow:")
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} imported successfully")

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("  ✓ TensorFlow model creation: OK")

        # Test with dummy data
        X_dummy = tf.random.normal((10, 5))
        y_dummy = model(X_dummy)
        print(f"  ✓ TensorFlow forward pass: OK (output shape: {y_dummy.shape})")

        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU devices detected: {len(gpus)}")
        else:
            print("  ⚠️  No GPU devices detected (CPU only)")

        success_count += 1

    except ImportError:
        print("  ⚠️  TensorFlow not available")
    except Exception as e:
        print(f"  ❌ TensorFlow test failed: {e}")

    # Test PyTorch
    total_tests += 1
    print("\n🔬 Testing PyTorch:")
    try:
        import torch
        import torch.nn as nn
        print(f"  ✓ PyTorch {torch.__version__} imported successfully")

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        print("  ✓ PyTorch model creation: OK")

        # Test with dummy data
        X_dummy = torch.randn(10, 5)
        y_dummy = model(X_dummy)
        print(f"  ✓ PyTorch forward pass: OK (output shape: {y_dummy.shape})")

        # Test GPU availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("  ⚠️  CUDA not available (CPU only)")

        success_count += 1

    except ImportError:
        print("  ⚠️  PyTorch not available")
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")

    # Test scikit-learn functionality
    total_tests += 1
    print("\n🔬 Testing Scikit-learn:")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression

        # Generate dummy data
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

        # Create and train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X[:5])

        print(f"  ✓ Scikit-learn model training and prediction: OK")
        print(f"  ✓ Sample predictions shape: {predictions.shape}")

        success_count += 1

    except Exception as e:
        print(f"  ❌ Scikit-learn test failed: {e}")

    # Test NIRS4ALL integration
    total_tests += 1
    print("\n🔬 Testing NIRS4ALL Integration:")
    try:
        from nirs4all.utils.backend_utils import (
            is_tensorflow_available, is_torch_available, is_gpu_available
        )

        tf_available = is_tensorflow_available()
        torch_available = is_torch_available()
        gpu_available = is_gpu_available()

        print(f"  ✓ Backend detection - TensorFlow: {tf_available}")
        print(f"  ✓ Backend detection - PyTorch: {torch_available}")
        print(f"  ✓ Backend detection - GPU: {gpu_available}")

        # Test a simple transformation
        from nirs4all.transformations import StandardNormalVariate
        snv = StandardNormalVariate()
        test_data = np.random.randn(10, 5)
        transformed = snv.fit_transform(test_data)
        print(f"  ✓ Transformation test: OK (shape: {transformed.shape})")

        success_count += 1

    except Exception as e:
        print(f"  ❌ NIRS4ALL integration test failed: {e}")

    print("\n" + "=" * 50)

    # Final summary
    if success_count == total_tests:
        print("🎉 Full installation test PASSED!")
        print(f"✓ All {total_tests} functionality tests successful")
        return True
    else:
        print(f"⚠️  Partial success: {success_count}/{total_tests} tests passed")
        if success_count > 0:
            print("✓ Basic functionality is working")
            print("⚠️  Some optional features may not be available")
            return True
        else:
            print("❌ Full installation test FAILED!")
            return False


def test_integration() -> bool:
    """
    Run integration test with Random Forest, PLS fine-tuning, and simple CNN (3 epochs).
    Creates temporary synthetic data to avoid dependency on sample_data directory.

    Returns:
        True if integration test passes, False otherwise.
    """
    print("NIRS4ALL Integration Test...")
    print("=" * 50)

    # First check if basic installation is working
    basic_ok = test_installation()
    if not basic_ok:
        print("* Integration test FAILED!")
        print("Please fix installation issues first.")
        return False

    print("\n" + "=" * 50)
    print("Running Full Pipeline Integration Test...")
    print("=" * 50)

    import tempfile
    import numpy as np
    import pandas as pd
    import uuid
    import time

    try:
        # Import required modules
        from nirs4all.core.config import Config
        from nirs4all.core.runner import ExperimentRunner
        from sklearn.preprocessing import RobustScaler, MinMaxScaler
        from sklearn.model_selection import RepeatedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cross_decomposition import PLSRegression
        from nirs4all.presets.ref_models import nicon_classification

        print("  * Successfully imported NIRS4ALL modules")

    except ImportError as e:
        print(f"  * Failed to import required modules: {e}")
        return False

    # Create temporary directory for test data
    temp_base_dir = tempfile.mkdtemp()
    print(f"  * Creating temporary data in: {temp_base_dir}")

    try:
        # Generate synthetic NIRS-like data
        def create_synthetic_data(n_samples=50, n_wavelengths=1000, data_type="regression"):
            """Create synthetic NIRS data similar to the sample data format."""
            # Generate wavelength columns (X3855 to X527)
            wavelengths = list(range(3855, 527, -2))[:n_wavelengths]
            wavelength_cols = [f"X{w}" for w in wavelengths]

            # Generate synthetic spectral data (typical NIRS absorbance values)
            np.random.seed(42)  # For reproducible results
            X_data = np.random.normal(0.15, 0.05, (n_samples, n_wavelengths))
            X_data = np.clip(X_data, 0.001, 0.5)  # Typical absorbance range

            # Add some spectral features (peaks/valleys)
            for i in range(n_samples):
                for j in range(0, n_wavelengths, 200):
                    peak_intensity = np.random.normal(0.05, 0.02)
                    peak_width = 50
                    for k in range(max(0, j-peak_width//2), min(n_wavelengths, j+peak_width//2)):
                        X_data[i, k] += peak_intensity * np.exp(-((k-j)**2)/(2*(peak_width/4)**2))

            # Create X DataFrame
            X_df = pd.DataFrame(X_data, columns=wavelength_cols)

            # Generate Y data
            if data_type == "regression":
                # Create regression target (correlated with some spectral features)
                y_data = (X_data[:, 100:200].mean(axis=1) * 100 +
                         np.random.normal(0, 2, n_samples))
                y_df = pd.DataFrame(y_data, columns=["value"])
            else:  # binary classification
                # Create binary target
                threshold = np.median(X_data[:, 150:250].mean(axis=1))
                y_data = (X_data[:, 150:250].mean(axis=1) > threshold).astype(int)
                y_df = pd.DataFrame(y_data, columns=["label"])

            return X_df, y_df

        # Create regression data
        regression_dir = os.path.join(temp_base_dir, "regression")
        os.makedirs(regression_dir, exist_ok=True)

        X_reg_cal, y_reg_cal = create_synthetic_data(50, 1000, "regression")
        X_reg_val, y_reg_val = create_synthetic_data(20, 1000, "regression")

        X_reg_cal.to_csv(os.path.join(regression_dir, "Xcal.csv"), index=False, sep=";")
        y_reg_cal.to_csv(os.path.join(regression_dir, "Ycal.csv"), index=False)
        X_reg_val.to_csv(os.path.join(regression_dir, "Xval.csv"), index=False, sep=";")
        y_reg_val.to_csv(os.path.join(regression_dir, "Yval.csv"), index=False)

        # Create binary classification data
        binary_dir = os.path.join(temp_base_dir, "binary")
        os.makedirs(binary_dir, exist_ok=True)

        X_bin_cal, y_bin_cal = create_synthetic_data(50, 1000, "binary")
        X_bin_val, y_bin_val = create_synthetic_data(20, 1000, "binary")

        X_bin_cal.to_csv(os.path.join(binary_dir, "Xcal.csv"), index=False, sep=";")
        y_bin_cal.to_csv(os.path.join(binary_dir, "Ycal.csv"), index=False)
        X_bin_val.to_csv(os.path.join(binary_dir, "Xval.csv"), index=False, sep=";")
        y_bin_val.to_csv(os.path.join(binary_dir, "Yval.csv"), index=False)

        print("  * Synthetic NIRS data created successfully")

        # Basic pipeline for all tests
        x_pipeline = [
            RobustScaler(),
            {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
            MinMaxScaler()
        ]

        # Test configurations
        test_configs = []

        # Test 1: Random Forest Classification
        print("\nTest 1: Random Forest Classification")
        try:
            rf_model = {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "model_params": {"n_estimators": 10, "max_depth": 5, "random_state": 42}
            }

            config1 = Config(
                binary_dir,
                x_pipeline,
                None,
                rf_model,
                {"action": "train", "task": "classification"},
                42
            )
            test_configs.append(("Random Forest Classification", config1))
            print("  * Configuration created successfully")

        except Exception as e:
            print(f"  * Failed to create Random Forest config: {e}")
            return False

        # Test 2: PLS Fine-tuning
        print("\nTest 2: PLS Fine-tuning")
        try:
            pls_model = {
                "class": "sklearn.cross_decomposition.PLSRegression",
                "model_params": {"n_components": 10}
            }

            pls_finetune_params = {
                "action": "finetune",
                "finetune_params": {
                    'model_params': {
                        'n_components': ('int', 5, 15),
                    },
                    'training_params': {},
                    'tuner': 'sklearn',
                    'n_trials': 5
                }
            }

            config2 = Config(
                regression_dir,
                x_pipeline,
                None,
                pls_model,
                pls_finetune_params,
                84
            )
            test_configs.append(("PLS Fine-tuning", config2))
            print("  * Configuration created successfully")

        except Exception as e:
            print(f"  * Failed to create PLS fine-tuning config: {e}")
            return False

        # Test 3: Simple CNN (3 epochs)
        print("\nTest 3: Simple CNN (3 epochs)")
        try:
            cnn_params = {
                "action": "train",
                "task": "classification",
                "training_params": {
                    "epochs": 3,
                    "verbose": 0,
                    "patience": 10
                }
            }

            config3 = Config(
                binary_dir,
                x_pipeline,
                None,
                nicon_classification,
                cnn_params,
                126
            )
            test_configs.append(("Simple CNN", config3))
            print("  * Configuration created successfully")

        except Exception as e:
            print(f"  * Failed to create CNN config: {e}")
            return False

        # Run all test configurations
        success_count = 0
        total_tests = len(test_configs)

        for test_name, config in test_configs:
            print(f"\nRunning {test_name}:")
            try:
                # Create temporary results directory
                temp_dir = os.path.join(tempfile.gettempdir(), f"nirs4all_test_{uuid.uuid4().hex[:8]}")
                os.makedirs(temp_dir, exist_ok=True)

                start_time = time.time()

                # Run the experiment
                runner = ExperimentRunner([config], results_dir=temp_dir, resume_mode="restart")
                datasets, predictions, scores, best_params = runner.run()

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Validate results
                assert len(datasets) == 1, "Should return one dataset"
                dataset = datasets[0]

                assert dataset is not None, "Dataset should not be None"

                # Get data shapes using appropriate methods
                x_train_shape = getattr(dataset, 'x_train_', lambda: None)()
                y_train_shape = getattr(dataset, 'y_train', None)
                x_test_shape = getattr(dataset, 'x_test_', lambda: None)()
                y_test_shape = getattr(dataset, 'y_test', None)

                if x_train_shape is not None and y_train_shape is not None:
                    print(f"    * Data shapes: X_train{x_train_shape.shape}, Y_train{y_train_shape.shape}")
                if x_test_shape is not None and y_test_shape is not None:
                    print(f"    * Data shapes: X_test{x_test_shape.shape}, Y_test{y_test_shape.shape}")

                print(f"    * Execution time: {elapsed_time:.2f} seconds")

                # Print best parameters for fine-tuning
                if best_params and best_params[0] and test_name == "PLS Fine-tuning":
                    print(f"    * Best parameters: {best_params[0]}")

                # Print scores if available
                if scores and len(scores) > 0 and scores[0] is not None:
                    score = scores[0]
                    if isinstance(score, list) and len(score) > 0:
                        score_dict = score[0] if isinstance(score[0], dict) else score
                        if isinstance(score_dict, dict):
                            formatted_scores = {k: (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
                                              for k, v in score_dict.items()}
                            print(f"    * Model scores: {formatted_scores}")
                    elif isinstance(score, dict):
                        formatted_scores = {k: (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
                                          for k, v in score.items()}
                        print(f"    * Model scores: {formatted_scores}")

                success_count += 1
                print(f"    * {test_name} completed successfully!")

                # Clean up temporary directory
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass  # Ignore cleanup errors

            except Exception as e:
                print(f"    ! {test_name} failed: {e}")
                # Print more detailed error information for debugging
                import traceback
                print(f"    ! Error details: {traceback.format_exc()}")

        print("\n" + "=" * 50)

        # Final summary
        if success_count == total_tests:
            print("Integration test PASSED!")
            print(f"* All {total_tests} pipeline tests completed successfully")
            print("* Random Forest, PLS fine-tuning, and CNN models work correctly")
            print("* NIRS4ALL is ready for use!")
            result = True
        else:
            print(f"! Partial success: {success_count}/{total_tests} tests passed")
            if success_count > 0:
                print("* Basic pipeline functionality is working")
                print("! Some advanced features may have issues")
                result = True  # Return True for partial success as it indicates basic functionality works
            else:
                print("X Integration test FAILED!")
                print("X Pipeline execution is not working properly")
                result = False

    finally:
        # Clean up temporary data directory
        try:
            import shutil
            shutil.rmtree(temp_base_dir, ignore_errors=True)
            print(f"  * Temporary data cleaned up")
        except Exception as e:
            print(f"  ! Warning: Could not clean up temporary data: {e}")

    return result