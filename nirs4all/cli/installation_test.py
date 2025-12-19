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

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


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
    logger.info("Testing NIRS4ALL Installation...")
    logger.info("=" * 50)

    # Core required dependencies from pyproject.toml
    required_deps = {
        'numpy': '1.20.0',
        'pandas': '2.0.0',
        'scipy': '1.5.0',
        'sklearn': '0.24.0',  # scikit-learn is imported as sklearn
        'pywt': '1.1.0',      # PyWavelets is imported as pywt
        'joblib': '0.16.0',
        'jsonschema': '3.2.0',
        'optuna': '2.0.0',
        'matplotlib': '3.0.0',
        'polars': '0.18.0',
        'yaml': '5.4.0',      # pyyaml is imported as yaml
        'seaborn': '0.11.0',
        'h5py': '3.0.0',
        'packaging': '20.0',
        'shap': '0.41.0',
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
    logger.success(f"Python: {python_version}")

    if sys.version_info < (3, 7):
        logger.error(f"Python version {python_version} is not supported (requires >=3.7)")
        return False

    logger.info("")

    # Test required dependencies
    logger.info("\nRequired Dependencies:")
    all_required_ok = True

    for dep_name, min_version in required_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        if is_available:
            logger.success(f"  {dep_name}: {version}")
        else:
            logger.error(f"  {dep_name}: {version}")
            all_required_ok = False

    logger.info("")

    # Test optional dependencies
    logger.info("\nOptional ML Frameworks:")
    optional_available = {}

    for dep_name, min_version in optional_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        if is_available:
            logger.success(f"  {dep_name}: {version}")
        else:
            logger.warning(f"  {dep_name}: {version}")
        optional_available[dep_name] = is_available

    logger.info("")

    # Test nirs4all itself
    logger.info("\nNIRS4ALL Components:")
    try:
        # Test core pipeline components
        from nirs4all.pipeline.runner import PipelineRunner
        logger.success("  nirs4all.pipeline.runner: OK")

        from nirs4all.data.dataset import SpectroDataset
        logger.success("  nirs4all.dataset.dataset: OK")

        # Test controller system
        from nirs4all.controllers import register_controller, CONTROLLER_REGISTRY
        logger.success(f"  nirs4all.controllers: OK ({len(CONTROLLER_REGISTRY)} controllers registered)")

        # Test operators
        from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay
        logger.success("  nirs4all.operators.transforms: OK")

        # Test backend utils
        from nirs4all.utils.backend import (
            is_tensorflow_available, is_torch_available,
            is_gpu_available
        )
        logger.success("  nirs4all.utils.backend_utils: OK")

    except ImportError as e:
        logger.error(f"  nirs4all import error: {e}")
        all_required_ok = False

    logger.info("")

    # Summary
    if all_required_ok:
        logger.success("Basic installation test PASSED!")
        logger.info("All required dependencies are available")

        available_frameworks = [name for name, available in optional_available.items() if available]
        if available_frameworks:
            logger.info(f"Available ML frameworks: {', '.join(available_frameworks)}")
        else:
            logger.info("No optional ML frameworks detected")

        return True
    else:
        logger.error("Basic installation test FAILED!")
        logger.info("Please install missing dependencies using:")
        logger.info("  pip install nirs4all")
        return False


def test_integration() -> bool:
    """
    Run integration test with sklearn, tensorflow, and optuna pipelines.
    Based on examples Q1.py, Q1_finetune.py, Q2.py but using synthetic data.
    Monitors execution time of each test.

    Returns:
        True if integration test passes, False otherwise.
    """
    logger.info("NIRS4ALL Integration Test...")
    logger.info("=" * 50)

    # # First check if basic installation is working
    # basic_ok = test_installation()
    # if not basic_ok:
    #     logger.error("Integration test FAILED!")
    #     logger.info("Please fix installation issues first.")
    #     return False

    logger.info("\n" + "=" * 50)
    logger.info("Running Pipeline Integration Tests...")
    logger.info("=" * 50)

    # Store test results with timing
    test_results = []

    try:
        # Import required modules based on examples
        from nirs4all.pipeline import PipelineConfigs, PipelineRunner
        from nirs4all.data import DatasetConfigs
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import ShuffleSplit
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cross_decomposition import PLSRegression
        from nirs4all.operators.transforms import StandardNormalVariate

        logger.success("Successfully imported NIRS4ALL modules")

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False

    def create_synthetic_dataset_files(temp_dir, task_type="regression", n_samples=100, n_features=500):
        """Create synthetic CSV files matching the expected format."""
        np.random.seed(42)

        # Create realistic spectral-like data
        X = np.random.normal(0.1, 0.05, (n_samples, n_features))
        X = np.clip(X, 0.001, 0.5)  # Typical absorbance range

        # Add some spectral structure
        for i in range(0, n_features, 100):
            peak_width = 20
            for j in range(max(0, i - peak_width), min(n_features, i + peak_width)):
                X[:, j] += 0.02 * np.exp(-((j - i) ** 2) / (2 * (peak_width / 3) ** 2))

        # Create wavelength-like column names
        wavelengths = np.linspace(2500, 400, n_features)  # Typical NIR range
        columns = [f"X{int(w)}" for w in wavelengths]

        # Split data
        n_train = int(0.7 * n_samples)
        X_train, X_test = X[:n_train], X[n_train:]

        if task_type == "regression":
            # Regression target correlated with spectral features
            y = (X[:, 100:200].mean(axis=1) * 100 +
                 np.random.normal(0, 2, n_samples))
        else:  # classification
            # Binary target based on spectral threshold
            threshold = np.median(X[:, 150:250].mean(axis=1))
            y = (X[:, 150:250].mean(axis=1) > threshold).astype(int)

        y_train, y_test = y[:n_train], y[n_train:]

        # Create CSV files
        import pandas as pd

        # Training data
        pd.DataFrame(X_train, columns=columns).to_csv(
            os.path.join(temp_dir, "Xcal.csv"), index=False, sep=";"
        )
        pd.DataFrame(y_train, columns=["value" if task_type == "regression" else "label"]).to_csv(
            os.path.join(temp_dir, "Ycal.csv"), index=False
        )

        # Test data
        pd.DataFrame(X_test, columns=columns).to_csv(
            os.path.join(temp_dir, "Xval.csv"), index=False, sep=";"
        )
        pd.DataFrame(y_test, columns=["value" if task_type == "regression" else "label"]).to_csv(
            os.path.join(temp_dir, "Yval.csv"), index=False
        )

        return temp_dir

    def run_test(test_name, test_func):
        """Run a test with timing and error handling."""
        logger.info(f"\nTest: {test_name}")
        start_time = time.time()

        try:
            success = test_func()
            end_time = time.time()
            elapsed = end_time - start_time

            if success:
                logger.success(f"  {test_name} completed successfully ({elapsed:.2f}s)")
                test_results.append((test_name, True, elapsed, None))
                return True
            else:
                logger.error(f"  {test_name} failed ({elapsed:.2f}s)")
                test_results.append((test_name, False, elapsed, "Test function returned False"))
                return False

        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            logger.error(f"  {test_name} failed with error ({elapsed:.2f}s): {e}")
            test_results.append((test_name, False, elapsed, str(e)))
            return False

    def test_sklearn_pipeline():
        """Test sklearn-based pipeline (based on Q2.py) - Extended version."""
        # Create temporary dataset with more samples for thorough testing
        temp_dir = tempfile.mkdtemp()
        try:
            create_synthetic_dataset_files(temp_dir, "regression", 120, 500)  # More samples and features

            # Extended pipeline based on Q2.py example with more models
            pipeline = [
                MinMaxScaler(feature_range=(0.1, 0.8)),
                StandardNormalVariate(),
                ShuffleSplit(n_splits=4),  # More folds
                {"model": PLSRegression(n_components=2)},
                {"model": PLSRegression(n_components=3)},
                {"model": PLSRegression(n_components=4)},
                {"model": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)},
                {"model": RandomForestRegressor(n_estimators=30, max_depth=5, random_state=123)},
            ]

            pipeline_config = PipelineConfigs(pipeline, "sklearn_test")
            dataset_config = DatasetConfigs(temp_dir)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            # Verify results
            assert predictions is not None, "No predictions returned"
            num_predictions = predictions.num_predictions
            logger.info(f"    Pipeline executed successfully, {num_predictions} predictions generated")

            # Additional validation
            assert num_predictions >= 10, f"Expected at least 10 predictions, got {num_predictions}"

            return True

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_tensorflow_pipeline():
        """Test TensorFlow-based pipeline (based on Q2.py)."""
        try:
            import tensorflow as tf
            from nirs4all.operators.models.tensorflow.nicon import nicon
        except ImportError:
            logger.warning("    TensorFlow/NIRS models not available, skipping test")
            return True  # Skip but don't fail

        # Create temporary dataset
        temp_dir = tempfile.mkdtemp()
        try:
            create_synthetic_dataset_files(temp_dir, "regression", 60, 300)

            # Pipeline based on Q2.py example
            pipeline = [
                MinMaxScaler(),
                StandardNormalVariate(),
                ShuffleSplit(n_splits=2),  # Fewer splits for speed
                {
                    "model": nicon,
                    "train_params": {
                        "epochs": 3,  # Very few epochs for speed
                        "patience": 10,
                        "verbose": 0
                    },
                },
            ]

            pipeline_config = PipelineConfigs(pipeline, "tensorflow_test")
            dataset_config = DatasetConfigs(temp_dir)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            # Verify results
            assert predictions is not None, "No predictions returned"
            logger.info("    TensorFlow model trained successfully")

            return True

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_optuna_pipeline():
        """Test Optuna hyperparameter optimization (based on Q1_finetune.py) - Extended version."""
        try:
            import optuna
        except ImportError:
            logger.warning("    Optuna not available, skipping test")
            return True  # Skip but don't fail

        # Create temporary dataset with more samples
        temp_dir = tempfile.mkdtemp()
        try:
            create_synthetic_dataset_files(temp_dir, "regression", 100, 400)  # More samples

            # Extended pipeline based on Q1_finetune.py example with more comprehensive optimization
            pipeline = [
                MinMaxScaler(),
                StandardNormalVariate(),
                ShuffleSplit(n_splits=3),  # Keep splits moderate for speed
                {
                    "model": PLSRegression(),
                    "name": "PLS-Finetuned-Extended",
                    "finetune_params": {
                        "n_trials": 15,  # More trials for better optimization
                        "verbose": 0,
                        "approach": "grouped",  # Test grouped approach
                        "eval_mode": "best",
                        "model_params": {
                            'n_components': ('int', 1, 5),  # Safe range for small training sets
                        },
                    }
                },
                # Add a second optimization test
                {
                    "model": PLSRegression(),
                    "name": "PLS-Single-Optim",
                    "finetune_params": {
                        "n_trials": 25,  # More trials for comprehensive optimization
                        "verbose": 0,
                        "approach": "single",  # Test single approach
                        "model_params": {
                            'n_components': ('int', 1, 5),  # Safe range for small training sets
                        },
                    }
                },
            ]

            pipeline_config = PipelineConfigs(pipeline, "optuna_test")
            dataset_config = DatasetConfigs(temp_dir)

            runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
            predictions, _ = runner.run(pipeline_config, dataset_config)

            # Verify results
            assert predictions is not None, "No predictions returned"
            num_predictions = predictions.num_predictions
            logger.info(f"    Optuna optimization completed, {num_predictions} predictions generated")

            # Additional validation - should have predictions from both optimizations
            assert num_predictions >= 4, f"Expected at least 4 predictions from optimization, got {num_predictions}"

            return True

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)    # Run all tests
    tests = [
        ("Sklearn Extended Pipeline (Multiple PLS + RandomForest)", test_sklearn_pipeline),
        ("TensorFlow Pipeline (NICON Neural Network)", test_tensorflow_pipeline),
        ("Optuna Extended Pipeline (Comprehensive PLS Optimization)", test_optuna_pipeline),
    ]

    success_count = 0
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            success_count += 1

    # Print summary with timing
    logger.info("\n" + "=" * 50)
    logger.info("Integration Test Summary")
    logger.info("=" * 50)

    total_time = sum(result[2] for result in test_results)

    for name, success, elapsed, error in test_results:
        if success:
            logger.success(f"PASS {name}: {elapsed:.2f}s")
        else:
            logger.error(f"FAIL {name}: {elapsed:.2f}s")
        if error and not success:
            logger.info(f"     Error: {error}")

    logger.info(f"\nTotal execution time: {total_time:.2f}s")

    if success_count == len(tests):
        logger.success("Integration test PASSED!")
        logger.success(f"All {len(tests)} pipeline tests completed successfully")
        logger.info("NIRS4ALL is ready for use!")
        return True
    else:
        logger.warning(f"Partial success: {success_count}/{len(tests)} tests passed")
        if success_count > 0:
            logger.success("Basic pipeline functionality is working")
            logger.warning("Some optional features may have issues")
            return True  # Return True for partial success
        else:
            logger.error("Integration test FAILED!")
            logger.error("Pipeline execution is not working properly")
            return False







