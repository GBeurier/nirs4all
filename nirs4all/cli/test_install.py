"""
Installation testing utilities for nirs4all CLI.
"""

import sys
import importlib
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
    print("🔍 Testing NIRS4ALL Installation...")
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
    print(f"✓ Python: {python_version}")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python version {python_version} is not supported (requires >=3.7)")
        return False
    
    print()
    
    # Test required dependencies
    print("📦 Required Dependencies:")
    all_required_ok = True
    
    for dep_name, min_version in required_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        status = "✓" if is_available else "❌"
        print(f"  {status} {dep_name}: {version}")
        
        if not is_available:
            all_required_ok = False
    
    print()
    
    # Test optional dependencies
    print("🔧 Optional ML Frameworks:")
    optional_available = {}
    
    for dep_name, min_version in optional_deps.items():
        is_available, version = check_dependency(dep_name, min_version)
        status = "✓" if is_available else "⚠️"
        print(f"  {status} {dep_name}: {version}")
        optional_available[dep_name] = is_available
    
    print()
    
    # Test nirs4all itself
    print("🎯 NIRS4ALL Components:")
    try:
        from nirs4all.utils.backend_utils import (
            is_tensorflow_available, is_torch_available,
            is_keras_available, is_jax_available
        )
        print("  ✓ nirs4all.utils.backend_utils: OK")
        
        from nirs4all.core.runner import ExperimentRunner
        print("  ✓ nirs4all.core.runner: OK")
        
        from nirs4all.data.dataset_loader import get_dataset
        print("  ✓ nirs4all.data.dataset_loader: OK")
        
        from nirs4all.transformations import StandardNormalVariate, SavitzkyGolay
        print("  ✓ nirs4all.transformations: OK")
        
    except ImportError as e:
        print(f"  ❌ nirs4all import error: {e}")
        all_required_ok = False
    
    print()
    
    # Summary
    if all_required_ok:
        print("🎉 Basic installation test PASSED!")
        print(f"✓ All required dependencies are available")
        
        available_frameworks = [name for name, available in optional_available.items() if available]
        if available_frameworks:
            print(f"✓ Available ML frameworks: {', '.join(available_frameworks)}")
        else:
            print("⚠️  No optional ML frameworks detected")
            
        return True
    else:
        print("❌ Basic installation test FAILED!")
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