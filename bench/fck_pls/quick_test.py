#!/usr/bin/env python
"""
Quick test to validate FCK-PLS Torch implementation.

Run from bench/fck-pls/:
    python quick_test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Local imports
from fckpls_torch import FCKPLSTorch, TrainConfig, create_fckpls_v1, create_fckpls_v2
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


def generate_simple_data(n_samples=200, n_features=500, n_components=5, noise=0.1, seed=42):
    """Generate simple synthetic spectral data."""
    rng = np.random.default_rng(seed)

    # Latent factors
    T = rng.standard_normal((n_samples, n_components))
    W = rng.standard_normal((n_components, n_features))

    # Linear relationship
    X = T @ W
    y = T[:, 0] * 2 + T[:, 1] * 1.5 + noise * rng.standard_normal(n_samples)

    # Add some spectral structure
    wavelengths = np.linspace(1000, 2500, n_features)
    for i in range(n_samples):
        X[i] += 0.1 * np.sin(wavelengths / 200 + rng.random())

    return X, y

def test_v1_basic():
    """Test V1 (learnable kernels) basic functionality."""
    print("\n" + "="*50)
    print("Test V1 (Learnable Kernels)")
    print("="*50)

    X, y = generate_simple_data(n_samples=200, n_features=500)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Create and fit model
    model = create_fckpls_v1(
        n_kernels=8,
        kernel_size=15,
        n_components=5,
        epochs=100,
        lr=1e-3,
        verbose=1,
    )

    print(f"Training on {X_train.shape}...")
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nResults V1:")
    print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Check kernels
    kernels = model.get_kernels()
    print(f"  Learned kernels shape: {kernels.shape}")

    # Transform
    T = model.transform(X_test)
    print(f"  Transform output shape: {T.shape}")

    return test_r2 > 0.5  # Basic sanity check

def test_v2_basic():
    """Test V2 (alpha/sigma parametric) basic functionality."""
    print("\n" + "="*50)
    print("Test V2 (Alpha/Sigma Parametric)")
    print("="*50)

    X, y = generate_simple_data(n_samples=200, n_features=500)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Create and fit model
    model = create_fckpls_v2(
        n_kernels=8,
        kernel_size=15,
        n_components=5,
        alpha_max=2.0,
        epochs=100,
        lr=1e-3,
        verbose=1,
    )

    print(f"Training on {X_train.shape}...")
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nResults V2:")
    print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Check learned parameters
    alphas = model.model_.extractor.get_alphas()
    sigmas = model.model_.extractor.get_sigmas()
    print(f"  Learned alphas: {alphas}")
    print(f"  Learned sigmas: {sigmas}")

    return test_r2 > 0.3  # V2 might be less stable

def test_compare_pls():
    """Compare with standard PLS."""
    print("\n" + "="*50)
    print("Comparison with Standard PLS")
    print("="*50)

    X, y = generate_simple_data(n_samples=200, n_features=500)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Standard PLS
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, y_train)
    pls_pred = pls.predict(X_test).ravel()
    pls_r2 = r2_score(y_test, pls_pred)

    # FCK-PLS Torch V1
    fckpls = create_fckpls_v1(n_kernels=16, n_components=5, epochs=150, verbose=0)
    fckpls.fit(X_train, y_train)
    fckpls_pred = fckpls.predict(X_test)
    fckpls_r2 = r2_score(y_test, fckpls_pred)

    print(f"Standard PLS R²: {pls_r2:.4f}")
    print(f"FCK-PLS Torch V1 R²: {fckpls_r2:.4f}")
    print(f"Difference: {fckpls_r2 - pls_r2:+.4f}")

    return True

def test_different_init_modes():
    """Test different kernel initialization modes."""
    print("\n" + "="*50)
    print("Test Different Initialization Modes")
    print("="*50)

    X, y = generate_simple_data(n_samples=200, n_features=500)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    for init_mode in ["random", "derivative", "fractional"]:
        cfg = TrainConfig(epochs=80, lr=1e-3, verbose=0)
        model = FCKPLSTorch(
            version="v1",
            n_kernels=8,
            n_components=5,
            init_mode=init_mode,
            train_cfg=cfg,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"  init_mode={init_mode}: R²={r2:.4f}")

    return True

def main():
    print("FCK-PLS Torch Quick Test")
    print("="*60)

    results = []

    # Test V1
    try:
        ok = test_v1_basic()
        results.append(("V1 Basic", ok))
    except Exception as e:
        print(f"V1 Basic FAILED: {e}")
        results.append(("V1 Basic", False))

    # Test V2
    try:
        ok = test_v2_basic()
        results.append(("V2 Basic", ok))
    except Exception as e:
        print(f"V2 Basic FAILED: {e}")
        results.append(("V2 Basic", False))

    # Compare with PLS
    try:
        ok = test_compare_pls()
        results.append(("Compare PLS", ok))
    except Exception as e:
        print(f"Compare PLS FAILED: {e}")
        results.append(("Compare PLS", False))

    # Test init modes
    try:
        ok = test_different_init_modes()
        results.append(("Init Modes", ok))
    except Exception as e:
        print(f"Init Modes FAILED: {e}")
        results.append(("Init Modes", False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
