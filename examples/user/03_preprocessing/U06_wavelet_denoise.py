#!/usr/bin/env python3
"""
Wavelet Denoising for NIRS Preprocessing
=========================================

This example demonstrates how to use wavelet denoising to reduce noise in NIRS data
while preserving spectral features. Wavelet denoising is particularly effective for
NIRS data because it can separate signal from noise in the frequency domain and
reconstruct a clean signal with the original length intact.

The WaveletDenoise operator uses multi-level discrete wavelet decomposition with
thresholding on detail coefficients, ideal for PLS regression and other ML methods.

Key Parameters
--------------
- wavelet: Wavelet family ('db4', 'db8', 'sym8', 'coif3', 'haar')
- level: Decomposition level (higher = captures lower frequencies)
- threshold_mode: 'soft' (smoother) or 'hard' (preserves peaks)
- noise_estimator: 'median' (robust) or 'std' (Gaussian assumption)

When to Use
-----------
- High-frequency instrumental noise
- Baseline drift combined with noise
- Before PLS or other regression methods
- When you want to preserve spectral shape while reducing noise

"""

import numpy as np
import matplotlib.pyplot as plt

import nirs4all
from nirs4all.operators.transforms import WaveletDenoise
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler


def create_noisy_signal():
    """Create a synthetic NIRS-like signal with noise for demonstration."""
    n_samples = 100
    n_features = 1200
    wavelengths = np.linspace(1000, 2500, n_features)

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Baseline + absorption bands + random variations
        baseline = 0.3 + 0.05 * np.random.randn()
        band1 = 0.05 * np.exp(-((wavelengths - 1450) ** 2) / 10000)
        band2 = 0.02 * np.exp(-((wavelengths - 1930) ** 2) / 8000)
        band3 = 0.01 * np.exp(-((wavelengths - 2100) ** 2) / 12000)

        # Clean signal
        clean = baseline + band1 + band2 + band3

        # Add instrumental noise
        noise = 0.01 * np.random.randn(n_features)

        X[i] = clean + noise
        y[i] = baseline * 10 + 0.1 * np.random.randn()  # Target correlated with baseline

    return X, y, wavelengths


def example_basic_denoising():
    """Example 1: Basic wavelet denoising."""
    print("=" * 70)
    print("Example 1: Basic Wavelet Denoising")
    print("=" * 70)

    X, y, wavelengths = create_noisy_signal()

    from sklearn.model_selection import ShuffleSplit

    # Simple pipeline with wavelet denoising
    pipeline = [
        WaveletDenoise(wavelet="db8", level=5, threshold_mode="soft"),
        MinMaxScaler(),
        ShuffleSplit(n_splits=5, test_size=0.25),
        {"model": PLSRegression(n_components=5)},
    ]

    result = nirs4all.run(pipeline=pipeline, dataset=(X, y, {"train": 75}), verbose=1)

    print(f"\nBest RMSE: {result.best_rmse:.4f}")
    print(f"Best R²: {result.best_r2:.4f}")
    print("\n")


def example_compare_methods():
    """Example 2: Compare different denoising methods."""
    print("=" * 70)
    print("Example 2: Compare Wavelet Families and Thresholding")
    print("=" * 70)

    X, y, wavelengths = create_noisy_signal()

    from sklearn.model_selection import ShuffleSplit

    results = {}

    # Test different wavelet families and thresholding modes
    configs = [
        ("db4", "soft"),
        ("db8", "soft"),
        ("sym8", "soft"),
        ("coif3", "soft"),
        ("db8", "hard"),
    ]

    for wavelet, threshold_mode in configs:
        pipeline = [
            WaveletDenoise(wavelet=wavelet, level=5, threshold_mode=threshold_mode),
            MinMaxScaler(),
            ShuffleSplit(n_splits=5, test_size=0.25),
            {"model": PLSRegression(n_components=5)},
        ]

        result = nirs4all.run(pipeline=pipeline, dataset=(X, y, {"train": 75}), verbose=0)
        results[f"{wavelet}_{threshold_mode}"] = result.best_rmse

    # Display comparison
    print("\nComparison of wavelet denoising configurations:")
    print("-" * 50)
    for config, rmse in sorted(results.items(), key=lambda x: x[1]):
        print(f"{config:20s}: RMSE = {rmse:.4f}")
    print("\n")


def example_with_cartesian():
    """Example 3: Use with _cartesian_ for automatic method selection."""
    print("=" * 70)
    print("Example 3: Automatic Wavelet Method Selection with _cartesian_")
    print("=" * 70)

    X, y, wavelengths = create_noisy_signal()

    from sklearn.model_selection import ShuffleSplit
    from nirs4all.operators.transforms import IdentityTransformer

    # Test multiple preprocessing combinations automatically
    pipeline = [
        {
            "_cartesian_": [
                # No preprocessing vs wavelet denoising
                [{"_or_": [IdentityTransformer(), WaveletDenoise(wavelet="db8", level=5)]}],
                # Scaling options
                [{"_or_": [MinMaxScaler()]}],
            ]
        },
        ShuffleSplit(n_splits=5, test_size=0.25),
        {"model": PLSRegression(n_components=5)},
    ]

    result = nirs4all.run(pipeline=pipeline, dataset=(X, y, {"train": 75}), verbose=1)

    print(f"\nBest RMSE: {result.best_rmse:.4f}")
    print(f"Best R²: {result.best_r2:.4f}")
    print("\nThe pipeline automatically selected the best preprocessing combination.")
    print("\n")


def example_visualization():
    """Example 4: Visualize denoising effect."""
    print("=" * 70)
    print("Example 4: Visualize Denoising Effect")
    print("=" * 70)

    X, y, wavelengths = create_noisy_signal()

    # Apply denoising
    denoiser = WaveletDenoise(wavelet="db8", level=5, threshold_mode="soft")
    X_denoised = denoiser.fit_transform(X)

    # Plot first 3 samples
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    for i in range(3):
        ax = axes[i]
        ax.plot(wavelengths, X[i], "b-", alpha=0.5, linewidth=1, label="Noisy")
        ax.plot(wavelengths, X_denoised[i], "r-", linewidth=2, label="Denoised")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"Sample {i + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("wavelet_denoise_comparison.png", dpi=150)
    print("Visualization saved to: wavelet_denoise_comparison.png")
    print("\n")


def example_noise_variance_comparison():
    """Example 5: Quantify noise reduction."""
    print("=" * 70)
    print("Example 5: Quantify Noise Reduction")
    print("=" * 70)

    X, y, wavelengths = create_noisy_signal()

    # Compare different levels
    levels = [3, 4, 5, 6, 7]
    noise_reduction = {}

    for level in levels:
        denoiser = WaveletDenoise(wavelet="db8", level=level, threshold_mode="soft")
        X_denoised = denoiser.fit_transform(X)

        # Measure high-frequency noise (second derivative std)
        noise_before = np.std(np.diff(X, n=2, axis=1))
        noise_after = np.std(np.diff(X_denoised, n=2, axis=1))
        reduction_pct = 100 * (1 - noise_after / noise_before)

        noise_reduction[level] = reduction_pct

    print("\nNoise reduction by decomposition level:")
    print("-" * 50)
    for level, reduction in noise_reduction.items():
        print(f"Level {level}: {reduction:5.1f}% noise reduction")

    print("\nNote: Higher levels remove more noise but may over-smooth the signal.")
    print("For NIRS data (~1000 features), level=5 is typically a good choice.")
    print("\n")


if __name__ == "__main__":
    # Run all examples
    example_basic_denoising()
    example_compare_methods()
    example_with_cartesian()
    example_noise_variance_comparison()

    # Run visualization (optional - requires matplotlib)
    try:
        example_visualization()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- WaveletDenoise preserves signal length (perfect for PLS)")
    print("- db8 with soft thresholding is generally a good default")
    print("- Level 4-6 works well for typical NIRS data (~1000 features)")
    print("- Use _cartesian_ to automatically test with/without denoising")
    print("- Stateless transformer: no parameters learned from training data")
