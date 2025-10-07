"""
Example demonstrating wavelength resampling with the Resampler operator.

This example shows how to:
1. Resample spectral data to a new wavelength grid
2. Use different interpolation methods
3. Handle multi-source datasets with per-source or shared targets
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

from nirs4all.operators.transformations import Resampler, StandardNormalVariate
from nirs4all.dataset import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner


def example_basic_resampling():
    """Basic example: resample to fewer wavelengths."""
    print("\n" + "="*70)
    print("Example 1: Basic Resampling")
    print("="*70)

    # Define target wavelengths (e.g., downsample from 200 to 100 points)
    # Assuming original data is in range 1000-2500 cm-1
    target_wavelengths = np.linspace(1000, 2500, 100)

    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        Resampler(target_wavelengths=target_wavelengths, method='linear'),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"y_processing": MinMaxScaler()},
        {"model": PLSRegression(n_components=10)},
    ]

    # Run pipeline (replace with your actual data path)
    pipeline_config = PipelineConfigs(pipeline, name="BasicResampling")
    dataset_config = DatasetConfigs("sample_data")  # Update with your path

    runner = PipelineRunner(save_files=False, verbose=1)
    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print(f"\n✓ Resampling successful!")
        print(f"  Original: ~200 wavelengths → Resampled: {len(target_wavelengths)} wavelengths")

        # Show top models
        top = predictions.top_k(3, 'rmse')
        print(f"\nTop 3 models:")
        for i, model in enumerate(top, 1):
            print(f"  {i}. {model['model_name']}: RMSE = {model['rmse']:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_cubic_interpolation():
    """Example with cubic spline interpolation."""
    print("\n" + "="*70)
    print("Example 2: Cubic Spline Interpolation")
    print("="*70)

    # Higher resolution target grid with cubic interpolation
    target_wavelengths = np.linspace(1000, 2500, 300)

    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        Resampler(
            target_wavelengths=target_wavelengths,
            method='cubic',  # Smooth interpolation
            fill_value=0.0
        ),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"y_processing": MinMaxScaler()},
        {"model": PLSRegression(n_components=15)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="CubicResampling")
    dataset_config = DatasetConfigs("sample_data")

    runner = PipelineRunner(save_files=False, verbose=1)
    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print(f"\n✓ Cubic resampling successful!")
        print(f"  Upsampled to: {len(target_wavelengths)} wavelengths")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_crop_and_resample():
    """Example with cropping and resampling."""
    print("\n" + "="*70)
    print("Example 3: Crop and Resample")
    print("="*70)

    # Focus on specific wavelength range
    target_wavelengths = np.linspace(1200, 2200, 150)

    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        Resampler(
            target_wavelengths=target_wavelengths,
            method='linear',
            crop_range=(1100, 2300),  # Crop before resampling
        ),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"y_processing": MinMaxScaler()},
        {"model": PLSRegression(n_components=10)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="CropResample")
    dataset_config = DatasetConfigs("sample_data")

    runner = PipelineRunner(save_files=False, verbose=1)
    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print(f"\n✓ Crop and resample successful!")
        print(f"  Cropped to 1100-2300 cm-1, then resampled to {len(target_wavelengths)} points")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_multi_source_resampling():
    """Example with per-source target wavelengths (multi-source datasets)."""
    print("\n" + "="*70)
    print("Example 4: Multi-Source Resampling")
    print("="*70)

    # Different target wavelengths for each source
    # If your dataset has 2 sources:
    target_wavelengths_source1 = np.linspace(1000, 2500, 100)
    target_wavelengths_source2 = np.linspace(1100, 2400, 120)

    # Pass as list for per-source targets
    target_wavelengths_list = [target_wavelengths_source1, target_wavelengths_source2]

    pipeline = [
        MinMaxScaler(),
        StandardNormalVariate(),
        Resampler(
            target_wavelengths=target_wavelengths_list,  # List for per-source
            method='linear'
        ),
        ShuffleSplit(n_splits=2, test_size=0.25),
        {"y_processing": MinMaxScaler()},
        {"model": PLSRegression(n_components=10)},
    ]

    pipeline_config = PipelineConfigs(pipeline, name="MultiSourceResampling")
    dataset_config = DatasetConfigs("sample_data")  # Must have multiple sources

    runner = PipelineRunner(save_files=False, verbose=1)
    try:
        predictions, _ = runner.run(pipeline_config, dataset_config)
        print(f"\n✓ Multi-source resampling successful!")
        print(f"  Source 1: {len(target_wavelengths_source1)} wavelengths")
        print(f"  Source 2: {len(target_wavelengths_source2)} wavelengths")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_comparison():
    """Compare different resampling strategies."""
    print("\n" + "="*70)
    print("Example 5: Comparison of Resampling Methods")
    print("="*70)

    target_wavelengths = np.linspace(1000, 2500, 100)

    methods = ['linear', 'cubic', 'nearest']

    for method in methods:
        print(f"\n--- Testing method: {method} ---")

        pipeline = [
            MinMaxScaler(),
            StandardNormalVariate(),
            Resampler(target_wavelengths=target_wavelengths, method=method),
            ShuffleSplit(n_splits=2, test_size=0.25),
            {"y_processing": MinMaxScaler()},
            {"model": PLSRegression(n_components=10)},
        ]

        pipeline_config = PipelineConfigs(pipeline, name=f"Resample_{method}")
        dataset_config = DatasetConfigs("sample_data")

        runner = PipelineRunner(save_files=False, verbose=0)
        try:
            predictions, _ = runner.run(pipeline_config, dataset_config)
            best = predictions.top_k(1, 'rmse')[0]
            print(f"  ✓ {method}: RMSE = {best['rmse']:.4f}")
        except Exception as e:
            print(f"  ✗ {method}: Error - {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NIRS4ALL Resampler Examples")
    print("="*70)
    print("\nThese examples demonstrate wavelength resampling capabilities.")
    print("Make sure to update 'sample_data' path to your actual dataset.")

    # Run examples (uncomment the ones you want to try)

    # Example 1: Basic linear resampling
    # example_basic_resampling()

    # Example 2: Cubic spline interpolation
    # example_cubic_interpolation()

    # Example 3: Crop and resample specific range
    # example_crop_and_resample()

    # Example 4: Multi-source with different target wavelengths
    # example_multi_source_resampling()

    # Example 5: Compare different methods
    # example_comparison()

    print("\n" + "="*70)
    print("Uncomment the examples above to run them!")
    print("="*70)
