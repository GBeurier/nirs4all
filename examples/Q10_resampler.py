"""
Example demonstrating the Resampler operator for wavelength resampling.

This example shows how to use the Resampler to interpolate spectral data
to a new wavelength grid, which is useful for:
- Combining data from different instruments with different wavelength grids
- Standardizing wavelength resolution across datasets
- Focusing on specific spectral regions

The resampler uses scipy interpolation to estimate spectral values at new
wavelengths based on the original wavelength-intensity relationship.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

from nirs4all.operators.transformations import Resampler, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.dataset import DatasetConfigs

def main():
    """Run resampler example."""

    print("=" * 70)
    print("NIRS4ALL - Resampler Example")
    print("=" * 70)
    print()

    # Example 1: Resample using wavelengths from another dataset (classification)
    print("Example 1: Resample using wavelengths from classification dataset")
    print("-" * 70)

    # # Get wavelengths from classification dataset as target
    classification_config = DatasetConfigs("sample_data/regression_2")
    ref_dataset = classification_config.iter_datasets().__next__()
    target_wl_from_other = ref_dataset.float_headers(0)
    # print(f"Target wavelengths from classification dataset: {len(target_wl_from_other)} points")
    # print(f"Range: {target_wl_from_other[0]:.1f} to {target_wl_from_other[-1]:.1f} cm-1")

    pipeline_other = [
        "chart_2d",
        Resampler(target_wavelengths=target_wl_from_other, method='linear'),
        "chart_2d",
    ]

    # Apply to regression dataset
    dataset_config = DatasetConfigs("sample_data/regression_3")
    pipeline_config = PipelineConfigs(pipeline_other, name="Other_Dataset_Pipeline")

    runner = PipelineRunner(save_files=True, verbose=1, plots_visible=True)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Example 2: Downsample to fewer points (descending order)
    print("\nExample 2: Downsample from 125 to 10 wavelengths (descending)")
    print("-" * 70)

    # Dataset wavelengths: 11012 down to 5966 cm⁻¹ (125 points, descending)
    # Create 10 evenly spaced points in descending order
    target_wl_downsample = np.linspace(11012, 5966, 10)  # Descending like original

    pipeline_downsample = [
        "chart_2d",
        Resampler(target_wavelengths=target_wl_downsample, method='linear'),
        "chart_2d",
    ]

    pipeline_config = PipelineConfigs(pipeline_downsample, name="Downsample_Pipeline")
    runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Example 3: Focus on fingerprint region (descending order)
    print("\nExample 3: Resample to fingerprint region (9500-7000 cm⁻¹)")
    print("-" * 70)

    # Focus on mid-infrared fingerprint region with higher resolution (descending)
    target_wl_cropped = np.linspace(9500, 7000, 50)  # Descending, 50 points

    pipeline_cropped = [
        'chart_2d',
        Resampler(
            target_wavelengths=target_wl_cropped,
            method='linear'  # Linear interpolation
        ),
        'chart_2d',
    ]

    pipeline_config = PipelineConfigs(pipeline_cropped, name="Cropped_Pipeline")

    runner = PipelineRunner(plots_visible=True)

    predictions, _ = runner.run(pipeline_config, dataset_config)


if __name__ == "__main__":
    main()
