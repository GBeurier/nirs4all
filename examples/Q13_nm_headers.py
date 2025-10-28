"""
Q13 Example - Working with Nanometer (nm) Headers
================================================
Demonstrates the new header_unit functionality for loading and processing
NIRS data with nanometer wavelength headers instead of wavenumber (cm⁻¹).

This example shows how nirs4all automatically:
- Handles nm wavelength data
- Converts nm to cm⁻¹ internally for resampling
- Displays proper axis labels in visualizations
- Maintains backward compatibility with cm⁻¹ data

Key Features:
- Load CSV files with nm headers using header_unit='nm'
- Automatic unit conversion for wavelength-based operations
- Proper visualization axis labels based on header units
- Seamless integration with preprocessing and modeling
"""

# Standard library imports
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import (
    StandardNormalVariate, SavitzkyGolay, Resampler
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner


def create_sample_nm_data(output_dir):
    """
    Create sample NIRS data with nanometer (nm) wavelength headers.

    Generates synthetic spectral data in the NIR range (780-2500 nm)
    which is the typical range for NIR spectroscopy.

    Args:
        output_dir: Directory to save the CSV files
    """
    # Create wavelength range in nanometers (NIR range: 780-2500 nm)
    # Using 50 wavelength points for this example
    wavelengths_nm = np.linspace(780, 2500, 50)

    # Generate 40 synthetic spectra with realistic NIR characteristics
    n_samples = 40
    spectra = []

    for i in range(n_samples):
        # Create baseline
        baseline = 0.5 + 0.3 * np.random.rand()

        # Add broad absorption bands (typical of NIR spectra)
        spectrum = baseline * np.ones(len(wavelengths_nm))

        # Add characteristic NIR absorption features
        # Water bands around 1450 nm and 1940 nm
        spectrum += 0.2 * np.exp(-((wavelengths_nm - 1450)**2) / 50000)
        spectrum += 0.15 * np.exp(-((wavelengths_nm - 1940)**2) / 60000)

        # Add C-H overtones around 1200 nm and 1700 nm
        spectrum += 0.1 * np.exp(-((wavelengths_nm - 1200)**2) / 40000)
        spectrum += 0.08 * np.exp(-((wavelengths_nm - 1700)**2) / 45000)

        # Add some noise
        spectrum += 0.02 * np.random.randn(len(wavelengths_nm))

        spectra.append(spectrum)

    X = np.array(spectra)

    # Generate target values (e.g., protein content, moisture, etc.)
    # Correlated with spectral features
    y = 10 + 5 * X[:, 15] + 3 * X[:, 30] + 2 * np.random.randn(n_samples)

    # Split into train/test
    train_size = 30
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create DataFrames with nm headers
    headers = [f"{wl:.1f}" for wl in wavelengths_nm]

    df_X_train = pd.DataFrame(X_train, columns=headers)
    df_X_test = pd.DataFrame(X_test, columns=headers)
    df_y_train = pd.DataFrame(y_train, columns=['target'])
    df_y_test = pd.DataFrame(y_test, columns=['target'])

    # Save to CSV files
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    df_X_train.to_csv(output_path / 'Xtrain.csv', index=False, sep=';')
    df_X_test.to_csv(output_path / 'Xtest.csv', index=False, sep=';')
    df_y_train.to_csv(output_path / 'Ytrain.csv', index=False, sep=';')
    df_y_test.to_csv(output_path / 'Ytest.csv', index=False, sep=';')

    print(f"Created sample nm data in {output_dir}")
    print(f"  Wavelength range: {wavelengths_nm[0]:.1f} - {wavelengths_nm[-1]:.1f} nm")
    print(f"  Number of wavelengths: {len(wavelengths_nm)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()


def example1_basic_nm_pipeline():
    """Example 1: Basic pipeline with nm wavelength data."""
    print("=" * 70)
    print("Example 1: Basic Pipeline with Nanometer Headers")
    print("=" * 70)
    print()

    # Create temporary directory for sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "nm_data"
        create_sample_nm_data(data_path)

        # Define pipeline
        pipeline = [
            "chart_2d",  # Visualize raw spectra (will show "Wavelength (nm)")
            MinMaxScaler(),
            StandardNormalVariate(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=10), "name": "PLS-10"}
        ]

        # Create configurations with header_unit='nm'
        pipeline_config = PipelineConfigs(pipeline, "NM_Pipeline")
        dataset_config = DatasetConfigs({
            'folder': str(data_path),
            'params': {'header_unit': 'nm'}  # Specify nm headers via global params
        })

        # Run pipeline
        runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Display results
        if len(predictions) > 0:
            best = predictions.top(1, 'rmse')[0]
            print(f"\nBest model: {best.model_name}")
            # Use simple string formatting like in other examples
            print(f"Model details: {Predictions.pred_short_string(best, metrics=['rmse', 'r2'])}")

    print()


def example2_nm_resampling():
    """Example 2: Resampling nm data (automatic conversion to cm⁻¹)."""
    print("=" * 70)
    print("Example 2: Resampling Nanometer Data")
    print("=" * 70)
    print("Demonstrates automatic nm → cm⁻¹ conversion for resampling")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "nm_data"
        create_sample_nm_data(data_path)

        # Define target wavelengths for resampling (in cm⁻¹)
        # Original nm range: 780-2500 nm
        # Equivalent cm⁻¹ range: ~12820-4000 cm⁻¹
        # Let's resample to a coarser grid
        target_wavelengths_cm1 = np.linspace(12000, 4500, 30)

        pipeline = [
            "chart_2d",  # Before resampling (nm labels)
            Resampler(
                target_wavelengths=target_wavelengths_cm1,
                method='linear'
            ),  # Resampler converts nm → cm⁻¹ automatically
            "chart_2d",  # After resampling (cm⁻¹ labels)
            MinMaxScaler(),
            StandardNormalVariate(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=8), "name": "PLS-8-Resampled"}
        ]

        pipeline_config = PipelineConfigs(pipeline, "Resampling_Pipeline")
        dataset_config = DatasetConfigs({
            'folder': str(data_path),
            'params': {'header_unit': 'nm'}
        })

        runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        if len(predictions) > 0:
            best = predictions.top(1, 'rmse')[0]
            print(f"\nAfter resampling from 50 to 30 wavelengths:")
            print(f"Model: {Predictions.pred_short_string(best, metrics=['rmse', 'r2'])}")
            print("\nNote: Resampled data now has cm⁻¹ headers")

    print()


def example3_mixed_cm1_and_nm():
    """Example 3: Compare cm⁻¹ and nm data in the same analysis."""
    print("=" * 70)
    print("Example 3: Working with Both cm⁻¹ and nm Data")
    print("=" * 70)
    print("Shows backward compatibility with existing cm⁻¹ data")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nm dataset
        nm_path = Path(tmpdir) / "nm_data"
        create_sample_nm_data(nm_path)

        # Use existing cm⁻¹ dataset
        cm1_path = "sample_data/regression_2"

        print("Analyzing nm dataset:")
        print("-" * 70)

        # Pipeline for nm data
        pipeline = [
            MinMaxScaler(),
            StandardNormalVariate(),
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, test_size=0.3),
            {"model": PLSRegression(n_components=10), "name": "PLS-10"}
        ]

        pipeline_config = PipelineConfigs(pipeline, "NM_Analysis")
        nm_config = DatasetConfigs({
            'folder': str(nm_path),
            'params': {'header_unit': 'nm'}
        })

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        nm_predictions, _ = runner.run(pipeline_config, nm_config)

        if len(nm_predictions) > 0:
            nm_best = nm_predictions.top(1, 'rmse')[0]
            print(f"nm data: {Predictions.pred_short_string(nm_best, metrics=['rmse'])}")

        print()
        print("Analyzing cm⁻¹ dataset (default behavior):")
        print("-" * 70)

        # Pipeline for cm⁻¹ data (no header_unit needed - defaults to cm-1)
        cm1_config = DatasetConfigs(cm1_path)
        cm1_predictions, _ = runner.run(pipeline_config, cm1_config)

        if len(cm1_predictions) > 0:
            cm1_best = cm1_predictions.top(1, 'rmse')[0]
            print(f"cm⁻¹ data: {Predictions.pred_short_string(cm1_best, metrics=['rmse'])}")

        print()
        print("Both header types work seamlessly! 🎉")

    print()


def example4_preprocessing_combinations():
    """Example 4: Feature augmentation with nm data."""
    print("=" * 70)
    print("Example 4: Preprocessing Combinations with nm Data")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "nm_data"
        create_sample_nm_data(data_path)

        # Pipeline with feature augmentation
        pipeline = [
            MinMaxScaler(),
            {
                "feature_augmentation": {
                    "_or_": [StandardNormalVariate, SavitzkyGolay],
                    "size": [1, 2],
                    "count": 3
                }
            },
            {"y_processing": MinMaxScaler()},
            ShuffleSplit(n_splits=2, test_size=0.3),
        ]

        # Add multiple PLS models
        for n_comp in [5, 10, 15]:
            pipeline.append({
                "model": PLSRegression(n_components=n_comp),
                "name": f"PLS-{n_comp}"
            })

        pipeline_config = PipelineConfigs(pipeline, "Augmentation_Pipeline")
        dataset_config = DatasetConfigs({
            'folder': str(data_path),
            'params': {'header_unit': 'nm'}
        })

        runner = PipelineRunner(save_files=False, verbose=0, plots_visible=False)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Display top 5 models
        print("Top 5 models by test RMSE:")
        top_models = predictions.top(5, 'rmse')
        for idx, pred in enumerate(top_models, 1):
            print(f"{idx}. {Predictions.pred_short_string(pred, metrics=['rmse'])} - Preprocessing: {pred['preprocessings']}")

    print()


def main():
    """Run all examples demonstrating nm header functionality."""
    print("\n" + "=" * 70)
    print("NIRS4ALL - Nanometer (nm) Header Support Examples")
    print("=" * 70)
    print()
    print("These examples demonstrate the new header_unit functionality")
    print("for working with wavelength data in nanometers (nm) instead of")
    print("the traditional wavenumber format (cm⁻¹).")
    print()
    print("Key points:")
    print("  • Use {'folder': path, 'params': {'header_unit': 'nm'}} to load nm data")
    print("  • Resampler automatically converts nm → cm⁻¹ for processing")
    print("  • Visualizations show appropriate axis labels")
    print("  • Full backward compatibility with cm⁻¹ data (default)")
    print("=" * 70)
    print()

    # Run all examples
    example1_basic_nm_pipeline()
    example2_nm_resampling()
    example3_mixed_cm1_and_nm()
    example4_preprocessing_combinations()

    print("=" * 70)
    print("All examples completed successfully! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
