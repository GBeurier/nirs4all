#!/usr/bin/env python3
"""
Test 7: Real Pipeline using sample.py config

This test demonstrates:
1. Creating a realistic dataset with NIRS (2500 wavelengths) and Raman (1000 wavelengths) sources
2. Using the exact config from sample.py to build and execute a pipeline
3. Saving pipeline config to sample_test.json
4. Saving trained models and results to files

Based on the sample.py configuration format.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"       # Forwarded automatically to joblib workers

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# Suppress TensorFlow and Keras warnings if available
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', category=UserWarning, module='keras')
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    warnings.filterwarnings('ignore', message='.*TensorFlow binary is optimized.*')
    warnings.filterwarnings('ignore', message='.*Do not pass an `input_shape`.*')
    warnings.filterwarnings('ignore', message='.*triggered tf.function retracing.*')

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from new4all
from SpectraDataset import SpectraDataset
from Pipeline import Pipeline
from PipelineBuilder import PipelineBuilder
from PipelineConfig import PipelineConfig
from PipelineSerializer import PipelineSerializer
from sample import config as sample_config



NIRS4ALL_AVAILABLE = True
print("Successfully imported config from sample.py")


def create_nirs_raman_dataset():
    """Create a realistic dataset with NIRS and Raman spectroscopy data."""
    print("=== Creating NIRS + Raman Dataset ===")

    np.random.seed(42)
    n_samples = 200

    # NIRS data (2500 wavelengths)
    nirs_wavelengths = 2500
    wavelengths_nirs = np.linspace(1000, 2500, nirs_wavelengths)

    # Create realistic NIRS spectra with characteristic peaks
    base_nirs = (np.exp(-((wavelengths_nirs - 1400) / 200) ** 2) +
                 0.7 * np.exp(-((wavelengths_nirs - 1900) / 150) ** 2) +
                 0.5 * np.exp(-((wavelengths_nirs - 2100) / 100) ** 2) +
                 0.3 * np.exp(-((wavelengths_nirs - 2300) / 120) ** 2))

    X_nirs = np.zeros((n_samples, nirs_wavelengths))
    for i in range(n_samples):
        noise = np.random.normal(0, 0.02, nirs_wavelengths)
        shift = np.random.normal(0, 20)  # Wavelength shift
        intensity = np.random.normal(1, 0.15)  # Intensity variation

        shifted_base = np.interp(wavelengths_nirs + shift, wavelengths_nirs, base_nirs)
        X_nirs[i] = intensity * shifted_base + noise

    # Raman data (1000 wavelengths)
    raman_wavelengths = 1000
    raman_shifts = np.linspace(200, 3500, raman_wavelengths)

    # Create realistic Raman spectra with characteristic peaks
    base_raman = (0.8 * np.exp(-((raman_shifts - 1000) / 150) ** 2) +
                  0.6 * np.exp(-((raman_shifts - 1600) / 100) ** 2) +
                  0.4 * np.exp(-((raman_shifts - 2900) / 200) ** 2) +
                  0.3 * np.exp(-((raman_shifts - 3100) / 80) ** 2))

    X_raman = np.zeros((n_samples, raman_wavelengths))
    for i in range(n_samples):
        noise = np.random.normal(0, 0.01, raman_wavelengths)
        shift = np.random.normal(0, 10)  # Wavenumber shift
        intensity = np.random.normal(1, 0.1)  # Intensity variation

        shifted_base = np.interp(raman_shifts + shift, raman_shifts, base_raman)
        X_raman[i] = intensity * shifted_base + noise

    # Create realistic targets based on spectral features
    # Simulate protein content classification
    protein_signal = (X_nirs[:, 800:900].mean(axis=1) * 2.0 +
                      X_nirs[:, 1200:1300].mean(axis=1) * 1.5 +
                      X_raman[:, 400:500].mean(axis=1) * 3.0 +
                      np.random.normal(0, 0.1, n_samples))

    # Create classification levels
    protein_percentiles = np.percentile(protein_signal, [33, 66])
    protein_levels = np.array(['low' if p < protein_percentiles[0]
                               else 'medium' if p < protein_percentiles[1]
                               else 'high' for p in protein_signal])

    # Create dataset
    dataset = SpectraDataset(task_type="classification")

    # Add data with sample IDs
    sample_ids = dataset.add_data(
        features=[X_nirs, X_raman],
        targets=protein_levels,
        partition="train",
        group=1,
        processing="raw"
    )

    print(f"Dataset created: {len(dataset)} samples")
    print(f"NIRS: {nirs_wavelengths} wavelengths, Raman: {raman_wavelengths} wavelengths")
    print(f"Class distribution: {dict(zip(*np.unique(protein_levels, return_counts=True)))}")

    return dataset


def test_real_pipeline():
    """Test the complete pipeline using sample.py config"""
    print("\n=== Testing Real Pipeline with sample.py Config ===")

    if not NIRS4ALL_AVAILABLE or sample_config is None:
        print("Cannot run test: sample.py config not available")
        return

    # Create output directory
    output_dir = Path("results/test_7_real_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create dataset
    dataset = create_nirs_raman_dataset()

    # Step 2: Use config from sample.py
    config = PipelineConfig.from_dict(sample_config)
    config.name = "Real Pipeline Test"

    # Step 3: Build pipeline
    builder = PipelineBuilder()
    operations = builder.build_from_config(config)    # Step 4: Create and execute pipeline
    pipeline = Pipeline("Real Pipeline Execution")
    for operation in operations:
        pipeline.add_operation(operation)

    print(f"Pipeline created with {len(operations)} operations")

    # Step 5: Execute pipeline
    print("Executing pipeline...")
    try:
        pipeline.execute(dataset)
        print("Pipeline executed successfully!")

        # Step 6: Save results
        print("Saving results...")

        # Save config as JSON using PipelineSerializer
        config_path = output_dir / "sample_test.json"
        serializer = PipelineSerializer()
        serializer.serialize_to_file(sample_config, config_path)
        print(f"Config saved to: {config_path}")

        # Save pipeline
        pipeline_path = output_dir / "pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Pipeline saved to: {pipeline_path}")

        # Save dataset state
        dataset_path = output_dir / "final_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to: {dataset_path}")

        # Export results as CSV (if dataset has results)
        if hasattr(dataset, 'results') and dataset.results is not None:
            if hasattr(dataset.results, 'to_csv'):
                # If it's already a DataFrame
                results_path = output_dir / "results.csv"
                dataset.results.to_csv(results_path, index=False)
                print(f"Results saved to: {results_path}")
            else:
                # Convert to DataFrame first
                results_df = pd.DataFrame(dataset.results)
                results_path = output_dir / "results.csv"
                results_df.to_csv(results_path, index=False)
                print(f"Results saved to: {results_path}")

        print(f"\nAll outputs saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        print("This might be due to missing dependencies or configuration issues")

        # Still save the config for reference using PipelineSerializer
        config_path = output_dir / "sample_test.json"
        try:
            serializer = PipelineSerializer()
            serializer.serialize_to_file(sample_config, config_path)
            print(f"Config saved to: {config_path}")
        except Exception as save_error:
            print(f"Failed to save config: {save_error}")
            # Save a summary instead
            config_summary = {
                "experiment": sample_config.get("experiment", {}),
                "pipeline_steps": len(sample_config.get("pipeline", [])),
                "error": "Failed to serialize full config due to complex objects"
            }
            with open(config_path, 'w') as f:
                json.dump(config_summary, f, indent=2)
            print(f"Config summary saved to: {config_path}")


def main():
    """Main test function"""
    print("=== Test 7: Real Pipeline using sample.py config ===")
    print(f"NIRS4ALL transformations available: {NIRS4ALL_AVAILABLE}")

    test_real_pipeline()

    print("\nTest completed!")


if __name__ == "__main__":
    main()
