#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced modular pipeline runner.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from spectraset import SpectraDataset
from runner import PipelineRunner


def create_sample_data():
    """Create sample spectral data for testing."""
    np.random.seed(42)
    
    # Create synthetic spectral data
    n_samples = 100
    n_wavelengths = 50
    
    # Generate base spectra with some structure
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    spectra = []
    targets = []
    
    for i in range(n_samples):
        # Create spectra with different profiles
        if i % 3 == 0:  # Type A
            spectrum = np.exp(-((wavelengths - 1000) / 200) ** 2) + 0.1 * np.random.randn(n_wavelengths)
            target = 10 + 0.5 * i + np.random.randn()
        elif i % 3 == 1:  # Type B
            spectrum = np.exp(-((wavelengths - 1500) / 300) ** 2) + 0.1 * np.random.randn(n_wavelengths)
            target = 15 + 0.3 * i + np.random.randn()
        else:  # Type C
            spectrum = np.exp(-((wavelengths - 800) / 150) ** 2) + 0.1 * np.random.randn(n_wavelengths)
            target = 8 + 0.4 * i + np.random.randn()
        
        spectra.append(spectrum.tolist())
        targets.append(target)
    
    return spectra, targets


def test_modular_pipeline():
    """Test the modular pipeline runner with various step types."""
    print("=== Testing Enhanced Modular Pipeline Runner ===")
    
    # Create sample dataset
    spectra, targets = create_sample_data()
    
    # Initialize dataset
    dataset = SpectraDataset()
    
    # Add data to dataset
    for i, (spectrum, target) in enumerate(zip(spectra, targets)):
        dataset.add_spectra(
            [[spectrum]],  # Wrap in list for augmentation structure
            target=target,
            origin=f"sample_{i}",
            sample=i,
            type="nirs",
            set="train" if i < 80 else "test"
        )
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Features shape: {dataset.features.shape if dataset.features is not None else 'None'}")
    print(f"Labels shape: {dataset.labels.shape if dataset.labels is not None else 'None'}")
    
    # Initialize pipeline runner
    runner = PipelineRunner()
    
    # Define a complex pipeline with various step types
    pipeline = [
        # Data preprocessing
        StandardScaler(),
        
        # Feature engineering
        {"feature_augmentation": [PCA(n_components=10)]},
        
        # Clustering
        {"cluster": KMeans(n_clusters=3, random_state=42)},
        
        # K-fold cross-validation with model training
        {
            "kfold": {
                "n_splits": 3,
                "n_repeats": 1,
                "random_state": 42,
                "pipeline": [
                    {
                        "model": RandomForestRegressor(n_estimators=10, random_state=42),
                        "y_pipeline": [],
                        "train_params": {"verbose": True}
                    }
                ]
            }
        },
        
        # Performance evaluation
        {"evaluate": {}}
    ]
    
    print("\n=== Running Enhanced Pipeline ===")
    runner.run_pipeline(pipeline, dataset)
    
    print("\n=== Pipeline Execution Complete ===")
    print(f"Final dataset size: {len(dataset)}")
    if dataset.results is not None:
        print(f"Predictions generated: {dataset.results.height}")
        print("Prediction columns:", dataset.results.columns)
    else:
        print("No predictions generated")


if __name__ == "__main__":
    test_modular_pipeline()
