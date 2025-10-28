"""
Synthetic test data generator for nirs4all integration tests.

This module creates consistent synthetic NIRS-like datasets with distinctive properties
for testing various components of the nirs4all pipeline.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Tuple, Dict, Optional


class SyntheticNIRSDataGenerator:
    """
    Generate synthetic NIRS-like spectral data for testing.

    Features:
    - Controlled noise levels for predictable test behavior
    - Distinctive spectral patterns per class/target
    - Multiple targets for multi-task testing
    - Consistent random seeds for reproducible tests
    """

    def __init__(self, random_state: int = 42):
        """Initialize the data generator with fixed random state."""
        self.random_state = random_state
        np.random.seed(random_state)

        # Standard NIRS wavelength range simulation
        self.n_wavelengths = 200  # Reduced for faster testing
        self.wavelengths = np.linspace(1000, 2500, self.n_wavelengths)

    def _generate_base_spectrum(self, pattern_type: str = "gaussian") -> np.ndarray:
        """Generate a base spectrum with characteristic patterns."""
        spectrum = np.zeros(self.n_wavelengths)

        if pattern_type == "gaussian":
            # Gaussian peaks at different wavelengths
            centers = [1200, 1500, 1900, 2200]
            for center in centers:
                idx = np.argmin(np.abs(self.wavelengths - center))
                spectrum += np.exp(-0.5 * ((np.arange(self.n_wavelengths) - idx) / 10) ** 2)

        elif pattern_type == "sawtooth":
            # Sawtooth pattern
            spectrum = np.mod(np.arange(self.n_wavelengths), 50) / 50.0

        elif pattern_type == "exponential":
            # Exponential decay
            spectrum = np.exp(-np.arange(self.n_wavelengths) / 50.0)

        return spectrum

    def generate_regression_data(self, n_samples: int = 100,
                                 noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate regression data with predictable target relationships.

        Args:
            n_samples: Number of samples to generate
            noise_level: Standard deviation of added noise

        Returns:
            X: Spectral data (n_samples, n_wavelengths)
            y: Continuous targets (n_samples,)
        """
        X = np.zeros((n_samples, self.n_wavelengths))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            # Target value influences spectrum type and intensity
            # Create clearly continuous values with multiple decimal places
            base_target = 10 + 40 * (i / n_samples)  # Range: 10-50
            # Add fine-grained random variation to ensure continuous values
            fine_variation = np.random.normal(0, 0.25, 1)[0]
            target = base_target + fine_variation
            # Round to 3 decimal places to ensure it's clearly continuous
            y[i] = round(target, 3)

            # Spectrum intensity correlates with target
            if target < 20:
                base = self._generate_base_spectrum("gaussian")
            elif target < 35:
                base = self._generate_base_spectrum("sawtooth")
            else:
                base = self._generate_base_spectrum("exponential")

            # Scale by target value with some nonlinearity
            intensity = 0.5 + 0.05 * target + 0.001 * target**2
            X[i] = intensity * base + noise_level * np.random.randn(self.n_wavelengths)

        return X, y

    def generate_classification_data(self, n_samples: int = 120,
                                     n_classes: int = 3,
                                     noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate classification data with distinct spectral signatures per class.

        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (2-5)
            noise_level: Standard deviation of added noise

        Returns:
            X: Spectral data (n_samples, n_wavelengths)
            y: Class labels (n_samples,) as integers 0, 1, 2, ...
        """
        samples_per_class = n_samples // n_classes
        X = np.zeros((n_samples, self.n_wavelengths))
        y = np.zeros(n_samples, dtype=int)

        pattern_types = ["gaussian", "sawtooth", "exponential", "gaussian", "sawtooth"][:n_classes]

        for class_idx in range(n_classes):
            start_idx = class_idx * samples_per_class
            end_idx = start_idx + samples_per_class if class_idx < n_classes - 1 else n_samples

            base_spectrum = self._generate_base_spectrum(pattern_types[class_idx])

            for i in range(start_idx, end_idx):
                # Add class-specific modifications
                spectrum = base_spectrum.copy()

                # Class-specific peak shifts
                if class_idx == 0:
                    spectrum *= (1.0 + 0.3 * np.sin(np.arange(self.n_wavelengths) * 0.1))
                elif class_idx == 1:
                    spectrum *= (1.0 + 0.2 * np.cos(np.arange(self.n_wavelengths) * 0.05))
                elif class_idx == 2:
                    spectrum += 0.2 * np.random.exponential(1, self.n_wavelengths)

                X[i] = spectrum + noise_level * np.random.randn(self.n_wavelengths)
                y[i] = class_idx

        return X, y

    def generate_multi_target_data(self, n_samples: int = 100,
                                   n_targets: int = 3,
                                   noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multi-target regression data.

        Args:
            n_samples: Number of samples
            n_targets: Number of target variables
            noise_level: Standard deviation of added noise

        Returns:
            X: Spectral data (n_samples, n_wavelengths)
            y: Multiple targets (n_samples, n_targets)
        """
        X = np.zeros((n_samples, self.n_wavelengths))
        y = np.zeros((n_samples, n_targets))

        for i in range(n_samples):
            # Generate base spectrum
            base = self._generate_base_spectrum("gaussian")

            # Each target correlates differently with spectral regions
            for t in range(n_targets):
                region_start = t * (self.n_wavelengths // n_targets)
                region_end = (t + 1) * (self.n_wavelengths // n_targets)

                # Target value based on specific spectral region
                base_value = 5 + 20 * np.mean(base[region_start:region_end])
                noise_value = 0.1 * noise_level * np.random.randn()
                y[i, t] = base_value + noise_value

            # Spectrum reflects all targets
            target_influence = np.sum(y[i]) / n_targets
            spectrum_base = (0.5 + 0.1 * target_influence) * base
            spectrum_noise = noise_level * np.random.randn(self.n_wavelengths)
            X[i] = spectrum_base + spectrum_noise

        return X, y


class TestDataManager:
    """Manages creation and cleanup of test datasets in temporary directories."""

    def __init__(self, base_temp_dir: Optional[str] = None):
        """
        Initialize test data manager.

        Args:
            base_temp_dir: Optional base directory for test data. If None, uses system temp.
        """
        if base_temp_dir:
            self.base_temp_dir = Path(base_temp_dir)
            self.base_temp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = tempfile.mkdtemp(dir=self.base_temp_dir)
        else:
            self.temp_dir = tempfile.mkdtemp()

        self.temp_path = Path(self.temp_dir)
        self.generator = SyntheticNIRSDataGenerator()

    def create_regression_dataset(self, name: str = "regression",
                                  n_train: int = 80, n_val: int = 20) -> Path:
        """Create a regression dataset in NIRS4ALL format."""
        dataset_path = self.temp_path / name
        dataset_path.mkdir(exist_ok=True)

        # Generate training data
        X_train, y_train = self.generator.generate_regression_data(n_train)

        # Generate validation data (different random state for variety)
        np.random.seed(self.generator.random_state + 1)
        X_val, y_val = self.generator.generate_regression_data(n_val)
        np.random.seed(self.generator.random_state)  # Reset

        # Save in NIRS4ALL format
        self._save_dataset(dataset_path, X_train, y_train, X_val, y_val)
        return dataset_path

    def create_classification_dataset(self, name: str = "classification",
                                      n_train: int = 90, n_val: int = 30,
                                      n_classes: int = 3) -> Path:
        """Create a classification dataset in NIRS4ALL format."""
        dataset_path = self.temp_path / name
        dataset_path.mkdir(exist_ok=True)

        # Generate training data
        X_train, y_train = self.generator.generate_classification_data(n_train, n_classes)

        # Generate validation data
        np.random.seed(self.generator.random_state + 2)
        X_val, y_val = self.generator.generate_classification_data(n_val, n_classes)
        np.random.seed(self.generator.random_state)  # Reset

        # Save in NIRS4ALL format
        self._save_dataset(dataset_path, X_train, y_train, X_val, y_val)
        return dataset_path

    def create_multi_target_dataset(self, name: str = "multi_target",
                                    n_train: int = 80, n_val: int = 20,
                                    n_targets: int = 3) -> Path:
        """Create a multi-target regression dataset in NIRS4ALL format."""
        dataset_path = self.temp_path / name
        dataset_path.mkdir(exist_ok=True)

        # Generate training data
        X_train, y_train = self.generator.generate_multi_target_data(n_train, n_targets)

        # Generate validation data
        np.random.seed(self.generator.random_state + 3)
        X_val, y_val = self.generator.generate_multi_target_data(n_val, n_targets)
        np.random.seed(self.generator.random_state)  # Reset

        # Save multi-target format
        self._save_multi_target_dataset(dataset_path, X_train, y_train, X_val, y_val)
        return dataset_path

    def _save_dataset(self, path: Path, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Save dataset in standard NIRS4ALL format."""
        # Save training data (using semicolon separator and no headers to match NIRS4ALL default)
        pd.DataFrame(X_train).to_csv(path / "Xcal.csv.gz", index=False, header=False,
                                     compression='gzip', sep=';')
        pd.DataFrame(y_train).to_csv(path / "Ycal.csv.gz", index=False, header=False,
                                     compression='gzip', sep=';')

        # Save validation data
        pd.DataFrame(X_val).to_csv(path / "Xval.csv.gz", index=False, header=False,
                                   compression='gzip', sep=';')
        pd.DataFrame(y_val).to_csv(path / "Yval.csv.gz", index=False, header=False,
                                   compression='gzip', sep=';')

    def _save_multi_target_dataset(self, path: Path, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray):
        """Save multi-target dataset."""
        # Save training data (using semicolon separator and no headers to match NIRS4ALL default)
        pd.DataFrame(X_train).to_csv(path / "Xcal.csv.gz", index=False, header=False,
                                     compression='gzip', sep=';')
        pd.DataFrame(y_train).to_csv(path / "Ycal.csv.gz", index=False, header=False,
                                     compression='gzip', sep=';')

        # Save validation data
        pd.DataFrame(X_val).to_csv(path / "Xval.csv.gz", index=False, header=False,
                                   compression='gzip', sep=';')
        pd.DataFrame(y_val).to_csv(path / "Yval.csv.gz", index=False, header=False,
                                   compression='gzip', sep=';')

    def get_temp_directory(self) -> Path:
        """Get the temporary directory path."""
        return self.temp_path

    def cleanup(self):
        """Clean up all temporary files and directories."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors in tests


def create_test_datasets() -> TestDataManager:
    """
    Convenience function to create all standard test datasets.

    Returns:
        TestDataManager instance with datasets created
    """
    manager = TestDataManager()

    # Create all standard test datasets
    manager.create_regression_dataset("regression")
    manager.create_classification_dataset("classification")
    manager.create_multi_target_dataset("multi_target")

    # Create additional datasets for multi-source testing
    manager.create_regression_dataset("regression_2")
    manager.create_regression_dataset("regression_3")

    return manager
