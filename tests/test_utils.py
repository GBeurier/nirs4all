"""
Test configuration and utilities for dataset testing.

This module provides common test utilities, fixtures, and configuration
used across all dataset tests.
"""

import pytest
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple
from pathlib import Path

from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.dataset.indexer import Indexer


class TestConfig:
    """Test configuration constants."""

    # Data dimensions
    N_TRAIN_SAMPLES = 80
    N_TEST_SAMPLES = 20
    N_VAL_SAMPLES = 15
    N_FEATURES_DEFAULT = 50
    N_FEATURES_LARGE = 200

    # Random seeds for reproducibility
    SEED_FEATURES = 42
    SEED_TARGETS = 43
    SEED_METADATA = 44
    SEED_AUGMENTATION = 45

    # Partitions
    PARTITIONS = ["train", "test", "val"]

    # Processing types
    PROCESSING_TYPES = ["raw", "normalized", "smoothed", "augmented"]

    # Augmentation types
    AUGMENTATION_TYPES = ["rotate", "translate", "noise", "scale"]

    # Groups for cross-validation
    CV_GROUPS = [0, 1, 2, 3, 4]

    # Branches for multi-branch processing
    BRANCHES = [0, 1, 2]


class SampleDataGenerator:
    """Utility class for generating realistic test data."""

    @staticmethod
    def create_spectral_features(
        n_samples: int,
        n_features: int = TestConfig.N_FEATURES_DEFAULT,
        seed: int = TestConfig.SEED_FEATURES,
        spectral_type: str = "nir"
    ) -> np.ndarray:
        """
        Create realistic spectral data.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of spectral features (wavelengths)
            seed: Random seed for reproducibility
            spectral_type: Type of spectral data ("nir", "vis", "raman")

        Returns:
            2D numpy array of shape (n_samples, n_features)
        """
        np.random.seed(seed)

        if spectral_type == "nir":
            # NIR wavelengths typically 700-2500 nm
            wavelengths = np.linspace(700, 2500, n_features)
        elif spectral_type == "vis":
            # Visible wavelengths typically 400-700 nm
            wavelengths = np.linspace(400, 700, n_features)
        elif spectral_type == "raman":
            # Raman shifts typically 0-4000 cm-1
            wavelengths = np.linspace(0, 4000, n_features)
        else:
            wavelengths = np.linspace(400, 2500, n_features)

        data = np.zeros((n_samples, n_features), dtype=np.float32)

        for i in range(n_samples):
            # Base spectrum with noise
            spectrum = np.random.randn(n_features) * 0.05

            # Add baseline drift
            baseline = np.random.randn() * 0.3
            drift = np.linspace(0, np.random.randn() * 0.2, n_features)
            spectrum += baseline + drift

            # Add realistic peaks based on spectral type
            if spectral_type == "nir":
                # NIR absorption bands
                spectrum += SampleDataGenerator._add_nir_peaks(wavelengths, i)
            elif spectral_type == "raman":
                # Raman peaks
                spectrum += SampleDataGenerator._add_raman_peaks(wavelengths, i)

            data[i] = spectrum

        return data

    @staticmethod
    def _add_nir_peaks(wavelengths: np.ndarray, sample_idx: int) -> np.ndarray:
        """Add realistic NIR absorption peaks."""
        peaks = np.zeros_like(wavelengths)

        # Water absorption around 1450 nm and 1940 nm
        peaks += np.exp(-((wavelengths - 1450) / 50) ** 2) * (0.5 + np.random.randn() * 0.1)
        peaks += np.exp(-((wavelengths - 1940) / 60) ** 2) * (0.3 + np.random.randn() * 0.1)

        # Protein absorption around 2180 nm
        peaks += np.exp(-((wavelengths - 2180) / 80) ** 2) * (0.4 + np.random.randn() * 0.1)

        # Sample-specific variations
        if sample_idx % 3 == 0:
            # High moisture samples
            peaks += np.exp(-((wavelengths - 1400) / 40) ** 2) * 0.2
        elif sample_idx % 3 == 1:
            # High protein samples
            peaks += np.exp(-((wavelengths - 2200) / 60) ** 2) * 0.3

        return peaks

    @staticmethod
    def _add_raman_peaks(wavelengths: np.ndarray, sample_idx: int) -> np.ndarray:
        """Add realistic Raman peaks."""
        peaks = np.zeros_like(wavelengths)

        # Common Raman peaks
        common_peaks = [1000, 1600, 2900, 3400]
        for peak_pos in common_peaks:
            intensity = 0.3 + np.random.randn() * 0.1
            width = 20 + np.random.randn() * 5
            peaks += np.exp(-((wavelengths - peak_pos) / width) ** 2) * intensity

        return peaks

    @staticmethod
    def create_classification_targets(
        n_samples: int,
        n_classes: int = 3,
        seed: int = TestConfig.SEED_TARGETS,
        class_balance: List[float] = None
    ) -> np.ndarray:
        """Create classification target data."""
        np.random.seed(seed)

        if class_balance is None:
            # Balanced classes
            targets = np.random.randint(0, n_classes, n_samples)
        else:
            # Custom class balance
            assert len(class_balance) == n_classes
            assert abs(sum(class_balance) - 1.0) < 1e-6

            targets = np.random.choice(
                n_classes,
                n_samples,
                p=class_balance
            )

        return targets.astype(np.int32)

    @staticmethod
    def create_regression_targets(
        n_samples: int,
        target_range: Tuple[float, float] = (0.0, 100.0),
        noise_level: float = 0.1,
        seed: int = TestConfig.SEED_TARGETS
    ) -> np.ndarray:
        """Create regression target data."""
        np.random.seed(seed)

        # Create targets with some underlying pattern plus noise
        base_values = np.linspace(target_range[0], target_range[1], n_samples)
        noise = np.random.randn(n_samples) * noise_level * (target_range[1] - target_range[0])

        # Shuffle to remove linear pattern
        indices = np.random.permutation(n_samples)
        targets = base_values[indices] + noise

        return targets.astype(np.float32)

    @staticmethod
    def create_metadata(
        n_samples: int,
        seed: int = TestConfig.SEED_METADATA
    ) -> pl.DataFrame:
        """Create realistic metadata."""
        np.random.seed(seed)

        return pl.DataFrame({
            "sample_id": [f"sample_{i:04d}" for i in range(n_samples)],
            "batch": np.random.choice(["batch_A", "batch_B", "batch_C", "batch_D"], n_samples),
            "instrument": np.random.choice(["NIR_001", "NIR_002", "VIS_001"], n_samples),
            "operator": np.random.choice(["operator_1", "operator_2", "operator_3"], n_samples),
            "temperature": np.random.uniform(18.0, 25.0, n_samples),
            "humidity": np.random.uniform(35.0, 65.0, n_samples),
            "measurement_date": [
                f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
                for _ in range(n_samples)
            ],
            "quality_flag": np.random.choice(["good", "acceptable", "poor"], n_samples, p=[0.7, 0.2, 0.1])
        })


class TestDatasetBuilder:
    """Builder class for creating test datasets with specific configurations."""

    def __init__(self):
        self.dataset = SpectroDataset()
        self.data_generator = SampleDataGenerator()

    def with_train_test_split(
        self,
        n_train: int = TestConfig.N_TRAIN_SAMPLES,
        n_test: int = TestConfig.N_TEST_SAMPLES,
        n_features: int = TestConfig.N_FEATURES_DEFAULT
    ) -> 'TestDatasetBuilder':
        """Add train/test split to dataset."""
        # Add training data
        train_features = self.data_generator.create_spectral_features(n_train, n_features, seed=42)
        train_targets = self.data_generator.create_classification_targets(n_train, seed=42)

        self.dataset.add_features({"partition": "train"}, train_features)
        self.dataset.add_targets(train_targets)

        # Add test data
        test_features = self.data_generator.create_spectral_features(n_test, n_features, seed=43)
        test_targets = self.data_generator.create_classification_targets(n_test, seed=43)

        self.dataset.add_features({"partition": "test"}, test_features)
        self.dataset.add_targets(test_targets)

        return self

    def with_cross_validation(self, n_folds: int = 5) -> 'TestDatasetBuilder':
        """Add cross-validation folds."""
        # TODO: Implement cross-validation setup
        return self

    def with_augmentation(self, augmentation_factor: int = 2) -> 'TestDatasetBuilder':
        """Add sample augmentation."""
        # TODO: Implement augmentation setup
        return self

    def with_metadata(self) -> 'TestDatasetBuilder':
        """Add metadata to dataset."""
        # TODO: Implement metadata addition
        return self

    def with_multi_source(self, n_sources: int = 2) -> 'TestDatasetBuilder':
        """Add multiple feature sources."""
        # TODO: Implement multi-source setup
        return self

    def build(self) -> SpectroDataset:
        """Return the built dataset."""
        return self.dataset


@pytest.fixture
def empty_dataset():
    """Fixture providing an empty SpectroDataset."""
    return SpectroDataset()


@pytest.fixture
def empty_indexer():
    """Fixture providing an empty Indexer."""
    return Indexer()


@pytest.fixture
def sample_data_generator():
    """Fixture providing a SampleDataGenerator instance."""
    return SampleDataGenerator()


@pytest.fixture
def basic_dataset():
    """Fixture providing a basic dataset with train/test split."""
    return TestDatasetBuilder().with_train_test_split().build()


@pytest.fixture
def complex_dataset():
    """Fixture providing a complex dataset with multiple features."""
    return (TestDatasetBuilder()
            .with_train_test_split()
            .with_cross_validation()
            .with_metadata()
            .build())


# Test data validation utilities
class ValidationUtils:
    """Utilities for validating test results."""

    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
        """Validate array has expected shape."""
        return array.shape == expected_shape

    @staticmethod
    def validate_data_consistency(dataset: SpectroDataset, filter_dict: Dict[str, Any]) -> bool:
        """Validate that features and targets are consistent for given filter."""
        try:
            features = dataset.features(filter_dict)
            targets = dataset.targets(filter_dict)

            if isinstance(features, tuple):
                # Multi-source case
                return all(f.shape[0] == targets.shape[0] for f in features)
            else:
                # Single source case
                return features.shape[0] == targets.shape[0]
        except Exception:
            return False

    @staticmethod
    def validate_indexer_consistency(indexer: Indexer) -> bool:
        """Validate indexer internal consistency."""
        # TODO: Implement comprehensive indexer validation
        return True


if __name__ == "__main__":
    # Example usage
    generator = SampleDataGenerator()

    # Generate sample data
    features = generator.create_spectral_features(100, 50)
    targets = generator.create_classification_targets(100)
    metadata = generator.create_metadata(100)

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Metadata shape: {metadata.shape}")

    # Build test dataset
    dataset = (TestDatasetBuilder()
               .with_train_test_split()
               .with_metadata()
               .build())

    print("Test dataset created successfully")
