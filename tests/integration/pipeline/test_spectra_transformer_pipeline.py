"""
Integration tests for SpectraTransformerMixin operators in pipelines.

Tests end-to-end pipeline execution with wavelength-aware transformers.
"""

from typing import Optional

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.base import SpectraTransformerMixin
from nirs4all.pipeline import PipelineRunner
from tests.fixtures.data_generators import TestDataManager


class WavelengthRecordingTransformer(SpectraTransformerMixin):
    """Test transformer that records the wavelengths it receives.

    Used to verify that wavelengths are correctly passed through the pipeline.
    """

    _requires_wavelengths = True
    _fit_wavelengths_record = None
    _transform_wavelengths_record = None

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def fit(self, X, y=None, **fit_params):
        WavelengthRecordingTransformer._fit_wavelengths_record = fit_params.get('wavelengths')
        return self

    def _transform_impl(self, X, wavelengths):
        WavelengthRecordingTransformer._transform_wavelengths_record = wavelengths
        return X * self.scale

    @classmethod
    def reset_records(cls):
        """Reset the recorded wavelengths."""
        cls._fit_wavelengths_record = None
        cls._transform_wavelengths_record = None

    @classmethod
    def get_fit_wavelengths(cls) -> np.ndarray | None:
        """Get the wavelengths that were passed to fit."""
        return cls._fit_wavelengths_record

    @classmethod
    def get_transform_wavelengths(cls) -> np.ndarray | None:
        """Get the wavelengths that were passed to transform."""
        return cls._transform_wavelengths_record

class WavelengthDependentScaler(SpectraTransformerMixin):
    """Transformer that applies wavelength-dependent scaling.

    Demonstrates a realistic use case where the transformation
    depends on the wavelength values.
    """

    _requires_wavelengths = True

    def __init__(self, target_wavelength: float = 1500.0, boost_factor: float = 1.5):
        self.target_wavelength = target_wavelength
        self.boost_factor = boost_factor

    def _transform_impl(self, X, wavelengths):
        """Boost signal intensity near the target wavelength."""
        X_out = X.copy()

        # Find wavelengths within 100nm of target
        distance = np.abs(wavelengths - self.target_wavelength)
        weight = np.exp(-distance / 100.0)

        # Apply wavelength-dependent scaling
        X_out = X_out * (1.0 + (self.boost_factor - 1.0) * weight)

        return X_out

class OptionalWavelengthTransformer(SpectraTransformerMixin):
    """Transformer that can work with or without wavelengths."""

    _requires_wavelengths = False

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.wavelengths_used = None

    def _transform_impl(self, X, wavelengths):
        self.wavelengths_used = wavelengths
        return X * self.scale

class TestSpectraTransformerPipeline:
    """Integration tests for SpectraTransformerMixin in pipelines."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    @pytest.fixture(autouse=True)
    def reset_recording_transformer(self):
        """Reset the recording transformer before each test."""
        WavelengthRecordingTransformer.reset_records()
        yield

    def test_spectra_transformer_in_simple_pipeline(self, test_data_manager):
        """Test that SpectraTransformerMixin receives wavelengths in a simple pipeline."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)
        dataset = dataset_config.get_datasets()[0]

        # Get expected wavelengths from dataset
        n_features = dataset.x({"partition": "all"}).shape[-1]

        pipeline = [
            WavelengthRecordingTransformer(scale=1.0),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify wavelengths were passed to transform (controller extracts from dataset)
        transform_wavelengths = WavelengthRecordingTransformer.get_transform_wavelengths()
        assert transform_wavelengths is not None
        assert len(transform_wavelengths) == n_features

        # Verify wavelengths were passed to fit
        fit_wavelengths = WavelengthRecordingTransformer.get_fit_wavelengths()
        assert fit_wavelengths is not None
        assert len(fit_wavelengths) == n_features

        # Verify both are the same wavelengths
        np.testing.assert_array_equal(transform_wavelengths, fit_wavelengths)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_spectra_transformer_with_standard_transformer(self, test_data_manager):
        """Test SpectraTransformerMixin mixed with standard sklearn transformers."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)
        dataset = dataset_config.get_datasets()[0]

        # Get expected wavelengths from dataset
        n_features = dataset.x({"partition": "all"}).shape[-1]

        pipeline = [
            StandardScaler(),  # Standard transformer (no wavelengths)
            WavelengthRecordingTransformer(scale=1.0),  # Spectra transformer (with wavelengths)
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify spectra transformer received wavelengths
        transform_wavelengths = WavelengthRecordingTransformer.get_transform_wavelengths()
        assert transform_wavelengths is not None
        assert len(transform_wavelengths) == n_features

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_wavelength_dependent_transformation(self, test_data_manager):
        """Test that wavelength-dependent transformations work correctly."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            WavelengthDependentScaler(target_wavelength=100.0, boost_factor=1.5),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_optional_wavelength_transformer(self, test_data_manager):
        """Test that transformers with optional wavelengths work without wavelengths."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            OptionalWavelengthTransformer(scale=2.0),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_multiple_spectra_transformers_in_pipeline(self, test_data_manager):
        """Test multiple SpectraTransformerMixin operators in sequence."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)
        dataset = dataset_config.get_datasets()[0]

        # Get expected wavelengths from dataset
        n_features = dataset.x({"partition": "all"}).shape[-1]

        pipeline = [
            WavelengthDependentScaler(target_wavelength=50.0, boost_factor=1.2),
            WavelengthDependentScaler(target_wavelength=150.0, boost_factor=1.3),
            WavelengthRecordingTransformer(scale=1.0),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify the last spectra transformer received wavelengths
        transform_wavelengths = WavelengthRecordingTransformer.get_transform_wavelengths()
        assert transform_wavelengths is not None
        assert len(transform_wavelengths) == n_features

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

class TestSpectraTransformerBackwardCompatibility:
    """Tests ensuring backward compatibility with existing pipelines."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    def test_existing_sklearn_transformers_unchanged(self, test_data_manager):
        """Test that existing sklearn transformers work without changes."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_nirs4all_existing_transforms_unchanged(self, test_data_manager):
        """Test that existing nirs4all transforms work without changes."""
        from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate

        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        dataset_config = DatasetConfigs(dataset_folder)

        pipeline = [
            StandardNormalVariate(),
            SavitzkyGolay(window_length=11, polyorder=2),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

class TestEnvironmentalAugmentersPipeline:
    """Integration tests for environmental/scattering augmenters in pipelines."""

    @pytest.fixture
    def test_data_manager(self):
        """Create test data manager with datasets."""
        manager = TestDataManager()
        manager.create_regression_dataset("regression")
        yield manager
        manager.cleanup()

    @pytest.fixture
    def dataset_config(self, test_data_manager):
        """Create dataset config with header_unit set to index for stable wavelengths."""
        dataset_folder = str(test_data_manager.get_temp_directory() / "regression")
        return DatasetConfigs({
            "folder": dataset_folder,
            "global_params": {"header_unit": "index"}
        })

    def test_temperature_augmenter_in_pipeline(self, dataset_config):
        """Test TemperatureAugmenter in a complete pipeline."""
        from nirs4all.operators import TemperatureAugmenter

        pipeline = [
            TemperatureAugmenter(temperature_range=(-5, 10), random_state=42),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_moisture_augmenter_in_pipeline(self, dataset_config):
        """Test MoistureAugmenter in a complete pipeline."""
        from nirs4all.operators import MoistureAugmenter

        pipeline = [
            MoistureAugmenter(water_activity_range=(-0.2, 0.2), random_state=42),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_particle_size_augmenter_in_pipeline(self, dataset_config):
        """Test ParticleSizeAugmenter in a complete pipeline."""
        from nirs4all.operators import ParticleSizeAugmenter

        pipeline = [
            ParticleSizeAugmenter(size_range_um=(20, 100), random_state=42),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_emsc_distortion_augmenter_in_pipeline(self, dataset_config):
        """Test EMSCDistortionAugmenter in a complete pipeline."""
        from nirs4all.operators import EMSCDistortionAugmenter

        pipeline = [
            EMSCDistortionAugmenter(polynomial_order=2, random_state=42),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_combined_environmental_augmenters(self, dataset_config):
        """Test multiple environmental augmenters in sequence."""
        from nirs4all.operators import (
            MoistureAugmenter,
            ParticleSizeAugmenter,
            TemperatureAugmenter,
        )

        pipeline = [
            TemperatureAugmenter(temperature_delta=5.0, random_state=42),
            MoistureAugmenter(water_activity_delta=0.1, random_state=43),
            ParticleSizeAugmenter(mean_size_um=40.0, random_state=44),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_environmental_augmenters_with_preprocessing(self, dataset_config):
        """Test environmental augmenters combined with standard preprocessing."""
        from nirs4all.operators import EMSCDistortionAugmenter, TemperatureAugmenter
        from nirs4all.operators.transforms import StandardNormalVariate

        pipeline = [
            # Apply environmental augmentation first
            TemperatureAugmenter(temperature_range=(-5, 10), random_state=42),
            EMSCDistortionAugmenter(multiplicative_range=(0.9, 1.1), random_state=43),
            # Then preprocessing
            StandardNormalVariate(),
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0

    def test_environmental_augmenters_with_standard_scaler(self, dataset_config):
        """Test environmental augmenters with sklearn StandardScaler."""
        from nirs4all.operators import ParticleSizeAugmenter

        pipeline = [
            StandardScaler(),  # Standard sklearn transformer
            ParticleSizeAugmenter(mean_size_um=50.0, random_state=42),  # Spectra transformer
            StandardScaler(),  # Another standard transformer
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        runner = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
        predictions, _ = runner.run(pipeline, dataset_config)

        # Verify pipeline ran successfully
        assert predictions.num_predictions > 0
