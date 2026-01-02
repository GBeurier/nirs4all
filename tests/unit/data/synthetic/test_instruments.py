"""
Unit tests for the instruments module (Phase 2.1).

Tests cover:
- Instrument archetypes and registry
- Multi-sensor configuration
- Multi-scan configuration
- InstrumentSimulator behavior
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.synthetic.instruments import (
    InstrumentCategory,
    DetectorType,
    MonochromatorType,
    SensorConfig,
    MultiSensorConfig,
    MultiScanConfig,
    InstrumentArchetype,
    INSTRUMENT_ARCHETYPES,
    get_instrument_archetype,
    list_instrument_archetypes,
    get_instruments_by_category,
    InstrumentSimulator,
)


class TestEnums:
    """Tests for instrument-related enums."""

    def test_instrument_category_values(self):
        """Test that all expected categories exist."""
        categories = [c.value for c in InstrumentCategory]
        assert "benchtop" in categories
        assert "handheld" in categories
        assert "process" in categories

    def test_detector_type_values(self):
        """Test that all expected detector types exist."""
        detectors = [d.value for d in DetectorType]
        assert "si" in detectors
        assert "ingaas" in detectors
        assert "ingaas_ext" in detectors
        assert "pbs" in detectors
        assert "mems" in detectors

    def test_monochromator_type_values(self):
        """Test that all expected monochromator types exist."""
        monos = [m.value for m in MonochromatorType]
        assert "grating" in monos
        assert "fourier_transform" in monos
        assert "dmd" in monos


class TestSensorConfig:
    """Tests for SensorConfig dataclass."""

    def test_sensor_config_creation(self):
        """Test creating a sensor configuration."""
        config = SensorConfig(
            detector_type=DetectorType.INGAAS,
            wavelength_range=(900, 1700),
            spectral_resolution=10.0,
            noise_level=1.5,
        )
        assert config.detector_type == DetectorType.INGAAS
        assert config.wavelength_range == (900, 1700)
        assert config.spectral_resolution == 10.0
        assert config.noise_level == 1.5

    def test_sensor_config_defaults(self):
        """Test sensor config default values."""
        config = SensorConfig(
            detector_type=DetectorType.SI,
            wavelength_range=(400, 1100),
        )
        assert config.spectral_resolution == 8.0  # Default
        assert config.noise_level == 1.0  # Default
        assert config.gain == 1.0  # Default
        assert config.overlap_range == 20.0  # Default


class TestMultiSensorConfig:
    """Tests for MultiSensorConfig dataclass."""

    def test_multi_sensor_config_creation(self):
        """Test creating a multi-sensor configuration."""
        sensors = [
            SensorConfig(DetectorType.SI, (400, 1100)),
            SensorConfig(DetectorType.INGAAS, (900, 1700)),
        ]
        config = MultiSensorConfig(
            enabled=True,
            sensors=sensors,
            stitch_method="weighted",
            stitch_smoothing=10.0,
            add_stitch_artifacts=True,
            artifact_intensity=0.02,
        )
        assert len(config.sensors) == 2
        assert config.stitch_method == "weighted"
        assert config.enabled is True

    def test_multi_sensor_stitch_methods(self):
        """Test that different stitch methods are accepted."""
        sensors = [
            SensorConfig(DetectorType.SI, (400, 1100)),
            SensorConfig(DetectorType.INGAAS, (900, 1700)),
        ]
        for method in ["weighted", "average", "first", "last", "optimal"]:
            config = MultiSensorConfig(enabled=True, sensors=sensors, stitch_method=method)
            assert config.stitch_method == method


class TestMultiScanConfig:
    """Tests for MultiScanConfig dataclass."""

    def test_multi_scan_config_creation(self):
        """Test creating a multi-scan configuration."""
        config = MultiScanConfig(
            enabled=True,
            n_scans=32,
            averaging_method="mean",
            scan_to_scan_noise=0.001,
            discard_outliers=True,
            outlier_threshold=3.0,
        )
        assert config.n_scans == 32
        assert config.averaging_method == "mean"
        assert config.discard_outliers is True
        assert config.enabled is True

    def test_multi_scan_averaging_methods(self):
        """Test that different averaging methods are accepted."""
        for method in ["mean", "median", "weighted", "savgol"]:
            config = MultiScanConfig(enabled=True, n_scans=16, averaging_method=method)
            assert config.averaging_method == method


class TestInstrumentArchetype:
    """Tests for InstrumentArchetype dataclass."""

    def test_instrument_archetype_creation(self):
        """Test creating an instrument archetype."""
        archetype = InstrumentArchetype(
            name="test_instrument",
            category=InstrumentCategory.BENCHTOP,
            detector_type=DetectorType.INGAAS,
            monochromator_type=MonochromatorType.GRATING,
            wavelength_range=(1100, 2500),
            spectral_resolution=2.0,
            snr=10000.0,
            scan_speed=10.0,
        )
        assert archetype.name == "test_instrument"
        assert archetype.category == InstrumentCategory.BENCHTOP
        assert archetype.wavelength_range == (1100, 2500)

    def test_instrument_with_multi_sensor(self):
        """Test creating an instrument with multi-sensor config."""
        multi_sensor = MultiSensorConfig(
            sensors=[
                SensorConfig(DetectorType.SI, (400, 1100)),
                SensorConfig(DetectorType.INGAAS, (900, 2500)),
            ],
            stitch_method="weighted",
        )
        archetype = InstrumentArchetype(
            name="multi_sensor_test",
            category=InstrumentCategory.BENCHTOP,
            detector_type=DetectorType.INGAAS,
            monochromator_type=MonochromatorType.GRATING,
            wavelength_range=(400, 2500),
            multi_sensor=multi_sensor,
        )
        assert archetype.multi_sensor is not None
        assert len(archetype.multi_sensor.sensors) == 2

    def test_instrument_with_multi_scan(self):
        """Test creating an instrument with multi-scan config."""
        multi_scan = MultiScanConfig(n_scans=32, averaging_method="mean")
        archetype = InstrumentArchetype(
            name="multi_scan_test",
            category=InstrumentCategory.BENCHTOP,
            detector_type=DetectorType.INGAAS,
            monochromator_type=MonochromatorType.FT,
            wavelength_range=(1100, 2500),
            multi_scan=multi_scan,
        )
        assert archetype.multi_scan is not None
        assert archetype.multi_scan.n_scans == 32


class TestInstrumentRegistry:
    """Tests for instrument archetype registry."""

    def test_instrument_archetypes_not_empty(self):
        """Test that the registry contains instruments."""
        assert len(INSTRUMENT_ARCHETYPES) > 0

    def test_get_instrument_archetype(self):
        """Test getting an instrument by name."""
        # Should have some common instruments
        archetype = get_instrument_archetype("foss_xds")
        assert archetype is not None
        assert archetype.name == "foss_xds"

    def test_get_unknown_instrument_raises(self):
        """Test that getting an unknown instrument raises KeyError."""
        with pytest.raises(KeyError):
            get_instrument_archetype("nonexistent_instrument")

    def test_list_instrument_archetypes(self):
        """Test listing all available instruments."""
        instruments = list_instrument_archetypes()
        assert len(instruments) > 0
        assert "foss_xds" in instruments

    def test_list_instruments_by_category(self):
        """Test listing instruments filtered by category."""
        all_by_cat = get_instruments_by_category()

        benchtop = all_by_cat.get("benchtop", [])
        process = all_by_cat.get("process", [])
        handheld = all_by_cat.get("handheld", [])

        # Each category should have at least one instrument
        assert len(benchtop) > 0
        assert len(handheld) > 0
        # Process may or may not have instruments

        # No overlap between benchtop and handheld
        assert set(benchtop).isdisjoint(set(handheld))

    def test_known_instruments_exist(self):
        """Test that expected instruments are in the registry."""
        expected = ["foss_xds", "bruker_mpa", "viavi_micronir", "scio"]
        available = list_instrument_archetypes()
        for name in expected:
            assert name in available, f"{name} should be in registry"


class TestInstrumentSimulator:
    """Tests for InstrumentSimulator class."""

    @pytest.fixture
    def sample_archetype(self):
        """Create a sample instrument archetype."""
        return get_instrument_archetype("foss_xds")

    @pytest.fixture
    def sample_spectra(self):
        """Create sample spectra for testing."""
        n_samples = 10
        n_wl = 100
        return np.random.default_rng(42).normal(0.5, 0.1, (n_samples, n_wl))

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelength array."""
        return np.linspace(1100, 2500, 100)

    def test_simulator_creation(self, sample_archetype):
        """Test creating an instrument simulator."""
        simulator = InstrumentSimulator(sample_archetype, random_state=42)
        assert simulator is not None
        assert simulator.archetype == sample_archetype

    def test_simulator_apply(self, sample_archetype, sample_spectra, sample_wavelengths):
        """Test applying instrument effects to spectra."""
        simulator = InstrumentSimulator(sample_archetype, random_state=42)
        result, output_wl = simulator.apply(sample_spectra, sample_wavelengths)

        # Output should have same number of samples
        assert result.shape[0] == sample_spectra.shape[0]
        # Output wavelengths should be returned
        assert len(output_wl) > 0
        # Output should be within instrument range
        wl_min, wl_max = sample_archetype.wavelength_range
        assert output_wl.min() >= wl_min
        assert output_wl.max() <= wl_max

    def test_simulator_apply_noise(self, sample_archetype, sample_spectra, sample_wavelengths):
        """Test that simulator adds noise."""
        # Apply twice with same seed should give same result
        sim1 = InstrumentSimulator(sample_archetype, random_state=42)
        sim2 = InstrumentSimulator(sample_archetype, random_state=42)

        result1, wl1 = sim1.apply(sample_spectra.copy(), sample_wavelengths)
        result2, wl2 = sim2.apply(sample_spectra.copy(), sample_wavelengths)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_simulator_different_seeds(self, sample_archetype, sample_spectra, sample_wavelengths):
        """Test that different seeds produce different results."""
        sim1 = InstrumentSimulator(sample_archetype, random_state=42)
        sim2 = InstrumentSimulator(sample_archetype, random_state=123)

        result1, _ = sim1.apply(sample_spectra.copy(), sample_wavelengths)
        result2, _ = sim2.apply(sample_spectra.copy(), sample_wavelengths)

        # Results should differ
        assert not np.allclose(result1, result2)


class TestMultiSensorInstruments:
    """Tests for instruments with multi-sensor configurations."""

    def test_foss_xds_has_multi_sensor(self):
        """Test that FOSS XDS has multi-sensor configuration."""
        archetype = get_instrument_archetype("foss_xds")
        assert archetype.multi_sensor is not None
        # XDS typically has Si + PbS
        assert len(archetype.multi_sensor.sensors) >= 2

    def test_multi_sensor_overlap_regions(self):
        """Test multi-sensor overlap region handling."""
        archetype = get_instrument_archetype("foss_xds")
        if archetype.multi_sensor is None:
            pytest.skip("Instrument doesn't have multi-sensor")

        sensors = archetype.multi_sensor.sensors

        # Check for overlap between adjacent sensors
        for i in range(len(sensors) - 1):
            sensor1_end = sensors[i].wavelength_range[1]
            sensor2_start = sensors[i + 1].wavelength_range[0]
            # Either overlap or adjacent
            assert sensor1_end >= sensor2_start or sensor2_start - sensor1_end < 100


class TestMultiScanInstruments:
    """Tests for instruments with multi-scan configurations."""

    def test_ft_instrument_has_multi_scan(self):
        """Test that FT instruments typically have multi-scan."""
        archetype = get_instrument_archetype("bruker_mpa")
        # FT instruments often use multi-scan averaging
        assert archetype.monochromator_type == MonochromatorType.FT
        if archetype.multi_scan is not None:
            assert archetype.multi_scan.n_scans > 1

    def test_handheld_typical_scans(self):
        """Test typical scan counts for handheld instruments."""
        archetype = get_instrument_archetype("scio")
        # Handheld MEMS instruments often use many scans to improve SNR
        if archetype.multi_scan is not None:
            assert archetype.multi_scan.n_scans >= 1
            # Some handhelds use median averaging for robustness
            assert archetype.multi_scan.averaging_method in ["mean", "median", "weighted"]
