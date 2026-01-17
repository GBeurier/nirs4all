"""
Unit tests for Phase 6: Custom Wavelength Support.

Tests for custom wavelength grids in SyntheticNIRSGenerator and SyntheticDatasetBuilder.
"""

import numpy as np
import pytest


class TestInstrumentWavelengths:
    """Tests for INSTRUMENT_WAVELENGTHS and related functions."""

    def test_instrument_wavelengths_dict_exists(self):
        """Test INSTRUMENT_WAVELENGTHS dictionary exists and has entries."""
        from nirs4all.data.synthetic import INSTRUMENT_WAVELENGTHS

        assert isinstance(INSTRUMENT_WAVELENGTHS, dict)
        assert len(INSTRUMENT_WAVELENGTHS) > 0

    def test_instrument_wavelengths_are_arrays(self):
        """Test all instrument wavelengths are numpy arrays."""
        from nirs4all.data.synthetic import INSTRUMENT_WAVELENGTHS

        for name, wl in INSTRUMENT_WAVELENGTHS.items():
            assert isinstance(wl, np.ndarray), f"{name} is not a numpy array"
            assert len(wl) > 0, f"{name} is empty"
            assert wl.ndim == 1, f"{name} is not 1D"

    def test_get_instrument_wavelengths(self):
        """Test get_instrument_wavelengths function."""
        from nirs4all.data.synthetic import get_instrument_wavelengths

        wl = get_instrument_wavelengths("micronir_onsite")

        assert isinstance(wl, np.ndarray)
        assert len(wl) == 125  # MicroNIR has 125 wavelengths
        assert wl[0] == pytest.approx(908, rel=0.01)
        assert wl[-1] == pytest.approx(1676, rel=0.01)

    def test_get_instrument_wavelengths_copy(self):
        """Test that get_instrument_wavelengths returns a copy."""
        from nirs4all.data.synthetic import get_instrument_wavelengths

        wl1 = get_instrument_wavelengths("micronir_onsite")
        wl2 = get_instrument_wavelengths("micronir_onsite")

        # Modify first one
        wl1[0] = 9999

        # Second should be unaffected
        assert wl2[0] != 9999

    def test_get_instrument_wavelengths_unknown(self):
        """Test get_instrument_wavelengths raises for unknown instrument."""
        from nirs4all.data.synthetic import get_instrument_wavelengths

        with pytest.raises(ValueError, match="Unknown instrument"):
            get_instrument_wavelengths("nonexistent_instrument")

    def test_list_instrument_wavelength_grids(self):
        """Test list_instrument_wavelength_grids function."""
        from nirs4all.data.synthetic import list_instrument_wavelength_grids

        grids = list_instrument_wavelength_grids()

        assert isinstance(grids, list)
        assert "micronir_onsite" in grids
        assert "foss_xds" in grids

    def test_get_instrument_wavelength_info(self):
        """Test get_instrument_wavelength_info function."""
        from nirs4all.data.synthetic import get_instrument_wavelength_info

        info = get_instrument_wavelength_info()

        assert isinstance(info, dict)
        assert "micronir_onsite" in info

        micro_info = info["micronir_onsite"]
        assert "n_wavelengths" in micro_info
        assert "wavelength_start" in micro_info
        assert "wavelength_end" in micro_info
        assert "mean_step" in micro_info


class TestSyntheticNIRSGeneratorWavelengths:
    """Tests for custom wavelengths in SyntheticNIRSGenerator."""

    def test_default_wavelengths(self):
        """Test default wavelength grid is used when none specified."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(random_state=42)

        # Default is 350-2500 nm at 2nm step
        assert gen.wavelengths[0] == 350
        assert gen.wavelengths[-1] == 2500
        assert len(gen.wavelengths) > 1000

    def test_custom_wavelengths_array(self):
        """Test custom wavelengths array parameter."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        custom_wl = np.linspace(1000, 2000, 100)

        gen = SyntheticNIRSGenerator(
            wavelengths=custom_wl,
            random_state=42,
        )

        assert len(gen.wavelengths) == 100
        assert np.allclose(gen.wavelengths, custom_wl)
        assert gen.wavelength_start == 1000
        assert gen.wavelength_end == 2000

    def test_instrument_wavelength_grid(self):
        """Test instrument_wavelength_grid parameter."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(
            instrument_wavelength_grid="micronir_onsite",
            random_state=42,
        )

        assert len(gen.wavelengths) == 125
        assert gen.wavelengths[0] == pytest.approx(908, rel=0.01)

    def test_generate_with_custom_wavelengths(self):
        """Test spectrum generation with custom wavelengths."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        custom_wl = np.linspace(1000, 2000, 50)

        gen = SyntheticNIRSGenerator(
            wavelengths=custom_wl,
            complexity="simple",
            random_state=42,
        )

        X, Y, E = gen.generate(n_samples=10)

        assert X.shape == (10, 50)
        assert E.shape[1] == 50

    def test_generate_with_instrument_grid(self):
        """Test spectrum generation with instrument wavelength grid."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(
            instrument_wavelength_grid="micronir_onsite",
            complexity="simple",
            random_state=42,
        )

        X, Y, E = gen.generate(n_samples=10)

        assert X.shape == (10, 125)

    def test_wavelengths_override_range(self):
        """Test that custom wavelengths override wavelength_start/end/step."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        custom_wl = np.linspace(1000, 2000, 100)

        gen = SyntheticNIRSGenerator(
            wavelength_start=500,  # Should be overridden
            wavelength_end=3000,   # Should be overridden
            wavelengths=custom_wl,
            random_state=42,
        )

        # wavelengths param should win
        assert gen.wavelength_start == 1000
        assert gen.wavelength_end == 2000

    def test_instrument_grid_unknown(self):
        """Test that unknown instrument grid raises error."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        with pytest.raises(ValueError, match="Unknown instrument"):
            SyntheticNIRSGenerator(
                instrument_wavelength_grid="nonexistent_instrument",
                random_state=42,
            )


class TestSyntheticDatasetBuilderWavelengths:
    """Tests for with_wavelengths() in SyntheticDatasetBuilder."""

    def test_with_wavelengths_array(self):
        """Test with_wavelengths with array parameter."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        custom_wl = np.linspace(1000, 2000, 100)

        builder = (
            SyntheticDatasetBuilder(n_samples=50, random_state=42)
            .with_wavelengths(wavelengths=custom_wl)
        )

        assert np.allclose(builder.state.custom_wavelengths, custom_wl)

    def test_with_wavelengths_instrument_grid(self):
        """Test with_wavelengths with instrument_grid parameter."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        builder = (
            SyntheticDatasetBuilder(n_samples=50, random_state=42)
            .with_wavelengths(instrument_grid="micronir_onsite")
        )

        assert builder.state.instrument_wavelength_grid == "micronir_onsite"
        assert builder.state.custom_wavelengths is None

    def test_with_wavelengths_both_raises(self):
        """Test that specifying both wavelengths and instrument_grid raises."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        custom_wl = np.linspace(1000, 2000, 100)

        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)

        with pytest.raises(ValueError, match="Cannot specify both"):
            builder.with_wavelengths(
                wavelengths=custom_wl,
                instrument_grid="micronir_onsite",
            )

    def test_with_wavelengths_unknown_instrument(self):
        """Test that unknown instrument grid raises error."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)

        with pytest.raises(ValueError, match="Unknown instrument"):
            builder.with_wavelengths(instrument_grid="nonexistent")

    def test_build_with_custom_wavelengths(self):
        """Test building dataset with custom wavelengths."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        custom_wl = np.linspace(1000, 2000, 50)

        X, y = (
            SyntheticDatasetBuilder(n_samples=20, random_state=42)
            .with_wavelengths(wavelengths=custom_wl)
            .with_features(complexity="simple")
            .build_arrays()
        )

        assert X.shape == (20, 50)

    def test_build_with_instrument_grid(self):
        """Test building dataset with instrument wavelength grid."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        X, y = (
            SyntheticDatasetBuilder(n_samples=20, random_state=42)
            .with_wavelengths(instrument_grid="micronir_onsite")
            .with_features(complexity="simple")
            .build_arrays()
        )

        # MicroNIR has 125 wavelengths
        assert X.shape == (20, 125)

    def test_chaining_with_wavelengths(self):
        """Test that with_wavelengths can be chained with other methods."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        custom_wl = np.linspace(1000, 2000, 50)

        dataset = (
            SyntheticDatasetBuilder(n_samples=30, random_state=42)
            .with_wavelengths(wavelengths=custom_wl)
            .with_features(complexity="simple")
            .with_targets(range=(0, 100))
            .with_partitions(train_ratio=0.8)
            .build()
        )

        # Should build successfully
        assert hasattr(dataset, "x")


class TestWavelengthInterpolation:
    """Tests for wavelength interpolation behavior."""

    def test_nonuniform_wavelength_grid(self):
        """Test generation with non-uniform wavelength grid."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        # Create non-uniform grid (like some real instruments)
        wl1 = np.linspace(1000, 1500, 30)
        wl2 = np.linspace(1500, 2000, 50)  # Higher resolution in 1500-2000
        custom_wl = np.concatenate([wl1[:-1], wl2])

        gen = SyntheticNIRSGenerator(
            wavelengths=custom_wl,
            complexity="simple",
            random_state=42,
        )

        X, _, _ = gen.generate(n_samples=5)

        assert X.shape[1] == len(custom_wl)

    def test_sparse_wavelength_grid(self):
        """Test generation with very sparse wavelength grid."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        # Very sparse grid (like some handheld devices)
        custom_wl = np.linspace(900, 1700, 20)

        gen = SyntheticNIRSGenerator(
            wavelengths=custom_wl,
            complexity="simple",
            random_state=42,
        )

        X, _, _ = gen.generate(n_samples=5)

        assert X.shape[1] == 20
        # Should still produce valid spectra
        assert not np.any(np.isnan(X))

    def test_high_resolution_wavelength_grid(self):
        """Test generation with high resolution wavelength grid."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        # High resolution grid (0.5 nm step)
        custom_wl = np.arange(1000, 2000, 0.5)

        gen = SyntheticNIRSGenerator(
            wavelengths=custom_wl,
            complexity="simple",
            random_state=42,
        )

        X, _, _ = gen.generate(n_samples=3)

        assert X.shape[1] == len(custom_wl)
        assert not np.any(np.isnan(X))
