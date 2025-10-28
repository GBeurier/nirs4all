"""
Test wavelength conversion methods in Dataset - Step 2
"""

import pytest
import numpy as np
from nirs4all.data.dataset import SpectroDataset


class TestDatasetWavelengthConversion:
    """Test wavelength conversion methods in Dataset"""

    def test_header_unit_method(self):
        """Test that header_unit method returns correct unit"""
        dataset = SpectroDataset(name="test")

        # Add samples with cm-1 headers
        samples = np.random.rand(10, 3)
        headers = ["4000.0", "5000.0", "6000.0"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="cm-1")

        assert dataset.header_unit(0) == "cm-1"

    def test_wavelengths_cm1_no_conversion(self):
        """Test getting cm-1 wavelengths when already in cm-1"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["4000.0", "5000.0", "6000.0"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="cm-1")

        wavelengths = dataset.wavelengths_cm1(0)

        np.testing.assert_array_equal(wavelengths, [4000.0, 5000.0, 6000.0])

    def test_wavelengths_cm1_from_nm(self):
        """Test converting nm to cm-1"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["1000.0", "2000.0", "2500.0"]  # nm
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="nm")

        wavelengths = dataset.wavelengths_cm1(0)

        # 1000 nm = 10000 cm-1, 2000 nm = 5000 cm-1, 2500 nm = 4000 cm-1
        expected = np.array([10000.0, 5000.0, 4000.0])
        np.testing.assert_array_almost_equal(wavelengths, expected, decimal=1)

    def test_wavelengths_nm_no_conversion(self):
        """Test getting nm wavelengths when already in nm"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["780.0", "1000.0", "2500.0"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="nm")

        wavelengths = dataset.wavelengths_nm(0)

        np.testing.assert_array_equal(wavelengths, [780.0, 1000.0, 2500.0])

    def test_wavelengths_nm_from_cm1(self):
        """Test converting cm-1 to nm"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["12820.51", "10000.0", "4000.0"]  # cm-1
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="cm-1")

        wavelengths = dataset.wavelengths_nm(0)

        # 12820.51 cm-1 ≈ 780 nm, 10000 cm-1 = 1000 nm, 4000 cm-1 = 2500 nm
        expected = np.array([780.0, 1000.0, 2500.0])
        np.testing.assert_array_almost_equal(wavelengths, expected, decimal=1)

    def test_conversion_accuracy(self):
        """Test conversion math accuracy for known values"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 1)

        # Test 780 nm <-> 12820.51 cm-1
        dataset.add_samples(samples, headers=["780.0"])
        dataset._features.sources[0].set_headers(["780.0"], unit="nm")

        cm1 = dataset.wavelengths_cm1(0)
        assert abs(cm1[0] - 12820.51282) < 0.01

        # Test round-trip conversion
        dataset._features.sources[0].set_headers([f"{cm1[0]}"], unit="cm-1")
        nm_back = dataset.wavelengths_nm(0)
        assert abs(nm_back[0] - 780.0) < 0.01

    def test_wavelengths_cm1_with_none_unit(self):
        """Test wavelengths_cm1 with 'none' unit returns indices"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 5)
        headers = ["0", "1", "2", "3", "4"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="none")

        wavelengths = dataset.wavelengths_cm1(0)

        np.testing.assert_array_equal(wavelengths, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_wavelengths_nm_with_index_unit(self):
        """Test wavelengths_nm with 'index' unit returns indices"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["0", "1", "2"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="index")

        wavelengths = dataset.wavelengths_nm(0)

        np.testing.assert_array_equal(wavelengths, [0.0, 1.0, 2.0])

    def test_wavelengths_cm1_with_text_unit_raises_error(self):
        """Test that text unit raises ValueError"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["feature_A", "feature_B", "feature_C"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="text")

        with pytest.raises(ValueError, match="Cannot convert unit 'text'"):
            dataset.wavelengths_cm1(0)

    def test_wavelengths_nm_with_invalid_unit_raises_error(self):
        """Test that invalid unit raises ValueError"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["1", "2", "3"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="invalid")

        with pytest.raises(ValueError, match="Cannot convert unit 'invalid'"):
            dataset.wavelengths_nm(0)

    def test_multi_source_different_units(self):
        """Test that different sources can have different units"""
        dataset = SpectroDataset(name="test")

        # Source 0: cm-1
        samples1 = np.random.rand(10, 3)
        headers1 = ["4000.0", "5000.0", "6000.0"]
        # Add as multi-source list
        dataset.add_samples([samples1], headers=[headers1])
        dataset._features.sources[0].set_headers(headers1, unit="cm-1")

        # Source 1: nm - manually add second source
        from nirs4all.data.feature_source import FeatureSource
        source1 = FeatureSource()
        samples2 = np.random.rand(10, 2)
        headers2 = ["780.0", "1000.0"]
        source1.add_samples(samples2, headers=headers2)
        source1.set_headers(headers2, unit="nm")
        dataset._features.sources.append(source1)
        dataset._features.preprocessing_str.append(["raw"])

        # Check units
        assert dataset.header_unit(0) == "cm-1"
        assert dataset.header_unit(1) == "nm"

        # Check conversions work independently
        wl_cm1_src0 = dataset.wavelengths_cm1(0)
        np.testing.assert_array_equal(wl_cm1_src0, [4000.0, 5000.0, 6000.0])

        wl_cm1_src1 = dataset.wavelengths_cm1(1)
        # 780 nm = 12820.51 cm-1, 1000 nm = 10000 cm-1
        expected = np.array([12820.51, 10000.0])
        np.testing.assert_array_almost_equal(wl_cm1_src1, expected, decimal=1)

    def test_float_headers_legacy_method(self):
        """Test that float_headers still works (legacy compatibility)"""
        dataset = SpectroDataset(name="test")

        samples = np.random.rand(10, 3)
        headers = ["1000.5", "2000.3", "3000.7"]
        dataset.add_samples(samples, headers=headers)
        dataset._features.sources[0].set_headers(headers, unit="cm-1")

        float_hdrs = dataset.float_headers(0)

        np.testing.assert_array_equal(float_hdrs, [1000.5, 2000.3, 3000.7])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
