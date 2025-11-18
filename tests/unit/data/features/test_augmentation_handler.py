"""
Step 6 Tests: Augmentation with Header Units
Test that header_unit is properly preserved during sample augmentation.
"""
import numpy as np
import pytest
from nirs4all.data.dataset import SpectroDataset


class TestAugmentationHeaderUnit:
    """Test that augmentation preserves header units"""

    def test_augment_preserves_header_unit(self):
        """Test that augmentation doesn't change header_unit"""
        dataset = SpectroDataset(name="test")

        # Add initial samples with nm unit
        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        headers = ['780', '850', '1000']
        dataset.add_samples(x_data, {"partition": "train"}, headers=headers, header_unit="nm")

        # Verify unit before augmentation
        assert dataset.header_unit(0) == "nm"

        # Augment samples
        augmented_data = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        dataset.augment_samples(
            augmented_data,
            processings=["augmented"],
            augmentation_id="aug1",
            selector={"partition": "train"}
        )

        # Verify unit after augmentation
        assert dataset.header_unit(0) == "nm"

        # Verify headers unchanged
        assert dataset.headers(0) == headers

    def test_augment_with_cm1_unit(self):
        """Test augmentation with cm-1 unit (default)"""
        dataset = SpectroDataset(name="test")

        # Add initial samples with cm-1 unit (default)
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        headers = ['1000', '1100']
        dataset.add_samples(x_data, {"partition": "train"}, headers=headers, header_unit="cm-1")

        assert dataset.header_unit(0) == "cm-1"

        # Augment
        augmented_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        dataset.augment_samples(
            augmented_data,
            processings=["augmented"],
            augmentation_id="aug1"
        )

        assert dataset.header_unit(0) == "cm-1"
        assert dataset.headers(0) == headers

    def test_augment_multi_source_preserves_units(self):
        """Test that multi-source augmentation preserves per-source units"""
        dataset = SpectroDataset(name="test")

        # Add multi-source data with different units
        x1_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        x2_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        headers1 = ['1000', '1100']
        headers2 = ['780', '850']

        dataset.add_samples(
            [x1_data, x2_data],
            {"partition": "train"},
            headers=[headers1, headers2],
            header_unit=["cm-1", "nm"]
        )

        # Verify units before augmentation
        assert dataset.header_unit(0) == "cm-1"
        assert dataset.header_unit(1) == "nm"

        # Augment both sources
        aug1_data = np.array([[9.0, 10.0], [11.0, 12.0]])
        aug2_data = np.array([[13.0, 14.0], [15.0, 16.0]])

        dataset.augment_samples(
            [aug1_data, aug2_data],
            processings=["augmented"],
            augmentation_id="aug1"
        )

        # Verify units after augmentation
        assert dataset.header_unit(0) == "cm-1"
        assert dataset.header_unit(1) == "nm"

        # Verify headers unchanged
        assert dataset.headers(0) == headers1
        assert dataset.headers(1) == headers2

    def test_augment_with_none_unit(self):
        """Test augmentation with none unit (headerless data)"""
        dataset = SpectroDataset(name="test")

        # Add samples with none unit
        x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        headers = ['f0', 'f1', 'f2']
        dataset.add_samples(x_data, {"partition": "train"}, headers=headers, header_unit="none")

        assert dataset.header_unit(0) == "none"

        # Augment
        augmented_data = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        dataset.augment_samples(
            augmented_data,
            processings=["augmented"],
            augmentation_id="aug1"
        )

        assert dataset.header_unit(0) == "none"

    def test_wavelength_conversion_after_augmentation(self):
        """Test that wavelength conversion still works after augmentation"""
        dataset = SpectroDataset(name="test")

        # Add samples with nm unit
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        headers = ['780', '850']
        dataset.add_samples(x_data, {"partition": "train"}, headers=headers, header_unit="nm")

        # Augment
        augmented_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        dataset.augment_samples(
            augmented_data,
            processings=["augmented"],
            augmentation_id="aug1"
        )

        # Test wavelength conversion still works
        wavelengths_nm = dataset.wavelengths_nm(0)
        wavelengths_cm1 = dataset.wavelengths_cm1(0)

        assert len(wavelengths_nm) == 2
        assert len(wavelengths_cm1) == 2
        assert wavelengths_nm[0] == 780
        assert wavelengths_nm[1] == 850

        # Verify conversion accuracy
        expected_cm1_0 = 10_000_000 / 780
        expected_cm1_1 = 10_000_000 / 850
        assert abs(wavelengths_cm1[0] - expected_cm1_0) < 0.01
        assert abs(wavelengths_cm1[1] - expected_cm1_1) < 0.01

    def test_augment_multiple_times_preserves_unit(self):
        """Test that multiple augmentations preserve header_unit"""
        dataset = SpectroDataset(name="test")

        # Add initial samples
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        headers = ['780', '850']
        dataset.add_samples(x_data, {"partition": "train"}, headers=headers, header_unit="nm")

        assert dataset.header_unit(0) == "nm"

        # First augmentation
        aug1_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        dataset.augment_samples(
            aug1_data,
            processings=["aug1"],
            augmentation_id="augmentation_1"
        )
        assert dataset.header_unit(0) == "nm"

        # Second augmentation
        aug2_data = np.array([[9.0, 10.0], [11.0, 12.0]])
        dataset.augment_samples(
            aug2_data,
            processings=["aug2"],
            augmentation_id="augmentation_2"
        )
        assert dataset.header_unit(0) == "nm"

        # Verify headers still correct
        assert dataset.headers(0) == headers
