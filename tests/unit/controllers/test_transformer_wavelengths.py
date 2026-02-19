"""
Unit tests for TransformerMixinController wavelength passing functionality.

Tests that SpectraTransformerMixin operators receive wavelengths from the controller.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.operators.base import SpectraTransformerMixin


class MockSpectraTransformer(SpectraTransformerMixin):
    """Mock SpectraTransformerMixin that tracks if wavelengths were passed."""

    def __init__(self):
        self.fit_wavelengths = None
        self.transform_wavelengths = None
        self.fit_called = False
        self.transform_called = False

    def fit(self, X, y=None, **kwargs):
        self.fit_called = True
        self.fit_wavelengths = kwargs.get('wavelengths')
        super().fit(X, y, **kwargs)
        return self

    def _transform_impl(self, X, wavelengths):
        self.transform_called = True
        self.transform_wavelengths = wavelengths
        return X * 2

class MockOptionalWavelengthsTransformer(SpectraTransformerMixin):
    """Mock SpectraTransformerMixin that doesn't require wavelengths."""

    _requires_wavelengths = False

    def __init__(self):
        self.fit_wavelengths = None
        self.transform_wavelengths = None

    def fit(self, X, y=None, **kwargs):
        self.fit_wavelengths = kwargs.get('wavelengths')
        super().fit(X, y, **kwargs)
        return self

    def _transform_impl(self, X, wavelengths):
        self.transform_wavelengths = wavelengths
        return X

class MockStandardTransformer(TransformerMixin, BaseEstimator):
    """Mock standard sklearn TransformerMixin (no wavelengths)."""

    def __init__(self):
        self.fit_called = False
        self.transform_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        return self

    def transform(self, X):
        self.transform_called = True
        return X * 3

class TestNeedsWavelengths:
    """Tests for _needs_wavelengths static method."""

    def test_spectra_transformer_with_required_wavelengths(self):
        """Test that SpectraTransformerMixin with _requires_wavelengths=True returns True."""
        transformer = MockSpectraTransformer()
        assert TransformerMixinController._needs_wavelengths(transformer) is True

    def test_spectra_transformer_with_optional_wavelengths(self):
        """Test that SpectraTransformerMixin with _requires_wavelengths=False returns False."""
        transformer = MockOptionalWavelengthsTransformer()
        assert TransformerMixinController._needs_wavelengths(transformer) is False

    def test_spectra_transformer_with_optional_string(self):
        """Test that SpectraTransformerMixin with _requires_wavelengths='optional' returns truthy."""

        class OptionalTransformer(SpectraTransformerMixin):
            _requires_wavelengths = "optional"

            def _transform_impl(self, X, wavelengths):
                return X

        transformer = OptionalTransformer()
        assert TransformerMixinController._needs_wavelengths(transformer)

    def test_standard_transformer(self):
        """Test that standard TransformerMixin returns False."""
        transformer = MockStandardTransformer()
        assert TransformerMixinController._needs_wavelengths(transformer) is False

    def test_non_transformer_object(self):
        """Test that non-transformer objects return False."""
        assert TransformerMixinController._needs_wavelengths("not a transformer") is False
        assert TransformerMixinController._needs_wavelengths(None) is False
        assert TransformerMixinController._needs_wavelengths(42) is False

class TestExtractWavelengths:
    """Tests for _extract_wavelengths static method."""

    def test_extract_wavelengths_nm(self):
        """Test wavelength extraction using wavelengths_nm method."""
        mock_dataset = MagicMock()
        expected_wavelengths = np.linspace(1100, 2500, 100)
        mock_dataset.wavelengths_nm.return_value = expected_wavelengths

        result = TransformerMixinController._extract_wavelengths(
            mock_dataset, source_index=0, operator_name="TestOp"
        )

        np.testing.assert_array_equal(result, expected_wavelengths)
        mock_dataset.wavelengths_nm.assert_called_once_with(0)

    def test_extract_wavelengths_fallback_to_float_headers(self):
        """Test fallback to float_headers when wavelengths_nm fails."""
        mock_dataset = MagicMock()
        expected_wavelengths = np.linspace(1100, 2500, 100)
        mock_dataset.wavelengths_nm.side_effect = ValueError("No unit info")
        mock_dataset.float_headers.return_value = expected_wavelengths

        result = TransformerMixinController._extract_wavelengths(
            mock_dataset, source_index=0, operator_name="TestOp"
        )

        np.testing.assert_array_equal(result, expected_wavelengths)
        mock_dataset.float_headers.assert_called_once_with(0)

    def test_extract_wavelengths_raises_when_unavailable(self):
        """Test that ValueError is raised when no wavelengths available."""
        mock_dataset = MagicMock()
        mock_dataset.wavelengths_nm.side_effect = ValueError("No unit info")
        mock_dataset.float_headers.return_value = None

        with pytest.raises(ValueError, match="requires wavelengths but dataset has no"):
            TransformerMixinController._extract_wavelengths(
                mock_dataset, source_index=0, operator_name="TestOp"
            )

    def test_extract_wavelengths_multi_source(self):
        """Test wavelength extraction for multi-source dataset."""
        mock_dataset = MagicMock()
        wavelengths_source_0 = np.linspace(1100, 2500, 100)
        wavelengths_source_1 = np.linspace(900, 1700, 80)

        def wavelengths_nm_side_effect(source_index):
            if source_index == 0:
                return wavelengths_source_0
            elif source_index == 1:
                return wavelengths_source_1
            raise ValueError(f"Unknown source {source_index}")

        mock_dataset.wavelengths_nm.side_effect = wavelengths_nm_side_effect

        result_0 = TransformerMixinController._extract_wavelengths(
            mock_dataset, source_index=0, operator_name="TestOp"
        )
        result_1 = TransformerMixinController._extract_wavelengths(
            mock_dataset, source_index=1, operator_name="TestOp"
        )

        np.testing.assert_array_equal(result_0, wavelengths_source_0)
        np.testing.assert_array_equal(result_1, wavelengths_source_1)

    def test_error_message_includes_source_index(self):
        """Test that error message includes source index."""
        mock_dataset = MagicMock()
        mock_dataset.wavelengths_nm.side_effect = ValueError("No unit info")
        mock_dataset.float_headers.return_value = None

        with pytest.raises(ValueError) as exc_info:
            TransformerMixinController._extract_wavelengths(
                mock_dataset, source_index=2, operator_name="MyOperator"
            )

        assert "source 2" in str(exc_info.value)
        assert "MyOperator" in str(exc_info.value)

class TestControllerMatches:
    """Tests for controller matches method."""

    def test_matches_spectra_transformer_mixin(self):
        """Test that controller matches SpectraTransformerMixin operators."""
        transformer = MockSpectraTransformer()
        assert TransformerMixinController.matches(transformer, transformer, "") is True

    def test_matches_standard_transformer_mixin(self):
        """Test that controller matches standard TransformerMixin operators."""
        transformer = MockStandardTransformer()
        assert TransformerMixinController.matches(transformer, transformer, "") is True

    def test_matches_dict_with_model_key(self):
        """Test that controller matches dict with model key."""
        transformer = MockSpectraTransformer()
        step = {"model": transformer}
        assert TransformerMixinController.matches(step, transformer, "model") is True

class TestControllerWavelengthLogic:
    """Tests for wavelength passing logic through controller.

    These tests verify the key behaviors without requiring full controller execution.
    Full end-to-end tests are in integration tests.
    """

    def test_needs_wavelengths_is_checked_correctly(self):
        """Verify the controller correctly identifies operators that need wavelengths."""
        controller = TransformerMixinController()

        # Required wavelengths
        transformer_required = MockSpectraTransformer()
        assert controller._needs_wavelengths(transformer_required) is True

        # Optional wavelengths
        transformer_optional = MockOptionalWavelengthsTransformer()
        assert controller._needs_wavelengths(transformer_optional) is False

        # Standard transformer
        transformer_standard = MockStandardTransformer()
        assert controller._needs_wavelengths(transformer_standard) is False

    def test_extract_wavelengths_is_called_correctly(self):
        """Verify wavelength extraction works with mock dataset."""
        mock_dataset = MagicMock()
        expected_wavelengths = np.linspace(1100, 2500, 100)
        mock_dataset.wavelengths_nm.return_value = expected_wavelengths

        result = TransformerMixinController._extract_wavelengths(
            mock_dataset, source_index=0, operator_name="TestOp"
        )

        np.testing.assert_array_equal(result, expected_wavelengths)
        mock_dataset.wavelengths_nm.assert_called_once_with(0)

    def test_fit_with_wavelengths_signature(self):
        """Test that fit can accept wavelengths keyword argument."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        # Fit should work with wavelengths kwarg
        transformer.fit(X, wavelengths=wavelengths)

        assert transformer.fit_called is True
        np.testing.assert_array_equal(transformer.fit_wavelengths, wavelengths)

    def test_transform_with_wavelengths_signature(self):
        """Test that transform accepts wavelengths parameter."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        # Transform should work with wavelengths kwarg
        result = transformer.transform(X, wavelengths=wavelengths)

        assert transformer.transform_called is True
        np.testing.assert_array_equal(transformer.transform_wavelengths, wavelengths)
        assert result.shape == X.shape

    def test_optional_wavelengths_transformer_works_without_wavelengths(self):
        """Test that optional wavelength transformer works without wavelengths."""
        transformer = MockOptionalWavelengthsTransformer()
        X = np.random.rand(10, 100)

        # Should not raise even without wavelengths
        transformer.fit(X)
        result = transformer.transform(X)

        assert result.shape == X.shape
        assert transformer.fit_wavelengths is None

    def test_required_wavelengths_transformer_raises_without_wavelengths(self):
        """Test that required wavelength transformer raises without wavelengths."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)

        # Transform should raise without wavelengths
        with pytest.raises(ValueError, match="requires wavelengths"):
            transformer.transform(X)

class TestControllerBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    def test_sklearn_standard_scaler_still_works(self):
        """Test that sklearn StandardScaler still works without wavelengths."""
        from sklearn.preprocessing import StandardScaler

        controller = TransformerMixinController()

        # StandardScaler should not need wavelengths
        scaler = StandardScaler()
        assert controller._needs_wavelengths(scaler) is False

    def test_sklearn_pca_still_works(self):
        """Test that sklearn PCA still works without wavelengths."""
        from sklearn.decomposition import PCA

        controller = TransformerMixinController()

        # PCA should not need wavelengths
        pca = PCA(n_components=5)
        assert controller._needs_wavelengths(pca) is False

    def test_minmax_scaler_still_works(self):
        """Test that sklearn MinMaxScaler still works without wavelengths."""
        from sklearn.preprocessing import MinMaxScaler

        controller = TransformerMixinController()

        # MinMaxScaler should not need wavelengths
        scaler = MinMaxScaler()
        assert controller._needs_wavelengths(scaler) is False
