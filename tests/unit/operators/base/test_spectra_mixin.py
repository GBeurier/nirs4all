"""
Unit tests for SpectraTransformerMixin base class.
"""

import numpy as np
import pytest

from nirs4all.operators.base import SpectraTransformerMixin


class MockSpectraTransformer(SpectraTransformerMixin):
    """Mock transformer that doubles the spectra."""

    def _transform_impl(self, X, wavelengths):
        return X * 2

class MockOptionalWavelengthsTransformer(SpectraTransformerMixin):
    """Mock transformer that doesn't require wavelengths."""

    _requires_wavelengths = "optional"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def _transform_impl(self, X, wavelengths):
        if wavelengths is not None:
            # Use wavelengths if provided
            return X * self.scale * (wavelengths.mean() / 1000)
        return X * self.scale

class MockDisabledWavelengthsTransformer(SpectraTransformerMixin):
    """Mock transformer that ignores wavelengths."""

    _requires_wavelengths = False

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def _transform_impl(self, X, wavelengths):
        return X * self.scale

class MockWavelengthDependentTransformer(SpectraTransformerMixin):
    """Mock transformer that uses wavelengths to modify spectra."""

    def __init__(self, region_start: float = 1400, region_end: float = 1600):
        self.region_start = region_start
        self.region_end = region_end

    def _transform_impl(self, X, wavelengths):
        X_out = X.copy()
        mask = (wavelengths >= self.region_start) & (wavelengths <= self.region_end)
        X_out[:, mask] = X_out[:, mask] * 1.5
        return X_out

class TestSpectraTransformerMixinAbstract:
    """Tests for abstract method behavior."""

    def test_transform_impl_is_abstract(self):
        """Test that _transform_impl raises NotImplementedError when not overridden."""

        class IncompleteTransformer(SpectraTransformerMixin):
            pass

        transformer = IncompleteTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        with pytest.raises(NotImplementedError, match="must implement _transform_impl"):
            transformer.transform(X, wavelengths=wavelengths)

    def test_abstract_method_exists(self):
        """Test that the abstract method exists."""
        assert hasattr(SpectraTransformerMixin, "_transform_impl")

class TestSpectraTransformerMixinFit:
    """Tests for fit method."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)
        result = transformer.fit(X, wavelengths=wavelengths)
        assert result is transformer

    def test_fit_with_y(self):
        """Test that fit works with y parameter."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        y = np.random.rand(10)
        wavelengths = np.linspace(1100, 2500, 100)
        result = transformer.fit(X, y, wavelengths=wavelengths)
        assert result is transformer

    def test_fit_with_wavelengths(self):
        """Test that fit accepts wavelengths keyword argument and caches them."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)
        result = transformer.fit(X, wavelengths=wavelengths)
        assert result is transformer
        np.testing.assert_array_equal(transformer._wavelengths, wavelengths)

    def test_fit_caches_wavelengths_for_transform(self):
        """Test that wavelengths cached in fit are used in transform."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        transformer.fit(X, wavelengths=wavelengths)
        # transform without passing wavelengths -- should use cached
        X_transformed = transformer.transform(X)
        np.testing.assert_array_almost_equal(X_transformed, X * 2)

    def test_fit_validates_wavelength_length(self):
        """Test that fit validates wavelength length against features."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 50)  # Wrong length

        with pytest.raises(ValueError, match="wavelengths length"):
            transformer.fit(X, wavelengths=wavelengths)

class TestSpectraTransformerMixinTransform:
    """Tests for transform method."""

    def test_transform_requires_wavelengths_by_default(self):
        """Test that transform raises error when wavelengths not provided."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)

        with pytest.raises(ValueError, match="requires wavelengths"):
            transformer.transform(X)

    def test_transform_with_wavelengths(self):
        """Test that transform works when wavelengths are provided."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        X_transformed = transformer.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X_transformed, X * 2)

    def test_transform_output_shape(self):
        """Test that transform preserves input shape."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(50, 200)
        wavelengths = np.linspace(900, 2500, 200)

        X_transformed = transformer.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_transform_error_message_includes_class_name(self):
        """Test that error message includes the class name."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)

        with pytest.raises(ValueError) as exc_info:
            transformer.transform(X)

        assert "MockSpectraTransformer" in str(exc_info.value)

    def test_transform_validates_wavelength_length(self):
        """Test that transform validates wavelength length against features."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 50)  # Wrong length

        with pytest.raises(ValueError, match="wavelengths length"):
            transformer.transform(X, wavelengths=wavelengths)

class TestSpectraTransformerMixinOptionalWavelengths:
    """Tests for operators with optional wavelengths."""

    def test_optional_wavelengths_without_providing(self):
        """Test that optional wavelength operators work without wavelengths."""
        transformer = MockOptionalWavelengthsTransformer(scale=3.0)
        X = np.random.rand(10, 100)

        X_transformed = transformer.transform(X)

        np.testing.assert_array_almost_equal(X_transformed, X * 3.0)

    def test_optional_wavelengths_with_providing(self):
        """Test that optional wavelength operators can use wavelengths."""
        transformer = MockOptionalWavelengthsTransformer(scale=2.0)
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1000, 2000, 100)  # mean = 1500

        X_transformed = transformer.transform(X, wavelengths=wavelengths)

        expected = X * 2.0 * 1.5  # scale * (mean / 1000)
        np.testing.assert_array_almost_equal(X_transformed, expected)

    def test_requires_wavelengths_flag_optional(self):
        """Test that _requires_wavelengths can be set to 'optional'."""
        transformer = MockOptionalWavelengthsTransformer()
        assert transformer._requires_wavelengths == "optional"

    def test_requires_wavelengths_flag_false(self):
        """Test that _requires_wavelengths can be set to False."""
        transformer = MockDisabledWavelengthsTransformer()
        assert transformer._requires_wavelengths is False

    def test_disabled_wavelengths_without_providing(self):
        """Test that disabled wavelength operators work without wavelengths."""
        transformer = MockDisabledWavelengthsTransformer(scale=2.0)
        X = np.random.rand(10, 100)

        X_transformed = transformer.transform(X)

        np.testing.assert_array_almost_equal(X_transformed, X * 2.0)

class TestSpectraTransformerMixinWavelengthDependent:
    """Tests for wavelength-dependent transformations."""

    def test_wavelength_region_modification(self):
        """Test transformer that modifies specific wavelength regions."""
        transformer = MockWavelengthDependentTransformer(
            region_start=1400, region_end=1600
        )
        X = np.ones((5, 100))
        wavelengths = np.linspace(1100, 2000, 100)

        X_transformed = transformer.transform(X, wavelengths=wavelengths)

        # Check that region was modified
        mask = (wavelengths >= 1400) & (wavelengths <= 1600)
        np.testing.assert_array_almost_equal(X_transformed[:, mask], 1.5)
        np.testing.assert_array_almost_equal(X_transformed[:, ~mask], 1.0)

    def test_wavelength_array_length_validation(self):
        """Test that wavelength array length must match feature count."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 50)  # Wrong length

        with pytest.raises(ValueError, match="wavelengths length"):
            transformer.transform(X, wavelengths=wavelengths)

class TestSpectraTransformerMixinSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_fit_transform(self):
        """Test fit_transform workflow."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        # fit_transform should work
        X_transformed = transformer.fit(X, wavelengths=wavelengths).transform(X, wavelengths=wavelengths)
        np.testing.assert_array_almost_equal(X_transformed, X * 2)

    def test_fit_then_transform_without_wavelengths(self):
        """Test that fit caches wavelengths for subsequent transform calls."""
        transformer = MockSpectraTransformer()
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(1100, 2500, 100)

        transformer.fit(X, wavelengths=wavelengths)
        X_transformed = transformer.transform(X)  # No wavelengths -- uses cached
        np.testing.assert_array_almost_equal(X_transformed, X * 2)

    def test_more_tags(self):
        """Test that _more_tags returns expected sklearn tags."""
        transformer = MockSpectraTransformer()
        tags = transformer._more_tags()

        assert "allow_nan" in tags
        assert tags["allow_nan"] is False
        assert "requires_wavelengths" in tags
        assert tags["requires_wavelengths"] is True

    def test_more_tags_optional_wavelengths(self):
        """Test _more_tags for operators with optional wavelengths."""
        transformer = MockOptionalWavelengthsTransformer()
        tags = transformer._more_tags()

        assert tags["requires_wavelengths"] == "optional"

    def test_more_tags_disabled_wavelengths(self):
        """Test _more_tags for operators with disabled wavelengths."""
        transformer = MockDisabledWavelengthsTransformer()
        tags = transformer._more_tags()

        assert tags["requires_wavelengths"] is False

    def test_get_params(self):
        """Test sklearn get_params works."""
        transformer = MockOptionalWavelengthsTransformer(scale=5.0)
        params = transformer.get_params()

        assert "scale" in params
        assert params["scale"] == 5.0

    def test_set_params(self):
        """Test sklearn set_params works."""
        transformer = MockOptionalWavelengthsTransformer(scale=1.0)
        transformer.set_params(scale=10.0)

        assert transformer.scale == 10.0

    def test_clone(self):
        """Test sklearn clone works."""
        from sklearn.base import clone

        transformer = MockOptionalWavelengthsTransformer(scale=7.0)
        cloned = clone(transformer)

        assert cloned.scale == 7.0
        assert cloned is not transformer

class TestSpectraTransformerMixinImport:
    """Tests for import paths."""

    def test_import_from_operators(self):
        """Test that SpectraTransformerMixin can be imported from operators."""
        from nirs4all.operators import SpectraTransformerMixin as STM

        assert STM is SpectraTransformerMixin

    def test_import_from_operators_base(self):
        """Test that SpectraTransformerMixin can be imported from operators.base."""
        from nirs4all.operators.base import SpectraTransformerMixin as STM

        assert STM is SpectraTransformerMixin

    def test_in_operators_all(self):
        """Test that SpectraTransformerMixin is in operators.__all__."""
        import nirs4all.operators as ops

        assert "SpectraTransformerMixin" in ops.__all__
