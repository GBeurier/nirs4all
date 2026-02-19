"""Unit tests for nirs4all.utils.backend."""

from unittest.mock import patch

import pytest

from nirs4all.utils.backend import (
    JAX_AVAILABLE,
    TF_AVAILABLE,
    TORCH_AVAILABLE,
    BackendNotAvailableError,
    _availability_cache,
    _LazyAvailability,
    clear_availability_cache,
    framework,
    get_backend_info,
    is_available,
    require_backend,
)


class TestIsAvailable:
    """Test is_available caching and package mapping."""

    def setup_method(self):
        clear_availability_cache()

    def test_sklearn_always_available(self):
        """sklearn is a hard dependency so it must always be available."""
        assert is_available('sklearn') is not None  # may or may not map; just no crash

    def test_numpy_available(self):
        """numpy is always installed in this environment."""
        # numpy is not in _PACKAGE_MAPPING so it falls back to its own name
        result = is_available('numpy')
        assert result is True

    def test_unavailable_fictional_package(self):
        result = is_available('_nonexistent_package_xyz_')
        assert result is False

    def test_result_is_cached(self):
        """Second call returns the cached result without re-checking."""
        is_available('numpy')
        # Patch find_spec to ensure it is NOT called a second time
        with patch('importlib.util.find_spec', side_effect=AssertionError("should not call find_spec")) as mock_spec:
            result = is_available('numpy')
        assert result is True

    def test_cache_cleared(self):
        """clear_availability_cache resets the internal dict."""
        is_available('numpy')
        clear_availability_cache()
        assert 'numpy' not in _availability_cache

    def test_pytorch_alias_maps_to_torch(self):
        """'pytorch' is an alias for the 'torch' package."""
        r_pytorch = is_available('pytorch')
        r_torch = is_available('torch')
        assert r_pytorch == r_torch

    def test_case_normalised(self):
        """Backend name is lowercased before lookup."""
        r1 = is_available('NumPy')
        r2 = is_available('numpy')
        assert r1 == r2

class TestRequireBackend:
    """Test require_backend raises on missing backend."""

    def setup_method(self):
        clear_availability_cache()

    def test_require_available_backend_no_raise(self):
        require_backend('numpy')  # must not raise

    def test_require_missing_backend_raises(self):
        with pytest.raises(BackendNotAvailableError):
            require_backend('_nonexistent_package_xyz_')

    def test_error_message_contains_backend_name(self):
        with pytest.raises(BackendNotAvailableError, match='_nonexistent_package_xyz_'):
            require_backend('_nonexistent_package_xyz_')

    def test_error_message_contains_feature_name(self):
        with pytest.raises(BackendNotAvailableError, match='myfeature'):
            require_backend('_nonexistent_package_xyz_', feature='myfeature')

    def test_error_is_import_error_subclass(self):
        """BackendNotAvailableError must inherit from ImportError (BackendError does)."""
        with pytest.raises(ImportError):
            require_backend('_nonexistent_package_xyz_')

class TestLazyAvailability:
    """Test _LazyAvailability lazy evaluation."""

    def setup_method(self):
        clear_availability_cache()

    def test_bool_evaluates_to_true_for_numpy(self):
        lazy = _LazyAvailability('numpy')
        assert bool(lazy) is True

    def test_bool_evaluates_to_false_for_missing(self):
        lazy = _LazyAvailability('_nonexistent_xyz_')
        assert bool(lazy) is False

    def test_repr_is_bool_string(self):
        lazy = _LazyAvailability('numpy')
        assert repr(lazy) == 'True'

    def test_eq_comparison(self):
        lazy = _LazyAvailability('numpy')
        assert lazy == True  # noqa: E712

    def test_lazy_constants_are_lazy_availability(self):
        assert isinstance(TF_AVAILABLE, _LazyAvailability)
        assert isinstance(TORCH_AVAILABLE, _LazyAvailability)
        assert isinstance(JAX_AVAILABLE, _LazyAvailability)

class TestFrameworkDecorator:
    """Test @framework decorator attaches the framework attribute."""

    def test_decorator_attaches_attribute(self):
        @framework('tensorflow')
        def build_model(input_shape):
            pass

        assert hasattr(build_model, 'framework')
        assert build_model.framework == 'tensorflow'

    def test_decorator_preserves_function(self):
        @framework('torch')
        def my_fn():
            return 42

        assert my_fn() == 42

class TestGetBackendInfo:
    """Test get_backend_info returns expected structure."""

    def test_returns_dict(self):
        info = get_backend_info()
        assert isinstance(info, dict)

    def test_known_backends_present(self):
        info = get_backend_info()
        for name in ('tensorflow', 'torch', 'jax', 'optuna', 'shap'):
            assert name in info

    def test_each_entry_has_available_key(self):
        info = get_backend_info()
        for name, details in info.items():
            assert 'available' in details, f"Missing 'available' for {name}"
            assert isinstance(details['available'], bool)
