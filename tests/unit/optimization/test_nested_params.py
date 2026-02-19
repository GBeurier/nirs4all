"""Unit tests for nested parameter flatten/unflatten (Phase 5 - ISSUE-16a)."""

import optuna
import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestIsParamSpec:
    """Tests for _is_param_spec helper."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_dict_with_type_is_param_spec(self, manager):
        assert manager._is_param_spec({"type": "int", "min": 1, "max": 10}) is True

    def test_dict_with_min_max_is_param_spec(self, manager):
        assert manager._is_param_spec({"min": 1, "max": 10}) is True

    def test_dict_with_low_high_is_param_spec(self, manager):
        assert manager._is_param_spec({"low": 1, "high": 10}) is True

    def test_dict_with_choices_is_param_spec(self, manager):
        assert manager._is_param_spec({"choices": [1, 2, 3]}) is True

    def test_dict_with_values_is_param_spec(self, manager):
        assert manager._is_param_spec({"values": [1, 2, 3]}) is True

    def test_nested_group_is_not_param_spec(self, manager):
        """A dict containing sub-parameter specs is NOT itself a param spec."""
        assert manager._is_param_spec({"PARAM_A": [True, False], "PARAM_B": [1, 2]}) is False

    def test_empty_dict_is_not_param_spec(self, manager):
        assert manager._is_param_spec({}) is False

    def test_non_dict_is_not_param_spec(self, manager):
        assert manager._is_param_spec([1, 2, 3]) is False
        assert manager._is_param_spec(42) is False
        assert manager._is_param_spec("string") is False

class TestFlattenNestedParams:
    """Tests for _flatten_nested_params."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_flat_params_unchanged(self, manager):
        """Already flat params should remain unchanged."""
        params = {"n_components": [1, 5, 10], "alpha": ("float", 0.01, 1.0)}
        result = manager._flatten_nested_params(params)
        assert result == params

    def test_single_level_nesting(self, manager):
        """One level of nesting should produce __ separated keys."""
        params = {
            "inference_config": {
                "FINGERPRINT_FEATURE": [True, False],
                "OUTLIER_REMOVAL_STD": [None, 7.0, 12.0],
            },
        }
        result = manager._flatten_nested_params(params)
        assert "inference_config__FINGERPRINT_FEATURE" in result
        assert "inference_config__OUTLIER_REMOVAL_STD" in result
        assert result["inference_config__FINGERPRINT_FEATURE"] == [True, False]
        assert result["inference_config__OUTLIER_REMOVAL_STD"] == [None, 7.0, 12.0]

    def test_mixed_flat_and_nested(self, manager):
        """Flat params alongside nested groups."""
        params = {
            "softmax_temperature": ("float", 0.7, 1.1),
            "inference_config": {
                "FINGERPRINT_FEATURE": [True, False],
            },
        }
        result = manager._flatten_nested_params(params)
        assert "softmax_temperature" in result
        assert result["softmax_temperature"] == ("float", 0.7, 1.1)
        assert "inference_config__FINGERPRINT_FEATURE" in result

    def test_deep_nesting(self, manager):
        """Multiple levels of nesting."""
        params = {
            "config": {
                "sub_config": {
                    "param": [1, 2, 3],
                },
            },
        }
        result = manager._flatten_nested_params(params)
        assert "config__sub_config__param" in result
        assert result["config__sub_config__param"] == [1, 2, 3]

    def test_dict_param_spec_not_flattened(self, manager):
        """A dict that IS a param spec should not be recursed into."""
        params = {
            "n_components": {"type": "int", "min": 1, "max": 30},
        }
        result = manager._flatten_nested_params(params)
        assert "n_components" in result
        assert result["n_components"] == {"type": "int", "min": 1, "max": 30}

    def test_empty_nested_group(self, manager):
        """Empty nested group produces no keys."""
        params = {"config": {}}
        result = manager._flatten_nested_params(params)
        assert result == {}

class TestUnflattenParams:
    """Tests for _unflatten_params."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_flat_keys_unchanged(self, manager):
        """Keys without __ should remain flat."""
        flat = {"n_components": 10, "alpha": 0.5}
        result = manager._unflatten_params(flat)
        assert result == {"n_components": 10, "alpha": 0.5}

    def test_single_level_unflatten(self, manager):
        """__ separated keys should reconstruct nested dict."""
        flat = {
            "inference_config__FINGERPRINT_FEATURE": True,
            "inference_config__OUTLIER_REMOVAL_STD": 7.0,
        }
        result = manager._unflatten_params(flat)
        assert result == {
            "inference_config": {
                "FINGERPRINT_FEATURE": True,
                "OUTLIER_REMOVAL_STD": 7.0,
            },
        }

    def test_mixed_flat_and_nested(self, manager):
        """Mix of flat and nested keys."""
        flat = {
            "softmax_temperature": 0.9,
            "inference_config__FINGERPRINT_FEATURE": False,
        }
        result = manager._unflatten_params(flat)
        assert result["softmax_temperature"] == 0.9
        assert result["inference_config"]["FINGERPRINT_FEATURE"] is False

    def test_deep_unflatten(self, manager):
        """Multiple __ levels."""
        flat = {"a__b__c": 42}
        result = manager._unflatten_params(flat)
        assert result == {"a": {"b": {"c": 42}}}

    def test_conflicting_keys_raises(self, manager):
        """Scalar key conflicting with nested group should raise ValueError."""
        flat = {
            "config": 42,
            "config__param": 100,
        }
        with pytest.raises(ValueError, match="Conflicting nested parameter structure"):
            manager._unflatten_params(flat)

    def test_roundtrip(self, manager):
        """flatten -> unflatten should reconstruct original structure."""
        original = {
            "alpha": 0.5,
            "inference_config": {
                "FINGERPRINT_FEATURE": True,
                "OUTLIER_REMOVAL_STD": 7.0,
            },
        }
        flat = manager._flatten_nested_params(original)
        restored = manager._unflatten_params(flat)
        assert restored == original

class TestNestedParamsSampling:
    """Integration tests for nested params going through sample_hyperparameters."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def _make_trial(self):
        study = optuna.create_study()
        return study.ask()

    def test_nested_params_sampled_and_reconstructed(self, manager):
        """Nested model_params should be sampled and reconstructed."""
        fp = {
            "model_params": {
                "inference_config": {
                    "FINGERPRINT_FEATURE": [True, False],
                    "OUTLIER_REMOVAL_STD": [None, 7.0, 12.0],
                },
                "softmax_temperature": ("float", 0.7, 1.1),
            },
        }
        trial = self._make_trial()
        model_params, train_params = manager.sample_hyperparameters(trial, fp)

        # Should have nested structure
        assert "inference_config" in model_params
        assert isinstance(model_params["inference_config"], dict)
        assert "FINGERPRINT_FEATURE" in model_params["inference_config"]
        assert model_params["inference_config"]["FINGERPRINT_FEATURE"] in [True, False]
        assert model_params["inference_config"]["OUTLIER_REMOVAL_STD"] in [None, 7.0, 12.0]
        assert 0.7 <= model_params["softmax_temperature"] <= 1.1

    def test_sklearn_double_underscore_params_preserved(self, manager):
        """Params with __ for sklearn meta-estimators should work."""
        fp = {
            "model_params": {
                "final_estimator__alpha": ("float_log", 1e-4, 1e-1),
                "final_estimator__fit_intercept": [True, False],
            },
        }
        trial = self._make_trial()
        model_params, _ = manager.sample_hyperparameters(trial, fp)

        # These use __ as part of the key name (sklearn convention)
        # They should be unflattened into nested structure
        assert "final_estimator" in model_params
        assert "alpha" in model_params["final_estimator"]
        assert "fit_intercept" in model_params["final_estimator"]
