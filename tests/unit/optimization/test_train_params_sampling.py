"""Unit tests for train_params sampling in OptunaManager (Phase 3 - ISSUE-4)."""

import pytest
import optuna

from nirs4all.optimization.optuna import OptunaManager


class TestTrainParamsSampling:
    """Tests for sample_hyperparameters returning (model_params, train_params) tuple."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def _make_trial(self, finetune_params):
        """Create a real Optuna trial for testing."""
        study = optuna.create_study()
        # We need to run a trial to get a trial object
        trial = study.ask()
        return trial

    def test_returns_tuple(self, manager):
        """sample_hyperparameters must return a (model_params, train_params) tuple."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "train_params": {"epochs": 100},
        }
        trial = self._make_trial(fp)
        result = manager.sample_hyperparameters(trial, fp)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_model_params_sampled(self, manager):
        """model_params range specs should be sampled."""
        fp = {"model_params": {"n_components": [5, 10, 15]}}
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert "n_components" in model_params
        assert model_params["n_components"] in [5, 10, 15]
        assert train_params == {}

    def test_static_train_params_passed_through(self, manager):
        """Static train_params (scalar values) should be passed through unchanged."""
        fp = {
            "model_params": {"n_components": [5, 10]},
            "train_params": {"verbose": 0, "epochs": 100},
        }
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert train_params["verbose"] == 0
        assert train_params["epochs"] == 100

    def test_train_params_range_sampled(self, manager):
        """train_params with range tuples should be sampled by Optuna."""
        fp = {
            "model_params": {"n_components": [5, 10]},
            "train_params": {"epochs": ("int", 50, 300)},
        }
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert "epochs" in train_params
        assert 50 <= train_params["epochs"] <= 300

    def test_train_params_list_sampled(self, manager):
        """train_params with list values should be sampled as categorical."""
        fp = {
            "model_params": {"n_components": [5, 10]},
            "train_params": {"batch_size": [16, 32, 64]},
        }
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert train_params["batch_size"] in [16, 32, 64]

    def test_train_params_dict_spec_sampled(self, manager):
        """train_params with dict param spec should be sampled."""
        fp = {
            "model_params": {"n_components": [5, 10]},
            "train_params": {
                "epochs": {"type": "int", "min": 5, "max": 50},
                "verbose": 0,
            },
        }
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert 5 <= train_params["epochs"] <= 50
        assert train_params["verbose"] == 0

    def test_mixed_static_and_sampable_train_params(self, manager):
        """Mix of static and sampable train_params."""
        fp = {
            "model_params": {},
            "train_params": {
                "epochs": ("int", 10, 100),
                "batch_size": [16, 32, 64],
                "verbose": 0,
                "learning_rate": ("float_log", 1e-5, 1e-1),
            },
        }
        trial = self._make_trial(fp)
        _, train_params = manager.sample_hyperparameters(trial, fp)

        assert 10 <= train_params["epochs"] <= 100
        assert train_params["batch_size"] in [16, 32, 64]
        assert train_params["verbose"] == 0
        assert 1e-5 <= train_params["learning_rate"] <= 1e-1

    def test_empty_train_params(self, manager):
        """No train_params should return empty dict."""
        fp = {"model_params": {"n_components": [5, 10]}}
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert train_params == {}

    def test_no_model_params_no_train_params(self, manager):
        """Empty finetune_params should return empty dicts."""
        fp = {"n_trials": 10}
        trial = self._make_trial(fp)
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert model_params == {}
        assert train_params == {}


class TestIsSampable:
    """Tests for _is_sampable helper."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_scalar_int_not_sampable(self, manager):
        assert manager._is_sampable(100) is False

    def test_scalar_float_not_sampable(self, manager):
        assert manager._is_sampable(0.5) is False

    def test_scalar_str_not_sampable(self, manager):
        assert manager._is_sampable("adam") is False

    def test_scalar_bool_not_sampable(self, manager):
        assert manager._is_sampable(True) is False

    def test_scalar_none_not_sampable(self, manager):
        assert manager._is_sampable(None) is False

    def test_list_is_sampable(self, manager):
        assert manager._is_sampable([16, 32, 64]) is True

    def test_tuple_range_is_sampable(self, manager):
        assert manager._is_sampable(("int", 1, 30)) is True

    def test_tuple_pair_is_sampable(self, manager):
        assert manager._is_sampable((1, 30)) is True

    def test_dict_with_type_is_sampable(self, manager):
        assert manager._is_sampable({"type": "int", "min": 1, "max": 10}) is True

    def test_dict_with_min_is_sampable(self, manager):
        assert manager._is_sampable({"min": 1, "max": 10}) is True

    def test_dict_with_low_is_sampable(self, manager):
        assert manager._is_sampable({"low": 1, "high": 10}) is True

    def test_empty_dict_not_sampable(self, manager):
        assert manager._is_sampable({}) is False
