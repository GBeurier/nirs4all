"""Unit tests for multi-phase optimization (Phase 5 - ISSUE-11)."""

import pytest
import optuna

from nirs4all.optimization.optuna import OptunaManager


class TestMultiphaseValidation:
    """Tests for phases validation in _validate_and_normalize_finetune_params."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_valid_phases_accepted(self, manager):
        """Well-formed phases list should pass validation."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": [
                {"sampler": "random", "n_trials": 20},
                {"sampler": "tpe", "n_trials": 30},
            ],
        }
        result = manager._validate_and_normalize_finetune_params(fp)
        assert len(result["phases"]) == 2

    def test_phases_must_be_list(self, manager):
        """phases must be a list, not a dict."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": {"sampler": "random", "n_trials": 20},
        }
        with pytest.raises(ValueError, match="non-empty list"):
            manager._validate_and_normalize_finetune_params(fp)

    def test_phases_must_not_be_empty(self, manager):
        """Empty phases list should raise."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": [],
        }
        with pytest.raises(ValueError, match="non-empty list"):
            manager._validate_and_normalize_finetune_params(fp)

    def test_phase_must_be_dict(self, manager):
        """Each phase must be a dict."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": ["random"],
        }
        with pytest.raises(ValueError, match="must be a dict"):
            manager._validate_and_normalize_finetune_params(fp)

    def test_phase_must_have_n_trials(self, manager):
        """Each phase must include n_trials."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": [{"sampler": "random"}],
        }
        with pytest.raises(ValueError, match="n_trials"):
            manager._validate_and_normalize_finetune_params(fp)

    def test_phase_invalid_sampler_raises(self, manager):
        """Invalid sampler in a phase should raise."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": [{"sampler": "invalid_sampler", "n_trials": 10}],
        }
        with pytest.raises(ValueError, match="Unknown sampler"):
            manager._validate_and_normalize_finetune_params(fp)

    def test_phases_none_is_ignored(self, manager):
        """phases=None should not trigger validation."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "phases": None,
        }
        result = manager._validate_and_normalize_finetune_params(fp)
        assert result["phases"] is None


class TestMultiphaseOptimization:
    """Tests for _optimize_multiphase execution."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_multiphase_runs_all_phases(self, manager):
        """Multi-phase search should run all phases and return best params."""
        fp = {
            "model_params": {"x": [1, 2, 3, 4, 5]},
            "sampler": "grid",
            "phases": [
                {"sampler": "random", "n_trials": 3},
                {"sampler": "tpe", "n_trials": 3},
            ],
        }
        fp = manager._validate_and_normalize_finetune_params(fp)

        # Build a study and objective manually to verify behavior
        study = manager._create_study(fp)

        def objective(trial):
            x = trial.suggest_categorical("x", [1, 2, 3, 4, 5])
            return (x - 3) ** 2  # Minimum at x=3

        # Run phase 1
        sampler1 = manager._create_sampler("random", fp, seed=42)
        study.sampler = sampler1
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        # Run phase 2
        sampler2 = manager._create_sampler("tpe", fp, seed=42)
        study.sampler = sampler2
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        # Should have 6 total trials
        assert len(study.trials) == 6
        # Best should be near x=3
        assert study.best_params["x"] == 3

    def test_phases_key_excluded_from_legacy_model_params(self, manager):
        """'phases' key should not leak into sampled model params."""
        fp = {
            "phases": [{"sampler": "random", "n_trials": 5}],
            "n_trials": 50,
        }
        trial = optuna.create_study().ask()
        model_params, train_params = manager.sample_hyperparameters(trial, fp)
        assert "phases" not in model_params
        assert "n_trials" not in model_params


class TestCreateSampler:
    """Tests for _create_sampler helper."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_tpe_sampler(self, manager):
        from optuna.samplers import TPESampler
        fp = {"model_params": {"x": ("float", 0, 1)}}
        sampler = manager._create_sampler("tpe", fp, seed=42)
        assert isinstance(sampler, TPESampler)

    def test_random_sampler(self, manager):
        from optuna.samplers import RandomSampler
        fp = {"model_params": {"x": ("float", 0, 1)}}
        sampler = manager._create_sampler("random", fp, seed=42)
        assert isinstance(sampler, RandomSampler)

    def test_cmaes_sampler(self, manager):
        from optuna.samplers import CmaEsSampler
        fp = {"model_params": {"x": ("float", 0, 1)}}
        sampler = manager._create_sampler("cmaes", fp, seed=42)
        assert isinstance(sampler, CmaEsSampler)

    def test_grid_sampler_with_categorical(self, manager):
        from optuna.samplers import GridSampler
        fp = {"model_params": {"x": [1, 2, 3]}}
        sampler = manager._create_sampler("grid", fp, seed=42)
        assert isinstance(sampler, GridSampler)

    def test_grid_sampler_falls_back_to_tpe(self, manager):
        """Grid sampler with continuous params should fall back to TPE."""
        from optuna.samplers import TPESampler
        fp = {"model_params": {"x": ("float", 0, 1)}}
        sampler = manager._create_sampler("grid", fp, seed=42)
        assert isinstance(sampler, TPESampler)

    def test_auto_selects_grid_for_categorical(self, manager):
        from optuna.samplers import GridSampler
        fp = {"model_params": {"x": [1, 2, 3]}}
        sampler = manager._create_sampler("auto", fp, seed=42)
        assert isinstance(sampler, GridSampler)

    def test_auto_selects_tpe_for_continuous(self, manager):
        from optuna.samplers import TPESampler
        fp = {"model_params": {"x": ("float", 0, 1)}}
        sampler = manager._create_sampler("auto", fp, seed=42)
        assert isinstance(sampler, TPESampler)
