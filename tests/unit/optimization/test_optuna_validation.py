"""Unit tests for OptunaManager config validation (Phase 2 + Phase 4)."""

import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestValidateAndNormalizeFinetuneParams:
    """Tests for _validate_and_normalize_finetune_params method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    # ---- Sampler validation ----

    def test_valid_sampler_auto(self, manager):
        params = manager._validate_and_normalize_finetune_params({"sampler": "auto"})
        assert params["sampler"] == "auto"

    def test_valid_sampler_tpe(self, manager):
        params = manager._validate_and_normalize_finetune_params({"sampler": "tpe"})
        assert params["sampler"] == "tpe"

    def test_valid_sampler_grid(self, manager):
        params = manager._validate_and_normalize_finetune_params({"sampler": "grid"})
        assert params["sampler"] == "grid"

    def test_valid_sampler_random(self, manager):
        params = manager._validate_and_normalize_finetune_params({"sampler": "random"})
        assert params["sampler"] == "random"

    def test_valid_sampler_cmaes(self, manager):
        """'cmaes' is implemented in Phase 4."""
        params = manager._validate_and_normalize_finetune_params({"sampler": "cmaes"})
        assert params["sampler"] == "cmaes"

    def test_invalid_sampler_hyperband_raises(self, manager):
        """'hyperband' is not a sampler — must raise, not silently fallback."""
        with pytest.raises(ValueError, match="Unknown sampler 'hyperband'"):
            manager._validate_and_normalize_finetune_params({"sampler": "hyperband"})

    def test_invalid_sampler_bogus_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown sampler"):
            manager._validate_and_normalize_finetune_params({"sampler": "bogus"})

    # ---- Approach validation ----

    def test_valid_approach_grouped(self, manager):
        params = manager._validate_and_normalize_finetune_params({"approach": "grouped"})
        assert params["approach"] == "grouped"

    def test_valid_approach_individual(self, manager):
        params = manager._validate_and_normalize_finetune_params({"approach": "individual"})
        assert params["approach"] == "individual"

    def test_valid_approach_single(self, manager):
        params = manager._validate_and_normalize_finetune_params({"approach": "single"})
        assert params["approach"] == "single"

    def test_invalid_approach_cross_raises(self, manager):
        """'cross' was never implemented — must raise."""
        with pytest.raises(ValueError, match="Unknown approach 'cross'"):
            manager._validate_and_normalize_finetune_params({"approach": "cross"})

    def test_invalid_approach_bogus_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown approach"):
            manager._validate_and_normalize_finetune_params({"approach": "bogus"})

    # ---- Eval mode validation ----

    def test_valid_eval_mode_best(self, manager):
        params = manager._validate_and_normalize_finetune_params({"eval_mode": "best"})
        assert params["eval_mode"] == "best"

    def test_valid_eval_mode_mean(self, manager):
        params = manager._validate_and_normalize_finetune_params({"eval_mode": "mean"})
        assert params["eval_mode"] == "mean"

    def test_valid_eval_mode_robust_best(self, manager):
        params = manager._validate_and_normalize_finetune_params({"eval_mode": "robust_best"})
        assert params["eval_mode"] == "robust_best"

    def test_invalid_eval_mode_bogus_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            manager._validate_and_normalize_finetune_params({"eval_mode": "bogus"})

    # ---- Pruner validation (Phase 4) ----

    def test_valid_pruner_none(self, manager):
        params = manager._validate_and_normalize_finetune_params({"pruner": "none"})
        assert params["pruner"] == "none"

    def test_valid_pruner_median(self, manager):
        params = manager._validate_and_normalize_finetune_params({"pruner": "median"})
        assert params["pruner"] == "median"

    def test_valid_pruner_successive_halving(self, manager):
        params = manager._validate_and_normalize_finetune_params({"pruner": "successive_halving"})
        assert params["pruner"] == "successive_halving"

    def test_valid_pruner_hyperband(self, manager):
        params = manager._validate_and_normalize_finetune_params({"pruner": "hyperband"})
        assert params["pruner"] == "hyperband"

    def test_invalid_pruner_bogus_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown pruner"):
            manager._validate_and_normalize_finetune_params({"pruner": "bogus"})

    def test_default_pruner_none(self, manager):
        """Default pruner is 'none' — no pruning."""
        params = manager._validate_and_normalize_finetune_params({})
        # No error; default pruner is 'none'

    # ---- Alias normalization ----

    def test_sample_key_normalized_to_sampler(self, manager):
        """Legacy 'sample' key must be normalized to 'sampler'."""
        params = manager._validate_and_normalize_finetune_params({"sample": "grid"})
        assert "sampler" in params
        assert "sample" not in params
        assert params["sampler"] == "grid"

    def test_sample_key_ignored_when_sampler_present(self, manager):
        """When both 'sample' and 'sampler' exist, 'sampler' wins."""
        params = manager._validate_and_normalize_finetune_params({
            "sample": "grid",
            "sampler": "tpe",
        })
        assert params["sampler"] == "tpe"
        assert "sample" not in params

    def test_avg_eval_mode_normalized_to_mean(self, manager):
        """Legacy 'avg' must be normalized to 'mean'."""
        params = manager._validate_and_normalize_finetune_params({"eval_mode": "avg"})
        assert params["eval_mode"] == "mean"

    # ---- Default values ----

    def test_defaults_valid_when_empty(self, manager):
        """Empty params should pass validation (defaults are valid)."""
        params = manager._validate_and_normalize_finetune_params({})
        # No error raised; defaults are 'auto', 'grouped', 'best'
        assert "sampler" not in params  # Not set, so uses default in caller

    def test_other_keys_preserved(self, manager):
        """Validation must not strip unknown keys like model_params."""
        original = {
            "n_trials": 50,
            "model_params": {"n_components": ("int", 1, 30)},
            "train_params": {"epochs": 10},
            "seed": 42,
            "pruner": "median",
        }
        params = manager._validate_and_normalize_finetune_params(original)
        assert params["n_trials"] == 50
        assert params["model_params"] == {"n_components": ("int", 1, 30)}
        assert params["train_params"] == {"epochs": 10}
        assert params["seed"] == 42
        assert params["pruner"] == "median"
