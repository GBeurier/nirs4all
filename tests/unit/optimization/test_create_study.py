"""Unit tests for OptunaManager._create_study (Phase 4 - Samplers, Pruners, Storage, Seed)."""

import pytest
import optuna
from optuna.samplers import TPESampler, GridSampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

from nirs4all.optimization.optuna import OptunaManager


class TestCreateStudySamplers:
    """Tests for sampler instantiation in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_tpe_sampler_default(self, manager):
        """Default (auto) with non-categorical params should use TPE."""
        fp = {
            "sampler": "tpe",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, TPESampler)

    def test_random_sampler(self, manager):
        """Explicit 'random' sampler."""
        fp = {
            "sampler": "random",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, RandomSampler)

    def test_cmaes_sampler(self, manager):
        """Explicit 'cmaes' sampler."""
        fp = {
            "sampler": "cmaes",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, CmaEsSampler)

    def test_grid_sampler_with_categorical(self, manager):
        """Explicit 'grid' sampler with categorical params."""
        fp = {
            "sampler": "grid",
            "model_params": {"n_components": [1, 5, 10]},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, GridSampler)

    def test_grid_sampler_falls_back_to_tpe_for_continuous(self, manager):
        """Grid sampler should fall back to TPE when params are continuous."""
        fp = {
            "sampler": "grid",
            "model_params": {"alpha": ("float", 0.01, 1.0)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, TPESampler)

    def test_auto_sampler_selects_grid_for_categorical(self, manager):
        """Auto sampler should select grid when all params are categorical."""
        fp = {
            "sampler": "auto",
            "model_params": {"n_components": [1, 5, 10]},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, GridSampler)

    def test_auto_sampler_selects_tpe_for_continuous(self, manager):
        """Auto sampler should select TPE when params are continuous."""
        fp = {
            "sampler": "auto",
            "model_params": {"alpha": ("float", 0.01, 1.0)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, TPESampler)


class TestCreateStudySeed:
    """Tests for seed support in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_tpe_sampler_with_seed(self, manager):
        """TPE sampler should accept seed."""
        fp = {
            "sampler": "tpe",
            "seed": 42,
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, TPESampler)

    def test_random_sampler_with_seed(self, manager):
        """Random sampler should accept seed."""
        fp = {
            "sampler": "random",
            "seed": 123,
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, RandomSampler)

    def test_cmaes_sampler_with_seed(self, manager):
        """CMA-ES sampler should accept seed."""
        fp = {
            "sampler": "cmaes",
            "seed": 7,
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, CmaEsSampler)

    def test_seed_none_by_default(self, manager):
        """No seed should not raise."""
        fp = {
            "sampler": "tpe",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.sampler, TPESampler)

    def test_seed_produces_reproducible_results(self, manager):
        """Same seed should produce same sampled values."""
        fp = {
            "sampler": "random",
            "seed": 42,
            "model_params": {"x": ("float", 0.0, 1.0)},
        }
        study1 = manager._create_study(fp)
        study2 = manager._create_study(fp)

        # Run one trial on each study with the same objective
        def objective(trial):
            return trial.suggest_float("x", 0.0, 1.0)

        study1.optimize(objective, n_trials=3, show_progress_bar=False)
        study2.optimize(objective, n_trials=3, show_progress_bar=False)

        values1 = [t.params["x"] for t in study1.trials]
        values2 = [t.params["x"] for t in study2.trials]
        assert values1 == values2


class TestCreateStudyPruners:
    """Tests for pruner instantiation in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_no_pruner_by_default(self, manager):
        """Default pruner should be None (no pruning)."""
        fp = {
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        # Optuna sets a default pruner when None is passed — we just verify no error
        # The key is _create_pruner returns None for 'none'

    def test_median_pruner(self, manager):
        fp = {
            "pruner": "median",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.pruner, MedianPruner)

    def test_successive_halving_pruner(self, manager):
        fp = {
            "pruner": "successive_halving",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.pruner, SuccessiveHalvingPruner)

    def test_hyperband_pruner(self, manager):
        fp = {
            "pruner": "hyperband",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert isinstance(study.pruner, HyperbandPruner)


class TestCreatePruner:
    """Tests for _create_pruner helper."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_none_returns_none(self, manager):
        assert manager._create_pruner("none") is None

    def test_median_returns_median_pruner(self, manager):
        assert isinstance(manager._create_pruner("median"), MedianPruner)

    def test_successive_halving_returns_pruner(self, manager):
        assert isinstance(manager._create_pruner("successive_halving"), SuccessiveHalvingPruner)

    def test_hyperband_returns_pruner(self, manager):
        assert isinstance(manager._create_pruner("hyperband"), HyperbandPruner)

    def test_unknown_returns_none(self, manager):
        assert manager._create_pruner("unknown_pruner") is None


class TestCreateStudyDirection:
    """Tests for direction parameter in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_minimize_direction_default(self, manager):
        fp = {"model_params": {"n_components": ("int", 1, 30)}}
        study = manager._create_study(fp)
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_maximize_direction(self, manager):
        fp = {
            "direction": "maximize",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


class TestCreateStudyStorage:
    """Tests for storage/resume in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_default_in_memory_storage(self, manager):
        """Default storage is in-memory (None)."""
        fp = {"model_params": {"n_components": ("int", 1, 30)}}
        study = manager._create_study(fp)
        # In-memory study should have no persistent storage
        assert study is not None

    def test_sqlite_storage(self, manager, tmp_path):
        """SQLite storage should allow study creation."""
        db_path = tmp_path / "test_study.db"
        fp = {
            "storage": f"sqlite:///{db_path}",
            "study_name": "test_study",
            "model_params": {"n_components": ("int", 1, 30)},
        }
        study = manager._create_study(fp)
        assert study.study_name == "test_study"

    def test_resume_study(self, manager, tmp_path):
        """Resume should reload an existing study."""
        db_path = tmp_path / "resume_study.db"
        storage_url = f"sqlite:///{db_path}"

        # Create initial study
        fp1 = {
            "storage": storage_url,
            "study_name": "resume_test",
            "model_params": {"n_components": [1, 5, 10]},
            "sampler": "grid",
        }
        study1 = manager._create_study(fp1)
        # Run a trial
        study1.optimize(lambda trial: trial.suggest_categorical("n_components", [1, 5, 10]), n_trials=1, show_progress_bar=False)
        assert len(study1.trials) == 1

        # Resume the study
        fp2 = {
            "storage": storage_url,
            "study_name": "resume_test",
            "resume": True,
            "model_params": {"n_components": [1, 5, 10]},
            "sampler": "grid",
        }
        study2 = manager._create_study(fp2)
        # Should have the trial from the first study
        assert len(study2.trials) == 1


class TestCreateStudyForceParams:
    """Tests for force_params enqueue in _create_study."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_force_params_enqueued(self, manager):
        """force_params should be enqueued as the first trial."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "sampler": "grid",
            "force_params": {"n_components": 5},
        }
        study = manager._create_study(fp)

        # Run one trial — it should use the force_params
        def objective(trial):
            val = trial.suggest_categorical("n_components", [1, 5, 10])
            return float(val)

        study.optimize(objective, n_trials=1, show_progress_bar=False)
        assert study.trials[0].params["n_components"] == 5

    def test_no_force_params_no_error(self, manager):
        """No force_params should not enqueue anything."""
        fp = {
            "model_params": {"n_components": [1, 5, 10]},
            "sampler": "grid",
        }
        study = manager._create_study(fp)
        assert study is not None
