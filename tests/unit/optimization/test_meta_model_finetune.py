"""Unit tests for MetaModel finetune parameter handling (Phase 5 - ISSUE-9/16b)."""

import pytest
from sklearn.linear_model import Ridge, Lasso


class TestMetaModelGetFinetuneParams:
    """Tests for MetaModel.get_finetune_params() — control key separation."""

    def test_no_finetune_space_returns_none(self):
        from nirs4all.operators.models.meta import MetaModel
        meta = MetaModel(model=Ridge())
        assert meta.get_finetune_params() is None

    def test_separates_control_keys_from_model_params(self):
        from nirs4all.operators.models.meta import MetaModel
        meta = MetaModel(
            model=Ridge(),
            finetune_space={
                "model__alpha": ("float_log", 1e-4, 1e-1),
                "n_trials": 30,
                "sampler": "tpe",
                "approach": "grouped",
            },
        )
        fp = meta.get_finetune_params()
        assert fp is not None

        # Control keys
        assert fp["n_trials"] == 30
        assert fp["sampler"] == "tpe"
        assert fp["approach"] == "grouped"

        # Model params should only contain the actual hyperparameter spec
        assert "model__alpha" in fp["model_params"]
        assert "n_trials" not in fp["model_params"]
        assert "sampler" not in fp["model_params"]
        assert "approach" not in fp["model_params"]

    def test_default_control_values(self):
        from nirs4all.operators.models.meta import MetaModel
        meta = MetaModel(
            model=Ridge(),
            finetune_space={"model__alpha": ("float_log", 1e-4, 1e-1)},
        )
        fp = meta.get_finetune_params()

        assert fp["n_trials"] == 50
        assert fp["approach"] == "grouped"
        assert fp["eval_mode"] == "best"
        assert fp["verbose"] == 0

    def test_all_control_keys_forwarded(self):
        """All recognized control keys should be forwarded."""
        from nirs4all.operators.models.meta import MetaModel
        meta = MetaModel(
            model=Ridge(),
            finetune_space={
                "model__alpha": [0.01, 0.1, 1.0],
                "n_trials": 100,
                "sampler": "random",
                "pruner": "median",
                "seed": 42,
                "direction": "minimize",
                "force_params": {"model__alpha": 0.1},
                "phases": [
                    {"sampler": "random", "n_trials": 50},
                    {"sampler": "tpe", "n_trials": 50},
                ],
            },
        )
        fp = meta.get_finetune_params()

        assert fp["sampler"] == "random"
        assert fp["pruner"] == "median"
        assert fp["seed"] == 42
        assert fp["direction"] == "minimize"
        assert fp["force_params"] == {"model__alpha": 0.1}
        assert len(fp["phases"]) == 2

        # Model params should only have the alpha spec
        assert list(fp["model_params"].keys()) == ["model__alpha"]


class TestMetaModelControllerModelPrefix:
    """Tests for MetaModelController._get_model_instance model__ prefix stripping."""

    def test_model_prefix_stripped(self):
        """model__alpha should be stripped to alpha for the inner model."""
        from nirs4all.controllers.models.meta_model import MetaModelController
        from nirs4all.operators.models.meta import MetaModel

        controller = MetaModelController()
        meta_op = MetaModel(model=Ridge(alpha=1.0))

        model_config = {"model_instance": meta_op}
        model = controller._get_model_instance(
            dataset=None,
            model_config=model_config,
            force_params={"model__alpha": 0.01}
        )

        assert model.alpha == 0.01

    def test_plain_params_passed_through(self):
        """Params without model__ prefix should be passed directly."""
        from nirs4all.controllers.models.meta_model import MetaModelController
        from nirs4all.operators.models.meta import MetaModel

        controller = MetaModelController()
        meta_op = MetaModel(model=Ridge(alpha=1.0))

        model_config = {"model_instance": meta_op}
        model = controller._get_model_instance(
            dataset=None,
            model_config=model_config,
            force_params={"alpha": 0.05}
        )

        assert model.alpha == 0.05

    def test_mixed_prefixed_and_plain(self):
        """Mix of prefixed and plain params."""
        from nirs4all.controllers.models.meta_model import MetaModelController
        from nirs4all.operators.models.meta import MetaModel

        controller = MetaModelController()
        meta_op = MetaModel(model=Ridge(alpha=1.0, fit_intercept=True))

        model_config = {"model_instance": meta_op}
        model = controller._get_model_instance(
            dataset=None,
            model_config=model_config,
            force_params={
                "model__alpha": 0.01,
                "fit_intercept": False,
            }
        )

        assert model.alpha == 0.01
        assert model.fit_intercept is False

    def test_no_force_params_returns_model_unchanged(self):
        """No force_params should return the original model."""
        from nirs4all.controllers.models.meta_model import MetaModelController
        from nirs4all.operators.models.meta import MetaModel

        controller = MetaModelController()
        meta_op = MetaModel(model=Ridge(alpha=5.0))

        model_config = {"model_instance": meta_op}
        model = controller._get_model_instance(
            dataset=None,
            model_config=model_config,
        )

        assert model.alpha == 5.0
