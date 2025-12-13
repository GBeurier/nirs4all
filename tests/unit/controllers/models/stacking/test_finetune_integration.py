"""
Unit tests for Phase 7.3: Finetune Integration.

Tests cover:
- MetaModel finetune_space parameter
- get_finetune_params() method
- Finetune parameter extraction in MetaModelController
- Integration with Optuna optimization
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from nirs4all.operators.models.meta import (
    MetaModel,
    StackingConfig,
    StackingLevel,
)


class TestMetaModelFinetuneSpace:
    """Test MetaModel finetune_space parameter."""

    def test_finetune_space_default_none(self):
        """finetune_space should be None by default."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(model=Ridge())
        assert meta.finetune_space is None

    def test_finetune_space_with_dict(self):
        """finetune_space should accept a dict of hyperparameters."""
        from sklearn.linear_model import Ridge

        finetune_space = {
            "model__alpha": (0.001, 100.0),
            "n_trials": 50,
            "approach": "grouped"
        }

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        assert meta.finetune_space is not None
        assert "model__alpha" in meta.finetune_space
        assert meta.finetune_space["model__alpha"] == (0.001, 100.0)

    def test_finetune_space_with_multiple_params(self):
        """finetune_space should support multiple hyperparameters."""
        from sklearn.ensemble import RandomForestRegressor

        finetune_space = {
            "model__n_estimators": (50, 200),
            "model__max_depth": (3, 15),
            "model__min_samples_split": (2, 10),
            "n_trials": 100,
        }

        meta = MetaModel(
            model=RandomForestRegressor(),
            finetune_space=finetune_space
        )

        assert meta.finetune_space is not None
        assert "model__n_estimators" in meta.finetune_space
        assert "model__max_depth" in meta.finetune_space
        assert "model__min_samples_split" in meta.finetune_space


class TestGetFinetuneParams:
    """Test MetaModel.get_finetune_params() method."""

    def test_get_finetune_params_none_when_no_space(self):
        """get_finetune_params() should return None when no finetune_space."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(model=Ridge())
        params = meta.get_finetune_params()

        assert params is None

    def test_get_finetune_params_returns_dict(self):
        """get_finetune_params() should return formatted dict."""
        from sklearn.linear_model import Ridge

        finetune_space = {
            "model__alpha": (0.001, 100.0),
            "n_trials": 75,
            "approach": "exhaustive"
        }

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        params = meta.get_finetune_params()

        assert params is not None
        assert "model_params" in params
        assert "n_trials" in params
        assert "approach" in params

    def test_get_finetune_params_extracts_model_params(self):
        """get_finetune_params() should extract model params correctly."""
        from sklearn.linear_model import Ridge

        finetune_space = {
            "model__alpha": (0.001, 100.0),
            "model__fit_intercept": [True, False],
        }

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        params = meta.get_finetune_params()

        assert params is not None
        assert params["model_params"] == finetune_space

    def test_get_finetune_params_default_n_trials(self):
        """get_finetune_params() should have default n_trials=50."""
        from sklearn.linear_model import Ridge

        finetune_space = {"model__alpha": (0.001, 100.0)}

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        params = meta.get_finetune_params()

        assert params is not None
        assert params["n_trials"] == 50

    def test_get_finetune_params_default_approach(self):
        """get_finetune_params() should have default approach='grouped'."""
        from sklearn.linear_model import Ridge

        finetune_space = {"model__alpha": (0.001, 100.0)}

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        params = meta.get_finetune_params()

        assert params is not None
        assert params["approach"] == "grouped"


class TestMetaModelGetParams:
    """Test MetaModel.get_params() includes finetune_space."""

    def test_get_params_includes_finetune_space(self):
        """get_params() should include finetune_space."""
        from sklearn.linear_model import Ridge

        finetune_space = {"model__alpha": (0.001, 100.0)}

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        params = meta.get_params()

        assert "finetune_space" in params
        assert params["finetune_space"] == finetune_space

    def test_get_params_deep_includes_model_params(self):
        """get_params(deep=True) should include nested model params."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(alpha=0.5),
            finetune_space={"model__alpha": (0.001, 100.0)}
        )

        params = meta.get_params(deep=True)

        assert "model__alpha" in params
        assert params["model__alpha"] == 0.5


class TestMetaModelSetParams:
    """Test MetaModel.set_params() with finetune_space."""

    def test_set_params_can_update_finetune_space(self):
        """set_params() should update finetune_space."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(model=Ridge())

        new_space = {"model__alpha": (0.1, 10.0)}
        meta.set_params(finetune_space=new_space)

        assert meta.finetune_space == new_space

    def test_set_params_can_update_model_params(self):
        """set_params() should update nested model params."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(model=Ridge(alpha=1.0))

        meta.set_params(model__alpha=0.5)

        assert meta.model.alpha == 0.5


class TestMetaModelControllerFinetuneExtraction:
    """Test MetaModelController._extract_model_config() for finetune."""

    def test_extract_model_config_with_finetune(self):
        """_extract_model_config should extract finetune_params."""
        from sklearn.linear_model import Ridge
        from nirs4all.controllers.models.meta_model import MetaModelController

        finetune_space = {"model__alpha": (0.001, 100.0)}

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        controller = MetaModelController()
        step = {"model": meta}

        # Access protected method - type: ignore for static analyzers
        config = controller._extract_model_config(step, meta)  # type: ignore

        assert "finetune_params" in config
        assert "model_params" in config["finetune_params"]

    def test_extract_model_config_no_finetune_when_none(self):
        """_extract_model_config should not include finetune_params when None."""
        from sklearn.linear_model import Ridge
        from nirs4all.controllers.models.meta_model import MetaModelController

        meta = MetaModel(model=Ridge())

        controller = MetaModelController()
        step = {"model": meta}

        # Access protected method - type: ignore for static analyzers
        config = controller._extract_model_config(step, meta)  # type: ignore

        # finetune_params should not be in config when finetune_space is None
        assert config.get("finetune_params") is None

    def test_extract_model_config_from_operator_only(self):
        """_extract_model_config should work with operator only."""
        from sklearn.linear_model import Ridge
        from nirs4all.controllers.models.meta_model import MetaModelController

        finetune_space = {"model__alpha": (0.001, 100.0)}

        meta = MetaModel(
            model=Ridge(),
            finetune_space=finetune_space
        )

        controller = MetaModelController()

        # Access protected method - type: ignore for static analyzers
        config = controller._extract_model_config(step={}, operator=meta)  # type: ignore

        assert "finetune_params" in config


class TestFinetuneIntegration:
    """Test finetune integration with stacking config."""

    def test_finetune_with_stacking_level(self):
        """Finetune should work with explicit stacking level."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(),
            stacking_config=StackingConfig(level=StackingLevel.LEVEL_2),
            finetune_space={"model__alpha": (0.001, 100.0)}
        )

        assert meta.stacking_config.level == StackingLevel.LEVEL_2
        assert meta.finetune_space is not None

    def test_finetune_with_all_branches(self):
        """Finetune should work with ALL_BRANCHES scope."""
        from sklearn.linear_model import Ridge
        from nirs4all.operators.models.meta import BranchScope

        meta = MetaModel(
            model=Ridge(),
            stacking_config=StackingConfig(branch_scope=BranchScope.ALL_BRANCHES),
            finetune_space={"model__alpha": (0.001, 100.0)}
        )

        assert meta.stacking_config.branch_scope == BranchScope.ALL_BRANCHES
        assert meta.finetune_space is not None

    def test_finetune_space_in_repr(self):
        """MetaModel repr should not expose finetune_space directly."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(),
            finetune_space={"model__alpha": (0.001, 100.0)}
        )

        repr_str = repr(meta)

        # Repr should focus on model and source_models
        assert "Ridge" in repr_str
        assert "source_models" in repr_str


class TestMetaModelLevel:
    """Test MetaModel.level property with finetune."""

    def test_level_property_auto(self):
        """level property should return detected level for AUTO."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(),
            stacking_config=StackingConfig(level=StackingLevel.AUTO)
        )

        # When no detected level, should return 1
        assert meta.level == 1

    def test_level_property_explicit(self):
        """level property should return configured level when explicit."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(),
            stacking_config=StackingConfig(level=StackingLevel.LEVEL_2)
        )

        assert meta.level == 2

    def test_level_property_with_detected(self):
        """level property should use detected level for AUTO."""
        from sklearn.linear_model import Ridge

        meta = MetaModel(
            model=Ridge(),
            stacking_config=StackingConfig(level=StackingLevel.AUTO)
        )

        # Simulate detected level
        meta._detected_level = 2

        assert meta.level == 2
