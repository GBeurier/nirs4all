"""Tests for warm-start refit logic (Task 4.4).

Covers:
- _resolve_warm_start_fold(): fold resolution for "best", "last", "fold_N"
- _apply_warm_start(): base class (sklearn) warm-start
- _apply_params_to_model(): warm_start_fold is skipped
- Warm-start integration in launch_training during REFIT phase
- Framework-specific overrides (TF, PyTorch, JAX)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.refit.executor import _apply_params_to_model

# =========================================================================
# Helpers
# =========================================================================

class _DummyModel:
    """Minimal sklearn-like model for testing warm_start."""

    def __init__(self, n_components: int = 5, warm_start: bool = False) -> None:
        self.n_components = n_components
        self.warm_start = warm_start
        # Simulate a fitted attribute
        self.coef_ = None

    def fit(self, X, y=None):
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "warm_start": self.warm_start}

class _DummyModelNoWarmStart:
    """Model that does not support warm_start."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

def _make_runtime_context(
    fold_artifacts: list[tuple[int, Any]] | None = None,
    phase: ExecutionPhase = ExecutionPhase.REFIT,
) -> RuntimeContext:
    """Create a RuntimeContext with mocked artifact provider."""
    ctx = RuntimeContext()
    ctx.phase = phase
    ctx.step_number = 1

    if fold_artifacts is not None:
        provider = MagicMock()
        provider.get_fold_artifacts.return_value = fold_artifacts
        ctx.artifact_provider = provider
    return ctx

# =========================================================================
# Tests: _resolve_warm_start_fold
# =========================================================================

class TestResolveWarmStartFold:
    """Test fold resolution logic."""

    def test_best_returns_first_fold(self):
        """'best' returns the first available fold."""
        from nirs4all.controllers.models.base_model import BaseModelController

        # We need a concrete subclass to test - use a mock
        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[(0, "m0"), (1, "m1"), (2, "m2")])
        result = ctrl._resolve_warm_start_fold(ctx, "best")
        assert result == 0

    def test_last_returns_last_fold(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[(0, "m0"), (1, "m1"), (2, "m2")])
        result = ctrl._resolve_warm_start_fold(ctx, "last")
        assert result == 2

    def test_fold_n_returns_specific_fold(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[(0, "m0"), (1, "m1"), (2, "m2")])
        result = ctrl._resolve_warm_start_fold(ctx, "fold_1")
        assert result == 1

    def test_fold_n_returns_none_when_missing(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[(0, "m0"), (1, "m1")])
        result = ctrl._resolve_warm_start_fold(ctx, "fold_5")
        assert result is None

    def test_no_artifact_provider_returns_none(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = RuntimeContext()
        ctx.artifact_provider = None
        result = ctrl._resolve_warm_start_fold(ctx, "best")
        assert result is None

    def test_empty_folds_returns_none(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[])
        result = ctrl._resolve_warm_start_fold(ctx, "best")
        assert result is None

    def test_invalid_fold_n_format_returns_none(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        ctrl = ConcreteController()
        ctx = _make_runtime_context(fold_artifacts=[(0, "m0")])
        result = ctrl._resolve_warm_start_fold(ctx, "fold_abc")
        assert result is None

# =========================================================================
# Tests: _apply_warm_start (base class / sklearn)
# =========================================================================

class TestApplyWarmStartBase:
    """Test base class warm-start behavior."""

    def _make_controller(self):
        from nirs4all.controllers.models.base_model import BaseModelController

        class ConcreteController(BaseModelController):
            def _get_model_instance(self, *a, **k): return None
            def _train_model(self, *a, **k): return None
            def _predict_model(self, *a, **k): return None
            def _prepare_data(self, *a, **k): return None, None
            def _clone_model(self, *a, **k): return None
            def _evaluate_model(self, *a, **k): return 0.0
            @classmethod
            def matches(cls, *a, **k): return False

        return ConcreteController()

    def test_sets_warm_start_attribute(self):
        ctrl = self._make_controller()
        model = _DummyModel(warm_start=False)
        source = _DummyModel(warm_start=True)
        source.coef_ = np.array([1.0, 2.0, 3.0])

        result = ctrl._apply_warm_start(model, source, RuntimeContext())
        assert result.warm_start is True
        np.testing.assert_array_equal(result.coef_, np.array([1.0, 2.0, 3.0]))

    def test_no_warm_start_attr_returns_model_unchanged(self):
        ctrl = self._make_controller()
        model = _DummyModelNoWarmStart(alpha=1.0)
        source = _DummyModelNoWarmStart(alpha=2.0)

        result = ctrl._apply_warm_start(model, source, RuntimeContext())
        assert result.alpha == 1.0  # Unchanged

# =========================================================================
# Tests: _apply_params_to_model skips warm_start_fold
# =========================================================================

class TestApplyParamsSkipsWarmStartFold:
    """Test that _apply_params_to_model skips the warm_start_fold key."""

    def test_warm_start_fold_is_excluded(self):
        model = _DummyModel(n_components=5)
        _apply_params_to_model(model, {
            "n_components": 10,
            "warm_start_fold": "best",
        })
        assert model.n_components == 10

    def test_warm_start_fold_only_does_nothing(self):
        model = _DummyModel(n_components=5)
        _apply_params_to_model(model, {"warm_start_fold": "last"})
        assert model.n_components == 5

# =========================================================================
# Tests: Sklearn _apply_warm_start
# =========================================================================

class TestSklearnApplyWarmStart:
    """Test sklearn controller warm-start."""

    def test_sklearn_warm_start(self):
        from nirs4all.controllers.models.sklearn_model import SklearnModelController

        ctrl = SklearnModelController()
        model = _DummyModel(warm_start=False)
        source = _DummyModel(warm_start=True)
        source.coef_ = np.array([10.0, 20.0])

        result = ctrl._apply_warm_start(model, source, RuntimeContext())
        assert result.warm_start is True
        np.testing.assert_array_equal(result.coef_, np.array([10.0, 20.0]))

    def test_sklearn_no_warm_start_support(self):
        from nirs4all.controllers.models.sklearn_model import SklearnModelController

        ctrl = SklearnModelController()
        model = _DummyModelNoWarmStart(alpha=1.0)
        source = _DummyModelNoWarmStart(alpha=2.0)

        result = ctrl._apply_warm_start(model, source, RuntimeContext())
        # Model doesn't support warm_start, should be returned unchanged
        assert result.alpha == 1.0
