"""Unit tests for YTransformerMixinController (C-05).

Covers: matches(), _normalize_operators(), _match_transformer_by_class(),
and train-mode execute() via a minimal mock context.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleTransformer(TransformerMixin, BaseEstimator):
    """Minimal transformer for testing."""
    def fit(self, X, y=None):
        self.mean_ = X.mean()
        return self
    def transform(self, X):
        return X - self.mean_

class _NotATransformer:
    """An object that is NOT a TransformerMixin."""
    pass

# ---------------------------------------------------------------------------
# matches() classmethod
# ---------------------------------------------------------------------------

class TestYTransformerMixinControllerMatches:

    def setup_method(self):
        from nirs4all.controllers.transforms.y_transformer import YTransformerMixinController
        self.ctrl = YTransformerMixinController

    def test_matches_instance_with_y_processing_keyword(self):
        assert self.ctrl.matches(None, MinMaxScaler(), "y_processing") is True

    def test_matches_class_with_y_processing_keyword(self):
        assert self.ctrl.matches(None, MinMaxScaler, "y_processing") is True

    def test_no_match_wrong_keyword(self):
        assert self.ctrl.matches(None, MinMaxScaler(), "model") is False
        assert self.ctrl.matches(None, MinMaxScaler(), "exclude") is False
        assert self.ctrl.matches(None, MinMaxScaler(), "") is False

    def test_matches_list_of_transformers(self):
        ops = [MinMaxScaler(), StandardScaler()]
        assert self.ctrl.matches(None, ops, "y_processing") is True

    def test_matches_mixed_instance_and_class_list(self):
        ops = [MinMaxScaler, StandardScaler()]
        assert self.ctrl.matches(None, ops, "y_processing") is True

    def test_no_match_list_with_non_transformer(self):
        ops = [MinMaxScaler(), _NotATransformer()]
        assert self.ctrl.matches(None, ops, "y_processing") is False

    def test_no_match_non_transformer_instance(self):
        assert self.ctrl.matches(None, _NotATransformer(), "y_processing") is False

    def test_no_match_empty_list(self):
        assert self.ctrl.matches(None, [], "y_processing") is False

    def test_matches_custom_transformer(self):
        assert self.ctrl.matches(None, _SimpleTransformer(), "y_processing") is True

# ---------------------------------------------------------------------------
# _normalize_operators()
# ---------------------------------------------------------------------------

class TestNormalizeOperators:

    def setup_method(self):
        from nirs4all.controllers.transforms.y_transformer import YTransformerMixinController
        self.ctrl = YTransformerMixinController()

    def test_single_instance_wrapped_in_list(self):
        result = self.ctrl._normalize_operators(MinMaxScaler())
        assert len(result) == 1
        assert isinstance(result[0], MinMaxScaler)

    def test_class_is_instantiated(self):
        result = self.ctrl._normalize_operators(MinMaxScaler)
        assert len(result) == 1
        assert isinstance(result[0], MinMaxScaler)

    def test_list_of_instances_returned_as_is(self):
        ops = [MinMaxScaler(), StandardScaler()]
        result = self.ctrl._normalize_operators(ops)
        assert len(result) == 2
        assert isinstance(result[0], MinMaxScaler)
        assert isinstance(result[1], StandardScaler)

    def test_list_of_classes_instantiated(self):
        ops = [MinMaxScaler, StandardScaler]
        result = self.ctrl._normalize_operators(ops)
        assert all(isinstance(r, TransformerMixin) for r in result)

    def test_tuple_treated_same_as_list(self):
        ops = (MinMaxScaler(), StandardScaler())
        result = self.ctrl._normalize_operators(ops)
        assert len(result) == 2

    def test_mixed_class_and_instance(self):
        ops = [MinMaxScaler, StandardScaler()]
        result = self.ctrl._normalize_operators(ops)
        assert isinstance(result[0], MinMaxScaler)
        assert isinstance(result[1], StandardScaler)

# ---------------------------------------------------------------------------
# _match_transformer_by_class()
# ---------------------------------------------------------------------------

class TestMatchTransformerByClass:

    def setup_method(self):
        from nirs4all.controllers.transforms.y_transformer import YTransformerMixinController
        self.ctrl = YTransformerMixinController()

    def test_finds_first_match_by_index_zero(self):
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        artifacts = [("art1", scaler1), ("art2", scaler2)]
        result = self.ctrl._match_transformer_by_class("MinMaxScaler", artifacts, target_index=0)
        assert result is scaler1

    def test_finds_second_match_by_index_one(self):
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        artifacts = [("art1", scaler1), ("art2", scaler2)]
        result = self.ctrl._match_transformer_by_class("MinMaxScaler", artifacts, target_index=1)
        assert result is scaler2

    def test_returns_none_for_no_match(self):
        artifacts = [("art1", StandardScaler())]
        result = self.ctrl._match_transformer_by_class("MinMaxScaler", artifacts, target_index=0)
        assert result is None

    def test_returns_none_when_index_exceeds_matches(self):
        artifacts = [("art1", MinMaxScaler())]
        result = self.ctrl._match_transformer_by_class("MinMaxScaler", artifacts, target_index=1)
        assert result is None

    def test_ignores_non_matching_class(self):
        artifacts = [
            ("art1", StandardScaler()),
            ("art2", MinMaxScaler()),
            ("art3", StandardScaler()),
        ]
        result = self.ctrl._match_transformer_by_class("MinMaxScaler", artifacts, target_index=0)
        assert isinstance(result, MinMaxScaler)

# ---------------------------------------------------------------------------
# Class-level predicates
# ---------------------------------------------------------------------------

class TestClassPredicates:

    def setup_method(self):
        from nirs4all.controllers.transforms.y_transformer import YTransformerMixinController
        self.ctrl = YTransformerMixinController

    def test_use_multi_source_is_false(self):
        assert self.ctrl.use_multi_source() is False

    def test_supports_prediction_mode_is_true(self):
        assert self.ctrl.supports_prediction_mode() is True

    def test_priority_is_set(self):
        assert isinstance(self.ctrl.priority, int)
