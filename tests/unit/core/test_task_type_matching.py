"""Tests for task-type matching and SQL resolution in nirs4all.core.task_type."""

import pytest

from nirs4all.core.task_type import matches_task_type, resolve_task_type_sql


class TestMatchesTaskType:
    """Test fuzzy alias matching for task_type filtering."""

    def test_exact_regression(self):
        assert matches_task_type("regression", "regression") is True

    def test_reg_alias(self):
        assert matches_task_type("regression", "reg") is True

    def test_regression_vs_classification(self):
        assert matches_task_type("regression", "classification") is False

    def test_binary_matches_classification(self):
        assert matches_task_type("binary_classification", "classification") is True

    def test_multiclass_matches_classification(self):
        assert matches_task_type("multiclass_classification", "classification") is True

    def test_binary_matches_clf(self):
        assert matches_task_type("binary_classification", "clf") is True

    def test_multiclass_matches_clf(self):
        assert matches_task_type("multiclass_classification", "clf") is True

    def test_binary_matches_binary(self):
        assert matches_task_type("binary_classification", "binary") is True

    def test_multiclass_does_not_match_binary(self):
        assert matches_task_type("multiclass_classification", "binary") is False

    def test_multiclass_matches_multiclass(self):
        assert matches_task_type("multiclass_classification", "multiclass") is True

    def test_binary_does_not_match_multiclass(self):
        assert matches_task_type("binary_classification", "multiclass") is False

    def test_none_record_returns_false(self):
        assert matches_task_type(None, "regression") is False

    def test_case_insensitive_filter(self):
        assert matches_task_type("regression", "Regression") is True

    def test_case_insensitive_record(self):
        assert matches_task_type("REGRESSION", "reg") is True

    def test_regression_does_not_match_clf(self):
        assert matches_task_type("regression", "clf") is False

    def test_unknown_filter_falls_back_to_exact(self):
        """Unknown alias falls back to case-insensitive exact match."""
        assert matches_task_type("custom_type", "custom_type") is True
        assert matches_task_type("custom_type", "other") is False


class TestResolveTaskTypeSql:
    """Test SQL fragment generation for task_type filters."""

    def test_regression(self):
        sql, params = resolve_task_type_sql("regression")
        assert sql == "task_type = ?"
        assert params == ["regression"]

    def test_reg_alias(self):
        sql, params = resolve_task_type_sql("reg")
        assert sql == "task_type = ?"
        assert params == ["regression"]

    def test_classification_uses_like(self):
        sql, params = resolve_task_type_sql("classification")
        assert sql == "task_type LIKE ?"
        assert params == ["%classification%"]

    def test_clf_uses_like(self):
        sql, params = resolve_task_type_sql("clf")
        assert sql == "task_type LIKE ?"
        assert params == ["%classification%"]

    def test_binary(self):
        sql, params = resolve_task_type_sql("binary")
        assert sql == "task_type = ?"
        assert params == ["binary_classification"]

    def test_multiclass(self):
        sql, params = resolve_task_type_sql("multiclass")
        assert sql == "task_type = ?"
        assert params == ["multiclass_classification"]

    def test_unknown_filter_passthrough(self):
        """Unknown alias passes the raw value through."""
        sql, params = resolve_task_type_sql("custom_type")
        assert sql == "task_type = ?"
        assert params == ["custom_type"]
