"""Unit tests for OptunaManager grid suitability (ISSUE-17 fix)."""

import pytest

from nirs4all.optimization.optuna import OptunaManager


class TestGridSearchSuitability:
    """Tests for _is_grid_search_suitable method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_list_params_are_grid_suitable(self, manager):
        params = {"model_params": {"n_components": [1, 2, 3, 4, 5]}}
        assert manager._is_grid_search_suitable(params) is True

    def test_tuple_range_not_grid_suitable(self, manager):
        params = {"model_params": {"n_components": ("int", 1, 30)}}
        assert manager._is_grid_search_suitable(params) is False

    def test_dict_categorical_is_grid_suitable(self, manager):
        """ISSUE-17: dict with type='categorical' should be grid-compatible."""
        params = {"model_params": {
            "method": {"type": "categorical", "choices": ["a", "b", "c"]},
        }}
        assert manager._is_grid_search_suitable(params) is True

    def test_dict_int_not_grid_suitable(self, manager):
        params = {"model_params": {
            "n_components": {"type": "int", "min": 1, "max": 30},
        }}
        assert manager._is_grid_search_suitable(params) is False

    def test_mixed_list_and_dict_categorical_grid_suitable(self, manager):
        params = {"model_params": {
            "n_components": [1, 5, 10],
            "method": {"type": "categorical", "choices": ["a", "b"]},
        }}
        assert manager._is_grid_search_suitable(params) is True

    def test_mixed_list_and_tuple_not_grid_suitable(self, manager):
        params = {"model_params": {
            "n_components": [1, 5, 10],
            "alpha": ("float", 0.01, 1.0),
        }}
        assert manager._is_grid_search_suitable(params) is False

    def test_empty_params_not_grid_suitable(self, manager):
        params = {"model_params": {}}
        assert manager._is_grid_search_suitable(params) is False

    def test_range_list_disguised_not_grid_suitable(self, manager):
        """A list like ['int', 1, 30] is a range spec, not categorical."""
        params = {"model_params": {"n_components": ["int", 1, 30]}}
        assert manager._is_grid_search_suitable(params) is False

class TestCreateGridSearchSpace:
    """Tests for _create_grid_search_space method."""

    @pytest.fixture
    def manager(self):
        return OptunaManager()

    def test_list_params_in_search_space(self, manager):
        params = {"model_params": {"n_components": [1, 5, 10]}}
        space = manager._create_grid_search_space(params)
        assert space == {"n_components": [1, 5, 10]}

    def test_dict_categorical_in_search_space(self, manager):
        """ISSUE-17: dict-categorical should be expanded into search space."""
        params = {"model_params": {
            "method": {"type": "categorical", "choices": ["a", "b", "c"]},
        }}
        space = manager._create_grid_search_space(params)
        assert space == {"method": ["a", "b", "c"]}

    def test_dict_categorical_with_values_key(self, manager):
        params = {"model_params": {
            "method": {"type": "categorical", "values": [1, 2, 3]},
        }}
        space = manager._create_grid_search_space(params)
        assert space == {"method": [1, 2, 3]}
