"""
Unit tests for Phase 7.1: Multi-Level Stacking.

Tests cover:
- StackingLevel enum functionality
- ModelLevelInfo dataclass
- LevelValidationResult dataclass
- MultiLevelValidator circular dependency detection
- MultiLevelValidator level consistency validation
- Level detection from source models
- max_level enforcement
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional

from nirs4all.controllers.models.stacking.multilevel import (
    MultiLevelValidator,
    ModelLevelInfo,
    LevelValidationResult,
    validate_multi_level_stacking,
    detect_stacking_level,
)
from nirs4all.controllers.models.stacking.exceptions import (
    CircularDependencyError,
    MaxStackingLevelExceededError,
    InconsistentLevelError,
)
from nirs4all.operators.models.meta import StackingLevel, StackingConfig


class TestStackingLevel:
    """Test StackingLevel enum."""

    def test_auto_level(self):
        """AUTO should be the default for auto-detection."""
        assert StackingLevel.AUTO.value == "auto"

    def test_level_1(self):
        """LEVEL_1 should have value 1."""
        assert StackingLevel.LEVEL_1.value == 1

    def test_level_2(self):
        """LEVEL_2 should have value 2."""
        assert StackingLevel.LEVEL_2.value == 2

    def test_level_3(self):
        """LEVEL_3 should have value 3."""
        assert StackingLevel.LEVEL_3.value == 3

    def test_stacking_config_level_default(self):
        """Default StackingConfig level should be AUTO."""
        config = StackingConfig()
        assert config.level == StackingLevel.AUTO

    def test_stacking_config_explicit_level(self):
        """StackingConfig should accept explicit level."""
        config = StackingConfig(level=StackingLevel.LEVEL_2)
        assert config.level == StackingLevel.LEVEL_2

    def test_stacking_config_max_level_default(self):
        """Default max_level should be 3."""
        config = StackingConfig()
        assert config.max_level == 3

    def test_stacking_config_max_level_validation(self):
        """max_level should be validated (1-10)."""
        with pytest.raises(ValueError, match="max_level must be between"):
            StackingConfig(max_level=0)

        with pytest.raises(ValueError, match="max_level must be between"):
            StackingConfig(max_level=11)


class TestModelLevelInfo:
    """Test ModelLevelInfo dataclass."""

    def test_base_model_info(self):
        """Base model should have level 0 and not be a meta-model."""
        info = ModelLevelInfo(
            model_name="PLS",
            level=0,
            is_meta_model=False,
            source_models=[]
        )
        assert info.level == 0
        assert info.is_meta_model is False
        assert info.source_models == []

    def test_meta_model_info(self):
        """Meta-model should have level >= 1 and list source models."""
        info = ModelLevelInfo(
            model_name="MetaModel_Ridge",
            level=1,
            is_meta_model=True,
            source_models=["PLS", "RandomForest"]
        )
        assert info.level == 1
        assert info.is_meta_model is True
        assert info.source_models == ["PLS", "RandomForest"]

    def test_level_2_meta_model(self):
        """Level 2 meta-model should have level 2."""
        info = ModelLevelInfo(
            model_name="SuperMeta",
            level=2,
            is_meta_model=True,
            source_models=["PLS", "MetaModel_Ridge"]
        )
        assert info.level == 2
        assert "MetaModel_Ridge" in info.source_models


class TestLevelValidationResult:
    """Test LevelValidationResult dataclass."""

    def test_valid_result(self):
        """Valid result should have is_valid=True and no errors."""
        result = LevelValidationResult(
            is_valid=True,
            detected_level=1,
            errors=[],
            warnings=[],
            source_levels={"PLS": 0, "RF": 0},
            circular_dependencies=[]
        )
        assert result.is_valid is True
        assert result.detected_level == 1
        assert len(result.errors) == 0

    def test_invalid_result_with_circular_deps(self):
        """Invalid result with circular dependencies."""
        result = LevelValidationResult(
            is_valid=False,
            detected_level=1,
            errors=["Circular dependency detected"],
            warnings=[],
            source_levels={"A": 1, "B": 1},
            circular_dependencies=[["A", "B", "A"]]
        )
        assert result.is_valid is False
        assert result.circular_dependencies == [["A", "B", "A"]]

    def test_add_error(self):
        """add_error should append error and mark invalid."""
        result = LevelValidationResult()
        result.add_error("Test error")
        assert "Test error" in result.errors
        assert result.is_valid is False

    def test_add_warning(self):
        """add_warning should append warning."""
        result = LevelValidationResult()
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings


class TestMultiLevelValidator:
    """Test MultiLevelValidator class."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = 5
        return context

    @pytest.fixture
    def mock_candidates(self):
        """Create mock model candidates."""
        c1 = Mock()
        c1.model_name = "PLS"
        c1.model_classname = "PLSRegression"
        c1.step_idx = 1
        c1.branch_id = None

        c2 = Mock()
        c2.model_name = "RF"
        c2.model_classname = "RandomForestRegressor"
        c2.step_idx = 2
        c2.branch_id = None

        return [c1, c2]

    def test_validator_initialization(self, mock_prediction_store):
        """Validator should initialize with prediction_store."""
        validator = MultiLevelValidator(
            prediction_store=mock_prediction_store,
            max_level=3
        )
        assert validator.max_level == 3

    def test_validate_base_models(
        self, mock_prediction_store, mock_context, mock_candidates
    ):
        """Validating base models should succeed with level 1."""
        mock_prediction_store.filter_predictions.return_value = [
            {'model_name': 'PLS', 'model_classname': 'PLSRegression', 'step_idx': 1},
            {'model_name': 'RF', 'model_classname': 'RandomForestRegressor', 'step_idx': 2},
        ]

        validator = MultiLevelValidator(
            prediction_store=mock_prediction_store,
            max_level=3
        )

        result = validator.validate_sources(
            meta_model_name="TestMeta",
            source_candidates=mock_candidates,
            context=mock_context,
            allow_meta_sources=True
        )

        assert result.is_valid is True
        assert result.detected_level == 1

    def test_detect_level_base_models(
        self, mock_prediction_store, mock_context, mock_candidates
    ):
        """Level detection for base models should return 1."""
        mock_prediction_store.filter_predictions.return_value = [
            {'model_name': 'PLS', 'model_classname': 'PLSRegression', 'step_idx': 1},
        ]

        validator = MultiLevelValidator(
            prediction_store=mock_prediction_store,
            max_level=3
        )

        level = validator.detect_level(mock_candidates, mock_context)
        assert level == 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def mock_prediction_store(self):
        """Create a mock prediction store."""
        store = Mock()
        store.filter_predictions = Mock(return_value=[])
        return store

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock()
        context.selector = Mock()
        context.selector.branch_id = None
        context.state = Mock()
        context.state.step_number = 5
        return context

    @pytest.fixture
    def mock_candidates(self):
        """Create mock model candidates."""
        c1 = Mock()
        c1.model_name = "PLS"
        c1.model_classname = "PLSRegression"
        c1.step_idx = 1
        c1.branch_id = None
        return [c1]

    def test_validate_multi_level_stacking_function(
        self, mock_prediction_store, mock_context, mock_candidates
    ):
        """Test the convenience function for multi-level validation."""
        mock_prediction_store.filter_predictions.return_value = [
            {'model_name': 'PLS', 'model_classname': 'PLSRegression', 'step_idx': 1},
        ]

        result = validate_multi_level_stacking(
            prediction_store=mock_prediction_store,
            meta_model_name="TestMeta",
            source_candidates=mock_candidates,
            context=mock_context
        )

        assert isinstance(result, LevelValidationResult)

    def test_detect_stacking_level_function(
        self, mock_prediction_store, mock_context, mock_candidates
    ):
        """Test the convenience function for level detection."""
        mock_prediction_store.filter_predictions.return_value = [
            {'model_name': 'PLS', 'model_classname': 'PLSRegression', 'step_idx': 1},
        ]

        level = detect_stacking_level(
            prediction_store=mock_prediction_store,
            source_candidates=mock_candidates,
            context=mock_context
        )

        assert isinstance(level, int)
        assert level >= 1


class TestMultiLevelExceptions:
    """Test multi-level stacking exceptions."""

    def test_circular_dependency_error(self):
        """CircularDependencyError should contain dependency chain."""
        with pytest.raises(CircularDependencyError) as exc_info:
            raise CircularDependencyError(
                source_model="ModelB",
                meta_model="ModelA",
                dependency_chain=["ModelA", "ModelB", "ModelA"]
            )

        assert "circular" in str(exc_info.value).lower()
        assert exc_info.value.dependency_chain == ["ModelA", "ModelB", "ModelA"]

    def test_max_stacking_level_exceeded_error(self):
        """MaxStackingLevelExceededError should contain level info."""
        with pytest.raises(MaxStackingLevelExceededError) as exc_info:
            raise MaxStackingLevelExceededError(
                current_level=4,
                max_level=3,
                source_models=["PLS", "RF"]
            )

        assert "4" in str(exc_info.value)
        assert "3" in str(exc_info.value)
        assert exc_info.value.current_level == 4
        assert exc_info.value.max_level == 3

    def test_inconsistent_level_error(self):
        """InconsistentLevelError should contain model and level info."""
        with pytest.raises(InconsistentLevelError) as exc_info:
            raise InconsistentLevelError(
                expected_levels=[0, 1],
                found_levels={"PLS": 0, "MetaModel_Ridge": 2},
                problematic_models=["MetaModel_Ridge"]
            )

        assert "MetaModel_Ridge" in str(exc_info.value)
