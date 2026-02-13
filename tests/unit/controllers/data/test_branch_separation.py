"""
Unit tests for separation branches in BranchController.

Tests the new separation branch modes:
- by_tag: Branch by tag values
- by_metadata: Branch by metadata column
- by_filter: Branch by filter result
- by_source: Branch by data source
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from nirs4all.controllers.data.branch import (
    BranchController,
    SEPARATION_KEYWORDS,
)
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestBranchModeDetection:
    """Test branch mode detection."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_list_is_duplication(self, controller):
        """List syntax should be detected as duplication."""
        raw_def = [[Mock()], [Mock()]]
        assert controller._detect_branch_mode(raw_def) == "duplication"

    def test_dict_without_separation_keys_is_duplication(self, controller):
        """Dict without separation keywords is duplication."""
        raw_def = {"snv": [Mock()], "msc": [Mock()]}
        assert controller._detect_branch_mode(raw_def) == "duplication"

    def test_by_tag_is_separation(self, controller):
        """by_tag keyword triggers separation mode."""
        raw_def = {"by_tag": "outlier", "steps": []}
        assert controller._detect_branch_mode(raw_def) == "separation"

    def test_by_metadata_is_separation(self, controller):
        """by_metadata keyword triggers separation mode."""
        raw_def = {"by_metadata": "site", "steps": []}
        assert controller._detect_branch_mode(raw_def) == "separation"

    def test_by_filter_is_separation(self, controller):
        """by_filter keyword triggers separation mode."""
        raw_def = {"by_filter": Mock(), "steps": []}
        assert controller._detect_branch_mode(raw_def) == "separation"

    def test_by_source_is_separation(self, controller):
        """by_source keyword triggers separation mode."""
        raw_def = {"by_source": True, "steps": {}}
        assert controller._detect_branch_mode(raw_def) == "separation"

    def test_legacy_by_patterns_raise_error(self, controller):
        """Legacy 'by' patterns should raise clear error."""
        legacy_patterns = [
            {"by": "outlier_excluder"},
            {"by": "sample_partitioner"},
            {"by": "metadata_partitioner"},
        ]

        for raw_def in legacy_patterns:
            with pytest.raises(ValueError, match="no longer supported"):
                controller._detect_branch_mode(raw_def)


class TestSeparationKeywordsConstant:
    """Test the SEPARATION_KEYWORDS constant."""

    def test_contains_expected_keywords(self):
        """Verify all separation keywords are defined."""
        expected = {"by_tag", "by_metadata", "by_filter", "by_source"}
        assert SEPARATION_KEYWORDS == expected

    def test_is_frozen_set(self):
        """SEPARATION_KEYWORDS should be a set for O(1) lookup."""
        assert isinstance(SEPARATION_KEYWORDS, set)


class TestBranchControllerHelpers:
    """Test helper methods in BranchController."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_get_source_names_from_dataset(self, controller):
        """Test source name extraction from dataset."""
        mock_dataset = Mock()
        mock_dataset.source_name = Mock(side_effect=["NIR", "markers"])

        names = controller._get_source_names(mock_dataset, 2)
        assert names == ["NIR", "markers"]

    def test_get_source_names_fallback(self, controller):
        """Test fallback when source_name not available."""
        mock_dataset = Mock(spec=[])  # No source_name method

        names = controller._get_source_names(mock_dataset, 3)
        assert names == ["source_0", "source_1", "source_2"]

    def test_get_step_names_list(self, controller):
        """Test step name extraction from list."""
        class FakeStep:
            pass

        steps = [FakeStep(), FakeStep()]
        names = controller._get_step_names(steps)
        assert names == "FakeStep > FakeStep"

    def test_get_step_names_empty(self, controller):
        """Test step name extraction from empty list."""
        names = controller._get_step_names([])
        assert names == ""

    def test_get_step_names_dict_steps(self, controller):
        """Test step name extraction from dict steps.

        _get_step_names recurses into dict values for known keywords
        (model, preprocessing), extracting the wrapped operator's class name.
        """
        steps = [{"model": Mock()}, {"preprocessing": Mock()}]
        names = controller._get_step_names(steps)
        # Dict steps now recurse into model/preprocessing values
        assert "Mock" in names


class TestMultiplyBranchContexts:
    """Test branch context multiplication for nested branching."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_multiply_creates_cartesian_product(self, controller):
        """Test that multiply creates all combinations."""
        # Create mock contexts
        def make_context(branch_id, branch_name):
            ctx = Mock()
            ctx.selector = Mock()
            ctx.selector.branch_path = [branch_id]
            ctx.selector.with_branch = Mock(return_value=Mock())
            ctx.copy = Mock(return_value=Mock(selector=Mock(
                branch_path=[branch_id],
                with_branch=Mock(return_value=Mock())
            )))
            return ctx

        existing = [
            {"branch_id": 0, "name": "A", "context": make_context(0, "A")},
            {"branch_id": 1, "name": "B", "context": make_context(1, "B")},
        ]

        new = [
            {"branch_id": 0, "name": "X", "context": make_context(0, "X")},
            {"branch_id": 1, "name": "Y", "context": make_context(1, "Y")},
        ]

        result = controller._multiply_branch_contexts(existing, new)

        # Should have 2 x 2 = 4 branches
        assert len(result) == 4

        # Check names are combined
        names = {r["name"] for r in result}
        assert names == {"A_X", "A_Y", "B_X", "B_Y"}

        # Check flattened IDs
        ids = [r["branch_id"] for r in result]
        assert ids == [0, 1, 2, 3]


class TestExtractSubstepInfo:
    """Test substep info extraction for trace recording."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_extract_model_step(self, controller):
        """Test extraction from model step."""
        class FakeModel:
            pass

        step = {"model": FakeModel()}
        op_type, op_class = controller._extract_substep_info(step)

        assert op_type == "model"
        assert op_class == "FakeModel"

    def test_extract_preprocessing_step(self, controller):
        """Test extraction from preprocessing step."""
        class SNV:
            pass

        step = {"preprocessing": SNV()}
        op_type, op_class = controller._extract_substep_info(step)

        assert op_type == "preprocessing"
        assert op_class == "SNV"

    def test_extract_tag_step(self, controller):
        """Test extraction from tag step."""
        class YOutlierFilter:
            pass

        step = {"tag": YOutlierFilter()}
        op_type, op_class = controller._extract_substep_info(step)

        assert op_type == "tag"
        assert op_class == "YOutlierFilter"

    def test_extract_exclude_step(self, controller):
        """Test extraction from exclude step."""
        class HighLeverageFilter:
            pass

        step = {"exclude": HighLeverageFilter()}
        op_type, op_class = controller._extract_substep_info(step)

        assert op_type == "exclude"
        assert op_class == "HighLeverageFilter"

    def test_extract_class_instance(self, controller):
        """Test extraction from raw class instance."""
        class PLSRegression:
            pass

        op_type, op_class = controller._extract_substep_info(PLSRegression())

        assert op_type == "transform"
        assert op_class == "PLSRegression"

    def test_extract_class_type(self, controller):
        """Test extraction from class type (not instance)."""
        class MyTransformer:
            pass

        op_type, op_class = controller._extract_substep_info(MyTransformer)

        assert op_type == "transform"
        assert op_class == "MyTransformer"

    def test_extract_dict_with_class_key(self, controller):
        """Test extraction from serialized dict format."""
        step = {"class": "sklearn.preprocessing.StandardScaler"}
        op_type, op_class = controller._extract_substep_info(step)

        assert op_type == "transform"
        assert op_class == "StandardScaler"


class TestGetOperatorClassName:
    """Test operator class name extraction."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_none_returns_none(self, controller):
        """Test None input."""
        assert controller._get_operator_class_name(None) == "None"

    def test_empty_list_returns_empty(self, controller):
        """Test empty list input."""
        assert controller._get_operator_class_name([]) == "Empty"

    def test_single_item_list(self, controller):
        """Test single item list delegates to item."""
        class MyOp:
            pass

        result = controller._get_operator_class_name([MyOp()])
        assert result == "MyOp"

    def test_multi_item_list_joins_names(self, controller):
        """Test multi-item list joins names."""
        class Op1:
            pass
        class Op2:
            pass
        class Op3:
            pass

        result = controller._get_operator_class_name([Op1(), Op2(), Op3()])
        assert result == "Op1, Op2, Op3"

    def test_long_list_truncates(self, controller):
        """Test long list truncates with count."""
        ops = [Mock() for _ in range(5)]
        for i, op in enumerate(ops):
            type(op).__name__ = f"Op{i}"

        result = controller._get_operator_class_name(ops)
        assert "..." in result
        assert "+2" in result

    def test_string_with_dot(self, controller):
        """Test string with module path."""
        result = controller._get_operator_class_name("sklearn.preprocessing.StandardScaler")
        assert result == "StandardScaler"

    def test_string_without_dot(self, controller):
        """Test simple string."""
        result = controller._get_operator_class_name("MyOperator")
        assert result == "MyOperator"

    def test_dict_with_class_key(self, controller):
        """Test dict with 'class' key."""
        result = controller._get_operator_class_name({
            "class": "nirs4all.operators.SNV"
        })
        assert result == "SNV"

    def test_dict_without_class_key(self, controller):
        """Test dict without 'class' key."""
        result = controller._get_operator_class_name({"some": "config"})
        assert result == "Config"

    def test_class_type(self, controller):
        """Test class type (not instance)."""
        class MyClass:
            pass

        result = controller._get_operator_class_name(MyClass)
        assert result == "MyClass"

    def test_instance(self, controller):
        """Test class instance."""
        class MyInstance:
            pass

        result = controller._get_operator_class_name(MyInstance())
        assert result == "MyInstance"
