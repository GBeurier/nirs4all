"""
Unit tests for BranchController.

Tests the core branching functionality including:
- Branch context creation
- Context isolation between branches
- Post-branch step execution on all branches
- Branch metadata tracking
- Generator syntax integration (Phase 3)
"""

import pytest
import numpy as np
from copy import deepcopy
from unittest.mock import Mock, MagicMock, patch

from nirs4all.controllers.data.branch import BranchController
from nirs4all.pipeline.config.context import (
    DataSelector,
    PipelineState,
    StepMetadata,
    ExecutionContext,
    RuntimeContext
)
from nirs4all.pipeline.execution.result import StepOutput, StepResult
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class TestDataSelectorBranchFields:
    """Test branch_id and branch_name fields in DataSelector."""

    def test_dataselector_default_branch_values(self):
        """DataSelector should have None for branch fields by default."""
        selector = DataSelector()
        assert selector.branch_id is None
        assert selector.branch_name is None

    def test_dataselector_with_branch(self):
        """DataSelector.with_branch should create copy with updated fields."""
        selector = DataSelector(partition="train")
        new_selector = selector.with_branch(branch_id=0, branch_name="snv_pca")

        # Original should be unchanged
        assert selector.branch_id is None
        assert selector.branch_name is None

        # New selector should have branch info
        assert new_selector.branch_id == 0
        assert new_selector.branch_name == "snv_pca"

        # Other fields should be preserved
        assert new_selector.partition == "train"

    def test_dataselector_copy_includes_branch(self):
        """DataSelector.copy() should include branch fields."""
        selector = DataSelector(partition="test", branch_id=1, branch_name="msc_d1")
        copied = selector.copy()

        assert copied.branch_id == 1
        assert copied.branch_name == "msc_d1"
        assert copied.partition == "test"


class TestExecutionContextBranch:
    """Test branch methods in ExecutionContext."""

    def test_context_with_branch(self):
        """ExecutionContext.with_branch should update selector."""
        context = ExecutionContext()
        new_context = context.with_branch(branch_id=2, branch_name="derivative")

        # Original unchanged
        assert context.selector.branch_id is None

        # New context has branch info
        assert new_context.selector.branch_id == 2
        assert new_context.selector.branch_name == "derivative"

    def test_context_copy_preserves_branch(self):
        """ExecutionContext.copy() should preserve branch info."""
        context = ExecutionContext()
        context.selector.branch_id = 3
        context.selector.branch_name = "test_branch"

        copied = context.copy()
        assert copied.selector.branch_id == 3
        assert copied.selector.branch_name == "test_branch"


class TestBranchControllerMatches:
    """Test BranchController.matches() method."""

    def test_matches_branch_keyword(self):
        """Should match when keyword is 'branch'."""
        step = {"branch": [[Mock()], [Mock()]]}
        assert BranchController.matches(step, None, "branch") is True

    def test_not_matches_other_keywords(self):
        """Should not match other keywords."""
        assert BranchController.matches({}, None, "model") is False
        assert BranchController.matches({}, None, "preprocessing") is False
        assert BranchController.matches({}, None, "feature_augmentation") is False


class TestBranchControllerParsing:
    """Test branch definition parsing."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    @pytest.fixture
    def mock_step_info(self):
        """Create mock ParsedStep."""
        return ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={},
            metadata={}
        )

    def test_parse_list_of_lists(self, controller, mock_step_info):
        """Parse [[steps], [steps]] format."""
        mock_step_info.original_step = {
            "branch": [
                ["step1", "step2"],
                ["step3"]
            ]
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 2
        assert result[0]["name"] == "branch_0"
        assert result[0]["steps"] == ["step1", "step2"]
        assert result[1]["name"] == "branch_1"
        assert result[1]["steps"] == ["step3"]

    def test_parse_named_dict(self, controller, mock_step_info):
        """Parse {"name": [steps]} format."""
        mock_step_info.original_step = {
            "branch": {
                "snv_pca": ["snv", "pca"],
                "msc": ["msc"]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "snv_pca" in names
        assert "msc" in names

    def test_parse_empty_branch(self, controller, mock_step_info):
        """Empty branch definition returns empty list."""
        mock_step_info.original_step = {"branch": []}

        result = controller._parse_branch_definitions(mock_step_info)
        assert result == []


class TestBranchControllerExecution:
    """Test BranchController.execute() method."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    @pytest.fixture
    def mock_dataset(self):
        dataset = Mock()
        dataset.name = "test_dataset"
        return dataset

    @pytest.fixture
    def mock_context(self):
        return ExecutionContext(
            selector=DataSelector(partition="train", processing=[["raw"]]),
            state=PipelineState(step_number=1),
            metadata=StepMetadata()
        )

    @pytest.fixture
    def mock_runtime_context(self):
        runtime = RuntimeContext()
        runtime.substep_number = 0

        # Create mock step_runner that returns a StepResult
        mock_runner = Mock()

        def execute_side_effect(step, dataset, context, runtime_context, **kwargs):
            # Return a StepResult with the context
            return StepResult(
                updated_context=context,
                artifacts=[]
            )

        mock_runner.execute = Mock(side_effect=execute_side_effect)
        runtime.step_runner = mock_runner

        return runtime

    def test_execute_creates_branch_contexts(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute should create branch contexts in custom dict."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": [
                    ["step1"],
                    ["step2"]
                ]
            },
            metadata={}
        )

        result_context, output = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        # Should have branch_contexts in custom dict
        assert "branch_contexts" in result_context.custom
        assert len(result_context.custom["branch_contexts"]) == 2

        # Check branch info
        branches = result_context.custom["branch_contexts"]
        assert branches[0]["branch_id"] == 0
        assert branches[0]["name"] == "branch_0"
        assert branches[1]["branch_id"] == 1
        assert branches[1]["name"] == "branch_1"

    def test_branch_contexts_have_correct_selector(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Each branch context should have correct branch_id in selector."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": {
                    "alpha": ["step_a"],
                    "beta": ["step_b"]
                }
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]
        for branch in branches:
            ctx = branch["context"]
            assert ctx.selector.branch_id == branch["branch_id"]
            assert ctx.selector.branch_name == branch["name"]

    def test_branch_mode_flag_set(
        self, controller, mock_dataset, mock_context, mock_runtime_context
    ):
        """Execute should set in_branch_mode flag."""
        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={"branch": [["step1"]]},
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=mock_dataset,
            context=mock_context,
            runtime_context=mock_runtime_context,
            mode="train"
        )

        assert result_context.custom.get("in_branch_mode") is True


class TestBranchContextIsolation:
    """Test that branch contexts are properly isolated."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_branch_processing_isolation(self, controller):
        """Each branch should start with same processing, then diverge."""
        initial_processing = [["raw"]]
        context = ExecutionContext(
            selector=DataSelector(processing=deepcopy(initial_processing))
        )

        # Create mock that modifies processing differently per branch
        mock_runner = Mock()
        call_count = [0]

        def modify_processing(step, dataset, context, runtime_context, **kwargs):
            call_count[0] += 1
            # Modify processing based on step
            new_context = context.copy()
            new_context.selector.processing = [[f"modified_{call_count[0]}"]]
            return StepResult(updated_context=new_context, artifacts=[])

        mock_runner.execute = modify_processing

        runtime = RuntimeContext()
        runtime.step_runner = mock_runner
        runtime.substep_number = 0

        step_info = ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={
                "branch": [
                    ["step1"],
                    ["step2"]
                ]
            },
            metadata={}
        )

        result_context, _ = controller.execute(
            step_info=step_info,
            dataset=Mock(name="test"),
            context=context,
            runtime_context=runtime,
            mode="train"
        )

        branches = result_context.custom["branch_contexts"]

        # Each branch should have different processing
        assert branches[0]["context"].selector.processing != branches[1]["context"].selector.processing


class TestBranchControllerSupportsPredict:
    """Test that BranchController works in prediction mode."""

    def test_supports_prediction_mode(self):
        """Controller should support prediction mode."""
        assert BranchController.supports_prediction_mode() is True

    def test_use_multi_source(self):
        """Controller should support multi-source datasets."""
        assert BranchController.use_multi_source() is True


class TestBranchMultiplication:
    """Test nested branch context multiplication."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_multiply_branch_contexts(self, controller):
        """Test multiplication of existing and new branch contexts."""
        # Existing branches
        existing = [
            {"branch_id": 0, "name": "A", "context": Mock()},
            {"branch_id": 1, "name": "B", "context": Mock()},
        ]

        # New branches to multiply with
        new = [
            {"branch_id": 0, "name": "X", "context": Mock()},
            {"branch_id": 1, "name": "Y", "context": Mock()},
        ]

        # Mock context copy method
        for i, item in enumerate(existing + new):
            mock_ctx = Mock()
            mock_selector = Mock()
            mock_selector.branch_id = None
            mock_selector.branch_name = None
            # branch_path should be a list or None - the code uses it with list concatenation
            mock_selector.branch_path = [i] if i < len(existing) else None
            mock_selector.with_branch = Mock(return_value=mock_selector)
            mock_ctx.selector = mock_selector
            mock_ctx.copy = Mock(return_value=mock_ctx)
            item["context"] = mock_ctx

        result = controller._multiply_branch_contexts(existing, new)

        # Should have 2 x 2 = 4 combinations
        assert len(result) == 4

        # Check names are combined
        names = [r["name"] for r in result]
        assert "A_X" in names
        assert "A_Y" in names
        assert "B_X" in names
        assert "B_Y" in names

        # Check flattened IDs are sequential
        ids = [r["branch_id"] for r in result]
        assert ids == [0, 1, 2, 3]


class TestBranchGeneratorIntegration:
    """Test generator syntax integration with branching (Phase 3)."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    @pytest.fixture
    def mock_step_info(self):
        """Create mock ParsedStep."""
        return ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={},
            metadata={}
        )

    def test_parse_or_generator_simple(self, controller, mock_step_info):
        """Parse {"_or_": [A, B, C]} format -> 3 branches."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": [
                    {"class": "nirs4all.preprocessing.SNV"},
                    {"class": "nirs4all.preprocessing.MSC"},
                    {"class": "nirs4all.preprocessing.FirstDerivative"},
                ]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 3
        # Each branch should have one step
        for branch in result:
            assert len(branch["steps"]) == 1
            assert "generator_choice" in branch

    def test_parse_or_generator_with_lists(self, controller, mock_step_info):
        """Parse {"_or_": [[A, B], [C]]} format -> 2 branches with multi-step."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": [
                    [
                        {"class": "nirs4all.preprocessing.SNV"},
                        {"class": "nirs4all.preprocessing.PCA", "params": {"n_components": 10}}
                    ],
                    [
                        {"class": "nirs4all.preprocessing.MSC"}
                    ]
                ]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 2
        # First branch has 2 steps
        assert len(result[0]["steps"]) == 2
        # Second branch has 1 step
        assert len(result[1]["steps"]) == 1

    def test_parse_range_generator(self, controller, mock_step_info):
        """Parse {"_range_": [5, 15, 5]} format -> 3 branches (5, 10, 15)."""
        mock_step_info.original_step = {
            "branch": {
                "_range_": [5, 16, 5]  # 5, 10, 15
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # _range_ [5, 16, 5] generates [5, 10, 15]
        assert len(result) == 3

    def test_parse_nested_or_in_list(self, controller, mock_step_info):
        """Parse list containing _or_ generator."""
        mock_step_info.original_step = {
            "branch": [
                {"_or_": [
                    {"class": "nirs4all.preprocessing.SNV"},
                    {"class": "nirs4all.preprocessing.MSC"}
                ]},
                [{"class": "nirs4all.preprocessing.FirstDerivative"}]
            ]
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # First item expands to 2 branches, second is 1 branch
        assert len(result) == 3

    def test_branch_names_from_generator(self, controller, mock_step_info):
        """Branch names should be derived from step class names."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": [
                    {"class": "nirs4all.preprocessing.SNV"},
                    {"class": "nirs4all.preprocessing.MSC"},
                ]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 2
        names = [r["name"] for r in result]
        # Names should include class short names
        assert any("SNV" in name for name in names)
        assert any("MSC" in name for name in names)

    def test_generator_choice_preserved(self, controller, mock_step_info):
        """Generator choice should be stored in branch definition."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": ["optionA", "optionB", "optionC"]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        assert len(result) == 3
        choices = [r.get("generator_choice") for r in result]
        assert "optionA" in choices
        assert "optionB" in choices
        assert "optionC" in choices

    def test_mixed_explicit_and_generator_branches(self, controller, mock_step_info):
        """Support mixing explicit branches with generator syntax."""
        mock_step_info.original_step = {
            "branch": [
                ["explicit_step_1", "explicit_step_2"],  # Explicit branch
                {"_or_": ["gen_a", "gen_b"]}  # Generator expands to 2 branches
            ]
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # 1 explicit + 2 from generator = 3 branches
        assert len(result) == 3

    def test_generator_with_pick_modifier(self, controller, mock_step_info):
        """Test _or_ with pick modifier for combinations."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": ["A", "B", "C"],
                "pick": 2  # C(3,2) = 3 combinations
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # pick 2 from 3 = 3 combinations: [A,B], [A,C], [B,C]
        assert len(result) == 3
        for branch in result:
            # Each combination is a list of 2 items
            assert len(branch["steps"]) == 2

    def test_generator_with_count_limit(self, controller, mock_step_info):
        """Test _or_ with count to limit number of branches."""
        mock_step_info.original_step = {
            "branch": {
                "_or_": ["A", "B", "C", "D", "E"],
                "count": 2  # Limit to 2 branches
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # Should be limited to 2 branches
        assert len(result) == 2


class TestBranchGeneratorNaming:
    """Test branch name generation from different step types."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_name_from_class_dict(self, controller):
        """Extract name from {"class": "module.ClassName"} format."""
        step = {"class": "sklearn.preprocessing.StandardScaler"}
        name = controller._get_single_step_name(step)
        assert name == "StandardScaler"

    def test_name_from_string(self, controller):
        """Extract name from string step."""
        name = controller._get_single_step_name("SNV")
        assert name == "SNV"

    def test_name_from_model_dict(self, controller):
        """Extract name from model step format."""
        step = {"model": {"class": "sklearn.linear_model.Ridge"}}
        name = controller._get_single_step_name(step)
        assert name == "Ridge"

    def test_name_from_preprocessing_dict(self, controller):
        """Extract name from preprocessing step format."""
        step = {"preprocessing": {"class": "nirs4all.preprocessing.SNV"}}
        name = controller._get_single_step_name(step)
        assert name == "SNV"

    def test_name_from_explicit_name_key(self, controller):
        """Use 'name' key if present."""
        step = {"name": "my_custom_step", "class": "SomeClass"}
        name = controller._get_single_step_name(step)
        assert name == "my_custom_step"

    def test_name_from_list_of_steps(self, controller):
        """Generate combined name from list of steps."""
        steps = [
            {"class": "nirs4all.preprocessing.SNV"},
            {"class": "sklearn.decomposition.PCA"},
        ]
        name = controller._generate_step_name(steps, 0)
        assert "SNV" in name
        assert "PCA" in name

    def test_fallback_to_index(self, controller):
        """Fall back to branch_N if no name can be extracted."""
        step = None
        name = controller._generate_step_name(step, 5)
        assert name == "branch_5"


class TestBranchGeneratorExpansion:
    """Test internal generator expansion methods."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    def test_expand_generator_branches_or(self, controller):
        """_expand_generator_branches with _or_ keyword."""
        generator_node = {"_or_": ["A", "B", "C"]}
        result = controller._expand_generator_branches(generator_node)

        assert len(result) == 3
        steps_values = [r["steps"] for r in result]
        assert ["A"] in steps_values
        assert ["B"] in steps_values
        assert ["C"] in steps_values

    def test_expand_generator_branches_range(self, controller):
        """_expand_generator_branches with _range_ keyword."""
        generator_node = {"_range_": [1, 4]}  # 1, 2, 3, 4
        result = controller._expand_generator_branches(generator_node)

        assert len(result) == 4
        # Each should have numeric value as step
        values = [r["steps"][0] for r in result]
        assert values == [1, 2, 3, 4]

    def test_expand_list_with_generators(self, controller):
        """Test Cartesian product expansion of list with generators."""
        items = [
            {"_or_": ["X", "Y"]},
            "fixed",
            {"_or_": ["1", "2"]}
        ]
        result = controller._expand_list_with_generators(items)

        # 2 x 1 x 2 = 4 combinations
        assert len(result) == 4

        # Each result should have 3 items
        for r in result:
            assert len(r) == 3
            assert r[1] == "fixed"

    def test_expand_list_no_generators(self, controller):
        """List without generators should return as-is."""
        items = ["a", "b", "c"]
        result = controller._expand_list_with_generators(items)

        assert len(result) == 1
        assert result[0] == ["a", "b", "c"]

    def test_expand_list_flattens_cartesian_results(self, controller):
        """Cartesian generator producing lists should be flattened into step list."""
        items = [
            {"_cartesian_": [
                {"_or_": ["A", "B"]},
                {"_or_": ["X", "Y"]},
            ]},
            "model_step",
        ]
        result = controller._expand_list_with_generators(items)

        # 2x2 = 4 cartesian combos, each combined with model_step
        assert len(result) == 4

        # Each result should be FLAT: [preproc1, preproc2, model_step]
        for r in result:
            assert len(r) == 3
            assert r[-1] == "model_step"
            assert not isinstance(r[0], list)  # should NOT be nested

    def test_expand_list_cartesian_with_multiple_non_generators(self, controller):
        """Cartesian with multiple trailing non-generator steps."""
        items = [
            {"_or_": [["A1", "A2"], ["B1"]]},
            "model1",
            "model2",
        ]
        result = controller._expand_list_with_generators(items)

        # 2 options from _or_ × 1 × 1 = 2 combos
        assert len(result) == 2

        # First option [A1, A2] should be flattened: [A1, A2, model1, model2]
        assert result[0] == ["A1", "A2", "model1", "model2"]
        # Second option [B1] should be flattened: [B1, model1, model2]
        assert result[1] == ["B1", "model1", "model2"]


class TestBranchGeneratorInNamedBranches:
    """Test generator expansion inside named branch step lists."""

    @pytest.fixture
    def controller(self):
        return BranchController()

    @pytest.fixture
    def mock_step_info(self):
        return ParsedStep(
            operator=None,
            keyword="branch",
            step_type=StepType.WORKFLOW,
            original_step={},
            metadata={}
        )

    def test_named_branch_with_or_generator_in_steps(self, controller, mock_step_info):
        """Named branch with _or_ generator inside step list."""
        mock_step_info.original_step = {
            "branch": {
                "my_branch": [
                    {"_or_": ["SNV", "MSC"]},
                    "model_step",
                ]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # Should expand to 2 branches
        assert len(result) == 2
        for branch in result:
            assert branch["name"].startswith("my_branch_")
            # Each branch: [preprocessing, model_step]
            assert len(branch["steps"]) == 2
            assert branch["steps"][-1] == "model_step"

    def test_named_branch_with_cartesian_generator_in_steps(self, controller, mock_step_info):
        """Named branch with _cartesian_ generator + model steps."""
        mock_step_info.original_step = {
            "branch": {
                "linear": [
                    {"_cartesian_": [
                        {"_or_": ["A", "B"]},
                        {"_or_": ["X", "Y"]},
                    ]},
                    "PLS",
                    "Ridge",
                ]
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # _cartesian_ produces 2×2=4 combos, combined with PLS + Ridge
        assert len(result) == 4
        for branch in result:
            assert branch["name"].startswith("linear_")
            # Each branch: [preproc1, preproc2, PLS, Ridge]
            assert len(branch["steps"]) == 4
            assert branch["steps"][-2] == "PLS"
            assert branch["steps"][-1] == "Ridge"

    def test_multiple_named_branches_independent_expansion(self, controller, mock_step_info):
        """Multiple named branches with generators expand independently (N+M, not N×M)."""
        mock_step_info.original_step = {
            "branch": {
                "branch_A": [
                    {"_or_": ["A1", "A2", "A3"]},
                    "model_A",
                ],
                "branch_B": [
                    {"_or_": ["B1", "B2"]},
                    "model_B",
                ],
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # 3 from A + 2 from B = 5 (NOT 3×2=6)
        assert len(result) == 5

        # Check A branches
        a_branches = [b for b in result if b["name"].startswith("branch_A_")]
        assert len(a_branches) == 3
        for b in a_branches:
            assert b["steps"][-1] == "model_A"

        # Check B branches
        b_branches = [b for b in result if b["name"].startswith("branch_B_")]
        assert len(b_branches) == 2
        for b in b_branches:
            assert b["steps"][-1] == "model_B"

    def test_mixed_named_branches_with_and_without_generators(self, controller, mock_step_info):
        """Mix of named branches: some with generators, some without."""
        mock_step_info.original_step = {
            "branch": {
                "static": ["step1", "step2"],
                "dynamic": [
                    {"_or_": ["X", "Y"]},
                    "model",
                ],
            }
        }

        result = controller._parse_branch_definitions(mock_step_info)

        # 1 static + 2 dynamic = 3
        assert len(result) == 3

        static = [b for b in result if b["name"] == "static"]
        assert len(static) == 1
        assert static[0]["steps"] == ["step1", "step2"]

        dynamic = [b for b in result if b["name"].startswith("dynamic_")]
        assert len(dynamic) == 2
