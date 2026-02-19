"""
Tests for FoldFileLoaderController and FoldFileParser.

These tests verify:
- FoldFileParser can parse various fold file formats
- FoldFileLoaderController correctly matches pipeline steps
- FoldFileLoaderController loads folds and sets them on dataset
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nirs4all.controllers.splitters.fold_file_loader import (
    FoldFileLoaderController,
    FoldFileParser,
)
from nirs4all.data.dataset import SpectroDataset


class TestFoldFileParser:
    """Tests for FoldFileParser class."""

    @pytest.fixture
    def parser(self):
        """Create a FoldFileParser instance."""
        return FoldFileParser()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files."""
        return tmp_path

    def test_parse_csv_nirs4all_format(self, parser, temp_dir):
        """Test parsing nirs4all CSV format (fold_0, fold_1, ... columns)."""
        csv_content = """fold_0,fold_1
0,5
1,6
2,7
3,8
4,9
"""
        csv_path = temp_dir / "folds.csv"
        csv_path.write_text(csv_content)

        folds = parser.parse(csv_path)

        assert len(folds) == 2
        # Fold 0: train=[0,1,2,3,4], val=[5,6,7,8,9]
        assert folds[0][0] == [0, 1, 2, 3, 4]
        assert folds[0][1] == [5, 6, 7, 8, 9]
        # Fold 1: train=[5,6,7,8,9], val=[0,1,2,3,4]
        assert folds[1][0] == [5, 6, 7, 8, 9]
        assert folds[1][1] == [0, 1, 2, 3, 4]

    def test_parse_csv_nirs4all_format_unequal_folds(self, parser, temp_dir):
        """Test parsing CSV with unequal fold sizes."""
        csv_content = """fold_0,fold_1,fold_2
0,3,6
1,4,7
2,5,
"""
        csv_path = temp_dir / "folds.csv"
        csv_path.write_text(csv_content)

        folds = parser.parse(csv_path)

        assert len(folds) == 3
        assert folds[0][0] == [0, 1, 2]
        assert folds[1][0] == [3, 4, 5]
        assert folds[2][0] == [6, 7]

    def test_parse_csv_assignment_format(self, parser, temp_dir):
        """Test parsing CSV with sample assignment format."""
        csv_content = """sample_id,fold
0,0
1,0
2,1
3,1
4,2
5,2
"""
        csv_path = temp_dir / "folds_assign.csv"
        csv_path.write_text(csv_content)

        folds = parser.parse(csv_path)

        assert len(folds) == 3
        # Fold 0: samples 0,1 are val, others are train
        assert sorted(folds[0][0]) == [2, 3, 4, 5]  # train
        assert sorted(folds[0][1]) == [0, 1]  # val
        # Fold 1: samples 2,3 are val
        assert sorted(folds[1][0]) == [0, 1, 4, 5]
        assert sorted(folds[1][1]) == [2, 3]
        # Fold 2: samples 4,5 are val
        assert sorted(folds[2][0]) == [0, 1, 2, 3]
        assert sorted(folds[2][1]) == [4, 5]

    def test_parse_json(self, parser, temp_dir):
        """Test parsing JSON fold file."""
        json_content = [
            {"train": [0, 1, 2], "val": [3, 4, 5]},
            {"train": [3, 4, 5], "val": [0, 1, 2]}
        ]
        json_path = temp_dir / "folds.json"
        json_path.write_text(json.dumps(json_content))

        folds = parser.parse(json_path)

        assert len(folds) == 2
        assert folds[0] == ([0, 1, 2], [3, 4, 5])
        assert folds[1] == ([3, 4, 5], [0, 1, 2])

    def test_parse_json_with_test_key(self, parser, temp_dir):
        """Test parsing JSON with 'test' key instead of 'val'."""
        json_content = [
            {"train": [0, 1, 2], "test": [3, 4, 5]}
        ]
        json_path = temp_dir / "folds.json"
        json_path.write_text(json.dumps(json_content))

        folds = parser.parse(json_path)

        assert len(folds) == 1
        assert folds[0] == ([0, 1, 2], [3, 4, 5])

    def test_parse_yaml(self, parser, temp_dir):
        """Test parsing YAML fold file."""
        pytest.importorskip("yaml")

        yaml_content = """
- train: [0, 1, 2, 3, 4]
  val: [5, 6, 7, 8, 9]
- train: [5, 6, 7, 8, 9]
  val: [0, 1, 2, 3, 4]
"""
        yaml_path = temp_dir / "folds.yaml"
        yaml_path.write_text(yaml_content)

        folds = parser.parse(yaml_path)

        assert len(folds) == 2
        assert folds[0] == ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        assert folds[1] == ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])

    def test_parse_txt(self, parser, temp_dir):
        """Test parsing TXT fold file (alternating train/val lines)."""
        txt_content = """0,1,2,3,4
5,6,7,8,9
5,6,7,8,9
0,1,2,3,4
"""
        txt_path = temp_dir / "folds.txt"
        txt_path.write_text(txt_content)

        folds = parser.parse(txt_path)

        assert len(folds) == 2
        assert folds[0] == ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        assert folds[1] == ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])

    def test_file_not_found(self, parser, temp_dir):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            parser.parse(temp_dir / "nonexistent.csv")

    def test_unsupported_extension(self, parser, temp_dir):
        """Test error for unsupported file extension."""
        bad_path = temp_dir / "folds.xyz"
        bad_path.write_text("content")

        with pytest.raises(ValueError, match="Cannot detect fold file format"):
            parser.parse(bad_path)

    def test_detect_format(self, parser):
        """Test format detection from extensions."""
        assert parser._detect_format(Path("folds.csv")) == "csv"
        assert parser._detect_format(Path("folds.json")) == "json"
        assert parser._detect_format(Path("folds.yaml")) == "yaml"
        assert parser._detect_format(Path("folds.yml")) == "yaml"
        assert parser._detect_format(Path("folds.txt")) == "txt"

class TestFoldFileLoaderController:
    """Tests for FoldFileLoaderController class."""

    @pytest.fixture
    def dataset(self):
        """Create a test dataset with samples."""
        ds = SpectroDataset("test_dataset")
        # Add 10 train samples
        X = np.random.randn(10, 100)
        ds.add_samples(X, {"partition": "train"})
        ds.add_targets(np.arange(10))
        return ds

    @pytest.fixture
    def context(self):
        """Create a mock execution context."""
        from nirs4all.pipeline.config.context import ExecutionContext
        return ExecutionContext()

    @pytest.fixture
    def runtime_context(self):
        """Create a mock runtime context."""
        mock = MagicMock()
        mock.verbose = 0
        return mock

    def test_matches_split_with_csv_path(self):
        """Test matches() returns True for split with CSV path."""
        assert FoldFileLoaderController.matches(
            step={"split": "path/to/folds.csv"},
            operator="path/to/folds.csv",
            keyword="split"
        )

    def test_matches_split_with_json_path(self):
        """Test matches() returns True for split with JSON path."""
        assert FoldFileLoaderController.matches(
            step={"split": "folds.json"},
            operator="folds.json",
            keyword="split"
        )

    def test_matches_split_with_yaml_path(self):
        """Test matches() returns True for split with YAML path."""
        assert FoldFileLoaderController.matches(
            step={"split": "folds.yaml"},
            operator="folds.yaml",
            keyword="split"
        )

    def test_not_matches_split_with_splitter_object(self):
        """Test matches() returns False for split with splitter object."""
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=5)
        assert not FoldFileLoaderController.matches(
            step={"split": splitter},
            operator=splitter,
            keyword="split"
        )

    def test_not_matches_non_split_keyword(self):
        """Test matches() returns False for non-split keywords."""
        assert not FoldFileLoaderController.matches(
            step={"model": "folds.csv"},
            operator="folds.csv",
            keyword="model"
        )

    def test_not_matches_unsupported_extension(self):
        """Test matches() returns False for unsupported file extensions."""
        assert not FoldFileLoaderController.matches(
            step={"split": "file.unknown"},
            operator="file.unknown",
            keyword="split"
        )

    def test_execute_loads_folds_from_csv(self, dataset, context, runtime_context, tmp_path):
        """Test execute() loads folds from CSV and sets on dataset."""
        # Create a fold file
        csv_content = """fold_0,fold_1
0,5
1,6
2,7
3,8
4,9
"""
        csv_path = tmp_path / "folds.csv"
        csv_path.write_text(csv_content)

        # Create parsed step mock
        step_info = MagicMock()
        step_info.operator = str(csv_path)
        step_info.original_step = {"split": str(csv_path)}

        # Execute
        controller = FoldFileLoaderController()
        new_context, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context
        )

        # Verify folds were set
        assert dataset.num_folds == 2
        folds = dataset.folds
        assert len(folds) == 2
        assert folds[0][0] == [0, 1, 2, 3, 4]  # train
        assert folds[0][1] == [5, 6, 7, 8, 9]  # val

    def test_execute_file_not_found(self, dataset, context, runtime_context):
        """Test execute() raises error for missing file."""
        step_info = MagicMock()
        step_info.operator = "nonexistent.csv"
        step_info.original_step = {"split": "nonexistent.csv"}

        controller = FoldFileLoaderController()
        with pytest.raises(ValueError, match="Failed to parse fold file"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context
            )

    def test_execute_handles_missing_sample_ids(self, dataset, context, runtime_context, tmp_path):
        """Test execute() warns when fold file has some sample IDs not in dataset."""
        # Create fold file with 1 sample ID that doesn't exist (10)
        # which is < 10% of total (1 out of 11), so it should warn and filter
        csv_content = """fold_0,fold_1
0,5
1,6
2,7
3,8
4,9
10,
"""
        csv_path = tmp_path / "folds.csv"
        csv_path.write_text(csv_content)

        step_info = MagicMock()
        step_info.operator = str(csv_path)
        step_info.original_step = {"split": str(csv_path)}

        controller = FoldFileLoaderController()

        # Should not raise since it's < 10% missing (1 out of 11 total IDs)
        new_context, output = controller.execute(
            step_info=step_info,
            dataset=dataset,
            context=context,
            runtime_context=runtime_context
        )

        # Folds should be filtered to only include valid IDs
        folds = dataset.folds
        assert len(folds) == 2
        # Check that ID 10 is not in any fold
        all_ids = set()
        for train, val in folds:
            all_ids.update(train)
            all_ids.update(val)
        assert all(i < 10 for i in all_ids)

    def test_execute_raises_too_many_missing_sample_ids(self, dataset, context, runtime_context, tmp_path):
        """Test execute() raises error when too many sample IDs are missing."""
        # Create fold file with many sample IDs that don't exist (>10%)
        csv_content = """fold_0,fold_1
0,10
1,11
2,12
3,13
4,14
"""
        csv_path = tmp_path / "folds.csv"
        csv_path.write_text(csv_content)

        step_info = MagicMock()
        step_info.operator = str(csv_path)
        step_info.original_step = {"split": str(csv_path)}

        controller = FoldFileLoaderController()

        # Should raise since >10% of sample IDs are missing
        with pytest.raises(ValueError, match="sample IDs not in dataset"):
            controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=runtime_context
            )

    def test_supports_prediction_mode(self):
        """Test that controller supports prediction mode."""
        assert FoldFileLoaderController.supports_prediction_mode() is True

    def test_use_multi_source(self):
        """Test that controller is single-source."""
        assert FoldFileLoaderController.use_multi_source() is False

    def test_priority_higher_than_cross_validator(self):
        """Test that FoldFileLoaderController has higher priority than CrossValidatorController."""
        from nirs4all.controllers.splitters.split import CrossValidatorController
        assert FoldFileLoaderController.priority < CrossValidatorController.priority

class TestFoldConfigSchema:
    """Tests for FoldConfig schema."""

    def test_inline_fold_definition(self):
        """Test creating FoldConfig with inline fold definitions."""
        from nirs4all.data.schema.config import FoldConfig, FoldDefinition

        config = FoldConfig(
            folds=[
                FoldDefinition(train=[0, 1, 2], val=[3, 4, 5]),
                FoldDefinition(train=[3, 4, 5], val=[0, 1, 2])
            ]
        )

        fold_list = config.to_fold_list()
        assert fold_list is not None
        assert len(fold_list) == 2
        assert fold_list[0] == ([0, 1, 2], [3, 4, 5])

    def test_file_reference(self):
        """Test creating FoldConfig with file reference."""
        from nirs4all.data.schema.config import FoldConfig

        config = FoldConfig(file="path/to/folds.csv")

        assert config.file == "path/to/folds.csv"
        assert config.to_fold_list() is None  # No inline folds

    def test_column_reference(self):
        """Test creating FoldConfig with column reference."""
        from nirs4all.data.schema.config import FoldConfig

        config = FoldConfig(column="cv_fold")

        assert config.column == "cv_fold"
        assert config.to_fold_list() is None

    def test_multiple_sources_error(self):
        """Test that specifying multiple sources raises error."""
        from nirs4all.data.schema.config import FoldConfig, FoldDefinition

        with pytest.raises(ValueError, match="Multiple fold sources"):
            FoldConfig(
                folds=[FoldDefinition(train=[0, 1], val=[2, 3])],
                file="folds.csv"
            )
