"""
Tests for PartitionAssigner class.

Tests partition assignment by static, column-based, percentage, index, and file methods.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.partition import PartitionAssigner, PartitionError, PartitionResult


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": list(range(100)),
        "category": ["A", "B", "C", "D"] * 25,
        "value": np.random.randn(100),
        "quality": np.random.rand(100),
        "split": (["train"] * 60 + ["test"] * 30 + ["val"] * 10),
        "label": [0, 1] * 50,
    })

@pytest.fixture
def small_df():
    """Create a small DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "split": ["train", "train", "train", "train", "train",
                  "test", "test", "test", "test", "test"],
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })

class TestPartitionAssignerStatic:
    """Tests for static partition assignment."""

    def test_static_train(self, small_df):
        """Test assigning entire DataFrame to train partition."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, "train")

        assert result.has_train
        assert not result.has_test
        assert not result.has_predict
        assert len(result.train_indices) == 10
        assert len(result.train_data) == 10

    def test_static_test(self, small_df):
        """Test assigning entire DataFrame to test partition."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, "test")

        assert not result.has_train
        assert result.has_test
        assert not result.has_predict
        assert len(result.test_indices) == 10
        assert len(result.test_data) == 10

    def test_static_predict(self, small_df):
        """Test assigning entire DataFrame to predict partition."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, "predict")

        assert not result.has_train
        assert not result.has_test
        assert result.has_predict
        assert len(result.predict_indices) == 10
        assert len(result.predict_data) == 10

    def test_static_case_insensitive(self, small_df):
        """Test that partition names are case-insensitive."""
        assigner = PartitionAssigner()

        result1 = assigner.assign(small_df, "TRAIN")
        result2 = assigner.assign(small_df, "Train")
        result3 = assigner.assign(small_df, "TrAiN")

        assert result1.has_train
        assert result2.has_train
        assert result3.has_train

    def test_static_invalid_partition_raises(self, small_df):
        """Test that invalid partition name raises error."""
        assigner = PartitionAssigner()

        with pytest.raises(PartitionError, match="Invalid partition name"):
            assigner.assign(small_df, "invalid")

    def test_none_partition_returns_empty(self, small_df):
        """Test that None partition returns empty result."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, None)

        assert not result.has_train
        assert not result.has_test
        assert not result.has_predict

class TestPartitionAssignerColumnBased:
    """Tests for column-based partition assignment."""

    def test_column_based_default_values(self, small_df):
        """Test column-based partition with default value mappings."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, {"column": "split"})

        assert result.has_train
        assert result.has_test
        assert len(result.train_indices) == 5
        assert len(result.test_indices) == 5
        assert result.partition_column == "split"

    def test_column_based_custom_values(self, sample_df):
        """Test column-based partition with custom value mappings."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "column": "split",
            "train_values": ["train"],
            "test_values": ["test", "val"],
        })

        assert len(result.train_indices) == 60
        assert len(result.test_indices) == 40  # 30 test + 10 val

    def test_column_based_missing_column_raises(self, small_df):
        """Test that missing column raises error."""
        assigner = PartitionAssigner()

        with pytest.raises(PartitionError, match="not found in DataFrame"):
            assigner.assign(small_df, {"column": "nonexistent"})

    def test_column_based_unknown_values_error(self, sample_df):
        """Test that unknown values raise error by default."""
        # Create df with unknown value
        df = sample_df.copy()
        df.loc[0, "split"] = "unknown_value"

        assigner = PartitionAssigner()
        with pytest.raises(PartitionError, match="Unknown partition values"):
            assigner.assign(df, {
                "column": "split",
                "train_values": ["train"],
                "test_values": ["test", "val"],
            })

    def test_column_based_unknown_policy_ignore(self, sample_df):
        """Test that unknown values can be ignored."""
        df = sample_df.copy()
        df.loc[0, "split"] = "unknown_value"

        assigner = PartitionAssigner()
        result = assigner.assign(df, {
            "column": "split",
            "train_values": ["train"],
            "test_values": ["test", "val"],
            "unknown_policy": "ignore",
        })

        total_assigned = (
            len(result.train_indices) +
            len(result.test_indices) +
            len(result.predict_indices)
        )
        assert total_assigned == 99  # One row ignored

    def test_column_based_unknown_policy_train(self, sample_df):
        """Test that unknown values can be assigned to train."""
        df = sample_df.copy()
        df.loc[0, "split"] = "unknown_value"

        assigner = PartitionAssigner()
        result = assigner.assign(df, {
            "column": "split",
            "train_values": ["train"],
            "test_values": ["test", "val"],
            "unknown_policy": "train",
        })

        # Unknown row should be in train
        assert 0 in result.train_indices

class TestPartitionAssignerPercentage:
    """Tests for percentage-based partition assignment."""

    def test_percentage_simple(self, sample_df):
        """Test simple percentage split."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
        })

        assert len(result.train_indices) == 80
        assert len(result.test_indices) == 20

    def test_percentage_with_range(self, sample_df):
        """Test percentage range syntax."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "train": "0:60%",
            "test": "60%:80%",
            "predict": "80%:100%",
        })

        assert len(result.train_indices) == 60
        assert len(result.test_indices) == 20
        assert len(result.predict_indices) == 20

    def test_percentage_with_shuffle(self, sample_df):
        """Test percentage split with shuffling."""
        assigner = PartitionAssigner()

        # Without shuffle
        result1 = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
            "shuffle": False,
        })

        # With shuffle
        result2 = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
            "shuffle": True,
            "random_state": 42,
        })

        # Indices should be different (shuffled)
        assert set(result1.train_indices) != set(result2.train_indices)

    def test_percentage_with_random_state_reproducible(self, sample_df):
        """Test that same random state produces same split."""
        assigner = PartitionAssigner()

        result1 = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
            "shuffle": True,
            "random_state": 42,
        })

        result2 = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
            "shuffle": True,
            "random_state": 42,
        })

        assert result1.train_indices == result2.train_indices
        assert result1.test_indices == result2.test_indices

    def test_percentage_with_stratify(self, sample_df):
        """Test stratified percentage split."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "train": "80%",
            "test": "80%:100%",
            "stratify": "label",
            "random_state": 42,
        })

        # Check that proportions are maintained
        train_df = sample_df.iloc[result.train_indices]
        test_df = sample_df.iloc[result.test_indices]

        train_prop = train_df["label"].mean()
        test_prop = test_df["label"].mean()

        # Should be roughly equal (0.5 each)
        assert abs(train_prop - 0.5) < 0.1
        assert abs(test_prop - 0.5) < 0.1

class TestPartitionAssignerIndices:
    """Tests for index-based partition assignment."""

    def test_indices_explicit_list(self, small_df):
        """Test explicit index list partition."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, {
            "train": [0, 1, 2, 3, 4],
            "test": [5, 6, 7, 8, 9],
        })

        assert result.train_indices == [0, 1, 2, 3, 4]
        assert result.test_indices == [5, 6, 7, 8, 9]

    def test_indices_with_negative(self, small_df):
        """Test negative indices are resolved."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, {
            "train": [0, 1, 2, -3, -2, -1],
            "test": [3, 4, 5, 6],
        })

        # -3 -> 7, -2 -> 8, -1 -> 9
        assert result.train_indices == [0, 1, 2, 7, 8, 9]

    def test_indices_overlap_raises(self, small_df):
        """Test that overlapping indices raise error."""
        assigner = PartitionAssigner()

        with pytest.raises(PartitionError, match="overlap"):
            assigner.assign(small_df, {
                "train": [0, 1, 2, 3],
                "test": [3, 4, 5],  # 3 overlaps
            })

    def test_indices_out_of_range_raises(self, small_df):
        """Test that out-of-range indices raise error."""
        assigner = PartitionAssigner()

        with pytest.raises(PartitionError, match="out of range"):
            assigner.assign(small_df, {
                "train": [0, 1, 100],  # 100 is out of range
            })

class TestPartitionAssignerIndexFile:
    """Tests for index file-based partition assignment."""

    def test_index_file_txt(self, small_df):
        """Test loading indices from text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index files
            train_file = Path(tmpdir) / "train_idx.txt"
            test_file = Path(tmpdir) / "test_idx.txt"

            with open(train_file, "w") as f:
                f.write("0\n1\n2\n3\n4\n")

            with open(test_file, "w") as f:
                f.write("5\n6\n7\n8\n9\n")

            assigner = PartitionAssigner(base_path=Path(tmpdir))
            result = assigner.assign(small_df, {
                "train_file": str(train_file),
                "test_file": str(test_file),
            })

            assert result.train_indices == [0, 1, 2, 3, 4]
            assert result.test_indices == [5, 6, 7, 8, 9]

    def test_index_file_json(self, small_df):
        """Test loading indices from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = Path(tmpdir) / "train_idx.json"
            test_file = Path(tmpdir) / "test_idx.json"

            with open(train_file, "w") as f:
                json.dump([0, 1, 2, 3, 4], f)

            with open(test_file, "w") as f:
                json.dump([5, 6, 7, 8, 9], f)

            assigner = PartitionAssigner(base_path=Path(tmpdir))
            result = assigner.assign(small_df, {
                "train_file": str(train_file),
                "test_file": str(test_file),
            })

            assert result.train_indices == [0, 1, 2, 3, 4]
            assert result.test_indices == [5, 6, 7, 8, 9]

    def test_index_file_csv(self, small_df):
        """Test loading indices from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = Path(tmpdir) / "train_idx.csv"

            with open(train_file, "w") as f:
                f.write("0\n1\n2\n3\n4\n")

            assigner = PartitionAssigner(base_path=Path(tmpdir))
            result = assigner.assign(small_df, {
                "train_file": str(train_file),
            })

            assert result.train_indices == [0, 1, 2, 3, 4]

    def test_index_file_not_found_raises(self, small_df):
        """Test that missing index file raises error."""
        assigner = PartitionAssigner()

        with pytest.raises(PartitionError, match="not found"):
            assigner.assign(small_df, {
                "train_file": "/nonexistent/file.txt",
            })

class TestPartitionResult:
    """Tests for PartitionResult dataclass."""

    def test_get_indices(self, sample_df):
        """Test get_indices method."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "train": "0:50",
            "test": "50:100",
        })

        assert result.get_indices("train") == list(range(50))
        assert result.get_indices("test") == list(range(50, 100))

    def test_get_data(self, sample_df):
        """Test get_data method."""
        assigner = PartitionAssigner()
        result = assigner.assign(sample_df, {
            "train": "0:50",
            "test": "50:100",
        })

        train_data = result.get_data("train")
        test_data = result.get_data("test")

        assert len(train_data) == 50
        assert len(test_data) == 50

    def test_has_properties(self, sample_df):
        """Test has_* properties."""
        assigner = PartitionAssigner()

        # Only train
        result = assigner.assign(sample_df, "train")
        assert result.has_train
        assert not result.has_test
        assert not result.has_predict

        # Only test
        result = assigner.assign(sample_df, "test")
        assert not result.has_train
        assert result.has_test
        assert not result.has_predict

class TestPartitionAssignerConcatenate:
    """Tests for concatenating partition results."""

    def test_concatenate_single(self, small_df):
        """Test concatenating a single result."""
        assigner = PartitionAssigner()
        result = assigner.assign(small_df, "train")
        combined = assigner.concatenate_partitions([result])

        assert combined.train_indices == result.train_indices
        assert len(combined.train_data) == len(result.train_data)

    def test_concatenate_multiple(self, small_df):
        """Test concatenating multiple results."""
        assigner = PartitionAssigner()

        # Simulate two files, each going to train
        result1 = assigner.assign(small_df, "train")
        result2 = assigner.assign(small_df, "train")

        combined = assigner.concatenate_partitions([result1, result2])

        # Combined should have indices from both (offset adjusted)
        assert len(combined.train_indices) == 20
        assert len(combined.train_data) == 20

    def test_concatenate_mixed_partitions(self, small_df):
        """Test concatenating results with different partitions."""
        assigner = PartitionAssigner()

        result1 = assigner.assign(small_df, "train")
        result2 = assigner.assign(small_df, "test")

        combined = assigner.concatenate_partitions([result1, result2])

        assert len(combined.train_indices) == 10
        assert len(combined.test_indices) == 10

class TestPartitionConfigIntegration:
    """Tests for PartitionConfig.to_assigner_spec() integration."""

    def test_partition_config_static(self):
        """Test PartitionConfig for static partition."""
        from nirs4all.data.schema import PartitionConfig, PartitionType

        config = PartitionConfig(type=PartitionType.TRAIN)
        spec = config.to_assigner_spec()

        assert spec == "train"

    def test_partition_config_column_based(self):
        """Test PartitionConfig for column-based partition."""
        from nirs4all.data.schema import PartitionConfig

        config = PartitionConfig(
            column="split",
            train_values=["train"],
            test_values=["test"],
        )
        spec = config.to_assigner_spec()

        assert spec["column"] == "split"
        assert spec["train_values"] == ["train"]
        assert spec["test_values"] == ["test"]

    def test_partition_config_percentage(self):
        """Test PartitionConfig for percentage partition."""
        from nirs4all.data.schema import PartitionConfig

        config = PartitionConfig(
            train="80%",
            test="80%:100%",
            shuffle=True,
            random_state=42,
        )
        spec = config.to_assigner_spec()

        assert spec["train"] == "80%"
        assert spec["test"] == "80%:100%"
        assert spec["shuffle"] is True
        assert spec["random_state"] == 42

    def test_partition_config_file_based(self):
        """Test PartitionConfig for file-based partition."""
        from nirs4all.data.schema import PartitionConfig

        config = PartitionConfig(
            train_file="train.txt",
            test_file="test.txt",
        )
        spec = config.to_assigner_spec()

        assert spec["train_file"] == "train.txt"
        assert spec["test_file"] == "test.txt"

    def test_partition_config_empty_returns_none(self):
        """Test that empty PartitionConfig returns None."""
        from nirs4all.data.schema import PartitionConfig

        config = PartitionConfig()
        spec = config.to_assigner_spec()

        assert spec is None
