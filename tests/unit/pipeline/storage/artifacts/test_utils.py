"""
Unit tests for artifact utility functions.

Tests the execution path utilities for artifact ID generation and parsing.
"""

import pytest
from pathlib import Path

from nirs4all.pipeline.storage.artifacts.utils import (
    ExecutionPath,
    generate_artifact_id,
    parse_artifact_id,
    generate_filename,
    parse_filename,
    compute_content_hash,
    get_short_hash,
    get_binaries_path,
    validate_artifact_id,
    extract_pipeline_id_from_artifact_id,
    artifact_id_matches_context,
)


class TestGenerateArtifactId:
    """Tests for generate_artifact_id function."""

    def test_simple_id_no_branch(self):
        """Test ID without branching."""
        artifact_id = generate_artifact_id(
            pipeline_id="0001_pls",
            branch_path=[],
            step_index=3,
            fold_id=None
        )
        assert artifact_id == "0001_pls:3:all"

    def test_id_with_fold(self):
        """Test ID with fold."""
        artifact_id = generate_artifact_id(
            pipeline_id="0001_pls",
            branch_path=[],
            step_index=3,
            fold_id=0
        )
        assert artifact_id == "0001_pls:3:0"

    def test_id_with_single_branch(self):
        """Test ID with single branch."""
        artifact_id = generate_artifact_id(
            pipeline_id="0001_pls",
            branch_path=[0],
            step_index=3,
            fold_id=None
        )
        assert artifact_id == "0001_pls:0:3:all"

    def test_id_with_nested_branches(self):
        """Test ID with nested branches."""
        artifact_id = generate_artifact_id(
            pipeline_id="0001_pls",
            branch_path=[0, 2],
            step_index=3,
            fold_id=None
        )
        assert artifact_id == "0001_pls:0:2:3:all"

    def test_id_with_branch_and_fold(self):
        """Test ID with both branch and fold."""
        artifact_id = generate_artifact_id(
            pipeline_id="0001_pls",
            branch_path=[1],
            step_index=5,
            fold_id=2
        )
        assert artifact_id == "0001_pls:1:5:2"


class TestParseArtifactId:
    """Tests for parse_artifact_id function."""

    def test_parse_simple_id(self):
        """Test parsing simple ID without branch."""
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            "0001_pls:3:all"
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == []
        assert step_index == 3
        assert fold_id is None
        assert sub_index is None

    def test_parse_id_with_fold(self):
        """Test parsing ID with fold."""
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            "0001_pls:3:0"
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == []
        assert step_index == 3
        assert fold_id == 0
        assert sub_index is None

    def test_parse_id_with_branch(self):
        """Test parsing ID with branch."""
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            "0001_pls:0:3:all"
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == [0]
        assert step_index == 3
        assert fold_id is None
        assert sub_index is None

    def test_parse_id_with_nested_branches(self):
        """Test parsing ID with nested branches."""
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            "0001_pls:0:2:3:all"
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == [0, 2]
        assert step_index == 3
        assert fold_id is None
        assert sub_index is None

    def test_parse_id_with_sub_index(self):
        """Test parsing ID with sub_index."""
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            "0001_pls:3.1:all"
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == []
        assert step_index == 3
        assert fold_id is None
        assert sub_index == 1

    def test_parse_invalid_id(self):
        """Test parsing invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid artifact ID format"):
            parse_artifact_id("invalid")

        with pytest.raises(ValueError, match="Invalid artifact ID format"):
            parse_artifact_id("0001:all")  # Missing step

    def test_roundtrip(self):
        """Test that generate and parse are inverses."""
        original = {
            "pipeline_id": "0001_pls_abc123",
            "branch_path": [1, 2],
            "step_index": 5,
            "fold_id": 3
        }

        artifact_id = generate_artifact_id(**original)
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            artifact_id
        )

        assert pipeline_id == original["pipeline_id"]
        assert branch_path == original["branch_path"]
        assert step_index == original["step_index"]
        assert fold_id == original["fold_id"]
        assert sub_index is None

    def test_roundtrip_with_sub_index(self):
        """Test that generate and parse are inverses with sub_index."""
        original = {
            "pipeline_id": "0001_pls_abc123",
            "branch_path": [1, 2],
            "step_index": 5,
            "fold_id": 3,
            "sub_index": 2
        }

        artifact_id = generate_artifact_id(**original)
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            artifact_id
        )

        assert pipeline_id == original["pipeline_id"]
        assert branch_path == original["branch_path"]
        assert step_index == original["step_index"]
        assert fold_id == original["fold_id"]
        assert sub_index == original["sub_index"]


class TestExecutionPath:
    """Tests for ExecutionPath dataclass."""

    def test_to_artifact_id(self):
        """Test converting ExecutionPath to artifact ID."""
        path = ExecutionPath(
            pipeline_id="0001_pls",
            branch_path=[0],
            step_index=3,
            fold_id=0
        )
        assert path.to_artifact_id() == "0001_pls:0:3:0"

    def test_from_artifact_id(self):
        """Test creating ExecutionPath from artifact ID."""
        path = ExecutionPath.from_artifact_id("0001_pls:0:2:3:all")

        assert path.pipeline_id == "0001_pls"
        assert path.branch_path == [0, 2]
        assert path.step_index == 3
        assert path.fold_id is None

    def test_roundtrip(self):
        """Test ExecutionPath roundtrip."""
        original = ExecutionPath(
            pipeline_id="0002_rf",
            branch_path=[1],
            step_index=7,
            fold_id=5
        )

        artifact_id = original.to_artifact_id()
        restored = ExecutionPath.from_artifact_id(artifact_id)

        assert restored.pipeline_id == original.pipeline_id
        assert restored.branch_path == original.branch_path
        assert restored.step_index == original.step_index
        assert restored.fold_id == original.fold_id


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_filename(
            artifact_type="model",
            class_name="PLSRegression",
            content_hash="abc123def456",
            extension="joblib"
        )
        assert filename == "model_PLSRegression_abc123def456.joblib"

    def test_filename_strips_sha256_prefix(self):
        """Test that sha256: prefix is stripped from hash."""
        filename = generate_filename(
            artifact_type="transformer",
            class_name="StandardScaler",
            content_hash="sha256:abc123def456",
            extension="pkl"
        )
        assert filename == "transformer_StandardScaler_abc123def456.pkl"

    def test_filename_truncates_hash(self):
        """Test that hash is truncated to 12 chars."""
        filename = generate_filename(
            artifact_type="model",
            class_name="Test",
            content_hash="abcdef123456789extra",
            extension="pkl"
        )
        assert "abcdef123456" in filename
        assert "extra" not in filename


class TestParseFilename:
    """Tests for parse_filename function."""

    def test_parse_new_format(self):
        """Test parsing new format filename."""
        result = parse_filename("model_PLSRegression_abc123.joblib")
        assert result == ("model", "PLSRegression", "abc123")

    def test_parse_legacy_format(self):
        """Test parsing legacy format (no type prefix)."""
        result = parse_filename("PLSRegression_abc123.pkl")
        assert result == ("", "PLSRegression", "abc123")

    def test_parse_class_with_underscore(self):
        """Test parsing class name with underscore."""
        result = parse_filename("transformer_My_Custom_Scaler_abc123.pkl")
        assert result[0] == "transformer"
        assert result[1] == "My_Custom_Scaler"
        assert result[2] == "abc123"

    def test_parse_invalid(self):
        """Test parsing invalid filename."""
        result = parse_filename("invalid.pkl")
        assert result is None


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_hash_with_prefix(self):
        """Test hash includes sha256 prefix."""
        content = b"test content"
        hash_value = compute_content_hash(content)
        assert hash_value.startswith("sha256:")

    def test_hash_deterministic(self):
        """Test hash is deterministic."""
        content = b"same content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_content(self):
        """Test different content produces different hash."""
        hash1 = compute_content_hash(b"content 1")
        hash2 = compute_content_hash(b"content 2")
        assert hash1 != hash2


class TestGetShortHash:
    """Tests for get_short_hash function."""

    def test_with_prefix(self):
        """Test with sha256 prefix."""
        short = get_short_hash("sha256:abcdef123456789", length=6)
        assert short == "abcdef"

    def test_without_prefix(self):
        """Test without prefix."""
        short = get_short_hash("abcdef123456789", length=6)
        assert short == "abcdef"

    def test_default_length(self):
        """Test default length is 12."""
        short = get_short_hash("sha256:abcdef123456789")
        assert len(short) == 12
        assert short == "abcdef123456"


class TestGetBinariesPath:
    """Tests for get_binaries_path function."""

    def test_path_construction(self):
        """Test binaries path construction."""
        workspace = Path("/home/user/workspace")
        path = get_binaries_path(workspace, "corn_m5")
        assert path == Path("/home/user/workspace/binaries/corn_m5")


class TestValidateArtifactId:
    """Tests for validate_artifact_id function."""

    def test_valid_ids(self):
        """Test valid artifact IDs."""
        assert validate_artifact_id("0001:3:all") is True
        assert validate_artifact_id("0001:0:3:0") is True
        assert validate_artifact_id("0001:0:2:3:all") is True

    def test_invalid_ids(self):
        """Test invalid artifact IDs."""
        assert validate_artifact_id("invalid") is False
        assert validate_artifact_id("0001:all") is False
        assert validate_artifact_id("") is False


class TestExtractPipelineId:
    """Tests for extract_pipeline_id_from_artifact_id function."""

    def test_extract(self):
        """Test extracting pipeline ID."""
        assert extract_pipeline_id_from_artifact_id("0001:3:all") == "0001"
        assert extract_pipeline_id_from_artifact_id("0001_pls:0:3:0") == "0001_pls"
        assert extract_pipeline_id_from_artifact_id(
            "0001_pls_abc123:0:2:3:all"
        ) == "0001_pls_abc123"


class TestArtifactIdMatchesContext:
    """Tests for artifact_id_matches_context function."""

    def test_match_pipeline_id(self):
        """Test matching by pipeline_id."""
        assert artifact_id_matches_context(
            "0001:3:all", pipeline_id="0001"
        ) is True
        assert artifact_id_matches_context(
            "0001:3:all", pipeline_id="0002"
        ) is False

    def test_match_step_index(self):
        """Test matching by step_index."""
        assert artifact_id_matches_context(
            "0001:3:all", step_index=3
        ) is True
        assert artifact_id_matches_context(
            "0001:3:all", step_index=5
        ) is False

    def test_match_branch_path(self):
        """Test matching by branch_path."""
        assert artifact_id_matches_context(
            "0001:0:3:all", branch_path=[0]
        ) is True
        assert artifact_id_matches_context(
            "0001:3:all", branch_path=[]
        ) is True
        assert artifact_id_matches_context(
            "0001:0:3:all", branch_path=[1]
        ) is False

    def test_match_fold_id(self):
        """Test matching by fold_id."""
        assert artifact_id_matches_context(
            "0001:3:0", fold_id=0
        ) is True
        assert artifact_id_matches_context(
            "0001:3:0", fold_id=1
        ) is False

    def test_match_multiple(self):
        """Test matching multiple criteria."""
        assert artifact_id_matches_context(
            "0001:0:3:0",
            pipeline_id="0001",
            branch_path=[0],
            step_index=3,
            fold_id=0
        ) is True

    def test_match_invalid_id(self):
        """Test matching with invalid ID returns False."""
        assert artifact_id_matches_context("invalid", pipeline_id="0001") is False
