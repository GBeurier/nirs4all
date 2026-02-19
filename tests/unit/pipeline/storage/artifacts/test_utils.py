"""
Unit tests for artifact utility functions (V3).

Tests the V3 artifact ID utilities, execution path helpers, and file utilities.
"""

from pathlib import Path

import pytest

from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain, OperatorNode, generate_artifact_id_v3
from nirs4all.pipeline.storage.artifacts.utils import (
    ExecutionPath,
    artifact_id_matches_context,
    compute_content_hash,
    extract_pipeline_id_from_artifact_id,
    generate_filename,
    get_binaries_path,
    get_short_hash,
    is_v3_artifact_id,
    parse_artifact_id,
    parse_artifact_id_v3,
    parse_filename,
    validate_artifact_id,
)


class TestGenerateArtifactIdV3:
    """Tests for generate_artifact_id_v3 function (V3 chain-based IDs)."""

    def test_simple_chain_no_fold(self):
        """Test ID generation with simple chain."""
        chain = OperatorChain([OperatorNode(step_index=1, operator_class="MinMaxScaler")])
        artifact_id = generate_artifact_id_v3(
            pipeline_id="0001_pls",
            chain=chain,
            fold_id=None
        )
        # V3 format: pipeline_id$chain_hash:fold
        assert artifact_id.startswith("0001_pls$")
        assert artifact_id.endswith(":all")
        assert is_v3_artifact_id(artifact_id)

    def test_chain_with_fold(self):
        """Test ID generation with fold."""
        chain = OperatorChain([
            OperatorNode(step_index=1, operator_class="MinMaxScaler"),
            OperatorNode(step_index=3, operator_class="PLSRegression")
        ])
        artifact_id = generate_artifact_id_v3(
            pipeline_id="0001_pls",
            chain=chain,
            fold_id=0
        )
        assert artifact_id.startswith("0001_pls$")
        assert artifact_id.endswith(":0")

    def test_chain_with_branch(self):
        """Test ID generation with branching."""
        chain = OperatorChain([
            OperatorNode(step_index=1, operator_class="MinMaxScaler"),
            OperatorNode(step_index=3, operator_class="SNV", branch_path=[0]),
            OperatorNode(step_index=4, operator_class="PLS", branch_path=[0])
        ])
        artifact_id = generate_artifact_id_v3(
            pipeline_id="0001_pls",
            chain=chain,
            fold_id=None
        )
        assert is_v3_artifact_id(artifact_id)

    def test_string_chain_path(self):
        """Test ID generation with string chain path."""
        chain_path = "s1.MinMaxScaler>s3.PLS[br=0]"
        artifact_id = generate_artifact_id_v3(
            pipeline_id="0001_pls",
            chain=chain_path,
            fold_id=2
        )
        assert is_v3_artifact_id(artifact_id)
        assert artifact_id.endswith(":2")

class TestParseArtifactIdV3:
    """Tests for parse_artifact_id function (V3 only)."""

    def test_parse_v3_id_no_fold(self):
        """Test parsing V3 ID without fold."""
        artifact_id = "0001_pls$a1b2c3d4e5f6:all"
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            artifact_id
        )
        assert pipeline_id == "0001_pls"
        assert branch_path == []  # V3 doesn't encode branch in ID (use ArtifactRecord)
        assert step_index == 0  # V3 doesn't encode step in ID
        assert fold_id is None
        assert sub_index is None

    def test_parse_v3_id_with_fold(self):
        """Test parsing V3 ID with fold."""
        artifact_id = "0001_pls$a1b2c3d4e5f6:0"
        pipeline_id, branch_path, step_index, fold_id, sub_index = parse_artifact_id(
            artifact_id
        )
        assert pipeline_id == "0001_pls"
        assert fold_id == 0

    def test_parse_v3_id_with_different_folds(self):
        """Test parsing V3 IDs with different fold values."""
        for fold in [0, 1, 5, 42]:
            artifact_id = f"0001_pls$abc123def456:{fold}"
            _, _, _, fold_id, _ = parse_artifact_id(artifact_id)
            assert fold_id == fold

    def test_parse_v2_raises_error(self):
        """Test that parsing V2 IDs raises an error (no backward compat)."""
        with pytest.raises(ValueError, match="V2 artifact format is no longer supported"):
            parse_artifact_id("0001_pls:3:all")  # V2 format

        with pytest.raises(ValueError, match="V2 artifact format is no longer supported"):
            parse_artifact_id("0001_pls:0:3:0")  # V2 with branch

    def test_v3_roundtrip(self):
        """Test V3 ID generation and parsing roundtrip."""
        chain = OperatorChain([
            OperatorNode(step_index=1, operator_class="MinMaxScaler"),
            OperatorNode(step_index=3, operator_class="PLS", branch_path=[0])
        ])

        for fold_id in [None, 0, 1, 5]:
            artifact_id = generate_artifact_id_v3("0001_pls", chain, fold_id)
            parsed_pipeline_id, _, _, parsed_fold_id, _ = parse_artifact_id(artifact_id)
            assert parsed_pipeline_id == "0001_pls"
            assert parsed_fold_id == fold_id

class TestIsV3ArtifactId:
    """Tests for is_v3_artifact_id function."""

    def test_v3_format_detected(self):
        """Test V3 format detection."""
        assert is_v3_artifact_id("0001_pls$abc123def456:all") is True
        assert is_v3_artifact_id("0001_pls$abc123def456:0") is True

    def test_v2_format_not_v3(self):
        """Test V2 format is not detected as V3."""
        assert is_v3_artifact_id("0001_pls:3:all") is False
        assert is_v3_artifact_id("0001_pls:0:3:0") is False

class TestExecutionPath:
    """Tests for ExecutionPath dataclass."""

    def test_to_artifact_id(self):
        """Test converting ExecutionPath to artifact ID (V3 format)."""
        path = ExecutionPath(
            pipeline_id="0001_pls",
            branch_path=[0],
            step_index=3,
            fold_id=0,
            chain_path="s1.MinMaxScaler>s3.PLS[br=0]"
        )
        # V3 format: pipeline_id$chain_hash:fold_id
        artifact_id = path.to_artifact_id()
        assert artifact_id.startswith("0001_pls$")
        assert artifact_id.endswith(":0")
        assert "$" in artifact_id

    def test_from_artifact_id_v3(self):
        """Test creating ExecutionPath from V3 artifact ID."""
        # First create a valid V3 ID
        path = ExecutionPath(
            pipeline_id="0001_pls",
            branch_path=[0, 2],
            step_index=3,
            fold_id=None,
            chain_path="s1.MinMaxScaler>s3.PLS[br=0,br=2]"
        )
        artifact_id = path.to_artifact_id()

        # Now parse it back
        restored = ExecutionPath.from_artifact_id_v3(artifact_id)

        assert restored.pipeline_id == "0001_pls"
        assert restored.fold_id is None
        # Note: V3 stores chain_path as hash, so we get chain_path from the hash reference
        assert restored.chain_path is not None

    def test_roundtrip(self):
        """Test ExecutionPath roundtrip with V3 format.

        Note: V3 uses one-way hashing of chain_path, so full roundtrip
        requires passing the original chain_path to from_artifact_id_v3.
        """
        original = ExecutionPath(
            pipeline_id="0002_rf",
            branch_path=[1],
            step_index=7,
            fold_id=5,
            chain_path="s1.SNV>s7.RF[br=1]"
        )

        artifact_id = original.to_artifact_id()

        # Roundtrip requires passing the original chain_path
        restored = ExecutionPath.from_artifact_id_v3(
            artifact_id,
            chain_path=original.chain_path
        )

        assert restored.pipeline_id == original.pipeline_id
        assert restored.fold_id == original.fold_id
        assert restored.chain_path == original.chain_path
        # The artifact IDs should match when chain_path is preserved
        assert restored.to_artifact_id() == artifact_id

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
        """Test binaries path construction.

        V3 uses content-addressed storage at workspace/artifacts (shared).
        The dataset parameter is accepted for backward compatibility but unused.
        """
        workspace = Path("/home/user/workspace")
        path = get_binaries_path(workspace, "corn_m5")
        assert path == Path("/home/user/workspace/artifacts")

class TestValidateArtifactId:
    """Tests for validate_artifact_id function (V3 only)."""

    def test_valid_v3_ids(self):
        """Test valid V3 artifact IDs."""
        assert validate_artifact_id("0001$abc123def456:all") is True
        assert validate_artifact_id("0001_pls$abc123def456:0") is True
        assert validate_artifact_id("0001_pls_abc$xyz789012345:5") is True

    def test_invalid_ids(self):
        """Test invalid artifact IDs."""
        assert validate_artifact_id("invalid") is False
        assert validate_artifact_id("0001:3:all") is False  # V2 format
        assert validate_artifact_id("") is False

class TestExtractPipelineId:
    """Tests for extract_pipeline_id_from_artifact_id function."""

    def test_extract_v3(self):
        """Test extracting pipeline ID from V3 IDs."""
        assert extract_pipeline_id_from_artifact_id(
            "0001$abc123def456:all"
        ) == "0001"
        assert extract_pipeline_id_from_artifact_id(
            "0001_pls$abc123def456:0"
        ) == "0001_pls"
        assert extract_pipeline_id_from_artifact_id(
            "0001_pls_abc123$xyz789012345:3"
        ) == "0001_pls_abc123"

class TestArtifactIdMatchesContext:
    """Tests for artifact_id_matches_context function (V3 only)."""

    def test_match_pipeline_id(self):
        """Test matching by pipeline_id."""
        assert artifact_id_matches_context(
            "0001$abc123def456:all", pipeline_id="0001"
        ) is True
        assert artifact_id_matches_context(
            "0001$abc123def456:all", pipeline_id="0002"
        ) is False

    def test_v3_ignores_step_index(self):
        """Test that V3 matching returns True for step_index (not in ID)."""
        # V3 can't match step_index from ID alone - returns True
        assert artifact_id_matches_context(
            "0001$abc123def456:all", step_index=3
        ) is True
        assert artifact_id_matches_context(
            "0001$abc123def456:all", step_index=5
        ) is True

    def test_v3_ignores_branch_path(self):
        """Test that V3 matching returns True for branch_path (not in ID)."""
        # V3 can't match branch_path from ID alone - returns True
        assert artifact_id_matches_context(
            "0001$abc123def456:all", branch_path=[0]
        ) is True
        assert artifact_id_matches_context(
            "0001$abc123def456:all", branch_path=[]
        ) is True

    def test_match_fold_id(self):
        """Test matching by fold_id."""
        assert artifact_id_matches_context(
            "0001$abc123def456:0", fold_id=0
        ) is True
        assert artifact_id_matches_context(
            "0001$abc123def456:0", fold_id=1
        ) is False
        assert artifact_id_matches_context(
            "0001$abc123def456:all", fold_id=None
        ) is True

    def test_match_multiple(self):
        """Test matching multiple criteria."""
        assert artifact_id_matches_context(
            "0001$abc123def456:0",
            pipeline_id="0001",
            fold_id=0
        ) is True
        # V3 ignores branch_path and step_index (not in ID)
        assert artifact_id_matches_context(
            "0001$abc123def456:0",
            pipeline_id="0001",
            branch_path=[0],
            step_index=3,
            fold_id=0
        ) is True

    def test_match_invalid_id(self):
        """Test matching with invalid or V2 ID returns False."""
        assert artifact_id_matches_context("invalid", pipeline_id="0001") is False
        assert artifact_id_matches_context("0001:3:all", pipeline_id="0001") is False
