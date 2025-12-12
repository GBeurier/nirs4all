"""
Unit tests for BinaryLoader with Branch Support.

Tests branch-aware artifact loading:
- Loading artifacts by step and branch_id
- Fallback behavior for legacy artifacts
- Branch-specific queries
- Error handling for missing branch artifacts
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import joblib

from nirs4all.pipeline.storage.artifacts.binary_loader import BinaryLoader


class TestBinaryLoaderBasics:
    """Test basic BinaryLoader functionality."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def test_loader_initialization(self, workspace_path):
        """Test BinaryLoader initialization with artifacts."""
        artifacts = [
            {"name": "scaler", "step": 1, "path": "ab/abc.pkl", "format": "joblib"},
            {"name": "model", "step": 2, "path": "cd/cde.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)

        info = loader.get_cache_info()
        assert "available_steps" in info
        assert 1 in info["available_steps"]
        assert 2 in info["available_steps"]

    def test_loader_empty_artifacts(self, workspace_path):
        """Test BinaryLoader with no artifacts."""
        loader = BinaryLoader([], workspace_path)

        info = loader.get_cache_info()
        assert len(info["available_steps"]) == 0

    def test_has_binaries_for_step(self, workspace_path):
        """Test has_binaries_for_step method."""
        artifacts = [
            {"name": "scaler", "step": 1, "path": "ab/abc.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)

        assert loader.has_binaries_for_step(1) is True
        assert loader.has_binaries_for_step(2) is False
        assert loader.has_binaries_for_step(99) is False


class TestBinaryLoaderBranchAware:
    """Test branch-aware artifact loading."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def branched_artifacts(self):
        """Create artifacts with branch metadata."""
        return [
            # Pre-branch artifact (shared)
            {"name": "splitter", "step": 1, "path": "aa/aaa.pkl", "format": "joblib"},
            # Branch 0 artifacts
            {"name": "StandardScaler", "step": 2, "branch_id": 0, "branch_name": "branch_0",
             "path": "ab/abc.pkl", "format": "joblib"},
            {"name": "Ridge", "step": 3, "branch_id": 0, "branch_name": "branch_0",
             "path": "ac/acd.pkl", "format": "joblib"},
            # Branch 1 artifacts
            {"name": "MinMaxScaler", "step": 2, "branch_id": 1, "branch_name": "branch_1",
             "path": "ba/bab.pkl", "format": "joblib"},
            {"name": "Ridge", "step": 3, "branch_id": 1, "branch_name": "branch_1",
             "path": "bc/bcd.pkl", "format": "joblib"},
        ]

    def test_get_available_branches(self, workspace_path, branched_artifacts):
        """Test get_available_branches method."""
        loader = BinaryLoader(branched_artifacts, workspace_path)

        # Step 1 has no branch (shared)
        branches_1 = loader.get_available_branches(1)
        assert None in branches_1

        # Step 2 has branches 0 and 1
        branches_2 = loader.get_available_branches(2)
        assert 0 in branches_2
        assert 1 in branches_2

        # Step 3 has branches 0 and 1
        branches_3 = loader.get_available_branches(3)
        assert 0 in branches_3
        assert 1 in branches_3

    def test_has_binaries_for_step_and_branch(self, workspace_path, branched_artifacts):
        """Test checking binaries by step and branch."""
        loader = BinaryLoader(branched_artifacts, workspace_path)

        # Shared step (branch=None)
        assert loader.has_binaries_for_step(1, branch_id=None) is True

        # Branch-specific steps
        assert loader.has_binaries_for_step(2, branch_id=0) is True
        assert loader.has_binaries_for_step(2, branch_id=1) is True
        assert loader.has_binaries_for_step(2, branch_id=99) is False

    def test_get_artifacts_for_branch(self, workspace_path, branched_artifacts):
        """Test getting artifacts filtered by branch."""
        loader = BinaryLoader(branched_artifacts, workspace_path)

        # Get branch 0 artifacts
        branch_0_arts = loader.get_artifacts_for_branch(0)
        assert len(branch_0_arts) == 2
        for art in branch_0_arts:
            assert art.get("branch_id") == 0

        # Get branch 1 artifacts
        branch_1_arts = loader.get_artifacts_for_branch(1)
        assert len(branch_1_arts) == 2
        for art in branch_1_arts:
            assert art.get("branch_id") == 1

    def test_get_shared_artifacts(self, workspace_path, branched_artifacts):
        """Test getting shared (non-branch) artifacts."""
        loader = BinaryLoader(branched_artifacts, workspace_path)

        shared_arts = loader.get_artifacts_for_branch(None)
        assert len(shared_arts) == 1
        assert shared_arts[0]["name"] == "splitter"


class TestBinaryLoaderLegacyCompatibility:
    """Test backward compatibility with legacy artifacts."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def legacy_artifacts(self):
        """Create legacy artifacts without branch_id."""
        return [
            {"name": "scaler", "step": 1, "path": "scaler_abc.pkl", "format": "joblib"},
            {"name": "pca", "step": 2, "path": "pca_def.pkl", "format": "joblib"},
            {"name": "model", "step": 3, "path": "model_ghi.pkl", "format": "joblib"},
        ]

    def test_legacy_artifacts_treated_as_no_branch(self, workspace_path, legacy_artifacts):
        """Test that legacy artifacts without branch_id work correctly."""
        loader = BinaryLoader(legacy_artifacts, workspace_path)

        # All steps should be available
        assert loader.has_binaries_for_step(1) is True
        assert loader.has_binaries_for_step(2) is True
        assert loader.has_binaries_for_step(3) is True

        # All should have None branch
        for step in [1, 2, 3]:
            branches = loader.get_available_branches(step)
            assert None in branches

    def test_legacy_artifacts_queryable_without_branch(self, workspace_path, legacy_artifacts):
        """Test querying legacy artifacts without specifying branch."""
        loader = BinaryLoader(legacy_artifacts, workspace_path)

        # Should be able to query without branch_id
        info = loader.get_cache_info()
        assert len(info["available_steps"]) == 3

    def test_mixed_legacy_and_branched(self, workspace_path):
        """Test mix of legacy and branched artifacts."""
        artifacts = [
            # Legacy artifact (no branch_id)
            {"name": "splitter", "step": 1, "path": "split.pkl", "format": "joblib"},
            # Branched artifacts
            {"name": "scaler", "step": 2, "branch_id": 0, "path": "scaler_b0.pkl", "format": "joblib"},
            {"name": "scaler", "step": 2, "branch_id": 1, "path": "scaler_b1.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)

        # Step 1 has no branch
        branches_1 = loader.get_available_branches(1)
        assert None in branches_1

        # Step 2 has branches 0 and 1
        branches_2 = loader.get_available_branches(2)
        assert 0 in branches_2
        assert 1 in branches_2


class TestBinaryLoaderStepBranchQueries:
    """Test combined step and branch queries."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def multi_branch_artifacts(self):
        """Create artifacts with multiple branches per step."""
        return [
            # Step 2 - Branch 0
            {"name": "StandardScaler", "step": 2, "branch_id": 0,
             "path": "b0_scaler.pkl", "format": "joblib"},
            # Step 2 - Branch 1
            {"name": "MinMaxScaler", "step": 2, "branch_id": 1,
             "path": "b1_scaler.pkl", "format": "joblib"},
            # Step 2 - Branch 2
            {"name": "RobustScaler", "step": 2, "branch_id": 2,
             "path": "b2_scaler.pkl", "format": "joblib"},
            # Step 3 - Models for each branch
            {"name": "Ridge", "step": 3, "branch_id": 0,
             "path": "b0_model.pkl", "format": "joblib"},
            {"name": "Ridge", "step": 3, "branch_id": 1,
             "path": "b1_model.pkl", "format": "joblib"},
            {"name": "Ridge", "step": 3, "branch_id": 2,
             "path": "b2_model.pkl", "format": "joblib"},
        ]

    def test_get_step_binaries_with_branch(self, workspace_path, multi_branch_artifacts):
        """Test getting binaries for specific step and branch."""
        loader = BinaryLoader(multi_branch_artifacts, workspace_path)

        # Get all branches for step 2
        all_branches = loader.get_available_branches(2)
        assert len(all_branches) == 3
        assert 0 in all_branches
        assert 1 in all_branches
        assert 2 in all_branches

    def test_count_artifacts_per_branch(self, workspace_path, multi_branch_artifacts):
        """Test counting artifacts per branch."""
        loader = BinaryLoader(multi_branch_artifacts, workspace_path)

        # Each branch should have 2 artifacts (scaler + model)
        for branch_id in [0, 1, 2]:
            branch_arts = loader.get_artifacts_for_branch(branch_id)
            assert len(branch_arts) == 2, f"Branch {branch_id} should have 2 artifacts"


class TestBinaryLoaderErrorHandling:
    """Test error handling in BinaryLoader."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def test_missing_step_returns_empty(self, workspace_path):
        """Test that querying missing step returns empty."""
        artifacts = [
            {"name": "scaler", "step": 1, "path": "scaler.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)

        # Step 99 doesn't exist
        branches = loader.get_available_branches(99)
        assert len(branches) == 0

    def test_missing_branch_returns_empty(self, workspace_path):
        """Test that querying missing branch returns empty."""
        artifacts = [
            {"name": "scaler", "step": 1, "branch_id": 0, "path": "scaler.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)

        # Branch 99 doesn't exist
        branch_arts = loader.get_artifacts_for_branch(99)
        assert len(branch_arts) == 0

    def test_invalid_artifact_format_handled(self, workspace_path):
        """Test that invalid artifact format is handled gracefully."""
        artifacts = [
            {"name": "scaler", "step": 1},  # Missing path and format
        ]

        # Should not raise during initialization
        loader = BinaryLoader(artifacts, workspace_path)
        assert loader.has_binaries_for_step(1) is True


class TestBinaryLoaderCacheInfo:
    """Test cache info with branch metadata."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    def test_cache_info_includes_branch_count(self, workspace_path):
        """Test that cache info includes branch information."""
        artifacts = [
            {"name": "scaler", "step": 2, "branch_id": 0, "path": "s0.pkl", "format": "joblib"},
            {"name": "scaler", "step": 2, "branch_id": 1, "path": "s1.pkl", "format": "joblib"},
            {"name": "model", "step": 3, "branch_id": 0, "path": "m0.pkl", "format": "joblib"},
            {"name": "model", "step": 3, "branch_id": 1, "path": "m1.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)
        info = loader.get_cache_info()

        assert "available_steps" in info
        assert 2 in info["available_steps"]
        assert 3 in info["available_steps"]

    def test_cache_info_total_artifacts(self, workspace_path):
        """Test that cache info includes total artifact count."""
        artifacts = [
            {"name": "a", "step": 1, "path": "a.pkl", "format": "joblib"},
            {"name": "b", "step": 2, "path": "b.pkl", "format": "joblib"},
            {"name": "c", "step": 3, "path": "c.pkl", "format": "joblib"},
        ]

        loader = BinaryLoader(artifacts, workspace_path)
        info = loader.get_cache_info()

        assert info.get("total_artifacts", len(artifacts)) >= 3
