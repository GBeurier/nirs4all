"""
Unit tests for ArtifactRecord and ArtifactType.

Tests the v2 artifact type definitions and data structures.
"""

import pytest
from datetime import datetime, timezone

from nirs4all.pipeline.storage.artifacts.types import (
    ArtifactRecord,
    ArtifactType,
    MetaModelConfig,
)


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_artifact_type_values(self):
        """Test all artifact type values."""
        assert ArtifactType.MODEL.value == "model"
        assert ArtifactType.TRANSFORMER.value == "transformer"
        assert ArtifactType.SPLITTER.value == "splitter"
        assert ArtifactType.ENCODER.value == "encoder"
        assert ArtifactType.META_MODEL.value == "meta_model"

    def test_artifact_type_str(self):
        """Test string conversion."""
        assert str(ArtifactType.MODEL) == "model"
        assert str(ArtifactType.TRANSFORMER) == "transformer"

    def test_artifact_type_from_string(self):
        """Test creating type from string."""
        assert ArtifactType("model") == ArtifactType.MODEL
        assert ArtifactType("meta_model") == ArtifactType.META_MODEL


class TestMetaModelConfig:
    """Tests for MetaModelConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = MetaModelConfig()
        assert config.source_models == []
        assert config.feature_columns == []

    def test_initialization_with_values(self):
        """Test initialization with values."""
        config = MetaModelConfig(
            source_models=[
                {"artifact_id": "0001:3:all", "feature_index": 0},
                {"artifact_id": "0001:4:all", "feature_index": 1},
            ],
            feature_columns=["pls_pred", "rf_pred"]
        )
        assert len(config.source_models) == 2
        assert config.feature_columns == ["pls_pred", "rf_pred"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MetaModelConfig(
            source_models=[{"artifact_id": "0001:3:all"}],
            feature_columns=["pred"]
        )
        d = config.to_dict()
        assert d["source_models"] == [{"artifact_id": "0001:3:all"}]
        assert d["feature_columns"] == ["pred"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "source_models": [{"artifact_id": "test"}],
            "feature_columns": ["col1", "col2"]
        }
        config = MetaModelConfig.from_dict(data)
        assert len(config.source_models) == 1
        assert config.feature_columns == ["col1", "col2"]


class TestArtifactRecord:
    """Tests for ArtifactRecord dataclass."""

    def test_minimal_initialization(self):
        """Test initialization with required fields."""
        record = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc123",
            path="model_PLSRegression_abc123.joblib",
            pipeline_id="0001"
        )
        assert record.artifact_id == "0001:3:all"
        assert record.content_hash == "sha256:abc123"
        assert record.pipeline_id == "0001"

    def test_full_initialization(self):
        """Test initialization with all fields."""
        record = ArtifactRecord(
            artifact_id="0001:0:3:0",
            content_hash="sha256:def456789abc",
            path="model_PLSRegression_def456.joblib",
            pipeline_id="0001_pls",
            branch_path=[0],
            step_index=3,
            fold_id=0,
            artifact_type=ArtifactType.MODEL,
            class_name="PLSRegression",
            depends_on=["0001_pls:0:all"],
            format="joblib",
            format_version="sklearn==1.5.0",
            nirs4all_version="0.5.0",
            size_bytes=2048,
            params={"n_components": 10}
        )
        assert record.branch_path == [0]
        assert record.fold_id == 0
        assert record.artifact_type == ArtifactType.MODEL
        assert record.depends_on == ["0001_pls:0:all"]
        assert record.params == {"n_components": 10}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc123",
            path="model_PLSRegression_abc123.joblib",
            pipeline_id="0001",
            artifact_type=ArtifactType.MODEL,
            class_name="PLSRegression"
        )
        d = record.to_dict()

        assert d["artifact_id"] == "0001:3:all"
        assert d["artifact_type"] == "model"  # String, not enum
        assert d["class_name"] == "PLSRegression"
        assert "meta_config" not in d  # None should be excluded

    def test_to_dict_with_meta_config(self):
        """Test to_dict includes meta_config when present."""
        record = ArtifactRecord(
            artifact_id="0001:5:all",
            content_hash="sha256:xyz",
            path="meta_model_Ridge_xyz.joblib",
            pipeline_id="0001",
            artifact_type=ArtifactType.META_MODEL,
            meta_config=MetaModelConfig(
                source_models=[{"artifact_id": "0001:3:all"}],
                feature_columns=["pls_pred"]
            )
        )
        d = record.to_dict()

        assert "meta_config" in d
        assert d["meta_config"]["source_models"] == [{"artifact_id": "0001:3:all"}]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "artifact_id": "0001:0:3:0",
            "content_hash": "sha256:abc123def456",
            "path": "model_PLSRegression_abc123.joblib",
            "pipeline_id": "0001_pls",
            "branch_path": [0],
            "step_index": 3,
            "fold_id": 0,
            "artifact_type": "model",
            "class_name": "PLSRegression",
            "depends_on": ["0001_pls:0:all"],
            "format": "joblib",
            "format_version": "sklearn==1.5.0",
            "size_bytes": 2048
        }
        record = ArtifactRecord.from_dict(data)

        assert record.artifact_id == "0001:0:3:0"
        assert record.branch_path == [0]
        assert record.fold_id == 0
        assert record.artifact_type == ArtifactType.MODEL
        assert record.depends_on == ["0001_pls:0:all"]

    def test_from_dict_with_meta_config(self):
        """Test from_dict with meta_config."""
        data = {
            "artifact_id": "0001:5:all",
            "content_hash": "sha256:xyz",
            "path": "meta_model_Ridge_xyz.joblib",
            "pipeline_id": "0001",
            "artifact_type": "meta_model",
            "meta_config": {
                "source_models": [{"artifact_id": "0001:3:all"}],
                "feature_columns": ["pred"]
            }
        }
        record = ArtifactRecord.from_dict(data)

        assert record.artifact_type == ArtifactType.META_MODEL
        assert record.meta_config is not None
        assert record.meta_config.feature_columns == ["pred"]

    def test_short_hash(self):
        """Test short_hash property."""
        record = ArtifactRecord(
            artifact_id="test",
            content_hash="sha256:abcdef123456789",
            path="test.pkl",
            pipeline_id="0001"
        )
        assert record.short_hash == "abcdef123456"

    def test_short_hash_without_prefix(self):
        """Test short_hash with hash that has no prefix."""
        record = ArtifactRecord(
            artifact_id="test",
            content_hash="abcdef123456789",
            path="test.pkl",
            pipeline_id="0001"
        )
        assert record.short_hash == "abcdef123456"

    def test_is_branch_specific(self):
        """Test is_branch_specific property."""
        # Not branch-specific
        record1 = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            branch_path=[]
        )
        assert not record1.is_branch_specific

        # Branch-specific
        record2 = ArtifactRecord(
            artifact_id="0001:0:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            branch_path=[0]
        )
        assert record2.is_branch_specific

    def test_is_fold_specific(self):
        """Test is_fold_specific property."""
        # Shared (not fold-specific)
        record1 = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            fold_id=None
        )
        assert not record1.is_fold_specific

        # Fold-specific
        record2 = ArtifactRecord(
            artifact_id="0001:3:0",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            fold_id=0
        )
        assert record2.is_fold_specific

    def test_is_meta_model(self):
        """Test is_meta_model property."""
        # Regular model
        record1 = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            artifact_type=ArtifactType.MODEL
        )
        assert not record1.is_meta_model

        # Meta model
        record2 = ArtifactRecord(
            artifact_id="0001:5:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            artifact_type=ArtifactType.META_MODEL
        )
        assert record2.is_meta_model

    def test_get_branch_path_str(self):
        """Test get_branch_path_str method."""
        record1 = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            branch_path=[]
        )
        assert record1.get_branch_path_str() == ""

        record2 = ArtifactRecord(
            artifact_id="0001:0:2:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            branch_path=[0, 2]
        )
        assert record2.get_branch_path_str() == "0:2"

    def test_get_fold_str(self):
        """Test get_fold_str method."""
        record1 = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            fold_id=None
        )
        assert record1.get_fold_str() == "all"

        record2 = ArtifactRecord(
            artifact_id="0001:3:5",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            fold_id=5
        )
        assert record2.get_fold_str() == "5"

    def test_repr(self):
        """Test string representation."""
        record = ArtifactRecord(
            artifact_id="0001:3:all",
            content_hash="sha256:abc",
            path="test.pkl",
            pipeline_id="0001",
            artifact_type=ArtifactType.MODEL,
            class_name="PLSRegression"
        )
        repr_str = repr(record)

        assert "0001:3:all" in repr_str
        assert "model" in repr_str
        assert "PLSRegression" in repr_str

    def test_roundtrip_dict(self):
        """Test that to_dict and from_dict are inverses."""
        original = ArtifactRecord(
            artifact_id="0001:0:2:3:0",
            content_hash="sha256:abcdef123456",
            path="model_PLSRegression_abcdef.joblib",
            pipeline_id="0001_pls",
            branch_path=[0, 2],
            step_index=3,
            fold_id=0,
            artifact_type=ArtifactType.MODEL,
            class_name="PLSRegression",
            depends_on=["0001_pls:0:all", "0001_pls:1:all"],
            format="joblib",
            format_version="sklearn==1.5.0",
            nirs4all_version="0.5.0",
            size_bytes=4096,
            params={"n_components": 15}
        )

        d = original.to_dict()
        restored = ArtifactRecord.from_dict(d)

        assert restored.artifact_id == original.artifact_id
        assert restored.content_hash == original.content_hash
        assert restored.branch_path == original.branch_path
        assert restored.fold_id == original.fold_id
        assert restored.artifact_type == original.artifact_type
        assert restored.depends_on == original.depends_on
        assert restored.params == original.params
