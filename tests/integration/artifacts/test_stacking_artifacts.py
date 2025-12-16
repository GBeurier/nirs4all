"""
Integration tests for stacking/meta-model artifact handling.

Tests meta-model persistence and loading:
- Meta-models are saved with source model references
- Source model dependencies are tracked correctly
- Feature column order is preserved
- Meta-model loading includes source models
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist
from nirs4all.pipeline.storage.artifacts.types import (
    ArtifactType,
    ArtifactRecord,
    MetaModelConfig,
)


class TestMetaModelRegistration:
    """Tests for meta-model artifact registration."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_register_meta_model_basic(self, registry):
        """Test basic meta-model registration."""
        # Register source models first
        lr = LinearRegression()
        lr.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))

        registry.register(
            obj=lr,
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL
        )

        # Register meta-model
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))

        record = registry.register_meta_model(
            obj=meta,
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all"],
            feature_columns=["LinearRegression_pred"]
        )

        assert record.artifact_type == ArtifactType.META_MODEL
        assert record.meta_config is not None
        assert len(record.meta_config.source_models) == 1
        assert record.meta_config.feature_columns == ["LinearRegression_pred"]

    def test_register_meta_model_multiple_sources(self, registry):
        """Test meta-model with multiple source models."""
        # Register source models
        lr = LinearRegression()
        lr.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register(obj=lr, artifact_id="0001:3:all", artifact_type=ArtifactType.MODEL)

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register(obj=rf, artifact_id="0001:4:all", artifact_type=ArtifactType.MODEL)

        # Register meta-model
        meta = Ridge()
        meta.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))

        record = registry.register_meta_model(
            obj=meta,
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all", "0001:4:all"],
            feature_columns=["LR_pred", "RF_pred"]
        )

        assert len(record.meta_config.source_models) == 2
        assert len(record.depends_on) == 2
        assert "0001:3:all" in record.depends_on
        assert "0001:4:all" in record.depends_on

    def test_register_meta_model_auto_feature_columns(self, registry):
        """Test that feature columns are auto-generated from class names."""
        # Register source models
        lr = LinearRegression()
        lr.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register(obj=lr, artifact_id="0001:3:all", artifact_type=ArtifactType.MODEL)

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register(obj=rf, artifact_id="0001:4:all", artifact_type=ArtifactType.MODEL)

        # Register meta-model without feature_columns
        meta = Ridge()
        meta.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))

        record = registry.register_meta_model(
            obj=meta,
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all", "0001:4:all"]
        )

        # Should auto-generate feature columns
        assert record.meta_config.feature_columns == [
            "LinearRegression_pred",
            "RandomForestRegressor_pred"
        ]

    def test_meta_model_missing_source_fails(self, registry):
        """Meta-model registration should fail if sources don't exist."""
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))

        with pytest.raises(ValueError, match="missing source model"):
            registry.register_meta_model(
                obj=meta,
                artifact_id="0001:5:all",
                source_model_ids=["nonexistent:3:all"]
            )

    def test_meta_model_dependency_graph(self, registry):
        """Meta-model dependencies should be tracked in graph."""
        # Register source
        lr = LinearRegression()
        lr.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register(obj=lr, artifact_id="0001:3:all", artifact_type=ArtifactType.MODEL)

        # Register meta
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))
        registry.register_meta_model(
            obj=meta,
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all"]
        )

        # Check dependency graph
        deps = registry.get_dependencies("0001:5:all")
        assert "0001:3:all" in deps


class TestMetaModelLoading:
    """Tests for meta-model artifact loading."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        binaries_dir = workspace / "binaries" / "test_dataset"
        binaries_dir.mkdir(parents=True)
        return workspace

    def test_load_meta_model_with_sources(self, workspace_path):
        """Test loading meta-model with its source models."""
        binaries_dir = workspace_path / "binaries" / "test_dataset"

        # Create source models
        lr = LinearRegression()
        lr.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
        lr_meta = persist(lr, binaries_dir, "lr_model")

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
        rf_meta = persist(rf, binaries_dir, "rf_model")

        # Create meta-model
        meta = Ridge()
        meta.fit(np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2]))
        meta_meta = persist(meta, binaries_dir, "meta_model")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:3:all",
                        "content_hash": lr_meta["hash"],
                        "path": lr_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "LinearRegression",
                        "format": lr_meta["format"],
                    },
                    {
                        "artifact_id": "0001:4:all",
                        "content_hash": rf_meta["hash"],
                        "path": rf_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 4,
                        "artifact_type": "model",
                        "class_name": "RandomForestRegressor",
                        "format": rf_meta["format"],
                    },
                    {
                        "artifact_id": "0001:5:all",
                        "content_hash": meta_meta["hash"],
                        "path": meta_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 5,
                        "artifact_type": "meta_model",
                        "class_name": "Ridge",
                        "format": meta_meta["format"],
                        "depends_on": ["0001:3:all", "0001:4:all"],
                        "meta_config": {
                            "source_models": [
                                {"artifact_id": "0001:3:all", "feature_index": 0},
                                {"artifact_id": "0001:4:all", "feature_index": 1}
                            ],
                            "feature_columns": ["LR_pred", "RF_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Load meta-model with sources
        meta_model, source_models, feature_columns = loader.load_meta_model_with_sources(
            "0001:5:all"
        )

        assert isinstance(meta_model, Ridge)
        assert len(source_models) == 2
        assert feature_columns == ["LR_pred", "RF_pred"]

        # Check source models
        assert source_models[0][0] == "0001:3:all"
        assert isinstance(source_models[0][1], LinearRegression)
        assert source_models[1][0] == "0001:4:all"
        assert isinstance(source_models[1][1], RandomForestRegressor)

    def test_load_meta_model_feature_order_preserved(self, workspace_path):
        """Feature column order should match source model order."""
        binaries_dir = workspace_path / "binaries" / "test_dataset"

        # Create models
        models_info = []
        for i, (name, cls) in enumerate([
            ("model_a", LinearRegression),
            ("model_b", Ridge),
            ("model_c", RandomForestRegressor)
        ]):
            model = cls() if cls != RandomForestRegressor else cls(n_estimators=5)
            model.fit(np.array([[0], [1]]), np.array([0, 1]))
            meta = persist(model, binaries_dir, name)
            models_info.append((f"0001:{i}:all", meta, cls.__name__))

        # Meta-model
        meta = Ridge()
        meta.fit(np.array([[0, 0, 0], [1, 1, 1]]), np.array([0, 1]))
        meta_meta = persist(meta, binaries_dir, "meta")

        items = [
            {
                "artifact_id": aid,
                "content_hash": m["hash"],
                "path": m["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": i,
                "artifact_type": "model",
                "class_name": cname,
                "format": m["format"],
            }
            for i, (aid, m, cname) in enumerate(models_info)
        ]

        # Add meta-model with specific feature order
        source_ids = [m[0] for m in models_info]
        items.append({
            "artifact_id": "0001:5:all",
            "content_hash": meta_meta["hash"],
            "path": meta_meta["path"],
            "pipeline_id": "0001",
            "branch_path": [],
            "step_index": 5,
            "artifact_type": "meta_model",
            "class_name": "Ridge",
            "format": meta_meta["format"],
            "depends_on": source_ids,
            "meta_config": {
                "source_models": [
                    {"artifact_id": aid, "feature_index": i}
                    for i, aid in enumerate(source_ids)
                ],
                "feature_columns": ["pred_a", "pred_b", "pred_c"]
            }
        })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        _, source_models, feature_columns = loader.load_meta_model_with_sources(
            "0001:5:all"
        )

        # Feature columns should be in order
        assert feature_columns == ["pred_a", "pred_b", "pred_c"]

        # Source models should be in order
        assert source_models[0][0] == "0001:0:all"
        assert source_models[1][0] == "0001:1:all"
        assert source_models[2][0] == "0001:2:all"


class TestMetaModelWithBranches:
    """Tests for meta-model artifacts with branching."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        binaries_dir = workspace / "binaries" / "test_dataset"
        binaries_dir.mkdir(parents=True)
        return workspace

    def test_meta_model_uses_shared_pre_branch_sources(self, workspace_path):
        """Meta-model in branch can use shared (pre-branch) sources."""
        binaries_dir = workspace_path / "binaries" / "test_dataset"

        # Shared source (no branch)
        source = LinearRegression()
        source.fit(np.array([[0], [1]]), np.array([0, 1]))
        source_meta = persist(source, binaries_dir, "source")

        # Meta-model in branch
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))
        meta_meta = persist(meta, binaries_dir, "meta")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:3:all",
                        "content_hash": source_meta["hash"],
                        "path": source_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],  # Shared
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "LinearRegression",
                        "format": source_meta["format"],
                    },
                    {
                        "artifact_id": "0001:0:5:all",
                        "content_hash": meta_meta["hash"],
                        "path": meta_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],  # In branch
                        "step_index": 5,
                        "artifact_type": "meta_model",
                        "class_name": "Ridge",
                        "format": meta_meta["format"],
                        "depends_on": ["0001:3:all"],
                        "meta_config": {
                            "source_models": [{"artifact_id": "0001:3:all", "feature_index": 0}],
                            "feature_columns": ["LR_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Should load successfully (shared source is valid)
        meta_model, source_models, _ = loader.load_meta_model_with_sources(
            "0001:0:5:all",
            validate_branch=True
        )

        assert meta_model is not None
        assert len(source_models) == 1


class TestMetaModelSerialization:
    """Tests for meta-model config serialization."""

    def test_meta_config_serialization_roundtrip(self):
        """MetaModelConfig should serialize and deserialize correctly."""
        config = MetaModelConfig(
            source_models=[
                {"artifact_id": "0001:3:all", "feature_index": 0},
                {"artifact_id": "0001:4:all", "feature_index": 1},
            ],
            feature_columns=["model_a_pred", "model_b_pred"]
        )

        d = config.to_dict()
        restored = MetaModelConfig.from_dict(d)

        assert len(restored.source_models) == 2
        assert restored.feature_columns == ["model_a_pred", "model_b_pred"]

    def test_artifact_record_with_meta_config_roundtrip(self):
        """ArtifactRecord with meta_config should roundtrip correctly."""
        record = ArtifactRecord(
            artifact_id="0001:5:all",
            content_hash="sha256:abc123",
            path="meta_model_Ridge_abc123.joblib",
            pipeline_id="0001",
            branch_path=[],
            step_index=5,
            artifact_type=ArtifactType.META_MODEL,
            class_name="Ridge",
            depends_on=["0001:3:all", "0001:4:all"],
            meta_config=MetaModelConfig(
                source_models=[
                    {"artifact_id": "0001:3:all", "feature_index": 0},
                    {"artifact_id": "0001:4:all", "feature_index": 1},
                ],
                feature_columns=["pls_pred", "rf_pred"]
            )
        )

        d = record.to_dict()
        restored = ArtifactRecord.from_dict(d)

        assert restored.artifact_type == ArtifactType.META_MODEL
        assert restored.meta_config is not None
        assert len(restored.meta_config.source_models) == 2
        assert restored.meta_config.feature_columns == ["pls_pred", "rf_pred"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
