"""
Unit tests for ArtifactLoader.

Tests artifact loading functionality:
- Loading artifacts by ID
- Loading artifacts by step/branch/fold context
- Dependency resolution
- Per-fold model loading
- Caching behavior (LRU cache)
- Backward compatibility with v1 manifests
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader, LRUCache
from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord, ArtifactType
from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist


@pytest.fixture
def workspace_path(tmp_path):
    """Create temporary workspace directory with artifacts subdirectory."""
    workspace = tmp_path / "workspace"
    artifacts_dir = workspace / "artifacts"
    artifacts_dir.mkdir(parents=True)
    return workspace


@pytest.fixture
def results_dir(tmp_path):
    """Create temporary results directory."""
    results = tmp_path / "results"
    results.mkdir(parents=True)
    (results / "_binaries").mkdir(parents=True)
    return results


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_basic_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=3)

        cache.put("a", 1)
        cache.put("b", 2)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

    def test_lru_eviction(self):
        """Test that oldest item is evicted when full."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_updates_order(self):
        """Test that get moves item to end (most recently used)."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)
        cache.put("b", 2)

        # Access "a" to make it most recently used
        cache.get("a")

        # Add new item - should evict "b" now (not "a")
        cache.put("c", 3)

        assert cache.get("a") == 1  # Still present
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3

    def test_put_existing_key(self):
        """Test that putting existing key updates value and order."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10)  # Update "a"

        assert cache.get("a") == 10

        # "a" should now be most recently used
        cache.put("c", 3)  # Should evict "b"

        assert cache.get("a") == 10
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_contains(self):
        """Test contains without updating order."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)

        assert cache.contains("a") is True
        assert cache.contains("b") is False

    def test_remove(self):
        """Test removing items from cache."""
        cache = LRUCache(max_size=3)

        cache.put("a", 1)
        cache.put("b", 2)

        cache.remove("a")

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.size == 1

    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUCache(max_size=3)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Record a hit

        cache.clear()

        assert cache.size == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=3)

        cache.put("a", 1)
        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        stats = cache.stats
        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestArtifactLoaderBasics:
    """Test basic ArtifactLoader functionality."""

    def test_loader_initialization(self, workspace_path):
        """Test ArtifactLoader initialization."""
        loader = ArtifactLoader(workspace_path, "test_dataset")

        assert loader.workspace == workspace_path
        assert loader.dataset == "test_dataset"
        assert loader.binaries_dir == workspace_path / "artifacts"

    def test_loader_empty_manifest(self, workspace_path):
        """Test loader with empty manifest."""
        loader = ArtifactLoader(workspace_path, "test_dataset")

        manifest = {"artifacts": {"schema_version": "2.0", "items": []}}
        loader.import_from_manifest(manifest)

        assert len(loader.get_all_records()) == 0

    def test_get_record_by_id(self, workspace_path):
        """Test getting artifact record by ID."""
        loader = ArtifactLoader(workspace_path, "test_dataset")

        manifest = {
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001_pls:0:all",
                        "content_hash": "sha256:abc123",
                        "path": "transformer_StandardScaler_abc123.pkl",
                        "pipeline_id": "0001_pls",
                        "branch_path": [],
                        "step_index": 0,
                        "fold_id": None,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "depends_on": [],
                        "format": "joblib",
                        "format_version": "sklearn==1.5.0",
                        "nirs4all_version": "0.5.0",
                        "size_bytes": 2048,
                        "created_at": "2025-12-12T10:00:00Z",
                        "params": {},
                    }
                ]
            }
        }
        loader.import_from_manifest(manifest)

        record = loader.get_record("0001_pls:0:all")
        assert record is not None
        assert record.artifact_id == "0001_pls:0:all"
        assert record.class_name == "StandardScaler"

    def test_get_cache_info(self, workspace_path):
        """Test getting cache information."""
        loader = ArtifactLoader(workspace_path, "test_dataset")

        info = loader.get_cache_info()
        assert "cached_count" in info
        assert "max_size" in info
        assert "hits" in info
        assert "misses" in info
        assert "hit_rate" in info
        assert info["cached_count"] == 0


class TestArtifactLoaderFromManifest:
    """Test creating loader from manifest."""

    def test_from_manifest_v2_format(self, results_dir):
        """Test creating loader from v2 manifest."""
        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:all",
                        "content_hash": "sha256:abc123def456",
                        "path": "transformer_StandardScaler_abc123def456.pkl",
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 0,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": "joblib",
                    }
                ]
            }
        }

        loader = ArtifactLoader.from_manifest(manifest, results_dir)

        assert len(loader.get_all_records()) == 1
        assert loader.dataset == "test_dataset"

    def test_from_manifest_v1_format(self, results_dir):
        """Test creating loader from v1 manifest (flat list)."""
        manifest = {
            "dataset": "test_dataset",
            "artifacts": [
                {
                    "name": "scaler",
                    "step": 0,
                    "hash": "sha256:abc123",
                    "path": "StandardScaler_abc123.pkl",
                    "format": "joblib",
                    "size": 2048,
                }
            ]
        }

        loader = ArtifactLoader.from_manifest(manifest, results_dir)

        # Should convert v1 to v2 format
        records = loader.get_all_records()
        assert len(records) == 1


class TestArtifactLoaderStepLoading:
    """Test loading artifacts by step context."""

    def test_load_for_step_basic(self, workspace_path, results_dir):
        """Test loading artifacts for a specific step."""
        # Create actual artifact file
        artifacts_dir = workspace_path / "artifacts"
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))
        artifact_meta = persist(scaler, artifacts_dir, "scaler")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:all",
                        "content_hash": artifact_meta["hash"],
                        "path": artifact_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 0,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": artifact_meta["format"],
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load artifacts for step 0
        artifacts = loader.load_for_step(step_index=0)
        assert len(artifacts) == 1
        artifact_id, loaded_scaler = artifacts[0]
        assert hasattr(loaded_scaler, 'mean_')

    def test_load_for_step_with_branch(self, workspace_path, results_dir):
        """Test loading artifacts for a specific step and branch."""
        artifacts_dir = workspace_path / "artifacts"

        # Create two scalers for different branches
        scaler0 = StandardScaler()
        scaler0.fit(np.array([[0], [1]]))
        meta0 = persist(scaler0, artifacts_dir, "scaler_b0")

        scaler1 = StandardScaler()
        scaler1.fit(np.array([[2], [3]]))
        meta1 = persist(scaler1, artifacts_dir, "scaler_b1")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:2:all",
                        "content_hash": meta0["hash"],
                        "path": meta0["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 2,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta0["format"],
                    },
                    {
                        "artifact_id": "0001:1:2:all",
                        "content_hash": meta1["hash"],
                        "path": meta1["path"],
                        "pipeline_id": "0001",
                        "branch_path": [1],
                        "step_index": 2,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta1["format"],
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load only branch 0 artifacts
        artifacts = loader.load_for_step(step_index=2, branch_path=[0])
        assert len(artifacts) == 1

    def test_import_builds_step_indexes(self, workspace_path, results_dir):
        """Step/branch/source indexes should be populated at import."""
        artifacts_dir = workspace_path / "artifacts"
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))
        meta = persist(scaler, artifacts_dir, "scaler_index")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:2:all",
                        "content_hash": meta["hash"],
                        "path": meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 2,
                        "source_index": 1,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta["format"],
                    }
                ]
            },
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        assert loader._by_step[2] == ["0001:0:2:all"]
        assert loader._by_step_branch[(2, (0,))] == ["0001:0:2:all"]
        assert loader._by_step_branch_source[(2, (0,), 1)] == ["0001:0:2:all"]

    def test_load_for_step_with_branch_and_source_uses_indexes(self, workspace_path, results_dir):
        """Branch+source lookups should include shared and branch-specific artifacts."""
        artifacts_dir = workspace_path / "artifacts"

        def _artifact(name: str, value: float) -> dict:
            scaler = StandardScaler()
            scaler.fit(np.array([[value], [value + 1.0]]))
            return persist(scaler, artifacts_dir, name)

        meta_shared_all = _artifact("shared_all", 0.0)
        meta_shared_source = _artifact("shared_source", 2.0)
        meta_branch_all = _artifact("branch_all", 4.0)
        meta_branch_source = _artifact("branch_source", 6.0)
        meta_other_branch = _artifact("other_branch", 8.0)

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "a_shared_all",
                        "content_hash": meta_shared_all["hash"],
                        "path": meta_shared_all["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 2,
                        "source_index": None,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta_shared_all["format"],
                    },
                    {
                        "artifact_id": "a_shared_source",
                        "content_hash": meta_shared_source["hash"],
                        "path": meta_shared_source["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 2,
                        "source_index": 1,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta_shared_source["format"],
                    },
                    {
                        "artifact_id": "a_branch_all",
                        "content_hash": meta_branch_all["hash"],
                        "path": meta_branch_all["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 2,
                        "source_index": None,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta_branch_all["format"],
                    },
                    {
                        "artifact_id": "a_branch_source",
                        "content_hash": meta_branch_source["hash"],
                        "path": meta_branch_source["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 2,
                        "source_index": 1,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta_branch_source["format"],
                    },
                    {
                        "artifact_id": "a_other_branch",
                        "content_hash": meta_other_branch["hash"],
                        "path": meta_other_branch["path"],
                        "pipeline_id": "0001",
                        "branch_path": [1],
                        "step_index": 2,
                        "source_index": 1,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": meta_other_branch["format"],
                    },
                ]
            },
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        artifacts = loader.load_for_step(step_index=2, branch_path=[0], source_index=1)
        artifact_ids = [artifact_id for artifact_id, _obj in artifacts]

        assert set(artifact_ids) == {
            "a_shared_all",
            "a_shared_source",
            "a_branch_all",
            "a_branch_source",
        }
        assert "a_other_branch" not in artifact_ids


class TestArtifactLoaderFoldModels:
    """Test loading fold-specific models."""

    def test_load_fold_models(self, workspace_path, results_dir):
        """Test loading all fold models for CV averaging."""
        artifacts_dir = workspace_path / "artifacts"

        # Create models for 3 folds
        models = []
        for i in range(3):
            model = Ridge(alpha=1.0)
            model.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
            meta = persist(model, artifacts_dir, f"model_fold{i}")
            models.append(meta)

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": f"0001:3:{i}",
                        "content_hash": models[i]["hash"],
                        "path": models[i]["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 3,
                        "fold_id": i,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": models[i]["format"],
                    }
                    for i in range(3)
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load all fold models
        fold_models = loader.load_fold_models(step_index=3)
        assert len(fold_models) == 3

        # Check they are sorted by fold_id
        fold_ids = [fold_id for fold_id, _ in fold_models]
        assert fold_ids == [0, 1, 2]


class TestArtifactLoaderDependencies:
    """Test dependency resolution."""

    def test_load_with_dependencies(self, workspace_path, results_dir):
        """Test loading artifact with all its dependencies."""
        artifacts_dir = workspace_path / "artifacts"

        # Create scaler and model
        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1], [2]]))
        scaler_meta = persist(scaler, artifacts_dir, "scaler")

        model = Ridge(alpha=1.0)
        model.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
        model_meta = persist(model, artifacts_dir, "model")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:all",
                        "content_hash": scaler_meta["hash"],
                        "path": scaler_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 0,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": scaler_meta["format"],
                        "depends_on": [],
                    },
                    {
                        "artifact_id": "0001:1:all",
                        "content_hash": model_meta["hash"],
                        "path": model_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [],
                        "step_index": 1,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": model_meta["format"],
                        "depends_on": ["0001:0:all"],
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load model with dependencies
        result = loader.load_with_dependencies("0001:1:all")

        # Should include both scaler and model
        assert "0001:0:all" in result
        assert "0001:1:all" in result


class TestArtifactLoaderCaching:
    """Test caching behavior."""

    def test_cache_hit(self, workspace_path, results_dir):
        """Test that second load hits cache."""
        artifacts_dir = workspace_path / "artifacts"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        meta = persist(scaler, artifacts_dir, "scaler")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # First load
        obj1 = loader.load_by_id("0001:0:all")
        cache_info = loader.get_cache_info()
        assert cache_info["misses"] == 1
        assert cache_info["hits"] == 0

        # Second load should return same object
        obj2 = loader.load_by_id("0001:0:all")
        cache_info = loader.get_cache_info()
        assert cache_info["hits"] == 1

        assert obj1 is obj2  # Same object from cache

    def test_clear_cache(self, workspace_path, results_dir):
        """Test cache clearing."""
        artifacts_dir = workspace_path / "artifacts"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        meta = persist(scaler, artifacts_dir, "scaler")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load to populate cache
        loader.load_by_id("0001:0:all")
        assert loader.get_cache_info()["cached_count"] == 1

        # Clear cache
        loader.clear_cache()
        assert loader.get_cache_info()["cached_count"] == 0

    def test_lru_eviction(self, workspace_path, results_dir):
        """Test LRU cache evicts oldest items when full."""
        artifacts_dir = workspace_path / "artifacts"

        # Create multiple scalers
        items = []
        for i in range(5):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i+1]]))
            meta = persist(scaler, artifacts_dir, f"scaler_{i}")
            items.append({
                "artifact_id": f"0001:{i}:all",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": i,
                "artifact_type": "transformer",
                "class_name": "StandardScaler",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        # Create loader with small cache
        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir, cache_size=3)
        loader.import_from_manifest(manifest)

        # Load 5 items into cache of size 3
        for i in range(5):
            loader.load_by_id(f"0001:{i}:all")

        # Only 3 items should be cached
        cache_info = loader.get_cache_info()
        assert cache_info["cached_count"] == 3
        assert cache_info["max_size"] == 3

    def test_set_cache_size(self, workspace_path, results_dir):
        """Test dynamically changing cache size."""
        artifacts_dir = workspace_path / "artifacts"

        items = []
        for i in range(5):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i+1]]))
            meta = persist(scaler, artifacts_dir, f"scaler_{i}")
            items.append({
                "artifact_id": f"0001:{i}:all",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": i,
                "artifact_type": "transformer",
                "class_name": "StandardScaler",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir, cache_size=5)
        loader.import_from_manifest(manifest)

        # Load all items
        for i in range(5):
            loader.load_by_id(f"0001:{i}:all")

        assert loader.get_cache_info()["cached_count"] == 5

        # Reduce cache size - should evict oldest
        loader.set_cache_size(2)
        assert loader.get_cache_info()["cached_count"] == 2
        assert loader.get_cache_info()["max_size"] == 2

    def test_preload_artifacts(self, workspace_path, results_dir):
        """Test preloading artifacts into cache."""
        artifacts_dir = workspace_path / "artifacts"

        items = []
        for i in range(3):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i+1]]))
            meta = persist(scaler, artifacts_dir, f"scaler_{i}")
            items.append({
                "artifact_id": f"0001:{i}:all",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": i,
                "artifact_type": "transformer",
                "class_name": "StandardScaler",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Preload all artifacts
        count = loader.preload_artifacts()
        assert count == 3

        # All should be cached
        assert loader.get_cache_info()["cached_count"] == 3

        # Preloading again should return 0 (already cached)
        count = loader.preload_artifacts()
        assert count == 0

    def test_preload_specific_artifacts(self, workspace_path, results_dir):
        """Test preloading specific artifacts."""
        artifacts_dir = workspace_path / "artifacts"

        items = []
        for i in range(3):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i+1]]))
            meta = persist(scaler, artifacts_dir, f"scaler_{i}")
            items.append({
                "artifact_id": f"0001:{i}:all",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": i,
                "artifact_type": "transformer",
                "class_name": "StandardScaler",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Preload only first two
        count = loader.preload_artifacts(artifact_ids=["0001:0:all", "0001:1:all"])
        assert count == 2
        assert loader.get_cache_info()["cached_count"] == 2


class TestArtifactLoaderLegacyCompatibility:
    """Test backward compatibility with legacy artifacts."""

    def test_get_step_binaries_legacy_method(self, workspace_path, results_dir):
        """Test legacy get_step_binaries method."""
        artifacts_dir = workspace_path / "artifacts"

        scaler = StandardScaler()
        scaler.fit(np.array([[0], [1]]))
        meta = persist(scaler, artifacts_dir, "scaler")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Use legacy method
        binaries = loader.get_step_binaries(0)
        assert len(binaries) == 1
        name, obj = binaries[0]
        # v2 returns name with step suffix for compatibility
        assert "StandardScaler" in name

    def test_has_binaries_for_step_legacy_method(self, workspace_path, results_dir):
        """Test legacy has_binaries_for_step method."""
        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:2:all",
                    "content_hash": "sha256:abc123",
                    "path": "test.pkl",
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 2,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": "joblib",
                }]
            }
        }
        loader.import_from_manifest(manifest)

        assert loader.has_binaries_for_step(2) is True
        assert loader.has_binaries_for_step(0) is False
        assert loader.has_binaries_for_step(99) is False


class TestArtifactLoaderErrorHandling:
    """Test error handling."""

    def test_load_by_id_not_found(self, workspace_path):
        """Test loading non-existent artifact raises KeyError."""
        loader = ArtifactLoader(workspace_path, "test_dataset")

        with pytest.raises(KeyError):
            loader.load_by_id("nonexistent:0:all")

    def test_load_missing_file(self, workspace_path, results_dir):
        """Test loading artifact with missing file."""
        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:0:all",
                    "content_hash": "sha256:missing",
                    "path": "missing_file.pkl",
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 0,
                    "artifact_type": "transformer",
                    "class_name": "StandardScaler",
                    "format": "joblib",
                }]
            }
        }
        loader.import_from_manifest(manifest)

        with pytest.raises(FileNotFoundError):
            loader.load_by_id("0001:0:all")


class TestMetaModelLoading:
    """Test meta-model (stacking) loading functionality."""

    def test_load_meta_model_with_sources(self, workspace_path, results_dir):
        """Test loading meta-model with its source models."""
        artifacts_dir = workspace_path / "artifacts"

        # Create source models
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
        lr_meta = persist(lr, artifacts_dir, "lr_model")

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
        rf_meta = persist(rf, artifacts_dir, "rf_model")

        # Create meta-model
        meta = Ridge(alpha=1.0)
        meta.fit(np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2]))
        meta_meta = persist(meta, artifacts_dir, "meta_model")

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
                        "depends_on": [],
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
                        "depends_on": [],
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
                            "feature_columns": ["LinearRegression_pred", "RandomForestRegressor_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Load meta-model with sources
        meta_model, source_models, feature_columns = loader.load_meta_model_with_sources("0001:5:all")

        assert isinstance(meta_model, Ridge)
        assert len(source_models) == 2
        assert source_models[0][0] == "0001:3:all"
        assert source_models[1][0] == "0001:4:all"
        assert isinstance(source_models[0][1], LinearRegression)
        assert feature_columns == ["LinearRegression_pred", "RandomForestRegressor_pred"]

    def test_load_meta_model_non_meta_fails(self, workspace_path, results_dir):
        """Test that loading non-meta-model as meta fails."""
        artifacts_dir = workspace_path / "artifacts"

        model = Ridge()
        model.fit(np.array([[0], [1]]), np.array([0, 1]))
        meta = persist(model, artifacts_dir, "model")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [{
                    "artifact_id": "0001:3:all",
                    "content_hash": meta["hash"],
                    "path": meta["path"],
                    "pipeline_id": "0001",
                    "branch_path": [],
                    "step_index": 3,
                    "artifact_type": "model",  # Not meta_model
                    "class_name": "Ridge",
                    "format": meta["format"],
                }]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        with pytest.raises(ValueError, match="not a meta-model"):
            loader.load_meta_model_with_sources("0001:3:all")

    def test_load_meta_model_branch_validation(self, workspace_path, results_dir):
        """Test that branch context validation works for meta-models."""
        artifacts_dir = workspace_path / "artifacts"

        # Create source model with different branch
        source = Ridge()
        source.fit(np.array([[0], [1]]), np.array([0, 1]))
        source_meta = persist(source, artifacts_dir, "source")

        # Create meta-model
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))
        meta_meta = persist(meta, artifacts_dir, "meta")

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:1:3:all",  # Branch [1]
                        "content_hash": source_meta["hash"],
                        "path": source_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [1],  # Different branch
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": source_meta["format"],
                    },
                    {
                        "artifact_id": "0001:0:5:all",  # Branch [0]
                        "content_hash": meta_meta["hash"],
                        "path": meta_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],  # Different branch from source
                        "step_index": 5,
                        "artifact_type": "meta_model",
                        "class_name": "Ridge",
                        "format": meta_meta["format"],
                        "depends_on": ["0001:1:3:all"],
                        "meta_config": {
                            "source_models": [
                                {"artifact_id": "0001:1:3:all", "feature_index": 0}
                            ],
                            "feature_columns": ["Ridge_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Should fail because source is from different branch
        with pytest.raises(ValueError, match="Branch context mismatch"):
            loader.load_meta_model_with_sources("0001:0:5:all", validate_branch=True)

    def test_load_meta_model_shared_source_valid(self, workspace_path, results_dir):
        """Test that meta-model can use shared (pre-branch) sources."""
        artifacts_dir = workspace_path / "artifacts"

        # Create shared source model (no branch)
        source = Ridge()
        source.fit(np.array([[0], [1]]), np.array([0, 1]))
        source_meta = persist(source, artifacts_dir, "source")

        # Create meta-model in a branch
        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))
        meta_meta = persist(meta, artifacts_dir, "meta")

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
                        "branch_path": [],  # Shared (pre-branch)
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": source_meta["format"],
                    },
                    {
                        "artifact_id": "0001:0:5:all",
                        "content_hash": meta_meta["hash"],
                        "path": meta_meta["path"],
                        "pipeline_id": "0001",
                        "branch_path": [0],  # In a branch
                        "step_index": 5,
                        "artifact_type": "meta_model",
                        "class_name": "Ridge",
                        "format": meta_meta["format"],
                        "depends_on": ["0001:3:all"],
                        "meta_config": {
                            "source_models": [
                                {"artifact_id": "0001:3:all", "feature_index": 0}
                            ],
                            "feature_columns": ["Ridge_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Should succeed - shared source is valid for any branch
        meta_model, source_models, feature_columns = loader.load_meta_model_with_sources(
            "0001:0:5:all", validate_branch=True
        )
        assert len(source_models) == 1

    def test_load_meta_model_for_prediction(self, workspace_path, results_dir):
        """Test load_meta_model_for_prediction convenience method."""
        artifacts_dir = workspace_path / "artifacts"

        # Create source and meta models
        source = Ridge()
        source.fit(np.array([[0], [1]]), np.array([0, 1]))
        source_meta = persist(source, artifacts_dir, "source")

        meta = Ridge()
        meta.fit(np.array([[0], [1]]), np.array([0, 1]))
        meta_meta = persist(meta, artifacts_dir, "meta")

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
                        "branch_path": [],
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": source_meta["format"],
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
                        "depends_on": ["0001:3:all"],
                        "meta_config": {
                            "source_models": [
                                {"artifact_id": "0001:3:all", "feature_index": 0}
                            ],
                            "feature_columns": ["Ridge_pred"]
                        }
                    }
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset", results_dir)
        loader.import_from_manifest(manifest)

        # Use prediction helper
        meta_model, source_models, feature_columns = loader.load_meta_model_for_prediction(
            "0001:5:all"
        )

        assert meta_model is not None
        assert len(source_models) == 1
        assert feature_columns == ["Ridge_pred"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
