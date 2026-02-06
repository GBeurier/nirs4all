"""Tests for cross-run artifact caching (DuckDB-backed).

Validates the storage layer and query mechanisms for persisting
step-level cache keys across runs:

- Saving artifacts with cache keys
- Finding cached artifacts by key
- Cache miss when key doesn't match
- Cache invalidation when dataset changes
- Schema migration for new columns
- Backward compatibility for artifacts without cache keys
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nirs4all.pipeline.storage.store_schema import create_schema
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def conn():
    """Create an in-memory DuckDB connection with schema."""
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at a temp directory."""
    return WorkspaceStore(tmp_path / "workspace")


# =========================================================================
# Schema tests
# =========================================================================


class TestCacheKeySchema:
    """Verify the artifacts table has cache key columns."""

    def test_artifacts_table_has_cache_key_columns(self, conn):
        """The artifacts table includes chain_path_hash, input_data_hash, dataset_hash."""
        result = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'artifacts' ORDER BY ordinal_position"
        ).fetchall()
        columns = [row[0] for row in result]
        assert "chain_path_hash" in columns
        assert "input_data_hash" in columns
        assert "dataset_hash" in columns

    def test_cache_key_columns_are_nullable(self, conn):
        """Cache key columns allow NULL values (for existing artifacts)."""
        conn.execute(
            "INSERT INTO artifacts "
            "(artifact_id, artifact_path, content_hash, operator_class, "
            "artifact_type, format, size_bytes) "
            "VALUES ('a1', 'path/file.joblib', 'hash123', 'MyClass', "
            "'transformer', 'joblib', 1024)"
        )
        row = conn.execute(
            "SELECT chain_path_hash, input_data_hash, dataset_hash "
            "FROM artifacts WHERE artifact_id = 'a1'"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    def test_cache_key_index_exists(self, conn):
        """The composite index on (chain_path_hash, input_data_hash) is created."""
        result = conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'artifacts'"
        ).fetchall()
        index_names = [row[0] for row in result]
        assert "idx_artifacts_cache_key" in index_names

    def test_dataset_hash_index_exists(self, conn):
        """The index on dataset_hash is created."""
        result = conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'artifacts'"
        ).fetchall()
        index_names = [row[0] for row in result]
        assert "idx_artifacts_dataset_hash" in index_names


# =========================================================================
# Schema migration tests
# =========================================================================


class TestCacheKeyMigration:
    """Verify schema migration adds cache key columns to existing databases."""

    def test_migration_adds_cache_key_columns(self):
        """Calling create_schema on a database without cache key columns adds them."""
        conn = duckdb.connect(":memory:")
        # Create old schema without cache key columns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id VARCHAR PRIMARY KEY,
                artifact_path VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                operator_class VARCHAR,
                artifact_type VARCHAR,
                format VARCHAR DEFAULT 'joblib',
                size_bytes BIGINT,
                ref_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        # Also create required parent tables so schema creation succeeds
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                config JSON,
                datasets JSON,
                status VARCHAR DEFAULT 'running',
                created_at TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP,
                summary JSON,
                error VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipelines (
                pipeline_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL REFERENCES runs(run_id),
                name VARCHAR NOT NULL,
                expanded_config JSON,
                generator_choices JSON,
                dataset_name VARCHAR NOT NULL,
                dataset_hash VARCHAR,
                status VARCHAR DEFAULT 'running',
                created_at TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP,
                best_val DOUBLE,
                best_test DOUBLE,
                metric VARCHAR,
                duration_ms INTEGER,
                error VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                chain_id VARCHAR PRIMARY KEY,
                pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
                steps JSON NOT NULL,
                model_step_idx INTEGER NOT NULL,
                model_class VARCHAR NOT NULL,
                preprocessings VARCHAR DEFAULT '',
                fold_strategy VARCHAR DEFAULT 'per_fold',
                fold_artifacts JSON,
                shared_artifacts JSON,
                branch_path JSON,
                source_index INTEGER,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id VARCHAR PRIMARY KEY,
                pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
                chain_id VARCHAR REFERENCES chains(chain_id),
                dataset_name VARCHAR NOT NULL,
                model_name VARCHAR NOT NULL,
                model_class VARCHAR NOT NULL,
                fold_id VARCHAR NOT NULL,
                partition VARCHAR NOT NULL,
                val_score DOUBLE,
                test_score DOUBLE,
                train_score DOUBLE,
                metric VARCHAR NOT NULL,
                task_type VARCHAR NOT NULL,
                n_samples INTEGER,
                n_features INTEGER,
                scores JSON,
                best_params JSON,
                preprocessings VARCHAR DEFAULT '',
                branch_id INTEGER,
                branch_name VARCHAR,
                exclusion_count INTEGER DEFAULT 0,
                exclusion_rate DOUBLE DEFAULT 0.0,
                refit_context VARCHAR DEFAULT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_arrays (
                prediction_id VARCHAR PRIMARY KEY
                    REFERENCES predictions(prediction_id),
                y_true DOUBLE[],
                y_pred DOUBLE[],
                y_proba DOUBLE[],
                sample_indices INTEGER[],
                weights DOUBLE[]
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                log_id VARCHAR PRIMARY KEY,
                pipeline_id VARCHAR NOT NULL REFERENCES pipelines(pipeline_id),
                step_idx INTEGER NOT NULL,
                operator_class VARCHAR,
                event VARCHAR NOT NULL,
                duration_ms INTEGER,
                message VARCHAR,
                details JSON,
                level VARCHAR DEFAULT 'info',
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Insert an artifact before migration
        conn.execute(
            "INSERT INTO artifacts "
            "(artifact_id, artifact_path, content_hash, operator_class, "
            "artifact_type, format, size_bytes) "
            "VALUES ('old_art', 'path/old.joblib', 'oldhash', 'OldClass', "
            "'transformer', 'joblib', 512)"
        )

        # Run migration
        create_schema(conn)

        # Verify columns were added
        result = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'artifacts' ORDER BY ordinal_position"
        ).fetchall()
        columns = [row[0] for row in result]
        assert "chain_path_hash" in columns
        assert "input_data_hash" in columns
        assert "dataset_hash" in columns

        # Verify old data is preserved with NULL cache keys
        row = conn.execute(
            "SELECT chain_path_hash, input_data_hash, dataset_hash "
            "FROM artifacts WHERE artifact_id = 'old_art'"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

        conn.close()


# =========================================================================
# WorkspaceStore cache method tests
# =========================================================================


class TestSaveArtifactWithCacheKey:
    """Test saving artifacts with cross-run cache keys."""

    def test_save_artifact_with_cache_key(self, store):
        """Artifact is saved with cache key metadata."""
        from sklearn.preprocessing import MinMaxScaler

        art_id = store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chainhash_abc",
            input_data_hash="datahash_123",
            dataset_hash="dsethash_xyz",
        )
        assert art_id is not None

        # Verify cache key was stored
        row = store._fetch_one(
            "SELECT chain_path_hash, input_data_hash, dataset_hash "
            "FROM artifacts WHERE artifact_id = $1",
            [art_id],
        )
        assert row is not None
        assert row["chain_path_hash"] == "chainhash_abc"
        assert row["input_data_hash"] == "datahash_123"
        assert row["dataset_hash"] == "dsethash_xyz"

    def test_save_artifact_without_cache_key_has_null_fields(self, store):
        """Artifact saved via save_artifact() has NULL cache key fields."""
        from sklearn.preprocessing import StandardScaler

        art_id = store.save_artifact(
            StandardScaler(),
            "sklearn.preprocessing.StandardScaler",
            "transformer",
            "joblib",
        )
        row = store._fetch_one(
            "SELECT chain_path_hash, input_data_hash, dataset_hash "
            "FROM artifacts WHERE artifact_id = $1",
            [art_id],
        )
        assert row is not None
        assert row["chain_path_hash"] is None
        assert row["input_data_hash"] is None
        assert row["dataset_hash"] is None


class TestUpdateArtifactCacheKey:
    """Test retrofitting cache keys onto existing artifacts."""

    def test_update_cache_key_on_existing_artifact(self, store):
        """Cache key can be attached to an artifact after creation."""
        from sklearn.preprocessing import MinMaxScaler

        art_id = store.save_artifact(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )

        store.update_artifact_cache_key(
            art_id,
            chain_path_hash="chain_abc",
            input_data_hash="data_123",
            dataset_hash="dset_xyz",
        )

        row = store._fetch_one(
            "SELECT chain_path_hash, input_data_hash, dataset_hash "
            "FROM artifacts WHERE artifact_id = $1",
            [art_id],
        )
        assert row["chain_path_hash"] == "chain_abc"
        assert row["input_data_hash"] == "data_123"
        assert row["dataset_hash"] == "dset_xyz"


class TestFindCachedArtifact:
    """Test cache lookup by (chain_path_hash, input_data_hash) key."""

    def test_find_cached_artifact_hit(self, store):
        """Returns artifact_id when cache key matches."""
        from sklearn.preprocessing import MinMaxScaler

        art_id = store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_hit",
            input_data_hash="data_hit",
            dataset_hash="dset_1",
        )

        found = store.find_cached_artifact("chain_hit", "data_hit")
        assert found == art_id

    def test_find_cached_artifact_miss_wrong_chain(self, store):
        """Returns None when chain_path_hash doesn't match."""
        from sklearn.preprocessing import MinMaxScaler

        store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_a",
            input_data_hash="data_1",
            dataset_hash="dset_1",
        )

        found = store.find_cached_artifact("chain_b", "data_1")
        assert found is None

    def test_find_cached_artifact_miss_wrong_data(self, store):
        """Returns None when input_data_hash doesn't match."""
        from sklearn.preprocessing import MinMaxScaler

        store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_a",
            input_data_hash="data_1",
            dataset_hash="dset_1",
        )

        found = store.find_cached_artifact("chain_a", "data_2")
        assert found is None

    def test_find_cached_artifact_miss_empty_store(self, store):
        """Returns None when no artifacts exist."""
        found = store.find_cached_artifact("chain_x", "data_x")
        assert found is None

    def test_find_cached_artifact_after_update(self, store):
        """Cache lookup works after retrofitting cache keys."""
        from sklearn.preprocessing import MinMaxScaler

        art_id = store.save_artifact(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )

        # Initially no cache key, so miss
        assert store.find_cached_artifact("chain_retro", "data_retro") is None

        # Retrofit cache key
        store.update_artifact_cache_key(
            art_id,
            chain_path_hash="chain_retro",
            input_data_hash="data_retro",
            dataset_hash="dset_retro",
        )

        # Now it should be found
        found = store.find_cached_artifact("chain_retro", "data_retro")
        assert found == art_id


class TestInvalidateDatasetCache:
    """Test cache invalidation by dataset hash."""

    def test_invalidate_clears_cache_keys(self, store):
        """Invalidation clears cache keys for matching dataset_hash."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        art1 = store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_1",
            input_data_hash="data_1",
            dataset_hash="dataset_A",
        )
        art2 = store.save_artifact_with_cache_key(
            StandardScaler(),
            "sklearn.preprocessing.StandardScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_2",
            input_data_hash="data_2",
            dataset_hash="dataset_A",
        )

        count = store.invalidate_dataset_cache("dataset_A")
        assert count == 2

        # Cache lookups should now miss
        assert store.find_cached_artifact("chain_1", "data_1") is None
        assert store.find_cached_artifact("chain_2", "data_2") is None

        # But artifacts still exist (only cache keys cleared)
        row1 = store._fetch_one(
            "SELECT artifact_id FROM artifacts WHERE artifact_id = $1",
            [art1],
        )
        row2 = store._fetch_one(
            "SELECT artifact_id FROM artifacts WHERE artifact_id = $1",
            [art2],
        )
        assert row1 is not None
        assert row2 is not None

    def test_invalidate_does_not_affect_other_datasets(self, store):
        """Invalidation only affects artifacts with matching dataset_hash."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_a",
            input_data_hash="data_a",
            dataset_hash="dataset_A",
        )
        art_b = store.save_artifact_with_cache_key(
            StandardScaler(),
            "sklearn.preprocessing.StandardScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_b",
            input_data_hash="data_b",
            dataset_hash="dataset_B",
        )

        store.invalidate_dataset_cache("dataset_A")

        # dataset_A's cache is invalidated
        assert store.find_cached_artifact("chain_a", "data_a") is None

        # dataset_B's cache is untouched
        found = store.find_cached_artifact("chain_b", "data_b")
        assert found == art_b

    def test_invalidate_returns_zero_when_no_match(self, store):
        """Returns 0 when no artifacts match the dataset_hash."""
        count = store.invalidate_dataset_cache("nonexistent_dataset")
        assert count == 0

    def test_invalidate_skips_artifacts_without_cache_keys(self, store):
        """Artifacts without cache keys are not counted in invalidation."""
        from sklearn.preprocessing import MinMaxScaler

        store.save_artifact(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )

        count = store.invalidate_dataset_cache("some_hash")
        assert count == 0


# =========================================================================
# ArtifactRegistry cross-run cache persistence tests
# =========================================================================


class TestRegistryPersistCacheKeys:
    """Test ArtifactRegistry.persist_cache_keys_to_store."""

    def test_persist_cache_keys_to_store(self, store, tmp_path):
        """In-memory cache keys are pushed to the workspace store."""
        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        registry = ArtifactRegistry(
            workspace=tmp_path / "workspace",
            dataset="wheat",
            pipeline_id="0001_pls",
        )

        from sklearn.preprocessing import MinMaxScaler

        record = registry.register_with_chain(
            MinMaxScaler(),
            chain="s1.MinMaxScaler",
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash="input_hash_abc",
        )

        # Register the artifact in the workspace store so the update succeeds
        store.register_existing_artifact(
            artifact_id=record.artifact_id,
            path=record.path,
            content_hash=record.content_hash,
            operator_class=record.class_name,
            artifact_type=record.artifact_type.value,
            format=record.format,
            size_bytes=record.size_bytes,
        )

        count = registry.persist_cache_keys_to_store(
            store, dataset_hash="dset_hash_wheat"
        )
        assert count == 1

        # Verify the cache key is queryable from the store
        from nirs4all.pipeline.storage.artifacts.operator_chain import compute_chain_hash
        chain_path_hash = compute_chain_hash("s1.MinMaxScaler")
        found = store.find_cached_artifact(chain_path_hash, "input_hash_abc")
        assert found == record.artifact_id

    def test_load_cached_from_store(self, store, tmp_path):
        """Registry can query the workspace store for cached artifacts."""
        from sklearn.preprocessing import MinMaxScaler

        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

        registry = ArtifactRegistry(
            workspace=tmp_path / "workspace",
            dataset="wheat",
            pipeline_id="0001_pls",
        )

        # Save an artifact with cache key in the store
        art_id = store.save_artifact_with_cache_key(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_lookup",
            input_data_hash="data_lookup",
            dataset_hash="dset_lookup",
        )

        found = registry.load_cached_from_store(
            store, "chain_lookup", "data_lookup"
        )
        assert found == art_id

    def test_load_cached_from_store_miss(self, store, tmp_path):
        """Returns None when no cached artifact matches."""
        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

        registry = ArtifactRegistry(
            workspace=tmp_path / "workspace",
            dataset="wheat",
            pipeline_id="0001_pls",
        )

        found = registry.load_cached_from_store(
            store, "nonexistent_chain", "nonexistent_data"
        )
        assert found is None


# =========================================================================
# Artifact loading backward compatibility
# =========================================================================


class TestBackwardCompatibility:
    """Verify existing artifacts without cache keys still work."""

    def test_load_artifact_without_cache_key(self, store):
        """Artifacts saved without cache keys can still be loaded."""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        art_id = store.save_artifact(
            scaler,
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )

        loaded = store.load_artifact(art_id)
        assert type(loaded).__name__ == "MinMaxScaler"

    def test_get_artifact_path_without_cache_key(self, store):
        """get_artifact_path works for artifacts without cache keys."""
        from sklearn.preprocessing import MinMaxScaler

        art_id = store.save_artifact(
            MinMaxScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )

        path = store.get_artifact_path(art_id)
        assert path.exists()

    def test_deduplication_with_cache_key(self, store):
        """Content deduplication works with cache key artifacts."""
        from sklearn.preprocessing import MinMaxScaler

        same_obj = MinMaxScaler()

        art1 = store.save_artifact(
            same_obj,
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
        )
        art2 = store.save_artifact_with_cache_key(
            same_obj,
            "sklearn.preprocessing.MinMaxScaler",
            "transformer",
            "joblib",
            chain_path_hash="chain_dedup",
            input_data_hash="data_dedup",
            dataset_hash="dset_dedup",
        )

        # Same content -> same artifact_id (deduplication)
        assert art1 == art2

        # Cache key should be set on the deduplicated artifact
        found = store.find_cached_artifact("chain_dedup", "data_dedup")
        assert found == art1
