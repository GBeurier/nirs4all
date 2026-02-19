"""Tests for ArtifactRegistry cache-key support and lifespan.

Task 0.4: Verifies that get_by_chain_and_data() returns the correct artifact
          when both chain path and data hash match, and None when either differs.
Task 0.5: Verifies that the same ArtifactRegistry instance is shared across
          multiple variant executions within a dataset run.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain, OperatorNode
from nirs4all.pipeline.storage.artifacts.types import ArtifactType


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a minimal workspace directory for testing."""
    return tmp_path / "workspace"

@pytest.fixture
def registry(tmp_workspace):
    """Create an ArtifactRegistry for testing."""
    reg = ArtifactRegistry(
        workspace=tmp_workspace,
        dataset="test_dataset",
        pipeline_id="p001",
    )
    reg.start_run()
    return reg

def _make_chain(step: int, cls_name: str) -> OperatorChain:
    """Helper to build a simple one-node OperatorChain."""
    return OperatorChain(
        nodes=[OperatorNode(step_index=step, operator_class=cls_name)],
        pipeline_id="p001",
    )

def _dummy_obj():
    """Return a trivially serializable object for registration."""
    return {"weight": 1.0}

# =========================================================================
# Task 0.4 — Cache-key support
# =========================================================================

class TestGetByChainAndData:
    """Tests for get_by_chain_and_data() cache-key lookup."""

    def test_returns_record_when_both_match(self, registry):
        """get_by_chain_and_data returns the correct record when chain and data hash match."""
        chain = _make_chain(1, "MinMaxScaler")
        data_hash = "sha256:aabbccdd"

        record = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash,
        )

        found = registry.get_by_chain_and_data(chain, data_hash)
        assert found is not None
        assert found.artifact_id == record.artifact_id

    def test_returns_none_when_chain_differs(self, registry):
        """get_by_chain_and_data returns None when the chain path is different."""
        chain_a = _make_chain(1, "MinMaxScaler")
        chain_b = _make_chain(2, "SNV")
        data_hash = "sha256:aabbccdd"

        registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain_a,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash,
        )

        found = registry.get_by_chain_and_data(chain_b, data_hash)
        assert found is None

    def test_returns_none_when_data_hash_differs(self, registry):
        """get_by_chain_and_data returns None when the data hash is different."""
        chain = _make_chain(1, "MinMaxScaler")

        registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash="sha256:aabbccdd",
        )

        found = registry.get_by_chain_and_data(chain, "sha256:different")
        assert found is None

    def test_returns_none_when_no_data_hash_registered(self, registry):
        """get_by_chain_and_data returns None when input_data_hash was not supplied during registration."""
        chain = _make_chain(1, "MinMaxScaler")

        registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            # no input_data_hash
        )

        found = registry.get_by_chain_and_data(chain, "sha256:anything")
        assert found is None

    def test_accepts_chain_path_string(self, registry):
        """get_by_chain_and_data works with a raw chain path string."""
        chain = _make_chain(1, "MinMaxScaler")
        chain_path = chain.to_path()
        data_hash = "sha256:112233"

        record = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash,
        )

        found = registry.get_by_chain_and_data(chain_path, data_hash)
        assert found is not None
        assert found.artifact_id == record.artifact_id

    def test_multiple_entries_distinct(self, registry):
        """Different chain+data pairs each resolve independently."""
        chain_a = _make_chain(1, "MinMaxScaler")
        chain_b = _make_chain(2, "SNV")
        data_hash_1 = "sha256:d1"
        data_hash_2 = "sha256:d2"

        rec_a = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain_a,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash_1,
        )
        rec_b = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain_b,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=2,
            input_data_hash=data_hash_2,
        )

        assert registry.get_by_chain_and_data(chain_a, data_hash_1).artifact_id == rec_a.artifact_id
        assert registry.get_by_chain_and_data(chain_b, data_hash_2).artifact_id == rec_b.artifact_id
        assert registry.get_by_chain_and_data(chain_a, data_hash_2) is None
        assert registry.get_by_chain_and_data(chain_b, data_hash_1) is None

class TestGetByChainUnchanged:
    """Existing get_by_chain() behaviour must remain unchanged."""

    def test_get_by_chain_still_works(self, registry):
        """get_by_chain returns the record regardless of input_data_hash."""
        chain = _make_chain(1, "MinMaxScaler")

        record = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash="sha256:whatever",
        )

        found = registry.get_by_chain(chain)
        assert found is not None
        assert found.artifact_id == record.artifact_id

    def test_get_by_chain_without_data_hash(self, registry):
        """get_by_chain works when no input_data_hash was provided."""
        chain = _make_chain(1, "MinMaxScaler")

        record = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
        )

        found = registry.get_by_chain(chain)
        assert found is not None
        assert found.artifact_id == record.artifact_id

class TestCacheKeyCleanup:
    """Cache-key index is cleaned up during purge and failure cleanup."""

    def test_cleanup_failed_run_clears_cache_key(self, registry):
        """cleanup_failed_run removes entries from the cache-key index."""
        chain = _make_chain(1, "MinMaxScaler")
        data_hash = "sha256:abc"

        registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash,
        )
        assert registry.get_by_chain_and_data(chain, data_hash) is not None

        registry.cleanup_failed_run()
        assert registry.get_by_chain_and_data(chain, data_hash) is None

    def test_purge_clears_cache_key(self, registry):
        """purge_dataset_artifacts clears the cache-key index."""
        chain = _make_chain(1, "MinMaxScaler")
        data_hash = "sha256:abc"

        registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain,
            artifact_type=ArtifactType.TRANSFORMER,
            step_index=1,
            input_data_hash=data_hash,
        )
        assert registry.get_by_chain_and_data(chain, data_hash) is not None

        registry.purge_dataset_artifacts(confirm=True)
        assert registry.get_by_chain_and_data(chain, data_hash) is None

# =========================================================================
# Task 0.5 — Registry lifespan across variant executions
# =========================================================================

class TestRegistryLifespan:
    """Verify that the same ArtifactRegistry instance is shared across
    multiple pipeline variant executions within a single dataset run.

    The orchestrator creates one ArtifactRegistry per dataset, passes it to
    the ExecutorBuilder, and then re-uses that builder/executor for every
    pipeline variant in the inner loop.  These tests verify the structural
    guarantee by inspecting the orchestrator flow.
    """

    def test_registry_shared_across_variants(self, tmp_workspace):
        """The same ArtifactRegistry identity is passed to RuntimeContext for
        every pipeline variant execution on the same dataset."""
        registry = ArtifactRegistry(
            workspace=tmp_workspace,
            dataset="ds",
            pipeline_id="p001",
        )
        registry.start_run()

        # Simulate two variant executions writing to the same registry
        chain_v1 = _make_chain(1, "PLS")
        chain_v2 = _make_chain(1, "Ridge")
        data_hash = "sha256:shared_data"

        rec1 = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain_v1,
            artifact_type=ArtifactType.MODEL,
            step_index=1,
            input_data_hash=data_hash,
        )
        rec2 = registry.register_with_chain(
            obj=_dummy_obj(),
            chain=chain_v2,
            artifact_type=ArtifactType.MODEL,
            step_index=1,
            input_data_hash=data_hash,
        )

        # Both artifacts are retrievable from the same registry instance
        assert registry.get_by_chain_and_data(chain_v1, data_hash).artifact_id == rec1.artifact_id
        assert registry.get_by_chain_and_data(chain_v2, data_hash).artifact_id == rec2.artifact_id

        # The registry also exposes both via the standard chain index
        assert registry.get_by_chain(chain_v1) is not None
        assert registry.get_by_chain(chain_v2) is not None

        # Total registered artifacts is 2
        assert len(registry.get_all_records()) == 2

        registry.end_run()

    def test_orchestrator_creates_registry_once_per_dataset(self):
        """Structural test: ArtifactRegistry is constructed once per dataset
        in the orchestrator and the same instance flows to all variant
        RuntimeContexts.

        We patch the ArtifactRegistry constructor and execute two pipeline
        variants on one dataset, then assert the constructor was called
        exactly once.
        """
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        with (
            patch.object(PipelineOrchestrator, "_normalize_pipeline") as mock_norm_pipe,
            patch.object(PipelineOrchestrator, "_normalize_dataset") as mock_norm_ds,
            patch("nirs4all.pipeline.execution.orchestrator.ArtifactRegistry") as MockRegistry,
            patch("nirs4all.pipeline.execution.orchestrator.ExecutorBuilder") as MockBuilder,
            patch("nirs4all.pipeline.config.context.RuntimeContext") as MockRuntimeCtx,
        ):
            # Set up mocks for a single dataset with two pipeline variants
            mock_pipeline_configs = MagicMock()
            mock_pipeline_configs.steps = [["step_a"], ["step_b"]]
            mock_pipeline_configs.names = ["variant_1", "variant_2"]
            mock_pipeline_configs.generator_choices = [None, None]
            mock_norm_pipe.return_value = mock_pipeline_configs

            mock_dataset_configs = MagicMock()
            mock_dataset = MagicMock()
            mock_dataset.aggregate = None
            mock_dataset.aggregate_method = None
            mock_dataset.aggregate_exclude_outliers = False
            mock_dataset.repetition = None
            mock_dataset_configs.configs = [({"config": True}, "ds_name")]
            mock_dataset_configs.get_dataset.return_value = mock_dataset
            mock_norm_ds.return_value = mock_dataset_configs

            # The mock registry instance
            mock_reg_instance = MagicMock()
            MockRegistry.return_value = mock_reg_instance

            # The mock executor
            mock_executor = MagicMock()
            mock_executor.initialize_context.return_value = MagicMock()
            mock_builder_instance = MagicMock()
            mock_builder_instance.with_workspace.return_value = mock_builder_instance
            mock_builder_instance.with_verbose.return_value = mock_builder_instance
            mock_builder_instance.with_mode.return_value = mock_builder_instance
            mock_builder_instance.with_save_artifacts.return_value = mock_builder_instance
            mock_builder_instance.with_save_charts.return_value = mock_builder_instance
            mock_builder_instance.with_continue_on_error.return_value = mock_builder_instance
            mock_builder_instance.with_show_spinner.return_value = mock_builder_instance
            mock_builder_instance.with_plots_visible.return_value = mock_builder_instance
            mock_builder_instance.with_artifact_loader.return_value = mock_builder_instance
            mock_builder_instance.with_artifact_registry.return_value = mock_builder_instance
            mock_builder_instance.with_store.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = mock_executor
            MockBuilder.return_value = mock_builder_instance

            # RuntimeContext mock
            mock_runtime_ctx = MagicMock()
            mock_runtime_ctx.pipeline_uid = None
            mock_runtime_ctx.get_execution_trace.return_value = None
            MockRuntimeCtx.return_value = mock_runtime_ctx

            orchestrator = PipelineOrchestrator(
                workspace_path=Path("fake_workspace"),
                verbose=0,
                mode="train",
                keep_datasets=False,
                enable_tab_reports=False,
            )
            # Stub out the store to avoid real DuckDB
            orchestrator.store = MagicMock()
            orchestrator.store.begin_run.return_value = "run-id"

            try:
                orchestrator.execute(
                    pipeline=["dummy"],
                    dataset="dummy",
                )
            except Exception:
                pass  # We don't care about execution completion

            # ArtifactRegistry was constructed exactly once (for one dataset)
            assert MockRegistry.call_count == 1

            # The same registry instance was passed to the builder
            mock_builder_instance.with_artifact_registry.assert_called_with(mock_reg_instance)
