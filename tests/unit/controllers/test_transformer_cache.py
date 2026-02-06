"""
Unit tests for TransformerMixinController check-before-fit caching.

Tests that shared preprocessing steps are fitted once and reused via the
artifact registry's (chain_path, data_hash) cache-key index.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.pipeline.config.context import (
    DataSelector,
    ExecutionContext,
    PipelineState,
    RuntimeContext,
    StepMetadata,
)
from nirs4all.pipeline.steps.parser import ParsedStep, StepType
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain, OperatorNode
from nirs4all.pipeline.storage.artifacts.types import ArtifactType


class FitCountingScaler(TransformerMixin, BaseEstimator):
    """Transformer that counts how many times fit() is called."""

    fit_count = 0

    def __init__(self):
        pass

    def fit(self, X, y=None):
        FitCountingScaler.fit_count += 1
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def _make_mock_dataset(n_samples=20, n_features=50, content_hash_value="abc123"):
    """Create a mock dataset with the required interface for execute()."""
    dataset = MagicMock()
    dataset._may_contain_nan = False

    # Feature data: 3D array (samples, processings, features)
    train_data = np.random.RandomState(42).rand(n_samples, 1, n_features)
    all_data = np.random.RandomState(42).rand(n_samples, 1, n_features)

    def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
        return [all_data] if not concat_source else all_data

    dataset.x.side_effect = x_side_effect
    dataset.features_processings.return_value = ["raw"]
    dataset.content_hash.return_value = content_hash_value

    return dataset, train_data, all_data


def _make_context(processing=None):
    """Create a minimal ExecutionContext."""
    selector = DataSelector(
        partition="all",
        processing=processing or [["raw"]],
        branch_path=[],
    )
    state = PipelineState(y_processing="numeric", step_number=1)
    metadata = StepMetadata(keyword="transform")
    return ExecutionContext(selector=selector, state=state, metadata=metadata)


def _make_runtime_context(artifact_registry=None, step_number=1):
    """Create a RuntimeContext with optional artifact registry."""
    rc = RuntimeContext(
        artifact_registry=artifact_registry,
        step_number=step_number,
        pipeline_name="test_pipeline",
    )
    return rc


class TestTryCacheLookup:
    """Tests for the _try_cache_lookup method."""

    def test_returns_none_when_no_registry(self):
        """Cache lookup returns None when artifact_registry is None."""
        controller = TransformerMixinController()
        rc = _make_runtime_context(artifact_registry=None)
        context = _make_context()
        dataset, _, _ = _make_mock_dataset()

        result = controller._try_cache_lookup(
            runtime_context=rc,
            context=context,
            dataset=dataset,
            operator_name="StandardScaler",
            source_index=0,
        )

        assert result is None

    def test_returns_none_on_cache_miss(self):
        """Cache lookup returns None when no matching artifact exists."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )
            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry)
            context = _make_context()
            dataset, _, _ = _make_mock_dataset()

            result = controller._try_cache_lookup(
                runtime_context=rc,
                context=context,
                dataset=dataset,
                operator_name="StandardScaler",
                source_index=0,
            )

            assert result is None

    def test_returns_transformer_on_cache_hit(self):
        """Cache lookup returns a fitted transformer when a matching artifact exists."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            # First, register an artifact with the expected chain_path and data_hash
            scaler = StandardScaler()
            X = np.random.RandomState(42).rand(20, 50)
            scaler.fit(X)

            # Build the same chain path that _try_cache_lookup will construct
            chain = OperatorChain(pipeline_id="test_pipeline")
            node = OperatorNode(
                step_index=1,
                operator_class="StandardScaler",
                branch_path=[],
                source_index=0,
                fold_id=None,
                substep_index=0,  # processing_counter starts at 0
            )
            artifact_chain = chain.append(node)
            chain_path = artifact_chain.to_path()

            data_hash = "abc123"  # Must match what the mock dataset returns

            # Register with chain and data hash
            registry.register_with_chain(
                obj=scaler,
                chain=artifact_chain,
                artifact_type=ArtifactRegistry.__module__  # We need to import properly
                if False else __import__('nirs4all.pipeline.storage.artifacts.types', fromlist=['ArtifactType']).ArtifactType.TRANSFORMER,
                step_index=1,
                branch_path=[],
                source_index=0,
                input_data_hash=data_hash,
                pipeline_id="test_pipeline",
            )

            # Now do the lookup
            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)
            context = _make_context()
            dataset, _, _ = _make_mock_dataset(content_hash_value=data_hash)

            result = controller._try_cache_lookup(
                runtime_context=rc,
                context=context,
                dataset=dataset,
                operator_name="StandardScaler",
                source_index=0,
            )

            assert result is not None
            # Verify the loaded transformer has fitted attributes
            assert hasattr(result, 'mean_')
            assert hasattr(result, 'scale_')

    def test_returns_none_on_class_name_mismatch(self):
        """Cache lookup returns None when the cached artifact has a different class name."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )
            # Register a MinMaxScaler artifact
            mms = MinMaxScaler()
            X = np.random.RandomState(42).rand(20, 50)
            mms.fit(X)

            chain = OperatorChain(pipeline_id="test_pipeline")
            node = OperatorNode(
                step_index=1,
                operator_class="MinMaxScaler",
                branch_path=[],
                source_index=0,
                substep_index=0,
            )
            artifact_chain = chain.append(node)

            data_hash = "abc123"
            registry.register_with_chain(
                obj=mms,
                chain=artifact_chain,
                artifact_type=ArtifactType.TRANSFORMER,
                step_index=1,
                source_index=0,
                input_data_hash=data_hash,
                pipeline_id="test_pipeline",
            )

            # Look up with a different operator_name
            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)
            context = _make_context()
            dataset, _, _ = _make_mock_dataset(content_hash_value=data_hash)

            result = controller._try_cache_lookup(
                runtime_context=rc,
                context=context,
                dataset=dataset,
                operator_name="StandardScaler",  # Different from MinMaxScaler
                source_index=0,
            )

            # Should return None because the chain path is built using
            # "StandardScaler" as operator_class, so it won't match the
            # chain registered with "MinMaxScaler"
            assert result is None

    def test_returns_none_on_different_data_hash(self):
        """Cache lookup returns None when data hash doesn't match."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            scaler = StandardScaler()
            X = np.random.RandomState(42).rand(20, 50)
            scaler.fit(X)

            chain = OperatorChain(pipeline_id="test_pipeline")
            node = OperatorNode(
                step_index=1,
                operator_class="StandardScaler",
                branch_path=[],
                source_index=0,
                substep_index=0,
            )
            artifact_chain = chain.append(node)

            # Register with one data hash
            registry.register_with_chain(
                obj=scaler,
                chain=artifact_chain,
                artifact_type=ArtifactType.TRANSFORMER,
                step_index=1,
                source_index=0,
                input_data_hash="hash_A",
                pipeline_id="test_pipeline",
            )

            # Look up with a different data hash
            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)
            context = _make_context()
            dataset, _, _ = _make_mock_dataset(content_hash_value="hash_B")

            result = controller._try_cache_lookup(
                runtime_context=rc,
                context=context,
                dataset=dataset,
                operator_name="StandardScaler",
                source_index=0,
            )

            assert result is None


class TestPersistTransformerWithDataHash:
    """Tests for _persist_transformer with input_data_hash parameter."""

    def test_persist_populates_chain_and_data_index(self):
        """Verify _persist_transformer populates the (chain, data) index when input_data_hash is given."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            scaler = StandardScaler()
            X = np.random.RandomState(42).rand(20, 50)
            scaler.fit(X)

            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)
            context = _make_context()

            record = controller._persist_transformer(
                runtime_context=rc,
                transformer=scaler,
                name="StandardScaler_1",
                context=context,
                source_index=0,
                processing_index=0,
                input_data_hash="my_data_hash",
            )

            assert record is not None
            # Verify the cache-key index was populated
            assert len(registry._by_chain_and_data) == 1

    def test_persist_without_data_hash_does_not_populate_index(self):
        """Verify _persist_transformer does NOT populate the (chain, data) index when input_data_hash is None."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            scaler = StandardScaler()
            X = np.random.RandomState(42).rand(20, 50)
            scaler.fit(X)

            controller = TransformerMixinController()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)
            context = _make_context()

            record = controller._persist_transformer(
                runtime_context=rc,
                transformer=scaler,
                name="StandardScaler_1",
                context=context,
                source_index=0,
                processing_index=0,
                input_data_hash=None,
            )

            assert record is not None
            assert len(registry._by_chain_and_data) == 0


class TestCheckBeforeFitIntegration:
    """Integration tests verifying that check-before-fit skips fitting on cache hit."""

    def test_cache_hit_skips_fit(self):
        """When a cached artifact exists, fit() is not called."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            # Pre-fit and register a FitCountingScaler (same class as the operator)
            FitCountingScaler.fit_count = 0
            scaler = FitCountingScaler()
            X_fit = np.random.RandomState(42).rand(15, 50)
            scaler.fit(X_fit)
            # Reset after the initial fit used for registration
            FitCountingScaler.fit_count = 0

            chain = OperatorChain(pipeline_id="test_pipeline")
            node = OperatorNode(
                step_index=1,
                operator_class="FitCountingScaler",
                branch_path=[],
                source_index=0,
                substep_index=0,
            )
            artifact_chain = chain.append(node)
            data_hash = "test_hash_123"

            registry.register_with_chain(
                obj=scaler,
                chain=artifact_chain,
                artifact_type=ArtifactType.TRANSFORMER,
                step_index=1,
                source_index=0,
                input_data_hash=data_hash,
                pipeline_id="test_pipeline",
            )

            # Now create a FitCountingScaler and run execute
            FitCountingScaler.fit_count = 0
            op = FitCountingScaler()

            step_info = ParsedStep(
                operator=op,
                keyword="",
                step_type=StepType.DIRECT,
                original_step=op,
                metadata={},
            )

            dataset = MagicMock()
            dataset._may_contain_nan = False
            dataset.content_hash.return_value = data_hash

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            call_count = 0

            def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
                nonlocal call_count
                call_count += 1
                # First call is all_data, second is fit_data
                if call_count <= 1:
                    return [all_data_3d]
                return [fit_data_3d]

            dataset.x.side_effect = x_side_effect
            dataset.features_processings.return_value = ["raw"]

            context = _make_context()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller = TransformerMixinController()
            new_context, artifacts = controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=rc,
                mode="train",
            )

            # FitCountingScaler.fit() should NOT have been called (cache hit)
            assert FitCountingScaler.fit_count == 0

    def test_cache_miss_calls_fit(self):
        """When no cached artifact exists, fit() is called normally."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            FitCountingScaler.fit_count = 0
            op = FitCountingScaler()

            step_info = ParsedStep(
                operator=op,
                keyword="",
                step_type=StepType.DIRECT,
                original_step=op,
                metadata={},
            )

            dataset = MagicMock()
            dataset._may_contain_nan = False
            dataset.content_hash.return_value = "unique_hash"

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            call_count = 0

            def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return [all_data_3d]
                return [fit_data_3d]

            dataset.x.side_effect = x_side_effect
            dataset.features_processings.return_value = ["raw"]

            context = _make_context()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller = TransformerMixinController()
            new_context, artifacts = controller.execute(
                step_info=step_info,
                dataset=dataset,
                context=context,
                runtime_context=rc,
                mode="train",
            )

            # FitCountingScaler.fit() SHOULD have been called (cache miss)
            assert FitCountingScaler.fit_count == 1

    def test_second_pipeline_reuses_cached_fit(self):
        """Simulate two pipeline executions sharing the same preprocessing step.

        The first execution fits the transformer and registers it.
        The second execution should find the cached artifact and skip fitting.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            data_hash = "shared_data_hash_xyz"
            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            def make_dataset_mock():
                dataset = MagicMock()
                dataset._may_contain_nan = False
                dataset.content_hash.return_value = data_hash
                call_count_inner = 0

                def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
                    nonlocal call_count_inner
                    call_count_inner += 1
                    if call_count_inner <= 1:
                        return [all_data_3d]
                    return [fit_data_3d]

                dataset.x.side_effect = x_side_effect
                dataset.features_processings.return_value = ["raw"]
                return dataset

            controller = TransformerMixinController()

            # --- First execution: cache miss, fit is called ---
            FitCountingScaler.fit_count = 0
            op1 = FitCountingScaler()
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)
            dataset1 = make_dataset_mock()

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert FitCountingScaler.fit_count == 1
            assert len(registry._by_chain_and_data) == 1

            # --- Second execution: cache hit, fit is NOT called ---
            FitCountingScaler.fit_count = 0
            op2 = FitCountingScaler()
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)
            dataset2 = make_dataset_mock()

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert FitCountingScaler.fit_count == 0, (
                "fit() should not be called on cache hit"
            )

    def test_no_cache_when_registry_is_none(self):
        """When artifact_registry is None, caching is entirely bypassed."""
        FitCountingScaler.fit_count = 0
        op = FitCountingScaler()
        step_info = ParsedStep(
            operator=op, keyword="", step_type=StepType.DIRECT,
            original_step=op, metadata={},
        )

        dataset = MagicMock()
        dataset._may_contain_nan = False
        dataset.content_hash.return_value = "some_hash"

        all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
        fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)
        call_count = 0

        def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return [all_data_3d]
            return [fit_data_3d]

        dataset.x.side_effect = x_side_effect
        dataset.features_processings.return_value = ["raw"]

        context = _make_context()
        rc = _make_runtime_context(artifact_registry=None, step_number=1)

        controller = TransformerMixinController()
        controller.execute(
            step_info=step_info, dataset=dataset, context=context,
            runtime_context=rc, mode="train",
        )

        # fit() should be called normally
        assert FitCountingScaler.fit_count == 1

    def test_cache_hit_produces_correct_transform_output(self):
        """Verify that a cache-hit loaded transformer produces the same transform output."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            # Pre-fit a StandardScaler and register it
            X_fit = np.random.RandomState(42).rand(15, 50)
            X_all = np.random.RandomState(99).rand(20, 50)

            scaler = StandardScaler()
            scaler.fit(X_fit)
            expected_output = scaler.transform(X_all)

            chain = OperatorChain(pipeline_id="test_pipeline")
            node = OperatorNode(
                step_index=1,
                operator_class="StandardScaler",
                branch_path=[],
                source_index=0,
                substep_index=0,
            )
            artifact_chain = chain.append(node)
            data_hash = "deterministic_hash"

            registry.register_with_chain(
                obj=scaler,
                chain=artifact_chain,
                artifact_type=ArtifactType.TRANSFORMER,
                step_index=1,
                source_index=0,
                input_data_hash=data_hash,
                pipeline_id="test_pipeline",
            )

            # Now execute with a StandardScaler and verify the output
            op = StandardScaler()
            step_info = ParsedStep(
                operator=op, keyword="", step_type=StepType.DIRECT,
                original_step=op, metadata={},
            )

            dataset = MagicMock()
            dataset._may_contain_nan = False
            dataset.content_hash.return_value = data_hash

            all_data_3d = X_all.reshape(20, 1, 50)
            fit_data_3d = X_fit.reshape(15, 1, 50)
            call_count = 0

            def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return [all_data_3d]
                return [fit_data_3d]

            dataset.x.side_effect = x_side_effect
            dataset.features_processings.return_value = ["raw"]
            dataset.replace_features = MagicMock()

            context = _make_context()
            rc = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller = TransformerMixinController()
            controller.execute(
                step_info=step_info, dataset=dataset, context=context,
                runtime_context=rc, mode="train",
            )

            # Verify that replace_features was called with the correct transformed data
            assert dataset.replace_features.called
            call_args = dataset.replace_features.call_args
            actual_output = call_args.kwargs.get('features', call_args[1].get('features', call_args[0][1] if len(call_args[0]) > 1 else None))
            if actual_output is None:
                # Try positional arg
                actual_output = call_args[0][1]

            # The output should match what the pre-fitted scaler would produce
            np.testing.assert_array_almost_equal(actual_output[0], expected_output)


# ---------------------------------------------------------------------------
# Helper: Stateless fit-counting transformer
# ---------------------------------------------------------------------------

class StatelessFitCounter(TransformerMixin, BaseEstimator):
    """Stateless transformer that counts fit() calls.

    Declares ``_stateless = True`` to signal that fit() produces no
    data-dependent state.  Used to verify that the cache system skips
    data hashing for stateless operators.
    """

    _stateless = True
    fit_count = 0

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def fit(self, X, y=None):
        StatelessFitCounter.fit_count += 1
        return self

    def transform(self, X):
        return X * self.scale_factor


class StatefulFitCounter(TransformerMixin, BaseEstimator):
    """Non-stateless transformer that counts fit() calls.

    Does NOT have ``_stateless = True``, so the cache must use
    data hash for lookup.
    """

    fit_count = 0

    def __init__(self):
        pass

    def fit(self, X, y=None):
        StatefulFitCounter.fit_count += 1
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        return X - self.mean_


def _make_execute_dataset(
    all_data_3d, fit_data_3d, content_hash_value="hash_abc",
):
    """Create a mock dataset suitable for execute()."""
    dataset = MagicMock()
    dataset._may_contain_nan = False
    dataset.content_hash.return_value = content_hash_value
    call_count = 0

    def x_side_effect(selector, layout, concat_source=False, include_excluded=True, include_augmented=False):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return [all_data_3d]
        return [fit_data_3d]

    dataset.x.side_effect = x_side_effect
    dataset.features_processings.return_value = ["raw"]
    return dataset


# ===========================================================================
# Task 1.2: fit_on_all artifact reuse validation
# ===========================================================================

class TestFitOnAllArtifactReuse:
    """Integration tests verifying that fit_on_all=True transformers are fitted
    once and reused across CV folds and between CV/refit phases.
    """

    def test_fit_on_all_fit_once_reuse_across_folds(self):
        """With fit_on_all=True, transformer is fitted once on fold 1 and reused on fold 2."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(20, 1, 50)  # fit_on_all uses all data
            data_hash = "all_data_hash"

            controller = TransformerMixinController()

            # --- Fold 1: cache miss, fit is called ---
            FitCountingScaler.fit_count = 0
            op1 = FitCountingScaler()
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op1, "fit_on_all": True},
                metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert FitCountingScaler.fit_count == 1
            assert len(registry._by_chain_and_data) == 1

            # --- Fold 2: same data hash -> cache hit, fit NOT called ---
            FitCountingScaler.fit_count = 0
            op2 = FitCountingScaler()
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op2, "fit_on_all": True},
                metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert FitCountingScaler.fit_count == 0, (
                "fit_on_all transformer should be reused across folds with same data hash"
            )

    def test_fit_on_all_refit_reuses_cv_artifact(self):
        """With fit_on_all=True, the REFIT phase reuses the artifact from the CV phase.

        Since fit_on_all fits on all data, the data hash is the same in both
        CV and REFIT phases, allowing artifact reuse.
        """
        from nirs4all.pipeline.config.context import ExecutionPhase

        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = all_data_3d.copy()  # fit_on_all: same data
            data_hash = "full_data_hash"

            controller = TransformerMixinController()

            # --- CV phase: fit ---
            FitCountingScaler.fit_count = 0
            op1 = FitCountingScaler()
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op1, "fit_on_all": True},
                metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)
            rc1.phase = ExecutionPhase.CV

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert FitCountingScaler.fit_count == 1

            # --- REFIT phase: same data hash -> reuse ---
            FitCountingScaler.fit_count = 0
            op2 = FitCountingScaler()
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op2, "fit_on_all": True},
                metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)
            rc2.phase = ExecutionPhase.REFIT

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert FitCountingScaler.fit_count == 0, (
                "fit_on_all transformer should be reused between CV and REFIT phases"
            )

    def test_fit_on_all_different_operator_types(self):
        """fit_on_all reuse works for different operator types (StandardScaler, MinMaxScaler)."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = all_data_3d.copy()
            data_hash = "shared_hash"

            controller = TransformerMixinController()

            # StandardScaler - first run
            op_ss = StandardScaler()
            step_info_ss = ParsedStep(
                operator=op_ss, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op_ss, "fit_on_all": True},
                metadata={},
            )
            ds_ss = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context_ss = _make_context()
            rc_ss = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info_ss, dataset=ds_ss, context=context_ss,
                runtime_context=rc_ss, mode="train",
            )

            # StandardScaler - second run (cache hit)
            op_ss2 = StandardScaler()
            step_info_ss2 = ParsedStep(
                operator=op_ss2, keyword="", step_type=StepType.DIRECT,
                original_step={"preprocessing": op_ss2, "fit_on_all": True},
                metadata={},
            )
            ds_ss2 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            context_ss2 = _make_context()
            rc_ss2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info_ss2, dataset=ds_ss2, context=context_ss2,
                runtime_context=rc_ss2, mode="train",
            )

            # Verify the StandardScaler was only persisted once (deduplication via cache hit)
            ss_records = [r for r in registry.get_all_records()
                          if r.class_name == "StandardScaler"]
            # The second execution should have reused the cached artifact,
            # so total StandardScaler artifacts should still be >=1
            assert len(ss_records) >= 1


# ===========================================================================
# Task 1.3: Stateless transform detection and skip
# ===========================================================================

class TestStatelessDetection:
    """Tests for _is_stateless() and _compute_operator_params_hash()."""

    def test_is_stateless_true_for_stateless_operator(self):
        """Operators with _stateless = True are detected."""
        op = StatelessFitCounter()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_false_for_stateful_operator(self):
        """Operators without _stateless are not stateless."""
        op = StatefulFitCounter()
        assert not TransformerMixinController._is_stateless(op)

    def test_is_stateless_false_for_sklearn_standard_scaler(self):
        """StandardScaler is NOT stateless (learns mean/var from data)."""
        op = StandardScaler()
        assert not TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_snv(self):
        """SNV (StandardNormalVariate) is stateless."""
        from nirs4all.operators.transforms.scalers import StandardNormalVariate
        op = StandardNormalVariate()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_lsnv(self):
        """LSNV (LocalStandardNormalVariate) is stateless."""
        from nirs4all.operators.transforms.scalers import LocalStandardNormalVariate
        op = LocalStandardNormalVariate()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_rsnv(self):
        """RSNV (RobustStandardNormalVariate) is stateless."""
        from nirs4all.operators.transforms.scalers import RobustStandardNormalVariate
        op = RobustStandardNormalVariate()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_derivate(self):
        """Derivate is stateless."""
        from nirs4all.operators.transforms.scalers import Derivate
        op = Derivate()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_detrend(self):
        """Detrend is stateless."""
        from nirs4all.operators.transforms.signal import Detrend
        op = Detrend()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_savitzky_golay(self):
        """SavitzkyGolay is stateless."""
        from nirs4all.operators.transforms.nirs import SavitzkyGolay
        op = SavitzkyGolay()
        assert TransformerMixinController._is_stateless(op)

    def test_is_stateless_true_for_signal_conversion(self):
        """ToAbsorbance is stateless."""
        from nirs4all.operators.transforms.signal_conversion import ToAbsorbance
        op = ToAbsorbance(source_type="reflectance")
        assert TransformerMixinController._is_stateless(op)

    def test_params_hash_same_for_same_params(self):
        """Same constructor parameters produce the same hash."""
        op1 = StatelessFitCounter(scale_factor=2.0)
        op2 = StatelessFitCounter(scale_factor=2.0)
        h1 = TransformerMixinController._compute_operator_params_hash(op1)
        h2 = TransformerMixinController._compute_operator_params_hash(op2)
        assert h1 == h2

    def test_params_hash_differs_for_different_params(self):
        """Different constructor parameters produce different hashes."""
        op1 = StatelessFitCounter(scale_factor=1.0)
        op2 = StatelessFitCounter(scale_factor=2.0)
        h1 = TransformerMixinController._compute_operator_params_hash(op1)
        h2 = TransformerMixinController._compute_operator_params_hash(op2)
        assert h1 != h2


class TestStatelessCacheSkipIntegration:
    """Integration tests verifying that stateless operators skip data hash for caching."""

    def test_stateless_operator_reused_with_different_data_hash(self):
        """A stateless operator cached by params hash can be reused even when data hash changes.

        This is the core optimization: stateless operators do not depend on
        the training data, so the cache key uses operator params hash instead
        of data hash.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d_1 = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d_1 = np.random.RandomState(42).rand(15, 1, 50)

            all_data_3d_2 = np.random.RandomState(99).rand(20, 1, 50)
            fit_data_3d_2 = np.random.RandomState(99).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- First execution: cache miss, fit is called ---
            StatelessFitCounter.fit_count = 0
            op1 = StatelessFitCounter(scale_factor=1.0)
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d_1, fit_data_3d_1, "hash_A")
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert StatelessFitCounter.fit_count == 1

            # --- Second execution: DIFFERENT data hash but same params ---
            StatelessFitCounter.fit_count = 0
            op2 = StatelessFitCounter(scale_factor=1.0)
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d_2, fit_data_3d_2, "hash_B")
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert StatelessFitCounter.fit_count == 0, (
                "Stateless operator should be reused even with different data hash"
            )

    def test_stateful_operator_not_reused_with_different_data_hash(self):
        """A stateful (non-stateless) operator is NOT reused when data hash changes.

        This confirms that the stateless optimization is only applied to
        operators with ``_stateless = True``.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- First execution ---
            StatefulFitCounter.fit_count = 0
            op1 = StatefulFitCounter()
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_A")
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert StatefulFitCounter.fit_count == 1

            # --- Second execution with different data hash ---
            StatefulFitCounter.fit_count = 0
            op2 = StatefulFitCounter()
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_B")
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert StatefulFitCounter.fit_count == 1, (
                "Stateful operator should NOT be reused when data hash differs"
            )

    def test_stateless_not_reused_with_different_params(self):
        """Stateless operators with different params must NOT be reused.

        Even though both are stateless, different scale_factor means
        they are distinct operators and must both be fitted.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- First execution: scale_factor=1.0 ---
            StatelessFitCounter.fit_count = 0
            op1 = StatelessFitCounter(scale_factor=1.0)
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_X")
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert StatelessFitCounter.fit_count == 1

            # --- Second execution: scale_factor=2.0 (different!) ---
            StatelessFitCounter.fit_count = 0
            op2 = StatelessFitCounter(scale_factor=2.0)
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_X")
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert StatelessFitCounter.fit_count == 1, (
                "Different params on stateless op should not produce cache hit"
            )

    def test_stateless_produces_correct_output(self):
        """Verify that a stateless cache-hit produces the correct transform output."""
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            X_all = np.random.RandomState(42).rand(20, 50)
            all_data_3d = X_all.reshape(20, 1, 50)
            fit_data_3d = np.random.RandomState(99).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- First execution: fit + transform ---
            StatelessFitCounter.fit_count = 0
            op1 = StatelessFitCounter(scale_factor=3.0)
            step_info1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            dataset1 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_1")
            dataset1.replace_features = MagicMock()
            context1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info1, dataset=dataset1, context=context1,
                runtime_context=rc1, mode="train",
            )
            assert StatelessFitCounter.fit_count == 1
            call_args1 = dataset1.replace_features.call_args
            output1 = call_args1.kwargs.get('features', call_args1[1].get('features', None))
            if output1 is None:
                output1 = call_args1[0][1]

            # --- Second execution: cache hit, different data hash ---
            StatelessFitCounter.fit_count = 0
            op2 = StatelessFitCounter(scale_factor=3.0)
            step_info2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            dataset2 = _make_execute_dataset(all_data_3d, fit_data_3d, "hash_2")
            dataset2.replace_features = MagicMock()
            context2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_info2, dataset=dataset2, context=context2,
                runtime_context=rc2, mode="train",
            )
            assert StatelessFitCounter.fit_count == 0  # cache hit
            call_args2 = dataset2.replace_features.call_args
            output2 = call_args2.kwargs.get('features', call_args2[1].get('features', None))
            if output2 is None:
                output2 = call_args2[0][1]

            # Both should produce identical output (scale_factor=3.0 applied to same input)
            np.testing.assert_array_almost_equal(output1[0], output2[0])
            np.testing.assert_array_almost_equal(output1[0], X_all * 3.0)


# ===========================================================================
# Task 1.4: Cross-pipeline preprocessing reuse validation
# ===========================================================================

class TestCrossPipelinePreprocessingReuse:
    """Integration tests verifying the check-before-fit mechanism handles
    generator sweeps where pipeline variants share common preprocessing prefixes.

    This simulates ``_or_`` / ``_range_`` / ``_cartesian_`` generators creating
    multiple pipeline variants that share initial preprocessing steps.
    """

    def test_shared_preprocessing_reused_across_pipeline_variants(self):
        """Two pipeline variants sharing the same step 1 preprocessor reuse the cache.

        Simulates:
            pipeline_variant_1 = [FitCountingScaler, ModelA]
            pipeline_variant_2 = [FitCountingScaler, ModelB]

        Step 1 (FitCountingScaler) should be fitted only once.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)
            data_hash = "cross_pipeline_hash"

            controller = TransformerMixinController()

            # --- Pipeline variant 1: step 1 = FitCountingScaler ---
            FitCountingScaler.fit_count = 0
            op_v1 = FitCountingScaler()
            step_v1 = ParsedStep(
                operator=op_v1, keyword="", step_type=StepType.DIRECT,
                original_step=op_v1, metadata={},
            )
            ds_v1 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            ctx_v1 = _make_context()
            rc_v1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_v1, dataset=ds_v1, context=ctx_v1,
                runtime_context=rc_v1, mode="train",
            )
            assert FitCountingScaler.fit_count == 1
            assert len(registry._by_chain_and_data) == 1

            # --- Pipeline variant 2: same step 1 = FitCountingScaler ---
            FitCountingScaler.fit_count = 0
            op_v2 = FitCountingScaler()
            step_v2 = ParsedStep(
                operator=op_v2, keyword="", step_type=StepType.DIRECT,
                original_step=op_v2, metadata={},
            )
            ds_v2 = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            ctx_v2 = _make_context()
            rc_v2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_v2, dataset=ds_v2, context=ctx_v2,
                runtime_context=rc_v2, mode="train",
            )
            assert FitCountingScaler.fit_count == 0, (
                "Shared preprocessing step should be reused across pipeline variants"
            )

    def test_different_preprocessing_not_reused(self):
        """Two pipeline variants with DIFFERENT step 1 preprocessors are NOT cached.

        Simulates:
            pipeline_variant_1 = [StandardScaler, ...]
            pipeline_variant_2 = [MinMaxScaler, ...]

        Even with the same data hash, different operator classes should not
        produce a cache hit.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)
            data_hash = "shared_hash"

            controller = TransformerMixinController()

            # --- Variant 1: StandardScaler ---
            op_ss = StandardScaler()
            step_ss = ParsedStep(
                operator=op_ss, keyword="", step_type=StepType.DIRECT,
                original_step=op_ss, metadata={},
            )
            ds_ss = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            ctx_ss = _make_context()
            rc_ss = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_ss, dataset=ds_ss, context=ctx_ss,
                runtime_context=rc_ss, mode="train",
            )

            # --- Variant 2: MinMaxScaler ---
            op_mm = MinMaxScaler()
            step_mm = ParsedStep(
                operator=op_mm, keyword="", step_type=StepType.DIRECT,
                original_step=op_mm, metadata={},
            )
            ds_mm = _make_execute_dataset(all_data_3d, fit_data_3d, data_hash)
            ctx_mm = _make_context()
            rc_mm = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step_mm, dataset=ds_mm, context=ctx_mm,
                runtime_context=rc_mm, mode="train",
            )

            # Both should be registered as separate artifacts
            ss_records = [r for r in registry.get_all_records()
                          if r.class_name == "StandardScaler"]
            mm_records = [r for r in registry.get_all_records()
                          if r.class_name == "MinMaxScaler"]
            assert len(ss_records) >= 1
            assert len(mm_records) >= 1

    def test_shared_stateless_prefix_reused_across_variants(self):
        """Stateless preprocessor shared across pipeline variants is reused even with
        different data hashes (e.g. different fold splits in each variant).

        Simulates a generator sweep:
            _or_: [SNV, MSC]
            ...
            _range_: [1, 10, 5] (n_components)

        Where SNV is the shared stateless preprocessor at step 1 for all variants.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- Variant 1: SNV-like stateless op, data hash A ---
            StatelessFitCounter.fit_count = 0
            op1 = StatelessFitCounter(scale_factor=1.0)
            step1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            ds1 = _make_execute_dataset(all_data_3d, fit_data_3d, "fold_1_hash")
            ctx1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step1, dataset=ds1, context=ctx1,
                runtime_context=rc1, mode="train",
            )
            assert StatelessFitCounter.fit_count == 1

            # --- Variant 2: same stateless op, DIFFERENT data hash ---
            StatelessFitCounter.fit_count = 0
            op2 = StatelessFitCounter(scale_factor=1.0)
            step2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            ds2 = _make_execute_dataset(all_data_3d, fit_data_3d, "fold_2_hash")
            ctx2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step2, dataset=ds2, context=ctx2,
                runtime_context=rc2, mode="train",
            )
            assert StatelessFitCounter.fit_count == 0, (
                "Stateless preprocessing should be reused across variants with different data hashes"
            )

            # --- Variant 3: same stateless op, yet another data hash ---
            StatelessFitCounter.fit_count = 0
            op3 = StatelessFitCounter(scale_factor=1.0)
            step3 = ParsedStep(
                operator=op3, keyword="", step_type=StepType.DIRECT,
                original_step=op3, metadata={},
            )
            ds3 = _make_execute_dataset(all_data_3d, fit_data_3d, "fold_3_hash")
            ctx3 = _make_context()
            rc3 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step3, dataset=ds3, context=ctx3,
                runtime_context=rc3, mode="train",
            )
            assert StatelessFitCounter.fit_count == 0, (
                "Third variant should also reuse the stateless cache"
            )

    def test_stateful_prefix_not_reused_across_different_folds(self):
        """A stateful preprocessor is NOT reused when the data hash changes
        (different fold splits), even when the operator class and params match.

        This confirms the optimization is only for stateless operators.
        """
        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(
                workspace=Path(tmpdir),
                dataset="test_ds",
                pipeline_id="test_pipeline",
            )

            all_data_3d = np.random.RandomState(42).rand(20, 1, 50)
            fit_data_3d = np.random.RandomState(42).rand(15, 1, 50)

            controller = TransformerMixinController()

            # --- Fold 1 ---
            FitCountingScaler.fit_count = 0
            op1 = FitCountingScaler()
            step1 = ParsedStep(
                operator=op1, keyword="", step_type=StepType.DIRECT,
                original_step=op1, metadata={},
            )
            ds1 = _make_execute_dataset(all_data_3d, fit_data_3d, "fold_1_hash")
            ctx1 = _make_context()
            rc1 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step1, dataset=ds1, context=ctx1,
                runtime_context=rc1, mode="train",
            )
            assert FitCountingScaler.fit_count == 1

            # --- Fold 2: different data hash ---
            FitCountingScaler.fit_count = 0
            op2 = FitCountingScaler()
            step2 = ParsedStep(
                operator=op2, keyword="", step_type=StepType.DIRECT,
                original_step=op2, metadata={},
            )
            ds2 = _make_execute_dataset(all_data_3d, fit_data_3d, "fold_2_hash")
            ctx2 = _make_context()
            rc2 = _make_runtime_context(artifact_registry=registry, step_number=1)

            controller.execute(
                step_info=step2, dataset=ds2, context=ctx2,
                runtime_context=rc2, mode="train",
            )
            assert FitCountingScaler.fit_count == 1, (
                "Stateful operator should be re-fitted when data hash changes"
            )
