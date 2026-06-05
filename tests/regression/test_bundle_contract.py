"""Contract snapshot: the ``.n4a`` bundle format.

The ``.n4a`` bundle format is a stable 0.9.x contract (bundles are exported,
shared, and re-loaded for prediction). This file freezes two things:

  * ``BUNDLE_FORMAT_VERSION`` — the version string embedded in every bundle
    manifest. A bump here is a deliberate format-version change.
  * The top-level key set of the bundle manifest produced by
    ``BundleGenerator._create_bundle_manifest`` — the keys a loader relies on.

The manifest is produced cheaply (no pipeline run): a minimal
``ResolvedPrediction`` is hand-constructed from a small ``ExecutionTrace`` and a
mock artifact provider — the same lightweight fixture style used by the
existing bundle unit tests. We snapshot three cases:

  * the *core* keys that are always present (``include_metadata=False`` and no
    trace), via a subset assertion so the core contract cannot shrink;
  * the *non-partitioner full* keys present when metadata + a trace WITHOUT a
    metadata-partitioner step are supplied, asserted exactly so the conditional
    keys (``original_manifest``, ``trace_id``) are pinned and ``partitioner_routing``
    is confirmed absent;
  * the *partitioner full* keys, where the trace contains a
    ``MetadataPartitionerController`` step carrying routing metadata, asserted
    exactly so the additional optional ``partitioner_routing`` key is pinned.

``partitioner_routing`` is an additional optional manifest key that
``BundleGenerator`` emits only for traces that include a metadata-partitioner
step (``generator.py``; the loader reads it back as manifest metadata). It is
therefore part of the manifest contract even though it is absent from the
common single-branch case.

Snapshot captured from nirs4all 0.9.1.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from nirs4all.pipeline.bundle import BundleGenerator
from nirs4all.pipeline.bundle.generator import BUNDLE_FORMAT_VERSION
from nirs4all.pipeline.resolver import FoldStrategy, ResolvedPrediction, SourceType
from nirs4all.pipeline.trace import ExecutionStep, ExecutionTrace, StepArtifacts

# Frozen bundle format version (captured from nirs4all 0.9.1).
EXPECTED_BUNDLE_FORMAT_VERSION = "1.0"

# Core manifest keys always emitted, regardless of metadata/trace presence.
EXPECTED_CORE_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "bundle_format_version",
        "created_at",
        "fold_strategy",
        "model_step_index",
        "nirs4all_version",
        "pipeline_uid",
        "preprocessing_chain",
        "source_type",
    }
)

# Full manifest keys when metadata + a non-partitioner trace are supplied.
EXPECTED_FULL_MANIFEST_KEYS: frozenset[str] = EXPECTED_CORE_MANIFEST_KEYS | {
    "original_manifest",
    "trace_id",
}

# Full manifest keys when the trace contains a metadata-partitioner step:
# the optional ``partitioner_routing`` key is additionally emitted.
EXPECTED_PARTITIONER_MANIFEST_KEYS: frozenset[str] = EXPECTED_FULL_MANIFEST_KEYS | {
    "partitioner_routing",
}


def _minimal_resolved_with_trace() -> ResolvedPrediction:
    """A minimal ResolvedPrediction carrying a trace and a manifest."""
    trace = ExecutionTrace(
        trace_id="contract_trace",
        pipeline_uid="0001_contract",
        model_step_index=4,
        fold_weights={0: 0.5, 1: 0.5},
        preprocessing_chain="SNV>SG",
    )
    step_transform = ExecutionStep(step_index=1, operator_type="transform", operator_class="SNV")
    step_transform.artifacts = StepArtifacts(artifact_ids=["0001:1:all"])
    step_model = ExecutionStep(step_index=4, operator_type="model", operator_class="PLSRegression")
    step_model.artifacts = StepArtifacts(
        artifact_ids=["0001:4:0", "0001:4:1"],
        fold_artifact_ids={"0": "0001:4:0", "1": "0001:4:1"},
    )
    trace.add_step(step_transform)
    trace.add_step(step_model)

    provider = MagicMock()
    provider.get_artifacts_for_step.return_value = []

    return ResolvedPrediction(
        source_type=SourceType.PREDICTION,
        minimal_pipeline=[{"transform": "SNV"}, {"model": "PLSRegression"}],
        artifact_provider=provider,
        trace=trace,
        fold_strategy=FoldStrategy.WEIGHTED_AVERAGE,
        fold_weights={0: 0.5, 1: 0.5},
        model_step_index=4,
        pipeline_uid="0001_contract",
        manifest={"dataset": "wheat", "name": "pls_contract"},
    )


def _minimal_resolved_with_partitioner_trace() -> ResolvedPrediction:
    """A minimal ResolvedPrediction whose trace contains a metadata partitioner.

    ``BundleGenerator._extract_partitioner_routing_info`` only returns routing
    (and the manifest only gains ``partitioner_routing``) when the trace has a
    step whose ``operator_class`` is ``"MetadataPartitionerController"`` and that
    step carries partition metadata. This fixture supplies exactly that so the
    optional manifest key is exercised cheaply (no pipeline run).
    """
    trace = ExecutionTrace(
        trace_id="contract_trace_partitioner",
        pipeline_uid="0001_partitioner",
        model_step_index=4,
        preprocessing_chain="SNV>SG",
    )
    step_partitioner = ExecutionStep(
        step_index=2,
        operator_type="branch",
        operator_class="MetadataPartitionerController",
    )
    step_partitioner.metadata = {
        "column": "origin",
        "partitions": ["A", "B"],
        "branch_count": 2,
        "group_values": ["A", "B"],
        "min_samples": 1,
    }
    step_model = ExecutionStep(step_index=4, operator_type="model", operator_class="PLSRegression")
    step_model.artifacts = StepArtifacts(artifact_ids=["0001:4:all"])
    trace.add_step(step_partitioner)
    trace.add_step(step_model)

    provider = MagicMock()
    provider.get_artifacts_for_step.return_value = []

    return ResolvedPrediction(
        source_type=SourceType.PREDICTION,
        minimal_pipeline=[{"branch": "by_metadata"}, {"model": "PLSRegression"}],
        artifact_provider=provider,
        trace=trace,
        fold_strategy=FoldStrategy.WEIGHTED_AVERAGE,
        model_step_index=4,
        pipeline_uid="0001_partitioner",
        manifest={"dataset": "wheat", "name": "pls_partitioner"},
    )


def _minimal_resolved_without_trace() -> ResolvedPrediction:
    """A minimal ResolvedPrediction with no trace and no manifest."""
    provider = MagicMock()
    provider.get_artifacts_for_step.return_value = []
    return ResolvedPrediction(
        source_type=SourceType.PREDICTION,
        minimal_pipeline=[{"model": "PLSRegression"}],
        artifact_provider=provider,
        trace=None,
        fold_strategy=FoldStrategy.WEIGHTED_AVERAGE,
        fold_weights={},
        model_step_index=0,
        pipeline_uid="0001_minimal",
        manifest={},
    )


def test_bundle_format_version_frozen() -> None:
    """``BUNDLE_FORMAT_VERSION`` matches the frozen value."""
    assert BUNDLE_FORMAT_VERSION == EXPECTED_BUNDLE_FORMAT_VERSION


def test_core_manifest_keys_present(tmp_path: Path) -> None:
    """The always-present core manifest keys are a subset of every manifest."""
    generator = BundleGenerator(tmp_path)
    manifest = generator._create_bundle_manifest(
        _minimal_resolved_without_trace(), include_metadata=False
    )
    missing = EXPECTED_CORE_MANIFEST_KEYS - set(manifest.keys())
    assert not missing, f"bundle manifest dropped core keys: {sorted(missing)}"


def test_manifest_keys_without_partitioner_routing(tmp_path: Path) -> None:
    """Metadata + a non-partitioner trace emit exactly the frozen full key set.

    This is the common single-branch case: no metadata-partitioner step in the
    trace, so ``partitioner_routing`` must NOT appear. The exact-equality check
    pins both the conditional keys (``original_manifest``, ``trace_id``) and the
    absence of the optional ``partitioner_routing`` key.
    """
    generator = BundleGenerator(tmp_path)
    manifest = generator._create_bundle_manifest(
        _minimal_resolved_with_trace(), include_metadata=True
    )
    assert set(manifest.keys()) == EXPECTED_FULL_MANIFEST_KEYS
    assert "partitioner_routing" not in manifest
    assert manifest["bundle_format_version"] == EXPECTED_BUNDLE_FORMAT_VERSION


def test_manifest_keys_with_partitioner_routing(tmp_path: Path) -> None:
    """A metadata-partitioner trace additionally emits ``partitioner_routing``.

    ``partitioner_routing`` is an optional manifest key the loader consumes as
    manifest metadata. It is emitted only when the trace contains a
    ``MetadataPartitionerController`` step with routing metadata; freezing this
    case guards that contract key so a regression that stops emitting it (or
    renames it) is caught.
    """
    generator = BundleGenerator(tmp_path)
    manifest = generator._create_bundle_manifest(
        _minimal_resolved_with_partitioner_trace(), include_metadata=True
    )
    assert set(manifest.keys()) == EXPECTED_PARTITIONER_MANIFEST_KEYS
    assert "partitioner_routing" in manifest
    assert manifest["bundle_format_version"] == EXPECTED_BUNDLE_FORMAT_VERSION
