"""
Unit tests for Phase 4 edge cases - V3 artifact system.

Tests cover all edge cases from Section 1.2 of ARTIFACT_SYSTEM_V3_DESIGN.md:
- Multi-source reload
- Branching and nested branches
- Subpipeline models
- Meta-model stacking
- Bundle import with chain merging
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nirs4all.pipeline.storage.artifacts.operator_chain import (
    OperatorChain,
    OperatorNode,
    compute_chain_hash,
    generate_artifact_id_v3,
    is_v3_artifact_id,
    parse_artifact_id_v3,
)


class TestOperatorChainMerging:
    """Tests for chain merging functionality for bundle import."""

    def test_merge_with_prefix_basic(self):
        """Test basic chain merging with prefix."""
        # Bundle chain: s1.Scaler>s3.PLS
        bundle_chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Scaler"),
                OperatorNode(step_index=3, operator_class="PLS"),
            ],
            pipeline_id="bundle_123"
        )

        # Import context chain: s1.Import
        import_chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="Import")],
            pipeline_id="pipeline_456"
        )

        # Merge with step offset of 1
        merged = bundle_chain.merge_with_prefix(import_chain, step_offset=1)

        assert len(merged.nodes) == 3
        assert merged.nodes[0].operator_class == "Import"
        assert merged.nodes[0].step_index == 1
        assert merged.nodes[1].operator_class == "Scaler"
        assert merged.nodes[1].step_index == 2  # 1 + 1 offset
        assert merged.nodes[2].operator_class == "PLS"
        assert merged.nodes[2].step_index == 4  # 3 + 1 offset
        assert merged.pipeline_id == "pipeline_456"  # Uses prefix's pipeline_id

    def test_merge_with_prefix_preserves_branch_path(self):
        """Test that merging preserves branch paths."""
        bundle_chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="SNV", branch_path=[0]),
                OperatorNode(step_index=2, operator_class="PLS", branch_path=[0]),
            ],
            pipeline_id="bundle"
        )

        import_chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="Import")],
            pipeline_id="parent"
        )

        merged = bundle_chain.merge_with_prefix(import_chain, step_offset=2)

        assert merged.nodes[1].branch_path == [0]
        assert merged.nodes[2].branch_path == [0]

    def test_merge_with_prefix_preserves_source_index(self):
        """Test that merging preserves source indices for multi-source."""
        bundle_chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Scaler", source_index=0),
                OperatorNode(step_index=1, operator_class="Scaler", source_index=1),
                OperatorNode(step_index=1, operator_class="Scaler", source_index=2),
            ],
            pipeline_id="bundle"
        )

        import_chain = OperatorChain(nodes=[], pipeline_id="parent")

        merged = bundle_chain.merge_with_prefix(import_chain, step_offset=0)

        assert len(merged.nodes) == 3
        for i, node in enumerate(merged.nodes):
            assert node.source_index == i

    def test_merge_empty_prefix(self):
        """Test merging with empty prefix chain."""
        bundle_chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="PLS")],
            pipeline_id="bundle"
        )

        empty_prefix = OperatorChain(nodes=[], pipeline_id="parent")

        merged = bundle_chain.merge_with_prefix(empty_prefix, step_offset=0)

        assert len(merged.nodes) == 1
        assert merged.nodes[0].operator_class == "PLS"
        assert merged.pipeline_id == "parent"

class TestOperatorChainRemap:
    """Tests for chain step remapping."""

    def test_remap_steps_basic(self):
        """Test basic step remapping."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=3, operator_class="B"),
                OperatorNode(step_index=5, operator_class="C"),
            ],
            pipeline_id="test"
        )

        mapping = {1: 2, 3: 4, 5: 6}
        remapped = chain.remap_steps(mapping)

        assert remapped.nodes[0].step_index == 2
        assert remapped.nodes[1].step_index == 4
        assert remapped.nodes[2].step_index == 6

    def test_remap_steps_preserves_unmapped(self):
        """Test that unmapped steps are preserved."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=3, operator_class="B"),
            ],
            pipeline_id="test"
        )

        mapping = {1: 10}  # Only remap step 1
        remapped = chain.remap_steps(mapping)

        assert remapped.nodes[0].step_index == 10
        assert remapped.nodes[1].step_index == 3  # Unchanged

class TestOperatorChainWithPipelineId:
    """Tests for with_pipeline_id method."""

    def test_with_pipeline_id(self):
        """Test creating chain copy with new pipeline ID."""
        chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="A")],
            pipeline_id="old_pipeline"
        )

        new_chain = chain.with_pipeline_id("new_pipeline")

        assert new_chain.pipeline_id == "new_pipeline"
        assert chain.pipeline_id == "old_pipeline"  # Original unchanged
        assert len(new_chain.nodes) == 1

class TestMultiSourceChains:
    """Tests for multi-source artifact chain handling."""

    def test_multi_source_chain_path(self):
        """Test chain path for multi-source transformers."""
        # Create chains for 3 sources
        chains = []
        for src in range(3):
            node = OperatorNode(
                step_index=1,
                operator_class="MinMaxScaler",
                source_index=src
            )
            chain = OperatorChain(nodes=[node], pipeline_id="test")
            chains.append(chain)

        # Each chain should have unique path
        paths = [c.to_path() for c in chains]
        assert len(set(paths)) == 3  # All unique

        # Each path should contain source info
        for i, path in enumerate(paths):
            assert f"src={i}" in path

    def test_multi_source_filter(self):
        """Test filtering chain by source index."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Scaler", source_index=0),
                OperatorNode(step_index=1, operator_class="Scaler", source_index=1),
                OperatorNode(step_index=1, operator_class="Scaler", source_index=2),
                OperatorNode(step_index=2, operator_class="PLS"),  # No source_index
            ],
            pipeline_id="test"
        )

        filtered = chain.filter_source(1)

        assert len(filtered.nodes) == 2
        assert filtered.nodes[0].source_index == 1
        assert filtered.nodes[1].source_index is None  # Included

class TestBranchingChains:
    """Tests for branching pipeline chain handling."""

    def test_branch_path_in_chain(self):
        """Test branch path is correctly encoded in chain."""
        node = OperatorNode(
            step_index=3,
            operator_class="SNV",
            branch_path=[0]
        )

        key = node.to_key()
        assert "br=0" in key

    def test_nested_branch_path(self):
        """Test nested branch path encoding."""
        node = OperatorNode(
            step_index=3,
            operator_class="PLS",
            branch_path=[0, 1]
        )

        key = node.to_key()
        assert "br=0.1" in key

    def test_filter_by_branch(self):
        """Test filtering chain by branch path."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Shared"),  # No branch
                OperatorNode(step_index=3, operator_class="SNV", branch_path=[0]),
                OperatorNode(step_index=3, operator_class="MSC", branch_path=[1]),
                OperatorNode(step_index=4, operator_class="PLS", branch_path=[0]),
            ],
            pipeline_id="test"
        )

        # Filter for branch [0]
        filtered = chain.filter_branch([0])

        # Should include: shared (no branch), SNV (branch 0), PLS (branch 0)
        assert len(filtered.nodes) == 3
        assert filtered.nodes[0].operator_class == "Shared"
        assert filtered.nodes[1].operator_class == "SNV"
        assert filtered.nodes[2].operator_class == "PLS"

    def test_filter_nested_branch(self):
        """Test filtering with nested branches."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Shared"),
                OperatorNode(step_index=2, operator_class="A", branch_path=[0]),
                OperatorNode(step_index=3, operator_class="B", branch_path=[0, 0]),
                OperatorNode(step_index=3, operator_class="C", branch_path=[0, 1]),
                OperatorNode(step_index=4, operator_class="D", branch_path=[0, 0]),
            ],
            pipeline_id="test"
        )

        # Filter for branch [0, 0]
        filtered = chain.filter_branch([0, 0])

        # Should include: shared, A (parent [0]), B ([0,0]), D ([0,0])
        assert len(filtered.nodes) == 4
        classes = [n.operator_class for n in filtered.nodes]
        assert "C" not in classes  # C is [0,1], not matching

class TestSubpipelineModels:
    """Tests for subpipeline model chains [model1, model2]."""

    def test_substep_index_in_chain(self):
        """Test substep_index is correctly encoded in chain."""
        node = OperatorNode(
            step_index=3,
            operator_class="JaxMLP",
            substep_index=0
        )

        key = node.to_key()
        assert "sub=0" in key

    def test_subpipeline_unique_chains(self):
        """Test that subpipeline models have unique chains."""
        # Create chains for [model1, model2] at same step
        chains = []
        for substep in range(2):
            node = OperatorNode(
                step_index=3,
                operator_class=f"Model{substep}",
                substep_index=substep
            )
            chain = OperatorChain(nodes=[node], pipeline_id="test")
            chains.append(chain)

        paths = [c.to_path() for c in chains]
        hashes = [c.to_hash() for c in chains]

        # Each should be unique
        assert len(set(paths)) == 2
        assert len(set(hashes)) == 2

class TestMetaModelChains:
    """Tests for meta-model stacking chains."""

    def test_meta_model_chain_with_sources(self):
        """Test meta-model chain includes source model info."""
        # Source models at step 3
        source_chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Scaler"),
                OperatorNode(step_index=3, operator_class="PLS"),
            ],
            pipeline_id="test"
        )

        # Meta-model at step 4
        meta_node = OperatorNode(
            step_index=4,
            operator_class="Meta_Ridge"
        )
        meta_chain = source_chain.append(meta_node)

        path = meta_chain.to_path()
        assert "s1.Scaler" in path
        assert "s3.PLS" in path
        assert "s4.Meta_Ridge" in path

    def test_cross_branch_meta_model(self):
        """Test meta-model collecting from multiple branches."""
        # This would be represented in the chain as combining branches
        # The chain captures the meta-model's position, dependencies are in ArtifactRecord

        meta_node = OperatorNode(
            step_index=5,
            operator_class="Meta_Ridge",
            branch_path=[]  # Meta-model after branches merge
        )

        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="Scaler"),
                # Branch artifacts are tracked via ArtifactRecord.depends_on
                meta_node
            ],
            pipeline_id="test"
        )

        assert len(chain.nodes) == 2

class TestChainDeterminism:
    """Tests for chain determinism."""

    def test_same_chain_same_hash(self):
        """Test that identical chains produce identical hashes."""
        chain1 = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=3, operator_class="B", branch_path=[0]),
            ],
            pipeline_id="test"
        )

        chain2 = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=3, operator_class="B", branch_path=[0]),
            ],
            pipeline_id="test"
        )

        assert chain1.to_hash() == chain2.to_hash()
        assert chain1.to_path() == chain2.to_path()

    def test_different_chains_different_hashes(self):
        """Test that different chains produce different hashes."""
        chain1 = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="A")],
            pipeline_id="test"
        )

        chain2 = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="B")],
            pipeline_id="test"
        )

        assert chain1.to_hash() != chain2.to_hash()

    def test_order_matters(self):
        """Test that node order affects hash."""
        chain1 = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=2, operator_class="B"),
            ],
            pipeline_id="test"
        )

        chain2 = OperatorChain(
            nodes=[
                OperatorNode(step_index=2, operator_class="B"),
                OperatorNode(step_index=1, operator_class="A"),
            ],
            pipeline_id="test"
        )

        assert chain1.to_hash() != chain2.to_hash()

class TestOperatorNodeFromKey:
    """Tests for OperatorNode.from_key parsing."""

    def test_parse_simple_key(self):
        """Test parsing simple operator key."""
        node = OperatorNode.from_key("s3.PLSRegression")

        assert node.step_index == 3
        assert node.operator_class == "PLSRegression"
        assert node.branch_path == []
        assert node.source_index is None

    def test_parse_key_with_branch(self):
        """Test parsing key with branch path."""
        node = OperatorNode.from_key("s3.SNV[br=0]")

        assert node.step_index == 3
        assert node.operator_class == "SNV"
        assert node.branch_path == [0]

    def test_parse_key_with_nested_branch(self):
        """Test parsing key with nested branch path."""
        node = OperatorNode.from_key("s4.PLS[br=0.1]")

        assert node.step_index == 4
        assert node.operator_class == "PLS"
        assert node.branch_path == [0, 1]

    def test_parse_key_with_source(self):
        """Test parsing key with source index."""
        node = OperatorNode.from_key("s1.MinMaxScaler[src=2]")

        assert node.step_index == 1
        assert node.operator_class == "MinMaxScaler"
        assert node.source_index == 2

    def test_parse_key_with_multiple_qualifiers(self):
        """Test parsing key with multiple qualifiers."""
        node = OperatorNode.from_key("s3.SNV[br=0,src=1,sub=2]")

        assert node.step_index == 3
        assert node.operator_class == "SNV"
        assert node.branch_path == [0]
        assert node.source_index == 1
        assert node.substep_index == 2

class TestOperatorChainFromPath:
    """Tests for OperatorChain.from_path parsing."""

    def test_parse_simple_path(self):
        """Test parsing simple chain path."""
        chain = OperatorChain.from_path("s1.Scaler>s3.PLS", pipeline_id="test")

        assert len(chain.nodes) == 2
        assert chain.nodes[0].step_index == 1
        assert chain.nodes[0].operator_class == "Scaler"
        assert chain.nodes[1].step_index == 3
        assert chain.nodes[1].operator_class == "PLS"

    def test_parse_path_with_qualifiers(self):
        """Test parsing path with qualifiers."""
        chain = OperatorChain.from_path(
            "s1.Scaler[src=0]>s3.SNV[br=0]>s4.PLS[br=0]",
            pipeline_id="test"
        )

        assert len(chain.nodes) == 3
        assert chain.nodes[0].source_index == 0
        assert chain.nodes[1].branch_path == [0]
        assert chain.nodes[2].branch_path == [0]

    def test_parse_empty_path(self):
        """Test parsing empty path."""
        chain = OperatorChain.from_path("", pipeline_id="test")
        assert len(chain.nodes) == 0

    def test_roundtrip_path(self):
        """Test path generation and parsing roundtrip."""
        original = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A", source_index=0),
                OperatorNode(step_index=2, operator_class="B", branch_path=[0]),
                OperatorNode(step_index=3, operator_class="C", branch_path=[0], substep_index=1),
            ],
            pipeline_id="test"
        )

        path = original.to_path()
        restored = OperatorChain.from_path(path, pipeline_id="test")

        assert len(restored.nodes) == len(original.nodes)
        for i in range(len(original.nodes)):
            orig = original.nodes[i]
            rest = restored.nodes[i]
            assert orig.step_index == rest.step_index
            assert orig.operator_class == rest.operator_class
            assert orig.branch_path == rest.branch_path
            assert orig.source_index == rest.source_index
            assert orig.substep_index == rest.substep_index

class TestArtifactIdV3Generation:
    """Tests for V3 artifact ID generation."""

    def test_id_format(self):
        """Test V3 artifact ID format."""
        chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="Test")],
            pipeline_id="0001_pls"
        )

        artifact_id = generate_artifact_id_v3("0001_pls", chain, fold_id=None)

        # Format: pipeline$hash:fold
        assert "$" in artifact_id
        assert ":" in artifact_id
        assert artifact_id.startswith("0001_pls$")
        assert artifact_id.endswith(":all")

    def test_same_chain_same_id(self):
        """Test deterministic ID generation."""
        chain = OperatorChain(
            nodes=[
                OperatorNode(step_index=1, operator_class="A"),
                OperatorNode(step_index=3, operator_class="B", branch_path=[0]),
            ],
            pipeline_id="test"
        )

        id1 = generate_artifact_id_v3("0001", chain, fold_id=0)
        id2 = generate_artifact_id_v3("0001", chain, fold_id=0)

        assert id1 == id2

    def test_different_fold_different_id(self):
        """Test different folds produce different IDs."""
        chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="A")],
            pipeline_id="test"
        )

        id0 = generate_artifact_id_v3("0001", chain, fold_id=0)
        id1 = generate_artifact_id_v3("0001", chain, fold_id=1)

        assert id0 != id1
        assert id0.endswith(":0")
        assert id1.endswith(":1")

    def test_parse_v3_id(self):
        """Test parsing V3 artifact ID."""
        chain = OperatorChain(
            nodes=[OperatorNode(step_index=1, operator_class="A")],
            pipeline_id="test"
        )

        artifact_id = generate_artifact_id_v3("0001_pls", chain, fold_id=2)
        pipeline_id, chain_hash, fold_id = parse_artifact_id_v3(artifact_id)

        assert pipeline_id == "0001_pls"
        assert len(chain_hash) == 12  # Default hash length
        assert fold_id == 2
