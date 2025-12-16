"""
Operator Chain - V3 artifact identification system.

This module provides the core data structures for the V3 artifact system:
- OperatorNode: Represents a single operator in the execution path
- OperatorChain: Full path of operators that produced an artifact

The Operator Chain is the fundamental identifier for any artifact. It encodes
the complete chain of operators that produced the artifact, enabling:
- Complete chain tracking from input to output
- Deterministic replay of any execution path
- Unified handling of branching, multi-source, and stacking

Chain Path Format:
    "s{step}.{ClassName}[br={branch},src={source}]>s{step}.{ClassName}[...]"

Examples:
    - "s1.MinMaxScaler[src=0]" - Single transformer at step 1, source 0
    - "s1.MinMaxScaler>s3.SNV[br=0]>s4.PLS[br=0]" - Chain through branch 0
    - "s4.PLS[br=0]+s4.RF[br=1]>s5.Ridge" - Meta-model combining branches
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class OperatorNode:
    """Represents a single operator in the execution chain.

    An OperatorNode captures all the context needed to identify a specific
    operator execution within a pipeline, including its position, branch
    context, and source index for multi-source processing.

    Attributes:
        step_index: Pipeline step number (1-based)
        operator_class: Class name of the operator (e.g., "MinMaxScaler", "PLS")
        branch_path: Branch indices path (e.g., [0] for branch 0, [0, 1] for nested)
        source_index: Index for multi-source transformers (None for single source)
        fold_id: Fold number for CV models (None for shared artifacts)
        substep_index: Index within a substep (for [model1, model2] at same step)
        operator_name: Instance name if different from class name
    """

    step_index: int
    operator_class: str
    branch_path: List[int] = field(default_factory=list)
    source_index: Optional[int] = None
    fold_id: Optional[int] = None
    substep_index: Optional[int] = None
    operator_name: Optional[str] = None

    def to_key(self) -> str:
        """Generate compact key string for this node.

        Format: s{step}.{Class}[qualifiers]

        Qualifiers (only if present):
            br={branch_path} - Branch context
            src={source_index} - Multi-source index
            sub={substep_index} - Substep index

        Returns:
            Compact key string for this operator node

        Examples:
            >>> OperatorNode(1, "MinMaxScaler").to_key()
            's1.MinMaxScaler'
            >>> OperatorNode(3, "SNV", branch_path=[0]).to_key()
            's3.SNV[br=0]'
            >>> OperatorNode(3, "SNV", branch_path=[0], source_index=1).to_key()
            's3.SNV[br=0,src=1]'
        """
        parts = [f"s{self.step_index}.{self.operator_class}"]

        qualifiers = []
        if self.branch_path:
            br_str = ".".join(str(b) for b in self.branch_path)
            qualifiers.append(f"br={br_str}")
        if self.source_index is not None:
            qualifiers.append(f"src={self.source_index}")
        if self.substep_index is not None:
            qualifiers.append(f"sub={self.substep_index}")

        if qualifiers:
            parts.append(f"[{','.join(qualifiers)}]")

        return "".join(parts)

    def matches_context(
        self,
        step_index: Optional[int] = None,
        branch_path: Optional[List[int]] = None,
        source_index: Optional[int] = None,
        fold_id: Optional[int] = None
    ) -> bool:
        """Check if this node matches the given context filters.

        None values are treated as "match any".

        Args:
            step_index: Step number to match (None = any)
            branch_path: Branch path to match (None = any)
            source_index: Source index to match (None = any)
            fold_id: Fold ID to match (None = any)

        Returns:
            True if node matches all specified filters
        """
        if step_index is not None and self.step_index != step_index:
            return False
        if branch_path is not None and self.branch_path != branch_path:
            return False
        if source_index is not None and self.source_index != source_index:
            return False
        if fold_id is not None and self.fold_id != fold_id:
            return False
        return True

    def with_fold(self, fold_id: int) -> OperatorNode:
        """Create a copy of this node with a specific fold ID.

        Args:
            fold_id: The fold ID to set

        Returns:
            New OperatorNode with the specified fold_id
        """
        return OperatorNode(
            step_index=self.step_index,
            operator_class=self.operator_class,
            branch_path=self.branch_path.copy(),
            source_index=self.source_index,
            fold_id=fold_id,
            substep_index=self.substep_index,
            operator_name=self.operator_name,
        )

    def with_source(self, source_index: int) -> OperatorNode:
        """Create a copy of this node with a specific source index.

        Args:
            source_index: The source index to set

        Returns:
            New OperatorNode with the specified source_index
        """
        return OperatorNode(
            step_index=self.step_index,
            operator_class=self.operator_class,
            branch_path=self.branch_path.copy(),
            source_index=source_index,
            fold_id=self.fold_id,
            substep_index=self.substep_index,
            operator_name=self.operator_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for YAML/JSON
        """
        data = {
            "step_index": self.step_index,
            "operator_class": self.operator_class,
        }
        if self.branch_path:
            data["branch_path"] = self.branch_path
        if self.source_index is not None:
            data["source_index"] = self.source_index
        if self.fold_id is not None:
            data["fold_id"] = self.fold_id
        if self.substep_index is not None:
            data["substep_index"] = self.substep_index
        if self.operator_name:
            data["operator_name"] = self.operator_name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OperatorNode:
        """Create OperatorNode from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            OperatorNode instance
        """
        return cls(
            step_index=data.get("step_index", 0),
            operator_class=data.get("operator_class", ""),
            branch_path=data.get("branch_path", []),
            source_index=data.get("source_index"),
            fold_id=data.get("fold_id"),
            substep_index=data.get("substep_index"),
            operator_name=data.get("operator_name"),
        )

    @classmethod
    def from_key(cls, key: str) -> OperatorNode:
        """Parse an OperatorNode from its key string representation.

        Args:
            key: Key string like "s3.SNV[br=0,src=1]"

        Returns:
            OperatorNode instance

        Raises:
            ValueError: If key format is invalid
        """
        # Parse "s{step}.{Class}[qualifiers]"
        import re

        match = re.match(r"s(\d+)\.(\w+)(?:\[([^\]]+)\])?", key)
        if not match:
            raise ValueError(f"Invalid operator node key: {key}")

        step_index = int(match.group(1))
        operator_class = match.group(2)
        qualifiers_str = match.group(3)

        branch_path: List[int] = []
        source_index: Optional[int] = None
        substep_index: Optional[int] = None

        if qualifiers_str:
            for qualifier in qualifiers_str.split(","):
                if "=" in qualifier:
                    qkey, qval = qualifier.split("=", 1)
                    if qkey == "br":
                        branch_path = [int(b) for b in qval.split(".")]
                    elif qkey == "src":
                        source_index = int(qval)
                    elif qkey == "sub":
                        substep_index = int(qval)

        return cls(
            step_index=step_index,
            operator_class=operator_class,
            branch_path=branch_path,
            source_index=source_index,
            substep_index=substep_index,
        )

    def __repr__(self) -> str:
        return f"OperatorNode({self.to_key()})"


@dataclass
class OperatorChain:
    """Ordered sequence of OperatorNodes representing the full execution path.

    The OperatorChain captures the complete path of operators from input to
    the current artifact, enabling deterministic artifact identification
    and replay.

    Attributes:
        nodes: Ordered list of OperatorNode objects in the chain
        pipeline_id: Pipeline identifier this chain belongs to
    """

    nodes: List[OperatorNode] = field(default_factory=list)
    pipeline_id: str = ""

    def to_path(self) -> str:
        """Generate full path string from all nodes.

        Format: node1>node2>node3

        Returns:
            Chain path string

        Examples:
            >>> chain = OperatorChain([
            ...     OperatorNode(1, "MinMaxScaler"),
            ...     OperatorNode(3, "SNV", branch_path=[0])
            ... ])
            >>> chain.to_path()
            's1.MinMaxScaler>s3.SNV[br=0]'
        """
        return ">".join(node.to_key() for node in self.nodes)

    def to_hash(self, length: int = 12) -> str:
        """Compute deterministic hash of the chain path.

        Args:
            length: Number of hex characters to return (default: 12)

        Returns:
            Truncated SHA256 hash of the chain path
        """
        path = self.to_path()
        full_hash = hashlib.sha256(path.encode()).hexdigest()
        return full_hash[:length]

    def append(self, node: OperatorNode) -> OperatorChain:
        """Return new chain with node appended.

        Args:
            node: OperatorNode to append

        Returns:
            New OperatorChain with the node appended
        """
        return OperatorChain(
            nodes=self.nodes + [node],
            pipeline_id=self.pipeline_id,
        )

    def extend(self, other: OperatorChain) -> OperatorChain:
        """Return new chain with another chain's nodes appended.

        Args:
            other: OperatorChain to append

        Returns:
            New OperatorChain with all nodes from both chains
        """
        return OperatorChain(
            nodes=self.nodes + other.nodes,
            pipeline_id=self.pipeline_id or other.pipeline_id,
        )

    def filter_branch(self, target_branch_path: List[int]) -> OperatorChain:
        """Return chain with only nodes matching the branch path.

        Includes nodes that:
        - Have no branch path (shared/pre-branch artifacts)
        - Have a branch path that is a prefix of or equal to target

        Args:
            target_branch_path: Branch path to filter for

        Returns:
            New OperatorChain with only matching nodes
        """
        filtered = []
        for node in self.nodes:
            if self._branch_matches(node.branch_path, target_branch_path):
                filtered.append(node)
        return OperatorChain(nodes=filtered, pipeline_id=self.pipeline_id)

    def filter_step(self, step_index: int) -> OperatorChain:
        """Return chain with only nodes at the specified step.

        Args:
            step_index: Step index to filter for

        Returns:
            New OperatorChain with only matching nodes
        """
        filtered = [n for n in self.nodes if n.step_index == step_index]
        return OperatorChain(nodes=filtered, pipeline_id=self.pipeline_id)

    def filter_source(self, source_index: int) -> OperatorChain:
        """Return chain with only nodes for the specified source.

        Includes nodes that:
        - Have no source_index (single source)
        - Have matching source_index

        Args:
            source_index: Source index to filter for

        Returns:
            New OperatorChain with only matching nodes
        """
        filtered = []
        for node in self.nodes:
            if node.source_index is None or node.source_index == source_index:
                filtered.append(node)
        return OperatorChain(nodes=filtered, pipeline_id=self.pipeline_id)

    def get_last_node(self) -> Optional[OperatorNode]:
        """Get the last node in the chain.

        Returns:
            Last OperatorNode or None if chain is empty
        """
        return self.nodes[-1] if self.nodes else None

    def get_nodes_at_step(self, step_index: int) -> List[OperatorNode]:
        """Get all nodes at a specific step.

        Args:
            step_index: Step index to filter

        Returns:
            List of nodes at that step
        """
        return [n for n in self.nodes if n.step_index == step_index]

    def get_branch_path(self) -> List[int]:
        """Get the branch path from the last node.

        Returns:
            Branch path of the last node, or empty list if no nodes
        """
        last = self.get_last_node()
        return last.branch_path.copy() if last else []

    def is_empty(self) -> bool:
        """Check if chain has no nodes.

        Returns:
            True if chain is empty
        """
        return len(self.nodes) == 0

    def copy(self) -> OperatorChain:
        """Create a deep copy of this chain.

        Returns:
            New OperatorChain with copied nodes
        """
        return OperatorChain(
            nodes=[
                OperatorNode(
                    step_index=n.step_index,
                    operator_class=n.operator_class,
                    branch_path=n.branch_path.copy(),
                    source_index=n.source_index,
                    fold_id=n.fold_id,
                    substep_index=n.substep_index,
                    operator_name=n.operator_name,
                )
                for n in self.nodes
            ],
            pipeline_id=self.pipeline_id,
        )

    def merge_with_prefix(
        self,
        prefix_chain: OperatorChain,
        step_offset: int = 0
    ) -> OperatorChain:
        """Merge this chain with a prefix chain for bundle import.

        Used when importing a bundle into a pipeline, where the bundle's
        chain needs to be prefixed with the import context's chain.

        Args:
            prefix_chain: Chain to prepend (the import context)
            step_offset: Offset to add to step indices in this chain

        Returns:
            New merged OperatorChain

        Example:
            >>> bundle_chain = OperatorChain.from_path("s1.Scaler>s3.PLS")
            >>> import_chain = OperatorChain.from_path("s1.Import")
            >>> merged = bundle_chain.merge_with_prefix(import_chain, step_offset=1)
            # Result: "s1.Import>s2.Scaler>s4.PLS"
        """
        # Copy prefix chain nodes
        merged_nodes = [
            OperatorNode(
                step_index=n.step_index,
                operator_class=n.operator_class,
                branch_path=n.branch_path.copy(),
                source_index=n.source_index,
                fold_id=n.fold_id,
                substep_index=n.substep_index,
                operator_name=n.operator_name,
            )
            for n in prefix_chain.nodes
        ]

        # Add this chain's nodes with step offset
        for n in self.nodes:
            merged_nodes.append(OperatorNode(
                step_index=n.step_index + step_offset,
                operator_class=n.operator_class,
                branch_path=n.branch_path.copy(),
                source_index=n.source_index,
                fold_id=n.fold_id,
                substep_index=n.substep_index,
                operator_name=n.operator_name,
            ))

        return OperatorChain(
            nodes=merged_nodes,
            pipeline_id=prefix_chain.pipeline_id or self.pipeline_id,
        )

    def remap_steps(self, step_mapping: Dict[int, int]) -> OperatorChain:
        """Create new chain with remapped step indices.

        Args:
            step_mapping: Mapping from old step index to new step index

        Returns:
            New OperatorChain with remapped steps
        """
        remapped_nodes = [
            OperatorNode(
                step_index=step_mapping.get(n.step_index, n.step_index),
                operator_class=n.operator_class,
                branch_path=n.branch_path.copy(),
                source_index=n.source_index,
                fold_id=n.fold_id,
                substep_index=n.substep_index,
                operator_name=n.operator_name,
            )
            for n in self.nodes
        ]
        return OperatorChain(nodes=remapped_nodes, pipeline_id=self.pipeline_id)

    def with_pipeline_id(self, pipeline_id: str) -> OperatorChain:
        """Create a copy of this chain with a new pipeline ID.

        Args:
            pipeline_id: New pipeline ID to set

        Returns:
            New OperatorChain with the specified pipeline_id
        """
        chain = self.copy()
        chain.pipeline_id = pipeline_id
        return chain

    @staticmethod
    def _branch_matches(node_path: List[int], target_path: List[int]) -> bool:
        """Check if node's branch path is compatible with target.

        A node matches if:
        - It has no branch path (pre-branch/shared)
        - Its branch path is equal to target
        - Its branch path is a prefix of target (for nested branches)

        Args:
            node_path: Branch path of the node
            target_path: Target branch path to match

        Returns:
            True if node is compatible with target branch
        """
        if not node_path:
            return True  # Pre-branch/shared artifacts match any branch
        if node_path == target_path:
            return True
        # Check if node_path is a prefix of target_path
        return (
            len(node_path) < len(target_path)
            and target_path[: len(node_path)] == node_path
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "pipeline_id": self.pipeline_id,
            "chain_path": self.to_path(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OperatorChain:
        """Create OperatorChain from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            OperatorChain instance
        """
        nodes = [OperatorNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            nodes=nodes,
            pipeline_id=data.get("pipeline_id", ""),
        )

    @classmethod
    def from_path(cls, path: str, pipeline_id: str = "") -> OperatorChain:
        """Parse OperatorChain from a path string.

        Args:
            path: Chain path string like "s1.MinMaxScaler>s3.SNV[br=0]"
            pipeline_id: Pipeline identifier

        Returns:
            OperatorChain instance
        """
        if not path:
            return cls(nodes=[], pipeline_id=pipeline_id)

        nodes = [OperatorNode.from_key(key) for key in path.split(">")]
        return cls(nodes=nodes, pipeline_id=pipeline_id)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return len(self.nodes) > 0

    def __repr__(self) -> str:
        if not self.nodes:
            return "OperatorChain(empty)"
        return f"OperatorChain({self.to_path()})"


def compute_chain_hash(chain_path: str, length: int = 12) -> str:
    """Compute deterministic hash from chain path string.

    Args:
        chain_path: Full operator chain path
        length: Number of hex characters (default: 12)

    Returns:
        Truncated SHA256 hash
    """
    return hashlib.sha256(chain_path.encode()).hexdigest()[:length]


def generate_artifact_id_v3(
    pipeline_id: str,
    chain: Union[OperatorChain, str],
    fold_id: Optional[int] = None
) -> str:
    """Generate V3 artifact ID from chain.

    Format: {pipeline_id}${chain_hash}:{fold_id}

    Args:
        pipeline_id: Pipeline identifier
        chain: Operator chain object or chain path string for this artifact
        fold_id: Fold ID (None for shared artifacts)

    Returns:
        V3 artifact ID string

    Examples:
        >>> generate_artifact_id_v3("0001_pls", chain, None)
        '0001_pls$a1b2c3d4e5f6:all'
        >>> generate_artifact_id_v3("0001_pls", chain, 0)
        '0001_pls$a1b2c3d4e5f6:0'
    """
    if isinstance(chain, str):
        chain_hash = compute_chain_hash(chain, 12)
    else:
        chain_hash = chain.to_hash(12)
    fold_str = str(fold_id) if fold_id is not None else "all"
    return f"{pipeline_id}${chain_hash}:{fold_str}"


def parse_artifact_id_v3(artifact_id: str) -> Tuple[str, str, Optional[int]]:
    """Parse V3 artifact ID into components.

    Args:
        artifact_id: V3 artifact ID string

    Returns:
        Tuple of (pipeline_id, chain_hash, fold_id)

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_artifact_id_v3("0001_pls$a1b2c3d4e5f6:all")
        ('0001_pls', 'a1b2c3d4e5f6', None)
        >>> parse_artifact_id_v3("0001_pls$a1b2c3d4e5f6:0")
        ('0001_pls', 'a1b2c3d4e5f6', 0)
    """
    if "$" not in artifact_id or ":" not in artifact_id:
        raise ValueError(f"Invalid V3 artifact ID format: {artifact_id}")

    # Split pipeline_id from rest
    pipeline_part, rest = artifact_id.split("$", 1)

    # Split chain_hash from fold_id
    chain_hash, fold_str = rest.rsplit(":", 1)

    fold_id = None if fold_str == "all" else int(fold_str)

    return pipeline_part, chain_hash, fold_id


def is_v3_artifact_id(artifact_id: str) -> bool:
    """Check if an artifact ID is in V3 format.

    Args:
        artifact_id: Artifact ID to check

    Returns:
        True if V3 format, False otherwise
    """
    return "$" in artifact_id
